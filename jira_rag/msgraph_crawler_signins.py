#!/usr/bin/env python
# msgraph_signins_crawler.py
from __future__ import annotations

from typing import List, Dict, Any, Tuple, Iterable, Optional
import numpy as np

from .msgraph_client import MSGraphClient
from .embedder import Embedder
from .vector_store import FaissIndexer
from .config import log


def _iso_floor(dt: str) -> str:
    """Accept 'YYYY-MM-DD' or full ISO; floor to 00:00:00Z when no 'T'."""
    if not dt:
        return dt
    return dt if "T" in dt else f"{dt}T00:00:00Z"


def _iso_ceil(dt: str) -> str:
    """Accept 'YYYY-MM-DD' or full ISO; ceil to 23:59:59Z when no 'T'."""
    if not dt:
        return dt
    return dt if "T" in dt else f"{dt}T23:59:59Z"


def _ensure_iter_apps(app_display_name: Optional[str | Iterable[str]]) -> List[str]:
    """Normalize app filter to a list of display names."""
    if not app_display_name:
        return []
    if isinstance(app_display_name, str):
        # support comma-separated convenience
        parts = [p.strip() for p in app_display_name.split(",")]
        return [p for p in parts if p]
    return [str(x).strip() for x in app_display_name if str(x).strip()]


def _quote_odata(s: str) -> str:
    """Escape single quotes per OData (' becomes '') and wrap with single quotes."""
    return "'" + s.replace("'", "''") + "'"


def crawl_msgraph_signins(
    tenant_id: str,
    client_id: str,
    client_secret: str,
    stem: str = "msgraph_signins",
    start_date: str | None = None,  # ISO yyyy-mm-dd or yyyy-mm-ddThh:mm:ssZ
    end_date: str | None = None,
    app_display_name: str | Iterable[str] | None = None,  # "Teams" or ["Teams","Windows Sign In"]
    top: int | None = None,  # overall cap across pages (None = all that Graph returns)
) -> Tuple[int, str]:
    """
    Crawl Microsoft Graph sign-ins and build a FAISS index.

    Permissions: APPLICATION 'AuditLog.Read.All' (admin consented).
    """
    client = MSGraphClient(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
        baseline="graph-signins",
    )

    # -------------- Build $filter --------------
    filter_clauses: List[str] = []

    # Date filter on createdDateTime; allow start-only or end-only
    if start_date:
        sd = _iso_floor(start_date)
        filter_clauses.append(f"(createdDateTime ge {sd})")
    if end_date:
        ed = _iso_ceil(end_date)
        filter_clauses.append(f"(createdDateTime le {ed})")

    # App filter: support one or many names (OR'd)
    apps = _ensure_iter_apps(app_display_name)
    if apps:
        or_clause = " or ".join([f"(appDisplayName eq {_quote_odata(a)})" for a in apps])
        filter_clauses.append(f"({or_clause})")

    params: Dict[str, Any] = {}
    if filter_clauses:
        params["$filter"] = " and ".join(filter_clauses)

    # Keep payload lean; add fields later if you need them
    params["$select"] = (
        "id,createdDateTime,appDisplayName,userDisplayName,userPrincipalName,"
        "ipAddress,clientAppUsed,deviceDetail,location,status"
    )
    params["$orderby"] = "createdDateTime desc"

    # $top is per page; we'll also enforce our own overall cap below
    if top:
        # Ask Graph for at most 'top' per page (if smaller than their default)
        params["$top"] = int(top)

    base_url = "https://graph.microsoft.com/v1.0/auditLogs/signIns"

    # -------------- Fetch (follow @odata.nextLink) --------------
    all_records: List[Dict[str, Any]] = []
    next_url: Optional[str] = None
    total_cap = top if (isinstance(top, int) and top > 0) else None
    page = 0

    while True:
        page += 1
        if next_url:
            raw = client.query_msgraph(next_url, params=None, max_retries=10)
        else:
            raw = client.query_msgraph(base_url, params=params, max_retries=10)

        if not raw:
            break

        # Normalize payload
        values = raw.get("value", []) if isinstance(raw, dict) else (raw or [])
        if not isinstance(values, list):
            values = []

        all_records.extend(values)

        # Respect overall cap (stop once we have >= cap)
        if total_cap is not None and len(all_records) >= total_cap:
            all_records = all_records[:total_cap]
            break

        next_url = raw.get("@odata.nextLink") if isinstance(raw, dict) else None
        if not next_url:
            break

        # Optional: log progress
        try:
            log.debug("sign-ins page %d, total so far: %d", page, len(all_records))
        except Exception:
            pass

    if not all_records:
        log.warning("MS Graph /signIns returned no records (filters: %s).", params.get("$filter", "—"))

    # -------------- Shape → metas + embed --------------
    filtered: List[Dict[str, Any]] = []
    for r in all_records:
        if not r:
            continue

        # Extract with safe fallbacks
        sid = r.get("id") or ""
        when = r.get("createdDateTime") or ""
        app = r.get("appDisplayName") or ""
        upn = r.get("userPrincipalName") or ""
        uname = r.get("userDisplayName") or ""
        ip = r.get("ipAddress") or ""
        cap = r.get("clientAppUsed") or ""
        dev = r.get("deviceDetail") or {}
        os_name = (dev or {}).get("operatingSystem") or ""
        browser = (dev or {}).get("browser") or ""
        loc = r.get("location") or {}
        city = (loc or {}).get("city") or ""
        country = (loc or {}).get("countryOrRegion") or ""
        status_code = (r.get("status") or {}).get("errorCode", 0)
        status_text = "Success" if status_code == 0 else f"Error {status_code}"

        # Embedding-friendly doc
        doc = (
            f"Sign-in ID: {sid}\n"
            f"User: {uname} ({upn})\n"
            f"When: {when}\n"
            f"App: {app}\n"
            f"Client: {cap}\n"
            f"IP: {ip}\n"
            f"Device OS: {os_name}\n"
            f"Browser: {browser}\n"
            f"Location: {city}, {country}\n"
            f"Result: {status_text}\n"
            "Source: Microsoft Graph /auditLogs/signIns"
        ).strip()

        filtered.append({
            "id": sid,
            "createdDateTime": when,
            "appDisplayName": app,
            "userPrincipalName": upn,
            "userDisplayName": uname,
            "ipAddress": ip,
            "clientAppUsed": cap,
            "operatingSystem": os_name,
            "browser": browser,
            "city": city,
            "countryOrRegion": country,
            "result": status_text,
            "document": doc,
            "datasource": "signins",
        })

    # -------------- Index build --------------
    embedder = Embedder()
    index = FaissIndexer(dim=embedder.dim, stem=stem)

    if not filtered:
        log.warning("MS Graph /signIns produced 0 usable records after shaping; writing empty index.")
        index.add(vectors=np.empty((0, embedder.dim), dtype=np.float32), metas=[])
        index.save()
        return 0, stem

    vecs = [embedder.encode_one(r["document"]).squeeze() for r in filtered]
    index.add(vectors=np.stack(vecs), metas=filtered)
    index.save()

    log.info("Indexed %d MS Graph sign-ins into stem '%s'.", len(filtered), stem)
    return len(filtered), stem
