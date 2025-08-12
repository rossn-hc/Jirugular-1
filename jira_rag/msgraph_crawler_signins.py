# msgraph_signins_crawler.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np

from .msgraph_client import MSGraphClient
from .embedder import Embedder
from .vector_store import FaissIndexer
from .config import log

def crawl_msgraph_signins(
    tenant_id: str,
    client_id: str,
    client_secret: str,
    stem: str = "msgraph_signins",
    start_date: str | None = None,  # ISO yyyy-mm-dd or yyyy-mm-ddThh:mm:ssZ
    end_date: str | None = None,
    app_display_name: str | None = None,  # e.g., "Windows Sign In", "Microsoft Teams"
    top: int | None = None,
) -> Tuple[int, str]:
    """
    Crawl Microsoft Graph sign-ins and build a FAISS index.

    Permissions: APPLICATION 'AuditLog.Read.All' (admin consented)
    """
    client = MSGraphClient(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
        baseline="graph-signins",
    )

    # Build $filter
    filter_clauses = []
    # Date filter uses createdDateTime; build only when both provided
    if start_date and end_date:
        # accept yyyy-mm-dd; expand to whole day if no 'T' present
        sd = start_date if "T" in start_date else f"{start_date}T00:00:00Z"
        ed = end_date if "T" in end_date else f"{end_date}T23:59:59Z"
        filter_clauses.append(f"(createdDateTime ge {sd} and createdDateTime le {ed})")

    if app_display_name:
        # common scenario: limit to one or two well-known apps
        filter_clauses.append(f"(appDisplayName eq '{app_display_name}')")

    params: Dict[str, Any] = {}
    if filter_clauses:
        params["$filter"] = " and ".join(filter_clauses)

    # Thin select keeps index light; adjust if you need more fields later
    params["$select"] = (
        "id,createdDateTime,appDisplayName,userDisplayName,userPrincipalName,"
        "ipAddress,clientAppUsed,deviceDetail,location,status"
    )
    params["$orderby"] = "createdDateTime desc"
    if top:
        params["$top"] = top

    base_url = "https://graph.microsoft.com/v1.0/auditLogs/signIns"
    raw = client.query_msgraph(base_url, params=params, max_retries=10)

    records = raw.get("value", []) if isinstance(raw, dict) else (raw or [])
    filtered: List[Dict[str, Any]] = []

    for r in records:
        if not r:
            continue

        # Extract useful bits with safe fallbacks
        sid = r.get("id") or ""
        when = r.get("createdDateTime") or ""
        app = r.get("appDisplayName") or ""
        upn = r.get("userPrincipalName") or ""
        uname = r.get("userDisplayName") or ""
        ip = r.get("ipAddress") or ""
        cap = r.get("clientAppUsed") or ""
        dev = r.get("deviceDetail") or {}
        os_name = dev.get("operatingSystem") or ""
        browser = dev.get("browser") or ""
        loc = r.get("location") or {}
        city = loc.get("city") or ""
        country = loc.get("countryOrRegion") or ""
        status = (r.get("status") or {}).get("errorCode", 0)
        status_text = "Success" if status == 0 else f"Error {status}"

        # Embedding-friendly document (plain text, not Jira-shaped)
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

        # Metadata we’ll keep alongside the vector (no Jira fields)
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
            # handy tags so the chat layer knows what it is
            "datasource": "signins",
        })

    embedder = Embedder()
    index = FaissIndexer(dim=embedder.dim, stem=stem)

    if not filtered:
        log.warning("MS Graph /signIns returned 0 usable records.")
        index.add(vectors=np.empty((0, embedder.dim), dtype=np.float32), metas=[])
        index.save()
        return 0, stem

    vecs = [embedder.encode_one(r["document"]).squeeze() for r in filtered]
    index.add(vectors=np.stack(vecs), metas=filtered)
    index.save()

    log.info("Indexed %d MS Graph sign-ins into stem '%s'.", len(filtered), stem)
    return len(filtered), stem
