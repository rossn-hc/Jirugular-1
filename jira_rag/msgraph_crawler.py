# msgraph_crawler.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np

from .msgraph_client import MSGraphClient
from .embedder import Embedder
from .vector_store import FaissIndexer
from .config import log


def crawl_msgraph_people(
    tenant_id: str,
    client_id: str,
    client_secret: str,
    stem: str = "msgraph_people",
    top: int | None = None,
) -> Tuple[int, str]:
    """
    Build a FAISS index from Microsoft Graph /users.
    Only persists: displayName, userPrincipalName, mail, jobTitle, department, accountEnabled, (optional id), document.

    App-only friendly (requires APPLICATION permission: User.Read.All).
    """
    client = MSGraphClient(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
        baseline="graph",
    )

    # --- Fetch users ---
    params: Dict[str, Any] = {
        "$select": "displayName,jobTitle,department,mail,userPrincipalName,accountEnabled,id",
        "$orderby": "displayName",
    }
    if top:
        params["$top"] = top

    base_url = "https://graph.microsoft.com/v1.0/users"
    users = client.query_msgraph(base_url, params=params, max_retries=5)

    # Normalize to a list
    records = users.get("value", []) if isinstance(users, dict) else (users or [])
    filtered: List[Dict[str, Any]] = []

    for u in records:
        if not isinstance(u, dict):
            continue
        if u.get("accountEnabled") is False:
            continue

        name = (u.get("displayName") or "").strip()
        upn = (u.get("userPrincipalName") or "").strip()
        mail = (u.get("mail") or "").strip()
        job = (u.get("jobTitle") or "").strip()
        dept = (u.get("department") or "").strip()
        uid = (u.get("id") or "").strip()

        # minimal eligibility: at least a name or UPN
        if not (name or upn or mail):
            continue

        # Embedding-friendly document (kept small & neutral)
        doc = (
            f"Name: {name or '-'}\n"
            f"UPN: {upn or '-'}\n"
            f"Email: {mail or '-'}\n"
            f"Title: {job or '-'}\n"
            f"Department: {dept or '-'}\n"
            "Source: Microsoft Graph /users"
        ).strip()

        filtered.append(
            {
                # STRICTLY graph fields + our embedding doc
                "id": uid or None,
                "displayName": name or None,
                "userPrincipalName": upn or None,
                "mail": mail or None,
                "jobTitle": job or None,
                "department": dept or None,
                "accountEnabled": u.get("accountEnabled"),
                "document": doc,
            }
        )

    if not filtered:
        log.warning("MS Graph /users returned 0 usable records.")
        # still write an empty index so later loads won’t crash
        embedder = Embedder()
        index = FaissIndexer(dim=embedder.dim, stem=stem)
        index.add(vectors=np.empty((0, embedder.dim), dtype=np.float32), metas=[])
        index.save()
        return 0, stem

    # --- Embed + index ---
    embedder = Embedder()
    vecs: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []

    for r in filtered:
        v = embedder.encode_one(r["document"]).squeeze()
        vecs.append(v)
        metas.append(r)

    index = FaissIndexer(dim=embedder.dim, stem=stem)
    index.add(vectors=np.stack(vecs), metas=metas)
    index.save()

    count = len(filtered)
    log.info("Indexed %d MS Graph users into stem '%s'.", count, stem)
    return count, stem
