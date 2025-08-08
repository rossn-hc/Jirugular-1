#!/usr/bin/env python
"""jira_client.py – Thin wrapper around Jira REST API v2.

* Handles **pagination** transparently via the Search endpoint.
* Uses **tenacity** for retry/back‑off on transient failures.
* Adds two convenience helpers for the hybrid retriever:

    • `get_issue(key, fields=None)`   – fetch a single issue on demand.
    • `get_issues(keys, fields=None)` – batch fetch up to 50 keys per request
      (Jira’s max for the `id`/`issueKey` search parameters).

The class is deliberately stateless beyond the underlying `requests.Session`,
so you can share one instance across many crawls / live look‑ups.
"""
from __future__ import annotations

import itertools
import logging
import sys
from typing import Any, Dict, Iterator, List, Sequence

import requests
from requests.auth import HTTPBasicAuth
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)


class JiraClient:  # pylint: disable=too-few-public-methods
    """Minimal Jira REST v2 helper.

    Parameters
    ----------
    base_url : str
        Base URL to your Jira instance, e.g. ``https://jira.example.com/jira``.
    username, password : str
        Basic‑auth credentials (or token as *password*).
    verify_ssl : bool, default True
        Set to False to skip TLS verification (self‑signed certs, etc.).
    timeout : int, default 30
        Per‑request timeout in seconds.
    """

    SEARCH_ENDPOINT = "rest/api/2/search"
    ISSUE_ENDPOINT = "rest/api/2/issue/{key}"

    def __init__(self, base_url: str, username: str, password: str, *, verify_ssl: bool = True, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        sess = requests.Session()
        sess.auth = HTTPBasicAuth(username, password)
        sess.headers.update({"Content-Type": "application/json"})
        sess.verify = verify_ssl
        self.session = sess

    # ------------------------------------------------------------------
    # Internal GET with retry/back‑off
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def _get(self, endpoint: str, **params: Any) -> Any:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Pagination helper
    # ------------------------------------------------------------------

    def search(
        self,
        jql: str,
        *,
        fields: Sequence[str] | None = None,
        batch: int = 200,
        expand: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield every issue matching *JQL*, transparently handling pagination."""

        start_at = 0
        total = sys.maxsize
        while start_at < total:
            params = {
                "jql": jql,
                "startAt": start_at,
                "maxResults": batch,
            }
            if fields:
                params["fields"] = ",".join(fields)
            if expand:
                params["expand"] = expand

            page: dict[str, Any] = self._get(self.SEARCH_ENDPOINT, **params)
            total = page.get("total", 0)
            issues: list[Any] = page.get("issues", [])
            log.debug("Fetched %s/%s issues (batch=%s)", start_at + len(issues), total, batch)

            yield from issues
            start_at += len(issues)

    # ------------------------------------------------------------------
    # Live look‑ups for HybridRetriever
    # ------------------------------------------------------------------

    def get_issue(self, key: str, *, fields: Sequence[str] | None = None) -> dict[str, Any]:
        """Fetch a single issue by key."""
        endpoint = self.ISSUE_ENDPOINT.format(key=key)
        params = {"fields": ",".join(fields)} if fields else {}
        return self._get(endpoint, **params)

    def get_issues(
        self,
        keys: Sequence[str],
        *,
        fields: Sequence[str] | None = None,
        batch_size: int = 50,
    ) -> List[dict[str, Any]]:
        """Batch fetch up to *batch_size* keys per call via the Search API.

        Jira’s Search endpoint allows `issueKey=KEY1,KEY2,…` (or `id=`) so we
        chunk the list to minimise round‑trips.
        """

        if not keys:
            return []

        chunks = (keys[i : i + batch_size] for i in range(0, len(keys), batch_size))
        results: list[dict[str, Any]] = []
        for chunk in chunks:
            jql = "issueKey IN (" + ",".join(chunk) + ")"
            params = {
                "jql": jql,
                "maxResults": len(chunk),
            }
            if fields:
                params["fields"] = ",".join(fields)
            page = self._get(self.SEARCH_ENDPOINT, **params)
            results.extend(page.get("issues", []))
        return results


__all__ = ["JiraClient"]
