#!/usr/bin/env python
"""
crawler.py – Collect IssueNode objects from Jira, including rich metadata.

Depends on:
    • jira_rag.jira_client.JiraClient
    • jira_rag.models.IssueNode

Usage::

    from jira_rag.jira_client import JiraClient
    from jira_rag.crawler import JiraCrawler

    client = JiraClient(JIRA_URL, JIRA_USERNAME, JIRA_PASSWORD)
    crawler = JiraCrawler(client)
    issues = crawler.crawl()  # List[IssueNode]
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from .jira_client import JiraClient
from .models import IssueNode

__all__ = ["JiraCrawler"]

log = logging.getLogger("jira_rag.crawler")


class JiraCrawler:
    """Traverse Jira issues and convert them to IssueNode with full fields."""

    FIELDS = [
        "summary",
        "description",
        "status",
        "issuetype",
        "issuelinks",
        "project",
        "parent",
        "assignee",
        "reporter",
        "priority",
        "resolution",
        "created",
        "updated",
        "labels",
        "components",
        "fixVersions",
    ]

    def __init__(self, client: JiraClient):
        self.client = client

    def crawl(self, jql_scope: str | None = None) -> List[IssueNode]:
        """Return all issues visible to the authenticated user or within a custom JQL scope."""
        log.info("Starting Jira crawl …")
        nodes: List[IssueNode] = []
        for issue in self.client.search(jql_scope or "", fields=self.FIELDS):
            nodes.append(self._issue_to_node(issue))
        log.info("Crawl finished – %s issues", len(nodes))
        return nodes

    def _issue_to_node(self, issue: dict[str, Any]) -> IssueNode:
        f = issue["fields"]

        # Helper to safely extract displayName
        def _disp(o: Optional[dict[str, Any]], key: str = "displayName") -> Optional[str]:
            return o.get(key) if isinstance(o, dict) else None

        return IssueNode(
            key=issue["key"],
            issue_type=f["issuetype"]["name"],
            summary=f.get("summary", "") or "",
            description=f.get("description", "") or "",
            status=f["status"]["name"],
            project_key=f["project"]["key"],
            parent_key=(f.get("parent") or {}).get("key"),
            dependencies=self._extract_deps(f.get("issuelinks", [])),
            assignee=_disp(f.get("assignee")),
            reporter=_disp(f.get("reporter")),
            priority=(f.get("priority") or {}).get("name"),
            resolution=(f.get("resolution") or {}).get("name"),
            created=f.get("created"),
            updated=f.get("updated"),
            labels=f.get("labels", []),
            components=[
                c.get("name") for c in f.get("components", []) if isinstance(c, dict)
            ],
            fix_versions=[
                v.get("name") for v in f.get("fixVersions", []) if isinstance(v, dict)
            ],
        )

    @staticmethod
    def _extract_deps(links: List[dict[str, Any]]) -> List[dict[str, Any]]:
        """Extract inward/outward dependencies from issuelinks."""
        deps: List[dict[str, Any]] = []
        for link in links:
            if "outwardIssue" in link:
                deps.append({
                    "direction": "outward",
                    "key": link["outwardIssue"]["key"],
                    "link_type": link["type"]["outward"],  # e.g. "blocks"
                })
            if "inwardIssue" in link:
                deps.append({
                    "direction": "inward",
                    "key": link["inwardIssue"]["key"],
                    "link_type": link["type"]["inward"],  # e.g. "is blocked by"
                })
        return deps
