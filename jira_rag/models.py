#!/usr/bin/env python
"""models.py – lightweight data structures used across the package."""

from __future__ import annotations
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["IssueNode"]


@dataclass
class IssueNode:
    """Full representation of a Jira issue for embedding, chat, and metadata."""

    key: str
    issue_type: str
    summary: str
    description: str
    status: str
    project_key: str

    parent_key: Optional[str] = None
    dependencies: List[Dict[str, Any]] = field(default_factory=list)

    # — Newly added Jira fields —
    assignee: Optional[str] = None
    reporter: Optional[str] = None
    priority: Optional[str] = None
    resolution: Optional[str] = None
    created: Optional[str] = None
    updated: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    components: List[str] = field(default_factory=list)
    fix_versions: List[str] = field(default_factory=list)

    def to_document(self) -> str:
        """Return a single self‑contained text block suitable for embedding."""
        deps_txt = ", ".join(f"{d['direction'].upper()} {d['key']}" for d in self.dependencies) or "None"
        labels_txt = ", ".join(self.labels) or "None"
        comps_txt = ", ".join(self.components) or "None"
        fixes_txt = ", ".join(self.fix_versions) or "None"

        return (
            f"[{self.issue_type.upper()}] {self.key}\n"
            f"Status: {self.status}\n"
            f"Project: {self.project_key}\n"
            f"Parent: {self.parent_key or '-'}\n"
            f"Assignee: {self.assignee or '-'}\n"
            f"Reporter: {self.reporter or '-'}\n"
            f"Priority: {self.priority or '-'}\n"
            f"Resolution: {self.resolution or '-'}\n"
            f"Created: {self.created or '-'}\n"
            f"Updated: {self.updated or '-'}\n"
            f"Labels: {labels_txt}\n"
            f"Components: {comps_txt}\n"
            f"Fix Versions: {fixes_txt}\n"
            f"Dependencies: {deps_txt}\n\n"
            f"Summary:\n{self.summary}\n\n"
            f"Description:\n{self.description}"
        )

    def to_metadata(self) -> Dict[str, Any]:
        """
        Return a trimmed dict (no bulky description) for FAISS sidecar.
        Includes all key fields for filtering and live lookups.
        """
        d = {
            "key": self.key,
            "issue_type": self.issue_type,
            "status": self.status,
            "project_key": self.project_key,
            "parent_key": self.parent_key,
            "dependencies": self.dependencies,
            "assignee": self.assignee,
            "reporter": self.reporter,
            "priority": self.priority,
            "resolution": self.resolution,
            "created": self.created,
            "updated": self.updated,
            "labels": self.labels,
            "components": self.components,
            "fix_versions": self.fix_versions,
        }
        return d

    def to_json(self) -> str:
        """Convenience for raw JSON dumps (includes every field)."""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)
