#!/usr/bin/env python
"""
chat.py – ChatService that blends semantic recall (FAISS) with live Jira data,
with optional verbose mode, expanded metadata context, and enhanced summaries.

Features:
- Personas via `character` (pirate, yoda, shakespeare, executive-snark, or custom)
- Persona intensity: light | medium | heavy
- Multi-format 4-section output with localized headings + hard FORMAT LOCK
- Language directive first (e.g., "en", "fr-CA", "fr", "es", "de"...)
- Temperature / max_tokens knobs
- Role-aware base prompts (developer, manager, executive)
- Backward-compat for `pirate=True`
- Patricize: when True, model appends one grounded dad joke (localized) at the end
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from openai import OpenAI
import openai as oai_err

from .config import OPENAI_API_KEY, CHAT_MODEL, log
from .hybrid_retriever import HybridRetriever
from .vector_store import FaissIndexer
from .embedder import Embedder
from .jira_client import JiraClient


class ChatService:
    def __init__(
        self,
        indexer: FaissIndexer,
        embedder: Embedder,
        jira: JiraClient,
        model: Optional[str] = None,
    ) -> None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model or CHAT_MODEL or "gpt-4-turbo"
        self.retriever = HybridRetriever(indexer, embedder, jira)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def answer(
        self,
        question: str,
        top_k: int = 5,
        verbose: bool = False,
        multi_format: bool = False,
        role: Optional[str] = None,
        character: Optional[str] = None,
        intensity: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        language: Optional[str] = None,
        # legacy shim
        pirate: Optional[bool] = None,
        # NEW: ask the model to append a dad joke, grounded in its own answer
        patricize: bool = False,
    ) -> Dict[str, Any]:
        """
        Retrieve context from FAISS + live Jira fields, construct a robust prompt that
        respects persona/language/formatting, call OpenAI, and return:
            { "answer": <text>, "sources": <raw_hits>, "structured": <normalized_fields> }
        """
        # Map legacy pirate flag to new character param if provided
        if character is None and pirate is not None:
            character = "pirate" if pirate else None
            try:
                log.debug("Deprecated arg 'pirate' used; mapped to character=%r", character)
            except Exception:
                pass

        # ---------------- Retrieval + dedupe ----------------
        hits = self.retriever.retrieve(question, k=top_k)
        unique_hits: List[Dict[str, Any]] = []
        seen_keys: set[str] = set()
        for h in hits:
            key = h.get("key") or h.get("issue_key")
            if key and key not in seen_keys:
                seen_keys.add(key)
                unique_hits.append(h)
        hits = unique_hits

        # ---------------- System prompt stack ----------------
        # Build role base (includes your original multi_format prompt if enabled)
        role_base = self._make_system_prompt(multi_format=multi_format, role=role)

        # Language must be FIRST so output is actually localized
        lang_line = self._language_directive(language)

        # Localized format lock if multi_format is on (headings in target lang)
        format_lock = self._format_lock_text(language) if multi_format else None

        # Persona (with intensity) – told not to mess with headings/order
        persona_block = self._persona_instructions(character, intensity=intensity) if character else None

        system_top_lines: List[str] = []
        if lang_line:
            system_top_lines.append(lang_line)

        if persona_block:
            system_top_lines.append(
                persona_block + " Do not alter required section headings or their order when present."
            )
            system_top_lines.append(
                "Always maintain this persona unless explicitly told otherwise. "
                "Avoid corporate boilerplate phrasing; be concise and persona-consistent."
            )

        if format_lock:
            system_top_lines.append(format_lock)

        # Humor rule: model appends a grounded joke itself (not post-processed)
        if patricize:
            humor_rule = (
                "Humor rule (Patricize): After you finish your complete answer, append exactly one extra line:\n"
                "PS (Dad joke): <one short, corny, G-rated one-liner>\n"
                "The joke MUST be based on the content of the answer you just wrote (e.g., statuses, risks, [KEY]s, "
                "apps/tools mentioned, trends) — not the user's question wording. Keep it to one sentence. "
                "Do not repeat section headings. If multi-format is enabled, the joke comes after all sections."
            )
            system_top_lines.append(humor_rule)
            if language:
                system_top_lines.append("Apply the language/locale requirement to the dad joke as well.")

        # Your original role/multi-format guidance goes after the hard rules
        system_top_lines.append(role_base)
        system_prompt = "\n\n".join(system_top_lines)

        # ---------------- Build messages ----------------
        messages = self._build_prompt(
            system_prompt=system_prompt,
            question=question,
            hits=hits,
            verbose=verbose,
            format_lock=format_lock,  # final reminder (if any) right before answering
        )

        # ---------------- Call OpenAI ----------------
        temperature = 0.5 if temperature is None else float(temperature)
        max_tokens = 4096 if max_tokens is None else int(max_tokens)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except oai_err.PermissionDeniedError as exc:
            # Fall back to gpt-4-turbo if requested model isn't accessible
            if "model" in str(exc):
                try:
                    log.debug("Model %s not available – falling back to gpt-4-turbo", self.model)
                except Exception:
                    pass
                resp = self.client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=min(max_tokens, 2048),
                )
            else:
                raise

        answer = (resp.choices[0].message.content or "").strip()

        # Persona opener + tag for UI clarity
        if character:
            opener = self._persona_opener(character, intensity=intensity)
            if opener and not answer.startswith(opener):
                answer = f"{opener} {answer}"
            # Prefix the persona tag (matches the dropdown value)
            char_tag = str(character).strip()
            if char_tag:
                answer = f"({char_tag}) {answer}"

        # ---------------- Normalize sources ----------------
        structured = [
            {
                "key": h.get("key"),
                "issue_type": h.get("issue_type"),
                "project": h.get("project_key"),
                "parent": h.get("parent_key"),
                "assignee": h.get("live_assignee") or h.get("assignee"),
                "reporter": h.get("reporter"),
                "priority": h.get("live_priority") or h.get("priority"),
                "resolution": h.get("resolution"),
                "status": h.get("live_status") or h.get("status"),
                "created": h.get("created"),
                "updated": h.get("live_updated") or h.get("updated"),
                "labels": h.get("labels", []),
                "components": h.get("components", []),
                "fix_versions": h.get("fix_versions", []),
                "summary": h.get("summary"),
                "description": (h.get("document") or "").strip(),
            }
            for h in hits
        ]
        return {"answer": answer, "sources": hits, "structured": structured}

    # -------------------------------------------------------------------------
    # System prompt builders (role-aware & multi-format)
    # -------------------------------------------------------------------------
    def _make_system_prompt(self, multi_format: bool, role: Optional[str] = None) -> str:
        """
        Returns the base (role or generic) system prompt.
        When multi_format=True, returns your original 4-section guidance verbatim.
        """
        if role == "developer":
            return (
                "You are a senior Jira-savvy developer. Summarize issues with technical clarity, focusing on code impact, blockers, dependencies, and implementation progress. "
                "Include statuses, priorities, fix versions, and technical labels. Use [KEY] format for references."
            )
        elif role == "manager":
            return (
                "You are a project manager reviewing Jira issues. Your goal is to track task ownership, delays, risks, overdue work, and workload distribution. "
                "Summarize who is responsible, what's at risk, and what requires follow-up. Use [KEY] format to cite issues."
            )
        elif role == "executive":
            return (
                "You are preparing a briefing for senior leadership. Generate a high-level summary of Jira issues across projects, including project health, delivery risk, and resourcing trends. "
                "Do not mention individuals unless critical. Focus on portfolio-level risk and progress signals. Cite issue [KEY]s if relevant."
            )

        if multi_format:
            # Your original four-section guidance (kept as-is)
            return (
                "You are a seasoned Jira expert and analyst tasked with generating comprehensive summaries for a cross-functional audience. "
                "Your goal is to produce verbose, insightful narratives, not lists or field dumps.\n\n"
                "Output must include **four sections**:\n"
                "1. **Detailed Summary** – Paragraphs per issue using metadata fields. Give context, owners, status, risks.\n"
                "2. **Technical Summary** – Developer-oriented overview. Focus on progress, blockers, and priorities.\n"
                "3. **Management Summary** – High-level report for leadership. Include overall status, overdue/risk items, trends.\n"
                "4. **Overall Project Summary** – An integrated narrative summarizing health and risk across all issues provided.\n\n"
                "Use [KEY] when referencing issues."
            )

        # Generic (non-multi-format) narrative prompt, also from your original
        return (
            "You are a senior Jira analyst producing detailed summaries. Always write in verbose paragraph style.\n"
            "Start with an overall project-level overview (issue counts, open/closed status).\n"
            "Then, for each issue, write a paragraph covering:\n"
            "• Summary & description\n• Responsible parties (assignee, reporter)\n"
            "• Lifecycle status (status, resolution, updated date)\n"
            "• Labels, components, fix versions\n"
            "• Urgency or blocking context\n"
            "• Mention issue keys using [KEY] format\n"
            "Conclude with executive-style insights or risk highlights."
        )

    # -------------------------------------------------------------------------
    # Persona & language helpers
    # -------------------------------------------------------------------------
    def _persona_instructions(self, character: Optional[str], intensity: Optional[str] = None) -> str:
        """
        Strong persona directive with intensity controls (light/medium/heavy),
        plus guardrails to avoid corrupting technical details.
        """
        if not character:
            return ""
        c = character.strip().lower()
        lvl = (intensity or "medium").strip().lower()
        if lvl not in {"light", "medium", "heavy"}:
            lvl = "medium"

        freq = {"light": "at least one", "medium": "at least two", "heavy": "most"}[lvl]
        yoda_freq = {"light": "at least one", "medium": "at least two", "heavy": "the majority of"}[lvl]
        pirate_freq = {"light": "1", "medium": "1–2", "heavy": "2–3"}[lvl]

        PERSONAS: Dict[str, str] = {
            "pirate": (
                "Persona: Classic sea pirate. Use nautical slang and occasional 'Arrr'. "
                f"Voice guidance: {pirate_freq} pirate-flavored phrases per paragraph; do not overdo it."
            ),
            "yoda": (
                "Persona: Yoda. Frequently use inverted syntax (object before subject/verb), omit articles at times, "
                "and sprinkle brief interjections ('Hmm.', 'Hrrrm.', 'Yes.'). "
                f"Voice guidance: In EVERY paragraph, {yoda_freq} sentences must use Yoda-style inversion. "
                "Avoid corporate boilerplate phrasing. Be short, wise, and direct."
            ),
            "shakespeare": (
                "Persona: Elizabethan/Shakespearean. Slightly archaic diction ('thee', 'thou', 'hath') and rhythmic flourish. "
                f"Voice guidance: Include {freq} light flourishes per paragraph; keep business terms intact."
            ),
            "executive-snark": (
                "Persona: Executive with dry wit. Brief, incisive, slightly sardonic, but professional. "
                f"Voice guidance: Include {freq} pointed, wry sentences per paragraph; never disrespectful."
            ),
        }

        base_guardrails = (
            "General guardrails: Stay truthful to Jira metadata and retrieval context. "
            "Never invent issue keys or statuses. Do not rename technical fields or labels. "
            "Preserve [KEY] references, numbers, and dates exactly. If persona conflicts with clarity, prefer clarity."
        )
        enforcement = (
            "Enforcement: Each paragraph must visibly exhibit the persona. "
            "Do not alter code tokens, issue keys, statuses, dates, or numeric values."
        )
        persona = PERSONAS.get(
            c,
            f"Persona: {character}. Adopt a consistent, recognizable voice matching this persona. "
            f"Voice guidance: Ensure {freq} sentence(s) per paragraph reflect the requested style.",
        )
        return f"{persona} {base_guardrails} {enforcement}"

    def _persona_opener(self, character: Optional[str], intensity: Optional[str] = None) -> Optional[str]:
        """Small opener to immediately set the persona tone at the start of the answer."""
        if not character:
            return None
        key = character.strip().lower()
        lvl = (intensity or "medium").strip().lower()
        openers = {
            "yoda": "Hmm.",
            "pirate": "Arrr.",
            "shakespeare": "Forsooth,",
            "executive-snark": "Short version:",
        }
        op = openers.get(key)
        if not op:
            return None
        if lvl == "heavy" and key == "yoda":
            return "Hrrrm."
        if lvl == "heavy" and key == "pirate":
            return "Arrr!"
        return op

    def _language_directive(self, language: Optional[str]) -> Optional[str]:
        """
        Returns a system instruction to respond in the requested language/locale.
        Placed FIRST so the model actually switches language.
        """
        if not language:
            return None
        lang = language.strip()
        # Friendly aliases
        alias = {
            "english": "en",
            "francais": "fr",
            "français": "fr",
            "francais (quebec)": "fr-CA",
            "français (québec)": "fr-CA",
            "quebec french": "fr-CA",
            "qc": "fr-CA",
            "pt-br": "pt-BR",
            "zh": "zh-CN",
            "cn": "zh-CN",
        }
        code = alias.get(lang.lower(), lang)
        return (
            "LANGUAGE: Respond entirely in the requested language/locale "
            f"(use '{code}' conventions). Translate all headings and prose accordingly. "
            "Do not include English unless quoting issue keys or code."
        )

    # -------------------------------------------------------------------------
    # Multi-format localization
    # -------------------------------------------------------------------------
    def _localized_headings(self, language: Optional[str]) -> List[str]:
        """
        Return localized headings for the four-section format.
        Fallback: English. Normalizes e.g. 'fr-CA' -> 'fr-ca' -> 'fr'.
        """
        code = (language or "en").lower()
        headings_map: Dict[str, List[str]] = {
            "en": [
                "Detailed Summary",
                "Technical Summary",
                "Management Summary",
                "Overall Project Summary",
            ],
            "fr": [
                "Résumé détaillé",
                "Résumé technique",
                "Résumé pour la direction",
                "Résumé global du projet",
            ],
            "fr-ca": [
                "Résumé détaillé",
                "Résumé technique",
                "Résumé pour la direction",
                "Résumé global du projet",
            ],
            "es": [
                "Resumen detallado",
                "Resumen técnico",
                "Resumen para la dirección",
                "Resumen general del proyecto",
            ],
            "de": [
                "Detaillierte Zusammenfassung",
                "Technische Zusammenfassung",
                "Management-Zusammenfassung",
                "Gesamtzusammenfassung des Projekts",
            ],
            "it": [
                "Riepilogo dettagliato",
                "Riepilogo tecnico",
                "Riepilogo per la direzione",
                "Riepilogo complessivo del progetto",
            ],
            "pt-br": [
                "Resumo detalhado",
                "Resumo técnico",
                "Resumo para a diretoria",
                "Resumo geral do projeto",
            ],
            "ja": [
                "詳細サマリー",
                "技術サマリー",
                "マネジメントサマリー",
                "プロジェクト全体のサマリー",
            ],
            "ko": [
                "상세 요약",
                "기술 요약",
                "경영 요약",
                "프로젝트 전반 요약",
            ],
            "zh-cn": [
                "详细摘要",
                "技术摘要",
                "管理摘要",
                "项目总体摘要",
            ],
        }
        # Normalize region codes: try base language fallback
        if code not in headings_map and "-" in code:
            base = code.split("-")[0]
            if base in headings_map:
                code = base
        return headings_map.get(code, headings_map["en"])

    def _format_lock_text(self, language: Optional[str]) -> str:
        """
        Hard rule for multi-format outputs with headings localized to the target language.
        This appears TWICE: once at the top, and again right before the user message.
        """
        h1, h2, h3, h4 = self._localized_headings(language)
        return (
            "FORMAT LOCK: Produce exactly four sections with these exact headings, in this order:\n"
            f"1. {h1}\n"
            f"2. {h2}\n"
            f"3. {h3}\n"
            f"4. {h4}\n"
            "Do not rename the headings. If a section has no content, write 'No relevant items.'"
        )

    # -------------------------------------------------------------------------
    # Prompt assembly
    # -------------------------------------------------------------------------
    @staticmethod
    def _build_prompt(
        system_prompt: str,
        question: str,
        hits: List[Dict[str, Any]],
        verbose: bool = False,
        format_lock: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Assemble the chat messages in order:
          1) system (rules/persona/format/role)
          2) system (Context: ... compiled from hits)
          3) system (final format lock reminder, if any)
          4) user (question)
        """
        # Build context blocks
        context_blocks: List[str] = []
        for h in hits:
            status = h.get("live_status") or h.get("status") or "-"
            assignee = h.get("live_assignee") or h.get("assignee") or "-"
            reporter = h.get("reporter") or "-"
            priority = h.get("live_priority") or h.get("priority") or "-"
            resolution = h.get("resolution") or "-"
            created = h.get("created") or "-"
            updated = h.get("live_updated") or h.get("updated") or "-"
            labels = ", ".join(h.get("labels", [])) or "-"
            comps = ", ".join(h.get("components", [])) or "-"
            fixes = ", ".join(h.get("fix_versions", [])) or "-"

            doc_text = (h.get("document") or "").strip()
            if not doc_text:
                # Safe fallback when there's no description text available
                doc_text = (
                    f"No description provided for this issue titled '{h.get('summary', 'Untitled')}' "
                    f"in project {h.get('project_key', '-')}. It is classified as {h.get('issue_type', '-')}, "
                    f"currently {status}, and was last updated on {updated}."
                )

            if verbose:
                block = (
                    f"[{h['key']}] ({h.get('issue_type', '-')})\n"
                    f"Project: {h.get('project_key', '-')}"
                    f" | Assignee: {assignee} | Reporter: {reporter}\n"
                    f"Status: {status} | Resolution: {resolution} | Priority: {priority}\n"
                    f"Created: {created} | Updated: {updated}\n"
                    f"Labels: {labels} | Components: {comps} | Fix Versions: {fixes}\n"
                    f"Description:\n{doc_text}\n––––––\n"
                )
            else:
                block = (
                    f"[{h['key']}] – {h.get('summary', 'No summary')}. "
                    f"Status: {status}. Assignee: {assignee}. Updated: {updated}."
                )
            context_blocks.append(block)

        context_txt = "\n".join(context_blocks)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Context:\n{context_txt}"},
        ]

        # Final format reminder just before answering (helps the model obey headings)
        if format_lock:
            messages.append({"role": "system", "content": format_lock})

        messages.append({"role": "user", "content": question})
        return messages


__all__ = ["ChatService"]
