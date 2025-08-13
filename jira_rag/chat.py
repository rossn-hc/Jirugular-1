#!/usr/bin/env python
"""
chat.py – ChatService that blends semantic recall (FAISS) with Jira, MS Graph People, and MS Graph Sign-ins data,
with optional verbose mode, expanded metadata context, and enhanced summaries.

Features:
- Datasource-aware: "jira" (project/issues) vs "people" (MS Graph /users) vs "signins" (MS Graph /auditLogs/signIns)
- Personas via `character` (pirate, yoda, shakespeare, executive-snark, or custom)
- Persona intensity: light | medium | heavy
- Multi-format 4-section output with localized headings + hard FORMAT LOCK
- Language directive first (e.g., "en", "fr-CA", "fr", "es", "de"...)
- Temperature / max_tokens knobs
- Role-aware base prompts (developer, manager, executive) for Jira
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

    # ---------------- NEW: persona normalizer ----------------
    def _normalize_persona(self, character: Optional[str]) -> Optional[str]:
        """Normalize UI-provided persona; treat 'Default', 'None', etc. as no persona."""
        if character is None:
            return None
        s = str(character).strip().lower()
        if s in {"", "none", "default", "off", "no"}:
            return None
        return s

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
        # model appends a grounded dad joke itself
        patricize: bool = False,
        # "jira" | "people" (msgraph) | "signins" (msgraph audit logs)  # --- SIGNINS ---
        datasource: str = "jira",
    ) -> Dict[str, Any]:
        """
        Build a datasource-aware prompt, retrieve the right context, call OpenAI,
        and return {"answer", "sources", "structured"}.
        """
        # Normalize datasource
        ds = (datasource or "jira").strip().lower()
        if ds not in {"jira", "people", "signins"}:  # --- SIGNINS ---
            ds = "jira"

        # Legacy pirate -> character; then normalize persona & intensity (NEW)
        if character is None and pirate is not None:
            character = "pirate" if pirate else None
            try:
                log.debug("Deprecated arg 'pirate' used; mapped to character=%r", character)
            except Exception:
                pass
        character = self._normalize_persona(character)

        lvl = (intensity or "medium").strip().lower()
        if lvl not in {"light", "medium", "heavy"}:
            lvl = "medium"
        # Stronger style adherence for Yoda in multi-format
        if character == "yoda" and multi_format and lvl == "medium":
            lvl = "heavy"

        # ---------------- Retrieval ----------------
        if ds in {"people", "signins"}:  # --- SIGNINS ---
            # Pure FAISS search; do NOT call Jira APIs
            q_vec = self.retriever.embedder.encode_one(question)
            hits = self.retriever.indexer.search(q_vec, k=top_k)
        else:
            # Jira path: FAISS + live refresh
            hits = self.retriever.retrieve(question, k=top_k)

        # ---------------- Dedupe ----------------
        unique_hits: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for h in hits:
            if ds == "people":
                dedupe_key = (
                    h.get("userPrincipalName")
                    or h.get("mail")
                    or h.get("displayName")
                    or h.get("id")
                    or h.get("document")
                )
            elif ds == "signins":  # --- SIGNINS ---
                dedupe_key = (
                    h.get("id")
                    or f"{h.get('userPrincipalName')}|{h.get('createdDateTime')}|{h.get('appDisplayName')}"
                    or h.get("document")
                )
            else:
                dedupe_key = h.get("key") or h.get("issue_key")
            kstr = str(dedupe_key) if dedupe_key else None
            if not kstr or kstr not in seen:
                if kstr:
                    seen.add(kstr)
                unique_hits.append(h)
        hits = unique_hits

        # ---------------- System prompt stack ----------------
        lang_line = self._language_directive(language)
        role_base = self._make_system_prompt(multi_format=multi_format, role=role, datasource=ds, language=language)
        format_lock = self._format_lock_text(language, ds) if multi_format else None
        persona_block = self._persona_instructions(character, intensity=lvl) if character else None

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

        if patricize:
            humor_rule = (
                "Humor rule (Patricize): After you finish your complete answer, append exactly one extra line:\n"
                "PS (Dad joke): <one short, corny, G-rated one-liner>\n"
                "The joke MUST be grounded in the content of the answer you just wrote — not the user's question wording. "
                "Keep it to one sentence. If multi-format is enabled, the joke comes after all sections."
            )
            system_top_lines.append(humor_rule)
            if language:
                system_top_lines.append("Apply the language/locale requirement to the dad joke as well.")

        system_top_lines.append(role_base)
        system_prompt = "\n\n".join(system_top_lines)

        # ---------------- NEW: persona reminder just-in-time ----------------
        persona_reminder = None
        if character:
            persona_reminder = (
                "Persona enforcement:\n"
                "- Apply the requested persona consistently in EVERY paragraph.\n"
                "- Meet the per-paragraph quota described above (e.g., Yoda inversion count).\n"
                "- If any paragraph fails the quota, rewrite that paragraph before finalizing.\n"
                "- Keep section headings and all data values exactly as given."
            )

        # ---------------- Build messages ----------------
        messages = self._build_prompt(
            system_prompt=system_prompt,
            question=question,
            hits=hits,
            verbose=verbose,
            format_lock=format_lock,
            datasource=ds,
            persona_reminder=persona_reminder,  # NEW
        )

        # ---------------- Call OpenAI ----------------
        base_temp = 0.5 if temperature is None else float(temperature)
        if character:
            # Lower temp helps stick to style rules; don't go below 0.2
            base_temp = min(base_temp, 0.35)
        temperature = base_temp
        max_tokens = 4096 if max_tokens is None else int(max_tokens)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except oai_err.PermissionDeniedError as exc:
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

        # Persona opener + tag
        if character:
            opener = self._persona_opener(character, intensity=lvl)
            if opener and not answer.startswith(opener):
                answer = f"{opener} {answer}"
            tag = str(character).strip()
            if tag:
                answer = f"({tag}) {answer}"

        # ---------------- Normalize sources ----------------
        if ds == "people":
            structured = [
                {
                    "displayName": h.get("displayName"),
                    "userPrincipalName": h.get("userPrincipalName") or h.get("mail"),
                    "mail": h.get("mail"),
                    "jobTitle": h.get("jobTitle"),
                    "department": h.get("department"),
                    "accountEnabled": h.get("accountEnabled"),
                    "document": (h.get("document") or "").strip(),
                }
                for h in hits
            ]
        elif ds == "signins":  # --- SIGNINS ---
            structured = [
                {
                    "id": h.get("id"),
                    "createdDateTime": h.get("createdDateTime"),
                    "appDisplayName": h.get("appDisplayName"),
                    "userDisplayName": h.get("userDisplayName"),
                    "userPrincipalName": h.get("userPrincipalName"),
                    "ipAddress": h.get("ipAddress"),
                    "clientAppUsed": h.get("clientAppUsed"),
                    "operatingSystem": h.get("operatingSystem"),
                    "browser": h.get("browser"),
                    "city": h.get("city"),
                    "countryOrRegion": h.get("countryOrRegion"),
                    "result": h.get("result"),
                    "document": (h.get("document") or "").strip(),
                }
                for h in hits
            ]
        else:
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
    # System prompt builders (role-aware & datasource-aware)
    # -------------------------------------------------------------------------
    def _make_system_prompt(
        self,
        multi_format: bool,
        role: Optional[str],
        datasource: str = "jira",
        language: Optional[str] = None,
    ) -> str:
        """
        Returns the base system prompt tailored to the datasource.
        - datasource == "jira": project/issue summaries
        - datasource == "people": org/people summaries (MS Graph /users)
        - datasource == "signins": auth activity summaries (MS Graph /auditLogs/signIns)
        """
        ds = (datasource or "jira").lower()

        if ds == "people":
            if multi_format:
                return (
                    "You are an HR/People analytics assistant summarizing Microsoft 365 directory data (Microsoft Graph /users). "
                    "Use only the fields provided in context (displayName, userPrincipalName, mail, jobTitle, department, accountEnabled, plus a short document string). "
                    "Answer plainly about people, roles, and organizational hints. Avoid Jira terminology.\n\n"
                    "Output must include **four sections**:\n"
                    "1. **People Overview** – Who is in scope (counts, notable departments/titles, unknowns).\n"
                    "2. **Managers & Leads (Inferred)** – Based on job titles only (e.g., titles containing 'Manager', 'Lead', 'Director'). Call out uncertainty.\n"
                    "3. **Org Signals** – Any department or naming patterns that indicate teams or functions.\n"
                    "4. **Actions & Follow-ups** – Data gaps (missing titles/emails), suggested clarifications.\n"
                )
            return (
                "You are an HR/People analytics assistant summarizing Microsoft Graph /users records. "
                "Use displayName, userPrincipalName/mail, jobTitle, department, accountEnabled, and the provided notes. "
                "Answer about people and teams; do not mention Jira."
            )

        if ds == "signins":  # --- SIGNINS ---
            if multi_format:
                return (
                    "You are a security ops analyst summarizing Microsoft Graph Audit Logs sign-in events (/auditLogs/signIns). "
                    "Use only the fields in context (createdDateTime, appDisplayName, userDisplayName, userPrincipalName, ipAddress, clientAppUsed, operatingSystem, browser, city, countryOrRegion, result) plus the provided document text. "
                    "Identify failure patterns, risky geographies/devices, and actionable follow-ups.\n\n"
                    "Output must include **four sections**:\n"
                    "1. **Auth Activity Overview** – Volume, time window implied by the context, notable apps/users.\n"
                    "2. **Failures & Risk Signals** – Error trends, repeated failures, impossible travel hints, suspicious IP/device patterns.\n"
                    "3. **Geo & Device Patterns** – Cities/countries, OS/browser clusters, client app usage anomalies.\n"
                    "4. **Actions & Queries** – Concrete next steps (KQL/MS Graph filters, MFA checks, conditional access review).\n"
                )
            return (
                "You are a security ops analyst summarizing Microsoft Graph Audit Logs sign-in events. "
                "Focus on failures, anomalies, and practical next steps. Keep findings concise and actionable."
            )

        # ---- Jira (default path) ----
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
        if not character:
            return ""
        c = character.strip().lower()
        lvl = (intensity or "medium").strip().lower()
        if lvl not in {"light", "medium", "heavy"}:
            lvl = "medium"

        # Quantitative per-paragraph quotas
        yoda_quota = {"light": 1, "medium": 2, "heavy": 3}[lvl]
        pirate_quota = {"light": 1, "medium": 2, "heavy": 3}[lvl]
        shakespeare_quota = {"light": 1, "medium": 2, "heavy": 3}[lvl]
        execsnark_quota = {"light": 1, "medium": 2, "heavy": 3}[lvl]

        # Global style rules boost consistency
        global_rules = (
            "Style rules:\n"
            "- Keep average sentence length under ~18 words (concise).\n"
            "- Prefer simple clauses over compound, to maintain tone.\n"
            "- Avoid generic corporate phrasing.\n"
        )

        PERSONAS: Dict[str, str] = {
            "pirate": (
                "Persona: Classic sea pirate.\n"
                f"- Quota: include ≥{pirate_quota} pirate-flavored phrases per paragraph (nautical slang, 'Arrr').\n"
                "- Maintain clarity; never obscure identifiers.\n"
            ),
            "yoda": (
                "Persona: Yoda.\n"
                "- Use inverted syntax frequently (object before subject/verb) and brief interjections ('Hmm.', 'Hrrrm.', 'Yes.').\n"
                f"- Quota: include ≥{yoda_quota} Yoda-style **inverted** sentences per paragraph.\n"
                "- Example templates: 'Open the issue remains.'  'At risk, this project is.'  'Blocked by X, the team is.'\n"
            ),
            "shakespeare": (
                "Persona: Elizabethan/Shakespearean.\n"
                f"- Quota: include ≥{shakespeare_quota} light Elizabethan flourishes per paragraph.\n"
            ),
            "executive-snark": (
                "Persona: Executive with dry wit.\n"
                f"- Quota: include ≥{execsnark_quota} wry, incisive lines per paragraph.\n"
                "- Keep it professional; never disrespectful.\n"
            ),
        }

        base_guardrails = (
            "Truthfulness & data rules:\n"
            "- Stay strictly grounded in provided context.\n"
            "- Do not invent fields or values; preserve identifiers, numbers, statuses, and dates exactly.\n"
            "- If persona conflicts with clarity, prefer clarity but still meet the quota.\n"
        )
        enforcement = (
            "Self-check:\n"
            "- For EACH paragraph, verify the quota is met; if not, rewrite that paragraph before finalizing.\n"
        )
        persona = PERSONAS.get(
            c,
            f"Persona: {character}. Maintain a consistent, recognizable voice in every paragraph. "
            f"Quota: include ≥{ {'light':1,'medium':2,'heavy':3}[lvl] } persona-typical sentences per paragraph.\n"
        )
        return f"{persona}\n{global_rules}{base_guardrails}{enforcement}"

    def _persona_opener(self, character: Optional[str], intensity: Optional[str] = None) -> Optional[str]:
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
        Force all narrative/headings/jokes to the selected language/locale,
        but KEEP every value from Context (names, titles, depts, keys, UPNs, labels, statuses)
        exactly as given (no translation/normalization).
        """
        if not language:
            return None
        lang = language.strip()
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
            "LANGUAGE POLICY:\n"
            f"- Write ALL explanatory text, section headings, and conclusions in '{code}'.\n"
            "- DO NOT translate, rewrite, or normalize any literal values coming from the Context block: "
            "names, job titles, departments, group names, issue keys, statuses, labels, UPNs/emails, IDs, dates. "
            "Quote them verbatim as data.\n"
            "- If a value appears in another language/script in Context, leave it as-is. "
            "Only the surrounding narration/headings should be localized.\n"
            "- When listing fields, keep their values exactly as given; only the connective prose is localized."
        )

    # -------------------------------------------------------------------------
    # Multi-format localization
    # -------------------------------------------------------------------------
    def _localized_headings_jira(self, language: Optional[str]) -> List[str]:
        code = (language or "en").lower()
        headings_map: Dict[str, List[str]] = {
            "en": ["Detailed Summary", "Technical Summary", "Management Summary", "Overall Project Summary"],
            "fr": ["Résumé détaillé", "Résumé technique", "Résumé pour la direction", "Résumé global du projet"],
            "fr-ca": ["Résumé détaillé", "Résumé technique", "Résumé pour la direction", "Résumé global du projet"],
            "es": ["Resumen detallado", "Resumen técnico", "Resumen para la dirección", "Resumen general del proyecto"],
            "de": ["Detaillierte Zusammenfassung", "Technische Zusammenfassung", "Management-Zusammenfassung", "Gesamtzusammenfassung des Projekts"],
            "it": ["Riepilogo dettagliato", "Riepilogo tecnico", "Riepilogo per la direzione", "Riepilogo complessivo del progetto"],
            "pt-br": ["Resumo detalhado", "Resumo técnico", "Resumo para a diretoria", "Resumo geral do projeto"],
            "ja": ["詳細サマリー", "技術サマリー", "マネジメントサマリー", "プロジェクト全体のサマリー"],
            "ko": ["상세 요약", "기술 요약", "경영 요약", "프로젝트 전반 요약"],
            "zh-cn": ["详细摘要", "技术摘要", "管理摘要", "项目总体摘要"],
        }
        if code not in headings_map and "-" in code:
            base = code.split("-")[0]
            if base in headings_map:
                code = base
        return headings_map.get(code, headings_map["en"])

    def _localized_headings_people(self, language: Optional[str]) -> List[str]:
        code = (language or "en").lower()
        headings_map: Dict[str, List[str]] = {
            "en": ["People Overview", "Managers & Leads (Inferred)", "Org Signals", "Actions & Follow-ups"],
            "fr": ["Aperçu des personnes", "Gestionnaires et responsables (inférés)", "Signaux d’organisation", "Actions et suivis"],
            "fr-ca": ["Aperçu des personnes", "Gestionnaires et responsables (inférés)", "Signaux d’organisation", "Actions et suivis"],
            "es": ["Resumen de personas", "Gerentes y líderes (inferidos)", "Señales de organización", "Acciones y seguimientos"],
        }
        if code not in headings_map and "-" in code:
            base = code.split("-")[0]
            if base in headings_map:
                code = base
        return headings_map.get(code, headings_map["en"])

    def _localized_headings_signins(self, language: Optional[str]) -> List[str]:  # --- SIGNINS ---
        code = (language or "en").lower()
        headings_map: Dict[str, List[str]] = {
            "en": ["Auth Activity Overview", "Failures & Risk Signals", "Geo & Device Patterns", "Actions & Queries"],
            "fr": ["Aperçu de l’activité d’authentification", "Échecs et signaux de risque", "Schémas géo et appareils", "Actions et requêtes"],
            "fr-ca": ["Aperçu de l’activité d’authentification", "Échecs et signaux de risque", "Schémas géo et appareils", "Actions et requêtes"],
            "es": ["Resumen de actividad de autenticación", "Fallos y señales de riesgo", "Patrones geográficos y de dispositivos", "Acciones y consultas"],
        }
        if code not in headings_map and "-" in code:
            base = code.split("-")[0]
            if base in headings_map:
                code = base
        return headings_map.get(code, headings_map["en"])

    def _format_lock_text(self, language: Optional[str], datasource: str) -> str:
        ds = (datasource or "jira").lower()
        if ds == "people":
            h1, h2, h3, h4 = self._localized_headings_people(language)
        elif ds == "signins":  # --- SIGNINS ---
            h1, h2, h3, h4 = self._localized_headings_signins(language)
        else:
            h1, h2, h3, h4 = self._localized_headings_jira(language)
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
        datasource: str = "jira",
        persona_reminder: Optional[str] = None,  # NEW
    ) -> List[Dict[str, str]]:
        """
        Assemble the chat messages in order:
          1) system (rules/persona/format/role)
          2) system (Context: ... compiled from hits)
          3) system (final format lock reminder, if any)
          4) user (question)
        """
        ds = (datasource or "jira").lower()

        context_blocks: List[str] = []
        for h in hits:
            if ds == "people":
                name = h.get("displayName") or "-"
                upn = h.get("userPrincipalName") or h.get("mail") or "-"
                email = h.get("mail") or "-"
                title = h.get("jobTitle") or "-"
                dept = h.get("department") or "-"
                acct = h.get("accountEnabled")
                acct_str = "Enabled" if acct is True else ("Disabled" if acct is False else "-")
                doc_text = (h.get("document") or "").strip()

                if verbose:
                    block = (
                        f"{name} ({upn})\n"
                        f"Title: {title} | Department: {dept} | Account: {acct_str} | Email: {email}\n"
                        f"Notes:\n{(doc_text or '—')}\n––––––\n"
                    )
                else:
                    block = f"{name} – {title}. Dept: {dept}. Email: {email}. UPN: {upn}. Account: {acct_str}."
            elif ds == "signins":  # --- SIGNINS ---
                when = h.get("createdDateTime") or "-"
                app = h.get("appDisplayName") or "-"
                uname = h.get("userDisplayName") or "-"
                upn = h.get("userPrincipalName") or "-"
                ip = h.get("ipAddress") or "-"
                cap = h.get("clientAppUsed") or "-"
                osn = h.get("operatingSystem") or "-"
                brw = h.get("browser") or "-"
                city = h.get("city") or "-"
                country = h.get("countryOrRegion") or "-"
                result = h.get("result") or "-"
                doc_text = (h.get("document") or "").strip()

                if verbose:
                    block = (
                        f"{when} | {app}\n"
                        f"User: {uname} ({upn}) | Result: {result}\n"
                        f"IP: {ip} | Client: {cap} | OS: {osn} | Browser: {brw}\n"
                        f"Location: {city}, {country}\n"
                        f"Notes:\n{(doc_text or '—')}\n––––––\n"
                    )
                else:
                    block = (
                        f"{when} – {app}. {uname} ({upn}). Result: {result}. "
                        f"IP {ip}. {city}, {country}. Client {cap}. OS {osn}. Browser {brw}."
                    )
            else:
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
                    doc_text = (
                        f"No description provided for this issue titled '{h.get('summary', 'Untitled')}' "
                        f"in project {h.get('project_key', '-')}. It is classified as {h.get('issue_type', '-')}, "
                        f"currently {status}, and was last updated on {updated}."
                    )

                if verbose:
                    block = (
                        f"[{h.get('key', 'UNKNOWN')}] ({h.get('issue_type', '-')})\n"
                        f"Project: {h.get('project_key', '-')}"
                        f" | Assignee: {assignee} | Reporter: {reporter}\n"
                        f"Status: {status} | Resolution: {resolution} | Priority: {priority}\n"
                        f"Created: {created} | Updated: {updated}\n"
                        f"Labels: {labels} | Components: {comps} | Fix Versions: {fixes}\n"
                        f"Description:\n{doc_text}\n––––––\n"
                    )
                else:
                    block = (
                        f"[{h.get('key', 'UNKNOWN')}] – {h.get('summary', 'No summary')}. "
                        f"Status: {status}. Assignee: {assignee}. Updated: {updated}."
                    )
            context_blocks.append(block)

        context_txt = "\n".join(context_blocks)

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Context:\n{context_txt}"},
        ]

        if format_lock:
            messages.append({"role": "system", "content": format_lock})

        if persona_reminder:
            messages.append({"role": "system", "content": persona_reminder})

        messages.append({"role": "user", "content": question})
        return messages


__all__ = ["ChatService"]
