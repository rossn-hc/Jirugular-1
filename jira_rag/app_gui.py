#!/usr/bin/env python
"""
app_gui.py – Enhanced UI Dialog Box with Chat Window for Jira-RAG
Adds MS Graph (People) and MS Graph (Sign-ins) as separate datasources with their own FAISS stems.
"""

import tkinter as tk
from tkinter import simpledialog, messagebox, scrolledtext

from jira_rag.interface import (
    crawl_and_build,               # JIRA crawl
    crawl_msgraph_people,          # MS Graph People crawl
    crawl_msgraph_signins,         # MS Graph Sign-ins crawl (NEW)
    show_dependencies,
    ask_question,
)

# ---------- UI options ----------
PERSONA_OPTIONS = ["", "pirate", "yoda", "shakespeare", "executive-snark"]
ROLE_OPTIONS = ["", "developer", "manager", "executive"]
INTENSITY_OPTIONS = ["light", "medium", "heavy"]
LANGUAGE_OPTIONS = ["en", "fr-CA", "fr", "es", "de", "it", "pt-BR", "ja", "ko", "zh-CN"]
MAX_TOKEN_OPTIONS = [512, 1024, 2048, 4096]

# Data source -> FAISS stem mapping
STEMS = {
    "jira": "jira_vectors",
    "msgraph": "msgraph_people",
    "signins": "msgraph_signins",   # NEW
}


class JiraRAGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Jira-RAG Interactive")
        self.create_main_widgets()

    def create_main_widgets(self):
        tk.Label(self.root, text="Jira-RAG Interactive", font=('Helvetica', 16, 'bold')).pack(pady=10)

        # Crawl buttons
        btns = tk.Frame(self.root)
        btns.pack(fill='x', padx=20, pady=5)

        tk.Button(btns, text="Crawl / rebuild JIRA index", command=self.crawl_jira).pack(fill='x', pady=3)
        tk.Button(btns, text="Crawl / rebuild MS Graph (People) index", command=self.crawl_msgraph_people).pack(fill='x', pady=3)
        tk.Button(btns, text="Crawl / rebuild MS Graph (Sign-ins) index", command=self.crawl_msgraph_signins).pack(fill='x', pady=3)  # NEW

        # Chat buttons (open windows bound to specific stem)
        chats = tk.Frame(self.root)
        chats.pack(fill='x', padx=20, pady=5)
        tk.Button(chats, text="Open Chat (JIRA index)", command=lambda: self.open_chat_window("jira")).pack(fill='x', pady=3)
        tk.Button(chats, text="Open Chat (MS Graph – People)", command=lambda: self.open_chat_window("msgraph")).pack(fill='x', pady=3)
        tk.Button(chats, text="Open Chat (MS Graph – Sign-ins)", command=lambda: self.open_chat_window("signins")).pack(fill='x', pady=3)  # NEW

        tk.Button(self.root, text="Show dependencies for an issue", command=self.dependencies).pack(fill='x', padx=20, pady=5)
        tk.Button(self.root, text="Quit", command=self.root.quit).pack(fill='x', padx=20, pady=5)

    # ----------------- Crawl actions -----------------
    def crawl_jira(self):
        jql = self.show_jql_builder()
        if jql is None:
            return
        try:
            crawl_and_build(jql=jql, stem=STEMS["jira"])
            messagebox.showinfo("Success", f"JIRA crawl completed.\nJQL: {jql}\nStem: {STEMS['jira']}")
        except Exception as e:
            messagebox.showerror("Error", f"Crawl failed:\n{e}")

    def crawl_msgraph_people(self):
        # You can prompt for 'top' if desired; hardcode small default for PoC
        try:
            count = crawl_msgraph_people(stem=STEMS["msgraph"], top=500)
            messagebox.showinfo("Success", f"MS Graph People crawl completed.\nIndexed: {count} records\nStem: {STEMS['msgraph']}")
        except AttributeError:
            messagebox.showerror("Unavailable", "The MS Graph People crawl function is not available. Please add it to interface.py.")
        except Exception as e:
            messagebox.showerror("Error", f"MS Graph People crawl failed:\n{e}")

    def crawl_msgraph_signins(self):
        """
        Simple prompts for optional date window and app filter.
        Accepts YYYY-MM-DD or full ISO (YYYY-MM-DDThh:mm:ssZ). Leaving blank skips the filter.
        """
        try:
            start_date = simpledialog.askstring("Sign-ins Crawl", "Start date (YYYY-MM-DD or ISO), optional:")
            end_date = simpledialog.askstring("Sign-ins Crawl", "End date (YYYY-MM-DD or ISO), optional:")
            app_name = simpledialog.askstring("Sign-ins Crawl", "Filter by App Display Name (optional):")
            # Optional: limit fetch (debugging / PoC)
            top_s = simpledialog.askstring("Sign-ins Crawl", "Top (limit results; optional integer):")
            top = int(top_s) if (top_s and top_s.strip().isdigit()) else None

            count = crawl_msgraph_signins(
                stem=STEMS["signins"],
                start_date=(start_date or None),
                end_date=(end_date or None),
                app_display_name=(app_name or None),
                top=top,
            )
            messagebox.showinfo(
                "Success",
                f"MS Graph Sign-ins crawl completed.\nIndexed: {count} records\nStem: {STEMS['signins']}"
            )
        except AttributeError:
            messagebox.showerror("Unavailable", "The MS Graph Sign-ins crawl function is not available. Please add it to interface.py.")
        except Exception as e:
            messagebox.showerror("Error", f"MS Graph Sign-ins crawl failed:\n{e}")

    # ----------------- JQL builder -----------------
    def show_jql_builder(self):
        popup = tk.Toplevel(self.root)
        popup.title("Advanced JQL Builder")
        popup.geometry("600x200")

        fields = ['project', 'issuetype', 'status', 'assignee', 'reporter', 'priority', 'labels', 'created', 'updated']
        operators = ['=', '!=', '~', '>', '<', '>=', '<=']

        field_var = tk.StringVar(value=fields[0])
        operator_var = tk.StringVar(value='=')
        value_var = tk.StringVar()

        form_frame = tk.Frame(popup)
        form_frame.pack(pady=10)

        tk.OptionMenu(form_frame, field_var, *fields).grid(row=0, column=0, padx=5)
        tk.OptionMenu(form_frame, operator_var, *operators).grid(row=0, column=1, padx=5)
        tk.Entry(form_frame, textvariable=value_var, width=40).grid(row=0, column=2, padx=5)

        jql_preview = tk.StringVar(value="")
        preview_label = tk.Label(popup, textvariable=jql_preview, wraplength=580, justify='left')
        preview_label.pack(pady=10)

        jql_clauses = []

        def add_clause():
            clause = f'{field_var.get()} {operator_var.get()} "{value_var.get()}"'
            jql_clauses.append(clause)
            jql_preview.set(" AND ".join(jql_clauses))
            value_var.set("")

        def clear_clauses():
            jql_clauses.clear()
            jql_preview.set("")

        button_frame = tk.Frame(popup)
        button_frame.pack(pady=5)

        tk.Button(button_frame, text="Add Condition", command=add_clause).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Clear", command=clear_clauses).pack(side=tk.LEFT, padx=5)

        result = {}

        def on_submit():
            result['jql'] = jql_preview.get()
            popup.destroy()

        def on_cancel():
            popup.destroy()

        tk.Button(popup, text="Submit", command=on_submit).pack(side=tk.LEFT, padx=20, pady=10)
        tk.Button(popup, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=20, pady=10)

        popup.grab_set()
        self.root.wait_window(popup)
        return result.get('jql')

    # ----------------- Chat window -----------------
    def open_chat_window(self, datasource: str):
        """Open a chat window bound to a specific datasource stem."""
        stem = STEMS.get(datasource, STEMS["jira"])

        chat_win = tk.Toplevel(self.root)
        title_map = {
            "jira": "JIRA",
            "msgraph": "MS Graph (People)",
            "signins": "MS Graph (Sign-ins)",
        }
        chat_win.title(f"Chat Window – {title_map.get(datasource, 'JIRA')}")
        chat_win.geometry("980x560")

        # Keep stem on the instance
        chat_win._stem = stem

        options_frame = tk.Frame(chat_win)
        options_frame.pack(side=tk.TOP, fill='x', padx=5, pady=5)

        # Top-K
        tk.Label(options_frame, text="Top-K:").grid(row=0, column=0, padx=5, sticky="w")
        self.top_k_var = tk.IntVar(value=5)
        tk.Spinbox(options_frame, from_=1, to=100, textvariable=self.top_k_var, width=6).grid(row=0, column=1, sticky="w")

        # Role dropdown (used for Jira; harmless for others)
        tk.Label(options_frame, text="Role:").grid(row=0, column=2, padx=(12, 5), sticky="e")
        self.role_var = tk.StringVar(value="")
        tk.OptionMenu(options_frame, self.role_var, *ROLE_OPTIONS).grid(row=0, column=3, sticky="w")

        # Persona dropdown
        tk.Label(options_frame, text="Persona:").grid(row=0, column=4, padx=(12, 5), sticky="e")
        self.persona_var = tk.StringVar(value="")
        tk.OptionMenu(options_frame, self.persona_var, *PERSONA_OPTIONS).grid(row=0, column=5, sticky="w")

        # Verbose + Multi-format + Patricize
        self.verbose_var = tk.BooleanVar(value=False)
        self.multiformat_var = tk.BooleanVar(value=False)
        self.patricize_var = tk.BooleanVar(value=False)
        tk.Checkbutton(options_frame, text="Verbose", variable=self.verbose_var).grid(row=1, column=0, padx=5, pady=(6,0), sticky="w")
        tk.Checkbutton(options_frame, text="Multi-format", variable=self.multiformat_var).grid(row=1, column=1, padx=5, pady=(6,0), sticky="w")
        tk.Checkbutton(options_frame, text="Patricize (Dad joke)", variable=self.patricize_var).grid(row=1, column=2, padx=(12,5), sticky="w")

        # Persona Intensity
        tk.Label(options_frame, text="Intensity:").grid(row=1, column=3, padx=(12,5), sticky="e")
        self.intensity_var = tk.StringVar(value="medium")
        tk.OptionMenu(options_frame, self.intensity_var, *INTENSITY_OPTIONS).grid(row=1, column=4, sticky="w")

        # Temperature
        tk.Label(options_frame, text="Temperature:").grid(row=1, column=5, padx=(12,5), sticky="e")
        self.temperature_var = tk.DoubleVar(value=0.5)
        tk.Spinbox(options_frame, from_=0.0, to=1.5, increment=0.1, textvariable=self.temperature_var, width=6).grid(row=1, column=6, sticky="w")

        # Max tokens
        tk.Label(options_frame, text="Max tokens:").grid(row=2, column=0, padx=(5,5), sticky="w")
        self.max_tokens_var = tk.IntVar(value=2048)
        tk.OptionMenu(options_frame, self.max_tokens_var, *MAX_TOKEN_OPTIONS).grid(row=2, column=1, sticky="w")

        # Language
        tk.Label(options_frame, text="Language:").grid(row=2, column=2, padx=(12,5), sticky="e")
        self.language_var = tk.StringVar(value="en")
        tk.OptionMenu(options_frame, self.language_var, *LANGUAGE_OPTIONS).grid(row=2, column=3, sticky="w")

        # Chat display
        self.chat_display = scrolledtext.ScrolledText(chat_win, state='disabled', height=18)
        self.chat_display.pack(padx=10, pady=5, fill='both', expand=True)

        # Entry + Ask button
        entry_frame = tk.Frame(chat_win)
        entry_frame.pack(side=tk.BOTTOM, fill='x', padx=5, pady=5)

        self.question_entry = tk.Entry(entry_frame)
        self.question_entry.pack(side=tk.LEFT, fill='x', expand=True, padx=(0, 5))
        tk.Button(entry_frame, text="Ask", command=lambda win=chat_win: self.ask_question(win)).pack(side=tk.RIGHT)

    def ask_question(self, chat_win):
        question = self.question_entry.get().strip()
        if not question:
            return

        self.question_entry.delete(0, tk.END)
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, f"Q: {question}\n")

        # Collect options
        top_k = int(self.top_k_var.get())
        role = (self.role_var.get() or "").strip() or None
        character = (self.persona_var.get() or "").strip() or None
        intensity = (self.intensity_var.get() or "").strip() or None
        temperature = float(self.temperature_var.get())
        max_tokens = int(self.max_tokens_var.get())
        language = (self.language_var.get() or "").strip() or None
        verbose = bool(self.verbose_var.get())
        multi_format = bool(self.multiformat_var.get())
        patricize = bool(self.patricize_var.get())
        stem = getattr(chat_win, "_stem", "jira_vectors")

        # Try new signature first; on failure, gracefully fall back
        try:
            result = ask_question(
                question=question,
                top_k=top_k,
                role=role,
                character=character,
                intensity=intensity,
                temperature=temperature,
                max_tokens=max_tokens,
                language=language,
                verbose=verbose,
                multi_format=multi_format,
                stem=stem,
                patricize=patricize,
            )
        except TypeError as e:
            # Fallback to legacy pirate switch
            if "unexpected keyword argument" in str(e):
                legacy_pirate = (character or "").strip().lower() == "pirate"
                result = ask_question(
                    question=question,
                    top_k=top_k,
                    role=role,
                    pirate=legacy_pirate,
                    verbose=verbose,
                    multi_format=multi_format,
                    stem=stem,
                )
            else:
                result = f"[Error] {e}"
        except Exception as e:
            result = f"[Error] {e}"

        # Normalize to text for display
        if isinstance(result, dict):
            response_text = result.get("answer", "No response generated.")
        else:
            response_text = str(result)

        self.chat_display.insert(tk.END, f"A: {response_text}\n\n")
        self.chat_display.config(state='disabled')
        self.chat_display.see(tk.END)

    # ----------------- Misc -----------------
    def dependencies(self):
        key = simpledialog.askstring("Issue Dependencies", "Enter Issue Key (e.g., KSDS-19):")
        if key:
            show_dependencies(key)


if __name__ == '__main__':
    root = tk.Tk()
    app = JiraRAGApp(root)
    root.mainloop()
