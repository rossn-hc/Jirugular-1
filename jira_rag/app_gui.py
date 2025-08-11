#!/usr/bin/env python
"""
app_gui.py – Enhanced UI Dialog Box with Chat Window for Jira-RAG
"""

import tkinter as tk
from tkinter import simpledialog, messagebox, scrolledtext
from typing import Optional
from jira_rag.interface import crawl_and_build, show_dependencies, ask_question  # fallback to interface if chat_wrapper isn't in module path


PERSONA_OPTIONS = [
    "",                 # none
    "pirate",
    "yoda",
    "shakespeare",
    "executive-snark",
]

ROLE_OPTIONS = ["", "developer", "manager", "executive"]

INTENSITY_OPTIONS = ["light", "medium", "heavy"]

LANGUAGE_OPTIONS = [
    "en",      # English
    "fr-CA",   # Français (Québec)
    "fr",      # Français
    "es",      # Español
    "de",      # Deutsch
    "it",      # Italiano
    "pt-BR",   # Português (Brasil)
    "ja",      # 日本語
    "ko",      # 한국어
    "zh-CN",   # 简体中文
]

MAX_TOKEN_OPTIONS = [512, 1024, 2048, 4096]


class JiraRAGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Jira-RAG Interactive")
        self.create_main_widgets()

    def create_main_widgets(self):
        tk.Label(self.root, text="Jira-RAG Interactive", font=('Helvetica', 16, 'bold')).pack(pady=10)

        tk.Button(self.root, text="Crawl / rebuild index", command=self.crawl_index).pack(fill='x', padx=20, pady=5)
        tk.Button(self.root, text="Open Chat", command=self.open_chat_window).pack(fill='x', padx=20, pady=5)
        tk.Button(self.root, text="Show dependencies for an issue", command=self.dependencies).pack(fill='x', padx=20, pady=5)
        tk.Button(self.root, text="Quit", command=self.root.quit).pack(fill='x', padx=20, pady=5)

    def crawl_index(self):
        jql = self.show_jql_builder()
        if jql is not None:
            try:
                crawl_and_build(jql)
                messagebox.showinfo("Success", f"Crawling completed.\nJQL: {jql}")
            except Exception as e:
                messagebox.showerror("Error", f"Crawl failed:\n{e}")

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

    def open_chat_window(self):
        chat_win = tk.Toplevel(self.root)
        chat_win.title("Chat Window")

        options_frame = tk.Frame(chat_win)
        options_frame.pack(side=tk.TOP, fill='x', padx=5, pady=5)

        # Top-K
        tk.Label(options_frame, text="Number of issues:").grid(row=0, column=0, padx=5, sticky="w")
        self.top_k_var = tk.IntVar(value=5)
        tk.Spinbox(options_frame, from_=1, to=100, textvariable=self.top_k_var, width=6).grid(row=0, column=1, sticky="w")

        # Role dropdown
        tk.Label(options_frame, text="Role:").grid(row=0, column=2, padx=(12, 5), sticky="e")
        self.role_var = tk.StringVar(value="")
        tk.OptionMenu(options_frame, self.role_var, *ROLE_OPTIONS).grid(row=0, column=3, sticky="w")

        # Persona dropdown
        tk.Label(options_frame, text="Persona:").grid(row=0, column=4, padx=(12, 5), sticky="e")
        self.persona_var = tk.StringVar(value="")
        tk.OptionMenu(options_frame, self.persona_var, *PERSONA_OPTIONS).grid(row=0, column=5, sticky="w")

        # Verbose + Multi-format
        self.verbose_var = tk.BooleanVar(value=False)
        self.multiformat_var = tk.BooleanVar(value=False)
        tk.Checkbutton(options_frame, text="Verbose", variable=self.verbose_var).grid(row=1, column=0, padx=5, pady=(6,0), sticky="w")
        tk.Checkbutton(options_frame, text="Multi-format", variable=self.multiformat_var).grid(row=1, column=1, padx=5, pady=(6,0), sticky="w")

        # Persona Intensity
        tk.Label(options_frame, text="Intensity:").grid(row=1, column=2, padx=(12,5), sticky="e")
        self.intensity_var = tk.StringVar(value="medium")
        tk.OptionMenu(options_frame, self.intensity_var, *INTENSITY_OPTIONS).grid(row=1, column=3, sticky="w")

        # Temperature
        tk.Label(options_frame, text="Temperature:").grid(row=1, column=4, padx=(12,5), sticky="e")
        self.temperature_var = tk.DoubleVar(value=0.5)
        tk.Spinbox(options_frame, from_=0.0, to=1.5, increment=0.1, textvariable=self.temperature_var, width=6).grid(row=1, column=5, sticky="w")

        # Max tokens
        tk.Label(options_frame, text="Max tokens:").grid(row=2, column=0, padx=(5,5), sticky="w")
        self.max_tokens_var = tk.IntVar(value=2048)
        tk.OptionMenu(options_frame, self.max_tokens_var, *MAX_TOKEN_OPTIONS).grid(row=2, column=1, sticky="w")

        # Language
        tk.Label(options_frame, text="Language:").grid(row=2, column=2, padx=(12,5), sticky="e")
        self.language_var = tk.StringVar(value="en")
        tk.OptionMenu(options_frame, self.language_var, *LANGUAGE_OPTIONS).grid(row=2, column=3, sticky="w")

        # Chat display
        self.chat_display = scrolledtext.ScrolledText(chat_win, state='disabled', height=15)
        self.chat_display.pack(padx=10, pady=5, fill='both', expand=True)

        # Entry + Ask button
        entry_frame = tk.Frame(chat_win)
        entry_frame.pack(side=tk.BOTTOM, fill='x', padx=5, pady=5)

        self.question_entry = tk.Entry(entry_frame)
        self.question_entry.pack(side=tk.LEFT, fill='x', expand=True, padx=(0, 5))
        tk.Button(entry_frame, text="Ask", command=self.ask_question).pack(side=tk.RIGHT)

    def ask_question(self):
        question = self.question_entry.get().strip()
        if not question:
            return

        # Clear entry and show question
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
            )
        except TypeError as e:
            # If backend doesn't yet accept 'character' (or new knobs), retry with pirate flag iff persona == 'pirate'
            if "unexpected keyword argument" in str(e):
                legacy_pirate = (character or "").strip().lower() == "pirate"
                result = ask_question(
                    question=question,
                    top_k=top_k,
                    role=role,
                    pirate=legacy_pirate,
                    verbose=verbose,
                    multi_format=multi_format,
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


    def dependencies(self):
        key = simpledialog.askstring("Issue Dependencies", "Enter Issue Key (e.g., KSDS-19):")
        if key:
            show_dependencies(key)


if __name__ == '__main__':
    root = tk.Tk()
    app = JiraRAGApp(root)
    root.mainloop()
#!/usr/bin/env python
"""
app_gui.py – Enhanced UI Dialog Box with Chat Window for Jira-RAG
"""

import tkinter as tk
from tkinter import simpledialog, messagebox, scrolledtext
from typing import Optional
from jira_rag.interface import crawl_and_build, show_dependencies, ask_question  # fallback to interface if chat_wrapper isn't in module path


PERSONA_OPTIONS = [
    "",                 # none
    "pirate",
    "yoda",
    "shakespeare",
    "executive-snark",
]

ROLE_OPTIONS = ["", "developer", "manager", "executive"]

INTENSITY_OPTIONS = ["light", "medium", "heavy"]

LANGUAGE_OPTIONS = [
    "en",      # English
    "fr-CA",   # Français (Québec)
    "fr",      # Français
    "es",      # Español
    "de",      # Deutsch
    "it",      # Italiano
    "pt-BR",   # Português (Brasil)
    "ja",      # 日本語
    "ko",      # 한국어
    "zh-CN",   # 简体中文
]

MAX_TOKEN_OPTIONS = [512, 1024, 2048, 4096]


class JiraRAGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Jira-RAG Interactive")
        self.create_main_widgets()

    def create_main_widgets(self):
        tk.Label(self.root, text="Jira-RAG Interactive", font=('Helvetica', 16, 'bold')).pack(pady=10)

        tk.Button(self.root, text="Crawl / rebuild index", command=self.crawl_index).pack(fill='x', padx=20, pady=5)
        tk.Button(self.root, text="Open Chat", command=self.open_chat_window).pack(fill='x', padx=20, pady=5)
        tk.Button(self.root, text="Show dependencies for an issue", command=self.dependencies).pack(fill='x', padx=20, pady=5)
        tk.Button(self.root, text="Quit", command=self.root.quit).pack(fill='x', padx=20, pady=5)

    def crawl_index(self):
        jql = self.show_jql_builder()
        if jql is not None:
            try:
                crawl_and_build(jql)
                messagebox.showinfo("Success", f"Crawling completed.\nJQL: {jql}")
            except Exception as e:
                messagebox.showerror("Error", f"Crawl failed:\n{e}")

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

    def open_chat_window(self):
        chat_win = tk.Toplevel(self.root)
        chat_win.title("Chat Window")

        options_frame = tk.Frame(chat_win)
        options_frame.pack(side=tk.TOP, fill='x', padx=5, pady=5)

        # Top-K
        tk.Label(options_frame, text="Number of issues:").grid(row=0, column=0, padx=5, sticky="w")
        self.top_k_var = tk.IntVar(value=5)
        tk.Spinbox(options_frame, from_=1, to=100, textvariable=self.top_k_var, width=6).grid(row=0, column=1, sticky="w")

        # Role dropdown
        tk.Label(options_frame, text="Role:").grid(row=0, column=2, padx=(12, 5), sticky="e")
        self.role_var = tk.StringVar(value="")
        tk.OptionMenu(options_frame, self.role_var, *ROLE_OPTIONS).grid(row=0, column=3, sticky="w")

        # Persona dropdown
        tk.Label(options_frame, text="Persona:").grid(row=0, column=4, padx=(12, 5), sticky="e")
        self.persona_var = tk.StringVar(value="")
        tk.OptionMenu(options_frame, self.persona_var, *PERSONA_OPTIONS).grid(row=0, column=5, sticky="w")

        # Verbose + Multi-format
        self.verbose_var = tk.BooleanVar(value=False)
        self.multiformat_var = tk.BooleanVar(value=False)
        tk.Checkbutton(options_frame, text="Verbose", variable=self.verbose_var).grid(row=1, column=0, padx=5, pady=(6,0), sticky="w")
        tk.Checkbutton(options_frame, text="Multi-format", variable=self.multiformat_var).grid(row=1, column=1, padx=5, pady=(6,0), sticky="w")

        # Persona Intensity
        tk.Label(options_frame, text="Intensity:").grid(row=1, column=2, padx=(12,5), sticky="e")
        self.intensity_var = tk.StringVar(value="medium")
        tk.OptionMenu(options_frame, self.intensity_var, *INTENSITY_OPTIONS).grid(row=1, column=3, sticky="w")

        # Temperature
        tk.Label(options_frame, text="Temperature:").grid(row=1, column=4, padx=(12,5), sticky="e")
        self.temperature_var = tk.DoubleVar(value=0.5)
        tk.Spinbox(options_frame, from_=0.0, to=1.5, increment=0.1, textvariable=self.temperature_var, width=6).grid(row=1, column=5, sticky="w")

        # Max tokens
        tk.Label(options_frame, text="Max tokens:").grid(row=2, column=0, padx=(5,5), sticky="w")
        self.max_tokens_var = tk.IntVar(value=2048)
        tk.OptionMenu(options_frame, self.max_tokens_var, *MAX_TOKEN_OPTIONS).grid(row=2, column=1, sticky="w")

        # Language
        tk.Label(options_frame, text="Language:").grid(row=2, column=2, padx=(12,5), sticky="e")
        self.language_var = tk.StringVar(value="en")
        tk.OptionMenu(options_frame, self.language_var, *LANGUAGE_OPTIONS).grid(row=2, column=3, sticky="w")

        # Chat display
        self.chat_display = scrolledtext.ScrolledText(chat_win, state='disabled', height=15)
        self.chat_display.pack(padx=10, pady=5, fill='both', expand=True)

        # Entry + Ask button
        entry_frame = tk.Frame(chat_win)
        entry_frame.pack(side=tk.BOTTOM, fill='x', padx=5, pady=5)

        self.question_entry = tk.Entry(entry_frame)
        self.question_entry.pack(side=tk.LEFT, fill='x', expand=True, padx=(0, 5))
        tk.Button(entry_frame, text="Ask", command=self.ask_question).pack(side=tk.RIGHT)

    def ask_question(self):
        question = self.question_entry.get().strip()
        if not question:
            return

        # Clear entry and show question
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
            )
        except TypeError as e:
            # If backend doesn't yet accept 'character' (or new knobs), retry with pirate flag iff persona == 'pirate'
            if "unexpected keyword argument" in str(e):
                legacy_pirate = (character or "").strip().lower() == "pirate"
                result = ask_question(
                    question=question,
                    top_k=top_k,
                    role=role,
                    pirate=legacy_pirate,
                    verbose=verbose,
                    multi_format=multi_format,
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


    def dependencies(self):
        key = simpledialog.askstring("Issue Dependencies", "Enter Issue Key (e.g., KSDS-19):")
        if key:
            show_dependencies(key)


if __name__ == '__main__':
    root = tk.Tk()
    app = JiraRAGApp(root)
    root.mainloop()
