#!/usr/bin/env python
"""
app_gui.py – Enhanced UI Dialog Box with Chat Window for Jira-RAG
"""

import tkinter as tk
from tkinter import simpledialog, messagebox, scrolledtext
from typing import Optional
from jira_rag.interface import crawl_and_build, show_dependencies, ask_question  # fallback to interface if chat_wrapper isn't in module path


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
            clause = f"{field_var.get()} {operator_var.get()} \"{value_var.get()}\""
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

        tk.Label(options_frame, text="Number of issues:").grid(row=0, column=0, padx=5)
        self.top_k_var = tk.IntVar(value=5)
        tk.Spinbox(options_frame, from_=1, to=100, textvariable=self.top_k_var, width=5).grid(row=0, column=1)

        self.pirate_var = tk.BooleanVar()
        tk.Checkbutton(options_frame, text="Pirate Mode ☠️", variable=self.pirate_var).grid(row=0, column=2, padx=5)

        self.chat_display = scrolledtext.ScrolledText(chat_win, state='disabled', height=15)
        self.chat_display.pack(padx=10, pady=5, fill='both', expand=True)

        entry_frame = tk.Frame(chat_win)
        entry_frame.pack(side=tk.BOTTOM, fill='x', padx=5, pady=5)

        self.question_entry = tk.Entry(entry_frame)
        self.question_entry.pack(side=tk.LEFT, fill='x', expand=True, padx=(0, 5))
        tk.Button(entry_frame, text="Ask", command=self.ask_question).pack(side=tk.RIGHT)

    def ask_question(self):
        question = self.question_entry.get().strip()
        if question:
            self.question_entry.delete(0, tk.END)
            self.chat_display.config(state='normal')
            self.chat_display.insert(tk.END, f"Q: {question}\n")
            try:
                response = ask_question(
                    question=question,
                    top_k=self.top_k_var.get(),
                    pirate=self.pirate_var.get(),
                    verbose=False,
                    multi_format=False
                )
            except TypeError as e:
                if "unexpected keyword" in str(e):
                    response = ask_question(
                        question=question,
                        top_k=self.top_k_var.get(),
                        pirate=self.pirate_var.get()
                    )
                else:
                    response = f"[Error] {e}"
            except Exception as e:
                response = f"[Error] {e}"
            self.chat_display.insert(tk.END, f"A: {response}\n\n")
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
