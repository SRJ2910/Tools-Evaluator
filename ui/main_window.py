import tkinter as tk

from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

import threading
from queue import Queue, Empty

from evaluator import Evaluator
from model_runner import FunctionGemmaRunner

class MainWindow:

    def __init__(self, root):

        self.root = root

        self.root.title("FunctionGemma Evaluator")
        self.root.geometry("1500x900")

        self.results = []
        self.item_to_result = {}

        self.queue = Queue()

        self.exact_count = 0
        self.tool_count = 0
        self.args_count = 0
        self.processed_count = 0
        self.total_count = 0

        self.create_ui()

        self.root.after(
            100,
            self.process_queue
        )

    ####################################################################
    # UI
    ####################################################################

    def create_ui(self):

        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=10)

        ttk.Label(
            top,
            text="Model Path"
        ).grid(row=0, column=0, sticky="w")

        self.model_var = tk.StringVar()

        ttk.Entry(
            top,
            textvariable=self.model_var,
            width=100
        ).grid(row=0, column=1)

        ttk.Button(
            top,
            text="Browse",
            command=self.select_model
        ).grid(row=0, column=2)

        ttk.Label(
            top,
            text="Dataset"
        ).grid(row=1, column=0, sticky="w")

        self.dataset_var = tk.StringVar()

        ttk.Entry(
            top,
            textvariable=self.dataset_var,
            width=100
        ).grid(row=1, column=1)

        ttk.Button(
            top,
            text="Browse",
            command=self.select_dataset
        ).grid(row=1, column=2)

        ttk.Button(
            top,
            text="Run Evaluation",
            command=self.start_eval
        ).grid(row=2, column=1, pady=10)

        self.progress = ttk.Progressbar(
            self.root,
            mode="determinate"
        )

        self.progress.pack(
            fill="x",
            padx=10,
            pady=5
        )

        self.summary = tk.Text(
            self.root,
            height=7
        )

        self.summary.pack(
            fill="x",
            padx=10
        )

        columns = (
            "row",
            "status",
            "tool_gt",
            "tool_pred",
            "latency"
        )

        self.tree = ttk.Treeview(
            self.root,
            columns=columns,
            show="headings"
        )

        for col in columns:

            self.tree.heading(
                col,
                text=col
            )

            self.tree.column(
                col,
                width=150
            )

        self.tree.tag_configure(
            "pass",
            background="#d8ffd8"
        )

        self.tree.tag_configure(
            "fail",
            background="#ffd8d8"
        )

        self.tree.pack(
            fill="both",
            expand=True,
            padx=10,
            pady=5
        )

        self.tree.bind(
            "<<TreeviewSelect>>",
            self.show_details
        )

        self.details = tk.Text(
            self.root,
            height=18
        )

        self.details.pack(
            fill="x",
            padx=10,
            pady=5
        )

    ####################################################################
    # FILE PICKERS
    ####################################################################

    def select_model(self):

        path = filedialog.askdirectory()

        if path:
            self.model_var.set(path)

    def select_dataset(self):

        path = filedialog.askopenfilename(
            filetypes=[
                ("JSONL Files", "*.jsonl")
            ]
        )

        if path:
            self.dataset_var.set(path)

    ####################################################################
    # QUEUE
    ####################################################################

    def process_queue(self):

        try:

            while True:

                message = self.queue.get_nowait()

                msg_type = message[0]

                if msg_type == "init_progress":

                    total = message[1]

                    self.progress["maximum"] = total

                elif msg_type == "sample":

                    result = message[1]

                    self.add_result_row(result)

                elif msg_type == "progress":

                    current, total = message[1]

                    self.progress["value"] = current

                elif msg_type == "summary":

                    summary = message[1]

                    self.display_summary(summary)

                elif msg_type == "error":

                    self.progress["value"] = 0

                    messagebox.showerror(
                        "Error",
                        message[1]
                    )

        except Empty:
            pass

        self.root.after(
            100,
            self.process_queue
        )

    ####################################################################
    # START EVAL
    ####################################################################

    def start_eval(self):

        if not self.model_var.get():

            messagebox.showerror(
                "Error",
                "Select Model Folder"
            )

            return

        if not self.dataset_var.get():

            messagebox.showerror(
                "Error",
                "Select Dataset"
            )

            return

        self.results.clear()
        self.item_to_result.clear()

        self.exact_count = 0
        self.tool_count = 0
        self.args_count = 0
        self.processed_count = 0
        self.total_count = 0

        self.progress["value"] = 0

        self.tree.delete(
            *self.tree.get_children()
        )

        self.summary.delete(
            "1.0",
            tk.END
        )

        self.details.delete(
            "1.0",
            tk.END
        )

        threading.Thread(
            target=self.run_eval_worker,
            daemon=True
        ).start()

    ####################################################################
    # WORKER THREAD
    ####################################################################

    def run_eval_worker(self):

        try:

            model = FunctionGemmaRunner(
                self.model_var.get()
            )

            evaluator = Evaluator(model)

            dataset = evaluator.load_dataset(
                self.dataset_var.get()
            )

            total = len(dataset)

            self.queue.put(
                (
                    "init_progress",
                    total
                )
            )

            exact = 0
            tool_acc = 0
            args_acc = 0

            for idx, sample in enumerate(dataset):

                result = evaluator.evaluate_sample(
                    sample,
                    idx + 1
                )

                self.queue.put(
                    (
                        "sample",
                        result
                    )
                )

                self.queue.put(
                    (
                        "progress",
                        (
                            idx + 1,
                            total
                        )
                    )
                )

                if result["exact_match"]:
                    exact += 1

                if result["tool_match"]:
                    tool_acc += 1

                if result["args_match"]:
                    args_acc += 1

            summary = {
                "total": total,
                "exact": exact,
                "tool": tool_acc,
                "args": args_acc
            }

            self.queue.put(
                (
                    "summary",
                    summary
                )
            )

        except Exception as e:

            import traceback

            traceback.print_exc()

            self.queue.put(
                (
                    "error",
                    str(e)
                )
            )

    ####################################################################
    # LIVE ROW INSERTION
    ####################################################################

    def add_result_row(self, result):

        self.results.append(result)

        self.processed_count += 1

        if result["exact_match"]:
            self.exact_count += 1

        if result["tool_match"]:
            self.tool_count += 1

        if result["args_match"]:
            self.args_count += 1

        tag = (
            "pass"
            if result["exact_match"]
            else "fail"
        )

        item = self.tree.insert(
            "",
            "end",
            values=(
                result["row"],
                "PASS"
                if result["exact_match"]
                else "FAIL",
                result["tool_gt"],
                result["tool_pred"],
                f"{result['latency']:.1f}"
            ),
            tags=(tag,)
        )

        self.item_to_result[item] = result

        exact_acc = (
            self.exact_count /
            self.processed_count
        ) * 100

        tool_acc = (
            self.tool_count /
            self.processed_count
        ) * 100

        args_acc = (
            self.args_count /
            self.processed_count
        ) * 100

        self.summary.delete(
            "1.0",
            tk.END
        )

        self.summary.insert(
            tk.END,
f"""Processed: {self.processed_count}/{self.progress['maximum']}
*************************
Exact Match Accuracy: {exact_acc:.2f} %
Tool Accuracy: {tool_acc:.2f} %
Argument Accuracy: {args_acc:.2f} %
"""
        )


    ####################################################################
    # FINAL SUMMARY
    ####################################################################

    def display_summary(self, summary):

        total = summary["total"]

        messagebox.showinfo(
            "Evaluation Complete",
f"""Total Samples: {total}
Exact Match: {summary['exact']/total*100:.2f} %
Tool Accuracy: {summary['tool']/total*100:.2f} %
Argument Accuracy: {summary['args']/total*100:.2f} %
"""
    )


    ####################################################################
    # DETAILS PANEL
    ####################################################################

    def show_details(self, event):

        selected = self.tree.selection()

        if not selected:
            return

        item = selected[0]

        result = self.item_to_result[item]

        self.details.delete(
            "1.0",
            tk.END
        )

        self.details.insert(
            tk.END,
f"""Row: {result['row']}
*************************
Ground Truth Tool: {result['tool_gt']}
Predicted Tool: {result['tool_pred']}
Ground Truth Args: {result['args_gt']}
Predicted Args: {result['args_pred']}
Tool Match: {result['tool_match']}
Args Match: {result['args_match']}
Exact Match: {result['exact_match']}
Latency: {result['latency']:.2f} ms
Prompt Tokens: {result['prompt_tokens']}
Generated Tokens: {result['generated_tokens']}
Raw Output: {result['raw_output']}
"""
    )
