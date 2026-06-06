import tkinter as tk
import traceback

from ui.main_window import MainWindow


def report_callback_exception(exc, val, tb):
    traceback.print_exception(exc, val, tb)


root = tk.Tk()

root.report_callback_exception = report_callback_exception

MainWindow(root)

root.mainloop()