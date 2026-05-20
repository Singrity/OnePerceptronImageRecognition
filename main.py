import os
import tkinter as tk
import ssl
from app import MNISTApp


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
ssl._create_default_https_context = ssl._create_unverified_context


if __name__ == "__main__":
    root = tk.Tk()
    MNISTApp(root)
    root.mainloop()
