from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from perceptron import Perceptron

import time



class MNISTApp:
    def __init__(self, root):
        self.root = root
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.title("Персептрон: 7 vs не-7 (MNIST) | Лабораторная работа")
        self.root.geometry("1200x850")

        self.nn = None
        self.X_train = self.Y_train = self.X_test = self.Y_test = None
        self.y_train_orig = self.y_test_orig = None
        self.data_loaded = False

        self._build_ui()

    def _build_ui(self):
        frm = ttk.LabelFrame(self.root, text="Параметры")
        frm.pack(fill="x", padx=10, pady=5)

        ttk.Button(frm, text="📥 Загрузить MNIST", command=self.load_mnist).grid(
            row=0, column=0, padx=5, pady=5
        )

        ttk.Label(frm, text="η:").grid(row=0, column=1, padx=5)
        self.ent_lr = ttk.Entry(frm, width=6)
        self.ent_lr.insert(0, "0.1")
        self.ent_lr.grid(row=0, column=2)

        ttk.Label(frm, text="Эпохи:").grid(row=0, column=3, padx=5)
        self.ent_epochs = ttk.Entry(frm, width=6)
        self.ent_epochs.insert(0, "50")
        self.ent_epochs.grid(row=0, column=4)

        self.btn_train = ttk.Button(
            frm, text="🎓 Обучить", command=self.start_training, state="disabled"
        )
        self.btn_train.grid(row=0, column=5, padx=5)
        self.btn_test = ttk.Button(
            frm, text="🔍 Тест", command=self.run_test, state="disabled"
        )
        self.btn_test.grid(row=0, column=6, padx=5)

        self.lbl_status = ttk.Label(frm, text="Ожидание загрузки...", foreground="blue")
        self.lbl_status.grid(row=1, column=0, columnspan=7, sticky="w", padx=5)

        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=10, pady=5)

        self.tab_train = ttk.Frame(nb)
        self.tab_test = ttk.Frame(nb)
        nb.add(self.tab_train, text="📊 Обучение")
        nb.add(self.tab_test, text="🎯 Тестирование")

        ttk.Label(self.tab_train, text="Примеры из выборки:").pack(
            anchor="w", padx=10, pady=5
        )
        self.frm_samples = ttk.Frame(self.tab_train)
        self.frm_samples.pack(fill="x", padx=10, pady=5)
        self.log_txt = tk.Text(self.tab_train, height=12)
        self.log_txt.pack(fill="both", expand=True, padx=10, pady=5)

        # График (создаётся один раз)
        self.fig, self.ax = plt.subplots(figsize=(9, 3.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="x", padx=10, pady=5)

        self.frm_test_imgs = ttk.LabelFrame(
            self.tab_test, text="Тестируемые изображения"
        )
        self.frm_test_imgs.pack(fill="x", padx=10, pady=5)
        self.res_txt = tk.Text(self.tab_test, height=14)
        self.res_txt.pack(fill="both", expand=True, padx=10, pady=5)

    def load_mnist(self):
        try:
            from tensorflow.keras.datasets import mnist as tf_mnist
            """
                x_tr, x_te - (60000, 28, 28)
                y_tr, y_te - (60000,)
            """

            (x_tr, y_tr), (x_te, y_te) = tf_mnist.load_data()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить MNIST:\n{e}")
            return

        self.y_train_orig, self.y_test_orig = y_tr.copy(), y_te.copy()
        self.Y_train = (y_tr == 7).astype(np.float32) # 1.0 where 7 and 0.0 where not 7
        self.Y_test = (y_te == 7).astype(np.float32)
        self.X_train = (x_tr/255 ).astype(np.float32).reshape(-1, 28 * 28)
        self.X_test = (x_te/255).astype(np.float32).reshape(-1, 28 * 28)

        self.data_loaded = True
        self.btn_train.config(state="normal")
        n7 = int(self.Y_train.sum())
        self.lbl_status.config(
            text=f"✓ MNIST загружен | Всего: {len(y_tr)} | Семёрок: {n7} | Не-семёрок: {len(y_tr) - n7}",
            foreground="green",
        )
        self._show_samples(x_tr[:10], y_tr[:10])

    def _show_samples(self, imgs, labs):
        for w in self.frm_samples.winfo_children():
            w.destroy()
        for i in range(10):
            img = Image.fromarray(imgs[i]).resize((40, 40))
            ph = ImageTk.PhotoImage(img)
            lbl = ttk.Label(self.frm_samples, image=ph)
            lbl.image = ph
            lbl.grid(row=0, column=i, padx=2)
            txt = ttk.Label(
                self.frm_samples,
                text=str(labs[i]),
                foreground="green" if labs[i] == 7 else "red",
            )
            txt.grid(row=1, column=i, padx=2)

    def _update_log(self, epoch, err, mis, bias=None):
        self.log_txt.insert(
            "end", f"Эпоха {epoch:2d} | Ошибка: {err:4.0f} | Bias: {bias:.2f}\n"
        )
        self.log_txt.see("end")
        self.root.update_idletasks()

    def start_training(self):
        if not self.data_loaded:
            return
        self.log_txt.delete("1.0", "end")

        lr = float(self.ent_lr.get())
        epochs = int(self.ent_epochs.get())
        self.nn = Perceptron(n_inputs=784, lr=lr)  # Полная перезагрузка сети
        self.btn_train.config(state="disabled")

        # Сбалансированная подвыборка
        idx7 = np.where(self.Y_train == 1)[0]
        idx0 = np.where(self.Y_train == 0)[0]
        np.random.shuffle(idx7)
        np.random.shuffle(idx0)
        idx = np.concatenate([idx7[:3000], idx0[:1000]])
        X_s, Y_s = self.X_train[idx], self.Y_train[idx]

        t0 = time.time()
        log = self.nn.train(X_s, Y_s, max_epochs=epochs, ui_callback=self._update_log)
        self.log_txt.insert(
            "end", f"\n✓ Обучение завершено за {time.time() - t0:.2f} сек\n"
        )

        preds, _ = self.nn.predict_batch(X_s)
        acc = np.mean(preds == Y_s.astype(int))
        self.log_txt.insert("end", f"📊 Точность на обучающей подвыборке: {acc:.2%}\n")

        self.btn_train.config(state="normal")
        self.btn_test.config(state="normal")
        self._plot_curve(log)
        self.root.update_idletasks()  # Принудительное обновление GUI

    def _plot_curve(self, log):
        self.ax.clear()
        if log:
            epochs = [l["epoch"] for l in log]
            errors = [l["error"] for l in log]
            self.ax.plot(
                epochs, errors, "b-o", lw=2, markersize=4, label="Суммарная ошибка"
            )
            self.ax.set_xlabel("Эпоха")
            self.ax.set_ylabel("Ошибка")
            self.ax.set_title(f"Кривая обучения (η={self.ent_lr.get()})")
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def run_test(self):
        if not self.nn:
            return
        self.res_txt.delete("1.0", "end")

        # Очищаем предыдущие тестовые картинки
        for w in self.frm_test_imgs.winfo_children():
            w.destroy()

        # Случайная выборка БЕЗ фиксированного seed
        idx = np.random.choice(len(self.X_test), 10, replace=False)
        correct = 0

        for i, k in enumerate(idx):
            x, y = self.X_test[k], self.Y_test[k]
            net, out = self.nn.forward(x)
            ok = bool(out == y)
            if ok:
                correct += 1

            true_digit = int(self.y_test_orig[k])
            true_label = "7" if y == 1 else f"другая ({true_digit})"

            # Показываем изображение
            img_arr = x.reshape(28, 28) * 255
            img_pil = Image.fromarray(img_arr.astype(np.uint8)).resize((50, 50))
            img_tk = ImageTk.PhotoImage(img_pil)
            lbl = ttk.Label(self.frm_test_imgs, image=img_tk)
            lbl.image = img_tk
            lbl.grid(row=0, column=i, padx=2)

            res_txt = f"{i + 1}. {true_label:8s} → {'7' if out else 'не 7':6s} | Net: {net:5.1f} | {'✅' if ok else '❌'}"
            self.res_txt.insert("end", res_txt + "\n")

        self.res_txt.insert(
            "end", f"\n📊 Итоговая точность: {correct}/10 = {correct / 10:.0%}\n"
        )
        self._show_weights()

    def _show_weights(self):
        w = self.nn.weights.reshape(28, 28)
        w = ((w - w.min()) / (w.max() - w.min()) * 255).astype(np.uint8)
        img = ImageTk.PhotoImage(Image.fromarray(w).resize((140, 140)))
        lbl = ttk.Label(self.frm_test_imgs, image=img)
        lbl.image = img
        lbl.grid(row=1, column=0, columnspan=10, pady=5)
        ttk.Label(self.frm_test_imgs, text="Визуализация весов сети").grid(
            row=2, column=0, columnspan=10
        )

    def on_closing(self):
        plt.close(self.fig)
        self.root.destroy()


