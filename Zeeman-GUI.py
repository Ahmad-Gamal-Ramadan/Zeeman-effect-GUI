import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


MU_B_EV_T = 5.7883818060e-5  # Bohr magneton in eV/T


@dataclass
class ZeemanConfig:
    mode: str = "Hyperfine"
    L: float = 1.0
    S: float = 0.5
    I: float = 0.5
    A_eV: float = 1e-6
    B_min: float = 0.0
    B_max: float = 2.0
    n_points: int = 300
    title: str = "Zeeman Effect"


class ZeemanGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Zeeman Effect Simulator")
        self.root.geometry("1320x860")
        self.root.minsize(1120, 720)

        self.figure = plt.Figure(figsize=(8.8, 6.3), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = None
        self.toolbar = None

        self.B_values = None
        self.energy_levels = []

        self.mode_var = tk.StringVar(value="Hyperfine")
        self.title_var = tk.StringVar(value="Zeeman Effect")
        self.L_var = tk.DoubleVar(value=1.0)
        self.S_var = tk.DoubleVar(value=0.5)
        self.I_var = tk.DoubleVar(value=0.5)
        self.A_var = tk.DoubleVar(value=1e-6)
        self.Bmin_var = tk.DoubleVar(value=0.0)
        self.Bmax_var = tk.DoubleVar(value=2.0)
        self.npoints_var = tk.IntVar(value=300)
        self.show_labels_var = tk.BooleanVar(value=False)
        self.energy_unit_var = tk.StringVar(value="eV")
        self.linewidth_var = tk.DoubleVar(value=2.0)

        self._build_ui()
        self._draw_placeholder()
        self._toggle_hyperfine_controls()

    @staticmethod
    def _is_half_integer(x: float, tol: float = 1e-9) -> bool:
        return abs(2 * x - round(2 * x)) < tol and x >= 0

    @staticmethod
    def _unique_sorted(values):
        unique_vals = []
        for value in values:
            if not any(abs(value - existing) < 1e-9 for existing in unique_vals):
                unique_vals.append(value)
        return sorted(unique_vals)

    @staticmethod
    def _lande_g(L: float, S: float, J: float) -> float:
        if J == 0:
            return 0.0
        return 1.0 + (J * (J + 1) + S * (S + 1) - L * (L + 1)) / (2 * J * (J + 1))

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        controls = ttk.LabelFrame(main, text="Simulation Controls", padding=12)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 10))

        plot_frame = ttk.Frame(main)
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        row = 0
        ttk.Label(controls, text="Mode").grid(row=row, column=0, sticky="w")
        mode_combo = ttk.Combobox(
            controls,
            textvariable=self.mode_var,
            values=["Normal", "Anomalous", "Hyperfine"],
            state="readonly",
            width=18,
        )
        mode_combo.grid(row=row, column=1, sticky="ew", pady=4)
        mode_combo.bind("<<ComboboxSelected>>", lambda e: self._toggle_hyperfine_controls())

        row += 1
        ttk.Label(controls, text="Plot title").grid(row=row, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.title_var, width=22).grid(row=row, column=1, sticky="ew", pady=4)

        row += 1
        ttk.Label(controls, text="L").grid(row=row, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.L_var, width=14).grid(row=row, column=1, sticky="ew", pady=4)

        row += 1
        ttk.Label(controls, text="S").grid(row=row, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.S_var, width=14).grid(row=row, column=1, sticky="ew", pady=4)

        row += 1
        self.i_label = ttk.Label(controls, text="I")
        self.i_label.grid(row=row, column=0, sticky="w")
        self.i_entry = ttk.Entry(controls, textvariable=self.I_var, width=14)
        self.i_entry.grid(row=row, column=1, sticky="ew", pady=4)

        row += 1
        self.a_label = ttk.Label(controls, text="A (eV)")
        self.a_label.grid(row=row, column=0, sticky="w")
        self.a_entry = ttk.Entry(controls, textvariable=self.A_var, width=14)
        self.a_entry.grid(row=row, column=1, sticky="ew", pady=4)

        row += 1
        ttk.Separator(controls).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)

        row += 1
        ttk.Label(controls, text="B min (T)").grid(row=row, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.Bmin_var, width=14).grid(row=row, column=1, sticky="ew", pady=4)

        row += 1
        ttk.Label(controls, text="B max (T)").grid(row=row, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.Bmax_var, width=14).grid(row=row, column=1, sticky="ew", pady=4)

        row += 1
        ttk.Label(controls, text="Points").grid(row=row, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.npoints_var, width=14).grid(row=row, column=1, sticky="ew", pady=4)

        row += 1
        ttk.Label(controls, text="Energy unit").grid(row=row, column=0, sticky="w")
        ttk.Combobox(
            controls,
            textvariable=self.energy_unit_var,
            values=["eV", "meV", "μeV"],
            state="readonly",
            width=18,
        ).grid(row=row, column=1, sticky="ew", pady=4)

        row += 1
        ttk.Label(controls, text="Line width").grid(row=row, column=0, sticky="w")
        ttk.Scale(controls, from_=1.0, to=4.0, variable=self.linewidth_var, orient="horizontal").grid(
            row=row, column=1, sticky="ew", pady=4
        )

        row += 1
        ttk.Checkbutton(controls, text="Show end labels", variable=self.show_labels_var).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=4
        )

        row += 1
        ttk.Separator(controls).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)

        row += 1
        ttk.Button(controls, text="Simulate", command=self.simulate).grid(row=row, column=0, columnspan=2, sticky="ew", pady=3)

        row += 1
        ttk.Button(controls, text="Reset", command=self.reset_defaults).grid(row=row, column=0, columnspan=2, sticky="ew", pady=3)

        row += 1
        ttk.Button(controls, text="Save Plot", command=self.save_plot).grid(row=row, column=0, columnspan=2, sticky="ew", pady=3)

        row += 1
        ttk.Button(controls, text="Export Data", command=self.export_data).grid(row=row, column=0, columnspan=2, sticky="ew", pady=3)

        row += 1
        ttk.Label(controls, text="Notes", font=("Segoe UI", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(10, 4))

        row += 1
        notes = (
            "• Normal: S = 0, simplified Zeeman splitting\n"
            "• Anomalous: uses Landé g-factor\n"
            "• Hyperfine: includes F, mF, and A"
        )
        ttk.Label(controls, text=notes, justify="left", wraplength=240).grid(row=row, column=0, columnspan=2, sticky="w")

        controls.columnconfigure(1, weight=1)

        canvas_host = ttk.Frame(plot_frame)
        canvas_host.grid(row=0, column=0, sticky="nsew")
        canvas_host.rowconfigure(0, weight=1)
        canvas_host.columnconfigure(0, weight=1)

        self.canvas = FigureCanvasTkAgg(self.figure, master=canvas_host)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, canvas_host, pack_toolbar=False)
        self.toolbar.grid(row=1, column=0, sticky="ew")

    def _toggle_hyperfine_controls(self):
        is_hyperfine = self.mode_var.get() == "Hyperfine"
        state = "normal" if is_hyperfine else "disabled"
        for widget in (self.i_entry, self.a_entry):
            widget.configure(state=state)
        for label in (self.i_label, self.a_label):
            label.configure(foreground="black" if is_hyperfine else "gray")

    def _draw_placeholder(self):
        self.ax.clear()
        self.ax.set_title("Zeeman Effect Simulator")
        self.ax.set_xlabel("Magnetic Field B (T)")
        self.ax.set_ylabel("Energy")
        self.ax.grid(True, linestyle="--", alpha=0.5)
        self.ax.text(
            0.5,
            0.5,
            "Set parameters and click Simulate",
            transform=self.ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="gray",
        )
        self.canvas.draw()

    def _validate_inputs(self):
        try:
            L = float(self.L_var.get())
            S = float(self.S_var.get())
            I = float(self.I_var.get())
            A_eV = float(self.A_var.get())
            B_min = float(self.Bmin_var.get())
            B_max = float(self.Bmax_var.get())
            n_points = int(self.npoints_var.get())
        except Exception:
            raise ValueError("Please enter valid numeric values.")

        if not self._is_half_integer(L):
            raise ValueError("L must be a non-negative integer or half-integer.")
        if not self._is_half_integer(S):
            raise ValueError("S must be a non-negative integer or half-integer.")
        if self.mode_var.get() == "Hyperfine" and not self._is_half_integer(I):
            raise ValueError("I must be a non-negative integer or half-integer.")
        if B_max <= B_min:
            raise ValueError("B max must be greater than B min.")
        if n_points < 10:
            raise ValueError("Points must be at least 10.")
        if self.mode_var.get() != "Hyperfine":
            I = 0.0
            A_eV = 0.0

        return ZeemanConfig(
            mode=self.mode_var.get(),
            L=L,
            S=S,
            I=I,
            A_eV=A_eV,
            B_min=B_min,
            B_max=B_max,
            n_points=n_points,
            title=self.title_var.get().strip() or "Zeeman Effect",
        )

    def _unit_scale_and_label(self):
        unit = self.energy_unit_var.get()
        if unit == "meV":
            return 1e3, "Energy (meV)"
        if unit == "μeV":
            return 1e6, "Energy (μeV)"
        return 1.0, "Energy (eV)"

    def simulate(self):
        try:
            config = self._validate_inputs()
            self.B_values, self.energy_levels = self._compute_energy_levels(config)
            self._plot_energy_levels(config)
        except Exception as exc:
            messagebox.showerror("Simulation Error", str(exc))

    def _compute_energy_levels(self, config: ZeemanConfig):
        B_values = np.linspace(config.B_min, config.B_max, config.n_points)
        energy_levels = []

        if config.mode == "Normal":
            mL_values = np.arange(-config.L, config.L + 1, 1)
            for mL in mL_values:
                energy = MU_B_EV_T * mL * B_values
                energy_levels.append(
                    {
                        "label": f"mL={mL:.1f}",
                        "energy_eV": energy,
                    }
                )
            return B_values, energy_levels

        J_values = self._unique_sorted([config.L + config.S, abs(config.L - config.S)])
        if not J_values:
            raise ValueError("No valid J values were generated.")

        for J in J_values:
            g_J = self._lande_g(config.L, config.S, J)

            if config.mode == "Anomalous":
                mJ_values = np.arange(-J, J + 1, 1)
                for mJ in mJ_values:
                    energy = g_J * MU_B_EV_T * mJ * B_values
                    energy_levels.append(
                        {
                            "label": f"J={J:.1f}, mJ={mJ:.1f}",
                            "energy_eV": energy,
                        }
                    )
            else:
                F_values = self._unique_sorted([J + config.I, abs(J - config.I)])
                for F in F_values:
                    mF_values = np.arange(-F, F + 1, 1)
                    E_hf = 0.5 * config.A_eV * (F * (F + 1) - config.I * (config.I + 1) - J * (J + 1))
                    for mF in mF_values:
                        energy = E_hf + g_J * MU_B_EV_T * mF * B_values
                        energy_levels.append(
                            {
                                "label": f"J={J:.1f}, F={F:.1f}, mF={mF:.1f}",
                                "energy_eV": energy,
                            }
                        )

        return B_values, energy_levels

    def _plot_energy_levels(self, config: ZeemanConfig):
        if self.B_values is None or not self.energy_levels:
            return

        scale, ylabel = self._unit_scale_and_label()
        self.ax.clear()
        self.ax.grid(True, linestyle="--", alpha=0.5)
        cmap = plt.cm.plasma
        n_levels = len(self.energy_levels)

        for idx, level in enumerate(self.energy_levels):
            color = cmap(idx / max(n_levels - 1, 1))
            y = level["energy_eV"] * scale
            self.ax.plot(self.B_values, y, linewidth=self.linewidth_var.get(), color=color)
            if self.show_labels_var.get():
                self.ax.text(
                    self.B_values[-1],
                    y[-1],
                    "  " + level["label"],
                    fontsize=8,
                    va="center",
                    color=color,
                )

        subtitle = f"Mode: {config.mode} | L={config.L}, S={config.S}"
        if config.mode == "Hyperfine":
            subtitle += f", I={config.I}, A={config.A_eV:.2e} eV"

        self.ax.set_title(config.title + "\n" + subtitle, fontsize=12)
        self.ax.set_xlabel("Magnetic Field B (T)")
        self.ax.set_ylabel(ylabel)
        self.figure.tight_layout()
        self.canvas.draw()

    def save_plot(self):
        if self.B_values is None or not self.energy_levels:
            messagebox.showwarning("No plot", "Please run a simulation first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save plot",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
        )
        if path:
            self.figure.savefig(path, bbox_inches="tight")
            messagebox.showinfo("Saved", "Plot saved successfully.")

    def export_data(self):
        if self.B_values is None or not self.energy_levels:
            messagebox.showwarning("No data", "Please run a simulation first.")
            return

        path = filedialog.asksaveasfilename(
            title="Export data",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return

        scale, _ = self._unit_scale_and_label()
        headers = ["B_T"] + [level["label"] for level in self.energy_levels]
        data_columns = [self.B_values] + [level["energy_eV"] * scale for level in self.energy_levels]
        data = np.column_stack(data_columns)

        with open(path, "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            np.savetxt(f, data, delimiter=",", fmt="%.8e")

        messagebox.showinfo("Exported", "Simulation data exported successfully.")

    def reset_defaults(self):
        self.mode_var.set("Hyperfine")
        self.title_var.set("Zeeman Effect")
        self.L_var.set(1.0)
        self.S_var.set(0.5)
        self.I_var.set(0.5)
        self.A_var.set(1e-6)
        self.Bmin_var.set(0.0)
        self.Bmax_var.set(2.0)
        self.npoints_var.set(300)
        self.show_labels_var.set(False)
        self.energy_unit_var.set("eV")
        self.linewidth_var.set(2.0)
        self._toggle_hyperfine_controls()
        self.B_values = None
        self.energy_levels = []
        self._draw_placeholder()


if __name__ == "__main__":
    root = tk.Tk()
    app = ZeemanGUI(root)
    root.mainloop()
