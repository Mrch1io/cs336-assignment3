import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def fit_power_law(C, y):
    b, log_a = np.polyfit(np.log10(C), np.log10(y), 1)
    return 10**log_a, b


def plot_fit(C, y, a, b, ylabel, path):
    C_fit = np.logspace(np.log10(C.min()), 24, 100)
    plt.figure()
    plt.loglog(C, y, "o", label="Data")
    plt.loglog(C_fit, a * C_fit**b, "-", label=f"Fit: {a:.2e} * C^{b:.3f}")
    plt.xlabel("Compute Budget (FLOPs)")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(path)


def main():
    with open("data/isoflops_curves.json") as f:
        data = json.load(f)

    runs = defaultdict(list)
    for run in data:
        runs[run["compute_budget"]].append(run)

    C = np.array(sorted(runs))
    N = np.array([min(runs[c], key=lambda r: r["final_loss"])["parameters"] for c in C])
    D = C / (6 * N)

    a_n, b_n = fit_power_law(C, N)
    a_d, b_d = fit_power_law(C, D)
    plot_fit(C, N, a_n, b_n, "Optimal Model Size (Parameters)", "model_size_scaling.png")
    plot_fit(C, D, a_d, b_d, "Optimal Dataset Size (Tokens)", "dataset_size_scaling.png")

    for budget in [1e23, 1e24]:
        print(f"C = {budget:.0e}: N_opt = {a_n * budget**b_n:.3e} parameters, D_opt = {a_d * budget**b_d:.3e} tokens")


if __name__ == "__main__":
    main()
