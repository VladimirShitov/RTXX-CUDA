import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

results_df = pd.read_csv("benchmark_results.txt", sep="\t")

plt.figure(dpi=300)
sns.lineplot(results_df, x="N", y="RTXX_speedup", hue="depth", palette="tab10")
plt.xlabel("M = N = K ")
plt.ylabel("cuBLAS time comparing to RTXX time")
plt.xlim(0, results_df["N"].max() + 1)
plt.hlines(1, 0, results_df["N"].max() + 1, linestyles="dashed", color="black", label="cuBLAS time")
plt.savefig("images/benchmark_plot.png", bbox_inches="tight")