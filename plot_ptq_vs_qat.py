# %%
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import numpy as np
import json
import os
from glob import glob

FONT_SIZE = 12
colors = plt.cm.get_cmap("plasma")
colors = colors(np.linspace(0, 1, 6))
# %%
results_dir = "trained_models/hls150_robust_quantmodels"
ptq_slurm_ids = {32: 179826, 16: 179819, 8: 194424}
qat_slurm_ids = {32: 179606, 16: 179601, 8: 179595}
ptq_dirname = (
    "/shared/home/coppi/repositories/tn4hep/TTN4HEP/trained_models/hls150_models_robust"
)

datas = defaultdict(list)
FLS = np.arange(2, 17, 2)
float_perf = {8: 0.65329, 16: 0.72574, 32: 0.76876}
bd = 10

for file in glob(os.path.join(results_dir, "*.json")):
    with open(file, "r") as f:
        data = json.load(f)
        datas["bd"].append(data["hyperparams"]["model"]["bond_dim"])
        datas["nconst"].append(data["hyperparams"]["dataset"]["nconst"])
        datas["test_accs"].append(data["results"]["test_accs"])
        datas["train_accs"].append(data["results"]["train_accs"])
        datas["aucs"].append(data["results"]["aucs"])
        datas["fprs"].append(data["results"]["fprs_at_tpr"])
        if len(data["results"]["train_accs"]) != 8:
            print(f"Warning: {file} has unexpected number of bond dimensions.")

df = pd.DataFrame(datas)
df
# %%
for nconst in np.sort(df["nconst"].unique()):
    subset = df[df["nconst"] == nconst]
    ptq_fpop_means = np.load(
        os.path.join(ptq_dirname, f"ptq_fpop_test_mean_{ptq_slurm_ids[nconst]}.npy")
    )
    ptq_fpop_stds = np.load(
        os.path.join(ptq_dirname, f"ptq_fpop_test_std_{ptq_slurm_ids[nconst]}.npy")
    )
    ptq_qop_means = np.load(
        os.path.join(ptq_dirname, f"ptq_qop_test_mean_{ptq_slurm_ids[nconst]}.npy")
    )
    ptq_qop_stds = np.load(
        os.path.join(ptq_dirname, f"ptq_qop_test_std_{ptq_slurm_ids[nconst]}.npy")
    )
    qat_fpop_accs = np.load(
        os.path.join(results_dir, f"qat_fpop_test_accs_{qat_slurm_ids[nconst]}.npy")
    )
    plt.figure(figsize=(6, 4))

    bd_subset = subset[subset["bd"] == 10]
    plt.plot(
        FLS,
        bd_subset.loc[bd_subset.index[0], "train_accs"],
        label=f"QAT - quantized ops",
        marker="o",
        color="tab:blue",
        linestyle="--",
        zorder=20,
    )
    plt.plot(
        FLS,
        bd_subset.loc[bd_subset.index[0], "test_accs"],
        label=f"QAT - full precision ops",
        marker="o",
        color="tab:blue",
        zorder=25,
    )
    plt.errorbar(
        range(2, 17),
        ptq_fpop_means,
        yerr=ptq_fpop_stds,
        label=f"PTQ - full precision ops",
        marker="s",
        color="tab:orange",
        zorder=15,
    )
    plt.errorbar(
        range(2, 17),
        ptq_qop_means,
        yerr=ptq_qop_stds,
        label=f"PTQ - quantized ops",
        marker="s",
        color="tab:orange",
        linestyle="--",
        zorder=10,
    )
    plt.hlines(
        float_perf[nconst],
        xmin=FLS[0],
        xmax=FLS[-1],
        colors="k",
        linestyles="dashed",
        zorder=10,
    )
    # plt.plot(
    #     FLS, bd_subset.loc[bd_subset.index[0], "train_accs"], label="Train Acc"
    # )
    plt.title(f"Quantized Model Performance (nconst={nconst})", fontsize=FONT_SIZE + 2)
    plt.xlabel("Fractional bits", fontsize=FONT_SIZE)
    plt.ylabel("Test Accuracy", fontsize=FONT_SIZE)
    plt.tick_params(axis="both", which="major", labelsize=FONT_SIZE - 2)
    plt.legend(fontsize=FONT_SIZE - 2).set_zorder(40)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(f"images/qatvsptq_testacc_nconst{nconst}.pdf")
    plt.close()


# %%
results_dir = "trained_models/hls150_models_robust"

datas = defaultdict(list)
colors = ["tab:blue", "tab:orange", "tab:green"]

for file in glob(os.path.join(results_dir, "results_179*.json")):
    with open(file, "r") as f:
        data = json.load(f)
        datas["bd"].append(data["hyperparams"]["model"]["bond_dim"])
        datas["nconst"].append(data["hyperparams"]["dataset"]["nconst"])
        datas["test_acc"].append(data["results"]["avg_test_acc"])
        datas["train_acc"].append(data["results"]["avg_train_acc"])
        datas["std_test_acc"].append(data["results"]["std_test_acc"])
        datas["std_train_acc"].append(data["results"]["std_train_acc"])

df = pd.DataFrame(datas)
df
# %%
plt.figure(figsize=(6, 4))
handles = [plt.Rectangle((0, 0), 0, 0, color="w")]
labels = ["#Const"]
for i, nconst in enumerate(np.sort(df["nconst"].unique())):
    subset = df[df["nconst"] == nconst].sort_values(by="bd")

    (handle,) = plt.plot(
        subset["bd"],
        subset["test_acc"],
        marker="o",
        label=f"{nconst}",
        color=colors[i],
    )
    handles.append(handle)
    labels.append(f"{nconst}")
    plt.plot(
        subset["bd"],
        subset["train_acc"],
        marker="s",
        markerfacecolor="none",
        label=f"{nconst}",
        color=colors[i],
    )
    plt.fill_between(
        subset["bd"],
        subset["test_acc"] - subset["std_test_acc"],
        subset["test_acc"] + subset["std_test_acc"],
        alpha=0.2,
        color=colors[i],
    )
    plt.fill_between(
        subset["bd"],
        subset["train_acc"] - subset["std_train_acc"],
        subset["train_acc"] + subset["std_train_acc"],
        alpha=0.2,
        color=colors[i],
    )

# empty plot to create handle of round
handles.append(plt.Rectangle((0, 0), 0, 0, color="w"))
(round_handle,) = plt.plot([], [], marker="o", label="Test", color="black")
(square_handle,) = plt.plot(
    [], [], marker="s", markerfacecolor="none", label="Train", color="black"
)
handles.append(round_handle)
handles.append(square_handle)
labels.append("Set")
labels.append("Test")
labels.append("Train")
plt.title(f"Model Performance", fontsize=FONT_SIZE + 2)
plt.xlabel("Bond dim", fontsize=FONT_SIZE)
plt.ylabel("Accuracy", fontsize=FONT_SIZE)
plt.xticks(df["bd"].unique())
plt.tick_params(axis="both", which="major", labelsize=FONT_SIZE - 2)

legend = plt.legend(
    handles=handles,
    labels=labels,
    fontsize=FONT_SIZE - 2,
    loc="center right",
)
legend_texts = legend.get_texts()
legend_texts[0].set_fontsize(FONT_SIZE - 1)
legend_texts[0].set_position(np.array((-28, 0)) + legend_texts[0].get_position())
legend_texts[-3].set_fontsize(FONT_SIZE - 1)
legend_texts[-3].set_position(np.array((-28, 0)) + legend_texts[-3].get_position())
plt.savefig("images/testacc_byBD.pdf")
# %%
