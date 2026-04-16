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

datas = defaultdict(list)
FLS = np.arange(2, 17, 2)
float_perf = {8: 0.65329, 16: 0.72574, 32: 0.76876}

patterns_per_type = {
    "qgrad_wrong_nolog": ["195543", "1956*", "1957*", "1959*", "196*"],
    "qgrad_new_log": ["19763*"],
    "fpgrad_new_log": ["19764*", "197650"],
    "new_qgrad": ["23191*", "23192*"],
    "new_fpgrad": ["23193*", "23194*"],
    "new_fpgrad_samelr": ["23215*", "232160", "232161", "232162"],
    "new_fpgrad_0.12lr": [
        "232163",
        "232164",
        "232165",
        "232166",
        "232167",
        "232168",
        "232169",
        "232170",
        "232171",
    ],
    "new_fpgrad_8lr": [
        "232172",
        "232173",
        "232174",
        "232175",
        "232176",
        "232177",
        "232178",
        "232179",
        "232180",
    ],
    "new_qgrad_8lr": [
        "232182",
        "232183",
        "232184",
        "232185",
        "232186",
        "232187",
        "232188",
        "232189",
        "232190",
    ],
    "new_qgrad_0.12lr": [
        "232191",
        "232192",
        "232193",
        "232194",
        "232195",
        "232196",
        "232197",
        "232198",
        "232199",
    ],
    "new_qgrad_samelr": ["23220*"],
}

# %%
pattern_type = "new_qgrad_samelr"
files = []
for pattern in patterns_per_type[pattern_type]:
    files += glob(os.path.join(results_dir, f"results_{pattern}.json"))

for file in files:
    with open(file, "r") as f:
        data = json.load(f)
        datas["bd"].append(data["hyperparams"]["model"]["bond_dim"])
        datas["nconst"].append(data["hyperparams"]["dataset"]["nconst"])
        datas["qgrad"].append(data["hyperparams"]["training"]["gradient_quantization"])
        datas["test_accs"].append(data["results"]["test_accs"])
        datas["train_accs"].append(data["results"]["train_accs"])
        datas["qemu_accs"].append(data["results"]["qemu_accs"])
        datas["qemu_norm_accs"].append(data["results"]["qemu_norm_accs"])
        datas["aucs"].append(data["results"]["aucs"])
        datas["fprs"].append(data["results"]["fprs_at_tpr"])
        if len(data["results"]["train_accs"]) != 8:
            print(f"Warning: {file} has unexpected number of bond dimensions.")

df = pd.DataFrame(datas)
df
# %%
for nconst in np.sort(df["nconst"].unique()):
    subset = df[df["nconst"] == nconst]
    plt.figure(figsize=(6, 4))

    for i, bd in enumerate(np.sort(subset["bd"].unique())):
        bd_subset = subset[subset["bd"] == bd]
        plt.plot(
            FLS,
            bd_subset.loc[bd_subset.index[0], "train_accs"],
            label=f"{bd}",
            marker="o",
            zorder=20 + i,
            color=colors[i],
        )
        plt.plot(
            FLS,
            bd_subset.loc[bd_subset.index[0], "qemu_accs"],
            label=f"EMU",
            marker="x",
            zorder=20 + i,
            color=colors[i],
        )
        plt.plot(
            FLS,
            bd_subset.loc[bd_subset.index[0], "qemu_norm_accs"],
            label=f"EMU Norm",
            marker="^",
            zorder=20 + i,
            color=colors[i],
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
    plt.title(
        f"Quantized Model Performance (nconst={nconst})"
        + ("\nqgrad" if all(subset["qgrad"]) else ""),
        fontsize=FONT_SIZE + 2,
    )
    plt.xlabel("Fractional bits", fontsize=FONT_SIZE)
    plt.ylabel("Test Accuracy", fontsize=FONT_SIZE)
    plt.tick_params(axis="both", which="major", labelsize=FONT_SIZE - 2)
    plt.legend(
        title=r"$\chi$", title_fontsize=FONT_SIZE, fontsize=FONT_SIZE - 2
    ).set_zorder(40)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(f"images/quant_{pattern_type}_testacc_nconst{nconst}_new.pdf")
    plt.close()


# %%
DATASET = "bbdata"
results_dir = f"trained_models/{DATASET}_models"

slurm_ids = [202747, 202746, 203418, 203419, 203747, 203748, 203749, 203750, 204552]

# [202843, 202844, 203757, 203758, 203759, 203760, 203761, 203762, 204548]
# [202747, 202746, 203418, 203419, 203747, 203748, 203749, 203750, 204552]

datas = defaultdict(list)
colors = ["tab:blue", "tab:orange", "tab:green"]
filelist = []
for id in slurm_ids:
    filelist.extend(glob(os.path.join(results_dir, f"results_*{id}*.json")))

for file in filelist:
    with open(file, "r") as f:
        data = json.load(f)
        datas["bd"].append(data["hyperparams"]["model"]["bond_dim"])
        datas["test_acc"].append(data["results"]["avg_test_acc"])
        datas["train_acc"].append(data["results"]["avg_train_acc"])
        datas["std_test_acc"].append(data["results"]["std_test_acc"])
        datas["std_train_acc"].append(data["results"]["std_train_acc"])

df = pd.DataFrame(datas)
df
# %%
plt.figure(figsize=(6, 4))
handles = [plt.Rectangle((0, 0), 0, 0, color="w")]
subset = df.sort_values(by="bd")

i = 0

(handle,) = plt.plot(
    subset["bd"],
    subset["test_acc"],
    marker="o",
    label=f"Test set",
    color=colors[i],
)
handles.append(handle)
# labels.append(f"{nconst}")
plt.plot(
    subset["bd"],
    subset["train_acc"],
    marker="s",
    markerfacecolor="none",
    label=f"Training set",
    color=colors[i + 1],
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
    color=colors[i + 1],
)

# empty plot to create handle of round
# handles.append(plt.Rectangle((0, 0), 0, 0, color="w"))
# (round_handle,) = plt.plot([], [], marker="o", label="Test", color="black")
# (square_handle,) = plt.plot(
#    [], [], marker="s", markerfacecolor="none", label="Train", color="black"
# )
# handles.append(round_handle)
# handles.append(square_handle)
# labels.append("Set")
# labels.append("Test")
# labels.append("Train")
plt.title(f"Model Performance", fontsize=FONT_SIZE + 2)
plt.xlabel("Bond dim", fontsize=FONT_SIZE)
plt.ylabel("Accuracy", fontsize=FONT_SIZE)
plt.xticks(df["bd"].unique())
plt.tick_params(axis="both", which="major", labelsize=FONT_SIZE - 2)

legend = plt.legend(
    # handles=handles,
    # labels=labels,
    fontsize=FONT_SIZE - 2,
    loc="center right",
)
# legend_texts = legend.get_texts()
# legend_texts[0].set_fontsize(FONT_SIZE - 1)
# legend_texts[0].set_position(np.array((-28, 0)) + legend_texts[0].get_position())
# legend_texts[-3].set_fontsize(FONT_SIZE - 1)
# legend_texts[-3].set_position(np.array((-28, 0)) + legend_texts[-3].get_position())
plt.savefig(f"images/{DATASET}_testacc_byBD.pdf")
# %%
