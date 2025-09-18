"""
Script to plot feature relevance
"""

import random
import warnings
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

random_seed = 42

# Set seeds
np.random.seed(random_seed)
random.seed(random_seed)

warnings.filterwarnings("ignore")

df_meta = pd.read_csv('./results/meta_dataset.csv')
df = df_meta.drop(columns=['Model', 'Model Type'])

# Ensure only numeric meta-features are used
meta_features = df.drop(columns=["MCC"])
target = df["MCC"]

# Compute correlations
pcc = meta_features.corrwith(target, method="pearson")
srcc = meta_features.corrwith(target, method="spearman")

# Combine into a DataFrame
correlations = pd.DataFrame({
    "PCC": pcc,
    "SRCC": srcc
}).sort_values("PCC", ascending=False)  # sort by PCC or SRCC if preferred

# Sort by PCC
correlations = correlations.sort_values(by="PCC", ascending=True)

# Color palette (shared across both)
colors = sns.color_palette("coolwarm", len(correlations))[::-1]

# Plot
plt.rcParams.update({
    "font.size": 12,            # Base font size
    "axes.titlesize": 14,       # Title of each subplot
    "axes.labelsize": 13,       # X and Y axis labels
    "xtick.labelsize": 11,      # Tick labels
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
    "figure.titlesize": 16      # Main title
})

fig, axes = plt.subplots(ncols=2, figsize=(5.5, 4.4), sharey=True)

# PCC barplot
axes[0].barh(correlations.index, correlations["PCC"], color=colors)
axes[0].set_xlabel("PCC")
axes[0].axvline(0, color='gray', linestyle='--')
axes[0].set_ylabel("Input Features")

# SRCC barplot (right)
axes[1].barh(correlations.index, correlations["SRCC"], color=colors)
axes[1].set_xlabel("SRCC")
axes[1].axvline(0, color='gray', linestyle='--')

# Y-tick labels on both sides
axes[1].set_yticks(np.arange(len(correlations.index)))
axes[1].set_yticklabels(correlations.index)

# Dynamically set x-axis ticks at every 0.2 interval based on actual min/max values
for ax, col in zip(axes, ["PCC", "SRCC"]):
    min_val = correlations[col].min()
    max_val = correlations[col].max()
    ax.set_xticks(np.arange(np.floor(min_val * 5) / 5, np.ceil(max_val * 5) / 5 + 0.2, 0.2))
    if col == "SRCC":
        # Adjust the x-axis limits for SRCC based on correlations
        ax.set_xlim(min_val- 0.025,  max_val + 0.025)


# Final layout
# fig.suptitle("Correlation between Dataset Characteristics and Model Performance", fontsize=14)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.tight_layout()
plt.savefig('plots/pdf/f_relevance.pdf', format='pdf', bbox_inches='tight')
plt.show()

# exporting the plotted data to csv
# correlations.to_csv('plots/feature_relevance.csv', index=True)