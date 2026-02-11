import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# X-axis: missingness rates
missingness = np.array([10, 20, 30, 40, 50])

# Accuracy values (mean ± std) for each dataset (original decimals)
means = {
    "Buy":       [0.974, 0.977, 0.976, 0.965, 0.951],
    "Restaurant":[0.812, 0.810, 0.792, 0.800, 0.795],
    "Zomato":    [0.984, 0.983, 0.985, 0.982, 0.989],
    "Phone":     [0.970, 0.973, 0.975, 0.970, 0.971],
}

stds = {
    "Buy":       [0.026, 0.017, 0.013, 0.007, 0.020],
    "Restaurant":[0.007, 0.013, 0.028, 0.018, 0.020],
    "Zomato":    [0.015, 0.008, 0.016, 0.007, 0.009],
    "Phone":     [0.016, 0.016, 0.012, 0.005, 0.008],
}

# Convert to percentages
for k in means:
    means[k] = [v * 100 for v in means[k]]
    stds[k] = [s * 100 for s in stds[k]]

# Define markers and color palette
markers = ["o", "s", "^", "D"]  # circle, square, triangle, diamond
colors = plt.cm.tab10.colors      # 10 distinct colors

# Plot
plt.figure(figsize=(8, 4))
for i, dataset in enumerate(means.keys()):
    mean_vals = np.array(means[dataset])
    std_vals = np.array(stds[dataset])
    
    plt.plot(missingness, mean_vals, 
             marker=markers[i], 
             color=colors[i], 
             label=dataset, 
             linewidth=2, 
             markersize=8)
    
    plt.fill_between(missingness,
                     mean_vals - std_vals,
                     mean_vals + std_vals,
                     color=colors[i],
                     alpha=0.2)

plt.rcParams['legend.fontsize'] = 16
#plt.title("Impact of Missing Data on Model Accuracy")
plt.xlabel("Missing Rate (%)", fontsize=16)
plt.ylabel("Accuracy (%)", fontsize=16)
plt.ylim(0, 101)  # zero baseline, 100% top
plt.xticks(missingness, fontsize=14)
plt.yticks(fontsize=14)
#plt.legend(fontsize=14, markerscale=1.5, handlelength=2.5)
plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=100))  # format as %
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# Save as PDF
plt.savefig("fig.pdf", format="pdf")

plt.show()
