import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def plot_histograms_with_kde(df):

    # Plot histogram and density
    cols = df.select_dtypes(include=['float']).columns
    n_per_row = 3
    n_rows = int(np.ceil(len(cols) / n_per_row))

    fig, axes = plt.subplots(n_rows, n_per_row,
        figsize=(4 * n_per_row, 3 * n_rows)
    )

    axes = axes.flatten()

    for ax, col in zip(axes, cols):
        data = df[col].dropna()

        ax.hist(data, bins=20, density=True, edgecolor='skyblue')

        kde = gaussian_kde(data)
        x = np.linspace(data.min(), data.max(), 300)
        ax.plot(x, kde(x))

        ax.set_title(col)
        ax.set_ylabel('Density')

    for ax in axes[len(cols):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# plot_histograms_with_kde(df)
