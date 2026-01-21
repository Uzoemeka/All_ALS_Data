import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def forest_plot_fp_model_haz_scale(df_coef):
    # Create figure with subplots for labels and plot
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8, 5), 
                                        gridspec_kw={'width_ratios': [3, 4, 3]})

    y_positions = np.arange(len(df_coef))

    # Left panel - Variable names
    ax1.axis('off')
    ax1.text(0, len(df_coef), 'Variable', fontweight='bold', fontsize=9)
    for idx, var in enumerate(df_coef['Variable']):
        ax1.text(0, len(df_coef) - idx - 1, var, fontsize=8, va='center')
    ax1.set_ylim(-1, len(df_coef) + 0.5)

    # Middle panel - Forest plot
    for idx, row in df_coef.iterrows():
        y_pos = len(df_coef) - idx - 1
        color = '#d62728' if row['Significant'] else '#7f7f7f'

        # CI line
        ax2.plot([row['HR_lower'], row['HR_upper']], [y_pos, y_pos], 
                color=color, linewidth=2, alpha=0.8)

        # Point estimate
        ax2.scatter(row['HR'], y_pos, s=80, color=color, 
                zorder=3, edgecolors='black', linewidth=1)

    # Reference line
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    ax2.set_xlabel('Hazard Ratio', fontsize=10, fontweight='bold')
    ax2.set_yticks([])
    ax2.set_ylim(-1, len(df_coef) + 0.5)
    ax2.spines['left'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(axis='x', alpha=0.3, linestyle=':', linewidth=0.8)
    ax2.set_xlim(left=min(-2, df_coef['HR_lower'].min() - 0.5))

    # Right panel - HR values and p-values
    ax3.axis('off')
    ax3.text(0, len(df_coef), 'HR (95% CI)', fontweight='bold', fontsize=9)
    ax3.text(0.6, len(df_coef), 'p-value', fontweight='bold', fontsize=9)

    for idx, row in df_coef.iterrows():
        y_pos = len(df_coef) - idx - 1
        hr_text = f"{row['HR']:.2f} ({row['HR_lower']:.2f}, {row['HR_upper']:.2f})"
        p_text = f"{row['Pr(z)']:.3f}" if row['Pr(z)'] >= 0.001 else "<0.001"

        ax3.text(0, y_pos, hr_text, fontsize=7, va='center')
        ax3.text(0.6, y_pos, p_text, fontsize=7, va='center')

    ax3.set_ylim(-1, len(df_coef) + 0.5)

    plt.suptitle('Hazard Ratios from Flexible Parametric Survival Model', 
                fontsize=11, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
