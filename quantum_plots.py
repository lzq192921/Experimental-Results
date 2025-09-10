import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Prepare quantum algorithm data
quantum_data = {
    'Medium noise': [
        (2, 2, 41, 3.528314, 22, 0.9584),
        (3, 2, 60, 203.840280, 29, 0.8658),
        (3, 3, 61, 750.564494, 32, 0.6948),
        (4, 2, 85, 6891.101911, 36, 0.8280)
    ],
    'Low noise': [
        (2, 2, 41, 3.560275, 22, 0.9968),
        (3, 2, 60, 194.157927, 29, 0.9392),
        (3, 3, 61, 768.232335, 32, 0.7254),
        (4, 2, 85, 7022.866378, 36, 0.9410)
    ],
    'No noise': [
        (2, 2, 41, 0.039322, 22, 1.0000),
        (3, 2, 60, 0.057519, 29, 0.9416),
        (3, 3, 61, 0.153576, 32, 0.7080),
        (4, 2, 85, 1.171445, 36, 0.9570),
        (4, 3, 92, 18.681235, 42, 0.9644),
        (4, 4, 87, 16.367083, 42, 0.9616)
    ]
}

# Prepare classical algorithm data
classical_data = [
    (2, 2, 4, 1306, 1.0000, 0.000000),
    (3, 2, 8, 3308, 1.0000, 0.000010),
    (3, 3, 8, 2906, 1.0000, 0.000000),
    (4, 2, 16, 3710, 1.0000, 0.000010),
    (4, 3, 16, 3308, 1.0000, 0.000010),
    (4, 4, 16, 2906, 1.0000, 0.000010)
]

# Convert to DataFrames
quantum_dfs = {}
for noise_level in quantum_data:
    quantum_dfs[noise_level] = pd.DataFrame(
        quantum_data[noise_level],
        columns=['k', 'n', 'Total_gates', 'Simulation_time', 'Total_qubits', 'Success_prob']
    )

classical_df = pd.DataFrame(
    classical_data,
    columns=['k', 'n', 'Search_space', 'Total_ops', 'Success_prob', 'Avg_time']
)

# Define styles
markers = {
    'Medium noise': 'o', 
    'Low noise': 's', 
    'No noise': '^',
    'Classical': 'D'  # Diamond marker for classical algorithm
}

colors = {
    'Medium noise': 'red', 
    'Low noise': 'blue', 
    'No noise': 'green',
    'Classical': 'purple'  # Purple color for classical algorithm
}

# Define save directory and create if it doesn't exist
save_dir = r"D:\quantum"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ---------------------------
# Comprehensive chart: 3 rows x 2 columns layout
# ---------------------------
fig = plt.figure(figsize=(18, 20))
#fig.suptitle('Quantum vs. Classical Algorithm Performance Comparison', 
             #fontsize=20, y=0.98)

# ---------------------------
# Row 1, Column 1: Success rate for different (k,n) combinations
# ---------------------------
ax1 = plt.subplot(321)
# Quantum algorithm data
for noise_level in quantum_dfs:
    df = quantum_dfs[noise_level]
    labels = [f'k={k},n={n}' for k, n in zip(df['k'], df['n'])]
    ax1.plot(labels, df['Success_prob']*100,
             marker=markers[noise_level], color=colors[noise_level],
             label=noise_level, linewidth=2, markersize=8)

# Classical algorithm data
classical_labels = [f'k={k},n={n}' for k, n in zip(classical_df['k'], classical_df['n'])]
ax1.plot(classical_labels, classical_df['Success_prob']*100,
         marker=markers['Classical'], color=colors['Classical'],
         label='Classical', linewidth=2, markersize=8)

ax1.set_title('(a) Success Rate by (k,n) Configuration', fontsize=16)
ax1.set_ylabel('Success Rate (%)', fontsize=14)
ax1.set_ylim(60, 105)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11, loc='lower right')
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=11)
plt.setp(ax1.get_yticklabels(), fontsize=11)

# ---------------------------
# Row 1, Column 2: Impact of k value on average success rate
# ---------------------------
ax2 = plt.subplot(322)
# Quantum algorithm data
for noise_level in quantum_dfs:
    df = quantum_dfs[noise_level]
    avg_success = df.groupby('k')['Success_prob'].mean() * 100
    ax2.plot(avg_success.index, avg_success.values,
             marker=markers[noise_level], color=colors[noise_level],
             label=noise_level, linewidth=2, markersize=8)

# Classical algorithm data
classical_avg_success = classical_df.groupby('k')['Success_prob'].mean() * 100
ax2.plot(classical_avg_success.index, classical_avg_success.values,
         marker=markers['Classical'], color=colors['Classical'],
         label='Classical', linewidth=2, markersize=8)

ax2.set_title('(b) Average Success Rate by k Value', fontsize=16)
ax2.set_xlabel('k Value', fontsize=14)
ax2.set_ylabel('Average Success Rate (%)', fontsize=14)
ax2.set_xticks([2, 3, 4])
ax2.set_ylim(60, 105)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(fontsize=11)
plt.setp(ax2.get_xticklabels(), fontsize=11)
plt.setp(ax2.get_yticklabels(), fontsize=11)

# ---------------------------
# Row 2, Column 1: Runtime for different (k,n) combinations
# ---------------------------
ax3 = plt.subplot(323)
# Quantum algorithm data
for noise_level in quantum_dfs:
    df = quantum_dfs[noise_level]
    labels = [f'k={k},n={n}' for k, n in zip(df['k'], df['n'])]
    ax3.plot(labels, df['Simulation_time'],
             marker=markers[noise_level], color=colors[noise_level],
             label=noise_level, linewidth=2, markersize=8)

# Classical algorithm data
ax3.plot(classical_labels, classical_df['Avg_time'],
         marker=markers['Classical'], color=colors['Classical'],
         label='Classical', linewidth=2, markersize=8)

ax3.set_title('(c) Runtime by (k,n) Configuration', fontsize=16)
ax3.set_ylabel('Runtime (seconds)', fontsize=14)
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(fontsize=11, loc='upper left')
plt.setp(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=11)
plt.setp(ax3.get_yticklabels(), fontsize=11)

# ---------------------------
# Row 2, Column 2: Impact of k value on average runtime
# ---------------------------
ax4 = plt.subplot(324)
# Quantum algorithm data
for noise_level in quantum_dfs:
    df = quantum_dfs[noise_level]
    avg_time = df.groupby('k')['Simulation_time'].mean()
    ax4.plot(avg_time.index, avg_time.values,
             marker=markers[noise_level], color=colors[noise_level],
             label=noise_level, linewidth=2, markersize=8)

# Classical algorithm data
classical_avg_time = classical_df.groupby('k')['Avg_time'].mean()
ax4.plot(classical_avg_time.index, classical_avg_time.values,
         marker=markers['Classical'], color=colors['Classical'],
         label='Classical', linewidth=2, markersize=8)

ax4.set_title('(d) Average Runtime by k Value', fontsize=16)
ax4.set_xlabel('k Value', fontsize=14)
ax4.set_ylabel('Average Runtime (seconds)', fontsize=14)
ax4.set_xticks([2, 3, 4])
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.legend(fontsize=11)
plt.setp(ax4.get_xticklabels(), fontsize=11)
plt.setp(ax4.get_yticklabels(), fontsize=11)

# ---------------------------
# Row 3, Column 1: Operations/gates comparison
# ---------------------------
ax5 = plt.subplot(325)
# Quantum algorithm gates
for noise_level in quantum_dfs:
    df = quantum_dfs[noise_level]
    labels = [f'k={k},n={n}' for k, n in zip(df['k'], df['n'])]
    ax5.plot(labels, df['Total_gates'],
             marker=markers[noise_level], color=colors[noise_level],
             label=f'{noise_level} (gates)', linewidth=2, markersize=8)

# Classical algorithm operations
ax5.plot(classical_labels, classical_df['Total_ops'],
         marker=markers['Classical'], color=colors['Classical'],
         label='Classical (operations)', linewidth=2, markersize=8)

ax5.set_title('(e) Computational Operations by (k,n) Configuration', fontsize=16)
ax5.set_ylabel('Number of Operations/Gates', fontsize=14)
ax5.set_yscale('log')
ax5.grid(True, alpha=0.3, linestyle='--')
ax5.legend(fontsize=11, loc='upper left')
plt.setp(ax5.get_xticklabels(), rotation=45, ha='right', fontsize=11)
plt.setp(ax5.get_yticklabels(), fontsize=11)

# ---------------------------
# Row 3, Column 2: Impact of k value on average operations/gates
# ---------------------------
ax6 = plt.subplot(326)
# Average gates for quantum algorithms
for noise_level in quantum_dfs:
    df = quantum_dfs[noise_level]
    avg_gates = df.groupby('k')['Total_gates'].mean()
    ax6.plot(avg_gates.index, avg_gates.values,
             marker=markers[noise_level], color=colors[noise_level],
             label=f'{noise_level} (gates)', linewidth=2, markersize=8)

# Average operations for classical algorithm
classical_avg_ops = classical_df.groupby('k')['Total_ops'].mean()
ax6.plot(classical_avg_ops.index, classical_avg_ops.values,
         marker=markers['Classical'], color=colors['Classical'],
         label='Classical (operations)', linewidth=2, markersize=8)

ax6.set_title('(f) Average Operations by k Value', fontsize=16)
ax6.set_xlabel('k Value', fontsize=14)
ax6.set_ylabel('Average Number of Operations/Gates', fontsize=14)
ax6.set_xticks([2, 3, 4])
ax6.set_yscale('log')
ax6.grid(True, alpha=0.3, linestyle='--')
ax6.legend(fontsize=11)
plt.setp(ax6.get_xticklabels(), fontsize=11)
plt.setp(ax6.get_yticklabels(), fontsize=11)

# Adjust layout and save
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(save_dir, 'quantum_vs_classical_performance.png'), 
            dpi=300, bbox_inches='tight')
plt.show()
