import numpy as np
import matplotlib.pyplot as plt

def analyze_recoil_data(filename):
    """Load and preprocess recoil data from NPZ file."""
    try:
        data = np.load(filename, allow_pickle=True)
        raw_ys = data['ys']    # Y (Recoil Control Vertikal)
        raw_xs = data['xs']    # X axis
        
        print(f"Data berhasil dimuat dari {filename}. Jumlah percobaan: {len(raw_ys)}")
    except Exception as e:
        print(f"Error memuat file {filename}: {e}")
        return None, None

    # Find minimum length for trimming
    min_length = min([len(trial) for trial in raw_ys])
    print(f"Trimming data ke panjang minimum: {min_length} sampel")

    processed_trials = []
    
    for i in range(len(raw_ys)):
        y_trial = raw_ys[i]
        y_trimmed = y_trial[:min_length]
        y_relative = y_trimmed - y_trimmed[0]
        processed_trials.append(y_relative)

    # Bikin Matriks A (Baris = Waktu, Kolom = Percobaan)
    matrix_A = np.column_stack(processed_trials)
    
    print(f"Matriks terbentuk. Dimensi: {matrix_A.shape}")

    # COMPUTING SVD
    U, S, Vt = np.linalg.svd(matrix_A, full_matrices=False)
    
    print("\nNilai Singular (Sigma):")
    print(np.round(S, 4))
    
    # Energi (Variansi yang dijelaskan)
    energy = (S ** 2) / np.sum(S ** 2) * 100
    print("Kontribusi Energi per Rank (%):")
    print(np.round(energy, 2))
    print(f"Energy concentration (E1): {energy[0]:.2f}%\n")

    return matrix_A, S, energy

def compare_two_datasets(mouse_file, script_file):
    """Compare mouse_trials and script_trials with side-by-side visualizations."""
    
    print("="*60)
    print("MEMBANDINGKAN MOUSE TRIALS vs SCRIPT TRIALS")
    print("="*60)
    
    # Load and analyze both datasets
    print("\n[1] LOADING MOUSE TRIALS")
    print("-"*60)
    matrix_mouse, S_mouse, energy_mouse = analyze_recoil_data(mouse_file)
    
    print("\n[2] LOADING SCRIPT TRIALS")
    print("-"*60)
    matrix_script, S_script, energy_script = analyze_recoil_data(script_file)
    
    if matrix_mouse is None or matrix_script is None:
        print("Error: Gagal memuat salah satu atau kedua file.")
        return
    
    # ==========================================
    # VISUALISASI 1: TRAJEKTORI PERBANDINGAN
    # ==========================================
    
    fig1, ax1 = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mouse Trials Trajectory
    ax1[0].plot(matrix_mouse, linewidth=1.5, alpha=0.7)
    ax1[0].set_title(f"Mouse Trials: Trajektori Vertikal\n(E1: {energy_mouse[0]:.2f}%)", fontsize=11, fontweight='bold')
    ax1[0].set_xlabel("Waktu (Sampel)")
    ax1[0].set_ylabel("Pergeseran Pixel (Relatif)")
    ax1[0].grid(True, alpha=0.3)
    ax1[0].legend([f'Trial {i+1}' for i in range(matrix_mouse.shape[1])], fontsize=8)
    
    # Script Trials Trajectory
    ax1[1].plot(matrix_script, linewidth=1.5, alpha=0.7)
    ax1[1].set_title(f"Script Trials: Trajektori Vertikal\n(E1: {energy_script[0]:.2f}%)", fontsize=11, fontweight='bold')
    ax1[1].set_xlabel("Waktu (Sampel)")
    ax1[1].set_ylabel("Pergeseran Pixel (Relatif)")
    ax1[1].grid(True, alpha=0.3)
    ax1[1].legend([f'Trial {i+1}' for i in range(matrix_script.shape[1])], fontsize=8)
    
    plt.tight_layout()
    plt.savefig('fig1_trajectories.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ==========================================
    # VISUALISASI 2: SCREE PLOT (SINGULAR VALUES)
    # ==========================================
    
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mouse Trials Scree Plot
    ranks_mouse = np.arange(1, len(S_mouse) + 1)
    bars_mouse = ax2[0].bar(ranks_mouse, S_mouse, color='steelblue', edgecolor='black', alpha=0.7)
    ax2[0].set_title(f"Mouse Trials: Scree Plot\n(E1 = {energy_mouse[0]:.2f}%)", fontsize=11, fontweight='bold')
    ax2[0].set_xlabel("Rank (k)")
    ax2[0].set_ylabel("Nilai Singular")
    ax2[0].set_xticks(ranks_mouse)
    ax2[0].grid(True, axis='y', alpha=0.3)
    
    for bar, val, pct in zip(bars_mouse, S_mouse, energy_mouse):
        height = bar.get_height()
        ax2[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=8)
    
    # Script Trials Scree Plot
    ranks_script = np.arange(1, len(S_script) + 1)
    bars_script = ax2[1].bar(ranks_script, S_script, color='coral', edgecolor='black', alpha=0.7)
    ax2[1].set_title(f"Script Trials: Scree Plot\n(E1 = {energy_script[0]:.2f}%)", fontsize=11, fontweight='bold')
    ax2[1].set_xlabel("Rank (k)")
    ax2[1].set_ylabel("Nilai Singular")
    ax2[1].set_xticks(ranks_script)
    ax2[1].grid(True, axis='y', alpha=0.3)
    
    for bar, val, pct in zip(bars_script, S_script, energy_script):
        height = bar.get_height()
        ax2[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('fig2_scree_plots.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ==========================================
    # VISUALISASI 3: ENERGY DISTRIBUTION
    # ==========================================
    
    fig3, ax3 = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mouse Trials Energy
    ax3[0].bar(ranks_mouse, energy_mouse, color='steelblue', edgecolor='black', alpha=0.7)
    ax3[0].set_title("Mouse Trials: Distribusi Energi (%)", fontsize=11, fontweight='bold')
    ax3[0].set_xlabel("Rank (k)")
    ax3[0].set_ylabel("Energi (%)")
    ax3[0].set_xticks(ranks_mouse)
    ax3[0].grid(True, axis='y', alpha=0.3)
    ax3[0].axhline(y=90, color='red', linestyle='--', linewidth=2, label='Threshold (90%)')
    ax3[0].legend()
    
    # Script Trials Energy
    ax3[1].bar(ranks_script, energy_script, color='coral', edgecolor='black', alpha=0.7)
    ax3[1].set_title("Script Trials: Distribusi Energi (%)", fontsize=11, fontweight='bold')
    ax3[1].set_xlabel("Rank (k)")
    ax3[1].set_ylabel("Energi (%)")
    ax3[1].set_xticks(ranks_script)
    ax3[1].grid(True, axis='y', alpha=0.3)
    ax3[1].axhline(y=90, color='red', linestyle='--', linewidth=2, label='Threshold (90%)')
    ax3[1].legend()
    
    plt.tight_layout()
    plt.savefig('fig3_energy_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print Summary
    print("\n" + "="*60)
    print("RINGKASAN PERBANDINGAN")
    print("="*60)
    print(f"Mouse Trials   - E1: {energy_mouse[0]:.2f}% | Singular Values: {np.round(S_mouse, 2)}")
    print(f"Script Trials  - E1: {energy_script[0]:.2f}% | Singular Values: {np.round(S_script, 2)}")
    print(f"\nSelisih E1: {abs(energy_mouse[0] - energy_script[0]):.2f}%")
    
    if energy_script[0] > 90 and energy_mouse[0] < 85:
        print("\n✓ KESIMPULAN: Script trials menunjukkan struktur LOW-RANK (synthetic)")
        print("✓ Mouse trials menunjukkan struktur DISTRIBUTED (human)")
    print("="*60 + "\n")

# JALANKAN FUNGSI
# Bandingkan mouse_trials.npz dengan script_trials.npz
compare_two_datasets('mouse_trials.npz', 'script_trials.npz')