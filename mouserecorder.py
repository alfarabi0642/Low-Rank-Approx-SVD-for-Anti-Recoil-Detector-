"""
mouse_recorder_matrix.py

Rekam posisi mouse dari t=0 sampai t=2 detik, sebanyak N_TRIALS percobaan.
Output:
 - CSV: each row = one trial, columns = trial_id, x0,y0,x1,y1,... (time is implicit: 0..duration)
 - NPZ: contains arrays xs (trials, samples), ys (trials, samples), times (samples,)
"""

import time
import csv
import numpy as np
from pynput.mouse import Controller

# -------------------------
# CONFIG
# -------------------------
DURATION = 1.0            # seconds: rekam dari 0 sampai 2 detik
INTERVAL = 0.01           # sampling interval in seconds (0.01 ~ 100 Hz)
N_TRIALS = 3             # jumlah percobaan
CSV_PATH = "mouse_trials.csv"
NPZ_PATH = "mouse_trials.npz"
# -------------------------

# compute number of samples including t=0 and t=DURATION
n_samples = int(round(DURATION / INTERVAL)) + 1

print(f"Konfigurasi: duration={DURATION}s, interval={INTERVAL}s (~{1/INTERVAL:.0f} Hz), samples={n_samples}, trials={N_TRIALS}")

mouse = Controller()

# prepare storage
xs = np.zeros((N_TRIALS, n_samples), dtype=float)
ys = np.zeros((N_TRIALS, n_samples), dtype=float)
times = np.array([round(i * INTERVAL, 6) for i in range(n_samples)])  # 0, 0.01, 0.02, ..., 2.0

def record_one_trial(trial_index):
    """
    Rekam satu trial. Menggunakan scheduling dengan target times
    untuk meminimalkan drift: sample di waktu start + i*INTERVAL.
    """
    input(f"\n[Trial {trial_index+1}/{N_TRIALS}] Siap? posisikan mouse lalu tekan Enter untuk mulai rekaman.")
    print("Mulai rekaman dalam: 3..2..1")
    for i in (3,2,1):
        print(i, end=" ", flush=True)
        time.sleep(1)
    print("\nRekam...")

    start = time.perf_counter()
    for i in range(n_samples):
        target = start + i * INTERVAL
        # tunggu sampai target (sleep sebagian + busy-wait jika perlu)
        now = time.perf_counter()
        to_sleep = target - now
        if to_sleep > 0.002:
            time.sleep(to_sleep - 0.001)  # tidur sebagian
        # busy wait short interval to improve timing
        while time.perf_counter() < target:
            pass
        # read position (relative timestamp = actual - start)
        px, py = mouse.position
        xs[trial_index, i] = px
        ys[trial_index, i] = py

    actual_end = time.perf_counter() - start
    print(f"Trial {trial_index+1} selesai. Durasi aktual: {actual_end:.4f}s (target {DURATION}s)")

# run all trials
try:
    for t in range(N_TRIALS):
        record_one_trial(t)
except KeyboardInterrupt:
    print("\nDirem (KeyboardInterrupt). Data sampai trial terakhir yang lengkap akan disimpan.")

# Ensure we have valid numbers; if any trailing zeros because of interruption, you can handle separately.
# Save CSV: header + each trial flattened as x0,y0,x1,y1,...
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    # header: trial_id, then pairs t=0..t=end as "x_0.000","y_0.000",...
    header = ["trial_id"]
    for tt in times:
        header.append(f"x_{tt:.3f}")
        header.append(f"y_{tt:.3f}")
    writer.writerow(header)

    for trial_idx in range(xs.shape[0]):
        row = [trial_idx + 1]  # trial numbering 1-based
        # interleave x,y
        for i in range(n_samples):
            row.append(f"{xs[trial_idx, i]:.6f}")
            row.append(f"{ys[trial_idx, i]:.6f}")
        writer.writerow(row)

print(f"CSV disimpan ke: {CSV_PATH}")

# Save binary numpy archive for easy loading
np.savez(NPZ_PATH, xs=xs, ys=ys, times=times)
print(f"NPZ disimpan ke: {NPZ_PATH}")

# Example mapping functions (untuk analisis selanjutnya)
def load_npz_to_matrices(path):
    data = np.load(path)
    xs = data["xs"]   # shape (trials, samples)
    ys = data["ys"]   # shape (trials, samples)
    
    times = data["times"]  # shape (samples,)
    # Option A: interleaved flattened matrix where each row = trial
    # shape => (trials, samples*2)
    interleaved = np.empty((xs.shape[0], xs.shape[1] * 2), dtype=float)
    for i in range(xs.shape[0]):
        interleaved[i, 0::2] = xs[i]
        interleaved[i, 1::2] = ys[i]
    # Option B: 3D matrix (trials, samples, 2) useful for ML/plotting
    mat3d = np.stack((xs, ys), axis=2)  # shape (trials, samples, 2)
    return xs, ys, times, interleaved, mat3d

# quick demo how to load
# xs_loaded, ys_loaded, times_loaded, interleaved_matrix, mat3d = load_npz_to_matrices(NPZ_PATH)
# print("interleaved_matrix shape:", interleaved_matrix.shape)
# print("mat3d shape:", mat3d.shape)

print("Selesai.")
