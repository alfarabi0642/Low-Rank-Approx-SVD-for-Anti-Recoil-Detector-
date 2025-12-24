"""
lua_assumed_to_csv.py

Generate CSV + NPZ with assumed mouse motion from a Logitech-style macro:
 - assumption: MoveMouseRelative(0, +5) called every 6 ms, continuously for the trial
 - sampling: sample every INTERVAL seconds (default 0.01 s)
 - duration: default 2.0 s (0..2s)
 - N_TRIALS: default 5

Output:
 - CSV: mouse_trials.csv (each row = one trial, columns trial_id, x_0.000,y_0.000,...)
 - NPZ: mouse_trials.npz (xs, ys, times)
"""

import numpy as np
import csv

# -------------------------
# CONFIG (ubah jika perlu)
# -------------------------
DURATION = 2.0         # seconds, record from t=0 to t=2.0
INTERVAL = 0.01        # sampling interval in seconds (100 Hz)
N_TRIALS = 3           # number of trials
CSV_PATH = "script_trials.csv"
NPZ_PATH = "script_trials.npz"
# Assumed macro parameters (from your lua): every 6 ms move (0, +5)
MOVE_DX = 0
MOVE_DY = 5
MOVE_MS = 6            # milliseconds between MoveMouseRelative calls
# initial position for each trial (can change)
INITIAL_POS = (500.0, 300.0)
# -------------------------

def build_assumed_trace(duration, interval, move_dx, move_dy, move_ms, initial_pos):
    """
    Build sampled xs, ys arrays for one trial based on the assumption:
    - Move (move_dx, move_dy) occurs every move_ms milliseconds starting at t=0
    - We sample the mouse position every 'interval' seconds from 0..duration inclusive
    """
    n_samples = int(round(duration / interval)) + 1
    times = np.array([round(i * interval, 6) for i in range(n_samples)])
    xs = np.zeros(n_samples, dtype=float)
    ys = np.zeros(n_samples, dtype=float)

    # convert move interval to seconds
    move_dt = move_ms / 1000.0
    # next move time scheduled, starting at t=0
    next_move_time = 0.0

    x = float(initial_pos[0])
    y = float(initial_pos[1])

    for i, t in enumerate(times):
        # apply all moves whose scheduled time <= current sample time
        # (handles cases where multiple moves happen between two sample points)
        # careful with infinite loops if move_dt == 0 (shouldn't happen here)
        if move_dt > 0:
            # number of moves that have occurred up to time t (inclusive)
            # floor((t + epsilon) / move_dt) - floor((prev_t + epsilon) / move_dt)
            # but simpler: loop while next_move_time <= t
            safety = 0
            while next_move_time <= t + 1e-12:
                x += move_dx
                y += move_dy
                next_move_time += move_dt
                safety += 1
                # safety guard (should not trigger)
                if safety > 1000000:
                    raise RuntimeError("Too many iterations in move loop (safety stop).")
        else:
            # if move_dt <= 0, no periodic moves; do nothing
            pass

        xs[i] = x
        ys[i] = y

    return xs, ys, times

def write_outputs(xs_all, ys_all, times, csv_path, npz_path):
    n_trials, n_samples = xs_all.shape
    # CSV header
    header = ["trial_id"]
    for tt in times:
        header.append(f"x_{tt:.3f}")
        header.append(f"y_{tt:.3f}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for tr in range(n_trials):
            row = [tr + 1]  # 1-based trial id
            for i in range(n_samples):
                row.append(f"{xs_all[tr, i]:.6f}")
                row.append(f"{ys_all[tr, i]:.6f}")
            writer.writerow(row)

    # save NPZ
    np.savez(npz_path, xs=xs_all, ys=ys_all, times=times)
    print(f"Saved CSV -> {csv_path}")
    print(f"Saved NPZ -> {npz_path}")

def main():
    n_samples = int(round(DURATION / INTERVAL)) + 1
    xs_all = np.zeros((N_TRIALS, n_samples), dtype=float)
    ys_all = np.zeros((N_TRIALS, n_samples), dtype=float)
    times = np.array([round(i * INTERVAL, 6) for i in range(n_samples)])

    print("Generating assumed traces with parameters:")
    print(f" DURATION={DURATION}s, INTERVAL={INTERVAL}s, samples={n_samples}")
    print(f" MOVE: dx={MOVE_DX}, dy={MOVE_DY} every {MOVE_MS} ms")
    print(f" N_TRIALS={N_TRIALS}, INITIAL_POS={INITIAL_POS}")

    for t in range(N_TRIALS):
        xs, ys, _ = build_assumed_trace(DURATION, INTERVAL, MOVE_DX, MOVE_DY, MOVE_MS, INITIAL_POS)
        xs_all[t, :] = xs
        ys_all[t, :] = ys
        print(f" - trial {t+1} done")

    write_outputs(xs_all, ys_all, times, CSV_PATH, NPZ_PATH)
    print("All done.")

if __name__ == "__main__":
    main()
