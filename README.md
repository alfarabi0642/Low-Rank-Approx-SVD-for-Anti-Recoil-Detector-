# SVD-Based Recoil Control Detection

## Overview

This program detects automated recoil control macros in Valorant by analyzing mouse movement patterns using **Singular Value Decomposition (SVD)**. It works by examining the structure of mouse trajectories: synthetic automated movements (macros) produce perfectly reproducible patterns that concentrate most variance in a single mode (>97%), while human-controlled movement distributes energy across multiple modes due to natural variability.

## Key Features

- **Analyzes Mouse Patterns**: Uses SVD to detect structured patterns in mouse movement data
- **Fast Detection**: Quick analysis suitable for real-time classification
- **Three Analysis Modes**: Analyze existing data, record human movement, or generate synthetic patterns
- **Clear Results**: Outputs energy concentration metrics and classification (human vs. synthetic)

## Project Structure

```
├── main.py                 # Core SVD analysis and detection pipeline
├── mouserecorder.py        # Captures real human mouse movements
├── skrip_recorder.py       # Generates synthetic macro trajectories
├── mouse_trials.npz        # Human trial data (3 trials)
├── script_trials.npz       # Synthetic trial data (3 trials)
├── mouse_trials.csv        # Human trial data (CSV format)
└── script_trials.csv       # Synthetic trial data (CSV format)
```

## Installation

### Requirements
- Python 3.7+
- NumPy
- Matplotlib
- pynput (for mouse input capture)

### Setup

```bash
# Clone the repository
git clone https://github.com/alfarabi0642/Low-Rank-Approx-SVD-for-Anti-Recoil-Detector-.git
cd Low-Rank-Approx-SVD-for-Anti-Recoil-Detector-

# Install dependencies
pip install numpy matplotlib pynput
```

## Usage

### 1. Analyze Existing Data

Run the main analysis on pre-recorded trials:

```bash
python main.py
```

This script will:
- Load human trials from `mouse_trials.npz`
- Load synthetic trials from `script_trials.npz`
- Compute SVD decomposition for each
- Display singular values and energy concentration
- Generate visualization comparing trajectories, scree plots, and energy distribution
- Output detection classification for each trial

### 2. Record Human Mouse Movement

Capture your own human recoil control trials:

```bash
python mouserecorder.py
```

The script will:
- Prompt you to position your mouse
- Record 3 trials of ~2 seconds each at ~100 Hz sampling rate
- Save data to `mouse_trials.npz` and `mouse_trials.csv`

**Configuration** (in `mouserecorder.py`):
```python
DURATION = 2.0      # Recording duration in seconds
INTERVAL = 0.01     # Sampling interval (100 Hz = 10 ms)
N_TRIALS = 3        # Number of trials
```

### 3. Generate Synthetic Trajectories

Create synthetic macro trajectories (for testing):

```bash
python skrip_recorder.py
```

This simulates a Valorant recoil macro that:
- Executes identical downward mouse movements (+5 pixels every 6 ms)
- Repeats perfectly across multiple trials
- Produces near-rank-1 matrix structure

**Configuration** (in `skrip_recorder.py`):
```python
DURATION = 2.0      # Recording duration
INTERVAL = 0.01     # Sampling interval
N_TRIALS = 3        # Number of trials
MOVE_DY = 5         # Pixels moved per step
MOVE_MS = 6         # Milliseconds between steps
```

## Methodology

### How It Works

Mouse trajectories are represented as matrices where:
- **Rows**: Time samples (at 10 ms intervals)
- **Columns**: Individual trials
- **Entries**: Vertical (Y-axis) mouse displacement

The program computes the SVD decomposition of this matrix and analyzes the "energy concentration" in the first singular value:
- **Synthetic motion**: >97% of energy in the first mode (perfectly reproducible patterns)
- **Human motion**: <82% of energy in the first mode (natural variability)

### Detection Logic

```
If E₁ > 0.99 → SYNTHETIC MACRO DETECTED
Else → HUMAN CONTROLLED MOVEMENT
```

Where E₁ is the fraction of total matrix variance captured by the first singular value.

## Results

The program produces analysis output showing:
- Singular values of the motion matrix
- Energy concentration percentages for each mode
- Classification result (SYNTHETIC or HUMAN)
- Visualizations comparing trajectories, singular value decay, and energy distribution

### Example Output

**Synthetic Motion (Perfect Macro)**:
- E₁ = 98.22% → **SYNTHETIC DETECTED**

**Human Motion (Manual Control)**:
- E₁ = 73.9% ± 6.7% → **HUMAN DETECTED**


## Program Output

The analysis generates visualizations and metrics:
- **Trajectory Comparison**: Visual comparison of recorded movement patterns
- **Scree Plots**: Singular value decay across modes
- **Energy Distribution**: Shows concentration of variance in each mode
- **Classification**: Human vs. Synthetic determination


