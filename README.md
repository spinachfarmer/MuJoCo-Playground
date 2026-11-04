# Berkeley Humanoid in MuJoCo Playground

This repository stores reproducible code and notebooks. Large generated files
belong in Google Drive, not GitHub.

## Project layout

```text
learning/
├── berkeley_humanoid_keyboard.py   # Local interactive viewer
└── notebooks/
    └── berkeley_humanoid.ipynb     # Colab training and rollout notebook

requirements-macos.txt              # Local viewer dependencies
README.md                            # Setup and workflow
```

## Berkeley Humanoid

Open the focused training notebook in Colab:

[Berkeley Humanoid PPO notebook](https://colab.research.google.com/github/spinachfarmer/MuJoCo-Playground/blob/codex/clean-berkeley-humanoid-notebook/learning/notebooks/berkeley_humanoid.ipynb)

The notebook has a short smoke-test mode and the full 150-million-step mode.
It saves trained parameters, metrics, plots, and rollout videos separately from
the Git repository.

Each run gets a timestamped Google Drive folder. The run summary records the
Git branch and commit that produced the results.

The current full run is stored here:

[2026-11-04 full Berkeley Humanoid run](https://drive.google.com/drive/folders/1D_JHDCdBGLGt7AvYMRHVPOX3_DlQhUaC)

Each run folder can contain:

```text
policy_params.pkl              # Trained policy parameters
run_summary.json               # Training configuration and Git commit
learning_curve.png             # Evaluation reward plot
keyboard_command_rollout.mp4   # Colab-generated rollout
interactive_keyboard_demo.mov  # Local interactive demonstration
```

## Live keyboard control on macOS

The live viewer needs a completed `policy_params.pkl` file from Colab.

Create a Python 3.12 environment and install the matching packages:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements-macos.txt
```

Download `policy_params.pkl` from the run folder in Google Drive. Store it in
a timestamped folder under `Training Artifacts/BerkeleyHumanoid/runs/`.

Then start the viewer with MuJoCo's macOS launcher:

```bash
mjpython learning/berkeley_humanoid_keyboard.py /path/to/policy_params.pkl
```

The viewer uses hold-to-move controls:

- Hold `Up` or `Down` to move forward or backward.
- Hold `Left` or `Right` to turn.
- Hold `A` or `D` to move sideways.
- Release the movement keys to stop.
- Use `8`, `2`, `4`, or `6` to apply one manual push.
- Use `R` to reset and `Space` to pause.

macOS Input Monitoring must allow `MuJoCo_(mjpython)` for held-key detection.

## Storage rules

- GitHub stores code, notebooks, configuration, and documentation.
- Google Drive stores policies, videos, plots, checkpoints, and run metadata.
- The local `Training Artifacts` folder stores policies needed by the viewer.
- Never commit `.venv`, `policy_params.pkl`, videos, or notebook checkpoints.
