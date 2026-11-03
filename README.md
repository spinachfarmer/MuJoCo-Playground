# MuJoCo Playground Notes

## Berkeley Humanoid

Open the focused training notebook in Colab:

[Berkeley Humanoid PPO notebook](https://colab.research.google.com/github/spinachfarmer/MuJoCo-Playground/blob/codex/clean-berkeley-humanoid-notebook/learning/notebooks/berkeley_humanoid.ipynb)

The notebook has a short smoke-test mode and the full 150-million-step mode.
It saves trained parameters, metrics, plots, and rollout videos separately from
the Git repository.

Each run gets a timestamped Google Drive folder. The run summary records the
Git branch and commit that produced the results.

## Live keyboard control on macOS

The live viewer needs a completed `policy_params.pkl` file from Colab.

Create a Python 3.12 environment and install the matching packages:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements-macos.txt
```

Download `policy_params.pkl` from the run folder in Google Drive. Then start
the viewer with MuJoCo's macOS launcher:

```bash
mjpython learning/berkeley_humanoid_keyboard.py /path/to/policy_params.pkl
```

Use `W/S` for forward and backward commands. Use `A/D` for sideways commands.
Use `Q/E` to turn. Use `X` to stop. Use `R` to reset the humanoid.
