"""Run a trained Berkeley Humanoid policy with live keyboard commands.

On macOS, start this file with ``mjpython``. The MuJoCo viewer requires the
macOS main thread for rendering.
"""

from __future__ import annotations

import argparse
import pickle
import threading
import time
from pathlib import Path

import jax
from jax import numpy as jp
import mujoco
from mujoco import mjx
from mujoco import viewer
import numpy as np

from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params


ENV_NAME = "BerkeleyHumanoidJoystickFlatTerrain"

COMMAND_UPDATES = {
    "w": (0, 0.7),
    "s": (0, -0.5),
    "a": (1, 0.4),
    "d": (1, -0.4),
    "q": (2, 0.4),
    "e": (2, -0.4),
}

PUSHES = {
    "8": np.array([0.6, 0.0]),
    "2": np.array([-0.6, 0.0]),
    "4": np.array([0.0, 0.6]),
    "6": np.array([0.0, -0.6]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control a trained Berkeley Humanoid policy."
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to policy_params.pkl from a completed Colab run.",
    )
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def load_policy(environment, checkpoint: Path, reset_state):
    """Rebuild the PPO actor and load its saved parameters."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Policy checkpoint not found: {checkpoint}")

    with checkpoint.open("rb") as file:
        params = pickle.load(file)

    ppo_config = locomotion_params.brax_ppo_config(ENV_NAME)
    network_options = dict(ppo_config.network_factory)
    observation_shape = jax.tree_util.tree_map(
        lambda value: value.shape, reset_state.obs
    )
    networks = ppo_networks.make_ppo_networks(
        observation_shape,
        environment.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        **network_options,
    )
    make_policy = ppo_networks.make_inference_fn(networks)
    return jax.jit(make_policy(params, deterministic=True))


def main() -> None:
    args = parse_args()

    config = registry.get_default_config(ENV_NAME)
    config.impl = "jax"
    # Use manual pushes during the demo. The training environment otherwise
    # applies a random push every 5 to 10 seconds.
    config.push_config.enable = False
    environment = registry.load(ENV_NAME, config=config)

    rng = jax.random.PRNGKey(args.seed)
    state = environment.reset(rng)
    policy = load_policy(environment, args.checkpoint, state)
    step = jax.jit(environment.step)

    command = np.zeros(3, dtype=np.float32)
    reset_requested = False
    paused = False
    pending_push = np.zeros(2, dtype=np.float32)
    command_lock = threading.Lock()

    def on_key(keycode: int) -> None:
        nonlocal command, pending_push, reset_requested, paused
        try:
            key = chr(keycode).lower()
        except (ValueError, OverflowError):
            return

        with command_lock:
            if key in COMMAND_UPDATES:
                index, value = COMMAND_UPDATES[key]
                command[index] = value
                print(f"Command [forward, sideways, turn]: {command}")
            elif key == "x":
                command.fill(0.0)
                print(f"Command [forward, sideways, turn]: {command}")
            elif key in PUSHES:
                pending_push += PUSHES[key]
                print(f"Manual push [forward, sideways]: {PUSHES[key]}")
            elif key == "r":
                reset_requested = True
                print("Reset requested.")
            elif key == " ":
                paused = not paused
                print("Paused." if paused else "Running.")

    model = environment.mj_model
    display_data = mujoco.MjData(model)

    print("Motion: W/S forward, A/D sideways, Q/E turn, X stop.")
    print("Push: 8/2 forward/backward, 4/6 left/right.")
    print("Press R to reset. Press Space to pause.")
    print("Click the MuJoCo window before you press a control key.")

    with viewer.launch_passive(
        model,
        display_data,
        key_callback=on_key,
        show_left_ui=False,
        show_right_ui=False,
    ) as simulation_viewer:
        while simulation_viewer.is_running():
            step_started = time.monotonic()

            with command_lock:
                current_command = command.copy()
                current_push = pending_push.copy()
                pending_push.fill(0.0)
                should_reset = reset_requested
                reset_requested = False
                is_paused = paused

            if should_reset or bool(jax.device_get(state.done)):
                rng, reset_rng = jax.random.split(rng)
                state = environment.reset(reset_rng)

            if not is_paused:
                if np.any(current_push):
                    pushed_qvel = state.data.qvel.at[:2].add(
                        jp.asarray(current_push)
                    )
                    state = state.replace(
                        data=state.data.replace(qvel=pushed_qvel)
                    )
                state.info["command"] = jp.asarray(current_command)
                rng, action_rng = jax.random.split(rng)
                action, _ = policy(state.obs, action_rng)
                state = step(state, action)

            mjx.get_data_into(display_data, model, state.data)
            simulation_viewer.sync()

            remaining = float(environment.dt) - (time.monotonic() - step_started)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
