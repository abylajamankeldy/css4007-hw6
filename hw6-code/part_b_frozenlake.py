"""Part B: repeated FrozenLake-v1 simulations with Gymnasium.

Runs two policies (random, fixed) on the default 4x4 map, with and without
slipping, for three batches of 50 episodes each (batch seeds 11, 22, 33).

Usage:  python part_b_frozenlake.py
Writes: outputs/part_b_episodes.csv, outputs/part_b_summary.md, outputs/part_b_traces.txt
"""
import csv
import os
import re

import gymnasium as gym
import numpy as np

MAX_STEPS = 100
BATCH_SEEDS = [11, 22, 33]
EPISODES_PER_BATCH = 50
OUT_DIR = "outputs"

# FrozenLake action numbers (from the Gymnasium documentation)
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
ACTION_NAMES = {LEFT: "left", DOWN: "down", RIGHT: "right", UP: "up"}

# Map (state number = row * 4 + col):
#   S F F F      0  1  2  3
#   F H F H      4  5  6  7
#   F F F H      8  9 10 11
#   H F F G     12 13 14 15
#
# Fixed policy, written BEFORE running experiments.
# Idea: follow the shortest safe path 0 -> 4 -> 8 -> 9 -> 13 -> 14 -> 15,
# and from every other safe tile move back towards that path / the goal.
FIXED_POLICY = {
    0: DOWN,    # start: go down along the left column
    1: LEFT,    # back to the start column (down would be the hole at 5)
    2: DOWN,    # 2 -> 6 -> 10 -> 14 is a second safe route
    3: LEFT,    # down is the hole at 7
    4: DOWN,    # right is the hole at 5
    6: DOWN,    # right is the hole at 7
    8: RIGHT,   # down is the hole at 12
    9: DOWN,    # towards 13
    10: DOWN,   # right is the hole at 11
    13: RIGHT,  # towards 14
    14: RIGHT,  # goal
}


def random_policy(state, rng):
    """Uniformly random action; uses the seeded generator for reproducibility."""
    return int(rng.integers(0, 4))


def fixed_policy(state, rng):
    """Look up the action for the current state; never changes between runs."""
    return FIXED_POLICY[state]


POLICIES = {"random": random_policy, "fixed": fixed_policy}


def clean_render(text):
    """Replace the ANSI colour around the agent's tile with [ ] brackets."""
    return re.sub(r"\x1b\[41m(.)\x1b\[0m", r"[\1]", text).strip("\n")


def run_episode(env, policy, episode_seed, record=False):
    """Run one episode with reset()/step(); stop on terminated, truncated or 100 steps."""
    rng = np.random.default_rng(episode_seed)  # seeds the random action generator
    state, _ = env.reset(seed=episode_seed)    # seeds the environment (slipping)
    desc = env.unwrapped.desc
    total_reward = 0.0
    steps = 0
    terminated = truncated = False
    frames = [clean_render(env.render())] if record else []
    actions = []

    while steps < MAX_STEPS:
        action = policy(state, rng)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        if record:
            actions.append(ACTION_NAMES[action])
            frames.append(clean_render(env.render()))
        if terminated or truncated:
            break

    # The final state tells goal from hole (a terminal outcome on step 100 still counts).
    row, col = divmod(state, desc.shape[1])
    tile = desc[row, col].decode()
    if terminated and tile == "G":
        outcome = "goal"
    elif terminated and tile == "H":
        outcome = "hole"
    else:
        outcome = "time_limit"
    return steps, total_reward, outcome, actions, frames


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    traces = {}  # (condition, outcome) -> first trace observed

    for slippery in (False, True):
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=slippery,
                       render_mode="ansi", max_episode_steps=MAX_STEPS)
        for policy_name, policy in POLICIES.items():
            condition = f"{'slippery' if slippery else 'not_slippery'}_{policy_name}"
            for batch_seed in BATCH_SEEDS:
                for i in range(EPISODES_PER_BATCH):
                    episode_seed = 1000 * batch_seed + i
                    steps, total, outcome, actions, frames = run_episode(
                        env, policy, episode_seed, record=True)
                    rows.append({
                        "condition": condition, "slippery": slippery,
                        "batch_seed": batch_seed, "episode_seed": episode_seed,
                        "policy": policy_name, "steps": steps,
                        "total_reward": total, "outcome": outcome,
                    })
                    kind = "success" if outcome == "goal" else "failure"
                    if (condition, kind) not in traces:
                        traces[(condition, kind)] = (episode_seed, outcome, actions, frames)
        env.close()

    # 1) one row per episode
    with open(os.path.join(OUT_DIR, "part_b_episodes.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # 2) per-batch summary table
    lines = ["| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for slippery in (False, True):
        for policy_name in POLICIES:
            for batch_seed in BATCH_SEEDS:
                batch = [r for r in rows if r["slippery"] == slippery
                         and r["policy"] == policy_name and r["batch_seed"] == batch_seed]
                goals = sum(r["outcome"] == "goal" for r in batch)
                holes = sum(r["outcome"] == "hole" for r in batch)
                limits = sum(r["outcome"] == "time_limit" for r in batch)
                mean_steps = sum(r["steps"] for r in batch) / len(batch)
                lines.append(f"| {'yes' if slippery else 'no'} | {policy_name} | {batch_seed} | "
                             f"{goals} | {holes} | {limits} | {100 * goals / len(batch):.0f}% | {mean_steps:.2f} |")
    summary = "\n".join(lines)
    with open(os.path.join(OUT_DIR, "part_b_summary.md"), "w") as f:
        f.write(summary + "\n")
    print(summary)

    # 3) example traces (text rendering, [x] marks the agent)
    with open(os.path.join(OUT_DIR, "part_b_traces.txt"), "w") as f:
        for slippery in (False, True):
            for policy_name in POLICIES:
                condition = f"{'slippery' if slippery else 'not_slippery'}_{policy_name}"
                for kind in ("success", "failure"):
                    f.write(f"===== {condition} | first {kind} =====\n")
                    if (condition, kind) not in traces:
                        f.write(f"No {kind} was observed in this condition.\n\n")
                        continue
                    seed, outcome, actions, frames = traces[(condition, kind)]
                    f.write(f"episode seed {seed}, outcome {outcome}, steps {len(actions)}\n")
                    f.write("start:\n" + frames[0] + "\n")
                    for step, (a, frame) in enumerate(zip(actions, frames[1:]), 1):
                        f.write(f"step {step}: action {a}\n{frame}\n")
                    f.write("\n")


if __name__ == "__main__":
    main()
