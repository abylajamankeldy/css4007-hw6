# HW6 submission

## AI-use declaration

- **Claude (Anthropic)** — used to check my Part A arithmetic after I did it on paper, to help write the Python code for Parts B and C, and to help phrase the explanations in this file. I ran all programs myself, compared the solver output with my paper values and checked the saved results in `outputs/`.
- No other AI tools were used.

## Setup and reproduction

Versions used: Python 3.x (see `python --version`), `gymnasium==1.4.0`, `numpy==2.5.5` (pinned in `requirements.txt`).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python part_b_frozenlake.py         # Part B -> outputs/part_b_*
python part_c.py                    # Part C -> outputs/part_c_*
```

No API key is needed. All runs are seeded, so the commands reproduce the same files.

| File | Purpose |
| --- | --- |
| `part_b_frozenlake.py` | Part B episode runner, two policies, four conditions, summary and traces |
| `gridworld.py` | Part C model (transitions/rewards from `data/maps.json`), value iteration, policy extraction, simulator |
| `part_c.py` | Part C: paper comparison, both maps, simulations, rule checks |
| `data/maps.json` | Maps and floor rules (rewards, slipping, terminals) — kept separate from the solver |

## Part A — Paper calculations

Photos: [`part-a/`](part-a/) — pages 1–4 (committed before any code).

Summary: V1 = −1 on every nonterminal tile except (1,3) = 10. V2(1,2) = 8 (→), V2(1,3) = 10 (→), V2(2,3) = 0.8·8 + 0.1·(−1.9) + 0.1·(−10) = 5.21 (↑). No ties at these three tiles. Two rounds are not enough: V_k is only the best value with k moves left, so the +10 has only propagated two steps (e.g. V2 at the start is −1.9 while its converged value is +1.54), and the values the arrows depend on keep changing until convergence.

## Part B — FrozenLake

### Fixed policy, written before the experiments

Map states are numbered `row*4 + col`:

```text
S F F F     0  1  2  3
F H F H     4  5  6  7
F F F H     8  9 10 11
H F F G    12 13 14 15
```

| State | 0 | 1 | 2 | 3 | 4 | 6 | 8 | 9 | 10 | 13 | 14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Action | down | left | down | left | down | down | right | down | down | right | right |

Reasoning: follow the shortest hole-free path 0 → 4 → 8 → 9 → 13 → 14 → 15 (6 moves). From tiles off that path, move back towards it or use the second safe route 2 → 6 → 10 → 14, and never choose a move whose intended destination is a hole. The policy is designed from the map as if moves were exact; it was not tuned for slipping. It is the same in all experiments and depends only on the current state.

### Results

| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no | random | 11 | 0 | 50 | 0 | 0% | 7.98 |
| no | random | 22 | 0 | 50 | 0 | 0% | 6.42 |
| no | random | 33 | 1 | 49 | 0 | 2% | 6.66 |
| no | fixed | 11 | 50 | 0 | 0 | 100% | 6.00 |
| no | fixed | 22 | 50 | 0 | 0 | 100% | 6.00 |
| no | fixed | 33 | 50 | 0 | 0 | 100% | 6.00 |
| yes | random | 11 | 0 | 50 | 0 | 0% | 8.30 |
| yes | random | 22 | 2 | 48 | 0 | 4% | 7.76 |
| yes | random | 33 | 0 | 50 | 0 | 0% | 6.74 |
| yes | fixed | 11 | 1 | 49 | 0 | 2% | 5.22 |
| yes | fixed | 22 | 2 | 48 | 0 | 4% | 6.00 |
| yes | fixed | 33 | 1 | 49 | 0 | 2% | 5.14 |

Files: [`outputs/part_b_episodes.csv`](outputs/part_b_episodes.csv) (600 rows, one per episode), [`outputs/part_b_summary.md`](outputs/part_b_summary.md), [`outputs/part_b_traces.txt`](outputs/part_b_traces.txt) (text-rendered traces; `[x]` marks the agent).

**What slipping changed.** Without slipping the fixed policy reaches the goal in exactly 6 steps in all 150 episodes: the environment is deterministic, so every seed gives the same path. With `is_slippery=True` the intended move happens only with probability 1/3 (the other 2/3 go to the two perpendicular directions), so the plan is broken almost every time and the agent drifts into a hole — only 4/150 goals. The random policy is bad in both settings (1/150 and 2/150 goals) because the map has four holes close to the path and a random walk usually falls into one within ~7 steps. No episode reached the 100-step time limit in any condition, because every condition ends in a hole long before that.

**Consistency across batches.** Deterministic conditions are identical across batches (fixed: 50/50 every time). Slippery and random conditions vary a little (0–2 goals per batch) because each episode seed gives different slips/actions; the batches agree that success is close to 0–4%.

**Why one success is not evidence.** The outcome is random. In the slippery fixed condition, episode 11020 reached the goal (trace saved), yet 146 of 150 episodes failed. A single episode shows what *can* happen, not how often it happens; we need many seeded episodes, and several batches, to estimate a success rate and see its variation.

**Traces.** Saved for every condition: e.g. slippery fixed success (seed 11020, 14 steps) and failure (seed 11000, hole after 8 steps). The non-slippery fixed policy never failed, so no failure trace exists for that condition (stated in the file).

**Why this is not training.** Both policies are fixed before running: nothing is updated from rewards, no values are learned, and episode 150 uses exactly the same rule as episode 1. We only *evaluate* the policies. Training would mean changing the policy or value estimates based on the observed rewards.

## Part C — Custom solver

### Compare code with paper

Output of `python part_c.py` (section 1 of [`outputs/part_c_report.txt`](outputs/part_c_report.txt)):

```text
V1 (from V0 = 0):
|  -1.000 |  -1.000 |  10.000 |     G 0 |
|  -1.000 |   ##### |  -1.000 |     H 0 |
|  -1.000 |  -1.000 |  -1.000 |  -1.000 |

(1, 2) tile 'M': up=-3.90, right=8.00, down=-3.90, left=-1.90  ->  V2 = 8.00, best: ['right']
(1, 3) tile '.': up=8.00, right=10.00, down=-1.90, left=-3.90  ->  V2 = 10.00, best: ['right']
(2, 3) tile 'I': up=5.21, right=-7.39, down=-2.71, left=-0.91  ->  V2 = 5.21, best: ['up']
```

All values match my paper calculations; no discrepancies in V1, V2 or the arrows.

### Values, policies and convergence

Full maps: [`outputs/part_c_report.txt`](outputs/part_c_report.txt), sections 2–4.

| Map | Iterations | Final max change | Converged |
| --- | --- | --- | --- |
| paper `.M.G / .#IH / S...` | 12 | 2.17e-07 | yes |
| extended `.M.G / .#IH / SD.W` | 11 | 2.41e-07 | yes |

```text
Paper map V*                                   Paper policy
|   4.200 |   8.000 |  10.000 |     G 0 |      .> M> .> G
|   2.780 |   ##### |   5.824 |     H 0 |      .^ #  I^ H
|   1.536 |   2.818 |   4.242 |   2.818 |      S> .> .^ .<

Extended map V*                                Extended policy
|   4.200 |   8.000 |  10.000 |     G 0 |      .> M> .> G
|   2.780 |   ##### |   5.824 |     H 0 |      .^ #  I^ H
|   1.502 |   2.818 |   4.242 |   2.818 |      S^ D> .^ W<
```

**Tie-breaking.** After convergence I compute all four Q-values with the final V and take the maximum. Actions within 1e-9 of the maximum count as tied; the first in the fixed order up, right, down, left is chosen, and all tied actions are printed. With the final values there were **no ties** on either map.

### Simulation results

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
| --- | --- | --- | --- | --- | --- | --- |
| paper | 11 | 47 | 3 | 0 | 5.04 | 2.2297 |
| paper | 22 | 44 | 6 | 0 | 4.96 | 1.4424 |
| paper | 33 | 42 | 8 | 0 | 4.88 | 0.9700 |
| extended | 11 | 50 | 0 | 0 | 5.00 | 1.5020 |
| extended | 22 | 50 | 0 | 0 | 5.00 | 1.5020 |
| extended | 33 | 50 | 0 | 0 | 5.00 | 1.5020 |

Computed V(start): paper = 1.5358, extended = 1.5020. Episodes: [`outputs/part_c_episodes.csv`](outputs/part_c_episodes.csv); same seed scheme as Part B (`1000 × batch_seed + i`), 100-step limit.

### Explain your results

**What changed with sand and water.** Only the start action changed. On the paper map the start goes **right** through the bottom row and the ice: S → (3,2) → (3,3) → I → (1,3) → G. On the extended map (3,2) is sand, so the first step costs −2 instead of −1. Going right on the paper map is worth −1 + 0.9·2.818 = 1.536; on the extended map it would be −2 + 0.9·2.818 = 0.536, while going **up** the safe left column through mud gives −1 + 0.9·2.780 = 1.502, so the policy switches to the mud route S → (2,1) → (1,1) → M → (1,3) → G. All other values are unchanged: a tile's value depends on rewards of tiles *after* it, not on the cost of entering it, so V(sand) = V(3,2) and V(water) = V(3,4) stay 2.818. Water at (3,4) mattered for nobody: the policy never enters it ((3,3) goes up to the ice, not right).

**Cost versus risk, and the ice.** On the paper map the ice route is risky — from the ice "up" goes to the hole with probability 0.1 and stays on ice with probability 0.1, so the chance of eventually falling is 0.1/0.9 ≈ 11% — but it is short and cheap (V = 1.536). The mud route is safe but costs −3 on the mud (V = 1.502). The expected values are close and the risky route wins by 0.034. Adding sand to the risky route (−1 more) flips the decision. At the ice the policy always chooses **up**: "right" would aim straight at the hole, while "up" has the best Q-value (0.8 chance to reach (1,3), worth 8 at V2 and 10 at convergence).

**Simulations versus computed values.** V(s) is an expected discounted return over an infinite horizon; a simulation gives a finite sample of returns. On the paper map individual episodes give 3.122 (straight to goal), 1.810 (one slip that stays on ice) or −10.0 (hole) — never 1.536 itself. The batch means vary (0.97 to 2.23) because 3–8 holes per batch change the mean a lot; the mean over all 150 episodes is 1.547, close to 1.536, and 17/150 = 11.3% holes matches the ≈11% predicted by the model. On the extended map the chosen route never touches ice, so every episode is identical and the mean return equals V(start) = 1.502 exactly. The 100-step limit cuts off an infinite horizon, but here every episode ends in at most 6 steps, so the truncation has no effect; in general a finite run, finite sample and step limit mean simulations only approximate the values.

### Rule checks

Saved in [`outputs/part_c_checks.txt`](outputs/part_c_checks.txt) (the script stops with an error if any check fails):

1. **Walls/boundaries:** e.g. from (1,1) up → stays at (1,1), reward −1; from mud (1,2) down into the wall → stays on mud, reward −3; from (3,2) up into the wall → stays.
2. **Ice probabilities:** for every action from (2,3) the outcomes are 0.8 + 0.1 + 0.1 = 1.0, each resolved against walls (e.g. "up": 0.8 → (1,3), 0.1 → wall so stay on I, 0.1 → H). Non-ice tiles (S, M, D, W) have one outcome with probability 1.
3. **Floor rewards on entering:** mud −3, sand −2, water −5, goal +10, hole −10.
4. **Termination:** G and H return `(1.0, same state, reward 0, done)` for every action, and their values stay 0. The wall (2,2) is not a state.
