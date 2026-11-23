"""Part C: solve the paper map and the extended map, simulate the policies, run rule checks.

Usage:  python part_c.py
Writes: outputs/part_c_report.txt, outputs/part_c_episodes.csv,
        outputs/part_c_summary.md, outputs/part_c_checks.txt
"""
import csv
import os

from gridworld import ACTIONS, GridWorld, load_config

OUT_DIR = "outputs"


def section(title):
    return f"\n{'=' * 60}\n{title}\n{'=' * 60}\n"


def paper_check(world, positions):
    """V1 for every tile and the three V2 calculations, compared with Part A."""
    V0 = {s: 0.0 for s in world.states}
    V1 = world.bellman_update(V0)
    out = ["V1 (from V0 = 0):", world.value_map(V1)]
    out.append("\nV2 at the requested positions (all four actions, using only V1):")
    for pos in positions:
        s = tuple(pos)
        q = {a: world.q_value(V1, s, a) for a in ACTIONS}
        best = max(q.values())
        tied = [a for a in ACTIONS if best - q[a] <= 1e-9]
        parts = ", ".join(f"{a}={q[a]:.2f}" for a in ACTIONS)
        out.append(f"  {s} tile '{world.tile(s)}': {parts}  ->  V2 = {best:.2f}, best: {tied}")
    return "\n".join(out)


def solve(world, cfg, name):
    V, iters, delta, converged, _ = world.value_iteration(
        tol=cfg["solver"]["tolerance"], max_iter=cfg["solver"]["max_iterations"])
    policy, ties = world.extract_policy(V)
    out = [f"Map '{name}':", "\n".join("  " + "".join(r) for r in world.grid)]
    if converged:
        out.append(f"Converged after {iters} iterations; final max |change| = {delta:.3e}")
    else:
        out.append(f"NOT converged: safeguard of {iters} iterations reached; last change = {delta:.3e}")
    out += ["\nFinal values V*:", world.value_map(V),
            "\nPolicy (tile letter + arrow; ties broken by order up, right, down, left):",
            world.policy_map(policy)]
    tied = {s: t for s, t in ties.items() if len(t) > 1}
    out.append("Tied actions: " + (", ".join(f"{s}: {t}" for s, t in tied.items()) if tied else "none"))
    return V, policy, "\n".join(out)


def simulate(world, policy, cfg, map_name, rows):
    sim = cfg["simulation"]
    for batch_seed in sim["batch_seeds"]:
        for i in range(sim["episodes_per_batch"]):
            seed = 1000 * batch_seed + i
            steps, total, disc, outcome = world.run_episode(policy, seed, sim["max_steps"])
            rows.append({"map": map_name, "batch_seed": batch_seed, "episode_seed": seed,
                         "steps": steps, "total_reward": round(total, 6),
                         "discounted_return": round(disc, 6), "outcome": outcome})


def rule_checks(cfg):
    """Evidence that the model follows the rules before we trust the solver."""
    paper = GridWorld(cfg["paper_map"], cfg["tiles"], cfg["discount"])
    ext = GridWorld(cfg["extended_map"], cfg["tiles"], cfg["discount"])
    out = []

    def show(world, s, a):
        t = world.transitions(s, a)
        out.append(f"  from {s} '{world.tile(s)}' action {a:5}: " +
                   "; ".join(f"p={p} -> {s2} '{world.tile(s2)}' r={r} done={d}" for p, s2, r, d in t))
        return t

    out.append("1) Walls and boundaries (blocked move = stay, reward of the tile you stay on):")
    t = show(paper, (1, 1), "up");     assert t[0][1] == (1, 1) and t[0][2] == -1
    t = show(paper, (1, 2), "down");   assert t[0][1] == (1, 2) and t[0][2] == -3
    t = show(paper, (3, 2), "up");     assert t[0][1] == (3, 2)
    t = show(paper, (3, 4), "right");  assert t[0][1] == (3, 4)
    t = show(paper, (2, 1), "right");  assert t[0][1] == (2, 1)

    out.append("\n2) Ice probabilities (each action, outcomes resolved against walls):")
    for a in ACTIONS:
        t = show(paper, (2, 3), a)
        total = sum(p for p, *_ in t)
        out.append(f"     sum of probabilities = {total}")
        assert abs(total - 1.0) < 1e-12
    out.append("   Non-ice tiles are deterministic:")
    for s in [(3, 1), (1, 2), (3, 2), (3, 4)]:
        t = show(ext, s, "left"); assert len(t) == 1 and t[0][0] == 1.0

    out.append("\n3) Floor rewards on entering (extended map):")
    t = show(ext, (1, 1), "right");  assert t[0][2] == -3, "mud"
    t = show(ext, (3, 1), "right");  assert t[0][2] == -2, "sand"
    t = show(ext, (3, 3), "right");  assert t[0][2] == -5, "water"
    t = show(ext, (1, 3), "right");  assert t[0][2] == 10, "goal"
    t = show(ext, (3, 4), "up");     assert t[0][2] == -10, "hole"

    out.append("\n4) Termination: G and H give no further reward and keep value 0:")
    for s in [(1, 4), (2, 4)]:
        for a in ACTIONS:
            t = paper.transitions(s, a)
            assert t == [(1.0, s, 0.0, True)]
        out.append(f"  {s} '{paper.tile(s)}': every action -> {paper.transitions(s, 'up')}")
    V, *_ = paper.value_iteration()
    out.append(f"  V(G) = {V[(1, 4)]}, V(H) = {V[(2, 4)]}")
    assert V[(1, 4)] == 0 and V[(2, 4)] == 0
    out.append("  The wall (2,2) is not a state: " + str((2, 2) not in paper.states))
    assert (2, 2) not in paper.states

    out.append("\nAll checks passed.")
    return "\n".join(out)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = load_config()
    paper = GridWorld(cfg["paper_map"], cfg["tiles"], cfg["discount"])
    ext = GridWorld(cfg["extended_map"], cfg["tiles"], cfg["discount"])

    report = [section("1. Paper map: compare with Part A"),
              paper_check(paper, cfg["paper_v2_positions"])]

    report.append(section("2. Paper map: value iteration until convergence"))
    V_p, pol_p, text = solve(paper, cfg, "paper")
    report.append(text)

    report.append(section("3. Extended map (D = sand -2, W = water -5)"))
    V_e, pol_e, text = solve(ext, cfg, "extended")
    report.append(text)

    report.append(section("4. Differences extended - paper"))
    lines = []
    for s in paper.states:
        if paper.is_terminal(s):
            continue
        changed = " (action changed)" if pol_p[s] != pol_e[s] else ""
        lines.append(f"  {s} '{paper.tile(s)}'->'{ext.tile(s)}': V {V_p[s]:8.3f} -> {V_e[s]:8.3f} "
                     f"(diff {V_e[s] - V_p[s]:+.3f}); action {pol_p[s]} -> {pol_e[s]}{changed}")
    report.append("\n".join(lines))
    report.append(f"\nV(start) paper = {V_p[paper.start]:.4f}, extended = {V_e[ext.start]:.4f}")

    # simulations
    rows = []
    simulate(paper, pol_p, cfg, "paper", rows)
    simulate(ext, pol_e, cfg, "extended", rows)
    with open(os.path.join(OUT_DIR, "part_c_episodes.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    table = ["| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, V, world in (("paper", V_p, paper), ("extended", V_e, ext)):
        for b in cfg["simulation"]["batch_seeds"]:
            batch = [r for r in rows if r["map"] == name and r["batch_seed"] == b]
            n = len(batch)
            table.append(f"| {name} | {b} | {sum(r['outcome'] == 'goal' for r in batch)} | "
                         f"{sum(r['outcome'] == 'hole' for r in batch)} | "
                         f"{sum(r['outcome'] == 'time_limit' for r in batch)} | "
                         f"{sum(r['steps'] for r in batch) / n:.2f} | "
                         f"{sum(r['discounted_return'] for r in batch) / n:.4f} |")
    table_text = "\n".join(table)
    table_text += (f"\n\nComputed V(start): paper = {V_p[paper.start]:.4f}, "
                   f"extended = {V_e[ext.start]:.4f}\n")
    with open(os.path.join(OUT_DIR, "part_c_summary.md"), "w") as f:
        f.write(table_text)
    report.append(section("5. Simulation of the computed policies") + table_text)

    checks = rule_checks(cfg)
    with open(os.path.join(OUT_DIR, "part_c_checks.txt"), "w") as f:
        f.write(checks + "\n")
    report.append(section("6. Rule checks") + checks)

    full = "\n".join(report)
    with open(os.path.join(OUT_DIR, "part_c_report.txt"), "w") as f:
        f.write(full + "\n")
    print(full)


if __name__ == "__main__":
    main()
