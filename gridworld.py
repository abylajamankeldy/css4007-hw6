"""Grid World MDP for Part C: model, value iteration, policy extraction, simulation.

Floor rules (rewards, slipping, terminal tiles) are read from data/maps.json,
so a new floor type only needs a new entry there - the solver is not changed.
"""
import json

import numpy as np

ACTIONS = ["up", "right", "down", "left"]          # also the tie-breaking order
MOVES = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
PERPENDICULAR = {"up": ("left", "right"), "down": ("left", "right"),
                 "left": ("up", "down"), "right": ("up", "down")}
ARROWS = {"up": "^", "right": ">", "down": "v", "left": "<"}


def load_config(path="data/maps.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class GridWorld:
    def __init__(self, rows, tiles, discount=0.9):
        self.grid = [list(r) for r in rows]
        self.tiles = tiles                      # floor rules, kept outside the solver
        self.gamma = discount
        self.n_rows, self.n_cols = len(self.grid), len(self.grid[0])
        # A state is a (row, col) pair, 1-based like the assignment. Walls are not states.
        self.states = [(r + 1, c + 1) for r in range(self.n_rows) for c in range(self.n_cols)
                       if self.tiles[self.grid[r][c]].get("traversable", True)]
        self.start = next(s for s in self.states if self.tile(s) == "S")

    def tile(self, s):
        return self.grid[s[0] - 1][s[1] - 1]

    def is_terminal(self, s):
        return self.tiles[self.tile(s)].get("terminal", False)

    def reward(self, s):
        """Reward for a move that ENDS on tile s."""
        return self.tiles[self.tile(s)]["reward"]

    def move(self, s, direction):
        """Deterministic one-cell move; a wall or the edge leaves you in place."""
        dr, dc = MOVES[direction]
        r, c = s[0] + dr, s[1] + dc
        if not (1 <= r <= self.n_rows and 1 <= c <= self.n_cols):
            return s
        if not self.tiles[self.grid[r - 1][c - 1]].get("traversable", True):
            return s
        return (r, c)

    def direction_probs(self, s, action):
        """Which directions actually happen, decided by the tile we START on."""
        rule = self.tiles[self.tile(s)]
        if rule.get("movement") == "slippery":
            side_a, side_b = PERPENDICULAR[action]
            p_side = rule["perpendicular_each"]
            return [(rule["intended"], action), (p_side, side_a), (p_side, side_b)]
        return [(1.0, action)]

    def transitions(self, s, action):
        """List of (probability, next_state, reward, done) - the known model P and R."""
        if self.is_terminal(s):
            return [(1.0, s, 0.0, True)]          # no rewards after termination
        result = []
        for p, direction in self.direction_probs(s, action):
            s2 = self.move(s, direction)          # blocked moves resolved per direction
            result.append((p, s2, self.reward(s2), self.is_terminal(s2)))
        return result

    # ---------- value iteration ----------

    def q_value(self, V, s, action):
        """sum over s' of P(s'|s,a) * (R + gamma * V(s')); terminal V is always 0."""
        return sum(p * (r + self.gamma * (0.0 if done else V[s2]))
                   for p, s2, r, done in self.transitions(s, action))

    def bellman_update(self, V):
        """One full round: every state uses only the previous round's values V."""
        new_V = {}
        for s in self.states:
            if self.is_terminal(s):
                new_V[s] = 0.0
            else:
                new_V[s] = max(self.q_value(V, s, a) for a in ACTIONS)
        return new_V

    def value_iteration(self, tol=1e-6, max_iter=10000, keep_history=0):
        V = {s: 0.0 for s in self.states}          # V0 = 0
        history = [V]
        for k in range(1, max_iter + 1):
            new_V = self.bellman_update(V)
            delta = max(abs(new_V[s] - V[s]) for s in self.states)
            V = new_V
            if k <= keep_history:
                history.append(V)
            if delta < tol:
                return V, k, delta, True, history
        return V, max_iter, delta, False, history

    def extract_policy(self, V, tie_tol=1e-9):
        """Best action(s) per state by comparing Q-values with the final V.
        Ties (within tie_tol) are all listed; the first in ACTIONS order is used."""
        policy, ties = {}, {}
        for s in self.states:
            if self.is_terminal(s):
                continue
            q = {a: self.q_value(V, s, a) for a in ACTIONS}
            best = max(q.values())
            tied = [a for a in ACTIONS if best - q[a] <= tie_tol]
            policy[s], ties[s] = tied[0], tied
        return policy, ties

    # ---------- printing ----------

    def value_map(self, V, fmt="{:7.3f}"):
        lines = []
        for r in range(1, self.n_rows + 1):
            cells = []
            for c in range(1, self.n_cols + 1):
                s, t = (r, c), self.grid[r - 1][c - 1]
                if s not in V:
                    cells.append(f"{'#####':>7}")
                elif self.is_terminal(s):
                    cells.append(f"{t + ' 0':>7}")
                else:
                    cells.append(fmt.format(V[s]))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def policy_map(self, policy):
        lines = []
        for r in range(1, self.n_rows + 1):
            cells = []
            for c in range(1, self.n_cols + 1):
                s, t = (r, c), self.grid[r - 1][c - 1]
                cells.append(f"{t}{ARROWS[policy[s]]}" if s in policy else f"{t} ")
            lines.append(" ".join(cells))
        return "\n".join(lines)

    # ---------- simulation ----------

    def sample_step(self, s, action, rng):
        """Sample one outcome from the same transition model the solver uses."""
        outcomes = self.transitions(s, action)
        i = rng.choice(len(outcomes), p=[o[0] for o in outcomes])
        _, s2, r, done = outcomes[i]
        return s2, r, done

    def run_episode(self, policy, seed, max_steps=100):
        rng = np.random.default_rng(seed)
        s, steps, total, discounted = self.start, 0, 0.0, 0.0
        done = False
        while steps < max_steps:
            s, r, done = self.sample_step(s, policy[s], rng)
            discounted += (self.gamma ** steps) * r     # r0 + 0.9*r1 + 0.9^2*r2 + ...
            total += r
            steps += 1
            if done:
                break
        if done:
            outcome = "goal" if self.tile(s) == "G" else "hole"
        else:
            outcome = "time_limit"
        return steps, total, discounted, outcome
