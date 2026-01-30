"""
Data-driven demo definitions and filter. No Manim imports so tests can run without manim.
"""

ALL_ALGORITHMS = ["bfs", "dfs", "dijkstra", "bellman_ford"]

TREE_ADJ = {
    0: [1, 2],
    1: [0, 3, 4],
    2: [0, 5],
    3: [1, 6],
    4: [1],
    5: [2],
    6: [3],
}
DIJKSTRA_ADJ = {
    0: {1: 4, 2: 1},
    1: {0: 4, 2: 2, 3: 1, 4: 7},
    2: {0: 1, 1: 2, 3: 5, 5: 8},
    3: {1: 1, 2: 5, 4: 3, 5: 2},
    4: {1: 7, 3: 3, 5: 1},
    5: {2: 8, 3: 2, 4: 1},
}
BF_ADJ = {
    0: {1: 4, 2: 5},
    1: {2: -2, 3: 6},
    2: {3: 1},
    3: {},
}

DEMO_ITEMS = [
    {"adjacency_list": TREE_ADJ, "algorithm": "bfs", "kwargs": {"start": 0, "directed": False, "weighted": False}},
    {"adjacency_list": TREE_ADJ, "algorithm": "dfs", "kwargs": {"start": 0, "directed": False, "weighted": False}},
    {"adjacency_list": DIJKSTRA_ADJ, "algorithm": "dijkstra", "kwargs": {"start": 0, "directed": False, "weighted": True}},
    {"adjacency_list": BF_ADJ, "algorithm": "bellman_ford", "kwargs": {"start": 0, "directed": True, "weighted": True}},
]


def get_demos_to_run(algorithm_keys):
    """Return list of demos whose algorithm is in algorithm_keys (order preserved from DEMO_ITEMS)."""
    key_set = set(algorithm_keys)
    return [d for d in DEMO_ITEMS if d["algorithm"] in key_set]
