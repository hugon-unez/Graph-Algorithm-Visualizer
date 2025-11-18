from manim import *
from animator import GraphAlgorithmAnimator

class MultiAlgorithmScene(Scene):
    def construct(self):
        # 1. BFS on a tree
        tree_adj = {
            0: [1, 2],
            1: [0, 3, 4],
            2: [0, 5],
            3: [1, 6],
            4: [1],
            5: [2],
            6: [3],
        }


        bfs_animator = GraphAlgorithmAnimator(self)
        bfs_animator.animate(
            tree_adj,
            "bfs",
            start=0,
            directed=False,
            weighted=False,
        )
        self.wait(2)
        self.clear()

        # 2. DFS on the same tree
        dfs_animator = GraphAlgorithmAnimator(self)
        dfs_animator.animate(
            tree_adj,
            "dfs",
            start=0,
            directed=False,
            weighted=False,
        )
        self.wait(2)
        self.clear()
        
        
        # 3. Dijkstra on a weighted graph
        dijkstra_adj = {
            0: {1: 4, 2: 1},
            1: {0: 4, 2: 2, 3: 1, 4: 7},
            2: {0: 1, 1: 2, 3: 5, 5: 8},
            3: {1: 1, 2: 5, 4: 3, 5: 2},
            4: {1: 7, 3: 3, 5: 1},
            5: {2: 8, 3: 2, 4: 1},
        }

        dijkstra_animator = GraphAlgorithmAnimator(self)
        dijkstra_animator.animate(
            dijkstra_adj,
            "dijkstra",
            start=0,
            directed=False,
            weighted=True,
        )
        self.wait(2)
        self.clear()

        # 4. Bellman–Ford on a graph with a negative edge
        bf_adj = {
            0: {1: 4, 2: 5},
            1: {2: -2, 3: 6},
            2: {3: 1},
            3: {},
        }

        bf_animator = GraphAlgorithmAnimator(self)
        bf_animator.animate(
            bf_adj,
            "bellman_ford",
            start=0,
            directed=True,
            weighted=True,
        )
        self.wait(2)
