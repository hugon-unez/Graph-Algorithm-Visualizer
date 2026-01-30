from manim import *
from animator import GraphAlgorithmAnimator
from demo_data import get_demos_to_run, ALL_ALGORITHMS

<<<<<<< Current (Your changes)
# Part 2: stub for get_demos_to_run(algorithm_keys); tests expect this API.
def get_demos_to_run(algorithm_keys):
    """Return list of demos to run for given algorithm keys. Part 2: not implemented yet."""
    raise NotImplementedError("Part 2: implement data-driven demos and filter by algorithm_keys")


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
=======
# Set by run.py before creating the scene; None means "run all four"
ALGORITHMS_TO_RUN = None
>>>>>>> Incoming (Background Agent changes)


class MultiAlgorithmScene(Scene):
    def construct(self):
        algorithms = ALGORITHMS_TO_RUN if ALGORITHMS_TO_RUN is not None else ALL_ALGORITHMS
        demos = get_demos_to_run(algorithms)

        for demo in demos:
            animator = GraphAlgorithmAnimator(self)
            animator.animate(
                demo["adjacency_list"],
                demo["algorithm"],
                **demo["kwargs"],
            )
            self.wait(2)
            if demo is not demos[-1]:
                self.clear()
