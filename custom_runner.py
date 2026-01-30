from manim import *
from animator import GraphAlgorithmAnimator
from demo_data import get_demos_to_run, ALL_ALGORITHMS

# Set by run.py before creating the scene; None means "run all four"
ALGORITHMS_TO_RUN = None


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
