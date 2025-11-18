from manim import *
from animator import GraphAlgorithmAnimator

class MultiAlgorithmScene(Scene):
    def construct(self):
        animator = GraphAlgorithmAnimator(self)
        adj_list = {0: [1, 2], 1: [0, 3], 2: [0, 3], 3: [1, 2]}
        
        animator.animate(adj_list, 'bfs', start=0, layout='circular')
        self.wait(2)
        self.clear()  # Clear the scene
        
        animator.animate(adj_list, 'dfs', start=0, layout='circular')