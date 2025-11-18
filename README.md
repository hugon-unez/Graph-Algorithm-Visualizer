# Graph Algorithm Visualizer (built on top of manim)

This project is a manim-based visualizer for shortest path graph algorithms.  
It takes in an adjacency list, draws a graph, runs an algorithm step by step, and animates both:

- the graph itself (visited/discovered nodes, tree edges, relaxations), and  
- an accompanying distance/parent table for shortest-path algorithms for Dijkstra/Bellman Ford

Currently supported algorithms:

- **BFS**
- **DFS**
- **Dijkstra**
- **Bellman–Ford**

---

## Setup

1. **Create and activate a virtualenv (unless you have python 3.11):**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
2. **Use simple_runner.py boiler plate code to visualize an algorithm on your input graph**