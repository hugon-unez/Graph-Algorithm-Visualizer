# Graph Algorithm Visualizer

A [Manim](https://www.manim.community/)-based visualizer for graph traversal and shortest-path algorithms. It turns an adjacency list and an algorithm choice into a step-by-step animation: the graph is drawn, nodes and edges are highlighted as the algorithm runs, and (for Dijkstra and Bellman–Ford) a distance/parent table is updated alongside the graph.

**Supported algorithms:** BFS, DFS, Dijkstra, Bellman–Ford.

Before the spring, I plan to add subtitles explaining each event.

Currently supported algorithms:

---

## What you need

- **Python 3** (3.11 recommended)
- **Manim Community Edition**

```bash
pip install manim
```

(Use a virtual environment so the project’s dependencies don’t conflict with the rest of your system.)

---

## Setup

1. **Clone or download this project** and open a terminal in the project folder.

2. **Create and activate a virtual environment** (recommended):

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
2. **Use simple_runner.py boiler plate code to visualize an algorithm on your input graph**
