# Graph Algorithm Visualizer

A [Manim](https://www.manim.community/)-based visualizer for graph traversal and shortest-path algorithms. It turns an adjacency list and an algorithm choice into a step-by-step animation: the graph is drawn, nodes and edges are highlighted as the algorithm runs, and (for Dijkstra and Bellman–Ford) a distance/parent table is updated alongside the graph.

**Supported algorithms:** BFS, DFS, Dijkstra, Bellman–Ford.

Animations include on-screen captions (e.g. “Visit A”, “Relax B → C: dist(C) = 5”) and, for shortest-path algorithms, a live distance table. You can run built-in demos or plug in your own graphs.

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
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

3. **Install Manim** (if not already installed):

   ```bash
   pip install manim
   ```

4. **Check that it runs** by generating a single demo:

   ```bash
   python run.py --bfs -q l
   ```

   The first run may take a minute. When it finishes, you’ll get a video (and optional PNG frames) in a `media` folder inside the project.

---

## How to visualize

You can either run the **built-in demos** (quick, no code changes) or **visualize your own graph** (edit a small script).

### Option 1: Built-in demos (`run.py`)

Use this when you want to see BFS, DFS, Dijkstra, or Bellman–Ford on the included example graphs without editing code.

**Run all four algorithms** (each on its demo graph, one after another):

```bash
python run.py
```

**Run only specific algorithms:**

```bash
python run.py --bfs                    # BFS only
python run.py --dijkstra               # Dijkstra only
python run.py --bfs --dfs              # BFS then DFS
python run.py --bellman-ford           # Bellman–Ford only
python run.py --all                    # Same as no flags: all four
```

**Control quality and preview:**

- **Quality** affects resolution and frame rate. Use `-q` with `l`, `m`, or `h`:
  - `l` — low (480p, 15 fps), faster to render
  - `m` — medium (720p, 30 fps)
  - `h` — high (1080p, 60 fps), default

  Examples:

  ```bash
  python run.py -q l              # Low quality, all algorithms
  python run.py --dijkstra -q m   # Dijkstra at medium quality
  ```

- **Preview** opens the rendered video when done. Use `-p`:

  ```bash
  python run.py -p                # Render all, then open video
  python run.py --bfs -q h -p     # BFS at high quality, then open video
  ```

**Full help:**

```bash
python run.py --help
```

**Where the output goes:** Manim writes videos (and optional frames) into a `media` directory in the project. Paths look like `media/videos/<script_name>/<quality>/` (e.g. for `run.py` at default quality, the video is under `media/videos/run/1080p60/`).

---

### Option 2: Your own graph (`simple_runner.py`)

Use this when you want to animate **your** graph with a chosen algorithm.

1. **Open `simple_runner.py`** in an editor.

2. **Define your graph as an adjacency list.** Vertices are integers (0, 1, 2, …); they will be labeled A, B, C, … in the animation.

   - **Unweighted graph** — each key is a vertex, value is a list of neighbors:

     ```python
     adj_list = {
         0: [1, 2],      # 0 is adjacent to 1 and 2
         1: [0, 3],
         2: [0, 3],
         3: [1, 2],
     }
     ```

   - **Weighted graph** — each key is a vertex, value is a dict `neighbor: weight`:

     ```python
     adj_list = {
         0: {1: 4, 2: 1},   # 0→1 has weight 4, 0→2 has weight 1
         1: {0: 4, 2: 2, 3: 1},
         2: {0: 1, 1: 2, 3: 5},
         3: {1: 1, 2: 5},
     }
     ```

3. **Call `animate_algorithm`** with your adjacency list, the algorithm name, and the start vertex:

   ```python
   animate_algorithm(
       adj_list,
       "dijkstra",    # or "bfs", "dfs", "bellman_ford"
       start=0,
       weighted=True,   # True if you used weights
       directed=False,  # True for directed edges
   )
   ```

4. **Run the script:**

   ```bash
   python simple_runner.py
   ```

   The video (and optional PNGs) will appear under `media/`, same as with `run.py`. You can change the adjacency list and the `animate_algorithm(...)` arguments and re-run to try different graphs or algorithms.

---

## What the animations show

- **Graph:** Vertices (labeled A, B, C, …), edges, and (for weighted graphs) edge weights.
- **Start vertex:** Highlighted at the beginning (e.g. red, then the highlight fades).
- **Algorithm progress:** Nodes and edges change color as they are visited, discovered, or relaxed (e.g. yellow for in-queue, green for tree edges, blue for finalized).
- **Captions:** Short text at the bottom (e.g. “Visit A”, “Discover B from A”, “Relax A → B: dist(B) = 5”) describing each step.
- **Distance table:** For Dijkstra and Bellman–Ford, a table next to the graph shows distance and parent for each vertex and updates as the algorithm runs.

Layout (placement of nodes) is chosen automatically (e.g. tree layout for tree-like graphs, circular or force-directed for others). You do not need to specify positions.

---

## Project layout

| File | Purpose |
|------|--------|
| `run.py` | Entry point for built-in demos; parses CLI (e.g. `--bfs`, `-q`, `-p`) and runs the chosen algorithms. |
| `simple_runner.py` | Boilerplate to animate **your** graph: edit the adjacency list and `animate_algorithm(...)` call, then run. |
| `custom_runner.py` | Manim scene that runs one or more built-in demos; used by `run.py`. |
| `animator.py` | Core animation logic: builds the graph, runs the algorithm step-by-step, drives captions and distance table. |
| `graph.py` | Graph representation and algorithm generators (BFS, DFS, Dijkstra, Bellman–Ford) that yield events for the animator. |
| `demo_data.py` | Definitions of the built-in demo graphs and which algorithms run when you use `run.py`. |
| `test_run_cli.py` | Tests for CLI parsing and demo selection. Run with `python test_run_cli.py` or `pytest test_run_cli.py -v`. |
| `test_graph_algos.py` | Tests for the graph algorithms (no Manim required). |

---

## Troubleshooting

- **“No module named 'manim'”** — Install Manim in the same environment you use to run `python run.py` or `python simple_runner.py` (e.g. `pip install manim` inside your `.venv`).
- **Rendering is slow** — Use low quality for quicker runs: `python run.py -q l` or, in `simple_runner.py`, you’d need to run via a config that sets lower resolution/fps (the script uses fixed settings; for full control you can run a custom Manim scene with your own `tempconfig`).
- **Video doesn’t open** — Use `-p` to open the video when done: `python run.py -p`. If it still doesn’t open, find the file under `media/videos/...` and play it manually.
- **Wrong graph or algorithm** — For built-in demos, double-check the flags (`--bfs`, `--dijkstra`, etc.). For your own graph, check the adjacency list and the `animate_algorithm(..., "algorithm_name", ...)` and `weighted`/`directed` arguments in `simple_runner.py`.

---

## Summary

1. Install Python and Manim, set up a venv, and run `python run.py --bfs -q l` to confirm everything works.
2. Use **`python run.py`** (with optional `--bfs`, `--dfs`, `--dijkstra`, `--bellman-ford`, `-q l|m|h`, `-p`) to visualize the built-in demos.
3. Use **`simple_runner.py`** to define your own adjacency list and animate it with BFS, DFS, Dijkstra, or Bellman–Ford.
4. Output appears in the `media/` folder; use `-p` to open the video when rendering finishes.
