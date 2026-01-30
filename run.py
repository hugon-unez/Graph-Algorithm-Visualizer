"""
Entry point to run graph algorithm animations without typing `manim` or the scene name.
Usage: python run.py [ -p ] [ -q l|m|h ] [ --bfs ] [ --dfs ] [ --dijkstra ] [ --bellman-ford ] [ --all ]
  -p    preview (open video when done)
  -q    quality: l (low), m (medium), h (high). Default: h
  --bfs, --dfs, --dijkstra, --bellman-ford  run only those algorithms
  --all  run all four (default if no algorithm flags given)
"""
import argparse

# All four algorithm keys; used when --all or when no algorithm flags are given
ALL_ALGORITHMS = ["bfs", "dfs", "dijkstra", "bellman_ford"]

# Quality presets: (pixel_height, pixel_width, frame_rate)
QUALITY_PRESETS = {
    "l": (480, 854, 15),   # low
    "m": (720, 1280, 30),  # medium
    "h": (1080, 1920, 60), # high
}


def parse_args(argv=None):
    """Parse command-line args. Returns dict with preview, quality, and algorithms (None = all)."""
    parser = argparse.ArgumentParser(description="Run graph algorithm animations.")
    parser.add_argument("-p", "--preview", action="store_true", help="Open video when done")
    parser.add_argument("-q", "--quality", choices=("l", "m", "h"), default="h", help="Quality: l, m, h (default: h)")
    parser.add_argument("--bfs", action="store_true", help="Run BFS demo")
    parser.add_argument("--dfs", action="store_true", help="Run DFS demo")
    parser.add_argument("--dijkstra", action="store_true", help="Run Dijkstra demo")
    parser.add_argument("--bellman-ford", action="store_true", help="Run Bellman-Ford demo")
    parser.add_argument("--all", action="store_true", dest="all_algorithms", help="Run all four algorithms")
    args = parser.parse_args(argv)

    if args.all_algorithms:
        algorithms = list(ALL_ALGORITHMS)
    elif args.bfs or args.dfs or args.dijkstra or args.bellman_ford:
        algorithms = []
        if args.bfs:
            algorithms.append("bfs")
        if args.dfs:
            algorithms.append("dfs")
        if args.dijkstra:
            algorithms.append("dijkstra")
        if args.bellman_ford:
            algorithms.append("bellman_ford")
    else:
        algorithms = None  # run all (default)

    return {
        "preview": args.preview,
        "quality": args.quality,
        "algorithms": algorithms,
    }


def _tempconfig_from_opts(opts):
    """Build Manim tempconfig dict from parse_args result."""
    ph, pw, fps = QUALITY_PRESETS[opts["quality"]]
    config = {
        "save_pngs": True,
        "write_to_movie": True,
        "disable_caching": True,
        "pixel_height": ph,
        "pixel_width": pw,
        "frame_rate": fps,
    }
    if opts.get("preview"):
        config["preview"] = True
    return config


def main():
    opts = parse_args()
    import custom_runner
    custom_runner.ALGORITHMS_TO_RUN = opts["algorithms"]

    from manim import tempconfig
    from custom_runner import MultiAlgorithmScene

    with tempconfig(_tempconfig_from_opts(opts)):
        scene = MultiAlgorithmScene()
        scene.render()


if __name__ == "__main__":
    main()
