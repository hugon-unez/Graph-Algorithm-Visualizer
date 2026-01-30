"""
Part 2 TDD: tests for CLI parsing and demo filtering.
These tests are written first; they will fail until Part 2 implementation exists.
Run with: pytest test_run_cli.py -v   or   python test_run_cli.py
"""

# Expected "all four" algorithm keys for assertions
ALL_ALGORITHMS = ["bfs", "dfs", "dijkstra", "bellman_ford"]


# ---- CLI parsing tests (run.parse_args) ----

def test_parse_args_bfs_dijkstra_returns_two_algorithms():
    """Given --bfs --dijkstra, parsed 'algorithms' list is ['bfs', 'dijkstra']."""
    from run import parse_args
    opts = parse_args(["--bfs", "--dijkstra"])
    assert opts["algorithms"] == ["bfs", "dijkstra"]


def test_parse_args_all_returns_all_four():
    """Given --all, parsed 'algorithms' list is all four algorithms."""
    from run import parse_args
    opts = parse_args(["--all"])
    assert opts["algorithms"] == ALL_ALGORITHMS


def test_parse_args_no_flags_returns_all():
    """Given no algorithm flags, 'algorithms' means all (None or full list)."""
    from run import parse_args
    opts = parse_args([])
    assert opts["algorithms"] is None or opts["algorithms"] == ALL_ALGORITHMS


def test_parse_args_quality_only_returns_all():
    """Given only -q h, 'algorithms' means all (None or full list)."""
    from run import parse_args
    opts = parse_args(["-q", "h"])
    assert opts["algorithms"] is None or opts["algorithms"] == ALL_ALGORITHMS
    assert opts["quality"] == "h"


def test_parse_args_preview_flag():
    """Given -p, preview is True."""
    from run import parse_args
    opts = parse_args(["-p"])
    assert opts["preview"] is True


def test_parse_args_quality_mapping():
    """-q l/m/h maps to quality key."""
    from run import parse_args
    for q in ("l", "m", "h"):
        opts = parse_args(["-q", q])
        assert opts["quality"] == q


# ---- Demo filtering tests (custom_runner.get_demos_to_run) ----

def test_get_demos_to_run_bfs_returns_one_demo():
    """Given ['bfs'], get_demos_to_run returns one demo with algorithm 'bfs'."""
    from demo_data import get_demos_to_run
    demos = get_demos_to_run(["bfs"])
    assert len(demos) == 1
    assert demos[0]["algorithm"] == "bfs"


def test_get_demos_to_run_all_four_returns_four_demos():
    """Given all four algorithm keys, get_demos_to_run returns four demos."""
    from demo_data import get_demos_to_run
    demos = get_demos_to_run(ALL_ALGORITHMS)
    assert len(demos) == 4
    algorithms = [d["algorithm"] for d in demos]
    assert set(algorithms) == set(ALL_ALGORITHMS)


def test_get_demos_to_run_each_demo_has_required_keys():
    """Each returned demo has adjacency_list, algorithm, and kwargs (or equivalent)."""
    from demo_data import get_demos_to_run
    demos = get_demos_to_run(["bfs"])
    assert len(demos) >= 1
    d = demos[0]
    assert "algorithm" in d
    assert "adjacency_list" in d or "adj_list" in d  # allow either key
    # kwargs may be embedded (start, directed, weighted etc.)
    assert d["algorithm"] == "bfs"


if __name__ == "__main__":
    # Run test functions when pytest is not available
    tests = [
        test_parse_args_bfs_dijkstra_returns_two_algorithms,
        test_parse_args_all_returns_all_four,
        test_parse_args_no_flags_returns_all,
        test_parse_args_quality_only_returns_all,
        test_parse_args_preview_flag,
        test_parse_args_quality_mapping,
        test_get_demos_to_run_bfs_returns_one_demo,
        test_get_demos_to_run_all_four_returns_four_demos,
        test_get_demos_to_run_each_demo_has_required_keys,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except (Exception, SystemExit) as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
    print(f"\n{failed} failed, {len(tests) - failed} passed")
    raise SystemExit(1 if failed else 0)
