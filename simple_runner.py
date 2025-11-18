from animator import animate_algorithm
from manim import config


def main():
    config.save_pngs = True
    config.png_mode = "RGBA" 
    adj_list = {
        0: {1: 4, 2: 1},
        1: {0: 4, 2: 2, 3: 1, 4: 7},
        2: {0: 1, 1: 2, 3: 5, 5: 8},
        3: {1: 1, 2: 5, 4: 3, 5: 2},
        4: {1: 7, 3: 3, 5: 1},
        5: {2: 8, 3: 2, 4: 1},
    }
    animate_algorithm(
        adj_list,
        "dijkstra",
        start=0,
        weighted=True,
        directed=False,
    )

if __name__ == "__main__":
    main()
