"""
Assignment 5 - UGV Navigation with Static Obstacles

A 70 x 70 grid represents the battlefield.
The user can choose low, medium or high obstacle density.
Obstacles are known before the search begins.

A* with an 8-direction movement model finds a shortest path.
The program also reports Measures of Effectiveness (MOE).
"""

import heapq
import math
import random
import time

import matplotlib.pyplot as plt

SIZE = 70
DENSITIES = {
    "low": 0.15,
    "medium": 0.25,
    "high": 0.35,
}

MOVES = [
    (-1, 0, 1.0), (1, 0, 1.0),
    (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
]


def heuristic(a, b):
    # Octile distance for 8-direction movement.
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy)


def astar(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_set = [(heuristic(start, goal), 0.0, start)]
    g_score = {start: 0.0}
    parent = {start: None}
    expanded = 0

    while open_set:
        _, current_g, current = heapq.heappop(open_set)

        if current_g != g_score.get(current):
            continue

        expanded += 1

        if current == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path, current_g, expanded

        r, c = current

        for dr, dc, move_cost in MOVES:
            nr, nc = r + dr, c + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr][nc]:
                continue

            # Prevent cutting diagonally through the corner of two obstacles.
            if dr != 0 and dc != 0:
                if grid[r + dr][c] or grid[r][c + dc]:
                    continue

            neighbour = (nr, nc)
            new_g = current_g + move_cost

            if new_g < g_score.get(neighbour, float("inf")):
                g_score[neighbour] = new_g
                parent[neighbour] = current
                f = new_g + heuristic(neighbour, goal)
                heapq.heappush(open_set, (f, new_g, neighbour))

    return None, None, expanded


def create_grid(density, start, goal, seed):
    random.seed(seed)

    while True:
        grid = [
            [random.random() < density for _ in range(SIZE)]
            for _ in range(SIZE)
        ]

        grid[start[0]][start[1]] = False
        grid[goal[0]][goal[1]] = False

        path, _, _ = astar(grid, start, goal)

        if path:
            return grid


def count_turns(path):
    if len(path) < 3:
        return 0

    turns = 0
    previous_direction = None

    for a, b in zip(path, path[1:]):
        direction = (b[0] - a[0], b[1] - a[1])

        if previous_direction is not None and direction != previous_direction:
            turns += 1

        previous_direction = direction

    return turns


def plot_grid(grid, path, start, goal, density_name):
    plt.figure(figsize=(8, 8))

    obstacle_x = []
    obstacle_y = []

    for r in range(SIZE):
        for c in range(SIZE):
            if grid[r][c]:
                obstacle_x.append(c)
                obstacle_y.append(SIZE - 1 - r)

    if obstacle_x:
        plt.scatter(obstacle_x, obstacle_y, s=5, marker="s")

    if path:
        px = [c for r, c in path]
        py = [SIZE - 1 - r for r, c in path]
        plt.plot(px, py, linewidth=2, label="UGV path")

    plt.scatter([start[1]], [SIZE - 1 - start[0]], s=70, marker="o", label="Start")
    plt.scatter([goal[1]], [SIZE - 1 - goal[0]], s=70, marker="*", label="Goal")

    plt.title(f"UGV Static Obstacles - {density_name.title()} Density")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def main():
    print("UGV STATIC OBSTACLE NAVIGATION")
    print("Grid size: 70 x 70")

    density_name = input("Obstacle density (low/medium/high) [medium]: ").strip().lower()
    if density_name not in DENSITIES:
        density_name = "medium"

    start_text = input("Start row,column [2,2]: ").strip() or "2,2"
    goal_text = input("Goal row,column [67,67]: ").strip() or "67,67"

    try:
        start = tuple(map(int, start_text.split(",")))
        goal = tuple(map(int, goal_text.split(",")))
        if len(start) != 2 or len(goal) != 2:
            raise ValueError
        if not all(0 <= x < SIZE for x in start + goal):
            raise ValueError
    except ValueError:
        print("Invalid coordinates. Use values such as 2,2.")
        return

    seed = int(time.time()) % 100000
    grid = create_grid(DENSITIES[density_name], start, goal, seed)

    t0 = time.perf_counter()
    path, distance, expanded = astar(grid, start, goal)
    elapsed = time.perf_counter() - t0

    print("\nMeasures of Effectiveness")
    print("-" * 35)
    print("Obstacle density :", density_name)
    print("Start             :", start)
    print("Goal              :", goal)
    print("Path found        :", bool(path))
    print("Path distance     :", round(distance, 3) if distance else "N/A")
    print("Path cells        :", len(path) if path else 0)
    print("Nodes expanded    :", expanded)
    print("Number of turns   :", count_turns(path) if path else "N/A")
    print("Search time (sec) :", round(elapsed, 6))

    plot_grid(grid, path, start, goal, density_name)


if __name__ == "__main__":
    main()
