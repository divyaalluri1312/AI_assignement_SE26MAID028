"""
Assignment 6 - UGV Navigation with Dynamic Obstacles

Unlike the static version, the UGV does not know the complete
obstacle map. Obstacles move/change during execution.

At every step:
1. The UGV senses nearby obstacles.
2. Newly observed obstacles are added to its internal map.
3. A* replans a path from the current position.
4. The UGV moves one step and the environment changes again.

This is a simple model of online replanning.
"""

import heapq
import math
import random
import time

import matplotlib.pyplot as plt

SIZE = 50
SENSE_RADIUS = 3
INITIAL_DENSITY = 0.12

MOVES = [
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
]


def heuristic(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def inside(p):
    return 0 <= p[0] < SIZE and 0 <= p[1] < SIZE


def astar(blocked, start, goal):
    queue = [(heuristic(start, goal), 0, start)]
    g = {start: 0}
    parent = {start: None}

    while queue:
        _, cost, current = heapq.heappop(queue)

        if cost != g.get(current):
            continue

        if current == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            return list(reversed(path))

        for dr, dc in MOVES:
            nxt = (current[0] + dr, current[1] + dc)

            if not inside(nxt) or nxt in blocked:
                continue

            new_cost = cost + 1

            if new_cost < g.get(nxt, float("inf")):
                g[nxt] = new_cost
                parent[nxt] = current
                heapq.heappush(
                    queue, (new_cost + heuristic(nxt, goal), new_cost, nxt)
                )

    return None


def generate_environment(seed):
    random.seed(seed)
    blocked = set()

    for r in range(SIZE):
        for c in range(SIZE):
            if random.random() < INITIAL_DENSITY:
                blocked.add((r, c))

    return blocked


def move_dynamic_obstacles(blocked, start, goal):
    # A fraction of obstacles move to neighbouring cells.
    updated = set()

    for obstacle in blocked:
        if obstacle in (start, goal):
            continue

        if random.random() < 0.25:
            options = [
                (obstacle[0] + dr, obstacle[1] + dc)
                for dr, dc in MOVES
                if inside((obstacle[0] + dr, obstacle[1] + dc))
            ]

            if options:
                candidate = random.choice(options)
                if candidate not in (start, goal):
                    updated.add(candidate)
                else:
                    updated.add(obstacle)
            else:
                updated.add(obstacle)
        else:
            updated.add(obstacle)

    return updated


def sense_obstacles(actual_blocked, position):
    known = set()

    for obstacle in actual_blocked:
        if (
            abs(obstacle[0] - position[0]) <= SENSE_RADIUS
            and abs(obstacle[1] - position[1]) <= SENSE_RADIUS
        ):
            known.add(obstacle)

    return known


def plot_environment(actual, known, path, start, goal, current):
    plt.figure(figsize=(8, 8))

    if actual:
        x = [c for r, c in actual]
        y = [SIZE - 1 - r for r, c in actual]
        plt.scatter(x, y, s=8, marker="s", alpha=0.25, label="Actual obstacles")

    if known:
        x = [c for r, c in known]
        y = [SIZE - 1 - r for r, c in known]
        plt.scatter(x, y, s=18, marker="s", label="Known obstacles")

    if path:
        x = [c for r, c in path]
        y = [SIZE - 1 - r for r, c in path]
        plt.plot(x, y, linewidth=2, label="Current planned path")

    plt.scatter([start[1]], [SIZE - 1 - start[0]], s=80, label="Start")
    plt.scatter([goal[1]], [SIZE - 1 - goal[0]], s=100, marker="*", label="Goal")
    plt.scatter([current[1]], [SIZE - 1 - current[0]], s=70, label="UGV")

    plt.title("UGV Dynamic Obstacle Navigation")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def main():
    random.seed()
    start = (2, 2)
    goal = (SIZE - 3, SIZE - 3)

    actual_blocked = generate_environment(int(time.time()) % 100000)
    actual_blocked.discard(start)
    actual_blocked.discard(goal)

    known_blocked = set()
    current = start
    travelled = 0
    replans = 0
    collision_avoided = 0
    trajectory = [current]

    max_steps = 300

    for step in range(max_steps):
        if current == goal:
            break

        # Sense the environment around the current position.
        newly_seen = sense_obstacles(actual_blocked, current)
        known_blocked.update(newly_seen)

        # Plan using only the information available to the UGV.
        path = astar(known_blocked, current, goal)
        replans += 1

        if not path:
            print("No route is currently known. Waiting and sensing again...")
            actual_blocked = move_dynamic_obstacles(
                actual_blocked, current, goal
            )
            continue

        # Move one cell along the current plan.
        next_position = path[1] if len(path) > 1 else current

        if next_position in actual_blocked:
            # The obstacle was not known when the previous plan was made.
            known_blocked.add(next_position)
            collision_avoided += 1
            actual_blocked = move_dynamic_obstacles(
                actual_blocked, current, goal
            )
            continue

        current = next_position
        travelled += 1
        trajectory.append(current)

        # Dynamic environment changes after movement.
        actual_blocked = move_dynamic_obstacles(
            actual_blocked, current, goal
        )

    success = current == goal

    print("\nDYNAMIC UGV RESULTS")
    print("-" * 35)
    print("Goal reached       :", success)
    print("Steps travelled    :", travelled)
    print("Replanning count   :", replans)
    print("Unexpected blocks  :", collision_avoided)
    print("Final position     :", current)
    print("Known obstacles    :", len(known_blocked))

    # Plot the final state and the last planned route.
    final_path = astar(known_blocked, current, goal) if not success else [current]
    plot_environment(
        actual_blocked, known_blocked, final_path,
        start, goal, current
    )


if __name__ == "__main__":
    main()
