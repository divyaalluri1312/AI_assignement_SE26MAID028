"""
Assignment 2 - Informed Search

This program demonstrates:
1. Greedy Best-First Search
2. Best-First Search
3. Dijkstra's Algorithm
4. A* Search
5. Beam Search
6. Recursive Best-First Search (RBFS)
7. IDA*
8. Weighted A*

The same graph and heuristic are used so that the
different search strategies can be compared.
"""

import heapq
import math


# ---------------------------------------------------------
# Graph
# ---------------------------------------------------------

GRAPH = {
    "S": [("A", 2), ("B", 5)],
    "A": [("S", 2), ("C", 2), ("D", 5)],
    "B": [("S", 5), ("D", 2), ("E", 4)],
    "C": [("A", 2), ("G", 5)],
    "D": [("A", 5), ("B", 2), ("G", 3)],
    "E": [("B", 4), ("G", 2)],
    "G": [("C", 5), ("D", 3), ("E", 2)],
}


# Heuristic values h(n)
H = {
    "S": 7,
    "A": 5,
    "B": 5,
    "C": 4,
    "D": 3,
    "E": 2,
    "G": 0,
}


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def reconstruct(parent, goal):
    """Build the final path using the parent dictionary."""
    path = [goal]

    while path[-1] in parent and parent[path[-1]] is not None:
        path.append(parent[path[-1]])

    path.reverse()
    return path


def edge_cost(a, b):
    """Return the cost of the edge from a to b."""
    for nxt, cost in GRAPH[a]:
        if nxt == b:
            return cost

    raise ValueError(f"No edge from {a} to {b}")


def cost_of(path):
    """Calculate total cost of a path."""
    return sum(
        edge_cost(a, b)
        for a, b in zip(path, path[1:])
    )


# ---------------------------------------------------------
# 1. Greedy Best-First Search
# f(n) = h(n)
# ---------------------------------------------------------

def greedy_best_first(start, goal):

    frontier = [(H[start], start)]
    parent = {start: None}
    visited = set()

    expanded = 0

    while frontier:

        _, node = heapq.heappop(frontier)

        if node in visited:
            continue

        visited.add(node)
        expanded += 1

        if node == goal:
            path = reconstruct(parent, goal)
            return path, cost_of(path), expanded

        for nxt, _ in GRAPH[node]:

            if nxt not in visited:
                if nxt not in parent:
                    parent[nxt] = node

                heapq.heappush(
                    frontier,
                    (H[nxt], nxt)
                )

    return None, None, expanded


# ---------------------------------------------------------
# 2. Best-First Search
#
# A general evaluation function is used here:
# f(n) = g(n) + 2h(n)
#
# This keeps it different from:
# Greedy:      f = h
# A*:          f = g + h
# Weighted A*: f = g + 1.5h
# ---------------------------------------------------------

def best_first_search(start, goal):

    frontier = [
        (2 * H[start], 0, start)
    ]

    best_g = {start: 0}
    parent = {start: None}

    expanded = 0

    while frontier:

        _, g, node = heapq.heappop(frontier)

        if g != best_g.get(node):
            continue

        expanded += 1

        if node == goal:
            path = reconstruct(parent, goal)
            return path, g, expanded

        for nxt, weight in GRAPH[node]:

            new_g = g + weight

            if new_g < best_g.get(nxt, math.inf):

                best_g[nxt] = new_g
                parent[nxt] = node

                f_value = new_g + 2 * H[nxt]

                heapq.heappush(
                    frontier,
                    (f_value, new_g, nxt)
                )

    return None, None, expanded


# ---------------------------------------------------------
# Generic Best-First Evaluation
# ---------------------------------------------------------

def best_first_evaluation(start, goal, heuristic_weight):

    frontier = [
        (
            heuristic_weight * H[start],
            0,
            start
        )
    ]

    best_g = {start: 0}
    parent = {start: None}

    expanded = 0

    while frontier:

        _, g, node = heapq.heappop(frontier)

        if g != best_g.get(node):
            continue

        expanded += 1

        if node == goal:
            path = reconstruct(parent, goal)
            return path, g, expanded

        for nxt, weight in GRAPH[node]:

            new_g = g + weight

            if new_g < best_g.get(nxt, math.inf):

                best_g[nxt] = new_g
                parent[nxt] = node

                f_value = (
                    new_g
                    + heuristic_weight * H[nxt]
                )

                heapq.heappush(
                    frontier,
                    (f_value, new_g, nxt)
                )

    return None, None, expanded


# ---------------------------------------------------------
# 3. Dijkstra's Algorithm
# f(n) = g(n)
# ---------------------------------------------------------

def dijkstra(start, goal):

    return best_first_evaluation(
        start,
        goal,
        0
    )


# ---------------------------------------------------------
# 4. A* Search
# f(n) = g(n) + h(n)
# ---------------------------------------------------------

def a_star(start, goal):

    return best_first_evaluation(
        start,
        goal,
        1
    )


# ---------------------------------------------------------
# 5. Weighted A*
# f(n) = g(n) + w*h(n)
# ---------------------------------------------------------

def weighted_a_star(start, goal, weight=1.5):

    return best_first_evaluation(
        start,
        goal,
        weight
    )


# ---------------------------------------------------------
# 6. Beam Search
# ---------------------------------------------------------

def beam_search(start, goal, width=2):

    beam = [
        (start, [start], 0)
    ]

    expanded = 0

    while beam:

        candidates = []

        for node, path, current_cost in beam:

            expanded += 1

            if node == goal:
                return path, current_cost, expanded

            for nxt, weight in GRAPH[node]:

                if nxt in path:
                    continue

                new_path = path + [nxt]
                new_cost = current_cost + weight

                candidates.append(
                    (
                        H[nxt],
                        nxt,
                        new_path,
                        new_cost
                    )
                )

        if not candidates:
            return None, None, expanded

        candidates.sort(
            key=lambda item: (
                item[0],
                item[3],
                item[1]
            )
        )

        beam = [
            (node, path, cost)
            for _, node, path, cost
            in candidates[:width]
        ]

    return None, None, expanded


# ---------------------------------------------------------
# 7. Recursive Best-First Search (RBFS)
# ---------------------------------------------------------

def rbfs(start, goal):

    expanded = 0

    def search(node, path, g, f_limit):

        nonlocal expanded

        expanded += 1

        if node == goal:
            return path, g, H[node]

        successors = []

        for nxt, weight in GRAPH[node]:

            if nxt in path:
                continue

            new_g = g + weight
            new_f = max(
                new_g + H[nxt],
                g + H[node]
            )

            successors.append(
                [
                    nxt,
                    path + [nxt],
                    new_g,
                    new_f
                ]
            )

        if not successors:
            return None, math.inf, math.inf

        while True:

            successors.sort(
                key=lambda item: (
                    item[3],
                    item[0]
                )
            )

            best = successors[0]

            if best[3] > f_limit:
                return None, math.inf, best[3]

            if len(successors) > 1:
                alternative = successors[1][3]
            else:
                alternative = math.inf

            result_path, result_cost, result_f = search(
                best[0],
                best[1],
                best[2],
                min(f_limit, alternative)
            )

            best[3] = result_f

            if result_path is not None:
                return result_path, result_cost, result_f

    path, cost, _ = search(
        start,
        [start],
        0,
        math.inf
    )

    if path is None:
        return None, None, expanded

    return path, cost, expanded


# ---------------------------------------------------------
# 8. IDA*
# ---------------------------------------------------------

def ida_star(start, goal):

    expanded = 0

    def search(node, path, g, threshold):

        nonlocal expanded

        expanded += 1

        f_value = g + H[node]

        if f_value > threshold:
            return None, f_value

        if node == goal:
            return (path, g), g

        minimum = math.inf

        for nxt, weight in GRAPH[node]:

            if nxt in path:
                continue

            result, value = search(
                nxt,
                path + [nxt],
                g + weight,
                threshold
            )

            if result is not None:
                return result, value

            minimum = min(
                minimum,
                value
            )

        return None, minimum

    threshold = H[start]

    while threshold < math.inf:

        result, next_threshold = search(
            start,
            [start],
            0,
            threshold
        )

        if result is not None:

            path, cost = result

            return path, cost, expanded

        if next_threshold == math.inf:
            break

        threshold = next_threshold

    return None, None, expanded


# ---------------------------------------------------------
# Display function
# ---------------------------------------------------------

def show(name, result):

    path, cost, expanded = result

    print("\n" + name)
    print("-" * len(name))

    if path:

        print(
            "Path          :",
            " -> ".join(path)
        )

        print(
            "Path cost     :",
            cost
        )

    else:

        print("No path found.")

    print(
        "Nodes expanded:",
        expanded
    )


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------

def main():

    start = "S"
    goal = "G"

    print("INFORMED SEARCH DEMONSTRATION")
    print(f"Start = {start}, Goal = {goal}")

    show(
        "Greedy Best-First Search",
        greedy_best_first(start, goal)
    )

    show(
        "Best-First Search (f = g + 2h)",
        best_first_search(start, goal)
    )

    show(
        "Dijkstra's Algorithm (f = g)",
        dijkstra(start, goal)
    )

    show(
        "A* (f = g + h)",
        a_star(start, goal)
    )

    show(
        "Beam Search (width = 2)",
        beam_search(start, goal, width=2)
    )

    show(
        "RBFS",
        rbfs(start, goal)
    )

    show(
        "IDA*",
        ida_star(start, goal)
    )

    show(
        "Weighted A* (w = 1.5)",
        weighted_a_star(start, goal, 1.5)
    )

    print("\nHeuristic values:")

    for node, value in H.items():
        print(f"{node}: {value}")


if __name__ == "__main__":
    main()
