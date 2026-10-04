"""
Assignment 2 - Informed Search

Demonstrates:
Greedy Best-First, Best-First, Dijkstra, A*, Beam Search,
Recursive Best-First Search (RBFS), IDA* and Weighted A*.

The graph and heuristic are intentionally small so the algorithms
can be inspected and explained during a viva.
"""

import heapq
import math

GRAPH = {
    "S": [("A", 2), ("B", 5)],
    "A": [("S", 2), ("C", 2), ("D", 5)],
    "B": [("S", 5), ("D", 2), ("E", 4)],
    "C": [("A", 2), ("G", 5)],
    "D": [("A", 5), ("B", 2), ("G", 3)],
    "E": [("B", 4), ("G", 2)],
    "G": [("C", 5), ("D", 3), ("E", 2)],
}

# A consistent heuristic for this graph.
H = {
    "S": 7,
    "A": 5,
    "B": 5,
    "C": 4,
    "D": 3,
    "E": 2,
    "G": 0,
}


def reconstruct(parent, goal):
    path = [goal]
    while path[-1] in parent and parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def edge_cost(a, b):
    for nxt, cost in GRAPH[a]:
        if nxt == b:
            return cost
    raise ValueError(f"No edge from {a} to {b}")


def cost_of(path):
    return sum(edge_cost(a, b) for a, b in zip(path, path[1:]))


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
            if nxt not in visited and nxt not in parent:
                parent[nxt] = node
                heapq.heappush(frontier, (H[nxt], nxt))

    return None, None, expanded


def best_first(start, goal, evaluation):
    """Generic best-first search.

    evaluation receives (g, h) and returns the priority.
    """
    frontier = [(evaluation(0, H[start]), 0, start)]
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

        for nxt, w in GRAPH[node]:
            new_g = g + w
            if new_g < best_g.get(nxt, math.inf):
                best_g[nxt] = new_g
                parent[nxt] = node
                heapq.heappush(
                    frontier, (evaluation(new_g, H[nxt]), new_g, nxt)
                )

    return None, None, expanded


def dijkstra(start, goal):
    return best_first(start, goal, lambda g, h: g)


def a_star(start, goal):
    return best_first(start, goal, lambda g, h: g + h)


def weighted_a_star(start, goal, weight=1.5):
    return best_first(start, goal, lambda g, h: g + weight * h)


def beam_search(start, goal, width=2):
    """Layered beam search.

    At each depth only the best 'width' partial paths according
    to h(n) are retained.
    """
    beam = [(start, [start], 0)]
    expanded = 0

    while beam:
        candidates = []

        for node, path, g in beam:
            expanded += 1

            if node == goal:
                return path, g, expanded

            for nxt, w in GRAPH[node]:
                if nxt in path:
                    continue
                candidates.append(
                    (H[nxt], nxt, path + [nxt], g + w)
                )

        candidates.sort(key=lambda x: (x[0], x[3]))
        beam = [(node, path, g) for _, node, path, g in candidates[:width]]

    return None, None, expanded


def rbfs(start, goal):
    """Recursive Best-First Search."""

    expanded = 0

    def search(node, path, g, f_limit):
        nonlocal expanded
        expanded += 1

        if node == goal:
            return path, g, H[node]

        successors = []
        for nxt, w in GRAPH[node]:
            if nxt in path:
                continue
            new_g = g + w
            successors.append([nxt, path + [nxt], new_g, max(new_g + H[nxt], H[node])])

        if not successors:
            return None, math.inf, math.inf

        while True:
            successors.sort(key=lambda x: x[3])
            best = successors[0]

            if best[3] > f_limit:
                return None, math.inf, best[3]

            alternative = successors[1][3] if len(successors) > 1 else math.inf
            result_path, result_cost, result_f = search(
                best[0], best[1], best[2], min(f_limit, alternative)
            )

            best[3] = result_f

            if result_path is not None:
                return result_path, result_cost, result_f

    path, cost, _ = search(start, [start], 0, math.inf)
    return path, (cost if path else None), expanded


def ida_star(start, goal):
    """Iterative Deepening A* using f = g + h."""

    expanded = 0

    def search(node, path, g, threshold):
        nonlocal expanded
        expanded += 1

        f = g + H[node]
        if f > threshold:
            return None, f

        if node == goal:
            return (path, g), g

        minimum = math.inf

        for nxt, w in GRAPH[node]:
            if nxt in path:
                continue

            result, value = search(
                nxt, path + [nxt], g + w, threshold
            )

            if result is not None:
                return result, value

            minimum = min(minimum, value)

        return None, minimum

    threshold = H[start]

    while threshold < math.inf:
        result, next_threshold = search(start, [start], 0, threshold)

        if result is not None:
            path, cost = result
            return path, cost, expanded

        if next_threshold == math.inf:
            break

        threshold = next_threshold

    return None, None, expanded


def show(name, result):
    path, cost, expanded = result
    print(f"\n{name}")
    print("-" * len(name))
    if path:
        print("Path          :", " -> ".join(path))
        print("Path cost     :", cost)
    else:
        print("No path found.")
    print("Nodes expanded:", expanded)


def main():
    print("INFORMED SEARCH DEMONSTRATION")
    print("Start = S, Goal = G\n")

    show("Greedy Best-First Search", greedy_best_first("S", "G"))
    show("Best-First Search (f = g + h)", a_star("S", "G"))
    show("Dijkstra's Algorithm (f = g)", dijkstra("S", "G"))
    show("A* (f = g + h)", a_star("S", "G"))
    show("Beam Search (width = 2)", beam_search("S", "G", width=2))
    show("RBFS", rbfs("S", "G"))
    show("IDA*", ida_star("S", "G"))
    show("Weighted A* (w = 1.5)", weighted_a_star("S", "G", 1.5))

    print("\nHeuristic values:")
    for node, value in H.items():
        print(f"{node}: {value}")


if __name__ == "__main__":
    main()
