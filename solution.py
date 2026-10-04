"""
Assignment 1 - Uninformed Search

This program demonstrates:
BFS, Uniform-Cost Search, DFS, Depth-Limited Search,
Iterative Deepening Search and Bidirectional Search.

All algorithms use the same small weighted graph so their behaviour
can be compared easily.
"""

from collections import deque
import heapq

GRAPH = {
    "A": [("B", 2), ("C", 4)],
    "B": [("A", 2), ("D", 3), ("E", 5)],
    "C": [("A", 4), ("F", 3)],
    "D": [("B", 3), ("G", 2)],
    "E": [("B", 5), ("G", 1)],
    "F": [("C", 3), ("G", 2)],
    "G": [("D", 2), ("E", 1), ("F", 2)],
}

# Search criteria:
# - Completeness: whether a solution is guaranteed if one exists.
# - Optimality: whether the returned path is minimum-cost.
# - Time/space: number of states stored or expanded.
# The code below records expanded nodes and the final path cost.


def path_cost(graph, path):
    total = 0
    for a, b in zip(path, path[1:]):
        for nxt, cost in graph[a]:
            if nxt == b:
                total += cost
                break
    return total


def bfs(graph, start, goal):
    queue = deque([[start]])
    visited = {start}
    expanded = 0

    while queue:
        path = queue.popleft()
        node = path[-1]
        expanded += 1

        if node == goal:
            return path, path_cost(graph, path), expanded

        for nxt, _ in graph[node]:
            if nxt not in visited:
                visited.add(nxt)
                queue.append(path + [nxt])

    return None, None, expanded


def uniform_cost_search(graph, start, goal):
    pq = [(0, start, [start])]
    best_cost = {start: 0}
    expanded = 0

    while pq:
        cost, node, path = heapq.heappop(pq)
        if cost != best_cost.get(node):
            continue

        expanded += 1
        if node == goal:
            return path, cost, expanded

        for nxt, edge_cost in graph[node]:
            new_cost = cost + edge_cost
            if new_cost < best_cost.get(nxt, float("inf")):
                best_cost[nxt] = new_cost
                heapq.heappush(pq, (new_cost, nxt, path + [nxt]))

    return None, None, expanded


def dfs(graph, start, goal):
    stack = [(start, [start])]
    visited = set()
    expanded = 0

    while stack:
        node, path = stack.pop()
        if node in visited:
            continue

        visited.add(node)
        expanded += 1

        if node == goal:
            return path, path_cost(graph, path), expanded

        # Reverse only to make the demo deterministic.
        for nxt, _ in reversed(graph[node]):
            if nxt not in visited:
                stack.append((nxt, path + [nxt]))

    return None, None, expanded


def depth_limited_search(graph, start, goal, limit):
    expanded = 0

    def visit(node, path, depth):
        nonlocal expanded
        expanded += 1

        if node == goal:
            return path, "FOUND"

        if depth == limit:
            return None, "CUTOFF"

        cutoff_happened = False

        for nxt, _ in graph[node]:
            if nxt in path:
                continue
            result, status = visit(nxt, path + [nxt], depth + 1)

            if result is not None:
                return result, "FOUND"
            if status == "CUTOFF":
                cutoff_happened = True

        return None, "CUTOFF" if cutoff_happened else "FAILURE"

    result, status = visit(start, [start], 0)
    if result is None:
        return None, None, expanded, status
    return result, path_cost(graph, result), expanded, status


def iterative_deepening_search(graph, start, goal):
    total_expanded = 0

    for limit in range(len(graph)):
        path, cost, expanded, status = depth_limited_search(
            graph, start, goal, limit
        )
        total_expanded += expanded

        if path is not None:
            return path, cost, total_expanded

        if status == "FAILURE":
            break

    return None, None, total_expanded


def bidirectional_search(graph, start, goal):
    if start == goal:
        return [start], 0, 1

    # This implementation uses BFS from both ends.
    forward_q = deque([start])
    backward_q = deque([goal])
    forward_parent = {start: None}
    backward_parent = {goal: None}
    expanded = 0

    while forward_q and backward_q:
        # Expand one layer from the start.
        for _ in range(len(forward_q)):
            node = forward_q.popleft()
            expanded += 1

            for nxt, _ in graph[node]:
                if nxt not in forward_parent:
                    forward_parent[nxt] = node
                    forward_q.append(nxt)

                    if nxt in backward_parent:
                        return build_bidirectional_path(
                            forward_parent, backward_parent, nxt, graph, expanded
                        )

        # Expand one layer from the goal.
        for _ in range(len(backward_q)):
            node = backward_q.popleft()
            expanded += 1

            for nxt, _ in graph[node]:
                if nxt not in backward_parent:
                    backward_parent[nxt] = node
                    backward_q.append(nxt)

                    if nxt in forward_parent:
                        return build_bidirectional_path(
                            forward_parent, backward_parent, nxt, graph, expanded
                        )

    return None, None, expanded


def build_bidirectional_path(forward_parent, backward_parent, meeting, graph, expanded):
    left = []
    node = meeting
    while node is not None:
        left.append(node)
        node = forward_parent[node]
    left.reverse()

    right = []
    node = backward_parent[meeting]
    while node is not None:
        right.append(node)
        node = backward_parent[node]

    path = left + right
    return path, path_cost(graph, path), expanded


ALGORITHMS = {
    "1": "BFS",
    "2": "UCS",
    "3": "DFS",
    "4": "DLS",
    "5": "IDS",
    "6": "BIDIRECTIONAL",
}


def run():
    print("\nUNINFORMED SEARCH DEMONSTRATION")
    print("Graph: A is the start and G is the goal.")
    print("\n1. Breadth-First Search")
    print("2. Uniform-Cost Search")
    print("3. Depth-First Search")
    print("4. Depth-Limited Search")
    print("5. Iterative Deepening Search")
    print("6. Bidirectional Search")

    choice = input("\nChoose an algorithm (1-6): ").strip()
    start = input("Start node [A]: ").strip().upper() or "A"
    goal = input("Goal node [G]: ").strip().upper() or "G"

    if start not in GRAPH or goal not in GRAPH:
        print("Invalid node. Use one of:", ", ".join(GRAPH))
        return

    if choice == "1":
        path, cost, expanded = bfs(GRAPH, start, goal)
    elif choice == "2":
        path, cost, expanded = uniform_cost_search(GRAPH, start, goal)
    elif choice == "3":
        path, cost, expanded = dfs(GRAPH, start, goal)
    elif choice == "4":
        limit_text = input("Depth limit [3]: ").strip()
        limit = int(limit_text) if limit_text else 3
        path, cost, expanded, status = depth_limited_search(
            GRAPH, start, goal, limit
        )
        if path is None:
            print(f"\nNo solution within depth limit {limit}. Status: {status}")
    elif choice == "5":
        path, cost, expanded = iterative_deepening_search(
            GRAPH, start, goal
        )
    elif choice == "6":
        path, cost, expanded = bidirectional_search(
            GRAPH, start, goal
        )
    else:
        print("Please choose a number from 1 to 6.")
        return

    if path:
        print("\nResult")
        print("-" * 35)
        print("Path          :", " -> ".join(path))
        print("Path cost     :", cost)
        print("Nodes expanded:", expanded)
    else:
        print("\nNo path found.")


if __name__ == "__main__":
    run()
