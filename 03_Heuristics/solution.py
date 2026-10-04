"""
Assignment 3 - Heuristics

This program demonstrates:
1. Satisficing search
2. Admissible heuristics
3. Formulating a heuristic
4. Combining heuristics generated from subproblems

The 8-puzzle is used because its state, goal state and legal moves
are easy to understand.
"""

from collections import deque
import heapq

GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)


def neighbours(state):
    zero = state.index(0)
    row, col = divmod(zero, 3)

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    result = []

    for dr, dc in moves:
        nr, nc = row + dr, col + dc

        if 0 <= nr < 3 and 0 <= nc < 3:
            j = nr * 3 + nc
            new_state = list(state)
            new_state[zero], new_state[j] = new_state[j], new_state[zero]
            result.append(tuple(new_state))

    return result


def misplaced_tiles(state, selected=None):
    count = 0
    selected = set(selected) if selected else set(range(1, 9))

    for index, value in enumerate(state):
        if value == 0 or value not in selected:
            continue
        if value != GOAL[index]:
            count += 1

    return count


def manhattan_distance(state, selected=None):
    selected = set(selected) if selected else set(range(1, 9))
    distance = 0

    goal_positions = {
        value: divmod(index, 3)
        for index, value in enumerate(GOAL)
    }

    for index, value in enumerate(state):
        if value == 0 or value not in selected:
            continue

        r1, c1 = divmod(index, 3)
        r2, c2 = goal_positions[value]
        distance += abs(r1 - r2) + abs(c1 - c2)

    return distance


def subproblem_heuristic(state):
    # Solve two smaller relaxed problems:
    # P1 tracks tiles 1,2,3,4
    # P2 tracks tiles 5,6,7,8
    #
    # Each subproblem gives a lower bound on the real problem.
    # The maximum of admissible lower bounds is also admissible.
    h1 = manhattan_distance(state, {1, 2, 3, 4})
    h2 = manhattan_distance(state, {5, 6, 7, 8})
    return max(h1, h2)


def exact_distance(start, max_depth=12):
    """Return the exact distance for small test states.

    A small depth limit keeps this demonstration quick.
    """
    if start == GOAL:
        return 0

    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        state, depth = queue.popleft()

        if depth >= max_depth:
            continue

        for nxt in neighbours(state):
            if nxt in visited:
                continue

            if nxt == GOAL:
                return depth + 1

            visited.add(nxt)
            queue.append((nxt, depth + 1))

    return None


def satisficing_search(graph, start, goal, acceptable_cost):
    """Stop as soon as a path no more expensive than the target is found.

    This is an example of satisficing behaviour: the first acceptable
    solution is enough; the algorithm does not insist on the optimum.
    """
    frontier = [(0, start, [start])]
    best = {start: 0}

    while frontier:
        cost, node, path = heapq.heappop(frontier)

        if cost != best[node]:
            continue

        if node == goal:
            return path, cost

        for nxt, edge_cost in graph[node]:
            new_cost = cost + edge_cost

            if new_cost <= acceptable_cost and new_cost < best.get(nxt, float("inf")):
                best[nxt] = new_cost
                heapq.heappush(frontier, (new_cost, nxt, path + [nxt]))

    return None, None


def main():
    test_states = [
        (1, 2, 3, 4, 5, 6, 0, 7, 8),
        (1, 2, 3, 5, 0, 6, 4, 7, 8),
        (2, 8, 3, 1, 6, 4, 7, 0, 5),
    ]

    print("HEURISTIC GENERATION USING THE 8-PUZZLE")
    print("=" * 45)

    for state in test_states:
        print("\nState:", state)

        h_misplaced = misplaced_tiles(state)
        h_manhattan = manhattan_distance(state)
        h_subproblem = subproblem_heuristic(state)
        actual = exact_distance(state)

        print("Misplaced-tile heuristic :", h_misplaced)
        print("Manhattan heuristic      :", h_manhattan)
        print("Subproblem heuristic     :", h_subproblem)

        if actual is not None:
            print("Actual solution distance :", actual)
            print(
                "Manhattan admissible?    :",
                h_manhattan <= actual,
            )
            print(
                "Subproblem admissible?   :",
                h_subproblem <= actual,
            )
        else:
            print("Actual distance          : not calculated within depth limit")

    print("\nSATISFICING SEARCH EXAMPLE")
    graph = {
        "S": [("A", 2), ("B", 4)],
        "A": [("G", 8)],
        "B": [("G", 5)],
        "G": [],
    }

    path, cost = satisficing_search(
        graph, "S", "G", acceptable_cost=9
    )

    print("Acceptable cost:", 9)
    print("Path:", " -> ".join(path) if path else "No acceptable path")
    print("Cost:", cost if cost is not None else "N/A")


if __name__ == "__main__":
    main()
