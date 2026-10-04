"""
Assignment 3 - Heuristics

This program demonstrates:
1. Satisficing search
2. Admissible heuristics
3. Formulating a heuristic
4. Generating heuristics from subproblems

The 8-puzzle is used to demonstrate heuristic formulation.
"""

from collections import deque
import heapq


# ---------------------------------------------------------
# 8-Puzzle Goal State
# ---------------------------------------------------------

GOAL = (1, 2, 3,
        4, 5, 6,
        7, 8, 0)


# ---------------------------------------------------------
# Generate possible next states
# ---------------------------------------------------------

def neighbours(state):
    zero_index = state.index(0)
    row, col = divmod(zero_index, 3)

    moves = [
        (-1, 0),   # Up
        (1, 0),    # Down
        (0, -1),   # Left
        (0, 1)     # Right
    ]

    next_states = []

    for row_change, col_change in moves:

        new_row = row + row_change
        new_col = col + col_change

        if 0 <= new_row < 3 and 0 <= new_col < 3:

            new_index = new_row * 3 + new_col

            new_state = list(state)

            new_state[zero_index], new_state[new_index] = (
                new_state[new_index],
                new_state[zero_index]
            )

            next_states.append(tuple(new_state))

    return next_states


# ---------------------------------------------------------
# Heuristic 1: Misplaced Tiles
# ---------------------------------------------------------

def misplaced_tiles(state, selected=None):
    """
    Counts the number of tiles that are not in their
    goal position.

    The blank tile (0) is ignored.
    """

    if selected is None:
        selected = set(range(1, 9))
    else:
        selected = set(selected)

    count = 0

    for index, tile in enumerate(state):

        if tile == 0:
            continue

        if tile not in selected:
            continue

        if tile != GOAL[index]:
            count += 1

    return count


# ---------------------------------------------------------
# Heuristic 2: Manhattan Distance
# ---------------------------------------------------------

def manhattan_distance(state, selected=None):
    """
    Calculates the total Manhattan distance of the selected
    tiles from their goal positions.

    Manhattan distance:
    |current_row - goal_row| +
    |current_column - goal_column|
    """

    if selected is None:
        selected = set(range(1, 9))
    else:
        selected = set(selected)

    goal_positions = {}

    for index, tile in enumerate(GOAL):
        goal_positions[tile] = divmod(index, 3)

    distance = 0

    for index, tile in enumerate(state):

        if tile == 0:
            continue

        if tile not in selected:
            continue

        current_row, current_col = divmod(index, 3)
        goal_row, goal_col = goal_positions[tile]

        distance += (
            abs(current_row - goal_row)
            + abs(current_col - goal_col)
        )

    return distance


# ---------------------------------------------------------
# Heuristic 3: Subproblem Heuristic
# ---------------------------------------------------------

def subproblem_heuristic(state):
    """
    Generates heuristic values from two smaller relaxed
    subproblems.

    Subproblem 1:
        Tiles 1, 2, 3, 4

    Subproblem 2:
        Tiles 5, 6, 7, 8

    Each subproblem provides a lower bound on the
    original problem.

    The maximum of the two lower bounds is used because
    the maximum of admissible lower bounds remains
    admissible.
    """

    first_group = {1, 2, 3, 4}
    second_group = {5, 6, 7, 8}

    h1 = manhattan_distance(
        state,
        first_group
    )

    h2 = manhattan_distance(
        state,
        second_group
    )

    return max(h1, h2)


# ---------------------------------------------------------
# Exact Solution Distance
# ---------------------------------------------------------

def exact_distance(start, max_depth=12):
    """
    Finds the exact solution distance using BFS.

    A depth limit is used so that the demonstration
    remains computationally small.
    """

    if start == GOAL:
        return 0

    queue = deque()

    queue.append((start, 0))

    visited = {start}

    while queue:

        state, depth = queue.popleft()

        if depth >= max_depth:
            continue

        for next_state in neighbours(state):

            if next_state in visited:
                continue

            if next_state == GOAL:
                return depth + 1

            visited.add(next_state)

            queue.append(
                (next_state, depth + 1)
            )

    return None


# ---------------------------------------------------------
# Satisficing Search
# ---------------------------------------------------------

def satisficing_search(
    graph,
    start,
    goal,
    acceptable_cost
):
    """
    Finds a solution whose cost does not exceed the
    acceptable cost.

    The search stops once an acceptable solution reaches
    the goal. The objective is to obtain a good-enough
    solution rather than continue searching unnecessarily
    for a better one.
    """

    frontier = [
        (0, start, [start])
    ]

    best_cost = {
        start: 0
    }

    while frontier:

        cost, node, path = heapq.heappop(frontier)

        if cost != best_cost.get(node):
            continue

        if node == goal:

            if cost <= acceptable_cost:
                return path, cost

            continue

        for next_node, edge_cost in graph[node]:

            new_cost = cost + edge_cost

            if new_cost > acceptable_cost:
                continue

            if new_cost < best_cost.get(
                next_node,
                float("inf")
            ):

                best_cost[next_node] = new_cost

                heapq.heappush(
                    frontier,
                    (
                        new_cost,
                        next_node,
                        path + [next_node]
                    )
                )

    return None, None


# ---------------------------------------------------------
# Main Demonstration
# ---------------------------------------------------------

def main():

    test_states = [

        # Solution distance = 2
        (
            1, 2, 3,
            4, 5, 6,
            0, 7, 8
        ),

        # Solution distance = 4
        (
            1, 2, 3,
            5, 0, 6,
            4, 7, 8
        ),

        # A more difficult state
        (
            2, 8, 3,
            1, 6, 4,
            7, 0, 5
        )
    ]

    print("HEURISTIC GENERATION USING THE 8-PUZZLE")
    print("=" * 50)

    for state in test_states:

        print("\nState:", state)

        h_misplaced = misplaced_tiles(state)

        h_manhattan = manhattan_distance(state)

        h_subproblem = subproblem_heuristic(state)

        actual_distance = exact_distance(
            state,
            max_depth=12
        )

        print(
            "Misplaced-tile heuristic :",
            h_misplaced
        )

        print(
            "Manhattan heuristic      :",
            h_manhattan
        )

        print(
            "Subproblem heuristic     :",
            h_subproblem
        )

        if actual_distance is not None:

            print(
                "Actual solution distance :",
                actual_distance
            )

            print(
                "Manhattan admissible?    :",
                h_manhattan <= actual_distance
            )

            print(
                "Subproblem admissible?   :",
                h_subproblem <= actual_distance
            )

        else:

            print(
                "Actual distance          :",
                "not calculated within depth limit"
            )

            print(
                "Admissibility check      :",
                "not performed because the"
                " exact distance is unknown"
            )


    # -----------------------------------------------------
    # Satisficing Search Example
    # -----------------------------------------------------

    print("\n" + "=" * 50)
    print("SATISFICING SEARCH EXAMPLE")
    print("=" * 50)

    graph = {

        "S": [
            ("A", 2),
            ("B", 4)
        ],

        "A": [
            ("G", 8)
        ],

        "B": [
            ("G", 5)
        ],

        "G": []
    }

    acceptable_cost = 9

    path, cost = satisficing_search(
        graph,
        "S",
        "G",
        acceptable_cost
    )

    print(
        "Acceptable cost:",
        acceptable_cost
    )

    if path is not None:

        print(
            "Path:",
            " -> ".join(path)
        )

        print(
            "Cost:",
            cost
        )

    else:

        print(
            "No acceptable path found."
        )


# ---------------------------------------------------------
# Program Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
