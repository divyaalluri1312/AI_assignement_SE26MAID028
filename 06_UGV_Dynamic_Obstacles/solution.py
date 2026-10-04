"""
Assignment 6 - UGV Navigation with Dynamic Obstacles

The UGV does not initially know the complete obstacle map.
Obstacles can move during execution.

At each step:
1. The UGV senses nearby obstacles.
2. It plans a shortest path using A* and its current knowledge.
3. The environment changes dynamically.
4. If the next planned cell becomes blocked, the UGV detects
   the unexpected obstacle and replans.
5. Otherwise, the UGV moves one step.

This demonstrates online path planning and replanning in
a partially known dynamic environment.
"""

import heapq
import random
import matplotlib.pyplot as plt


# ================================================================
# PARAMETERS
# ================================================================

SIZE = 50

SENSE_RADIUS = 3

INITIAL_DENSITY = 0.12

DYNAMIC_MOVE_PROBABILITY = 0.25

MAX_STEPS = 300

# Fixed seed makes the experiment reproducible.
RANDOM_SEED = 42


# 8-connected movement
MOVES = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1)
]


# ================================================================
# HEURISTIC
# ================================================================

def heuristic(current, goal):
    """
    Chebyshev distance heuristic for an 8-connected grid.
    """

    return max(
        abs(current[0] - goal[0]),
        abs(current[1] - goal[1])
    )


# ================================================================
# GRID CHECK
# ================================================================

def inside(position):
    """
    Check whether a grid position is inside the environment.
    """

    row, col = position

    return (
        0 <= row < SIZE
        and 0 <= col < SIZE
    )


# ================================================================
# A* SEARCH
# ================================================================

def astar(blocked, start, goal):
    """
    Find the shortest currently known path using A*.

    The search uses only the obstacles currently known
    to the UGV.
    """

    priority_queue = [
        (heuristic(start, goal), 0, start)
    ]

    g_cost = {
        start: 0
    }

    parent = {
        start: None
    }

    while priority_queue:

        _, current_cost, current = heapq.heappop(
            priority_queue
        )

        # Ignore outdated queue entries.
        if current_cost != g_cost.get(current):
            continue

        # Goal reached.
        if current == goal:

            path = []

            node = goal

            while node is not None:

                path.append(node)

                node = parent[node]

            path.reverse()

            return path

        # Explore neighbours.
        for dr, dc in MOVES:

            next_cell = (
                current[0] + dr,
                current[1] + dc
            )

            if not inside(next_cell):
                continue

            if next_cell in blocked:
                continue

            new_cost = current_cost + 1

            if new_cost < g_cost.get(
                next_cell,
                float("inf")
            ):

                g_cost[next_cell] = new_cost

                parent[next_cell] = current

                f_cost = (
                    new_cost
                    + heuristic(next_cell, goal)
                )

                heapq.heappush(
                    priority_queue,
                    (
                        f_cost,
                        new_cost,
                        next_cell
                    )
                )

    return None


# ================================================================
# INITIAL ENVIRONMENT
# ================================================================

def generate_environment(seed):
    """
    Generate the actual environment containing obstacles.
    """

    random.seed(seed)

    blocked = set()

    for row in range(SIZE):

        for col in range(SIZE):

            if random.random() < INITIAL_DENSITY:

                blocked.add((row, col))

    return blocked


# ================================================================
# DYNAMIC OBSTACLE MOVEMENT
# ================================================================

def move_dynamic_obstacles(
    actual_blocked,
    start,
    goal
):
    """
    Move some obstacles to neighbouring cells.

    The UGV does not know about these changes until it
    senses the environment again.
    """

    updated = set()

    for obstacle in actual_blocked:

        # Never place an obstacle on the UGV
        # or on the destination.
        if obstacle in (start, goal):
            continue

        if random.random() < DYNAMIC_MOVE_PROBABILITY:

            possible_positions = []

            for dr, dc in MOVES:

                candidate = (
                    obstacle[0] + dr,
                    obstacle[1] + dc
                )

                if inside(candidate):

                    if candidate not in (
                        start,
                        goal
                    ):
                        possible_positions.append(
                            candidate
                        )

            if possible_positions:

                new_position = random.choice(
                    possible_positions
                )

                updated.add(new_position)

            else:

                updated.add(obstacle)

        else:

            updated.add(obstacle)

    return updated


# ================================================================
# OBSTACLE SENSING
# ================================================================

def sense_obstacles(
    actual_blocked,
    position
):
    """
    Detect obstacles within the UGV's sensing radius.
    """

    detected = set()

    for obstacle in actual_blocked:

        row_distance = abs(
            obstacle[0] - position[0]
        )

        col_distance = abs(
            obstacle[1] - position[1]
        )

        if (
            row_distance <= SENSE_RADIUS
            and col_distance <= SENSE_RADIUS
        ):

            detected.add(obstacle)

    return detected


# ================================================================
# VISUALIZATION
# ================================================================

def plot_environment(
    actual_obstacles,
    known_obstacles,
    trajectory,
    current_path,
    start,
    goal,
    current
):
    """
    Display the actual environment, known obstacles,
    travelled trajectory and current planned path.
    """

    plt.figure(figsize=(9, 9))

    # ------------------------------------------------------------
    # Actual obstacles
    # ------------------------------------------------------------

    if actual_obstacles:

        x_actual = [
            col
            for row, col in actual_obstacles
        ]

        y_actual = [
            SIZE - 1 - row
            for row, col in actual_obstacles
        ]

        plt.scatter(
            x_actual,
            y_actual,
            s=12,
            marker="s",
            alpha=0.25,
            label="Actual obstacles"
        )

    # ------------------------------------------------------------
    # Obstacles known to UGV
    # ------------------------------------------------------------

    if known_obstacles:

        x_known = [
            col
            for row, col in known_obstacles
        ]

        y_known = [
            SIZE - 1 - row
            for row, col in known_obstacles
        ]

        plt.scatter(
            x_known,
            y_known,
            s=20,
            marker="s",
            label="Known obstacles"
        )

    # ------------------------------------------------------------
    # UGV travelled trajectory
    # ------------------------------------------------------------

    if len(trajectory) > 1:

        x_trajectory = [
            col
            for row, col in trajectory
        ]

        y_trajectory = [
            SIZE - 1 - row
            for row, col in trajectory
        ]

        plt.plot(
            x_trajectory,
            y_trajectory,
            linewidth=2,
            label="UGV trajectory"
        )

    # ------------------------------------------------------------
    # Current planned path
    # ------------------------------------------------------------

    if current_path:

        x_path = [
            col
            for row, col in current_path
        ]

        y_path = [
            SIZE - 1 - row
            for row, col in current_path
        ]

        plt.plot(
            x_path,
            y_path,
            linestyle="--",
            linewidth=1.5,
            label="Current planned path"
        )

    # ------------------------------------------------------------
    # Start
    # ------------------------------------------------------------

    plt.scatter(
        [start[1]],
        [SIZE - 1 - start[0]],
        s=100,
        label="Start"
    )

    # ------------------------------------------------------------
    # Goal
    # ------------------------------------------------------------

    plt.scatter(
        [goal[1]],
        [SIZE - 1 - goal[0]],
        s=130,
        marker="*",
        label="Goal"
    )

    # ------------------------------------------------------------
    # Current UGV position
    # ------------------------------------------------------------

    plt.scatter(
        [current[1]],
        [SIZE - 1 - current[0]],
        s=80,
        marker="o",
        label="UGV"
    )

    plt.title(
        "UGV Dynamic Obstacle Navigation"
    )

    plt.xlabel("X")

    plt.ylabel("Y")

    plt.grid(alpha=0.2)

    plt.legend()

    plt.tight_layout()

    plt.show()


# ================================================================
# MAIN SIMULATION
# ================================================================

def main():

    # ------------------------------------------------------------
    # Start and goal
    # ------------------------------------------------------------

    start = (2, 2)

    goal = (
        SIZE - 3,
        SIZE - 3
    )

    # ------------------------------------------------------------
    # Generate actual environment
    # ------------------------------------------------------------

    actual_obstacles = generate_environment(
        RANDOM_SEED
    )

    actual_obstacles.discard(start)

    actual_obstacles.discard(goal)

    # ------------------------------------------------------------
    # UGV initially knows nothing about obstacles.
    # ------------------------------------------------------------

    known_obstacles = set()

    current = start

    travelled = 0

    replanning_count = 0

    unexpected_blocks = 0

    trajectory = [current]

    current_path = None

    # ============================================================
    # NAVIGATION LOOP
    # ============================================================

    for step in range(MAX_STEPS):

        # --------------------------------------------------------
        # Check whether destination has been reached.
        # --------------------------------------------------------

        if current == goal:

            break

        # --------------------------------------------------------
        # Sense nearby obstacles.
        # --------------------------------------------------------

        newly_detected = sense_obstacles(
            actual_obstacles,
            current
        )

        known_obstacles.update(
            newly_detected
        )

        # --------------------------------------------------------
        # Plan using currently known information.
        # --------------------------------------------------------

        current_path = astar(
            known_obstacles,
            current,
            goal
        )

        replanning_count += 1

        # --------------------------------------------------------
        # No currently known path.
        # --------------------------------------------------------

        if not current_path:

            actual_obstacles = move_dynamic_obstacles(
                actual_obstacles,
                current,
                goal
            )

            continue

        # --------------------------------------------------------
        # Determine next movement.
        # --------------------------------------------------------

        if len(current_path) > 1:

            next_position = current_path[1]

        else:

            next_position = current

        # --------------------------------------------------------
        # IMPORTANT:
        # Dynamic environment changes AFTER planning but
        # BEFORE movement.
        # --------------------------------------------------------

        actual_obstacles = move_dynamic_obstacles(
            actual_obstacles,
            current,
            goal
        )

        # --------------------------------------------------------
        # Check whether the planned next cell became blocked.
        # --------------------------------------------------------

        if next_position in actual_obstacles:

            # Newly discovered dynamic obstacle.
            known_obstacles.add(
                next_position
            )

            unexpected_blocks += 1

            # Stay in current position.
            # The next iteration will replan.
            continue

        # --------------------------------------------------------
        # Move UGV.
        # --------------------------------------------------------

        current = next_position

        travelled += 1

        trajectory.append(current)

    # ============================================================
    # FINAL RESULTS
    # ============================================================

    success = (
        current == goal
    )

    # Calculate final known path if possible.
    if success:

        final_path = [current]

    else:

        final_path = astar(
            known_obstacles,
            current,
            goal
        )

    print("\nDYNAMIC UGV RESULTS")
    print("-" * 40)

    print(
        "Goal reached       :",
        success
    )

    print(
        "Steps travelled    :",
        travelled
    )

    print(
        "Replanning count   :",
        replanning_count
    )

    print(
        "Unexpected blocks  :",
        unexpected_blocks
    )

    print(
        "Final position     :",
        current
    )

    print(
        "Known obstacles    :",
        len(known_obstacles)
    )

    print(
        "Actual obstacles   :",
        len(actual_obstacles)
    )

    print(
        "Trajectory cells   :",
        len(trajectory)
    )

    # ------------------------------------------------------------
    # Plot final environment
    # ------------------------------------------------------------

    plot_environment(
        actual_obstacles,
        known_obstacles,
        trajectory,
        final_path,
        start,
        goal,
        current
    )


# ================================================================
# PROGRAM ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()
