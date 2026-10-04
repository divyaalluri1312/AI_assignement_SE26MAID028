"""
Assignment 8 - Map Coloring for Telangana Districts

The state is represented using its 33 districts.

The problem is modelled as a Constraint Satisfaction Problem (CSP):

Variable    = District
Domain      = Available colors
Constraint  = Adjacent districts must have different colors

The adjacency graph is kept directly in the program so that the
assignment can run without requiring GIS packages.

The backtracking solver uses:
- MRV (Minimum Remaining Values)
- Degree heuristic for tie-breaking
- Least-constraining value ordering
- Forward checking
"""

import matplotlib.pyplot as plt


# ================================================================
# TELANGANA DISTRICTS
# ================================================================

DISTRICTS = [
    "Adilabad",
    "Bhadradri Kothagudem",
    "Hanamkonda",
    "Hyderabad",
    "Jagtial",
    "Jangaon",
    "Jayashankar Bhupalpally",
    "Jogulamba Gadwal",
    "Kamareddy",
    "Karimnagar",
    "Khammam",
    "Kumuram Bheem",
    "Mahabubabad",
    "Mahabubnagar",
    "Mancherial",
    "Medak",
    "Medchal-Malkajgiri",
    "Mulugu",
    "Nagarkurnool",
    "Nalgonda",
    "Narayanpet",
    "Nirmal",
    "Nizamabad",
    "Peddapalli",
    "Rajanna Sircilla",
    "Rangareddy",
    "Sangareddy",
    "Siddipet",
    "Suryapet",
    "Vikarabad",
    "Wanaparthy",
    "Warangal",
    "Yadadri Bhuvanagiri",
]


# ================================================================
# DISTRICT ADJACENCY
# ================================================================

# Each pair represents two neighbouring districts.
# The graph is undirected.

EDGES = [
    ("Adilabad", "Kumuram Bheem"),
    ("Adilabad", "Nirmal"),

    ("Kumuram Bheem", "Mancherial"),
    ("Kumuram Bheem", "Nirmal"),

    ("Mancherial", "Nirmal"),
    ("Mancherial", "Jagtial"),
    ("Mancherial", "Peddapalli"),

    ("Nirmal", "Nizamabad"),
    ("Nirmal", "Jagtial"),

    ("Nizamabad", "Kamareddy"),
    ("Nizamabad", "Jagtial"),

    ("Jagtial", "Rajanna Sircilla"),
    ("Jagtial", "Karimnagar"),
    ("Jagtial", "Peddapalli"),

    ("Rajanna Sircilla", "Karimnagar"),
    ("Rajanna Sircilla", "Siddipet"),

    ("Peddapalli", "Karimnagar"),
    ("Peddapalli", "Jayashankar Bhupalpally"),

    ("Karimnagar", "Siddipet"),
    ("Karimnagar", "Jayashankar Bhupalpally"),

    ("Siddipet", "Kamareddy"),
    ("Siddipet", "Medak"),
    ("Siddipet", "Jangaon"),
    ("Siddipet", "Yadadri Bhuvanagiri"),
    ("Siddipet", "Hanamkonda"),

    ("Kamareddy", "Medak"),
    ("Kamareddy", "Sangareddy"),

    ("Medak", "Sangareddy"),

    ("Sangareddy", "Vikarabad"),
    ("Sangareddy", "Medchal-Malkajgiri"),
    ("Sangareddy", "Rangareddy"),

    ("Medchal-Malkajgiri", "Hyderabad"),
    ("Medchal-Malkajgiri", "Rangareddy"),

    ("Hyderabad", "Rangareddy"),

    ("Rangareddy", "Vikarabad"),
    ("Rangareddy", "Mahabubnagar"),
    ("Rangareddy", "Nagarkurnool"),

    ("Vikarabad", "Mahabubnagar"),
    ("Vikarabad", "Narayanpet"),

    ("Mahabubnagar", "Narayanpet"),
    ("Mahabubnagar", "Wanaparthy"),
    ("Mahabubnagar", "Nagarkurnool"),

    ("Narayanpet", "Wanaparthy"),

    ("Wanaparthy", "Jogulamba Gadwal"),
    ("Wanaparthy", "Nagarkurnool"),

    ("Jogulamba Gadwal", "Nagarkurnool"),

    ("Nagarkurnool", "Nalgonda"),

    ("Nalgonda", "Suryapet"),
    ("Nalgonda", "Yadadri Bhuvanagiri"),

    ("Suryapet", "Yadadri Bhuvanagiri"),
    ("Suryapet", "Khammam"),

    ("Khammam", "Bhadradri Kothagudem"),
    ("Khammam", "Mahabubabad"),

    ("Bhadradri Kothagudem", "Jayashankar Bhupalpally"),
    ("Bhadradri Kothagudem", "Mulugu"),

    ("Mahabubabad", "Mulugu"),
    ("Mahabubabad", "Warangal"),
    ("Mahabubabad", "Jangaon"),

    ("Mulugu", "Jayashankar Bhupalpally"),
    ("Mulugu", "Warangal"),

    ("Jayashankar Bhupalpally", "Warangal"),

    ("Warangal", "Hanamkonda"),
    ("Warangal", "Jangaon"),

    ("Hanamkonda", "Jangaon"),

    ("Jangaon", "Yadadri Bhuvanagiri"),

    ("Yadadri Bhuvanagiri", "Nalgonda"),
]


# ================================================================
# BUILD GRAPH
# ================================================================

def build_graph():
    """
    Build an undirected adjacency-list graph.
    """

    graph = {
        district: set()
        for district in DISTRICTS
    }

    for district_a, district_b in EDGES:

        if (
            district_a in graph
            and district_b in graph
        ):

            graph[district_a].add(district_b)
            graph[district_b].add(district_a)

    return graph


# ================================================================
# CONSISTENCY CHECK
# ================================================================

def is_consistent(
    district,
    color,
    assignment,
    graph
):
    """
    Check whether assigning a color to a district violates
    any already assigned neighbouring district.
    """

    return all(
        assignment.get(neighbour) != color
        for neighbour in graph[district]
    )


# ================================================================
# MRV + DEGREE HEURISTIC
# ================================================================

def select_unassigned(
    assignment,
    domains,
    graph
):
    """
    Select the next district using:

    1. MRV - Minimum Remaining Values
    2. Degree heuristic for tie-breaking
    """

    remaining = [
        district
        for district in DISTRICTS
        if district not in assignment
    ]

    return min(
        remaining,
        key=lambda district: (
            len(domains[district]),
            -len(graph[district])
        )
    )


# ================================================================
# BACKTRACKING CSP SOLVER
# ================================================================

def backtrack(
    assignment,
    domains,
    graph
):
    """
    Backtracking search with forward checking.
    """

    # All districts have been assigned.
    if len(assignment) == len(DISTRICTS):
        return assignment.copy()

    # Select the next district.
    district = select_unassigned(
        assignment,
        domains,
        graph
    )

    # ------------------------------------------------------------
    # Least-constraining value ordering
    # ------------------------------------------------------------

    ordered_colors = sorted(
        domains[district],
        key=lambda color: sum(
            color in domains[neighbour]
            for neighbour in graph[district]
            if neighbour not in assignment
        )
    )

    # Try each possible color.
    for color in ordered_colors:

        if not is_consistent(
            district,
            color,
            assignment,
            graph
        ):
            continue

        assignment[district] = color

        removed = []
        valid = True

        # --------------------------------------------------------
        # Forward checking
        # --------------------------------------------------------

        for neighbour in graph[district]:

            if (
                neighbour not in assignment
                and color in domains[neighbour]
            ):

                domains[neighbour].remove(color)

                removed.append(neighbour)

                # Empty domain means this assignment is invalid.
                if not domains[neighbour]:

                    valid = False
                    break

        # Continue recursively if valid.
        if valid:

            result = backtrack(
                assignment,
                domains,
                graph
            )

            if result:
                return result

        # --------------------------------------------------------
        # Undo domain changes.
        # --------------------------------------------------------

        for neighbour in removed:
            domains[neighbour].add(color)

        del assignment[district]

    return None


# ================================================================
# SOLVE MAP COLORING
# ================================================================

def solve(num_colors):
    """
    Solve the Telangana map-coloring problem using the specified
    number of colors.
    """

    graph = build_graph()

    colors = [
        f"Color {i}"
        for i in range(1, num_colors + 1)
    ]

    domains = {
        district: set(colors)
        for district in DISTRICTS
    }

    return backtrack(
        {},
        domains,
        graph
    )


# ================================================================
# VALIDATE SOLUTION
# ================================================================

def validate_solution(assignment):
    """
    Check every adjacency constraint and return the conflicting
    edges, if any.
    """

    conflicts = []

    for district_a, district_b in EDGES:

        if assignment.get(district_a) == assignment.get(
            district_b
        ):

            conflicts.append(
                (district_a, district_b)
            )

    return conflicts


# ================================================================
# DISPLAY SOLUTION
# ================================================================

def show_solution(assignment):
    """
    Print the assigned color for every district and validate it.
    """

    print("\nTELANGANA MAP COLORING")
    print("-" * 35)

    for district in sorted(assignment):

        print(
            f"{district:30} "
            f"{assignment[district]}"
        )

    conflicts = validate_solution(
        assignment
    )

    print("\nConflicts:", len(conflicts))

    print(
        "Valid coloring:",
        len(conflicts) == 0
    )


# ================================================================
# GRAPH VISUALIZATION
# ================================================================

def plot_graph(assignment):
    """
    Display the Telangana district adjacency graph.

    This is an adjacency graph rather than a geographical GIS map.
    """

    graph = build_graph()

    # Approximate positions used only for readable visualization.
    positions = {
        "Adilabad": (1, 9),
        "Kumuram Bheem": (3, 9),
        "Mancherial": (5, 8),
        "Nirmal": (2, 7),
        "Nizamabad": (0, 6),
        "Jagtial": (4, 6),
        "Peddapalli": (6, 6),
        "Rajanna Sircilla": (5, 5),
        "Karimnagar": (7, 5),
        "Kamareddy": (1, 5),
        "Siddipet": (5, 4),
        "Sangareddy": (-1, 4),
        "Medak": (1, 4),
        "Jayashankar Bhupalpally": (8, 4),
        "Hanamkonda": (9, 3),
        "Warangal": (10, 2),
        "Mulugu": (9, 1),
        "Jangaon": (7, 2),
        "Yadadri Bhuvanagiri": (6, 2),
        "Nalgonda": (5, 1),
        "Suryapet": (7, 0),
        "Khammam": (9, 0),
        "Bhadradri Kothagudem": (11, 0),
        "Medchal-Malkajgiri": (0, 3),
        "Hyderabad": (1, 2),
        "Rangareddy": (1, 1),
        "Vikarabad": (-1, 1),
        "Mahabubnagar": (0, 0),
        "Narayanpet": (-2, 0),
        "Wanaparthy": (-1, -1),
        "Jogulamba Gadwal": (-2, -2),
        "Nagarkurnool": (2, -1),
    }

    # Ensure every district has a position.
    for index, district in enumerate(DISTRICTS):

        positions.setdefault(
            district,
            (
                index % 6,
                -(index // 6)
            )
        )

    plt.figure(
        figsize=(13, 9)
    )

    # ------------------------------------------------------------
    # Draw adjacency edges.
    # ------------------------------------------------------------

    for district_a, district_b in EDGES:

        xa, ya = positions[district_a]
        xb, yb = positions[district_b]

        plt.plot(
            [xa, xb],
            [ya, yb],
            linewidth=0.7,
            alpha=0.35
        )

    # ------------------------------------------------------------
    # Generate colors dynamically.
    # ------------------------------------------------------------

    color_names = sorted(
        set(assignment.values()),
        key=lambda name: int(
            name.split()[-1]
        )
    )

    available_colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]

    palette = {
        color_name: available_colors[index]
        for index, color_name in enumerate(
            color_names
        )
    }

    # ------------------------------------------------------------
    # Draw districts.
    # ------------------------------------------------------------

    for district, (x, y) in positions.items():

        plt.scatter(
            [x],
            [y],
            s=550,
            color=palette[assignment[district]],
            edgecolors="black"
        )

        short_name = district.replace(
            " ",
            "\n"
        )

        plt.text(
            x,
            y,
            short_name,
            ha="center",
            va="center",
            fontsize=6
        )

    plt.title(
        "Telangana District Map-Coloring "
        "as an Adjacency Graph"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.show()


# ================================================================
# MAIN PROGRAM
# ================================================================

def main():

    print(
        "TELANGANA DISTRICT MAP COLORING"
    )

    print(
        "Current district count:",
        len(DISTRICTS)
    )

    choice = input(
        "Number of colors [4]: "
    ).strip()

    if choice == "":
        choice = "4"

    try:

        num_colors = int(choice)

        if num_colors < 1:
            raise ValueError

    except ValueError:

        print(
            "Please enter a positive integer."
        )

        return

    assignment = solve(
        num_colors
    )

    if assignment is None:

        print(
            f"\nNo solution found using "
            f"{num_colors} colors."
        )

        return

    show_solution(
        assignment
    )

    show_plot = input(
        "\nShow adjacency graph? (y/n) [y]: "
    ).strip().lower()

    if show_plot != "n":

        plot_graph(
            assignment
        )


# ================================================================
# PROGRAM ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()
