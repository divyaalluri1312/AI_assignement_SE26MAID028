"""
Assignment 8 - Map Coloring for Telangana Districts

The state has 33 districts. The problem is modelled as a CSP:
- Variable  = district
- Domain    = available colors
- Constraint = neighbouring districts must have different colors

The adjacency list is based on the district map and is intentionally
kept in the program so the assignment can run without GIS packages.
The backtracking solver uses the MRV and degree heuristics.
"""

import matplotlib.pyplot as plt

DISTRICTS = [
    "Adilabad", "Bhadradri Kothagudem", "Hanamkonda", "Hyderabad",
    "Jagtial", "Jangaon", "Jayashankar Bhupalpally",
    "Jogulamba Gadwal", "Kamareddy", "Karimnagar", "Khammam",
    "Kumuram Bheem", "Mahabubabad", "Mahabubnagar", "Mancherial",
    "Medak", "Medchal-Malkajgiri", "Mulugu", "Nagarkurnool",
    "Nalgonda", "Narayanpet", "Nirmal", "Nizamabad", "Peddapalli",
    "Rajanna Sircilla", "Rangareddy", "Sangareddy", "Siddipet",
    "Suryapet", "Vikarabad", "Wanaparthy", "Warangal",
    "Yadadri Bhuvanagiri",
]

# Adjacent districts are represented as an undirected graph.
# The list follows the current 33-district arrangement.
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
    ("Medak", "Kamareddy"),
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


def build_graph():
    graph = {district: set() for district in DISTRICTS}

    for a, b in EDGES:
        if a in graph and b in graph:
            graph[a].add(b)
            graph[b].add(a)

    return graph


def is_consistent(district, color, assignment, graph):
    return all(
        assignment.get(neighbour) != color
        for neighbour in graph[district]
    )


def select_unassigned(assignment, domains, graph):
    remaining = [d for d in DISTRICTS if d not in assignment]

    # MRV: choose the variable with the smallest remaining legal domain.
    # Degree breaks ties.
    return min(
        remaining,
        key=lambda d: (
            len(domains[d]),
            -len(graph[d]),
        ),
    )


def backtrack(assignment, domains, graph):
    if len(assignment) == len(DISTRICTS):
        return assignment.copy()

    district = select_unassigned(assignment, domains, graph)

    # Least-constraining values first.
    ordered_colors = sorted(
        domains[district],
        key=lambda color: sum(
            color in domains[n]
            for n in graph[district]
            if n not in assignment
        ),
    )

    for color in ordered_colors:
        if not is_consistent(district, color, assignment, graph):
            continue

        assignment[district] = color

        removed = []
        valid = True

        for neighbour in graph[district]:
            if neighbour not in assignment and color in domains[neighbour]:
                domains[neighbour].remove(color)
                removed.append(neighbour)

                if not domains[neighbour]:
                    valid = False
                    break

        if valid:
            result = backtrack(assignment, domains, graph)
            if result:
                return result

        for neighbour in removed:
            domains[neighbour].add(color)

        del assignment[district]

    return None


def solve(num_colors):
    graph = build_graph()
    colors = [f"Color {i}" for i in range(1, num_colors + 1)]
    domains = {district: set(colors) for district in DISTRICTS}

    return backtrack({}, domains, graph)


def show_solution(assignment):
    print("\nTELANGANA MAP COLORING")
    print("-" * 35)

    for district in sorted(assignment):
        print(f"{district:28} {assignment[district]}")

    graph = build_graph()
    conflicts = []

    for a, b in EDGES:
        if assignment[a] == assignment[b]:
            conflicts.append((a, b))

    print("\nConflicts:", len(conflicts))
    print("Valid coloring:", len(conflicts) == 0)


def plot_graph(assignment):
    # This is a district-adjacency graph rather than a geographic GIS map.
    # The colours still show the CSP solution clearly.
    graph = build_graph()

    # Hand-positioned approximate layout for readability.
    positions = {
        "Adilabad": (1, 9), "Kumuram Bheem": (3, 9),
        "Mancherial": (5, 8), "Nirmal": (2, 7),
        "Nizamabad": (0, 6), "Jagtial": (4, 6),
        "Peddapalli": (6, 6), "Rajanna Sircilla": (5, 5),
        "Karimnagar": (7, 5), "Kamareddy": (1, 5),
        "Siddipet": (5, 4), "Sangareddy": (-1, 4),
        "Medak": (1, 4), "Jayashankar Bhupalpally": (8, 4),
        "Hanamkonda": (9, 3), "Warangal": (10, 2),
        "Mulugu": (9, 1), "Jangaon": (7, 2),
        "Yadadri Bhuvanagiri": (6, 2), "Nalgonda": (5, 1),
        "Suryapet": (7, 0), "Khammam": (9, 0),
        "Bhadradri Kothagudem": (11, 0),
        "Medchal-Malkajgiri": (0, 3), "Hyderabad": (1, 2),
        "Rangareddy": (1, 1), "Vikarabad": (-1, 1),
        "Mahabubnagar": (0, 0), "Narayanpet": (-2, 0),
        "Wanaparthy": (-1, -1), "Jogulamba Gadwal": (-2, -2),
        "Nagarkurnool": (2, -1),
    }

    # If a district has no hand position, place it at the origin.
    for i, district in enumerate(DISTRICTS):
        positions.setdefault(district, (i % 6, -(i // 6)))

    plt.figure(figsize=(13, 9))

    for a, b in EDGES:
        xa, ya = positions[a]
        xb, yb = positions[b]
        plt.plot([xa, xb], [ya, yb], linewidth=0.7, alpha=0.35)

    palette = {
        "Color 1": "tab:blue",
        "Color 2": "tab:orange",
        "Color 3": "tab:green",
        "Color 4": "tab:red",
        "Color 5": "tab:purple",
    }

    for district, (x, y) in positions.items():
        plt.scatter(
            [x], [y], s=550,
            color=palette[assignment[district]],
            edgecolors="black",
        )
        short_name = district.replace(" ", "\n")
        plt.text(x, y, short_name, ha="center", va="center", fontsize=6)

    plt.title("Telangana District Map-Coloring as an Adjacency Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    print("TELANGANA DISTRICT MAP COLORING")
    print("Current district count:", len(DISTRICTS))

    choice = input("Number of colors [4]: ").strip() or "4"

    try:
        num_colors = int(choice)
        if num_colors < 1:
            raise ValueError
    except ValueError:
        print("Please enter a positive integer.")
        return

    assignment = solve(num_colors)

    if assignment is None:
        print(f"\nNo solution found using {num_colors} colors.")
        return

    show_solution(assignment)

    show_plot = input("\nShow adjacency graph? (y/n) [y]: ").strip().lower()
    if show_plot != "n":
        plot_graph(assignment)


if __name__ == "__main__":
    main()
