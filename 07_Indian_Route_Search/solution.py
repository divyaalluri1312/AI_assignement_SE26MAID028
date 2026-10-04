"""
Assignment 7 - Search Route on an Indian City Map

The original problem uses the Romania map (Arad -> Bucharest).
This implementation adapts the same idea to an Indian city road
network.

A* search is used to find a shortest route between two Indian cities.

The heuristic is based on the straight-line geographical distance
between cities. It is scaled using the road network so that it
remains a lower-bound estimate of the actual road distance.
"""

import heapq
import math


# ================================================================
# APPROXIMATE CITY COORDINATES
# ================================================================

# Coordinates are given as:
# (latitude, longitude)

COORDS = {
    "Delhi": (28.6139, 77.2090),
    "Jaipur": (26.9124, 75.7873),
    "Ahmedabad": (23.0225, 72.5714),
    "Udaipur": (24.5854, 73.7125),

    "Mumbai": (19.0760, 72.8777),
    "Pune": (18.5204, 73.8567),

    "Hyderabad": (17.3850, 78.4867),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),

    "Kolkata": (22.5726, 88.3639),
    "Bhuvaneswar": (20.2961, 85.8245),
    "Visakhapatnam": (17.6868, 83.2185),

    "Lucknow": (26.8467, 80.9462),
    "Kanpur": (26.4499, 80.3319),
    "Varanasi": (25.3176, 82.9739),
    "Patna": (25.5941, 85.1376),

    "Kochi": (9.9312, 76.2673),
    "Thiruvananthapuram": (8.5241, 76.9366),

    "Goa": (15.4909, 73.8278),
}


# ================================================================
# INDIAN ROAD NETWORK
# ================================================================

# Each tuple represents:
# (City A, City B, Approximate road distance in km)

ROADS = [
    ("Delhi", "Jaipur", 307),
    ("Delhi", "Lucknow", 548),
    ("Delhi", "Ahmedabad", 911),
    ("Delhi", "Mumbai", 1452),
    ("Delhi", "Hyderabad", 1582),

    ("Jaipur", "Ahmedabad", 660),
    ("Jaipur", "Udaipur", 397),

    ("Ahmedabad", "Mumbai", 526),
    ("Ahmedabad", "Pune", 663),

    ("Mumbai", "Pune", 148),
    ("Mumbai", "Hyderabad", 708),
    ("Mumbai", "Goa", 585),

    ("Pune", "Hyderabad", 562),
    ("Pune", "Bengaluru", 839),

    ("Hyderabad", "Bengaluru", 576),
    ("Hyderabad", "Chennai", 626),
    ("Hyderabad", "Visakhapatnam", 618),

    ("Bengaluru", "Chennai", 346),
    ("Bengaluru", "Goa", 562),
    ("Bengaluru", "Kochi", 547),

    ("Chennai", "Kochi", 690),
    ("Chennai", "Kolkata", 1666),

    ("Kochi", "Goa", 755),
    ("Kochi", "Thiruvananthapuram", 206),

    ("Kolkata", "Bhuvaneswar", 442),
    ("Kolkata", "Patna", 554),
    ("Kolkata", "Varanasi", 680),

    ("Bhuvaneswar", "Visakhapatnam", 444),
    ("Bhuvaneswar", "Patna", 831),

    ("Visakhapatnam", "Kolkata", 882),

    ("Patna", "Varanasi", 256),

    ("Varanasi", "Kanpur", 328),
    ("Kanpur", "Lucknow", 115),
]


# ================================================================
# BUILD GRAPH
# ================================================================

def build_graph():
    """
    Convert the road list into an undirected adjacency-list graph.
    """

    graph = {
        city: []
        for city in COORDS
    }

    for city_a, city_b, distance in ROADS:

        graph[city_a].append(
            (city_b, distance)
        )

        graph[city_b].append(
            (city_a, distance)
        )

    return graph


# ================================================================
# HAVERSINE DISTANCE
# ================================================================

def haversine(city_a, city_b):
    """
    Calculate the straight-line geographical distance between
    two cities using the Haversine formula.

    Returns distance in kilometres.
    """

    lat1, lon1 = map(
        math.radians,
        COORDS[city_a]
    )

    lat2, lon2 = map(
        math.radians,
        COORDS[city_b]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    value = (
        math.sin(dlat / 2) ** 2
        +
        math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return (
        6371.0
        * 2
        * math.asin(math.sqrt(value))
    )


# ================================================================
# HEURISTIC SCALE
# ================================================================

def calculate_heuristic_scale():
    """
    Calculate a conservative scale factor for the geographical
    heuristic.

    For each road edge, the ratio:

        road distance / straight-line distance

    is calculated.

    The minimum ratio is used so that the heuristic remains a
    lower-bound estimate of the actual road distance.
    """

    ratios = []

    for city_a, city_b, road_distance in ROADS:

        straight_distance = haversine(
            city_a,
            city_b
        )

        if straight_distance > 0:

            ratio = (
                road_distance
                / straight_distance
            )

            ratios.append(ratio)

    if not ratios:
        return 1.0

    return min(ratios)


HEURISTIC_SCALE = calculate_heuristic_scale()


# ================================================================
# A* HEURISTIC
# ================================================================

def heuristic(city, goal):
    """
    Estimate the remaining road distance from city to goal.

    The Haversine distance is multiplied by a conservative scale
    factor to maintain a lower-bound estimate.
    """

    return (
        HEURISTIC_SCALE
        * haversine(city, goal)
    )


# ================================================================
# A* SEARCH
# ================================================================

def a_star(graph, start, goal):
    """
    Find a shortest route from start to goal using A* search.

    Returns:
        path      : list of cities in the route
        distance  : total road distance
        expanded  : number of expanded nodes
    """

    initial_h = heuristic(
        start,
        goal
    )

    priority_queue = [
        (
            initial_h,
            0,
            start
        )
    ]

    g_cost = {
        start: 0
    }

    parent = {
        start: None
    }

    expanded = 0

    while priority_queue:

        _, current_cost, current = heapq.heappop(
            priority_queue
        )

        # Ignore outdated queue entries.
        if current_cost != g_cost[current]:
            continue

        expanded += 1

        # Goal reached.
        if current == goal:

            path = []

            node = goal

            while node is not None:

                path.append(node)

                node = parent[node]

            path.reverse()

            return (
                path,
                current_cost,
                expanded
            )

        # Explore neighbouring cities.
        for neighbour, road_cost in graph[current]:

            new_cost = (
                current_cost
                + road_cost
            )

            if new_cost < g_cost.get(
                neighbour,
                float("inf")
            ):

                g_cost[neighbour] = new_cost

                parent[neighbour] = current

                f_cost = (
                    new_cost
                    + heuristic(
                        neighbour,
                        goal
                    )
                )

                heapq.heappush(
                    priority_queue,
                    (
                        f_cost,
                        new_cost,
                        neighbour
                    )
                )

    return (
        None,
        None,
        expanded
    )


# ================================================================
# MAIN PROGRAM
# ================================================================

def main():

    graph = build_graph()

    cities = sorted(graph)

    print("INDIAN ROUTE SEARCH USING A*")
    print("=" * 35)

    print("Available cities:")
    print(", ".join(cities))

    start_input = input(
        "\nStart city: "
    ).strip()

    goal_input = input(
        "Goal city: "
    ).strip()

    # Case-insensitive lookup.
    city_lookup = {
        city.lower(): city
        for city in cities
    }

    start = city_lookup.get(
        start_input.lower()
    )

    goal = city_lookup.get(
        goal_input.lower()
    )

    # Validate source city.
    if start is None:

        print(
            f"\nInvalid start city: {start_input}"
        )

        return

    # Validate destination city.
    if goal is None:

        print(
            f"\nInvalid goal city: {goal_input}"
        )

        return

    # Run A*.
    path, distance, expanded = a_star(
        graph,
        start,
        goal
    )

    if path is None:

        print("\nNo route found.")

        return

    print("\nA* Result")
    print("-" * 25)

    print(
        "Route:",
        " -> ".join(path)
    )

    print(
        f"Total road distance: {distance} km"
    )

    print(
        "Nodes expanded:",
        expanded
    )

    print(
        "Heuristic at start:",
        round(
            heuristic(start, goal),
            2
        ),
        "km"
    )

    print(
        "Heuristic scale:",
        round(
            HEURISTIC_SCALE,
            4
        )
    )


# ================================================================
# PROGRAM ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()
