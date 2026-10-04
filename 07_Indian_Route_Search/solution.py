"""
Assignment 7 - Search Route on an Indian City Map

The original question uses the Romania map (Arad -> Bucharest).
The assignment note says to implement the same idea using Indian
routes, so this program uses Indian cities instead.

A* is used with a straight-line heuristic calculated from city
coordinates. The heuristic is only used as a lower-bound estimate;
the route cost itself comes from the road graph.
"""

import heapq
import math

# Approximate city coordinates (latitude, longitude).
COORDS = {
    "Delhi": (28.6139, 77.2090),
    "Jaipur": (26.9124, 75.7873),
    "Ahmedabad": (23.0225, 72.5714),
    "Mumbai": (19.0760, 72.8777),
    "Pune": (18.5204, 73.8567),
    "Hyderabad": (17.3850, 78.4867),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Bhubaneswar": (20.2961, 85.8245),
    "Visakhapatnam": (17.6868, 83.2185),
    "Lucknow": (26.8467, 80.9462),
    "Kanpur": (26.4499, 80.3319),
    "Varanasi": (25.3176, 82.9739),
    "Patna": (25.5941, 85.1376),
    "Kochi": (9.9312, 76.2673),
    "Goa": (15.4909, 73.8278),
}

# Road graph. Each edge is an approximate road distance in km.
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
    ("Kolkata", "Bhubaneswar", 442),
    ("Kolkata", "Patna", 554),
    ("Kolkata", "Varanasi", 680),
    ("Bhubaneswar", "Visakhapatnam", 444),
    ("Bhubaneswar", "Patna", 831),
    ("Visakhapatnam", "Kolkata", 882),
    ("Patna", "Varanasi", 256),
    ("Varanasi", "Kanpur", 328),
    ("Kanpur", "Lucknow", 115),
]


def build_graph():
    graph = {city: [] for city in COORDS}

    for a, b, distance in ROADS:
        if a not in graph:
            graph[a] = []
        if b not in graph:
            graph[b] = []

        graph[a].append((b, distance))
        graph[b].append((a, distance))

    return graph


def haversine(a, b):
    """Straight-line distance between two coordinate points in km."""
    lat1, lon1 = map(math.radians, COORDS[a])
    lat2, lon2 = map(math.radians, COORDS[b])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    x = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )

    return 6371.0 * 2 * math.asin(math.sqrt(x))


def a_star(graph, start, goal):
    queue = [(haversine(start, goal), 0, start)]
    g = {start: 0}
    parent = {start: None}
    expanded = 0

    while queue:
        _, current_cost, current = heapq.heappop(queue)

        if current_cost != g[current]:
            continue

        expanded += 1

        if current == goal:
            path = []
            node = goal
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path, current_cost, expanded

        for nxt, road_cost in graph[current]:
            new_cost = current_cost + road_cost

            if new_cost < g.get(nxt, float("inf")):
                g[nxt] = new_cost
                parent[nxt] = current

                f = new_cost + haversine(nxt, goal)
                heapq.heappush(queue, (f, new_cost, nxt))

    return None, None, expanded


def main():
    graph = build_graph()
    cities = sorted(graph)

    print("INDIAN ROUTE SEARCH USING A*")
    print("=" * 35)
    print("Available cities:")
    print(", ".join(cities))

    start = input("\nStart city: ").strip()
    goal = input("Goal city: ").strip()

    lookup = {city.lower(): city for city in cities}
    start = lookup.get(start.lower())
    goal = lookup.get(goal.lower())

    if not start or not goal:
        print("City not found.")
        return

    path, distance, expanded = a_star(graph, start, goal)

    if path is None:
        print("No route found.")
        return

    print("\nA* result")
    print("-" * 25)
    print("Route:", " -> ".join(path))
    print(f"Total road distance: {distance} km")
    print("Nodes expanded:", expanded)
    print("Heuristic at start:", round(haversine(start, goal), 2), "km")


if __name__ == "__main__":
    main()
