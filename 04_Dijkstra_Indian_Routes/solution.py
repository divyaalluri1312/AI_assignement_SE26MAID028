"""
Assignment 4 - Dijkstra's Algorithm for Indian Routes

The graph below contains a practical set of major Indian cities and
road-distance edges. Distances are in kilometres.

The source data is based on an openly published Indian-city road
distance dataset. The assignment asks for Indian routes; this file
keeps the graph directly inside the program so it can run without
requiring a separate database.

To extend the graph, add more city-to-city edges to ROUTES.
"""

import heapq

# (city1, city2, road distance in km)
ROUTES = [
    ("Agra", "Delhi", 240),
    ("Agra", "Lucknow", 334),
    ("Agra", "Kanpur", 277),
    ("Ahmedabad", "Mumbai", 526),
    ("Ahmedabad", "Pune", 663),
    ("Ahmedabad", "Jaipur", 660),
    ("Ahmedabad", "Udaipur", 258),
    ("Bengaluru", "Pune", 839),
    ("Bengaluru", "Hyderabad", 576),
    ("Bengaluru", "Chennai", 346),
    ("Bengaluru", "Goa", 562),
    ("Bhubaneswar", "Kolkata", 442),
    ("Bhubaneswar", "Vishakhapatnam", 444),
    ("Bhubaneswar", "Patna", 831),
    ("Chennai", "Hyderabad", 626),
    ("Chennai", "Kochi", 690),
    ("Chennai", "Mumbai", 1335),
    ("Chennai", "Kolkata", 1666),
    ("Delhi", "Jaipur", 307),
    ("Delhi", "Lucknow", 548),
    ("Delhi", "Bengaluru", 2174),
    ("Delhi", "Mumbai", 1452),
    ("Delhi", "Hyderabad", 1582),
    ("Goa", "Hyderabad", 674),
    ("Goa", "Kochi", 755),
    ("Goa", "Mumbai", 585),
    ("Goa", "Pune", 442),
    ("Goa", "Thiruvananthapuram", 1305),
    ("Hyderabad", "Pune", 562),
    ("Hyderabad", "Mumbai", 708),
    ("Hyderabad", "Kolkata", 1489),
    ("Jaipur", "Udaipur", 397),
    ("Jaipur", "Mumbai", 1170),
    ("Jaipur", "Pune", 1191),
    ("Kanpur", "Lucknow", 115),
    ("Kanpur", "Varanasi", 328),
    ("Kochi", "Bengaluru", 547),
    ("Kochi", "Thiruvananthapuram", 206),
    ("Kolkata", "Varanasi", 680),
    ("Kolkata", "Patna", 554),
    ("Lucknow", "Varanasi", 312),
    ("Mumbai", "Pune", 148),
    ("Mumbai", "Bengaluru", 984),
    ("Mumbai", "Kolkata", 1886),
    ("Patna", "Varanasi", 256),
    ("Udaipur", "Mumbai", 767),
    ("Udaipur", "Delhi", 688),
    ("Vishakhapatnam", "Hyderabad", 618),
    ("Vishakhapatnam", "Kolkata", 882),
]


def build_graph(routes):
    graph = {}

    for city_a, city_b, distance in routes:
        graph.setdefault(city_a, []).append((city_b, distance))
        graph.setdefault(city_b, []).append((city_a, distance))

    return graph


def dijkstra(graph, start, goal):
    queue = [(0, start)]
    distances = {city: float("inf") for city in graph}
    parent = {start: None}
    distances[start] = 0

    while queue:
        current_distance, current = heapq.heappop(queue)

        if current_distance != distances[current]:
            continue

        if current == goal:
            break

        for neighbour, weight in graph[current]:
            new_distance = current_distance + weight

            if new_distance < distances[neighbour]:
                distances[neighbour] = new_distance
                parent[neighbour] = current
                heapq.heappush(queue, (new_distance, neighbour))

    if distances.get(goal, float("inf")) == float("inf"):
        return None, None

    path = []
    node = goal

    while node is not None:
        path.append(node)
        node = parent[node]

    path.reverse()
    return path, distances[goal]


def main():
    graph = build_graph(ROUTES)
    cities = sorted(graph)

    print("DIJKSTRA'S ALGORITHM - INDIAN ROAD NETWORK")
    print("=" * 50)
    print("Available cities:")
    print(", ".join(cities))

    start = input("\nEnter source city: ").strip()
    goal = input("Enter destination city: ").strip()

    lookup = {city.lower(): city for city in cities}
    start = lookup.get(start.lower())
    goal = lookup.get(goal.lower())

    if not start or not goal:
        print("City not found. Please use one of the listed cities.")
        return

    path, distance = dijkstra(graph, start, goal)

    if path is None:
        print("\nNo route exists between the selected cities.")
        return

    print("\nShortest route")
    print("-" * 30)
    print(" -> ".join(path))
    print(f"Total road distance: {distance} km")
    print(f"Number of road segments: {len(path) - 1}")


if __name__ == "__main__":
    main()
