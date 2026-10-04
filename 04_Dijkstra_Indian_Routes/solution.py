# Assignment 4 - Dijkstra's Algorithm for Indian Routes

## Objective

Implement Dijkstra's shortest-path algorithm for a weighted road
network representing selected major cities in India.

The program finds the shortest road route between a source city and
a destination city.

## Algorithm Used

Dijkstra's Algorithm

Dijkstra's algorithm is used to find the shortest path from a source
vertex to other vertices in a weighted graph when all edge weights
are non-negative.

A priority queue is used to efficiently select the city with the
smallest currently known distance.

## Graph Representation

The Indian road network is represented using an adjacency list.

Each road contains:

- Source city
- Destination city
- Road distance in kilometres

The graph is undirected because a road can be travelled in either
direction.

## Cities Included

The implementation contains a selected set of major Indian cities,
including:

- Agra
- Ahmedabad
- Bengaluru
- Bhubaneswar
- Chennai
- Delhi
- Goa
- Hyderabad
- Jaipur
- Kanpur
- Kochi
- Kolkata
- Lucknow
- Mumbai
- Patna
- Pune
- Thiruvananthapuram
- Udaipur
- Varanasi
- Vishakhapatnam

## Program Features

1. Builds the weighted Indian road graph.
2. Uses Dijkstra's algorithm to find the shortest route.
3. Reconstructs the shortest path using parent information.
4. Calculates total road distance.
5. Displays the number of road segments.
6. Accepts source and destination from the user.
7. Handles city names without case sensitivity.
8. Handles invalid city names.
9. Handles cases where no route exists.
10. Includes automatic test cases for verification.

## Example

For:

Source:
Jaipur

Destination:
Chennai

A shortest route produced by the program is:

Jaipur -> Pune -> Bengaluru -> Chennai

Total road distance:

2376 km

Number of road segments:

3

## Complexity

Using an adjacency list and a binary heap priority queue:

Time Complexity:
O((V + E) log V)

Space Complexity:
O(V + E)

where:

V = number of cities

E = number of road connections

## How to Run

### Python

Run:

```bash
python solution.py
