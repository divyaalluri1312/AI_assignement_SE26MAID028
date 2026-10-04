# Artificial Intelligence - Assignment 2

This repository contains the programming implementations for the AI assignment.

## Assignment topics

### Part A

1. Uninformed Search
   - Breadth-First Search
   - Uniform-Cost Search
   - Depth-First Search
   - Depth-Limited Search
   - Iterative Deepening Search
   - Bidirectional Search

2. Informed Search
   - Greedy Best-First Search
   - Best-First Search
   - Dijkstra's Algorithm
   - A*
   - Beam Search
   - RBFS
   - IDA*
   - Weighted A*

3. Heuristics
   - Satisficing search
   - Admissible heuristic
   - Heuristic formulation
   - Heuristics from subproblems

### Programming problems

4. Dijkstra's algorithm using Indian city road routes

5. UGV navigation with static obstacles

6. UGV navigation with dynamic obstacles

7. Search route finding using Indian cities

8. Map coloring for Telangana districts

## Folder structure

```text
01_uninformed_search/
02_informed_search/
03_heuristics/
04_dijkstra_indian_routes/
05_ugv_static/
06_ugv_dynamic/
07_indian_route_search/
08_ts_map_coloring/
```

Each assignment folder contains one main Python program and a short README.

## Requirements

Python 3.10 or later is recommended.

Install the only external package used for visualisation:

```bash
pip install -r requirements.txt
```

## Running a program

Open the required folder in a terminal and run:

```bash
python solution.py
```

The UGV and Telangana programs display visual output using Matplotlib.

## Note on data

The Indian route programs use a compact open-data-based road graph so that the algorithms remain easy to run and understand. The graph can be extended by adding more road edges.

The Telangana program models the districts as a constraint graph. The Telangana government currently lists 33 districts.

## Purpose

The programs are written to demonstrate the algorithms rather than hide the logic behind external AI or search libraries.
