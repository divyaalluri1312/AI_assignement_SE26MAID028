# Telangana District Map Coloring

## Requirement

Implement the map-coloring problem for Telangana districts.

## Approach

The districts are treated as CSP variables. Adjacent districts are connected by constraints, and a backtracking solver assigns colors so adjacent districts do not share a color.

MRV, degree and least-constraining-value ideas are used to reduce the search.

## Run

```bash
python solution.py
```

The program also offers an adjacency-graph visualization.
