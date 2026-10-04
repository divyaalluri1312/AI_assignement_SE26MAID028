# UGV - Dynamic Obstacles

## Requirement

Navigate when obstacles can move and are not known in advance.

## Approach

The UGV senses nearby obstacles, updates its internal map, replans with A*, and then moves one step. This repeats until the goal is reached or the step limit is reached.

## Run

```bash
python solution.py
```
