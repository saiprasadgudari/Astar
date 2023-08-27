import csv
import time
import heapq
import math
from collections import defaultdict
import numpy as np
import random
import resource  # To measure memory usage
import psutil
import gc

# Heuristic functions
def manhattan(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def euclidean(node, goal):
    return math.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

def chebyshev(node, goal):
    return max(abs(node[0] - goal[0]), abs(node[1] - goal[1]))

heuristics = [manhattan, euclidean, chebyshev]

# A* for Grid
def A_star_grid(start, goal, grid, heuristic):
    neighbors = [(0,1),(0,-1),(1,0),(-1,0)]
    priority_queue, visited = [(heuristic(start, goal), 0, start, [])], set()
    nodes_expanded = 0
    start_time = time.time()
    
    while priority_queue:
        (priority, current_distance, current_node, path) = heapq.heappop(priority_queue)
        if current_node in visited: continue
        nodes_expanded += 1
        path = path + [current_node]
        if current_node == goal:
            execution_time = time.time() - start_time
            return path, execution_time, nodes_expanded
        visited.add(current_node)
        for dx, dy in neighbors:
            next_node = current_node[0] + dx, current_node[1] + dy
            if (0 <= next_node[0] < len(grid) and
                0 <= next_node[1] < len(grid[0]) and
                grid[next_node[0]][next_node[1]] != 1):
                next_distance = current_distance + 1
                heapq.heappush(priority_queue, (next_distance + heuristic(next_node, goal), next_distance, next_node, path))
    
    execution_time = time.time() - start_time
    return [], execution_time, nodes_expanded  # Return an empty list if no path is found

import heapq
import time

def A_star_graph(start, goal, graph, heuristic):
    priority_queue, visited = [(heuristic(start, goal), 0, start, [])], set()
    nodes_expanded = 0
    start_time = time.time()
    
    while priority_queue:
        (priority, current_distance, current_node, path) = heapq.heappop(priority_queue)
        if current_node in visited: continue
        nodes_expanded += 1
        path = path + [current_node]
        if current_node == goal:
            execution_time = time.time() - start_time
            return path, execution_time, nodes_expanded
        visited.add(current_node)
        for (next_node, edge_cost) in graph.get(current_node, {}).items():  # Use .get() to avoid KeyError
            if next_node not in visited:  # Check if the neighbor has been visited before
                next_distance = current_distance + edge_cost
                heapq.heappush(priority_queue, (next_distance + heuristic(next_node, goal), next_distance, next_node, path))
    
    execution_time = time.time() - start_time
    return [], execution_time, nodes_expanded

# For the sake of brevity, we won't rerun the entire testing procedure here.
# But now, the A_star_graph function should record the correct number of nodes expanded when it's used.
# Random Test Case Generation
def generate_test_cases():
    np.random.seed(0)  # For reproducible results
    test_cases = []
    NUM_TEST_CASES = 100

    # Calculate step size to ensure we generate 1000 test cases
    step = (100 - 10) / (NUM_TEST_CASES ** 0.5)
    
    # Use the calculated step size for grid generation
    for grid_size in np.arange(10, 101, step, dtype=int):  
        for _ in range(int(NUM_TEST_CASES ** 0.5)):  
            obstacle_prob = np.random.uniform(0.1, 0.4)  # Choose a random obstacle probability between 0.1 and 0.4
            grid = np.random.choice([0, 1], size=(grid_size, grid_size), p=[1-obstacle_prob, obstacle_prob])
            
            # Use the enhanced graph generation function
            graph = enhanced_graph_generation(grid)

            # Set the range of valid nodes for the grid and graph
            valid_nodes = [(i, j) for i in range(grid_size) for j in range(grid_size) if grid[i][j] != 1]

            # Choose a random start and goal
            start, goal = random.sample(valid_nodes, 2)

            test_cases.append((start, goal, grid, graph, grid_size, obstacle_prob))

    return test_cases

def enhanced_graph_generation(grid):
    """
    Enhanced graph generation to ensure better connectivity.
    This function creates edges between neighboring nodes in the grid if they are not obstacles.
    """
    grid_size = len(grid)
    graph = defaultdict(dict)
    
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) not in graph:
                graph[(i, j)] = {}
            if grid[i][j] != 1:  # Only add nodes that aren't obstacles
                if i < grid_size - 1 and grid[i + 1][j] != 1:
                    weight = np.random.randint(1, grid_size + 1)
                    graph[(i, j)][(i + 1, j)] = weight
                if j < grid_size - 1 and grid[i][j + 1] != 1:
                    weight = np.random.randint(1, grid_size + 1)
                    graph[(i, j)][(i, j + 1)] = weight
                    
    return graph
def is_connected(graph, start, goal):
    """
    Check if the start and goal nodes are part of the same connected component in the graph.
    """
    visited = set()
    queue = [start]
    
    while queue:
        current = queue.pop()
        visited.add(current)
        queue.extend([neighbor for neighbor, _ in graph[current].items() if neighbor not in visited])
        
    return goal in visited

# Provide the functions


def generate_test_cases():
    np.random.seed(0)  # For reproducible results
    test_cases = []
    NUM_TEST_CASES = 100

    # Calculate step size to ensure we generate 1000 test cases
    step = (100 - 10) / (NUM_TEST_CASES ** 0.5)
    
    # Use the calculated step size for grid generation
    for grid_size in np.arange(10, 101, step, dtype=int):  
        for _ in range(int(NUM_TEST_CASES ** 0.5)):  
            obstacle_prob = np.random.uniform(0.1, 0.4)  # Choose a random obstacle probability between 0.1 and 0.4
            grid = np.random.choice([0, 1], size=(grid_size, grid_size), p=[1-obstacle_prob, obstacle_prob])
            
            # Use the enhanced graph generation
            graph = enhanced_graph_generation(grid)

            # Set the range of valid nodes for the grid and graph
            valid_nodes = [(i, j) for i in range(grid_size) for j in range(grid_size) if grid[i][j] != 1]

            # Choose a random start and goal
            start, goal = random.sample(valid_nodes, 2)
            # Check if start and goal are connected
            while not is_connected(graph, start, goal):
                start, goal = random.sample(valid_nodes, 2)
                
            test_cases.append((start, goal, grid, graph, grid_size, obstacle_prob))

    return test_cases

# Provide the modified generate_test_cases function

# Curated Test Case Generation
import numpy as np
from collections import defaultdict
import random

def curated_test_cases():
    cases = [
        ("Open Space", np.zeros((10, 10), dtype=int)),
        ("Completely Blocked", np.ones((10, 10), dtype=int)),
        ("Long Corridor", np.ones((10, 10), dtype=int)),
        ("Simple Maze", np.array([
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0, 1, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
        ])),
        ("Sparse Obstacles", np.random.choice([0, 1], size=(10, 10), p=[0.8, 0.2])),
        ("Dense Obstacles", np.random.choice([0, 1], size=(10, 10), p=[0.4, 0.6])),
        ("Diagonal Path", np.array([[(i != j) for j in range(10)] for i in range(10)])),
        ("Multiple Paths", np.zeros((10, 10), dtype=int)),
        ("Islands", np.zeros((10, 10), dtype=int)),
        ("Borders Blocked", np.zeros((10, 10), dtype=int))
    ]

    # Add corridors for the Long Corridor grid
    cases[2][1][:, 5] = 0

    # Add the walls for the Multiple Paths grid
    cases[7][1][4:6, 2:8] = 1

    # Add islands for the Islands grid
    cases[8][1][3:6, 3:6] = 1
    cases[8][1][7:9, 7:9] = 1

    # Add the border blocks
    cases[9][1][0, :] = 1
    cases[9][1][9, :] = 1
    cases[9][1][:, 0] = 1
    cases[9][1][:, 9] = 1

    return cases

def generate_curated_test_cases():
    curated = curated_test_cases()

    curated_test_cases_list = []
    for name, grid in curated:
        grid_size = len(grid)
        obstacle_prob = np.mean(grid)

        # Set the range of valid nodes for the grid
        valid_nodes = [(i, j) for i in range(grid_size) for j in range(grid_size) if grid[i][j] != 1]

        # Create a random graph
        graph = defaultdict(dict)
        for i in range(grid_size):
            for j in range(grid_size):
                if (i,j) not in graph:
                    graph[(i,j)] = {}
                if grid[i][j] != 1:  # Only add nodes that aren't obstacles
                    if i < grid_size - 1 and grid[i+1][j] != 1:
                        weight = np.random.randint(1, grid_size + 1) 
                        if np.random.choice([0, 1], p=[1-obstacle_prob, obstacle_prob]):
                            graph[(i,j)][(i+1,j)] = weight  # Random edge weights
                    if j < grid_size - 1 and grid[i][j+1] != 1:
                        weight = np.random.randint(1, grid_size + 1)
                        if np.random.choice([0, 1], p=[1-obstacle_prob, obstacle_prob]):
                            graph[(i,j)][(i,j+1)] = weight  # Random edge weights

        # Choose a random start and goal only if there are at least 2 valid nodes
        if len(valid_nodes) >= 2:
            start, goal = random.sample(valid_nodes, 2)
            curated_test_cases_list.append((start, goal, grid, graph, grid_size, obstacle_prob))
        else:
            print(f"Skipping test case '{name}' due to insufficient valid nodes.")

    return curated_test_cases_list


# Testing
print(generate_curated_test_cases())


# def evaluate_and_write_to_csv(test_num, start, goal, grid, graph, grid_size, obstacle_prob, writer):
#     for heuristic in heuristics:
#         gc.collect()  # Explicitly call the garbage collector

#         initial_mem_usage = psutil.Process().memory_info().rss # Get initial memory usage in KB
#         # Test grid environment
#         grid_path, grid_execution_time, grid_nodes_expanded = A_star_grid(start, goal, grid.tolist(), heuristic)
#         final_mem_usage = psutil.Process().memory_info().rss  # Get final memory usage in KB
#         grid_memory_consumption = final_mem_usage - initial_mem_usage
#         writer.writerow([test_num, "Grid", heuristic.__name__, grid_execution_time, len(grid_path), start, goal, grid_size, obstacle_prob, grid_nodes_expanded, grid_memory_consumption])
#         gc.collect()  # Explicitly call the garbage collector

#         initial_mem_usage = psutil.Process().memory_info().rss # Get initial memory usage in KB
#         # Test graph environment
#         print(graph)
#         print("Start:", start)
#         print("Goal:", goal)

#         graph_path, graph_execution_time, graph_nodes_expanded = A_star_graph(start, goal, graph, heuristic)
#         final_mem_usage = psutil.Process().memory_info().rss# Get final memory usage in KB
#         graph_memory_consumption = final_mem_usage - initial_mem_usage
#         writer.writerow([test_num, "Graph", heuristic.__name__, graph_execution_time, len(graph_path), start, goal, grid_size, obstacle_prob, graph_nodes_expanded, graph_memory_consumption])
import tracemalloc

def evaluate_and_write_to_csv(test_num, start, goal, grid, graph, grid_size, obstacle_prob, writer):
    for heuristic in heuristics:
        
        # Grid environment memory measurement
        tracemalloc.start()  # Start tracing memory allocations
        grid_path, grid_execution_time, grid_nodes_expanded = A_star_grid(start, goal, grid.tolist(), heuristic)
        current, peak = tracemalloc.get_traced_memory()  # Get traced memory    
        tracemalloc.stop()  # Stop tracing
        grid_memory_consumption = current
        
        writer.writerow([test_num, "Grid", heuristic.__name__, grid_execution_time, len(grid_path), start, goal, grid_size, obstacle_prob, grid_nodes_expanded, grid_memory_consumption])
        
        # Graph environment memory measurement
        tracemalloc.start()  # Start tracing memory allocations
        graph_path, graph_execution_time, graph_nodes_expanded = A_star_graph(start, goal, graph, heuristic)
        current, peak = tracemalloc.get_traced_memory()  # Get traced memory
        tracemalloc.stop()  # Stop tracing
        graph_memory_consumption = current
        
        writer.writerow([test_num, "Graph", heuristic.__name__, graph_execution_time, len(graph_path), start, goal, grid_size, obstacle_prob, graph_nodes_expanded, graph_memory_consumption])

# The provided function now uses tracemalloc for more precise memory measurements. 
# You can use this updated function in your environment to see if it provides more accurate and consistent memory consumption values.

def test_A_star():
    with open('A_star_results_random.csv', 'w', newline='') as random_file, \
         open('A_star_results_curated.csv', 'w', newline='') as curated_file:
        
        random_writer = csv.writer(random_file)
        curated_writer = csv.writer(curated_file)
        
        headers = ["Test Case", "Environment", "Heuristic", "Execution time", "Path length", "Start", "Goal", "Grid Size", "Obstacle Probability", "Nodes Expanded", "Memory Consumption (Bytes)"]
        random_writer.writerow(headers)
        curated_writer.writerow(headers)
        
        # Random test cases generation
        test_cases_random = generate_test_cases()
        for test_num, (start, goal, grid, graph, grid_size, obstacle_prob) in enumerate(test_cases_random, 1):
            evaluate_and_write_to_csv(test_num, start, goal, grid, graph, grid_size, obstacle_prob, random_writer)

        # Curated test cases generation
        test_cases_curated = generate_curated_test_cases()
        for test_num, (start, goal, grid, graph, grid_size, obstacle_prob) in enumerate(test_cases_curated, 1):
            evaluate_and_write_to_csv(test_num, start, goal, grid, graph, grid_size, obstacle_prob, curated_writer)

if __name__ == "__main__":
    test_A_star()
