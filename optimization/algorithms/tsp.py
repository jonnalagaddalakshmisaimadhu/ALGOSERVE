import time
import psutil
import numpy as np
from typing import List, Dict, Tuple, Any
import sys


def measure_performance(func):
    """Decorator to measure execution time and memory usage"""
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        return {
            'solution': result,
            'time': end_time - start_time,
            'memory': mem_after - mem_before
        }
    return wrapper


@measure_performance
def tsp_greedy_nearest_neighbor(distance_matrix: List[List[float]], coordinates: List[Tuple[float, float]] = None) -> Dict[str, Any]:
    """Greedy Nearest Neighbor approach for TSP"""
    n = len(distance_matrix)
    visited = [False] * n
    path = [0]
    visited[0] = True
    current = 0
    total_distance = 0
    
    for _ in range(n - 1):
        nearest = -1
        min_dist = float('inf')
        
        for j in range(n):
            if not visited[j] and distance_matrix[current][j] < min_dist:
                min_dist = distance_matrix[current][j]
                nearest = j
        
        if nearest != -1:
            path.append(nearest)
            visited[nearest] = True
            total_distance += min_dist
            current = nearest
    
    # Return to start
    total_distance += distance_matrix[current][0]
    path.append(0)
    
    return {
        'path': path,
        'total_distance': total_distance,
        'coordinates': coordinates
    }


@measure_performance
def tsp_dynamic_programming(distance_matrix: List[List[float]], coordinates: List[Tuple[float, float]] = None) -> Dict[str, Any]:
    """Held-Karp Dynamic Programming approach for TSP (exact solution)"""
    n = len(distance_matrix)
    
    # Limit size to avoid excessive computation
    if n > 15:
        return {
            'path': list(range(n)) + [0],
            'total_distance': -1,
            'coordinates': coordinates,
            'error': 'DP approach limited to 15 cities for performance'
        }
    
    # dp[mask][i] = minimum cost to visit all cities in mask ending at city i
    dp = [[float('inf')] * n for _ in range(1 << n)]
    parent = [[None] * n for _ in range(1 << n)]
    
    # Start from city 0
    dp[1][0] = 0
    
    for mask in range(1 << n):
        for last in range(n):
            if dp[mask][last] == float('inf'):
                continue
            
            for next_city in range(n):
                if mask & (1 << next_city):
                    continue
                
                new_mask = mask | (1 << next_city)
                new_cost = dp[mask][last] + distance_matrix[last][next_city]
                
                if new_cost < dp[new_mask][next_city]:
                    dp[new_mask][next_city] = new_cost
                    parent[new_mask][next_city] = last
    
    # Find the best ending city
    full_mask = (1 << n) - 1
    min_cost = float('inf')
    last_city = -1
    
    for i in range(n):
        cost = dp[full_mask][i] + distance_matrix[i][0]
        if cost < min_cost:
            min_cost = cost
            last_city = i
    
    # Reconstruct path
    path = []
    mask = full_mask
    current = last_city
    
    while current is not None:
        path.append(current)
        new_current = parent[mask][current]
        if new_current is not None:
            mask ^= (1 << current)
        current = new_current
    
    path.reverse()
    path.append(0)
    
    return {
        'path': path,
        'total_distance': min_cost,
        'coordinates': coordinates
    }


@measure_performance
def tsp_backtracking(distance_matrix: List[List[float]], coordinates: List[Tuple[float, float]] = None) -> Dict[str, Any]:
    """Backtracking approach for TSP"""
    n = len(distance_matrix)
    
    # Limit size for performance
    if n > 12:
        return {
            'path': list(range(n)) + [0],
            'total_distance': -1,
            'coordinates': coordinates,
            'error': 'Backtracking limited to 12 cities for performance'
        }
    
    best_path = [None]
    best_distance = [float('inf')]
    
    def backtrack(current: int, visited: List[bool], path: List[int], current_distance: float):
        if len(path) == n:
            # Return to start
            total = current_distance + distance_matrix[current][0]
            if total < best_distance[0]:
                best_distance[0] = total
                best_path[0] = path + [0]
            return
        
        # Pruning
        if current_distance >= best_distance[0]:
            return
        
        for next_city in range(n):
            if not visited[next_city]:
                visited[next_city] = True
                path.append(next_city)
                backtrack(next_city, visited, path, current_distance + distance_matrix[current][next_city])
                path.pop()
                visited[next_city] = False
    
    visited = [False] * n
    visited[0] = True
    backtrack(0, visited, [0], 0)
    
    return {
        'path': best_path[0] if best_path[0] else [0],
        'total_distance': best_distance[0] if best_distance[0] != float('inf') else 0,
        'coordinates': coordinates
    }


@measure_performance
def tsp_branch_and_bound(distance_matrix: List[List[float]], coordinates: List[Tuple[float, float]] = None) -> Dict[str, Any]:
    """Branch and Bound approach for TSP"""
    n = len(distance_matrix)
    
    # Limit size for performance
    if n > 13:
        return {
            'path': list(range(n)) + [0],
            'total_distance': -1,
            'coordinates': coordinates,
            'error': 'Branch and Bound limited to 13 cities for performance'
        }
    
    # Calculate lower bound using minimum edges
    def calculate_lower_bound(visited: List[bool], current_distance: float) -> float:
        bound = current_distance
        
        for i in range(n):
            if not visited[i]:
                min_edge = min(distance_matrix[i][j] for j in range(n) if i != j)
                bound += min_edge
        
        return bound
    
    best_path = [None]
    best_distance = [float('inf')]
    
    def branch_and_bound_helper(current: int, visited: List[bool], path: List[int], current_distance: float):
        if len(path) == n:
            total = current_distance + distance_matrix[current][0]
            if total < best_distance[0]:
                best_distance[0] = total
                best_path[0] = path + [0]
            return
        
        # Calculate lower bound
        lower_bound = calculate_lower_bound(visited, current_distance)
        if lower_bound >= best_distance[0]:
            return
        
        # Try all unvisited cities
        for next_city in range(n):
            if not visited[next_city]:
                visited[next_city] = True
                path.append(next_city)
                branch_and_bound_helper(next_city, visited, path, 
                                       current_distance + distance_matrix[current][next_city])
                path.pop()
                visited[next_city] = False
    
    visited = [False] * n
    visited[0] = True
    branch_and_bound_helper(0, visited, [0], 0)
    
    return {
        'path': best_path[0] if best_path[0] else [0],
        'total_distance': best_distance[0] if best_distance[0] != float('inf') else 0,
        'coordinates': coordinates
    }
