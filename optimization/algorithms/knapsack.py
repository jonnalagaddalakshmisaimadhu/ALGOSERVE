import time
import psutil
import numpy as np
from typing import List, Dict, Tuple, Any


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
def knapsack_greedy(weights: List[int], values: List[int], capacity: int) -> Dict[str, Any]:
    """Greedy approach: select items by value-to-weight ratio"""
    n = len(weights)
    items = list(range(n))
    
    # Calculate value-to-weight ratio
    ratios = [(values[i] / weights[i], i) for i in range(n) if weights[i] > 0]
    ratios.sort(reverse=True)
    
    total_weight = 0
    total_value = 0
    selected = []
    
    for ratio, i in ratios:
        if total_weight + weights[i] <= capacity:
            selected.append(i)
            total_weight += weights[i]
            total_value += values[i]
    
    return {
        'selected_items': selected,
        'total_value': total_value,
        'total_weight': total_weight
    }


@measure_performance
def knapsack_dynamic_programming(weights: List[int], values: List[int], capacity: int) -> Dict[str, Any]:
    """Dynamic Programming approach for 0/1 Knapsack"""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                dp[i][w] = dp[i - 1][w]
    
    # Backtrack to find selected items
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(i - 1)
            w -= weights[i - 1]
    
    selected.reverse()
    total_value = dp[n][capacity]
    total_weight = sum(weights[i] for i in selected)
    
    return {
        'selected_items': selected,
        'total_value': total_value,
        'total_weight': total_weight
    }


@measure_performance
def knapsack_backtracking(weights: List[int], values: List[int], capacity: int) -> Dict[str, Any]:
    """Backtracking approach for 0/1 Knapsack"""
    n = len(weights)
    best_value = [0]
    best_selection = [[]]
    
    def backtrack(index: int, current_weight: int, current_value: int, selected: List[int]):
        if index == n:
            if current_value > best_value[0]:
                best_value[0] = current_value
                best_selection[0] = selected.copy()
            return
        
        # Include current item
        if current_weight + weights[index] <= capacity:
            selected.append(index)
            backtrack(index + 1, current_weight + weights[index], current_value + values[index], selected)
            selected.pop()
        
        # Exclude current item
        backtrack(index + 1, current_weight, current_value, selected)
    
    backtrack(0, 0, 0, [])
    total_weight = sum(weights[i] for i in best_selection[0])
    
    return {
        'selected_items': best_selection[0],
        'total_value': best_value[0],
        'total_weight': total_weight
    }


@measure_performance
def knapsack_branch_and_bound(weights: List[int], values: List[int], capacity: int) -> Dict[str, Any]:
    """Branch and Bound approach for 0/1 Knapsack"""
    n = len(weights)
    
    # Calculate upper bound using fractional knapsack
    def upper_bound(index: int, current_weight: int, current_value: int) -> float:
        if current_weight >= capacity:
            return 0
        
        bound = current_value
        total_weight = current_weight
        
        for i in range(index, n):
            if total_weight + weights[i] <= capacity:
                total_weight += weights[i]
                bound += values[i]
            else:
                bound += (capacity - total_weight) * values[i] / weights[i]
                break
        
        return bound
    
    # Sort items by value-to-weight ratio
    items = [(values[i] / weights[i] if weights[i] > 0 else 0, i) for i in range(n)]
    items.sort(reverse=True)
    sorted_weights = [weights[i] for _, i in items]
    sorted_values = [values[i] for _, i in items]
    original_indices = [i for _, i in items]
    
    best_value = [0]
    best_selection = [[]]
    
    def branch_and_bound_helper(index: int, current_weight: int, current_value: int, selected: List[int]):
        if current_weight <= capacity and current_value > best_value[0]:
            best_value[0] = current_value
            best_selection[0] = selected.copy()
        
        if index >= n:
            return
        
        # Check upper bound
        if upper_bound(index, current_weight, current_value) <= best_value[0]:
            return
        
        # Include current item
        if current_weight + sorted_weights[index] <= capacity:
            selected.append(original_indices[index])
            branch_and_bound_helper(index + 1, current_weight + sorted_weights[index], 
                                   current_value + sorted_values[index], selected)
            selected.pop()
        
        # Exclude current item
        branch_and_bound_helper(index + 1, current_weight, current_value, selected)
    
    branch_and_bound_helper(0, 0, 0, [])
    total_weight = sum(weights[i] for i in best_selection[0])
    
    return {
        'selected_items': best_selection[0],
        'total_value': best_value[0],
        'total_weight': total_weight
    }
