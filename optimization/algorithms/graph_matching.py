import time
import psutil
import numpy as np
from typing import List, Dict, Tuple, Any, Set


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
def graph_matching_greedy(edges: List[Tuple[int, int, float]]) -> Dict[str, Any]:
    """Greedy approach for maximum weight matching"""
    # Sort edges by weight (descending)
    sorted_edges = sorted(edges, key=lambda x: x[2], reverse=True)
    
    matched = []
    matched_vertices = set()
    total_weight = 0
    
    for u, v, weight in sorted_edges:
        if u not in matched_vertices and v not in matched_vertices:
            matched.append((u, v, weight))
            matched_vertices.add(u)
            matched_vertices.add(v)
            total_weight += weight
    
    return {
        'matched_edges': matched,
        'total_weight': total_weight,
        'num_matched': len(matched)
    }


@measure_performance
def graph_matching_backtracking(edges: List[Tuple[int, int, float]], max_vertices: int = None) -> Dict[str, Any]:
    """Backtracking approach for maximum weight matching"""
    if not edges:
        return {
            'matched_edges': [],
            'total_weight': 0,
            'num_matched': 0
        }
    
    # Get all vertices
    vertices = set()
    for u, v, _ in edges:
        vertices.add(u)
        vertices.add(v)
    
    if max_vertices is None:
        max_vertices = max(vertices) + 1 if vertices else 0
    
    # Limit size for performance
    if len(edges) > 20:
        return {
            'matched_edges': [],
            'total_weight': -1,
            'num_matched': 0,
            'error': 'Backtracking limited to 20 edges for performance'
        }
    
    best_matching = [None]
    best_weight = [0]
    
    def backtrack(index: int, current_matching: List[Tuple[int, int, float]], 
                  matched_vertices: Set[int], current_weight: float):
        if index == len(edges):
            if current_weight > best_weight[0]:
                best_weight[0] = current_weight
                best_matching[0] = current_matching.copy()
            return
        
        u, v, weight = edges[index]
        
        # Include edge if both vertices are unmatched
        if u not in matched_vertices and v not in matched_vertices:
            current_matching.append((u, v, weight))
            matched_vertices.add(u)
            matched_vertices.add(v)
            backtrack(index + 1, current_matching, matched_vertices, current_weight + weight)
            current_matching.pop()
            matched_vertices.remove(u)
            matched_vertices.remove(v)
        
        # Exclude edge
        backtrack(index + 1, current_matching, matched_vertices, current_weight)
    
    backtrack(0, [], set(), 0)
    
    return {
        'matched_edges': best_matching[0] if best_matching[0] else [],
        'total_weight': best_weight[0],
        'num_matched': len(best_matching[0]) if best_matching[0] else 0
    }
