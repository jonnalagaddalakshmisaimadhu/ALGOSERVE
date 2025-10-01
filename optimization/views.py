from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import pandas as pd
import numpy as np
from .algorithms import knapsack, tsp, graph_matching


def home(request):
    """Home page"""
    return render(request, 'optimization/home.html')


@csrf_exempt
def upload(request):
    """Upload dataset page"""
    if request.method == 'GET':
        return render(request, 'optimization/upload.html')
    
    if request.method == 'POST':
        try:
            uploaded_file = request.FILES.get('dataset')
            problem_type = request.POST.get('problem_type')
            
            if not uploaded_file or not problem_type:
                return JsonResponse({'error': 'Missing file or problem type'}, status=400)
            
            # Parse dataset based on file type
            file_content = uploaded_file.read()
            
            if uploaded_file.name.endswith('.csv'):
                import io
                df = pd.read_csv(io.BytesIO(file_content))
                data = df.to_dict('records')
            elif uploaded_file.name.endswith('.json'):
                data = json.loads(file_content.decode('utf-8'))
            else:
                return JsonResponse({'error': 'Unsupported file format'}, status=400)
            
            # Store in session
            request.session['dataset'] = data
            request.session['problem_type'] = problem_type
            
            return redirect('algorithm_selection')
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)


def algorithm_selection(request):
    """Algorithm selection page"""
    if 'dataset' not in request.session:
        return redirect('upload')
    
    problem_type = request.session.get('problem_type')
    return render(request, 'optimization/algorithm_selection.html', {
        'problem_type': problem_type
    })


@csrf_exempt
def solve(request, problem):
    """Solve optimization problem"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=400)
    
    if 'dataset' not in request.session:
        return JsonResponse({'error': 'No dataset uploaded'}, status=400)
    
    try:
        data = json.loads(request.body)
        algorithms = data.get('algorithms', [])
        dataset = request.session['dataset']
        
        results = {}
        
        if problem == 'knapsack':
            results = solve_knapsack(dataset, algorithms)
        elif problem == 'tsp':
            results = solve_tsp(dataset, algorithms)
        elif problem == 'graph_matching':
            results = solve_graph_matching(dataset, algorithms)
        else:
            return JsonResponse({'error': 'Invalid problem type'}, status=400)
        
        return JsonResponse(results)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def solve_knapsack(dataset, algorithms):
    """Execute knapsack algorithms"""
    results = {}
    
    # Normalize keys (handle case sensitivity and whitespace)
    normalized_dataset = []
    for item in dataset:
        normalized_item = {k.strip().lower(): v for k, v in item.items()}
        normalized_dataset.append(normalized_item)
    
    # Validate required fields
    if not normalized_dataset or 'weight' not in normalized_dataset[0] or 'value' not in normalized_dataset[0]:
        raise ValueError("Dataset must contain 'weight' and 'value' columns")
    
    # Parse dataset
    weights = [int(item['weight']) for item in normalized_dataset]
    values = [int(item['value']) for item in normalized_dataset]
    capacity = int(normalized_dataset[0].get('capacity', 50))
    
    if 'greedy' in algorithms:
        results['greedy'] = knapsack.knapsack_greedy(weights, values, capacity)
    
    if 'dynamic_programming' in algorithms:
        results['dynamic_programming'] = knapsack.knapsack_dynamic_programming(weights, values, capacity)
    
    if 'backtracking' in algorithms:
        results['backtracking'] = knapsack.knapsack_backtracking(weights, values, capacity)
    
    if 'branch_and_bound' in algorithms:
        results['branch_and_bound'] = knapsack.knapsack_branch_and_bound(weights, values, capacity)
    
    return results


def solve_tsp(dataset, algorithms):
    """Execute TSP algorithms"""
    results = {}
    
    # Normalize keys
    normalized_dataset = []
    for item in dataset:
        normalized_item = {k.strip().lower(): v for k, v in item.items()}
        normalized_dataset.append(normalized_item)
    
    # Parse dataset - expect coordinates or distance matrix
    if 'x' in normalized_dataset[0] and 'y' in normalized_dataset[0]:
        coordinates = [(float(item['x']), float(item['y'])) for item in normalized_dataset]
        n = len(coordinates)
        
        # Calculate distance matrix
        distance_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = coordinates[i][0] - coordinates[j][0]
                    dy = coordinates[i][1] - coordinates[j][1]
                    distance_matrix[i][j] = np.sqrt(dx**2 + dy**2)
    else:
        # Assume distance matrix is provided
        distance_matrix = [list(item.values()) for item in dataset]
        coordinates = None
    
    if 'greedy' in algorithms:
        results['greedy'] = tsp.tsp_greedy_nearest_neighbor(distance_matrix, coordinates)
    
    if 'dynamic_programming' in algorithms:
        results['dynamic_programming'] = tsp.tsp_dynamic_programming(distance_matrix, coordinates)
    
    if 'backtracking' in algorithms:
        results['backtracking'] = tsp.tsp_backtracking(distance_matrix, coordinates)
    
    if 'branch_and_bound' in algorithms:
        results['branch_and_bound'] = tsp.tsp_branch_and_bound(distance_matrix, coordinates)
    
    return results


def solve_graph_matching(dataset, algorithms):
    """Execute graph matching algorithms"""
    results = {}
    
    # Normalize keys
    normalized_dataset = []
    for item in dataset:
        normalized_item = {k.strip().lower(): v for k, v in item.items()}
        normalized_dataset.append(normalized_item)
    
    # Validate required fields
    if not normalized_dataset or 'u' not in normalized_dataset[0] or 'v' not in normalized_dataset[0] or 'weight' not in normalized_dataset[0]:
        raise ValueError("Dataset must contain 'u', 'v', and 'weight' columns")
    
    # Parse dataset - expect edges with format: {u, v, weight}
    edges = [(int(item['u']), int(item['v']), float(item['weight'])) for item in normalized_dataset]
    
    if 'greedy' in algorithms:
        results['greedy'] = graph_matching.graph_matching_greedy(edges)
    
    if 'backtracking' in algorithms:
        results['backtracking'] = graph_matching.graph_matching_backtracking(edges)
    
    return results


def results(request):
    """Results display page"""
    return render(request, 'optimization/results.html')
