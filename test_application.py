import requests
import json

BASE_URL = "http://localhost:5000"

def test_knapsack():
    """Test Knapsack problem solving"""
    print("Testing Knapsack problem...")
    
    session = requests.Session()
    
    files = {'dataset': open('test_datasets/knapsack_sample.csv', 'rb')}
    data = {'problem_type': 'knapsack'}
    
    response = session.post(f"{BASE_URL}/upload/", files=files, data=data, allow_redirects=False)
    print(f"Upload status: {response.status_code}")
    
    if response.status_code == 302:
        algorithms = ['greedy', 'dynamic_programming', 'backtracking', 'branch_and_bound']
        solve_response = session.post(
            f"{BASE_URL}/solve/knapsack/",
            json={'algorithms': algorithms},
            headers={'Content-Type': 'application/json'}
        )
        
        if solve_response.status_code == 200:
            results = solve_response.json()
            print("Knapsack Results:")
            for alg, data in results.items():
                print(f"  {alg}: Value={data['solution']['total_value']}, Time={data['time']:.4f}s")
            return True
        else:
            print(f"Solve failed: {solve_response.text}")
            return False
    else:
        print(f"Upload failed")
        return False

def test_tsp():
    """Test TSP problem solving"""
    print("\nTesting TSP problem...")
    
    session = requests.Session()
    
    files = {'dataset': open('test_datasets/tsp_sample.csv', 'rb')}
    data = {'problem_type': 'tsp'}
    
    response = session.post(f"{BASE_URL}/upload/", files=files, data=data, allow_redirects=False)
    print(f"Upload status: {response.status_code}")
    
    if response.status_code == 302:
        algorithms = ['greedy', 'dynamic_programming']
        solve_response = session.post(
            f"{BASE_URL}/solve/tsp/",
            json={'algorithms': algorithms},
            headers={'Content-Type': 'application/json'}
        )
        
        if solve_response.status_code == 200:
            results = solve_response.json()
            print("TSP Results:")
            for alg, data in results.items():
                print(f"  {alg}: Distance={data['solution']['total_distance']:.2f}, Time={data['time']:.4f}s")
            return True
        else:
            print(f"Solve failed: {solve_response.text}")
            return False
    else:
        print(f"Upload failed")
        return False

def test_graph_matching():
    """Test Graph Matching problem solving"""
    print("\nTesting Graph Matching problem...")
    
    session = requests.Session()
    
    files = {'dataset': open('test_datasets/graph_matching_sample.csv', 'rb')}
    data = {'problem_type': 'graph_matching'}
    
    response = session.post(f"{BASE_URL}/upload/", files=files, data=data, allow_redirects=False)
    print(f"Upload status: {response.status_code}")
    
    if response.status_code == 302:
        algorithms = ['greedy', 'backtracking']
        solve_response = session.post(
            f"{BASE_URL}/solve/graph_matching/",
            json={'algorithms': algorithms},
            headers={'Content-Type': 'application/json'}
        )
        
        if solve_response.status_code == 200:
            results = solve_response.json()
            print("Graph Matching Results:")
            for alg, data in results.items():
                print(f"  {alg}: Weight={data['solution']['total_weight']}, Edges={data['solution']['num_matched']}, Time={data['time']:.4f}s")
            return True
        else:
            print(f"Solve failed: {solve_response.text}")
            return False
    else:
        print(f"Upload failed")
        return False

if __name__ == "__main__":
    print("Starting Application Tests\n" + "="*50)
    
    success_count = 0
    total_tests = 3
    
    if test_knapsack():
        success_count += 1
    
    if test_tsp():
        success_count += 1
    
    if test_graph_matching():
        success_count += 1
    
    print("\n" + "="*50)
    print(f"Tests Passed: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("All tests passed successfully!")
    else:
        print("Some tests failed.")
