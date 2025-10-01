# Combinatorial Optimization: Multi-Strategy Approach

## Overview

This is a full-stack web application built with Django that allows users to solve combinatorial optimization problems using multiple algorithmic strategies. Users can upload datasets, select from three problem types (Knapsack, Traveling Salesman Problem, Graph Matching), choose multiple algorithms to compare (Greedy, Dynamic Programming, Backtracking, Branch & Bound), and visualize results with performance comparisons.

The application is designed to be stateless - all data is stored temporarily in Django sessions and cleared on page refresh. No database persistence is used.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Architecture

**Framework**: Django 5.2.6 with Python

**Application Structure**: Single Django app called `optimization` within the `combinatorial_optimization` project

**Session-Based Storage**: All user data (uploaded datasets, problem types) is stored in Django sessions rather than a database. This makes the application completely stateless - data exists only during the user's session and is cleared on refresh.

**Algorithm Organization**: Algorithms are organized in the `optimization/algorithms/` directory with three modules:
- `knapsack.py` - Implements 4 algorithms for the knapsack problem
- `tsp.py` - Implements 4 algorithms for TSP
- `graph_matching.py` - Implements algorithms for graph matching

**Performance Measurement**: Each algorithm uses a decorator pattern (`@measure_performance`) that wraps algorithm functions to track:
- Execution time using Python's `time` module
- Memory usage using `psutil` library
- This allows consistent performance metrics across all algorithms

**Data Processing**: Uses pandas for CSV parsing and native JSON parsing for JSON files. Files are read into memory and converted to Python dictionaries for processing.

**API Design**: 
- `/upload/` - Handles file upload via POST, stores data in session, redirects to algorithm selection
- `/solve/<problem>/` - Accepts algorithm selections via POST, executes selected algorithms, returns JSON results
- All results are returned as JSON for frontend consumption

### Frontend Architecture

**Template System**: Django templates with template inheritance using `base.html` as the parent template

**CSS Framework**: Bootstrap 5.3.0 for responsive UI components and layout

**Icons**: Bootstrap Icons for visual elements

**Multi-Page Flow**:
1. Home page - Project overview and entry point
2. Upload page - File upload and problem type selection
3. Algorithm selection page - Checkbox selection of algorithms to compare
4. Results page - Displays solutions, performance charts, and visualizations

**Visualization Libraries**:
- Chart.js 4.4.0 for runtime and memory comparison bar charts
- Plotly.js 2.27.0 for TSP route visualization and graph matching displays

**State Management**: Uses browser `sessionStorage` to pass results data between pages (from solve endpoint to results page)

**AJAX Communication**: Fetch API used to submit algorithm selections and receive JSON results without full page reloads

### External Dependencies

**Python Libraries**:
- Django 5.2.6 - Web framework
- pandas - CSV data parsing
- numpy - Numerical operations in algorithms
- psutil - Memory usage tracking

**Frontend Libraries** (CDN-based):
- Bootstrap 5.3.0 - UI framework
- Bootstrap Icons 1.11.0 - Icon library
- Chart.js 4.4.0 - Performance comparison charts
- Plotly.js 2.27.0 - TSP and graph visualizations

**No Database**: The application intentionally avoids using Django models or any database. All data is ephemeral and lives only in session memory.

**Development Server**: Uses Django's built-in development server (runnable via `manage.py runserver`)

### Key Design Decisions

**Stateless Architecture**: Choosing session-based storage over database persistence simplifies deployment and keeps the application lightweight. Users can experiment with different datasets without worrying about data accumulation.

**Decorator Pattern for Performance Tracking**: Using a single decorator ensures all algorithms report consistent metrics without duplicating measurement code.

**Algorithm Modularity**: Separating algorithms by problem type into different modules makes the codebase easier to maintain and extend with new algorithms.

**Client-Side Result Display**: Having the solve endpoint return JSON and using JavaScript to render results allows for dynamic visualizations and reduces server-side rendering complexity.

**CSRF Exemption**: The `@csrf_exempt` decorator on upload and solve views allows flexible API access for testing and programmatic use, though this should be reviewed for production security.

## Recent Changes

**October 1, 2025** - Initial implementation completed:
- Created Django project with optimization app
- Implemented all three problem types with multiple algorithms each:
  - Knapsack: Greedy, Dynamic Programming, Backtracking, Branch & Bound
  - TSP: Nearest Neighbor (Greedy), Held-Karp (DP with n≤15 limit), Backtracking (n≤12), Branch & Bound (n≤13)
  - Graph Matching: Greedy, Backtracking (limited to 20 edges)
- Created responsive Bootstrap UI with 4 pages (home, upload, algorithm selection, results)
- Integrated Chart.js for performance comparison charts and Plotly.js for TSP route visualization
- Added performance tracking decorator for consistent time/memory metrics
- Implemented session-based in-memory data storage
- All end-to-end tests passing successfully

## Dataset Format Examples

**Knapsack (CSV)**:
```csv
weight,value,capacity
10,60,50
20,100,50
30,120,50
```

**TSP with Coordinates (CSV)**:
```csv
x,y
0,0
4,0
4,3
0,3
```

**Graph Matching (CSV)**:
```csv
u,v,weight
0,1,5
1,2,3
2,3,7
```

## Running the Application

1. Start the server: `python manage.py runserver 0.0.0.0:5000`
2. Access the app at `http://localhost:5000`
3. Upload a dataset in CSV or JSON format
4. Select problem type and algorithms to compare
5. View results with interactive charts and visualizations