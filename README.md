# Smart Route Planner

### An Intelligent Route-Finding System Using Dijkstra's and A* Search Algorithms

## About the Project

Smart Route Planner is an AI-based route-finding application that identifies the shortest road distance between two selected locations.

The project implements and compares two Artificial Intelligence search algorithms: **Dijkstra's Algorithm** and **A* Search Algorithm**.

It uses real-world road network data obtained from OpenStreetMap through OSMnx. The road network is represented as a weighted graph using NetworkX, where nodes represent road intersections and edges represent road connections with distance-based weights.

Both algorithms calculate the shortest route independently, and the results are displayed on an interactive map using Folium.

## Features

- Source and destination selection using dropdown menus.
- Real-world road network data from OpenStreetMap.
- Shortest-path calculation using Dijkstra's algorithm.
- Shortest-path calculation using A* search.
- Interactive map visualization using Folium.
- Separate route visualization for both algorithms.
- Route distance comparison.
- Execution-time measurement.
- Number of explored nodes comparison.
- Performance visualization using charts.

## Technology Stack

- **Python** – Programming language
- **Streamlit** – Web application interface
- **OSMnx** – Road network data collection
- **NetworkX** – Graph representation and management
- **OpenStreetMap** – Real-world geographical road data
- **Folium** – Interactive map visualization
- **Matplotlib** – Performance comparison charts
- **heapq** – Priority queue implementation

## Algorithms Used

### 1. Dijkstra's Algorithm

Dijkstra's algorithm is an uninformed search technique that finds the minimum-cost path by exploring nodes based on their accumulated distance from the source.

### 2. A* Search Algorithm

A* is an informed search algorithm that uses a heuristic to guide the search towards the destination.

It uses the evaluation function:

**f(n) = g(n) + h(n)**

Where:
- g(n) represents the actual distance from the source.
- h(n) represents the estimated distance to the destination.
- f(n) represents the estimated total route cost.

The Haversine formula is used to calculate geographical straight-line distance as the heuristic.

## Performance Evaluation

Both algorithms are compared using:

- Shortest route distance
- Execution time
- Number of nodes explored
- Number of nodes in the calculated path

In an initial Guntur–Vijayawada test, both algorithms calculated a shortest route distance of 32.136 km.

A* explored 12,817 nodes, compared with 35,997 nodes explored by Dijkstra.

## Installation and Setup

**1. Clone the repository**

```bash
git clone https://github.com/Pavithra-1209/Smart-Route-Planner.git
```

**2. Navigate to the project directory**

```bash
cd Smart-Route-Planner
```

**3. Create a virtual environment**

```bash
python -m venv .venv
```

**4. Activate the environment**

Windows:

```bash
.venv\Scripts\activate
```

**5. Install dependencies**

```bash
pip install -r requirements.txt
```

**6. Download the road network**

```bash
python road_network/download_graph.py
```

The road network is downloaded and saved locally in GraphML format.

**7. Run the application**

```bash
streamlit run app.py
```

## Project Structure

```text
Smart-Route-Planner/
│
├── algorithms/
│   ├── dijkstra.py
│   └── astar.py
│
├── road_network/
│   └── download_graph.py
│
├── app.py
├── test_algorithms.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Limitations

- The initial road network covers selected areas around Guntur and Vijayawada.
- The system focuses on shortest distance rather than fastest travel time.
- Live traffic conditions are not considered.
- Internet connectivity is required for downloading road data, geocoding and displaying interactive maps.

## Future Enhancements

- Support for additional geographical regions.
- Alternative route suggestions.
- Live traffic integration.
- Travel-time-based route planning.
- Comparison with additional search algorithms.

## Data Source

Road network data is obtained from OpenStreetMap.

https://www.openstreetmap.org/copyright

