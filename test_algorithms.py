
import osmnx as ox
import time
import math
from pathlib import Path

from algorithms.dijkstra import dijkstra
from algorithms.astar import astar


# Load the downloaded road network
GRAPH_FILE = (
    Path(__file__).parent
    / "road_network"
    / "guntur_vijayawada.graphml"
)

print("Loading road network...")

G = ox.load_graphml(filepath=GRAPH_FILE)

print("Road network loaded!")
print("Nodes:", len(G.nodes))
print("Edges:", len(G.edges))


# Select source and destination
print("\nFinding source and destination coordinates...")

source_lat, source_lon = ox.geocode(
    "Guntur, Andhra Pradesh, India"
)

time.sleep(1)

destination_lat, destination_lon = ox.geocode(
    "Vijayawada, Andhra Pradesh, India"
)


# Find nearest road nodes
source_node = ox.distance.nearest_nodes(
    G, X=source_lon, Y=source_lat
)

destination_node = ox.distance.nearest_nodes(
    G, X=destination_lon, Y=destination_lat
)

print("\nSource node:", source_node)
print("Destination node:", destination_node)


# Run Dijkstra
print("\nRunning Dijkstra...")

start = time.perf_counter()

dijkstra_path, dijkstra_distance, dijkstra_explored, dijkstra_order = (
    dijkstra(G, source_node, destination_node)
)



dijkstra_time = (time.perf_counter() - start) * 1000


# Run A*
print("Running A*...")

start = time.perf_counter()

astar_path, astar_distance, astar_explored, astar_order = (
    astar(G, source_node, destination_node)
)

astar_time = (time.perf_counter() - start) * 1000


# Display results
print("\n========== DIJKSTRA RESULTS ==========")

if dijkstra_path is not None:
    print("Distance:", round(dijkstra_distance / 1000, 3), "km")
    print("Execution time:", round(dijkstra_time, 3), "ms")
    print("Nodes explored:", dijkstra_explored)
    print("Path nodes:", len(dijkstra_path))
else:
    print("No route found.")


print("\n========== A* RESULTS ==========")

if astar_path is not None:
    print("Distance:", round(astar_distance / 1000, 3), "km")
    print("Execution time:", round(astar_time, 3), "ms")
    print("Nodes explored:", astar_explored)
    print("Path nodes:", len(astar_path))
else:
    print("No route found.")


# Verify shortest-distance agreement
print("\n========== COMPARISON ==========")

if dijkstra_path is not None and astar_path is not None:

    print(
        "Same shortest distance:",
        math.isclose(
            dijkstra_distance,
            astar_distance,
            rel_tol=1e-9,
            abs_tol=1e-6
        )
    )
