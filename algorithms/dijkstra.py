
import heapq
import math


def get_edge_distance(G, u, v):
    """
    Get the minimum distance between two connected nodes.
    Handles multiple edges between the same pair of nodes.
    """
    edge_data = G.get_edge_data(u, v)

    if G.is_multigraph():
        return min(data["length"] for data in edge_data.values())

    return edge_data["length"]


def dijkstra(G, source, destination):

    # Store the minimum distance to each node
    distances = {source: 0.0}

    # Store the previous node to reconstruct the route
    parent = {source: None}

    # Priority queue: (distance, node)
    pq = [(0.0, source)]

    explored_nodes = 0

    while pq:

        current_distance, current_node = heapq.heappop(pq)

        # Ignore outdated entries
        if current_distance > distances.get(current_node, math.inf):
            continue

        explored_nodes += 1

        # Destination reached
        if current_node == destination:
            break

        # Explore neighbouring nodes
        for neighbor in G.neighbors(current_node):

            edge_distance = get_edge_distance(
                G, current_node, neighbor
            )

            new_distance = current_distance + edge_distance

            # Update if a shorter path is found
            if new_distance < distances.get(neighbor, math.inf):

                distances[neighbor] = new_distance
                parent[neighbor] = current_node

                heapq.heappush(
                    pq, (new_distance, neighbor)
                )

    # Check whether destination is reachable
    if destination not in distances:
        return None, math.inf, explored_nodes

    # Reconstruct shortest path
    path = []
    node = destination

    while node is not None:
        path.append(node)
        node = parent[node]

    path.reverse()

    return path, distances[destination], explored_nodes
