
import heapq
import math


def get_edge_distance(G, u, v):

    edge_data = G.get_edge_data(u, v)

    if G.is_multigraph():
        return min(
            data["length"]
            for data in edge_data.values()
        )

    return edge_data["length"]


def dijkstra(G, source, destination):

    distances = {source: 0.0}

    parent = {source: None}

    pq = [(0.0, source)]

    explored_nodes = 0

    # Store the order in which nodes are explored
    explored_order = []

    while pq:

        current_distance, current_node = heapq.heappop(pq)

        # Ignore outdated queue entries
        if current_distance > distances.get(
            current_node, math.inf
        ):
            continue

        explored_nodes += 1

        # Record the explored node
        explored_order.append(current_node)

        # Destination reached
        if current_node == destination:
            break

        # Explore neighbours
        for neighbor in G.neighbors(current_node):

            edge_distance = get_edge_distance(
                G, current_node, neighbor
            )

            new_distance = current_distance + edge_distance

            if new_distance < distances.get(
                neighbor, math.inf
            ):

                distances[neighbor] = new_distance

                parent[neighbor] = current_node

                heapq.heappush(
                    pq,
                    (new_distance, neighbor)
                )

    # Destination not reachable
    if destination not in distances:

        return (
            None,
            math.inf,
            explored_nodes,
            explored_order
        )

    # Reconstruct the shortest path
    path = []

    node = destination

    while node is not None:

        path.append(node)

        node = parent[node]

    path.reverse()

    return (
        path,
        distances[destination],
        explored_nodes,
        explored_order
    )
