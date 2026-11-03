
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


def heuristic(G, node, destination):

    lat1 = math.radians(G.nodes[node]["y"])
    lon1 = math.radians(G.nodes[node]["x"])

    lat2 = math.radians(G.nodes[destination]["y"])
    lon2 = math.radians(G.nodes[destination]["x"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    a = min(1.0, max(0.0, a))

    earth_radius = 6371000

    distance = 2 * earth_radius * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return distance


def astar(G, source, destination):

    g_score = {source: 0.0}

    parent = {source: None}

    pq = [
        (
            heuristic(G, source, destination),
            0.0,
            source
        )
    ]

    explored_nodes = 0

    # Store the exploration sequence
    explored_order = []

    while pq:

        f_score, current_g, current_node = heapq.heappop(pq)

        # Ignore outdated queue entries
        if current_g > g_score.get(
            current_node, math.inf
        ):
            continue

        explored_nodes += 1

        # Record explored node
        explored_order.append(current_node)

        # Destination reached
        if current_node == destination:
            break

        for neighbor in G.neighbors(current_node):

            edge_distance = get_edge_distance(
                G, current_node, neighbor
            )

            tentative_g = current_g + edge_distance

            if tentative_g < g_score.get(
                neighbor, math.inf
            ):

                g_score[neighbor] = tentative_g

                parent[neighbor] = current_node

                h_score = heuristic(
                    G, neighbor, destination
                )

                f_score = tentative_g + h_score

                heapq.heappush(
                    pq,
                    (f_score, tentative_g, neighbor)
                )

    # Destination not reachable
    if destination not in g_score:

        return (
            None,
            math.inf,
            explored_nodes,
            explored_order
        )

    # Reconstruct shortest path
    path = []

    node = destination

    while node is not None:

        path.append(node)

        node = parent[node]

    path.reverse()

    return (
        path,
        g_score[destination],
        explored_nodes,
        explored_order
    )
