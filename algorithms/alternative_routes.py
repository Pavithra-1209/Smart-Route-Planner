"""Fast generation of short, diverse alternative routes.

The input should be a simple NetworkX Graph/DiGraph with non-negative
``length`` edge attributes. For OSMnx MultiDiGraph data, call
``build_simple_weighted_graph`` once and cache the result.

Instead of enumerating many K-shortest paths with Yen's algorithm, this module
runs a small number of bidirectional Dijkstra searches. Each later search
penalizes road segments already used by selected routes, encouraging the next
route to take different roads. The resulting routes are ranked by their true
(unpenalized) road distance.
"""

import math

import networkx as nx


def build_simple_weighted_graph(graph):
    """Convert an OSMnx graph to a simple weighted graph, preserving direction.

    When multiple OSM edges connect the same ordered node pair, retain the
    shortest edge by ``length``. This is compatible with
    ``nx.bidirectional_dijkstra`` and avoids repeatedly processing parallel
    edges during route searches.
    """
    simple_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()
    simple_graph.add_nodes_from(graph.nodes(data=True))

    for u, v, data in graph.edges(data=True):
        try:
            length = float(data.get("length", 1.0))
        except (TypeError, ValueError):
            continue

        if not math.isfinite(length) or length < 0:
            continue

        existing = simple_graph.get_edge_data(u, v)
        if existing is None or length < existing["length"]:
            simple_graph.add_edge(u, v, length=length)

    return simple_graph


def _path_edges(path, directed):
    edges = zip(path[:-1], path[1:])
    if directed:
        return set(edges)
    return {frozenset((u, v)) for u, v in edges}


def _shared_segment_ratio(path_a, path_b, directed):
    """Fraction of the smaller route's road segments shared by both paths."""
    edges_a = _path_edges(path_a, directed)
    edges_b = _path_edges(path_b, directed)
    denominator = min(len(edges_a), len(edges_b))
    if denominator == 0:
        return 0.0
    return len(edges_a & edges_b) / denominator


def _route_distance(graph, path):
    return float(sum(graph[u][v]["length"] for u, v in zip(path[:-1], path[1:])))


def find_alternative_routes(
    weighted_graph,
    source,
    destination,
    k=3,
    max_overlap=0.80,
    max_distance_ratio=1.50,
    penalty_levels=(1.8, 2.5, 4.0, 7.0),
):
    """Find up to ``k`` short, diverse routes using a bounded number of searches.

    The shortest route is found first. For each additional route, a few
    bidirectional Dijkstra searches are run with progressively larger
    multiplicative penalties on segments used by already selected routes.
    Candidate routes are evaluated using their original, unpenalized distance.

    This is a fast diversity-oriented heuristic, not an exhaustive guarantee
    of the mathematically exact K-shortest paths.

    Args:
        weighted_graph: Simple NetworkX Graph/DiGraph with ``length`` weights.
        source: Starting node ID.
        destination: Destination node ID.
        k: Maximum number of routes to return, including the shortest route.
        max_overlap: Maximum shared-segment ratio allowed against any selected
            route. 0.80 means at most 80% of the smaller route may be shared.
        max_distance_ratio: Maximum true route distance relative to the
            shortest route. Use None to disable this limit.
        penalty_levels: Small sequence of penalty multipliers. Higher values
            make already-used road segments less attractive in the next search.

    Returns:
        Route dictionaries with ``path``, ``distance`` (metres when OSMnx
        lengths are metres), and ``max_shared_ratio``.
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer.")
    if not 0.0 <= max_overlap <= 1.0:
        raise ValueError("max_overlap must be between 0 and 1.")
    if max_distance_ratio is not None and max_distance_ratio < 1.0:
        raise ValueError("max_distance_ratio must be at least 1.0 or None.")
    if not penalty_levels or any(float(value) <= 1.0 for value in penalty_levels):
        raise ValueError("Every penalty level must be greater than 1.0.")
    if source not in weighted_graph:
        raise nx.NodeNotFound(f"Source node {source} is not in the road graph.")
    if destination not in weighted_graph:
        raise nx.NodeNotFound(f"Destination node {destination} is not in the road graph.")
    if source == destination:
        return [{"path": [source], "distance": 0.0, "max_shared_ratio": 0.0}]

    directed = weighted_graph.is_directed()

    try:
        shortest_cost, shortest_path = nx.bidirectional_dijkstra(
            weighted_graph, source, destination, weight="length"
        )
    except nx.NetworkXNoPath:
        return []

    shortest_distance = _route_distance(weighted_graph, shortest_path)
    selected = [{
        "path": shortest_path,
        "distance": shortest_distance,
        "max_shared_ratio": 0.0,
    }]

    # We only perform a bounded number of searches: one baseline search plus
    # len(penalty_levels) searches for each requested additional route.
    while len(selected) < k:
        used_edges = set()
        for route in selected:
            used_edges.update(_path_edges(route["path"], directed))

        accepted = None
        for penalty in penalty_levels:
            penalty = float(penalty)

            def penalized_weight(u, v, data, used=used_edges, factor=penalty):
                base_length = float(data.get("length", 1.0))
                edge_key = (u, v) if directed else frozenset((u, v))
                return base_length * factor if edge_key in used else base_length

            try:
                _, candidate_path = nx.bidirectional_dijkstra(
                    weighted_graph,
                    source,
                    destination,
                    weight=penalized_weight,
                )
            except nx.NetworkXNoPath:
                continue

            # Avoid exact duplicate paths even if a penalty search returns one.
            if any(candidate_path == route["path"] for route in selected):
                continue

            candidate_distance = _route_distance(weighted_graph, candidate_path)
            if (
                max_distance_ratio is not None
                and candidate_distance > shortest_distance * max_distance_ratio
            ):
                continue

            overlap = max(
                _shared_segment_ratio(candidate_path, route["path"], directed)
                for route in selected
            )
            if overlap > max_overlap:
                continue

            accepted = {
                "path": candidate_path,
                "distance": candidate_distance,
                "max_shared_ratio": overlap,
            }
            break

        if accepted is None:
            break
        selected.append(accepted)

    # Show routes in ascending order of their actual road distance.
    selected.sort(key=lambda route: route["distance"])

    # Recalculate overlap metadata in the final display order.
    for index, route in enumerate(selected):
        if index == 0:
            route["max_shared_ratio"] = 0.0
        else:
            route["max_shared_ratio"] = max(
                _shared_segment_ratio(route["path"], previous["path"], directed)
                for previous in selected[:index]
            )

    return selected
