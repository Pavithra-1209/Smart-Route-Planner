"""Multi-stop route planning on a weighted road network.

The planner supports two modes:
- automatic: finds the minimum-total-distance order of up to five stops;
- selected: visits stops in the exact order supplied by the user.

It uses target-aware Dijkstra searches from each selected waypoint. Other
selected waypoints are treated as terminal nodes during each leg search, so a
leg cannot silently pass through a stop that is scheduled for a later leg.
The planner then evaluates at most 5! = 120 stop orders.
"""

from __future__ import annotations

import heapq
import itertools
import math
from typing import Any, Iterable

import networkx as nx


def _reconstruct_path(predecessors: dict[Any, Any], source: Any, target: Any) -> list[Any]:
    """Reconstruct a settled Dijkstra path from its predecessor map."""
    path = [target]
    current = target
    while current != source:
        if current not in predecessors:
            raise nx.NetworkXNoPath(
                f"No path exists between waypoint nodes {source} and {target}."
            )
        current = predecessors[current]
        path.append(current)
    path.reverse()
    return path


def _dijkstra_paths_to_targets(
    graph: nx.Graph,
    source: Any,
    targets: Iterable[Any],
) -> dict[Any, dict[str, Any]]:
    """Find shortest paths from source to targets, treating targets as terminals.

    Once a target waypoint is settled, it is recorded but not expanded. This
    prevents a route leg from passing through another selected waypoint before
    that waypoint's scheduled turn in the route.
    """
    targets_remaining = set(targets)
    targets_remaining.discard(source)
    if not targets_remaining:
        return {}

    distances: dict[Any, float] = {source: 0.0}
    predecessors: dict[Any, Any] = {}
    queue: list[tuple[float, int, Any]] = []
    sequence = itertools.count()
    heapq.heappush(queue, (0.0, next(sequence), source))
    settled: set[Any] = set()
    found: dict[Any, dict[str, Any]] = {}

    while queue and targets_remaining:
        current_distance, _, node = heapq.heappop(queue)
        if node in settled:
            continue
        if current_distance > distances.get(node, math.inf):
            continue

        settled.add(node)
        if node in targets_remaining:
            found[node] = {
                "distance": float(current_distance),
                "path": _reconstruct_path(predecessors, source, node),
            }
            targets_remaining.remove(node)
            # A selected waypoint is an endpoint for this leg, not a node to
            # travel through on the way to a different selected waypoint.
            continue

        for neighbor, edge_data in graph[node].items():
            try:
                edge_length = float(edge_data.get("length", 1.0))
            except (TypeError, ValueError):
                continue

            if not math.isfinite(edge_length) or edge_length < 0:
                continue

            new_distance = current_distance + edge_length
            if new_distance < distances.get(neighbor, math.inf):
                distances[neighbor] = new_distance
                predecessors[neighbor] = node
                heapq.heappush(
                    queue,
                    (new_distance, next(sequence), neighbor),
                )

    return found


def find_multi_stop_route(
    weighted_graph: nx.Graph,
    source: Any,
    stop_nodes: Iterable[Any],
    destination: Any,
    *,
    optimize_order: bool = True,
    max_stops: int = 5,
) -> dict[str, Any]:
    """Plan a shortest-distance road trip through multiple stops.

    Args:
        weighted_graph: Simple NetworkX Graph/DiGraph with nonnegative
            ``length`` edge attributes (metres for an OSMnx network).
        source: Road-graph node for the starting location.
        stop_nodes: Intermediate stop nodes. Up to five are supported.
        destination: Road-graph node for the final destination.
        optimize_order: If true, select the stop order with minimum total
            road distance. If false, preserve the supplied stop order.
        max_stops: Maximum permitted number of intermediate stops.

    Returns:
        A dictionary containing the selected stop order, complete path,
        individual leg paths/distances, total distance, and evaluated-order
        count.

    Raises:
        ValueError: Invalid or duplicate waypoint input.
        networkx.NetworkXNoPath: No connected route covers all waypoints.
    """
    stops = list(stop_nodes)

    if weighted_graph.is_multigraph():
        raise ValueError(
            "Pass a simple weighted Graph/DiGraph. Convert the OSMnx "
            "MultiDiGraph using build_simple_weighted_graph first."
        )
    if max_stops < 1:
        raise ValueError("max_stops must be at least 1.")
    if not stops:
        raise ValueError("Select at least one intermediate stop.")
    if len(stops) > max_stops:
        raise ValueError(f"Select no more than {max_stops} intermediate stops.")

    all_waypoints = [source, *stops, destination]
    if len(set(all_waypoints)) != len(all_waypoints):
        raise ValueError(
            "The source, destination, and intermediate stops must snap to "
            "different road-network intersections. Choose different locations."
        )

    missing_nodes = [node for node in all_waypoints if node not in weighted_graph]
    if missing_nodes:
        raise nx.NodeNotFound(f"Waypoint node(s) not found in graph: {missing_nodes}")

    # Each waypoint is treated as a terminal during a leg search. One
    # target-aware Dijkstra run per origin calculates the shortest admissible
    # path to every other waypoint without passing through another waypoint.
    pair_routes: dict[tuple[Any, Any], dict[str, Any]] = {}
    for origin in all_waypoints[:-1]:
        found = _dijkstra_paths_to_targets(
            weighted_graph,
            origin,
            (node for node in all_waypoints if node != origin),
        )
        for target, route in found.items():
            pair_routes[(origin, target)] = route

    if optimize_order:
        best_order: tuple[Any, ...] | None = None
        best_total = math.inf
        evaluated_orders = 0

        for candidate_order in itertools.permutations(stops):
            evaluated_orders += 1
            sequence = (source, *candidate_order, destination)
            total = 0.0
            feasible = True

            for from_node, to_node in zip(sequence[:-1], sequence[1:]):
                leg = pair_routes.get((from_node, to_node))
                if leg is None:
                    feasible = False
                    break
                total += leg["distance"]

            if feasible and total < best_total:
                best_total = total
                best_order = candidate_order

        if best_order is None:
            raise nx.NetworkXNoPath(
                "No complete route visits all selected stops and reaches the destination."
            )
        ordered_stops = list(best_order)
    else:
        ordered_stops = stops
        evaluated_orders = 1

    ordered_waypoints = [source, *ordered_stops, destination]
    leg_routes: list[dict[str, Any]] = []
    combined_path: list[Any] = []

    for from_node, to_node in zip(ordered_waypoints[:-1], ordered_waypoints[1:]):
        leg = pair_routes.get((from_node, to_node))
        if leg is None:
            raise nx.NetworkXNoPath(
                f"No road route exists between waypoint nodes {from_node} and {to_node}."
            )

        leg_routes.append(leg)
        if combined_path:
            combined_path.extend(leg["path"][1:])
        else:
            combined_path.extend(leg["path"])

    leg_paths = [route["path"] for route in leg_routes]
    leg_distances = [route["distance"] for route in leg_routes]

    return {
        "ordered_stops": ordered_stops,
        "ordered_waypoints": ordered_waypoints,
        "path": combined_path,
        "leg_paths": leg_paths,
        "leg_distances": leg_distances,
        "distance": float(sum(leg_distances)),
        "optimized": bool(optimize_order),
        "orders_evaluated": evaluated_orders,
    }
