import networkx as nx
import pytest

from algorithms.multi_stop import find_multi_stop_route


def make_graph():
    graph = nx.DiGraph()
    graph.add_edge("S", "A", length=2.0)
    graph.add_edge("A", "B", length=2.0)
    graph.add_edge("B", "T", length=2.0)
    graph.add_edge("S", "B", length=8.0)
    graph.add_edge("A", "T", length=8.0)
    graph.add_edge("B", "A", length=7.0)
    return graph


def test_automatic_mode_finds_best_order():
    result = find_multi_stop_route(
        make_graph(), "S", ["B", "A"], "T", optimize_order=True
    )
    assert result["ordered_stops"] == ["A", "B"]
    assert result["distance"] == pytest.approx(6.0)
    assert result["orders_evaluated"] == 2


def test_selected_order_is_preserved():
    result = find_multi_stop_route(
        make_graph(), "S", ["B", "A"], "T", optimize_order=False
    )
    assert result["ordered_stops"] == ["B", "A"]
    assert result["distance"] == pytest.approx(23.0)
    assert result["orders_evaluated"] == 1


def test_unreachable_stop_raises_no_path():
    graph = nx.DiGraph()
    graph.add_edge(1, 2, length=1.0)
    graph.add_edge(2, 4, length=1.0)
    graph.add_node(3)

    with pytest.raises(nx.NetworkXNoPath):
        find_multi_stop_route(graph, 1, [3], 4)


def test_duplicate_waypoint_is_rejected():
    with pytest.raises(ValueError, match="different road-network intersections"):
        find_multi_stop_route(make_graph(), "S", ["A"], "A")


def test_more_than_five_stops_is_rejected():
    graph = nx.DiGraph()
    nodes = list(range(7))
    for left, right in zip(nodes[:-1], nodes[1:]):
        graph.add_edge(left, right, length=1.0)

    with pytest.raises(ValueError, match="no more than 5"):
        find_multi_stop_route(graph, 0, [1, 2, 3, 4, 5, 6], 6)
