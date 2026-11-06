import math
import time
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import osmnx as ox
import streamlit as st
from streamlit_folium import st_folium

from algorithms.astar import astar
from algorithms.dijkstra import dijkstra
from algorithms.alternative_routes import (
    build_simple_weighted_graph,
    find_alternative_routes,
)
from algorithms.multi_stop import find_multi_stop_route


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Smart Route Planner",
    page_icon="🗺️",
    layout="wide",
)


# =====================================================
# 2. CUSTOM STYLING
# =====================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #F4F7FC;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(120deg, #14233B, #1E40AF);
        padding: 30px;
        border-radius: 18px;
        margin-bottom: 25px;
    }

    .hero-tag {
        color: #93C5FD;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
    }

    .hero h1 {
        color: white;
        font-size: 34px;
        font-weight: 750;
        margin: 8px 0 5px 0;
    }

    .hero p {
        color: #D8E5F7;
        font-size: 15px;
        margin-bottom: 0;
    }

    .section-heading {
        color: #172B4D;
        font-size: 21px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 43px;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #E2E8F0;
        border-radius: 13px;
        padding: 17px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    div[data-testid="stMetricLabel"] {
        color: #64748B;
    }

    div[data-testid="stMetricValue"] {
        color: #14233B;
        font-weight: 750;
    }

    .footer {
        text-align: center;
        color: #64748B;
        font-size: 12px;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# 3. HEADER
# =====================================================

st.markdown(
    """
<div class="hero">
<div class="hero-tag">INTELLIGENT NAVIGATION SYSTEM</div>
<h1>🗺️ Smart Route Planner</h1>
<p>Discover the shortest road route using Dijkstra's and A* Search Algorithms.</p>
</div>
""",
    unsafe_allow_html=True,
)


# =====================================================
# 4. LOCATION OPTIONS
# =====================================================

# These are suggested locations. A selected location must still fall
# within the downloaded graph's geographical coverage and be connected.
LOCATION_OPTIONS = {
    "Guntur Railway Station": (
        "Guntur Railway Station, Guntur, Andhra Pradesh, India"
    ),
    "Guntur NTR Bus Station": (
        "NTR Bus Station, Guntur, Andhra Pradesh, India"
    ),
    "Brodipet, Guntur": "Brodipet, Guntur, Andhra Pradesh, India",
    "Arundelpet, Guntur": "Arundelpet, Guntur, Andhra Pradesh, India",
    "Lakshmipuram, Guntur": "Lakshmipuram, Guntur, Andhra Pradesh, India",
    "Vijayawada Railway Station": (
        "Vijayawada Railway Station, Vijayawada, Andhra Pradesh, India"
    ),
    "Pandit Nehru Bus Station": (
        "Pandit Nehru Bus Station, Vijayawada, Andhra Pradesh, India"
    ),
    "Benz Circle, Vijayawada": (
        "Benz Circle, Vijayawada, Andhra Pradesh, India"
    ),
    "Mangalagiri": "Mangalagiri, Andhra Pradesh, India",
    "Kanaka Durga Temple": (
        "Kanaka Durga Temple, Vijayawada, Andhra Pradesh, India"
    ),
}


# =====================================================
# 5. SESSION STATE
# =====================================================

if "route_result" not in st.session_state:
    st.session_state.route_result = None

if "route_error" not in st.session_state:
    st.session_state.route_error = None

if "source_choice" not in st.session_state:
    st.session_state.source_choice = "Guntur Railway Station"

if "destination_choice" not in st.session_state:
    st.session_state.destination_choice = "Vijayawada Railway Station"


def swap_locations():
    """Swap the currently selected source and destination."""
    source = st.session_state.source_choice
    destination = st.session_state.destination_choice
    st.session_state.source_choice = destination
    st.session_state.destination_choice = source


# =====================================================
# 6. LOAD THE SAVED ROAD NETWORK
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent
GRAPH_FILE = PROJECT_DIR / "road_network" / "guntur_vijayawada.graphml"


@st.cache_resource(show_spinner=False)
def load_graph():
    """Load the saved OSMnx road graph once per app process."""
    if not GRAPH_FILE.exists():
        raise FileNotFoundError(
            "Road network file not found: "
            "road_network/guntur_vijayawada.graphml. "
            "Run road_network/download_graph.py or include the graph "
            "file in the deployment environment."
        )

    return ox.load_graphml(filepath=GRAPH_FILE)


@st.cache_resource(show_spinner=False)
def load_alternative_graph():
    """Build and cache a simple weighted graph for K-shortest paths."""
    return build_simple_weighted_graph(load_graph())


@st.cache_data(show_spinner=False)
def geocode_location(location):
    """Geocode a place name and cache the returned coordinates."""
    return ox.geocode(location)


@st.cache_data(show_spinner=False)
def get_graph_bounds():
    """Return cached latitude/longitude bounds for the saved road graph."""
    graph = load_graph()
    if graph.number_of_nodes() == 0:
        raise ValueError("The downloaded road network contains no nodes.")

    latitudes = [float(data["y"]) for _, data in graph.nodes(data=True)]
    longitudes = [float(data["x"]) for _, data in graph.nodes(data=True)]
    return min(latitudes), max(latitudes), min(longitudes), max(longitudes)


# =====================================================
# 7. ROUTE CALCULATION
# =====================================================

def calculate_routes(source_location, destination_location):
    """Run Dijkstra and A* on the same saved road graph."""
    graph = load_graph()

    # Convert place names to latitude and longitude.
    source_lat, source_lon = geocode_location(source_location)
    destination_lat, destination_lon = geocode_location(destination_location)

    # Check whether the geocoded points fall inside graph bounds.
    node_data = list(graph.nodes(data=True))
    if not node_data:
        raise ValueError("The downloaded road network contains no nodes.")

    min_lat = min(data["y"] for _, data in node_data)
    max_lat = max(data["y"] for _, data in node_data)
    min_lon = min(data["x"] for _, data in node_data)
    max_lon = max(data["x"] for _, data in node_data)

    for label, lat, lon in (
        ("Source", source_lat, source_lon),
        ("Destination", destination_lat, destination_lon),
    ):
        if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
            raise ValueError(
                f"{label} is outside the downloaded road network coverage. "
                "Choose a location within the supported Guntur–Vijayawada region."
            )

    # Snap the selected geographical points to the nearest graph nodes.
    source_node = ox.distance.nearest_nodes(
        graph, X=source_lon, Y=source_lat
    )
    destination_node = ox.distance.nearest_nodes(
        graph, X=destination_lon, Y=destination_lat
    )

    # Dijkstra returns: path, distance, explored count, explored order.
    start_time = time.perf_counter()
    (
        dijkstra_path,
        dijkstra_distance,
        dijkstra_explored,
        dijkstra_order,
    ) = dijkstra(graph, source_node, destination_node)
    dijkstra_time_ms = (time.perf_counter() - start_time) * 1000

    # A* returns: path, distance, explored count, explored order.
    start_time = time.perf_counter()
    (
        astar_path,
        astar_distance,
        astar_explored,
        astar_order,
    ) = astar(graph, source_node, destination_node)
    astar_time_ms = (time.perf_counter() - start_time) * 1000

    if dijkstra_path is None or astar_path is None:
        raise ValueError(
            "No connected road route was found between these locations."
        )

    return {
        "source_lat": source_lat,
        "source_lon": source_lon,
        "destination_lat": destination_lat,
        "destination_lon": destination_lon,
        "source_node": source_node,
        "destination_node": destination_node,
        "source_display": source_location,
        "destination_display": destination_location,
        "dijkstra_path": dijkstra_path,
        "dijkstra_distance": dijkstra_distance,
        "dijkstra_time": dijkstra_time_ms,
        "dijkstra_explored": dijkstra_explored,
        "dijkstra_order": dijkstra_order,
        "astar_path": astar_path,
        "astar_distance": astar_distance,
        "astar_time": astar_time_ms,
        "astar_explored": astar_explored,
        "astar_order": astar_order,
    }


# =====================================================
# 8. SOURCE AND DESTINATION SELECTION
# =====================================================

st.markdown(
    '<div class="section-heading">📍 Plan Your Journey</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    source_col, swap_col, destination_col = st.columns([5, 1, 5])
    location_names = list(LOCATION_OPTIONS.keys())

    with source_col:
        st.markdown("**STARTING LOCATION**")
        st.selectbox(
            "Select source",
            options=location_names,
            key="source_choice",
            label_visibility="collapsed",
        )

    with swap_col:
        st.markdown("<div style='height: 27px'></div>", unsafe_allow_html=True)
        st.button(
            "⇄",
            help="Swap source and destination",
            on_click=swap_locations,
            use_container_width=True,
        )

    with destination_col:
        st.markdown("**DESTINATION**")
        st.selectbox(
            "Select destination",
            options=location_names,
            key="destination_choice",
            label_visibility="collapsed",
        )

    st.caption(
        "Type in a dropdown to search its options. Locations must be "
        "within the downloaded road-network coverage."
    )

    search_button = st.button(
        "🔍 Find Shortest Route",
        type="primary",
        use_container_width=True,
    )


# =====================================================
# 9. HANDLE A NEW ROUTE SEARCH
# =====================================================

if search_button:
    source_choice = st.session_state.source_choice
    destination_choice = st.session_state.destination_choice

    st.session_state.route_result = None
    st.session_state.route_error = None

    if source_choice == destination_choice:
        st.session_state.route_error = (
            "Source and destination cannot be the same."
        )
    else:
        try:
            source_location = LOCATION_OPTIONS[source_choice]
            destination_location = LOCATION_OPTIONS[destination_choice]

            with st.spinner(
                "Finding locations and calculating routes with Dijkstra and A*..."
            ):
                result = calculate_routes(
                    source_location,
                    destination_location,
                )

            # Store the short labels for display and stale-result checking.
            result["source_display"] = source_choice
            result["destination_display"] = destination_choice
            st.session_state.route_result = result

        except Exception as exc:
            st.session_state.route_error = str(exc)


# =====================================================
# 10. DISPLAY ERRORS AND VALIDATE STORED RESULTS
# =====================================================

if st.session_state.route_error:
    st.error(st.session_state.route_error)

result = st.session_state.route_result

if result and (
    result["source_display"] != st.session_state.source_choice
    or result["destination_display"] != st.session_state.destination_choice
):
    st.info(
        "Your selected locations have changed. Click 'Find Shortest Route' "
        "to calculate the updated journey."
    )
    result = None


# =====================================================
# 11. DISPLAY RESULTS
# =====================================================

if result:
    st.success("Both algorithms calculated their routes successfully.")

    st.markdown(
        '<div class="section-heading">📊 Route Overview</div>',
        unsafe_allow_html=True,
    )

    source_col, destination_col = st.columns(2)
    source_col.info(f"**Source:** {result['source_display']}")
    destination_col.info(f"**Destination:** {result['destination_display']}")

    # Main distance cards.
    dijkstra_col, astar_col = st.columns(2)

    with dijkstra_col:
        with st.container(border=True):
            st.markdown("### 🔵 Dijkstra")
            st.metric(
                "Shortest Distance",
                f"{result['dijkstra_distance'] / 1000:.3f} km",
            )
            st.caption(f"{result['dijkstra_explored']:,} nodes explored")

    with astar_col:
        with st.container(border=True):
            st.markdown("### 🔴 A* Search")
            st.metric(
                "Shortest Distance",
                f"{result['astar_distance'] / 1000:.3f} km",
            )
            st.caption(f"{result['astar_explored']:,} nodes explored")

    # Derived performance metrics.
    distance_difference = abs(
        result["dijkstra_distance"] - result["astar_distance"]
    )

    if result["dijkstra_explored"] > 0:
        node_reduction = (
            (result["dijkstra_explored"] - result["astar_explored"])
            / result["dijkstra_explored"]
        ) * 100
    else:
        node_reduction = 0.0

    same_distance = math.isclose(
        result["dijkstra_distance"],
        result["astar_distance"],
        rel_tol=1e-9,
        abs_tol=1e-6,
    )
    same_path = result["dijkstra_path"] == result["astar_path"]

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Distance Difference", f"{distance_difference:.3f} m")
    metric2.metric("Dijkstra Time", f"{result['dijkstra_time']:.2f} ms")
    metric3.metric("A* Time", f"{result['astar_time']:.2f} ms")
    metric4.metric("A* Node Reduction", f"{node_reduction:+.1f}%")

    if same_distance:
        st.success("Both algorithms found the same minimum route distance.")
    else:
        st.warning(
            "The algorithms returned different route distances. "
            "Verify the graph weights and algorithm implementations."
        )

    if same_path:
        st.info("Both algorithms discovered exactly the same road path.")
    elif same_distance:
        st.info(
            "The algorithms found different paths with the same minimum distance."
        )

    if node_reduction > 0:
        st.caption(
            f"A* explored {node_reduction:.2f}% fewer nodes than Dijkstra "
            "for this route."
        )
    elif node_reduction < 0:
        st.caption(
            f"A* explored {abs(node_reduction):.2f}% more nodes than "
            "Dijkstra for this route."
        )
    else:
        st.caption("Both algorithms explored the same number of nodes.")

    # Application sections.
    route_tab, alternative_tab, multi_stop_tab, search_tab, performance_tab, details_tab = st.tabs([
        "🗺️ Route Map",
        "🛣️ Alternative Routes",
        "📍 Multi-Stop Planner",
        "🔍 Search Visualization",
        "📈 Performance Analysis",
        "📋 Route Details",
    ])
    
    # =================================================
    # 12. SEARCH PROCESS VISUALIZATION - OPTIMIZED
    # =================================================

    with search_tab:

        st.markdown("### 🔍 Search Process Visualization")

        st.write(
            "Observe how Dijkstra and A* explore the road network "
            "before reaching the destination."
        )

        # Select algorithm
        selected_algorithm = st.radio(
            "Choose Algorithm",
            ["Dijkstra", "A* Search"],
            horizontal=True,
            key="visualization_algorithm"
        )

        # Get algorithm data
        if selected_algorithm == "Dijkstra":

            explored_order = result["dijkstra_order"]
            final_path = result["dijkstra_path"]
            route_color = "#2563EB"

        else:

            explored_order = result["astar_order"]
            final_path = result["astar_path"]
            route_color = "#DC2626"

        total_explored = len(explored_order)

        if total_explored == 0:

            st.warning("No explored nodes available.")

        else:

            # -----------------------------------------
            # STATISTICS
            # -----------------------------------------

            col1, col2 = st.columns(2)

            col1.metric(
                "Total Nodes Explored",
                f"{total_explored:,}"
            )

            col2.metric(
                "Final Path Nodes",
                f"{len(final_path):,}"
            )

            st.divider()

            # -----------------------------------------
            # OPTIMIZED SLIDER
            # -----------------------------------------

            step_key = (
                f"visual_step_{result['source_node']}_"
                f"{result['destination_node']}_"
                f"{selected_algorithm}"
            )

            applied_key = step_key + "_applied"

            if applied_key not in st.session_state:
                st.session_state[applied_key] = min(
                    500, total_explored
                )

            # Use approximately 100 slider positions
            slider_step = max(1, total_explored // 100)

            with st.form(key=f"visual_form_{step_key}"):

                selected_step = st.slider(
                    "Exploration Progress",
                    min_value=0,
                    max_value=total_explored,
                    value=min(
                        st.session_state[applied_key],
                        total_explored
                    ),
                    step=slider_step,
                    key=step_key
                )

                apply_step = st.form_submit_button(
                    "▶ Apply Visualization",
                    use_container_width=True
                )

                if apply_step:

                    st.session_state[applied_key] = selected_step

            current_step = st.session_state[applied_key]

            # -----------------------------------------
            # PROGRESS
            # -----------------------------------------

            progress = current_step / total_explored

            st.progress(
                progress,
                text=(
                    f"Explored {current_step:,} "
                    f"out of {total_explored:,} nodes"
                )
            )

            # -----------------------------------------
            # SAMPLE NODES UNIFORMLY
            # -----------------------------------------

            visible_nodes = explored_order[:current_step]

            MAX_MARKERS = 150

            if len(visible_nodes) > MAX_MARKERS:

                interval = max(
                    1,
                    len(visible_nodes) // MAX_MARKERS
                )

                display_nodes = visible_nodes[::interval][
                    :MAX_MARKERS
                ]

            else:

                display_nodes = visible_nodes

            current_node = (
                explored_order[current_step - 1]
                if current_step > 0
                else None
            )

            st.caption(
                f"Showing {len(display_nodes)} sampled nodes "
                f"from {current_step:,} explored nodes."
            )

            # -----------------------------------------
            # CREATE MAP
            # -----------------------------------------

            graph = load_graph()

            source_point = [
                result["source_lat"],
                result["source_lon"]
            ]

            destination_point = [
                result["destination_lat"],
                result["destination_lon"]
            ]

            search_map = folium.Map(
                location=[
                    (source_point[0] + destination_point[0]) / 2,
                    (source_point[1] + destination_point[1]) / 2
                ],
                zoom_start=9,
                control_scale=True,
                prefer_canvas=True
            )

            # -----------------------------------------
            # EXPLORED NODES
            # -----------------------------------------

            explored_layer = folium.FeatureGroup(
                name="Explored Nodes"
            )

            for node in display_nodes:

                lat = graph.nodes[node]["y"]
                lon = graph.nodes[node]["x"]

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=3,
                    color="#F59E0B",
                    fill=True,
                    fill_color="#F59E0B",
                    fill_opacity=0.7,
                    weight=0
                ).add_to(explored_layer)

            explored_layer.add_to(search_map)

            # -----------------------------------------
            # CURRENT NODE
            # -----------------------------------------

            if current_node is not None:

                lat = graph.nodes[current_node]["y"]
                lon = graph.nodes[current_node]["x"]

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=8,
                    color="#7C3AED",
                    fill=True,
                    fill_color="#7C3AED",
                    fill_opacity=1,
                    tooltip="Current Node"
                ).add_to(search_map)

            # -----------------------------------------
            # FINAL SHORTEST PATH
            # -----------------------------------------

            final_coordinates = [
                (graph.nodes[node]["y"], graph.nodes[node]["x"])
                for node in final_path
            ]

            folium.PolyLine(
                locations=final_coordinates,
                color=route_color,
                weight=5,
                opacity=0.9,
                tooltip=f"{selected_algorithm} Final Route"
            ).add_to(search_map)

            # -----------------------------------------
            # SOURCE AND DESTINATION
            # -----------------------------------------

            folium.Marker(
                location=source_point,
                tooltip="Source",
                icon=folium.Icon(
                    color="green",
                    icon="play"
                )
            ).add_to(search_map)

            folium.Marker(
                location=destination_point,
                tooltip="Destination",
                icon=folium.Icon(
                    color="red",
                    icon="flag"
                )
            ).add_to(search_map)

            # -----------------------------------------
            # MAP BOUNDS
            # -----------------------------------------

            search_map.fit_bounds([
                [
                    min(p[0] for p in final_coordinates),
                    min(p[1] for p in final_coordinates)
                ],
                [
                    max(p[0] for p in final_coordinates),
                    max(p[1] for p in final_coordinates)
                ]
            ])

            # -----------------------------------------
            # DISPLAY MAP
            # -----------------------------------------

            st_folium(
                search_map,
                width=1200,
                height=550,
                key=(
                    f"optimized_map_{result['source_node']}_"
                    f"{result['destination_node']}_"
                    f"{selected_algorithm}_{current_step}"
                ),
                returned_objects=[]
            )

            # -----------------------------------------
            # LEGEND
            # -----------------------------------------

            st.markdown("### Map Legend")

            st.markdown(
                """
                🟡 Explored Nodes

                🟣 Current Explored Node

                🟢 Source Location

                🔴 Destination

                🔵 Blue / 🔴 Red Line: Final Shortest Route
                """
            )

    # =================================================
    # 13. FINAL ROUTE MAP
    # =================================================

    with route_tab:
        st.markdown("### Interactive Route Visualization")
        st.markdown(
            "🔵 **Blue solid line:** Dijkstra &nbsp;&nbsp;&nbsp; "
            "🔴 **Red dashed line:** A* Search",
            unsafe_allow_html=True,
        )

        graph = load_graph()
        dijkstra_coordinates = [
            (graph.nodes[node]["y"], graph.nodes[node]["x"])
            for node in result["dijkstra_path"]
        ]
        astar_coordinates = [
            (graph.nodes[node]["y"], graph.nodes[node]["x"])
            for node in result["astar_path"]
        ]

        source_point = [result["source_lat"], result["source_lon"]]
        destination_point = [
            result["destination_lat"],
            result["destination_lon"],
        ]

        route_map = folium.Map(
            location=[
                (source_point[0] + destination_point[0]) / 2,
                (source_point[1] + destination_point[1]) / 2,
            ],
            zoom_start=9,
            control_scale=True,
            prefer_canvas=True,
        )

        dijkstra_layer = folium.FeatureGroup(
            name="🔵 Dijkstra Route",
            show=True,
        )
        folium.PolyLine(
            locations=dijkstra_coordinates,
            color="#2563EB",
            weight=6,
            opacity=0.9,
            tooltip="Dijkstra Shortest Route",
        ).add_to(dijkstra_layer)
        dijkstra_layer.add_to(route_map)

        astar_layer = folium.FeatureGroup(
            name="🔴 A* Route",
            show=True,
        )
        folium.PolyLine(
            locations=astar_coordinates,
            color="#DC2626",
            weight=5,
            opacity=0.9,
            dash_array="10, 10",
            tooltip="A* Shortest Route",
        ).add_to(astar_layer)
        astar_layer.add_to(route_map)

        folium.Marker(
            location=source_point,
            popup=f"Source: {result['source_display']}",
            tooltip="START LOCATION",
            icon=folium.Icon(color="green", icon="play"),
        ).add_to(route_map)

        folium.Marker(
            location=destination_point,
            popup=f"Destination: {result['destination_display']}",
            tooltip="DESTINATION",
            icon=folium.Icon(color="red", icon="flag"),
        ).add_to(route_map)

        all_route_coordinates = (
            dijkstra_coordinates
            + astar_coordinates
            + [tuple(source_point), tuple(destination_point)]
        )
        south = min(point[0] for point in all_route_coordinates)
        north = max(point[0] for point in all_route_coordinates)
        west = min(point[1] for point in all_route_coordinates)
        east = max(point[1] for point in all_route_coordinates)
        route_map.fit_bounds([[south, west], [north, east]])
        folium.LayerControl(collapsed=False).add_to(route_map)

        st_folium(
            route_map,
            width=1200,
            height=600,
            key=(
                f"route_map_{result['source_node']}_"
                f"{result['destination_node']}"
            ),
            returned_objects=[],
        )

        st.caption(
            "Use the map layer control to show or hide each algorithm's route."
        )

    # =================================================
    # 14. ALTERNATIVE ROUTE FINDER
    # =================================================

    with alternative_tab:
        st.markdown("### 🛣️ Alternative Route Finder")
        st.write(
            "Generate up to three routes that are both short and use "
            "different road segments."
        )
        st.caption(
            "For faster results, the planner runs a small number of "
            "bidirectional shortest-path searches. It penalizes roads already "
            "used by selected routes, prefers alternatives with no more than "
            "80% shared road segments, and keeps routes within 1.5 times the "
            "shortest-route distance. If the road network does not contain "
            "enough sufficiently different routes, fewer routes will be shown."
        )

        alternative_key = (
            f"{result['source_node']}_{result['destination_node']}"
        )

        # Discard alternatives from a previous source-destination pair.
        if st.session_state.get("alternative_routes_key") != alternative_key:
            st.session_state["alternative_routes_key"] = alternative_key
            st.session_state["alternative_routes"] = None

        if st.button(
            "🧭 Generate Alternative Routes",
            type="primary",
            use_container_width=True,
            key=f"generate_alternatives_{alternative_key}",
        ):
            try:
                with st.spinner(
                    "Searching for a few distinct routes..."
                ):
                    alternative_graph = load_alternative_graph()
                    alternatives = find_alternative_routes(
                        alternative_graph,
                        result["source_node"],
                        result["destination_node"],
                        k=3,
                        max_overlap=0.80,
                        max_distance_ratio=1.50,
                        penalty_levels=(1.8, 2.5, 4.0, 7.0),
                    )

                st.session_state["alternative_routes"] = alternatives
            except Exception as exc:
                st.session_state["alternative_routes"] = None
                st.error(f"Could not generate alternative routes: {exc}")

        alternatives = st.session_state.get("alternative_routes")

        if alternatives is None:
            st.info(
                "Click **Generate Alternative Routes** to calculate up to "
                "three routes for this journey."
            )
        elif not alternatives:
            st.warning("No route was found for this source and destination.")
        else:
            st.markdown("#### Route Comparison")
            shortest_distance = alternatives[0]["distance"]
            route_names = [
                "Shortest Route" if index == 0 else f"Alternative {index}"
                for index in range(len(alternatives))
            ]

            comparison_rows = []
            for index, route in enumerate(alternatives):
                comparison_rows.append(
                    {
                        "Route": route_names[index],
                        "Distance (km)": round(route["distance"] / 1000, 3),
                        "Extra Distance (km)": round(
                            (route["distance"] - shortest_distance) / 1000, 3
                        ),
                        "Path Nodes": len(route["path"]),
                        "Max Shared Roads (%)": (
                            "—" if index == 0
                            else round(route.get("max_shared_ratio", 0.0) * 100, 1)
                        ),
                    }
                )

            st.dataframe(
                comparison_rows,
                use_container_width=True,
                hide_index=True,
            )

            if len(alternatives) == 1:
                st.info(
                    "Only one route met the diversity and distance limits. "
                    "The road network may not contain another sufficiently "
                    "different route for this journey."
                )
            else:
                st.success(
                    f"Found {len(alternatives)} routes with limited road "
                    "overlap, ranked by total road distance."
                )

            # Draw every available route so the user can compare them visually.
            graph = load_graph()
            route_colors = ["#2563EB", "#DC2626", "#059669"]
            route_dash_patterns = [None, "10, 8", "2, 8"]

            source_point = [result["source_lat"], result["source_lon"]]
            destination_point = [
                result["destination_lat"],
                result["destination_lon"],
            ]

            alternative_map = folium.Map(
                location=[
                    (source_point[0] + destination_point[0]) / 2,
                    (source_point[1] + destination_point[1]) / 2,
                ],
                zoom_start=9,
                control_scale=True,
                prefer_canvas=True,
            )

            all_coordinates = [tuple(source_point), tuple(destination_point)]

            for index, route in enumerate(alternatives):
                route_coordinates = [
                    (graph.nodes[node]["y"], graph.nodes[node]["x"])
                    for node in route["path"]
                ]
                all_coordinates.extend(route_coordinates)

                route_layer = folium.FeatureGroup(
                    name=(
                        f"{route_names[index]} "
                        f"({route['distance'] / 1000:.2f} km)"
                    ),
                    show=True,
                )

                line_options = {
                    "locations": route_coordinates,
                    "color": route_colors[index % len(route_colors)],
                    "weight": 6 if index == 0 else 5,
                    "opacity": 0.9,
                    "tooltip": (
                        f"{route_names[index]} — "
                        f"{route['distance'] / 1000:.3f} km"
                    ),
                }
                dash_pattern = route_dash_patterns[
                    index % len(route_dash_patterns)
                ]
                if dash_pattern:
                    line_options["dash_array"] = dash_pattern

                folium.PolyLine(**line_options).add_to(route_layer)
                route_layer.add_to(alternative_map)

            folium.Marker(
                location=source_point,
                popup=f"Source: {result['source_display']}",
                tooltip="START LOCATION",
                icon=folium.Icon(color="green", icon="play"),
            ).add_to(alternative_map)

            folium.Marker(
                location=destination_point,
                popup=f"Destination: {result['destination_display']}",
                tooltip="DESTINATION",
                icon=folium.Icon(color="red", icon="flag"),
            ).add_to(alternative_map)

            south = min(point[0] for point in all_coordinates)
            north = max(point[0] for point in all_coordinates)
            west = min(point[1] for point in all_coordinates)
            east = max(point[1] for point in all_coordinates)
            alternative_map.fit_bounds([[south, west], [north, east]])
            folium.LayerControl(collapsed=False).add_to(alternative_map)

            st.markdown("#### Alternative Routes on the Map")
            st_folium(
                alternative_map,
                width=1200,
                height=600,
                key=f"alternative_map_{alternative_key}",
                returned_objects=[],
            )
            st.caption(
                "Blue: shortest route · Red: alternative 1 · Green: alternative 2. "
                "The shared-road percentage in the table is measured against "
                "the most similar route already selected. Use the layer control "
                "to show or hide routes."
            )

    # =================================================
    # 15. MULTI-STOP ROUTE PLANNER
    # =================================================

    with multi_stop_tab:
        st.markdown("### 📍 Multi-Stop Route Planner")
        st.write(
            "Choose intermediate places between your current source and "
            "destination. The planner can automatically find the visiting "
            "order with the smallest total road distance."
        )
        st.caption(
            "Select up to five intermediate stops. Automatic optimization "
            "checks every possible stop order using road-network distances "
            "and then reconstructs the shortest path for each journey leg."
        )

        source_name = result["source_display"]
        destination_name = result["destination_display"]
        stop_options = [
            name for name in LOCATION_OPTIONS
            if name not in (source_name, destination_name)
        ]

        multi_stop_key = (
            f"{result['source_node']}_{result['destination_node']}"
        )
        selected_stops = st.multiselect(
            "Select intermediate stops (1–5)",
            options=stop_options,
            max_selections=5,
            key=f"multi_stop_selection_{multi_stop_key}",
            placeholder="Choose places to visit along the journey",
        )

        planning_mode = st.radio(
            "Stop visiting order",
            [
                "Automatically optimize order",
                "Keep my selected order",
            ],
            horizontal=True,
            key=f"multi_stop_mode_{multi_stop_key}",
            help=(
                "Automatic mode finds the lowest-total-distance order. "
                "Selected-order mode visits the stops in the order you chose."
            ),
        )
        optimize_order = planning_mode == "Automatically optimize order"

        request_key = "|".join(
            [
                str(result["source_node"]),
                str(result["destination_node"]),
                planning_mode,
                *selected_stops,
            ]
        )

        if st.button(
            "🧭 Calculate Multi-Stop Route",
            type="primary",
            use_container_width=True,
            key=f"calculate_multi_stop_{multi_stop_key}",
        ):
            if not selected_stops:
                st.warning("Please select at least one intermediate stop.")
                st.session_state["multi_stop_result"] = None
                st.session_state["multi_stop_result_key"] = None
            else:
                try:
                    graph = load_graph()
                    min_lat, max_lat, min_lon, max_lon = get_graph_bounds()
                    stop_details = []

                    with st.spinner(
                        "Finding road-network locations and optimizing the stop order..."
                    ):
                        for stop_name in selected_stops:
                            latitude, longitude = geocode_location(
                                LOCATION_OPTIONS[stop_name]
                            )

                            if not (
                                min_lat <= latitude <= max_lat
                                and min_lon <= longitude <= max_lon
                            ):
                                raise ValueError(
                                    f"{stop_name} is outside the downloaded "
                                    "Guntur–Vijayawada road-network coverage."
                                )

                            stop_node = ox.distance.nearest_nodes(
                                graph, X=longitude, Y=latitude
                            )
                            stop_details.append(
                                {
                                    "name": stop_name,
                                    "latitude": latitude,
                                    "longitude": longitude,
                                    "node": stop_node,
                                }
                            )

                        weighted_graph = load_alternative_graph()
                        multi_route = find_multi_stop_route(
                            weighted_graph,
                            result["source_node"],
                            [item["node"] for item in stop_details],
                            result["destination_node"],
                            optimize_order=optimize_order,
                            max_stops=5,
                        )

                    name_by_node = {
                        item["node"]: item["name"] for item in stop_details
                    }
                    multi_route["ordered_stop_names"] = [
                        name_by_node[node]
                        for node in multi_route["ordered_stops"]
                    ]
                    multi_route["stop_details"] = stop_details
                    multi_route["source_name"] = source_name
                    multi_route["destination_name"] = destination_name
                    multi_route["source_latitude"] = result["source_lat"]
                    multi_route["source_longitude"] = result["source_lon"]
                    multi_route["destination_latitude"] = result["destination_lat"]
                    multi_route["destination_longitude"] = result["destination_lon"]

                    st.session_state["multi_stop_result"] = multi_route
                    st.session_state["multi_stop_result_key"] = request_key

                except Exception as exc:
                    st.session_state["multi_stop_result"] = None
                    st.session_state["multi_stop_result_key"] = None
                    st.error(f"Could not calculate the multi-stop route: {exc}")

        multi_result = st.session_state.get("multi_stop_result")
        stored_request_key = st.session_state.get("multi_stop_result_key")

        if not selected_stops:
            st.info("Select at least one intermediate stop to begin.")
        elif multi_result is None or stored_request_key != request_key:
            st.info(
                "Choose your stops and planning order, then click "
                "**Calculate Multi-Stop Route**."
            )
        else:
            st.success(
                "Multi-stop journey calculated successfully. "
                + (
                    f"Evaluated {multi_result['orders_evaluated']} possible stop orders."
                    if multi_result["optimized"]
                    else "Your selected stop order was preserved."
                )
            )

            metric1, metric2, metric3 = st.columns(3)
            metric1.metric(
                "Total Journey Distance",
                f"{multi_result['distance'] / 1000:.3f} km",
            )
            metric2.metric(
                "Intermediate Stops",
                str(len(multi_result["ordered_stops"])),
            )
            metric3.metric(
                "Journey Legs",
                str(len(multi_result["leg_distances"])),
            )

            ordered_names = [
                multi_result["source_name"],
                *multi_result["ordered_stop_names"],
                multi_result["destination_name"],
            ]
            st.markdown("#### Recommended Visiting Order")
            st.markdown(" → ".join(ordered_names))

            leg_rows = []
            for index, distance in enumerate(multi_result["leg_distances"]):
                leg_rows.append(
                    {
                        "Leg": index + 1,
                        "From": ordered_names[index],
                        "To": ordered_names[index + 1],
                        "Distance (km)": round(distance / 1000, 3),
                        "Path Nodes": len(multi_result["leg_paths"][index]),
                    }
                )

            st.markdown("#### Journey Leg Details")
            st.dataframe(
                leg_rows,
                use_container_width=True,
                hide_index=True,
            )

            # Render the combined route and number each intermediate stop.
            graph = load_graph()
            route_coordinates = [
                (graph.nodes[node]["y"], graph.nodes[node]["x"])
                for node in multi_result["path"]
            ]
            source_point = [
                multi_result["source_latitude"],
                multi_result["source_longitude"],
            ]
            destination_point = [
                multi_result["destination_latitude"],
                multi_result["destination_longitude"],
            ]

            multi_map = folium.Map(
                location=[
                    (source_point[0] + destination_point[0]) / 2,
                    (source_point[1] + destination_point[1]) / 2,
                ],
                zoom_start=9,
                control_scale=True,
                prefer_canvas=True,
            )

            folium.PolyLine(
                locations=route_coordinates,
                color="#7C3AED",
                weight=6,
                opacity=0.9,
                tooltip=(
                    f"Multi-stop route — "
                    f"{multi_result['distance'] / 1000:.3f} km"
                ),
            ).add_to(multi_map)

            folium.Marker(
                location=source_point,
                popup=f"Start: {multi_result['source_name']}",
                tooltip="START",
                icon=folium.Icon(color="green", icon="play"),
            ).add_to(multi_map)

            all_map_coordinates = list(route_coordinates)
            all_map_coordinates.extend(
                [tuple(source_point), tuple(destination_point)]
            )

            stop_details_by_node = {
                item["node"]: item for item in multi_result["stop_details"]
            }
            for stop_index, stop_node in enumerate(
                multi_result["ordered_stops"], start=1
            ):
                stop_detail = stop_details_by_node[stop_node]
                stop_point = [
                    stop_detail["latitude"],
                    stop_detail["longitude"],
                ]
                all_map_coordinates.append(tuple(stop_point))
                folium.Marker(
                    location=stop_point,
                    popup=f"Stop {stop_index}: {stop_detail['name']}",
                    tooltip=f"STOP {stop_index}: {stop_detail['name']}",
                    icon=folium.Icon(color="blue", icon="flag"),
                ).add_to(multi_map)

            folium.Marker(
                location=destination_point,
                popup=f"Destination: {multi_result['destination_name']}",
                tooltip="DESTINATION",
                icon=folium.Icon(color="red", icon="flag"),
            ).add_to(multi_map)

            south = min(point[0] for point in all_map_coordinates)
            north = max(point[0] for point in all_map_coordinates)
            west = min(point[1] for point in all_map_coordinates)
            east = max(point[1] for point in all_map_coordinates)
            multi_map.fit_bounds([[south, west], [north, east]])

            st.markdown("#### Multi-Stop Route Map")
            st_folium(
                multi_map,
                width=1200,
                height=600,
                key=f"multi_stop_map_{multi_stop_key}_{abs(hash(request_key))}",
                returned_objects=[],
            )
            st.caption(
                "Green: starting location · Blue: numbered intermediate stops · "
                "Red: final destination · Purple: complete road route."
            )

    # =================================================
    # 16. PERFORMANCE ANALYSIS
    # =================================================

    with performance_tab:
        st.markdown("### Algorithm Performance Comparison")

        comparison = [
            {
                "Metric": "Route Distance (km)",
                "Dijkstra": round(result["dijkstra_distance"] / 1000, 3),
                "A*": round(result["astar_distance"] / 1000, 3),
            },
            {
                "Metric": "Execution Time (ms)",
                "Dijkstra": round(result["dijkstra_time"], 3),
                "A*": round(result["astar_time"], 3),
            },
            {
                "Metric": "Nodes Explored",
                "Dijkstra": result["dijkstra_explored"],
                "A*": result["astar_explored"],
            },
            {
                "Metric": "Path Nodes",
                "Dijkstra": len(result["dijkstra_path"]),
                "A*": len(result["astar_path"]),
            },
        ]

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.markdown("#### Execution Time")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(
                ["Dijkstra", "A*"],
                [result["dijkstra_time"], result["astar_time"]],
                color=["#2563EB", "#DC2626"],
            )
            ax.set_ylabel("Milliseconds")
            ax.set_title("Execution Time Comparison")
            ax.grid(axis="y", alpha=0.2)
            st.pyplot(fig)
            plt.close(fig)

        with chart_col2:
            st.markdown("#### Nodes Explored")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(
                ["Dijkstra", "A*"],
                [result["dijkstra_explored"], result["astar_explored"]],
                color=["#2563EB", "#DC2626"],
            )
            ax.set_ylabel("Number of Nodes")
            ax.set_title("Explored Nodes Comparison")
            ax.grid(axis="y", alpha=0.2)
            st.pyplot(fig)
            plt.close(fig)

        if node_reduction > 0:
            st.success(
                f"A* explored {node_reduction:.2f}% fewer nodes than Dijkstra."
            )
        elif node_reduction < 0:
            st.info(
                f"A* explored {abs(node_reduction):.2f}% more nodes "
                "than Dijkstra for this test."
            )
        else:
            st.info("Both algorithms explored the same number of nodes.")

    # =================================================
    # 15. ROUTE DETAILS
    # =================================================

    with details_tab:
        st.markdown("### Journey Details")
        st.write(f"**Source:** {result['source_display']}")
        st.write(f"**Destination:** {result['destination_display']}")
        st.divider()

        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            st.markdown("#### 🔵 Dijkstra")
            st.write(
                f"Distance: {result['dijkstra_distance'] / 1000:.3f} km"
            )
            st.write(f"Execution time: {result['dijkstra_time']:.3f} ms")
            st.write(f"Nodes explored: {result['dijkstra_explored']:,}")
            st.write(f"Path nodes: {len(result['dijkstra_path'])}")
            st.write(f"Recorded exploration nodes: {len(result['dijkstra_order']):,}")

        with detail_col2:
            st.markdown("#### 🔴 A* Search")
            st.write(
                f"Distance: {result['astar_distance'] / 1000:.3f} km"
            )
            st.write(f"Execution time: {result['astar_time']:.3f} ms")
            st.write(f"Nodes explored: {result['astar_explored']:,}")
            st.write(f"Path nodes: {len(result['astar_path'])}")
            st.write(f"Recorded exploration nodes: {len(result['astar_order']):,}")

        st.divider()
        st.markdown("**Algorithm Interpretation**")
        if same_distance:
            st.write(
                "Both algorithms found the same minimum route distance "
                "using the same road graph and distance weights."
            )
        else:
            st.write(
                "The route distances differ. Check graph directionality, "
                "edge weights and the heuristic implementation."
            )

    # =================================================
    # 16. CLEAR RESULTS
    # =================================================

    st.divider()
    if st.button("🗑️ Clear Route Results", use_container_width=True):
        st.session_state.route_result = None
        st.session_state.route_error = None
        st.rerun()

else:
    if not st.session_state.route_error:
        st.info(
            "Select a source and destination, then click "
            "'Find Shortest Route' to begin."
        )


# =====================================================
# 17. FOOTER
# =====================================================

st.markdown(
    """
    <div class="footer">
        Smart Route Planner | Dijkstra's Algorithm | A* Search |
        OpenStreetMap Contributors | OSMnx | Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)
