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


@st.cache_data(show_spinner=False)
def geocode_location(location):
    """Geocode a place name and cache the returned coordinates."""
    return ox.geocode(location)


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

    # Four application sections.
    route_tab, search_tab, performance_tab, details_tab = st.tabs([
    "🗺️ Route Map",
    "🔍 Search Visualization",
    "📈 Performance Analysis",
    "📋 Route Details"
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
    # 14. PERFORMANCE ANALYSIS
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
