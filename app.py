
import streamlit as st
import osmnx as ox
import folium
import matplotlib.pyplot as plt
import time
import math

from pathlib import Path
from streamlit_folium import st_folium

from algorithms.dijkstra import dijkstra
from algorithms.astar import astar


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Smart Route Planner",
    page_icon="🗺️",
    layout="wide"
)


# =====================================================
# 2. CUSTOM CSS
# =====================================================

st.markdown("""
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

.hero h1 {
    color: white;
    font-size: 34px;
    font-weight: 750;
    margin-bottom: 5px;
}

.hero p {
    color: #D8E5F7;
    font-size: 15px;
    margin-bottom: 0;
}

.hero-tag {
    color: #93C5FD;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
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
""", unsafe_allow_html=True)


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
    unsafe_allow_html=True
)


# =====================================================
# 4. AVAILABLE LOCATIONS
# =====================================================

LOCATION_OPTIONS = {

    "Guntur Railway Station":
        "Guntur Railway Station, Guntur, Andhra Pradesh, India",

    "Guntur NTR Bus Station":
        "NTR Bus Station, Guntur, Andhra Pradesh, India",

    "Brodipet, Guntur":
        "Brodipet, Guntur, Andhra Pradesh, India",

    "Arundelpet, Guntur":
        "Arundelpet, Guntur, Andhra Pradesh, India",

    "Lakshmipuram, Guntur":
        "Lakshmipuram, Guntur, Andhra Pradesh, India",

    "Vijayawada Railway Station":
        "Vijayawada Railway Station, Vijayawada, Andhra Pradesh, India",

    "Pandit Nehru Bus Station":
        "Pandit Nehru Bus Station, Vijayawada, Andhra Pradesh, India",

    "Benz Circle, Vijayawada":
        "Benz Circle, Vijayawada, Andhra Pradesh, India",

    "Mangalagiri":
        "Mangalagiri, Andhra Pradesh, India",

    "Kanaka Durga Temple":
        "Kanaka Durga Temple, Vijayawada, Andhra Pradesh, India"
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


# =====================================================
# 6. SWAP LOCATIONS
# =====================================================

def swap_locations():

    source = st.session_state.source_choice

    destination = st.session_state.destination_choice

    st.session_state.source_choice = destination

    st.session_state.destination_choice = source


# =====================================================
# 7. LOAD ROAD NETWORK
# =====================================================

GRAPH_FILE = (
    Path(__file__).resolve().parent
    / "road_network"
    / "guntur_vijayawada.graphml"
)


@st.cache_resource(show_spinner=False)
def load_graph():

    if not GRAPH_FILE.exists():

        raise FileNotFoundError(
            "Road network not found. "
            "Please run download_graph.py first."
        )

    return ox.load_graphml(filepath=GRAPH_FILE)


# =====================================================
# 8. GEOCODING
# =====================================================

@st.cache_data(show_spinner=False)
def geocode_location(location):

    return ox.geocode(location)


# =====================================================
# 9. ROUTE CALCULATION
# =====================================================

def calculate_routes(source_location, destination_location):

    G = load_graph()

    # ---------------------------------------------
    # Get geographical coordinates
    # ---------------------------------------------

    source_lat, source_lon = geocode_location(
        source_location
    )

    time.sleep(1)

    destination_lat, destination_lon = geocode_location(
        destination_location
    )

    # ---------------------------------------------
    # Check graph coverage
    # ---------------------------------------------

    graph_nodes = list(G.nodes(data=True))

    min_lat = min(data["y"] for _, data in graph_nodes)
    max_lat = max(data["y"] for _, data in graph_nodes)

    min_lon = min(data["x"] for _, data in graph_nodes)
    max_lon = max(data["x"] for _, data in graph_nodes)

    locations = [
        ("Source", source_lat, source_lon),
        ("Destination", destination_lat, destination_lon)
    ]

    for name, lat, lon in locations:

        if not (
            min_lat <= lat <= max_lat
            and min_lon <= lon <= max_lon
        ):

            raise ValueError(
                f"{name} is outside the downloaded road network. "
                "Please select a location within the supported area."
            )

    # ---------------------------------------------
    # Find nearest road nodes
    # ---------------------------------------------

    source_node = ox.distance.nearest_nodes(
        G,
        X=source_lon,
        Y=source_lat
    )

    destination_node = ox.distance.nearest_nodes(
        G,
        X=destination_lon,
        Y=destination_lat
    )

    # ---------------------------------------------
    # Run Dijkstra
    # ---------------------------------------------

    start_time = time.perf_counter()

    dijkstra_path, dijkstra_distance, dijkstra_explored = (
        dijkstra(G, source_node, destination_node)
    )

    dijkstra_time = (
        time.perf_counter() - start_time
    ) * 1000

    # ---------------------------------------------
    # Run A*
    # ---------------------------------------------

    start_time = time.perf_counter()

    astar_path, astar_distance, astar_explored = (
        astar(G, source_node, destination_node)
    )

    astar_time = (
        time.perf_counter() - start_time
    ) * 1000

    # ---------------------------------------------
    # Check routes
    # ---------------------------------------------

    if dijkstra_path is None or astar_path is None:

        raise ValueError(
            "No connected road route found between these locations."
        )

    # ---------------------------------------------
    # Return results
    # ---------------------------------------------

    return {

        "source_lat": source_lat,
        "source_lon": source_lon,

        "destination_lat": destination_lat,
        "destination_lon": destination_lon,

        "source_node": source_node,
        "destination_node": destination_node,

        "dijkstra_path": dijkstra_path,
        "dijkstra_distance": dijkstra_distance,
        "dijkstra_time": dijkstra_time,
        "dijkstra_explored": dijkstra_explored,

        "astar_path": astar_path,
        "astar_distance": astar_distance,
        "astar_time": astar_time,
        "astar_explored": astar_explored,

        "source_display": source_choice,
        "destination_display": destination_choice
    }


# =====================================================
# 10. SOURCE AND DESTINATION SELECTION
# =====================================================

st.markdown(
    '<div class="section-heading">📍 Plan Your Journey</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    source_col, swap_col, destination_col = st.columns(
        [5, 1, 5]
    )

    location_names = list(LOCATION_OPTIONS.keys())

    with source_col:

        st.markdown("**STARTING LOCATION**")

        st.selectbox(
            "Select Source",
            options=location_names,
            key="source_choice",
            label_visibility="collapsed"
        )

    with swap_col:

        st.markdown(
            "<div style='height: 27px'></div>",
            unsafe_allow_html=True
        )

        st.button(
            "⇄",
            help="Swap source and destination",
            on_click=swap_locations,
            use_container_width=True
        )

    with destination_col:

        st.markdown("**DESTINATION**")

        st.selectbox(
            "Select Destination",
            options=location_names,
            key="destination_choice",
            label_visibility="collapsed"
        )

    st.caption(
        "Select your starting location and destination "
        "from the available locations. You can type in a dropdown "
        "to search its options."
    )

    search_button = st.button(
        "🔍 Find Shortest Route",
        type="primary",
        use_container_width=True
    )


# =====================================================
# 11. HANDLE ROUTE SEARCH
# =====================================================

if search_button:

    source_choice = st.session_state.source_choice

    destination_choice = st.session_state.destination_choice

    # Clear previous results
    st.session_state.route_result = None
    st.session_state.route_error = None

    if source_choice == destination_choice:

        st.session_state.route_error = (
            "Source and destination cannot be the same."
        )

    else:

        try:

            source_location = LOCATION_OPTIONS[source_choice]

            destination_location = LOCATION_OPTIONS[
                destination_choice
            ]

            with st.spinner(
                "Calculating shortest routes using Dijkstra and A*..."
            ):

                result = calculate_routes(
                    source_location,
                    destination_location
                )

                # Store readable selected locations
                result["source_display"] = source_choice

                result["destination_display"] = destination_choice

                st.session_state.route_result = result

        except Exception as e:

            st.session_state.route_error = str(e)


# =====================================================
# 12. DISPLAY ERRORS
# =====================================================

if st.session_state.route_error:

    st.error(st.session_state.route_error)


# =====================================================
# 13. DISPLAY RESULTS
# =====================================================

result = st.session_state.route_result

# Check if user changed locations after calculation

if result and (
    result["source_display"] != st.session_state.source_choice
    or result["destination_display"] != st.session_state.destination_choice
):

    st.info(
        "Your selected locations have changed. "
        "Click Find Shortest Route to calculate the updated route."
    )

    result = None


if result:

    st.success("Routes calculated successfully!")

    # ---------------------------------------------
    # SELECTED LOCATIONS
    # ---------------------------------------------

    st.markdown(
        '<div class="section-heading">📍 Selected Route</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            f"**Source:** {result['source_display']}"
        )

    with col2:

        st.info(
            f"**Destination:** {result['destination_display']}"
        )

    # ---------------------------------------------
    # DISTANCE COMPARISON
    # ---------------------------------------------

    st.markdown(
        '<div class="section-heading">📊 Route Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.markdown("### 🔵 Dijkstra")

            st.metric(
                "Shortest Distance",
                f"{result['dijkstra_distance'] / 1000:.3f} km"
            )

            st.caption(
                f"{result['dijkstra_explored']:,} nodes explored"
            )

    with col2:

        with st.container(border=True):

            st.markdown("### 🔴 A* Search")

            st.metric(
                "Shortest Distance",
                f"{result['astar_distance'] / 1000:.3f} km"
            )

            st.caption(
                f"{result['astar_explored']:,} nodes explored"
            )

    # ---------------------------------------------
    # EXTRA METRICS
    # ---------------------------------------------

    distance_difference = abs(
        result["dijkstra_distance"]
        - result["astar_distance"]
    )

    reduction = (
        (
            result["dijkstra_explored"]
            - result["astar_explored"]
        )
        / result["dijkstra_explored"] * 100
        if result["dijkstra_explored"] > 0
        else 0
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Distance Difference",
        f"{distance_difference:.3f} m"
    )

    metric2.metric(
        "Dijkstra Time",
        f"{result['dijkstra_time']:.2f} ms"
    )

    metric3.metric(
        "A* Time",
        f"{result['astar_time']:.2f} ms"
    )

    metric4.metric(
        "A* Node Change",
        f"{reduction:+.1f}%"
    )

    # ---------------------------------------------
    # DISTANCE VERIFICATION
    # ---------------------------------------------

    same_distance = math.isclose(
        result["dijkstra_distance"],
        result["astar_distance"],
        rel_tol=1e-9,
        abs_tol=1e-6
    )

    same_path = (
        result["dijkstra_path"] == result["astar_path"]
    )

    if same_distance:

        st.success(
            "Both algorithms found the same shortest distance!"
        )

    else:

        st.warning(
            "The route distances are different. "
            "Please verify the algorithm implementations."
        )

    if same_path:

        st.info(
            "Both algorithms discovered exactly the same road path."
        )

    elif same_distance:

        st.info(
            "Both algorithms found different paths "
            "with the same shortest distance."
        )

    # =================================================
    # 14. RESULTS TABS
    # =================================================

    map_tab, performance_tab, details_tab = st.tabs(
        [
            "🗺️ Route Map",
            "📈 Performance Analysis",
            "📋 Route Details"
        ]
    )

    # =================================================
    # 15. INTERACTIVE MAP
    # =================================================

    with map_tab:

        st.markdown("### Interactive Route Visualization")

        st.markdown(
            """
            <span style="color:#2563EB;font-weight:700;">
                ━━━━━ Dijkstra
            </span>
            &nbsp;&nbsp;&nbsp;
            <span style="color:#DC2626;font-weight:700;">
                ┅ ┅ ┅ A* Search
            </span>
            """,
            unsafe_allow_html=True
        )

        G = load_graph()

        # Convert Dijkstra path into coordinates

        dijkstra_coords = [

            (G.nodes[node]["y"], G.nodes[node]["x"])

            for node in result["dijkstra_path"]

        ]

        # Convert A* path into coordinates

        astar_coords = [

            (G.nodes[node]["y"], G.nodes[node]["x"])

            for node in result["astar_path"]

        ]

        source_point = [
            result["source_lat"],
            result["source_lon"]
        ]

        destination_point = [
            result["destination_lat"],
            result["destination_lon"]
        ]

        # Create Folium map

        route_map = folium.Map(

            location=[
                (source_point[0] + destination_point[0]) / 2,
                (source_point[1] + destination_point[1]) / 2
            ],

            zoom_start=9,
            control_scale=True,
            prefer_canvas=True

        )

        # Dijkstra route - Blue

        dijkstra_layer = folium.FeatureGroup(
            name="🔵 Dijkstra Route",
            show=True
        )

        folium.PolyLine(

            locations=dijkstra_coords,
            color="#2563EB",
            weight=6,
            opacity=0.9,
            tooltip="Dijkstra Shortest Route"

        ).add_to(dijkstra_layer)

        dijkstra_layer.add_to(route_map)

        # A* route - Red dashed

        astar_layer = folium.FeatureGroup(
            name="🔴 A* Route",
            show=True
        )

        folium.PolyLine(

            locations=astar_coords,
            color="#DC2626",
            weight=5,
            opacity=0.9,
            dash_array="10, 10",
            tooltip="A* Shortest Route"

        ).add_to(astar_layer)

        astar_layer.add_to(route_map)

        # Source marker

        folium.Marker(

            location=source_point,

            popup=result["source_display"],

            tooltip="START LOCATION",

            icon=folium.Icon(
                color="green",
                icon="play"
            )

        ).add_to(route_map)

        # Destination marker

        folium.Marker(

            location=destination_point,

            popup=result["destination_display"],

            tooltip="DESTINATION",

            icon=folium.Icon(
                color="red",
                icon="flag"
            )

        ).add_to(route_map)

        # Fit map bounds

        all_coords = (

            dijkstra_coords
            + astar_coords
            + [source_point, destination_point]

        )

        south = min(p[0] for p in all_coords)
        north = max(p[0] for p in all_coords)

        west = min(p[1] for p in all_coords)
        east = max(p[1] for p in all_coords)

        route_map.fit_bounds([

            [south, west],
            [north, east]

        ])

        # Route visibility controls

        folium.LayerControl(
            collapsed=False
        ).add_to(route_map)

        # Display map

        st_folium(

            route_map,

            width=1200,
            height=600,

            key=f"route_map_{result['source_node']}_{result['destination_node']}",

            returned_objects=[]

        )

        st.caption(
            "Blue solid line: Dijkstra | "
            "Red dashed line: A* Search. "
            "Use the map layer controls to show or hide routes."
        )

    # =================================================
    # 16. PERFORMANCE ANALYSIS
    # =================================================

    with performance_tab:

        st.markdown("### Algorithm Performance Comparison")

        comparison = [

            {
                "Metric": "Route Distance (km)",
                "Dijkstra": round(
                    result["dijkstra_distance"] / 1000, 3
                ),
                "A*": round(
                    result["astar_distance"] / 1000, 3
                )
            },

            {
                "Metric": "Execution Time (ms)",
                "Dijkstra": round(result["dijkstra_time"], 3),
                "A*": round(result["astar_time"], 3)
            },

            {
                "Metric": "Nodes Explored",
                "Dijkstra": result["dijkstra_explored"],
                "A*": result["astar_explored"]
            },

            {
                "Metric": "Path Nodes",
                "Dijkstra": len(result["dijkstra_path"]),
                "A*": len(result["astar_path"])
            }

        ]

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True
        )

        chart1, chart2 = st.columns(2)

        # Execution time chart

        with chart1:

            st.markdown("#### Execution Time")

            fig, ax = plt.subplots(figsize=(5, 4))

            ax.bar(

                ["Dijkstra", "A*"],

                [
                    result["dijkstra_time"],
                    result["astar_time"]
                ],

                color=["#2563EB", "#DC2626"]

            )

            ax.set_ylabel("Milliseconds")
            ax.set_title("Execution Time Comparison")
            ax.grid(axis="y", alpha=0.2)

            st.pyplot(fig)
            plt.close(fig)

        # Explored nodes chart

        with chart2:

            st.markdown("#### Nodes Explored")

            fig, ax = plt.subplots(figsize=(5, 4))

            ax.bar(

                ["Dijkstra", "A*"],

                [
                    result["dijkstra_explored"],
                    result["astar_explored"]
                ],

                color=["#2563EB", "#DC2626"]

            )

            ax.set_ylabel("Number of Nodes")
            ax.set_title("Explored Nodes Comparison")
            ax.grid(axis="y", alpha=0.2)

            st.pyplot(fig)
            plt.close(fig)

        if reduction > 0:

            st.success(
                f"A* explored {reduction:.2f}% fewer nodes than Dijkstra."
            )

        elif reduction < 0:

            st.info(
                f"A* explored {abs(reduction):.2f}% more nodes "
                "than Dijkstra in this test."
            )

        else:

            st.info(
                "Both algorithms explored the same number of nodes."
            )

    # =================================================
    # 17. ROUTE DETAILS
    # =================================================

    with details_tab:

        st.markdown("### Journey Details")

        st.write(
            f"**Source:** {result['source_display']}"
        )

        st.write(
            f"**Destination:** {result['destination_display']}"
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.markdown("#### 🔵 Dijkstra")

            st.write(
                f"Distance: "
                f"{result['dijkstra_distance'] / 1000:.3f} km"
            )

            st.write(
                f"Execution Time: {result['dijkstra_time']:.3f} ms"
            )

            st.write(
                f"Nodes Explored: {result['dijkstra_explored']:,}"
            )

            st.write(
                f"Path Nodes: {len(result['dijkstra_path'])}"
            )

        with col2:

            st.markdown("#### 🔴 A* Search")

            st.write(
                f"Distance: "
                f"{result['astar_distance'] / 1000:.3f} km"
            )

            st.write(
                f"Execution Time: {result['astar_time']:.3f} ms"
            )

            st.write(
                f"Nodes Explored: {result['astar_explored']:,}"
            )

            st.write(
                f"Path Nodes: {len(result['astar_path'])}"
            )

        st.divider()

        st.markdown("**Algorithm Interpretation**")

        if same_distance:

            st.write(
                "Both algorithms successfully found the same "
                "minimum-distance route cost."
            )

        else:

            st.write(
                "The algorithms returned different distances. "
                "Further verification is required."
            )

    # =================================================
    # 18. CLEAR RESULTS
    # =================================================

    st.divider()

    if st.button(
        "🗑️ Clear Route Results",
        use_container_width=True
    ):

        st.session_state.route_result = None
        st.session_state.route_error = None

        st.rerun()


# =====================================================
# 19. INITIAL MESSAGE
# =====================================================

else:

    if not st.session_state.route_error:

        st.info(
            "Select a source and destination from the dropdowns "
            "and click Find Shortest Route to begin."
        )


# =====================================================
# 20. FOOTER
# =====================================================

st.markdown("""
<div class="footer">
    Smart Route Planner |
    Dijkstra's Algorithm |
    A* Search |
    OpenStreetMap contributors |
    OSMnx |
    Streamlit
</div>
""", unsafe_allow_html=True)
