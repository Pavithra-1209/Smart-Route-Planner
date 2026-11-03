
import osmnx as ox
from pathlib import Path
import time

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Location to save the downloaded graph
GRAPH_FILE = PROJECT_DIR / "road_network" / "guntur_vijayawada.graphml"


def download_road_network():

    # Reuse the graph if it has already been downloaded
    if GRAPH_FILE.exists():
        print("Road network already exists. Loading it...")
        G = ox.io.load_graphml(GRAPH_FILE)

    else:
        print("Getting geographical coordinates...")

        # Get approximate city center coordinates
        guntur_lat, guntur_lon = ox.geocode(
            "Guntur, Andhra Pradesh, India"
        )

        vijayawada_lat, vijayawada_lon = ox.geocode(
            "Vijayawada, Andhra Pradesh, India"
        )

        # Create a bounding box around both cities
        margin = 0.04

        bbox = (
            min(guntur_lon, vijayawada_lon) - margin,
            min(guntur_lat, vijayawada_lat) - margin,
            max(guntur_lon, vijayawada_lon) + margin,
            max(guntur_lat, vijayawada_lat) + margin
        )

        print("Downloading road network from OpenStreetMap...")
        print("This may take some time.")

        # Download driving road network
        # /////
            # Download with automatic retry if Overpass is busy
        max_attempts = 4

        for attempt in range(1, max_attempts + 1):
            try:
                print(f"Download attempt {attempt}/{max_attempts}")

                G = ox.graph.graph_from_bbox(
                    bbox=bbox,
                    network_type="drive",
                    simplify=True
                )

                print("Road network downloaded successfully!")
                break

            except Exception as e:

                error = str(e).lower()

                temporary_error = any(
                    word in error
                    for word in ["502", "503", "504", "capacity", "gateway"]
                )

                if not temporary_error or attempt == max_attempts:
                    raise

                wait_time = 30 * (2 ** (attempt - 1))

                print(f"Overpass server is busy.")
                print(f"Retrying after {wait_time} seconds...")

                time.sleep(wait_time)


        #//////

        # Save graph locally
        ox.io.save_graphml(G, filepath=GRAPH_FILE)

        print("Road network downloaded and saved!")

    # Display graph information
    print("\n----- ROAD NETWORK DETAILS -----")

    print("Total nodes:", len(G.nodes))
    print("Total edges:", len(G.edges))

    # Display one sample node
    node_id, node_data = next(iter(G.nodes(data=True)))

    print("\nSample node:")
    print("Node ID:", node_id)
    print("Latitude:", node_data["y"])
    print("Longitude:", node_data["x"])

    # Display one sample edge
    u, v, edge_data = next(iter(G.edges(data=True)))

    print("\nSample edge:")
    print("From:", u)
    print("To:", v)
    print("Distance:", edge_data["length"], "meters")

    return G


if __name__ == "__main__":
    download_road_network()
