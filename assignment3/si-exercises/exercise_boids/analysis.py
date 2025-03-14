import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_velocity(df: pd.DataFrame):
    """
    Calculate the velocity of the boids in each time step
    """
    # Compute the differences in x and y coordinates
    df["dx"] = df.groupby("id")["x"].diff()
    df["dy"] = df.groupby("id")["y"].diff()
    # Fill the first NaN values with 0
    df["dx"] = df["dx"].fillna(0)
    df["dy"] = df["dy"].fillna(0)
    return df


def normalize_velocity(row: pd.DataFrame):
    mag = (row["dx"] ** 2 + row["dy"] ** 2) ** 0.5
    if mag == 0:
        return (0, 0)
    return (row["dx"] / mag, row["dy"] / mag)


def calculate_order_parameter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the order parameter over time for boids
    """
    df = calculate_velocity(df)

    # Apply normalization
    df[["vx", "vy"]] = df.apply(normalize_velocity, axis=1, result_type="expand")

    # Group the data by 'time' to aggregate the normalized velocities for each time step
    time_groups = df.groupby("time")

    order_parameter = []

    for time, group in time_groups:
        sum_vx = group["vx"].sum()
        sum_vy = group["vy"].sum()
        total_magnitude = np.hypot(sum_vx, sum_vy)  # Calculate the magnitude of the summed vectors
        num_boids = len(group)
        o = total_magnitude / num_boids
        order_parameter.append({"time": time, "O": o})

    # Convert to DataFrame
    order_df = pd.DataFrame(order_parameter)

    # optionally save the order values to csv
    # order_df.to_csv('order_parameter.csv', index=False)

    return order_df


def calculate_nearest_neighbour_distance(df: pd.DataFrame) -> pd.DataFrame:
    # get unique times
    unique_times = df["time"].unique()

    nearest_neighbour_distances = []

    for time in unique_times:
        # Filter data for the current time
        current_df = df[df["time"] == time]

        # If there is only one Boid, skip (no neighbors)
        if len(current_df) < 2:
            continue

        # Calculate pairwise distances
        for _, row in current_df.iterrows():
            # Calculate distances to all other Boids
            other_boids = current_df[current_df["id"] != row["id"]]
            distances = np.sqrt((other_boids["x"] - row["x"]) ** 2 + (other_boids["y"] - row["y"]) ** 2)
            nearest_distance = distances.min()
            nearest_neighbour_distances.append({"time": time, "id": row["id"], "distance": nearest_distance})

    nnd_df = pd.DataFrame(nearest_neighbour_distances)

    # optionally save the nearest neighbours distances to csv
    # nnd_df.to_csv('nearest_neighbour_distances.csv', index=False)

    return nnd_df


# print("majin duu")
# load the data
# df = pd.read_csv("test_exp.csv")

# order_df = calculate_order_parameter(df)

# # visualisations
# plt.plot(order_df['time'], order_df['O'])
# plt.xlabel('Time')
# plt.ylabel('Order Parameter (O)')
# plt.title('Order Parameter Over Time')
# plt.show()

# nnd_df = calculate_nearest_neighbour_distance(df)
