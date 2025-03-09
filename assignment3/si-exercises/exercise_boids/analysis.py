import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load the data
df = pd.read_csv("test_exp.csv")

print(df.head())
print(df.shape)

grouped = df.groupby('id')
print(grouped.head())


def compute_velocity(group):
    # Calculate the difference between consecutive rows to get dx and dy
    group['dx'] = group['x'].diff()
    group['dy'] = group['y'].diff()
    # Handle NaN in the first row
    group.loc[len(group)-1, 'dx'] = 0
    group.loc[len(group)-1, 'dy'] = 0  
    return group

# Apply the velocity computation to each group
df = grouped.apply(compute_velocity).reset_index(drop=True)


def normalize_velocity(row):
    mag = np.hypot(row['dx'], row['dy'])
    if mag == 0:
        return (0, 0)
    return (row['dx']/mag, row['dy']/mag)

# Apply normalization
df[['vx', 'vy']] = df.apply(normalize_velocity, axis=1)


# Group the data by time to aggregate velocities
time_groups = df.groupby('time')
sum_vectors = []

for time, group in time_groups:
    sum_vx = group['vx'].sum()
    sum_vy = group['vy'].sum()
    sum_vectors.append((time, sum_vx, sum_vy))

# Convert to DataFrame
sum_df = pd.DataFrame(sum_vectors, columns=['time', 'sum_vx', 'sum_vy'])

# Count the number of boids at each time
n_boids_per_time = time_groups.apply(lambda x: len(x)).reset_index()
n_boids_per_time.columns = ['time', 'n_boids']

# Merge with sum_df
sum_df = sum_df.merge(n_boids_per_time, on='time')

# Compute order parameter
sum_df['order_parameter'] = sum_df['magnitude'] / sum_df['n_boids']

# plot the results
plt.plot(sum_df['time'], sum_df['order_parameter'])
plt.xlabel('Time')
plt.ylabel('Order Parameter')
plt.title('Order Parameter Over Time')
plt.show()