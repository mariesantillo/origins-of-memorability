import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os 
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

data = pd.read_csv('/home/mariesantillo/vca/vca_results/bathsong_y4m32.csv')

poc_column = 'POC'

# Constants
fps = 25  # frames per second
tr = 0.61  # TR in seconds
frames_per_tr = round(fps * tr)  # Frames per TR

# Calculate TR intervals using the frame index
data['TR'] = (data.index // frames_per_tr)  # Create a column to group by TR intervals

# Ensure that the first row's TR value is explicitly set to 0
data.loc[data.index == 0, 'TR'] = 0

# Aggregate data by TR, using mean for complexity measures and mean POC for frame representation
aggregated_data = data.groupby('TR').mean().reset_index()

# Set the first POC value to 0
aggregated_data.loc[0, poc_column] = 0

# After aggregating data by TR, ensure the first POC value is set to 0
aggregated_data[poc_column] = data.groupby('TR')[poc_column].mean().values

# Save the aggregated data to a new CSV file
aggregated_data.drop('TR', axis=1, inplace=True)  # Drop the 'TR' column before saving
aggregated_data.to_csv('/home/mariesantillo/vca/results_aggregated/bathsong_y4m32_aggregated2.csv', index=False)
