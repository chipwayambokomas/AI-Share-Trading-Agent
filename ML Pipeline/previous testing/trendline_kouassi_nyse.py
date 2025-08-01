import pandas as pd
import numpy as np
import matplotlib
# FIX: Explicitly set the backend for matplotlib to TkAgg.
# This is often required to make interactive plotting work correctly
# in various script-running environments. This line must come
# before importing pyplot.
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import subprocess
import os

def download_data():
    """
    Executes the nyse_kouassi_data.py script to download the dataset.
    """
    script_name = 'nyse_kouassi_data.py'
    if not os.path.exists('nyse_composite_1965_2019.csv'):
        print(f"Running {script_name} to download NYSE data...")
        try:
            subprocess.run(['python', script_name], check=True)
            print("Data downloaded successfully.")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Error executing {script_name}: {e}")
            print("Please ensure the script is in the same directory and you have yfinance installed.")
            return False
    return True

def calculate_error(points):
    """
    Calculates the sum of squared errors for a set of points from a
    line connecting the first and last points.
    """
    if len(points) < 2:
        return 0
    
    # Get the start and end points
    x_coords = np.arange(len(points))
    y_coords = points
    
    # Equation of the line connecting the first and last points
    # y = mx + c
    m = (y_coords[-1] - y_coords[0]) / (x_coords[-1] - x_coords[0]) if len(points) > 1 else 0
    c = y_coords[0]

    # Calculate the y-values on the line for each x-coordinate
    line_y = m * x_coords + c
    
    # Calculate the sum of squared errors
    error = np.sum((y_coords - line_y) ** 2)
    return error

def bottom_up_segmentation(data, max_error):
    """
    Performs bottom-up piecewise linear segmentation on a time series.

    Args:
        data (np.array): The time series data.
        max_error (float): The maximum allowed error for merging segments.

    Returns:
        list: A list of arrays, where each array represents a segment.
    """
    # 1. Initial Fine Approximation
    # Start with n/2 segments, each connecting two adjacent points.
    segments = [data[i:i+2] for i in range(0, len(data) - 1, 1)]
    
    # 2. Calculate initial merging costs
    merge_costs = []
    for i in range(len(segments) - 1):
        merged_segment = np.concatenate((segments[i], segments[i+1][1:]))
        merge_costs.append(calculate_error(merged_segment))

    # 3. Iterative Merging
    while True:
        if not merge_costs:
            break
            
        min_cost = min(merge_costs)
        
        if min_cost > max_error:
            break
            
        # Find the index of the cheapest merge
        merge_idx = merge_costs.index(min_cost)
        
        # Merge the segments
        segments[merge_idx] = np.concatenate((segments[merge_idx], segments[merge_idx+1][1:]))
        
        # Remove the merged segment and its cost
        del segments[merge_idx+1]
        del merge_costs[merge_idx]
        
        # Recalculate cost for the new segment with its right neighbor
        if merge_idx < len(segments) - 1:
            merged_segment = np.concatenate((segments[merge_idx], segments[merge_idx+1][1:]))
            merge_costs[merge_idx] = calculate_error(merged_segment)
            
        # Recalculate cost for the new segment with its left neighbor
        if merge_idx > 0:
            merged_segment = np.concatenate((segments[merge_idx-1], segments[merge_idx][1:]))
            merge_costs[merge_idx-1] = calculate_error(merged_segment)
            
    return segments

def main():
    """
    Main function to run the data download, segmentation, and visualization.
    """
    if not download_data():
        return

    # Load the dataset
    try:
        df = pd.read_csv('nyse_composite_1965_2019.csv', index_col=0, parse_dates=True)
        df.index.name = 'Date'
    except FileNotFoundError:
        print("Could not find the NYSE data file. Please check the download step.")
        return

    # Ensure the 'Close' column is numeric to prevent TypeErrors.
    # The 'coerce' argument will turn any non-numeric values into NaN (Not a Number).
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    
    # Remove any rows that now have NaN in the 'Close' column.
    df.dropna(subset=['Close'], inplace=True)

    # Use the 'Close' price for segmentation
    price_data = df['Close'].values

    # --- Parameters for Segmentation ---
    # This is a crucial parameter. A smaller value results in more segments.
    # A larger value results in fewer, more coarse segments.
    # You may need to experiment to find a value that suits your needs.
    MAX_SEGMENTATION_ERROR = 1000000 

    print(f"Starting bottom-up segmentation with max_error = {MAX_SEGMENTATION_ERROR}...")
    segmented_data = bottom_up_segmentation(price_data, MAX_SEGMENTATION_ERROR)
    print(f"Segmentation complete. Found {len(segmented_data)} segments.")

    # --- Visualization ---
    plt.figure(figsize=(15, 8))
    
    # Plot original data
    plt.plot(df.index, price_data, label='NYSE Composite Close Price', color='blue', alpha=0.5)

    # Plot trendlines
    current_pos = 0
    for segment in segmented_data:
        segment_len = len(segment)
        x_coords = df.index[current_pos : current_pos + segment_len]
        
        # Create the line for the segment
        p = np.poly1d(np.polyfit(np.arange(segment_len), segment, 1))
        
        plt.plot(x_coords, p(np.arange(segment_len)), color='red', linestyle='-', linewidth=2)
        current_pos += segment_len -1

    plt.title('NYSE Composite Index with Bottom-Up Piecewise Trendlines')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend(['NYSE Close', 'Trendlines'])
    plt.grid(True)
    
    # FIX: Save the plot to a file as a robust way to view the output,
    # in case the interactive window doesn't display properly.
    output_filename = 'nyse_trendlines.png'
    plt.savefig(output_filename)
    print(f"\nPlot saved to {output_filename}")
    
    print("Displaying plot...")
    plt.show()

if __name__ == '__main__':
    main()
