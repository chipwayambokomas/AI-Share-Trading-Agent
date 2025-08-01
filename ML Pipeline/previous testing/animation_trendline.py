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
    This function is a generator, yielding the state of segments at each merge.
    """
    # 1. Initial Fine Approximation
    # Start with segments connecting adjacent points.
    segments = [data[i:i+2] for i in range(0, len(data) - 1, 1)]
    yield segments

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
        
        yield segments
            
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
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(subset=['Close'], inplace=True)

    # Use the 'Close' price for segmentation
    price_data = df['Close'].values

    # --- Parameters for Segmentation ---
    MAX_SEGMENTATION_ERROR = 1000000 
    
    # --- Visualization Setup ---
    # Number of final merges to animate
    ANIMATE_LAST_N_MERGES = 300
    
    print(f"Starting bottom-up segmentation with max_error = {MAX_SEGMENTATION_ERROR}...")
    
    segment_generator = bottom_up_segmentation(price_data, MAX_SEGMENTATION_ERROR)
    
    # Get initial number of segments
    initial_segments = next(segment_generator)
    num_initial_segments = len(initial_segments)
    
    # --- Process merges without plotting for speed ---
    final_segments = []
    merges_to_process = 0
    
    # Calculate how many merges to process before starting animation
    temp_segments = list(initial_segments)
    temp_merge_costs = []
    for i in range(len(temp_segments) - 1):
        merged_segment = np.concatenate((temp_segments[i], temp_segments[i+1][1:]))
        temp_merge_costs.append(calculate_error(merged_segment))

    num_merges = 0
    while min(temp_merge_costs) <= MAX_SEGMENTATION_ERROR:
        merge_idx = temp_merge_costs.index(min(temp_merge_costs))
        temp_segments[merge_idx] = np.concatenate((temp_segments[merge_idx], temp_segments[merge_idx+1][1:]))
        del temp_segments[merge_idx+1]
        del temp_merge_costs[merge_idx]
        if merge_idx < len(temp_segments) - 1:
            merged_segment = np.concatenate((temp_segments[merge_idx], temp_segments[merge_idx+1][1:]))
            temp_merge_costs[merge_idx] = calculate_error(merged_segment)
        if merge_idx > 0:
            merged_segment = np.concatenate((temp_segments[merge_idx-1], temp_segments[merge_idx][1:]))
            temp_merge_costs[merge_idx-1] = calculate_error(merged_segment)
        if not temp_merge_costs: break
        num_merges += 1

    merges_to_skip = max(0, num_merges - ANIMATE_LAST_N_MERGES)
    
    print(f"Processing {merges_to_skip} initial merges (this may take a moment)...")
    for i, segments in enumerate(segment_generator):
        if i >= merges_to_skip:
            final_segments = segments
            break # Start animation from this point
    
    # --- Animate the final merges ---
    print(f"Animating the final {ANIMATE_LAST_N_MERGES} merges...")
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.ion() # Turn on interactive mode
    
    # Use the rest of the generator for animation
    for segments in segment_generator:
        ax.cla()
        ax.plot(df.index, price_data, label='NYSE Composite Close Price', color='blue', alpha=0.5)
        
        current_pos = 0
        for segment in segments:
            segment_len = len(segment)
            x_coords = df.index[current_pos : current_pos + segment_len]
            p = np.poly1d(np.polyfit(np.arange(segment_len), segment, 1))
            ax.plot(x_coords, p(np.arange(segment_len)), color='red', linestyle='-', linewidth=2)
            current_pos += segment_len -1
        
        ax.set_title(f'NYSE Composite Index with Bottom-Up Piecewise Trendlines | Segments: {len(segments)}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Closing Price')
        ax.legend(['NYSE Close', 'Trendlines'])
        ax.grid(True)
        plt.pause(0.01)
        final_segments = segments

    plt.ioff()
    print(f"\nSegmentation complete. Found {len(final_segments)} segments.")
    
    # Save the final plot
    output_filename = 'nyse_trendlines.png'
    plt.savefig(output_filename)
    print(f"Final plot saved to {output_filename}")
    
    print("Displaying final plot...")
    plt.show()

if __name__ == '__main__':
    main()
