import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time

def load_data_from_excel(file_path):
    """
    Loads time series data from the first column of an Excel or CSV file.

    Args:
        file_path (str): The path to the .xlsx or .csv file.

    Returns:
        np.ndarray: A NumPy array containing the time series data, or None if an error occurs.
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, header=None)
        else:
            df = pd.read_excel(file_path, header=None)
        
        # Extract the first column and convert to a NumPy array of floats
        time_series = df.iloc[:, 0].dropna().to_numpy(dtype=float)
        
        if time_series.size < 2:
            print("Error: Not enough data points found in the first column.")
            return None
            
        print(f"Successfully loaded {time_series.size} data points.")
        return time_series
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def calculate_segment_error(points):
    """
    Calculates the sum of squared errors for a segment against its linear regression line.

    Args:
        points (np.ndarray): A 2D array of points, where each row is [x, y].

    Returns:
        float: The sum of squared errors.
    """
    if len(points) < 2:
        return 0.0
    
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    
    # Calculate linear regression parameters (slope and intercept)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x * x)
    
    denominator = n * sum_x2 - sum_x**2
    if np.isclose(denominator, 0):
        return np.sum((x - np.mean(x))**2)

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    
    predicted_y = slope * x + intercept
    error = np.sum((y - predicted_y)**2)
    
    return error

def visualize_step(original_data, segments, step_number, total_steps, output_dir):
    """
    Visualizes a single step of the segmentation process and saves it to a file.

    Args:
        original_data (np.ndarray): The full original time series data.
        segments (list): A list of tuples, where each tuple is (start_index, end_index).
        step_number (int): The current step number for the plot title.
        total_steps (int): The total number of merge operations.
        output_dir (str): The directory to save the plot image.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(original_data, color='gray', alpha=0.6, label='Original Data')
    
    for start, end in segments:
        plt.plot([start, end], [original_data[start], original_data[end]], color='red', linewidth=2)

    plt.title(f"Step {step_number} of {total_steps}: {len(segments)} Segments Remaining")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    file_path = os.path.join(output_dir, f"step_{step_number:04d}.png")
    plt.savefig(file_path)
    plt.close()

def bottom_up_segmentation(data, target_segments, visualize_last_n=20):
    """
    Performs bottom-up piecewise linear segmentation on time series data.

    Args:
        data (np.ndarray): The input time series data.
        target_segments (int): The desired final number of segments.
        visualize_last_n (int): The number of final steps to visualize.

    Returns:
        list: A list of (start_index, end_index) tuples representing the final segments.
    """
    segments = [(i, i + 1) for i in range(len(data) - 1)]
    
    merge_costs = []
    for i in range(len(segments) - 1):
        start_idx = segments[i][0]
        end_idx = segments[i+1][1]
        points_to_merge = np.array([[j, data[j]] for j in range(start_idx, end_idx + 1)])
        merge_costs.append(calculate_segment_error(points_to_merge))

    output_dir = "segmentation_steps"
    if visualize_last_n > 0 and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory '{output_dir}' for step visualizations.")

    total_merges = len(segments) - target_segments
    
    while len(segments) > target_segments:
        if not merge_costs:
            break
        min_cost_idx = np.argmin(merge_costs)
        
        s1_start, _ = segments[min_cost_idx]
        _, s2_end = segments[min_cost_idx + 1]
        new_segment = (s1_start, s2_end)
        
        segments[min_cost_idx] = new_segment
        del segments[min_cost_idx + 1]
        del merge_costs[min_cost_idx]
        
        if min_cost_idx > 0:
            left_neighbor_start = segments[min_cost_idx - 1][0]
            new_segment_end = segments[min_cost_idx][1]
            points_to_merge = np.array([[j, data[j]] for j in range(left_neighbor_start, new_segment_end + 1)])
            merge_costs[min_cost_idx - 1] = calculate_segment_error(points_to_merge)
            
        if min_cost_idx < len(merge_costs):
            new_segment_start = segments[min_cost_idx][0]
            right_neighbor_end = segments[min_cost_idx + 1][1]
            points_to_merge = np.array([[j, data[j]] for j in range(new_segment_start, right_neighbor_end + 1)])
            merge_costs[min_cost_idx] = calculate_segment_error(points_to_merge)

        current_step_num = total_merges - (len(segments) - target_segments)
        if visualize_last_n > 0 and (len(segments) - target_segments) < visualize_last_n:
            visualize_step(data, segments, current_step_num, total_merges, output_dir)
            print(f"Visualizing Step {current_step_num}/{total_merges}... Saved to '{output_dir}'")
            
    return segments

def calculate_segment_stats(segments, data):
    """
    Calculates statistics (duration, slope, average) for each segment.

    Args:
        segments (list): List of (start, end) index tuples for each segment.
        data (np.ndarray): The original time series data.

    Returns:
        list: A list of dictionaries, each containing stats for a segment.
    """
    stats_list = []
    for start, end in segments:
        if start >= end: continue
        
        # Duration: Number of data points covered
        duration = end - start + 1
        
        # Average value of the segment
        average_value = np.mean(data[start : end + 1])
        
        # Slope calculation
        delta_y = data[end] - data[start]
        delta_x = end - start
        slope = delta_y / delta_x
        
        # Convert slope to angle in degrees
        slope_angle = np.degrees(np.arctan(slope))
        
        stats_list.append({
            "start_index": start,
            "end_index": end,
            "duration": duration,
            "slope_angle_degrees": slope_angle,
            "average_value": average_value
        })
    return stats_list

def plot_final_result(data, segments, stats):
    """
    Plots the final segmentation result and annotates with stats.
    """
    plt.figure(figsize=(15, 7))
    plt.plot(data, color='c', label='Original Time Series', alpha=0.8)
    
    for i, segment_info in enumerate(stats):
        start = segment_info['start_index']
        end = segment_info['end_index']
        
        # Plot the segment line
        plt.plot([start, end], [data[start], data[end]], color='m', marker='o', linewidth=2.5, markersize=5)
        
        # Add text annotation for slope and duration
        mid_x = (start + end) / 2
        mid_y = (data[start] + data[end]) / 2
        
        # Alternate text position for readability
        vertical_offset = 5 if i % 2 == 0 else -10
        
        annotation_text = f"S: {segment_info['slope_angle_degrees']:.1f}°\nD: {segment_info['duration']}"
        plt.text(mid_x, mid_y + vertical_offset, annotation_text, 
                 ha='center', va='bottom', fontsize=9, 
                 bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
        
    plt.title('Final Bottom-Up Segmentation Result with Stats', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

if __name__ == "__main__":
    # --- User Inputs ---
    FILE_PATH = "sample_data.xlsx"
    TARGET_SEGMENTS = 15
    VISUALIZE_LAST_N_STEPS = 10

    if not os.path.exists(FILE_PATH):
        print(f"'{FILE_PATH}' not found. Creating a sample file for demonstration.")
        x = np.linspace(0, 50, 500)
        y = np.sin(x) * 10 + np.random.randn(500) * 1.5 + np.linspace(0, 5, 500)
        pd.DataFrame(y).to_excel(FILE_PATH, index=False, header=False)
        print("Sample file created.")

    time_series_data = load_data_from_excel(FILE_PATH)
    
    if time_series_data is not None:
        if TARGET_SEGMENTS >= len(time_series_data) / 2:
            print(f"Error: Target segments ({TARGET_SEGMENTS}) must be less than half the number of data points ({len(time_series_data)/2}).")
        else:
            print("\nStarting Bottom-Up Segmentation...")
            start_time = time.time()
            
            final_segments = bottom_up_segmentation(
                time_series_data, 
                target_segments=TARGET_SEGMENTS,
                visualize_last_n=VISUALIZE_LAST_N_STEPS
            )
            
            end_time = time.time()
            print(f"\nSegmentation finished in {end_time - start_time:.2f} seconds.")

            # --- Calculate and Display Segment Statistics ---
            segment_stats = calculate_segment_stats(final_segments, time_series_data)
            print("\n--- Final Segment Statistics ---")
            print(f"{'Segment':<10}{'Start':<10}{'End':<10}{'Duration':<12}{'Avg Value':<15}{'Slope (deg)':<15}")
            print("-" * 75)
            for i, stats in enumerate(segment_stats):
                print(f"{i+1:<10}"
                      f"{stats['start_index']:<10}"
                      f"{stats['end_index']:<10}"
                      f"{stats['duration']:<12}"
                      f"{stats['average_value']:<15.2f}"
                      f"{stats['slope_angle_degrees']:<15.2f}")
            
            if VISUALIZE_LAST_N_STEPS > 0:
                print(f"\nStep-by-step visualizations saved in '{os.path.abspath('segmentation_steps')}'")

            print("\nDisplaying the final annotated result...")
            plot_final_result(time_series_data, final_segments, segment_stats)
