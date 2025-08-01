import numpy as np
import matplotlib.pyplot as plt

class Segment:
    """A class to represent a line segment in the time series."""
    def __init__(self, start_index, end_index, data):
        self.start_index = start_index
        self.end_index = end_index
        self.points = data[start_index : end_index + 1]
        self.line = self._calculate_best_fit_line()

    def _calculate_best_fit_line(self):
        """Calculates the best-fit line (linear regression) for the segment's points."""
        x = np.arange(self.start_index, self.end_index + 1)
        y = self.points
        if len(x) < 2:
            return np.array([0, y[0]]) # Not enough points for a line, return a flat line
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return np.array([m, c])

    def get_error(self):
        """Calculates the sum of squared errors for the segment."""
        x = np.arange(self.start_index, self.end_index + 1)
        y = self.points
        m, c = self.line
        y_fit = m * x + c
        return np.sum((y - y_fit) ** 2)

    def __repr__(self):
        return f"Segment({self.start_index}, {self.end_index})"

def calculate_merge_cost(segment1, segment2, original_data):
    """Calculates the error of a potential merged segment."""
    start_index = segment1.start_index
    end_index = segment2.end_index
    
    # Create a temporary merged segment to calculate its error
    merged_segment = Segment(start_index, end_index, original_data)
    return merged_segment.get_error()

def bottom_up_segmentation(data, max_error):
    """
    Performs bottom-up time series segmentation.

    Args:
        data (np.array): The input time series data.
        max_error (float): The maximum allowed sum of squared errors for a segment merge.

    Returns:
        list: A list of Segment objects representing the final segmentation.
    """
    # --- Step 1: Create the finest possible initial approximation ---
    segments = [Segment(i, i + 1, data) for i in range(len(data) - 1)]

    if not segments:
        return []

    # --- Step 2: Calculate the cost of merging each adjacent pair ---
    merge_costs = [calculate_merge_cost(segments[i], segments[i+1], data) for i in range(len(segments) - 1)]

    # --- Step 3: Iteratively merge the lowest-cost pair ---
    while merge_costs and min(merge_costs) < max_error:
        # Find the index of the cheapest pair to merge
        min_cost_index = np.argmin(merge_costs)

        # Merge the segments at the found index
        segment1 = segments[min_cost_index]
        segment2 = segments[min_cost_index + 1]
        
        new_segment = Segment(segment1.start_index, segment2.end_index, data)
        
        # Replace the first segment with the new merged segment
        segments[min_cost_index] = new_segment
        # Remove the second segment
        del segments[min_cost_index + 1]
        
        # Remove the cost associated with the just-merged pair
        del merge_costs[min_cost_index]

        # --- Update costs for the neighbors of the new merged segment ---
        
        # Update cost for the segment to the left (if it exists)
        if min_cost_index > 0:
            left_neighbor = segments[min_cost_index - 1]
            new_cost_left = calculate_merge_cost(left_neighbor, new_segment, data)
            merge_costs[min_cost_index - 1] = new_cost_left

        # Update cost for the segment to the right (if it exists)
        if min_cost_index < len(segments) - 1:
            right_neighbor = segments[min_cost_index + 1]
            new_cost_right = calculate_merge_cost(new_segment, right_neighbor, data)
            # The cost list is now one shorter, so the index is the same
            merge_costs[min_cost_index] = new_cost_right
            
    return segments

def plot_segmentation(data, segments):
    """Plots the original data and the segmented approximation."""
    plt.figure(figsize=(15, 6))
    plt.plot(data, label='Original Time Series', color='blue', alpha=0.7)
    
    for seg in segments:
        x = np.arange(seg.start_index, seg.end_index + 1)
        m, c = seg.line
        y_fit = m * x + c
        plt.plot(x, y_fit, 'r-', linewidth=3)

    plt.title('Bottom-Up Time Series Segmentation')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # --- Create a sample time series ---
    np.random.seed(42)
    time = np.arange(0, 100, 0.5)
    data = (np.sin(time * 0.1) * 10 + 
            np.cos(time * 0.5) * 5 + 
            np.random.randn(len(time)) * 2)
    data[80:120] += 20  # Add a step change
    data[150:] -= 15 # Add another step change

    # --- Perform segmentation ---
    # The max_error is a crucial parameter. You'll need to tune it
    # for your specific dataset to get the desired level of detail.
    max_approximation_error = 50.0 
    final_segments = bottom_up_segmentation(data, max_approximation_error)

    print(f"Original number of segments: {len(data) - 1}")
    print(f"Final number of segments: {len(final_segments)}")
    
    # --- Visualize the result ---
    plot_segmentation(data, final_segments)
