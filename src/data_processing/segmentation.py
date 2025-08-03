import numpy as np

def create_segment_least_squares(sequence, seq_range):
    """Creates a line segment using linear regression (least squares)."""
    start, end = seq_range
    sub_sequence = sequence[start:end+1]
    if len(sub_sequence) < 2:
        return (start, sub_sequence[0], end, 0.0)
    
    # this essentiallly creates a range for the sub_sequence, assigning indices to the values eg [105,222,333] -> [0,1,2]
    x = np.arange(len(sub_sequence))
    # Perform linear regression to find slope and intercept
    # in our case the intercept is the value at the start of the segment
    slope, intercept = np.polyfit(x, sub_sequence, 1)
    return (start, intercept, end, slope)

def compute_sum_of_squared_error(sequence, segment):
    """Computes the Sum of Squared Errors (SSE) for a given segment."""
    start, intercept, end, slope = segment
    start, end = int(start), int(end)
    
    if start >= end:
        return 0.0
        
    sub_sequence = sequence[start:end+1]
    x = np.arange(len(sub_sequence))
    predicted_sequence = slope * x + intercept
    return np.sum((sub_sequence - predicted_sequence) ** 2)

def bottomupsegment(sequence, create_segment_func, compute_error_func, max_error):
    """
    Computes a list of line segments that approximate the sequence using the bottom-up technique.
    """
    # Start with the finest possible segmentation which is each pair of adjacent points
    segments = [create_segment_func(sequence, seq_range) for seq_range in zip(range(len(sequence))[:-1], range(len(sequence))[1:])]
    
    if not segments:
        return []
    
    # We create a new list of merged segments from the segments we have
    mergesegments = [create_segment_func(sequence, (seg1[0], seg2[2])) for seg1, seg2 in zip(segments[:-1], segments[1:])]
    # Compute the initial costs for merging segments
    mergecosts = [compute_error_func(sequence, segment) for segment in mergesegments]
    
    # we will continue merging segments until the minimum merge cost is greater than the max_error -> this means that either all segments are within the max_error threshold
    # or the segments are too large to merge without exceeding the max_error threshold
    # or there are no segments left to merge
    while mergecosts and min(mergecosts) < max_error:
        # Find the index of the segment with the minimum merge cost
        idx = mergecosts.index(min(mergecosts))
        
        # we update our original segments list by merging the segments at idx and idx+1
        segments[idx] = mergesegments[idx]
        del segments[idx+1]
        
        # Update cost for the segment to the left
        if idx > 0:
            mergesegments[idx-1] = create_segment_func(sequence, (segments[idx-1][0], segments[idx][2]))
            mergecosts[idx-1] = compute_error_func(sequence, mergesegments[idx-1])
        
        # Update cost for the segment at the merge point (now representing the one to the right)
        if idx < len(segments) - 1:
            mergesegments[idx] = create_segment_func(sequence, (segments[idx][0], segments[idx+1][2]))
            mergecosts[idx] = compute_error_func(sequence, mergesegments[idx])
        
        # Remove the now-merged cost entry
        del mergecosts[idx]
        if idx < len(mergesegments): # The mergesegments list is now one shorter
             del mergesegments[idx]
    
    return segments