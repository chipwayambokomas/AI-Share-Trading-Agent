import csv
from io import StringIO

# Sample data
data = """StockID,Actual_Slope_Angle,Predicted_Slope_Angle,Actual_Duration,Predicted_Duration
EXX.JO,64.842705,-85.38131,110.00001,68.594734
EXX.JO,76.52418,-83.65921,80.0,68.978065
DSY.JO,-65.364555,-85.96407,100.00001,68.75976
MRP.JO,86.58918,-82.061966,50.000004,68.78634
APN.JO,88.086586,51.21603,55.000004,67.59228"""

def check_slope_angles(csv_data):
    """
    Check if slope angles are within bounds (-90 to 90 degrees)
    Print rows with out-of-bounds angles and count violations
    """
    # Counter for rows with out-of-bounds angles
    out_of_bounds_count = 0
    row_num = 0  # Initialize row counter
    
    # Create CSV reader from string data
    csv_reader = csv.DictReader(StringIO(csv_data))
    
    print("Checking slope angles for bounds (-90 to 90 degrees)...")
    print("=" * 60)
    
    # Process each row
    for row_num, row in enumerate(csv_reader, start=1):
        stock_id = row['StockID']
        actual_slope = float(row['Actual_Slope_Angle'])
        predicted_slope = float(row['Predicted_Slope_Angle'])
        actual_duration = row['Actual_Duration']
        predicted_duration = row['Predicted_Duration']
        
        # Check if either angle is out of bounds
        actual_out_of_bounds = actual_slope < -90 or actual_slope > 90
        predicted_out_of_bounds = predicted_slope < -90 or predicted_slope > 90
        
        if actual_out_of_bounds or predicted_out_of_bounds:
            out_of_bounds_count += 1
            print(f"Row {row_num} - OUT OF BOUNDS:")
            print(f"{stock_id},{actual_slope},{predicted_slope},{actual_duration},{predicted_duration}")
            
            # Specify which angles are out of bounds
            violations = []
            if actual_out_of_bounds:
                violations.append(f"Actual_Slope_Angle: {actual_slope}")
            if predicted_out_of_bounds:
                violations.append(f"Predicted_Slope_Angle: {predicted_slope}")
            
            print(f"  Violations: {', '.join(violations)}")
            print()
    
    print("=" * 60)
    print(f"Total rows processed: {row_num}")
    print(f"Rows with out-of-bounds angles: {out_of_bounds_count}")
    
    if out_of_bounds_count == 0:
        print("✓ All slope angles are within bounds!")
    else:
        print(f"⚠ {out_of_bounds_count} row(s) have slope angles outside the valid range.")

# Alternative function to read from a CSV file
def check_slope_angles_from_file(filename):
    """
    Read CSV file and check slope angles
    """
    out_of_bounds_count = 0
    row_num = 0  # Initialize row counter
    
    try:
        with open(filename, 'r', newline='') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            
            print(f"Checking slope angles in '{filename}' for bounds (-90 to 90 degrees)...")
            print("=" * 60)
            
            for row_num, row in enumerate(csv_reader, start=1):
                stock_id = row['StockID']
                actual_slope = float(row['Actual_Slope_Angle'])
                predicted_slope = float(row['Predicted_Slope_Angle'])
                actual_duration = row['Actual_Duration']
                predicted_duration = row['Predicted_Duration']
                
                # Check if either angle is out of bounds
                actual_out_of_bounds = actual_slope < -90 or actual_slope > 90
                predicted_out_of_bounds = predicted_slope < -90 or predicted_slope > 90
                
                if actual_out_of_bounds or predicted_out_of_bounds:
                    out_of_bounds_count += 1
                    print(f"Row {row_num} - OUT OF BOUNDS:")
                    print(f"{stock_id},{actual_slope},{predicted_slope},{actual_duration},{predicted_duration}")
                    
                    # Specify which angles are out of bounds
                    violations = []
                    if actual_out_of_bounds:
                        violations.append(f"Actual_Slope_Angle: {actual_slope}")
                    if predicted_out_of_bounds:
                        violations.append(f"Predicted_Slope_Angle: {predicted_slope}")
                    
                    print(f"  Violations: {', '.join(violations)}")
                    print()
            
            print("=" * 60)
            print(f"Total rows processed: {row_num}")
            print(f"Rows with out-of-bounds angles: {out_of_bounds_count}")
            
            if out_of_bounds_count == 0:
                print("✓ All slope angles are within bounds!")
            else:
                print(f"⚠ {out_of_bounds_count} row(s) have slope angles outside the valid range.")
                
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except Exception as e:
        print(f"Error reading file: {e}")

# Run the function with the sample data
if __name__ == "__main__":
    # Check the sample data
    check_slope_angles("evaluation_results_TREND_GraphWaveNet.csv")
    
    # Uncomment the line below to check a CSV file instead
    # check_slope_angles_from_file('evaluation_results_TREND_MLP.csv')