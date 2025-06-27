import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
import json
from pathlib import Path

def find_card_corners(mask_path: str) -> Optional[List[Tuple[float, float]]]:
    """
    Extract 4 corner points from card mask using HoughLinesP for line detection.
    
    Args:
        mask_path: Path to the mask image
        
    Returns:
        List of 4 corner points [(x, y), ...] in clockwise order starting from top-left,
        or None if corners cannot be found
    """
    try:
        # Read mask image
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not read mask {mask_path}")
            return None
            
        # Threshold to ensure binary mask
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Apply edge detection
        edges = cv2.Canny(binary_mask, 50, 150, apertureSize=3)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, 
                               minLineLength=50, maxLineGap=10)
        
        if lines is None or len(lines) < 4:
            # Fallback to contour-based method if HoughLines fails
            return find_card_corners_contour_fallback(binary_mask, mask_path)
        
        # Filter and group lines by orientation
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angle = abs(angle)
            
            # Classify as horizontal or vertical based on angle
            if angle < 30 or angle > 150:  # Horizontal lines (±30°)
                horizontal_lines.append(line[0])
            elif 60 < angle < 120:  # Vertical lines (±30° from 90°)
                vertical_lines.append(line[0])
        
        # Need at least 2 horizontal and 2 vertical lines
        if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
            return find_card_corners_contour_fallback(binary_mask, mask_path)
        
        # Find extreme lines (top, bottom, left, right)
        h_lines = np.array(horizontal_lines)
        v_lines = np.array(vertical_lines)
        
        # Find top and bottom horizontal lines
        top_line = h_lines[np.argmin(np.minimum(h_lines[:, 1], h_lines[:, 3]))]
        bottom_line = h_lines[np.argmax(np.maximum(h_lines[:, 1], h_lines[:, 3]))]
        
        # Find left and right vertical lines
        left_line = v_lines[np.argmin(np.minimum(v_lines[:, 0], v_lines[:, 2]))]
        right_line = v_lines[np.argmax(np.maximum(v_lines[:, 0], v_lines[:, 2]))]
        
        # Calculate intersections to get corners
        corners = []
        line_pairs = [
            (top_line, left_line),    # Top-left
            (top_line, right_line),   # Top-right
            (bottom_line, right_line), # Bottom-right
            (bottom_line, left_line)   # Bottom-left
        ]
        
        for line1, line2 in line_pairs:
            intersection = line_intersection(line1, line2)
            if intersection is not None:
                corners.append(intersection)
        
        if len(corners) != 4:
            return find_card_corners_contour_fallback(binary_mask, mask_path)
        
        # Sort corners in clockwise order starting from top-left
        corners = np.array(corners)
        corners = sort_corners_clockwise(corners)
        
        # Convert to list of tuples with float coordinates
        corner_points = [(float(x), float(y)) for x, y in corners]
        
        return corner_points
        
    except Exception as e:
        print(f"Error processing {mask_path}: {str(e)}")
        return None


def line_intersection(line1: np.ndarray, line2: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Find intersection point of two lines.
    
    Args:
        line1: Line as [x1, y1, x2, y2]
        line2: Line as [x1, y1, x2, y2]
        
    Returns:
        Intersection point (x, y) or None if lines are parallel
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-10:  # Lines are parallel
        return None
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    
    return (x, y)


def find_card_corners_contour_fallback(binary_mask: np.ndarray, mask_path: str) -> Optional[List[Tuple[float, float]]]:
    """
    Fallback method using contour detection when HoughLines fails.
    
    Args:
        binary_mask: Binary mask image
        mask_path: Path to mask (for error reporting)
        
    Returns:
        List of 4 corner points or None
    """
    try:
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print(f"Warning: No contours found in {mask_path}")
            return None
            
        # Find the largest contour (should be the card)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate contour to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # If we don't get exactly 4 points, try different epsilon values
        if len(approx) != 4:
            for eps_factor in [0.01, 0.03, 0.04, 0.05]:
                epsilon = eps_factor * cv2.arcLength(largest_contour, True)
                approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                if len(approx) == 4:
                    break
        
        # If still not 4 points, use convex hull approach
        if len(approx) != 4:
            hull = cv2.convexHull(largest_contour)
            epsilon = 0.02 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)
            
        # If still not 4 points, find 4 extreme points
        if len(approx) != 4:
            # Find extreme points
            leftmost = tuple(largest_contour[largest_contour[:, :, 0].argmin()][0])
            rightmost = tuple(largest_contour[largest_contour[:, :, 0].argmax()][0])
            topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
            bottommost = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])
            
            # Create approximate rectangle from extreme points
            corners = np.array([leftmost, topmost, rightmost, bottommost])
        else:
            corners = approx.reshape(4, 2)
        
        # Sort corners in clockwise order starting from top-left
        corners = sort_corners_clockwise(corners)
        
        # Convert to list of tuples with float coordinates
        corner_points = [(float(x), float(y)) for x, y in corners]
        
        return corner_points
        
    except Exception as e:
        print(f"Error in contour fallback for {mask_path}: {str(e)}")
        return None

def sort_corners_clockwise(corners: np.ndarray) -> np.ndarray:
    """
    Sort corners in clockwise order starting from top-left.
    
    Args:
        corners: Array of 4 corner points
        
    Returns:
        Sorted corners [top-left, top-right, bottom-right, bottom-left]
    """
    # Calculate centroid
    center = np.mean(corners, axis=0)
    
    # Calculate angles from centroid to each corner
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    
    # Sort by angle to get clockwise order
    sorted_indices = np.argsort(angles)
    sorted_corners = corners[sorted_indices]
    
    # Find top-left corner (minimum x + y)
    sums = np.sum(sorted_corners, axis=1)
    top_left_idx = np.argmin(sums)
    
    # Reorder to start from top-left
    ordered_corners = np.roll(sorted_corners, -top_left_idx, axis=0)
    
    return ordered_corners

def process_dataset(dataset_root: str, output_file: str):
    """
    Process entire dataset to extract corner points from masks.
    
    Args:
        dataset_root: Path to dataset root directory
        output_file: Path to output JSON file
    """
    dataset_path = Path(os.path.realpath(dataset_root))
    
    # Process both train and test sets
    corner_data = {}
    
    for split in ['train', 'test']:
        split_path = dataset_path / split
        images_path = split_path / 'images'
        masks_path = split_path / 'masks'
        
        if not images_path.exists() or not masks_path.exists():
            print(f"Warning: {split} split not found at {split_path}")
            continue
            
        print(f"Processing {split} split...")
        
        corner_data[split] = {}
        
        # Get all mask files
        mask_files = list(masks_path.glob('*.png'))
        
        for i, mask_file in enumerate(mask_files):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(mask_files)} masks...")
                
            # Extract corners
            corners = find_card_corners(str(mask_file))
            
            if corners is not None:
                # Store with image filename as key
                image_name = mask_file.stem + '.jpg'
                corner_data[split][image_name] = corners
            else:
                print(f"  Failed to extract corners from {mask_file.name}")
        
        print(f"  Completed {split}: {len(corner_data[split])}/{len(mask_files)} successful")
    
    # Save to JSON file
    output_path = Path(os.path.realpath(output_file))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(corner_data, f, indent=2)
    
    print(f"Corner data saved to {output_path}")
    
    # Print summary statistics
    total_train = len(corner_data.get('train', {}))
    total_test = len(corner_data.get('test', {}))
    print(f"\nSummary:")
    print(f"  Train images with corners: {total_train}")
    print(f"  Test images with corners: {total_test}")
    print(f"  Total: {total_train + total_test}")

def visualize_corners(image_path: str, corners: List[Tuple[float, float]], output_path: str):
    """
    Visualize corner points on image for debugging.
    
    Args:
        image_path: Path to original image
        corners: List of corner points
        output_path: Path to save visualization
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image {image_path}")
        return
    
    # Draw corners
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  # Red, Green, Blue, Yellow
    labels = ['TL', 'TR', 'BR', 'BL']
    
    for i, (x, y) in enumerate(corners):
        cv2.circle(image, (int(x), int(y)), 8, colors[i], -1)
        cv2.putText(image, labels[i], (int(x) + 10, int(y) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[i], 2)
    
    # Draw lines connecting corners
    for i in range(4):
        pt1 = (int(corners[i][0]), int(corners[i][1]))
        pt2 = (int(corners[(i + 1) % 4][0]), int(corners[(i + 1) % 4][1]))
        cv2.line(image, pt1, pt2, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, image)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    # Set paths
    dataset_root = os.path.realpath("../dataset")
    output_file = os.path.realpath("corner_annotations.json")
    
    # Process dataset
    process_dataset(dataset_root, output_file)
    
    # Create some visualizations for debugging
    print("\nCreating sample visualizations...")
    
    with open(output_file, 'r') as f:
        corner_data = json.load(f)
    
    # Visualize a few samples from train set
    train_data = corner_data.get('train', {})
    sample_images = list(train_data.keys())[:5]
    
    vis_dir = Path(os.path.realpath("visualizations"))
    vis_dir.mkdir(exist_ok=True)
    
    for img_name in sample_images:
        img_path = os.path.realpath(f"../dataset/train/images/{img_name}")
        corners = train_data[img_name]
        vis_path = vis_dir / f"vis_{img_name}"
        
        if os.path.exists(img_path):
            visualize_corners(img_path, corners, str(vis_path))
    
    print("Sample visualizations created in 'visualizations' directory")