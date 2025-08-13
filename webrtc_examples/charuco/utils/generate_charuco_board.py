#!/usr/bin/env python3
"""
Generate a proper ChAruco calibration board with correct marker IDs.
"""

import cv2
import numpy as np
import argparse

def generate_charuco_board(
    squares_x=5,
    squares_y=4, 
    square_length=0.04,
    marker_length=0.02,
    dict_name="DICT_4X4_50",
    margins=0.005,
    dpi=200,
    output_file="charuco_board_5x4.png"
):
    """Generate a ChAruco board image file."""
    
    print(f"Generating ChAruco Board:")
    print(f"  Size: {squares_x}×{squares_y} squares")
    print(f"  Square length: {square_length*1000:.1f}mm")
    print(f"  Marker length: {marker_length*1000:.1f}mm")
    print(f"  Dictionary: {dict_name}")
    print(f"  Margins: {margins*1000:.1f}mm")
    print(f"  DPI: {dpi}")
    
    # Get ArUco dictionary
    dict_id = getattr(cv2.aruco, dict_name)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    
    # Create ChAruco board
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length,
        marker_length,
        aruco_dict
    )
    
    # Calculate image size in pixels
    # Board size in meters
    board_width = squares_x * square_length + 2 * margins
    board_height = squares_y * square_length + 2 * margins
    
    # Convert to pixels (1 inch = 0.0254 meters)
    pixels_per_meter = dpi / 0.0254
    img_width = int(board_width * pixels_per_meter)
    img_height = int(board_height * pixels_per_meter)
    
    print(f"\nImage dimensions: {img_width}×{img_height} pixels")
    print(f"Physical size: {board_width*1000:.1f}×{board_height*1000:.1f}mm")
    
    # Generate the board image
    board_image = board.generateImage((img_width, img_height))
    
    # Add white border/margins
    if margins > 0:
        margin_pixels = int(margins * pixels_per_meter)
        board_with_margin = np.ones((img_height, img_width), dtype=np.uint8) * 255
        
        # Calculate position to center the board
        y_start = margin_pixels
        y_end = img_height - margin_pixels
        x_start = margin_pixels  
        x_end = img_width - margin_pixels
        
        # Resize board to fit within margins
        board_resized = cv2.resize(board_image, (x_end - x_start, y_end - y_start))
        board_with_margin[y_start:y_end, x_start:x_end] = board_resized
        board_image = board_with_margin
    
    # Save the board
    cv2.imwrite(output_file, board_image)
    print(f"\n✅ Board saved to: {output_file}")
    
    # Print marker ID layout for verification
    print(f"\nMarker ID Layout (for {squares_x}×{squares_y} board):")
    print("The board will have the following marker IDs:")
    
    # Calculate and show expected marker IDs
    marker_id = 0
    for y in range(squares_y):
        row_ids = []
        for x in range(squares_x):
            # ChAruco places markers at alternating positions
            if (x + y) % 2 == 0:  # Even sum = black square (marker)
                if x < squares_x - 1 or y < squares_y - 1:  # Not the last corner
                    row_ids.append(f"{marker_id:2d}")
                    marker_id += 1
            else:  # Odd sum = white square
                row_ids.append("  ")
        print(f"Row {y}: [" + ", ".join(row_ids) + "]")
    
    print(f"\nTotal markers: {marker_id}")
    print(f"Expected ChAruco corners: {(squares_x-1)*(squares_y-1)}")
    
    # Create a verification script
    verification_script = f"""
# To verify this board:
# 1. Print at {dpi} DPI
# 2. Measure that squares are {square_length*1000:.1f}mm
# 3. Measure that markers are {marker_length*1000:.1f}mm
# 4. Use this configuration in your code:
charuco_config = {{
    "squares_x": {squares_x},
    "squares_y": {squares_y},
    "square_length": {square_length},
    "marker_length": {marker_length},
    "dictionary": "{dict_name}"
}}
"""
    
    with open(output_file.replace('.png', '_config.txt'), 'w') as f:
        f.write(verification_script)
    
    print(f"📝 Configuration saved to: {output_file.replace('.png', '_config.txt')}")
    
    return board_image


def main():
    parser = argparse.ArgumentParser(description="Generate ChAruco calibration board")
    parser.add_argument("--squares_x", type=int, default=5, help="Number of squares in X")
    parser.add_argument("--squares_y", type=int, default=4, help="Number of squares in Y")
    parser.add_argument("--square_length", type=float, default=0.04, help="Square size in meters")
    parser.add_argument("--marker_length", type=float, default=0.02, help="Marker size in meters")
    parser.add_argument("--dictionary", type=str, default="DICT_4X4_50", help="ArUco dictionary")
    parser.add_argument("--margins", type=float, default=0.005, help="Margin size in meters")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for printing")
    parser.add_argument("--output", type=str, default="charuco_board_5x4.png", help="Output filename")
    
    args = parser.parse_args()
    
    generate_charuco_board(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length=args.square_length,
        marker_length=args.marker_length,
        dict_name=args.dictionary,
        margins=args.margins,
        dpi=args.dpi,
        output_file=args.output
    )


if __name__ == "__main__":
    main()