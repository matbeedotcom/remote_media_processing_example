#!/usr/bin/env python3
"""
ChAruco Calibration Results Visualization Generator

Creates a comprehensive single-image visualization from a calibration results directory.
Shows original frames, warped frames, combined view, and detection statistics.
"""

import argparse
import os
import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CalibrationVisualizer:
    """Generate comprehensive calibration visualization."""
    
    def __init__(self, calibration_dir: str):
        self.calibration_dir = Path(calibration_dir)
        self.images = {}
        self.homographies = {}
        self.info = {}
        
        if not self.calibration_dir.exists():
            raise ValueError(f"Calibration directory not found: {calibration_dir}")
    
    def load_calibration_data(self) -> bool:
        """Load all calibration data from the directory."""
        try:
            logger.info(f"📂 Loading calibration data from: {self.calibration_dir}")
            
            # Load images
            image_files = {
                'combined_view': self.calibration_dir / "combined_view.jpg",
                'originals': list(self.calibration_dir.glob("camera_*_original.jpg")),
                'warped': list(self.calibration_dir.glob("camera_*_warped.jpg"))
            }
            
            # Load combined view
            if image_files['combined_view'].exists():
                self.images['combined_view'] = cv2.imread(str(image_files['combined_view']))
                logger.info(f"📸 Loaded combined view: {self.images['combined_view'].shape}")
            
            # Load original frames
            self.images['originals'] = []
            for img_path in sorted(image_files['originals']):
                img = cv2.imread(str(img_path))
                if img is not None:
                    self.images['originals'].append(img)
                    logger.info(f"📷 Loaded original: {img_path.name} ({img.shape})")
            
            # Load warped frames
            self.images['warped'] = []
            for img_path in sorted(image_files['warped']):
                img = cv2.imread(str(img_path))
                if img is not None:
                    self.images['warped'].append(img)
                    logger.info(f"🔧 Loaded warped: {img_path.name} ({img.shape})")
            
            # Load homographies
            homo_path = self.calibration_dir / "homographies.json"
            if homo_path.exists():
                with open(homo_path, 'r') as f:
                    self.homographies = json.load(f)
                logger.info(f"📊 Loaded homography data for {len(self.homographies)} cameras")
            
            # Load calibration info
            info_path = self.calibration_dir / "calibration_info.txt"
            if info_path.exists():
                with open(info_path, 'r') as f:
                    info_text = f.read()
                    self._parse_calibration_info(info_text)
                logger.info(f"📋 Loaded calibration info")
            
            return len(self.images['originals']) > 0 or self.images.get('combined_view') is not None
            
        except Exception as e:
            logger.error(f"❌ Error loading calibration data: {e}")
            return False
    
    def _parse_calibration_info(self, info_text: str):
        """Parse calibration info text file."""
        self.info = {
            'timestamp': '',
            'num_cameras': 0,
            'board_config': '',
            'detections': []
        }
        
        lines = info_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('Timestamp:'):
                self.info['timestamp'] = line.split(':', 1)[1].strip()
            elif line.startswith('Number of cameras:'):
                self.info['num_cameras'] = int(line.split(':', 1)[1].strip())
            elif line.startswith('Board configuration:'):
                self.info['board_config'] = line.split(':', 1)[1].strip()
            elif line.startswith('Camera ') and ':' in line:
                self.info['detections'].append(line)
    
    def create_grid_layout(self, images: List[np.ndarray], cols: int = 2, 
                          target_size: Tuple[int, int] = (400, 300),
                          titles: Optional[List[str]] = None) -> np.ndarray:
        """Create a grid layout of images."""
        if not images:
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        
        # Resize all images to target size
        resized_images = []
        for i, img in enumerate(images):
            if img is None:
                # Create blank image if None
                blank = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                cv2.putText(blank, "No Image", (target_size[0]//2-50, target_size[1]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
                resized_images.append(blank)
            else:
                resized = cv2.resize(img, target_size)
                
                # Add title if provided
                if titles and i < len(titles):
                    title_height = 30
                    titled_img = np.zeros((target_size[1] + title_height, target_size[0], 3), dtype=np.uint8)
                    titled_img[title_height:, :] = resized
                    
                    # Add title text
                    cv2.putText(titled_img, titles[i], (10, 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    resized_images.append(titled_img)
                else:
                    resized_images.append(resized)
        
        # Calculate grid dimensions
        rows = (len(resized_images) + cols - 1) // cols
        img_height = resized_images[0].shape[0]
        img_width = resized_images[0].shape[1]
        
        # Create grid canvas
        grid_height = rows * img_height
        grid_width = cols * img_width
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # Place images in grid
        for i, img in enumerate(resized_images):
            row = i // cols
            col = i % cols
            
            y_start = row * img_height
            y_end = y_start + img_height
            x_start = col * img_width
            x_end = x_start + img_width
            
            grid[y_start:y_end, x_start:x_end] = img
        
        return grid
    
    def create_info_panel(self, width: int = 800, height: int = 600) -> np.ndarray:
        """Create an information panel with calibration details."""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        panel.fill(40)  # Dark gray background
        
        # Title
        title = "ChAruco Calibration Results"
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
        title_x = (width - title_size[0]) // 2
        cv2.putText(panel, title, (title_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Underline
        cv2.line(panel, (title_x, 60), (title_x + title_size[0], 60), (255, 255, 255), 2)
        
        y_pos = 100
        line_height = 30
        
        # Basic info
        info_items = [
            f"Timestamp: {self.info.get('timestamp', 'N/A')}",
            f"Cameras: {self.info.get('num_cameras', 'N/A')}",
            f"Board: {self.info.get('board_config', 'N/A')}",
            "",  # Spacing
            "Detection Summary:"
        ]
        
        # Add detection results
        for detection in self.info.get('detections', []):
            info_items.append(f"  • {detection}")
        
        # Add homography info
        if self.homographies:
            info_items.extend([
                "",
                f"Homography matrices: {len(self.homographies)} cameras",
                "✅ Perspective correction applied"
            ])
        
        # Add file info
        info_items.extend([
            "",
            "Generated files:",
            "  • combined_view.jpg - Merged camera view",
            "  • camera_X_warped.jpg - Perspective corrected",
            "  • camera_X_original.jpg - With ChAruco overlay",
            "  • homographies.json - Transformation matrices"
        ])
        
        # Draw info text
        for item in info_items:
            if item.strip():
                if item.startswith("  •"):
                    color = (200, 200, 255)  # Light blue for bullet points
                    font_scale = 0.6
                elif item.endswith(":"):
                    color = (255, 255, 200)  # Light yellow for headers
                    font_scale = 0.8
                elif item.startswith("✅"):
                    color = (100, 255, 100)  # Green for success
                    font_scale = 0.7
                else:
                    color = (255, 255, 255)  # White for regular text
                    font_scale = 0.7
                
                cv2.putText(panel, item, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, color, 1)
            
            y_pos += line_height
            
            # Prevent overflow
            if y_pos >= height - 30:
                cv2.putText(panel, "...", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (128, 128, 128), 1)
                break
        
        return panel
    
    def generate_comprehensive_visualization(self, output_path: str = None) -> np.ndarray:
        """Generate a comprehensive single-image visualization."""
        logger.info("🎨 Generating comprehensive calibration visualization...")
        
        # Define layout parameters
        section_width = 800
        section_height = 600
        margin = 20
        
        sections = []
        
        # 1. Combined view (if available)
        if self.images.get('combined_view') is not None:
            combined_resized = cv2.resize(self.images['combined_view'], (section_width, section_height))
            
            # Add title
            title_img = np.zeros((50, section_width, 3), dtype=np.uint8)
            title_img.fill(60)
            cv2.putText(title_img, "Combined Multi-Camera View", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            combined_section = np.vstack([title_img, combined_resized])
            sections.append(combined_section)
        
        # 2. Original frames grid
        if self.images.get('originals'):
            original_titles = [f"Camera {i} (Original)" for i in range(len(self.images['originals']))]
            originals_grid = self.create_grid_layout(
                self.images['originals'], 
                cols=2, 
                target_size=(section_width//2-10, section_height//2-40),
                titles=original_titles
            )
            
            # Resize to match section width
            if originals_grid.shape[1] != section_width:
                originals_grid = cv2.resize(originals_grid, (section_width, 
                                          int(originals_grid.shape[0] * section_width / originals_grid.shape[1])))
            
            # Add title
            title_img = np.zeros((50, section_width, 3), dtype=np.uint8)
            title_img.fill(60)
            cv2.putText(title_img, "Original Frames with ChAruco Detection", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            originals_section = np.vstack([title_img, originals_grid])
            sections.append(originals_section)
        
        # 3. Warped frames grid
        if self.images.get('warped'):
            warped_titles = [f"Camera {i} (Warped)" for i in range(len(self.images['warped']))]
            warped_grid = self.create_grid_layout(
                self.images['warped'], 
                cols=2, 
                target_size=(section_width//2-10, section_height//2-40),
                titles=warped_titles
            )
            
            # Resize to match section width
            if warped_grid.shape[1] != section_width:
                warped_grid = cv2.resize(warped_grid, (section_width, 
                                       int(warped_grid.shape[0] * section_width / warped_grid.shape[1])))
            
            # Add title
            title_img = np.zeros((50, section_width, 3), dtype=np.uint8)
            title_img.fill(60)
            cv2.putText(title_img, "Perspective Corrected Frames", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            warped_section = np.vstack([title_img, warped_grid])
            sections.append(warped_section)
        
        # 4. Information panel
        info_panel = self.create_info_panel(section_width, section_height)
        title_img = np.zeros((50, section_width, 3), dtype=np.uint8)
        title_img.fill(60)
        cv2.putText(title_img, "Calibration Information", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        info_section = np.vstack([title_img, info_panel])
        sections.append(info_section)
        
        # Combine all sections vertically
        if sections:
            # Add margins between sections
            final_sections = []
            for i, section in enumerate(sections):
                final_sections.append(section)
                if i < len(sections) - 1:  # Don't add margin after last section
                    margin_img = np.zeros((margin, section_width, 3), dtype=np.uint8)
                    margin_img.fill(20)  # Dark margin
                    final_sections.append(margin_img)
            
            final_image = np.vstack(final_sections)
        else:
            # Fallback if no sections
            final_image = np.zeros((600, 800, 3), dtype=np.uint8)
            cv2.putText(final_image, "No calibration data found", (200, 300), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (128, 128, 128), 2)
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(output_path, final_image)
            logger.info(f"💾 Saved comprehensive visualization: {output_path}")
        
        return final_image


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate comprehensive ChAruco calibration visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_calibration_visualization.py calibration_results_20250807_175548
  python generate_calibration_visualization.py calibration_results_20250807_175548 -o my_calibration.jpg
  python generate_calibration_visualization.py calibration_results_20250807_175548 --preview
        """
    )
    
    parser.add_argument(
        "calibration_dir",
        help="Path to calibration results directory"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output image path (default: visualization_<timestamp>.jpg)"
    )
    
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show preview window instead of saving"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create visualizer
        visualizer = CalibrationVisualizer(args.calibration_dir)
        
        # Load data
        if not visualizer.load_calibration_data():
            logger.error("❌ Failed to load calibration data")
            return 1
        
        # Generate output filename if not provided
        output_path = args.output
        if not output_path and not args.preview:
            cal_dir_name = Path(args.calibration_dir).name
            output_path = f"visualization_{cal_dir_name}.jpg"
        
        # Generate visualization
        result_image = visualizer.generate_comprehensive_visualization(
            output_path if not args.preview else None
        )
        
        # Show preview if requested
        if args.preview:
            logger.info("👁️  Showing preview (press any key to close)")
            cv2.imshow("ChAruco Calibration Visualization", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            logger.info(f"✅ Generated visualization: {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())