
# Lane Curve Detection 🚗🛣️  

This project implements **lane curve detection** in video streams using **OpenCV** and **NumPy**. It transforms the perspective of a road image to a **bird's-eye view**, applies **HSV thresholding**, and detects lane curves using a **sliding window approach**. A beep sound is triggered when excessive curvature is detected.  

## Features ✨  
- **Perspective Transformation:** Converts the road view into a **bird's-eye perspective**.  
- **HSV Thresholding:** Dynamically adjusts lane detection parameters using trackbars.  
- **Lane Detection Algorithm:** Uses a **histogram-based sliding window approach** to track lanes.  
- **Curvature Calculation:** Fits a polynomial curve to detected lanes.  
- **Alert System:** Triggers an **audible beep** if lane curvature exceeds a threshold.  

## Requirements 📦  
- Python  
- OpenCV  
- NumPy  
- Winsound (Windows only)  

## How to Run 🚀  
1. Install dependencies:  
   ```bash
   pip install opencv-python numpy
   ```
2. Place a road video (`challenge.mp4`) in the project directory.  
3. Run the script:  
   ```bash
   python lane_curve_detection.py
   ```
4. Adjust HSV values using trackbars for optimal lane detection.  
5. Press **ESC** to exit.  

## Demo 🎥  
(TODO: Add a sample GIF or video of the lane detection in action.)  
