# Canny Edge Detection in Python

Edge detector written with NumPy for CST-585 Machine Vision Systems Comparison and Application Part Two.

The program opens an image, converts its pixels into a two-dimensional grayscale array, runs the Canny algorithm (Gaussian smoothing, Sobel gradients, non-maximum suppression, and double-threshold hysteresis), and saves an image showing the edges.

## Files

| File | Purpose |
| --- | --- |
| edge_detector.py | The program |
| test_edge_detector.py | 13 unit tests, including a comparison with OpenCV |
| sample_image.png | Sample photograph (coffee cup, CC0, from scikit-image) |
| output/edges.png | Edges only (white on black) |
| output/edge_detection_result.png | Four-panel figure: original, grayscale array, gradient strength, edges |
| run_output.txt, test_output.txt | Saved console output |

## Setup

    pip install numpy pillow matplotlib

The OpenCV comparison test is skipped unless `opencv-python` and `scipy` are also installed.

## How to run

    python edge_detector.py

Use your own image and settings:

    python edge_detector.py --image my_photo.jpg --sigma 1.4 --low 0.07 --high 0.20 --out my_output

Run the tests:

    python -m unittest -v

## Source of the algorithm

Canny, J. (1986). A computational approach to edge detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 8(6), 679-698.

OpenCV. (n.d.). Canny edge detection. OpenCV 4.13.0 documentation. https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
