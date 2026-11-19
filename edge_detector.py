import argparse
import os
import time

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 1. Read the image file and turn its pixels into a two-dimensional array
def load_pixels(path, show=True):
    image = Image.open(path)
    if show:
        print(f"Opened {path}: {image.width} x {image.height} pixels, mode {image.mode}")
    rgb = np.array(image.convert("RGB"), dtype=np.float64)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    if show:
        print(f"Color array: {rgb.shape} (rows, columns, channels)")
        print(f"Gray array:  {gray.shape} (rows, columns), values {gray.min():.0f} to {gray.max():.0f}")
        print("Top-left 4 x 4 corner of the gray array:")
        print(np.round(gray[:4, :4]).astype(int))
    return rgb, gray


# 2. Convolution: slide a small kernel over the array
def convolve(array, kernel):
    rows, cols = array.shape
    pad_r, pad_c = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(array, ((pad_r, pad_r), (pad_c, pad_c)), mode="edge")
    flipped = kernel[::-1, ::-1]
    result = np.zeros_like(array)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            result += flipped[i, j] * padded[i:i + rows, j:j + cols]
    return result


def gaussian_kernel(size, sigma):
    axis = np.arange(size) - size // 2
    x, y = np.meshgrid(axis, axis)
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


# 3. Canny stage 1: smooth the image to reduce noise
def smooth(gray, sigma):
    return convolve(gray, gaussian_kernel(5, sigma))


# 4. Canny stage 2: find the gradient strength and direction with Sobel kernels
def gradients(smoothed):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sobel_y = sobel_x.T
    gx = convolve(smoothed, sobel_x)
    gy = convolve(smoothed, sobel_y)
    magnitude = np.hypot(gx, gy)
    direction = np.degrees(np.arctan2(gy, gx)) % 180
    return magnitude, direction


# 5. Canny stage 3: keep only the strongest pixel across each edge
def thin_edges(magnitude, direction):
    p = np.pad(magnitude, 1)
    rows, cols = magnitude.shape
    left, right = p[1:rows + 1, 0:cols], p[1:rows + 1, 2:cols + 2]
    up, down = p[0:rows, 1:cols + 1], p[2:rows + 2, 1:cols + 1]
    up_left, down_right = p[0:rows, 0:cols], p[2:rows + 2, 2:cols + 2]
    up_right, down_left = p[0:rows, 2:cols + 2], p[2:rows + 2, 0:cols]

    horizontal = (direction < 22.5) | (direction >= 157.5)
    diagonal_down = (direction >= 22.5) & (direction < 67.5)
    vertical = (direction >= 67.5) & (direction < 112.5)

    before = np.where(horizontal, left, np.where(diagonal_down, up_left, np.where(vertical, up, up_right)))
    after = np.where(horizontal, right, np.where(diagonal_down, down_right, np.where(vertical, down, down_left)))
    return np.where((magnitude >= before) & (magnitude >= after), magnitude, 0.0)


# 6. Canny stage 4: double threshold, then keep weak edges that touch strong ones
def link_edges(thin, low, high):
    strong = thin >= high
    weak = (thin >= low) & ~strong
    edges = strong.copy()
    while True:
        p = np.pad(edges, 1)
        rows, cols = edges.shape
        near = np.zeros_like(edges)
        for dr in range(3):
            for dc in range(3):
                near |= p[dr:dr + rows, dc:dc + cols]
        added = weak & near & ~edges
        if not added.any():
            return edges
        edges |= added


# 7. Run the full detector
def canny(gray, sigma=1.4, low_ratio=0.07, high_ratio=0.20):
    smoothed = smooth(gray, sigma)
    magnitude, direction = gradients(smoothed)
    thin = thin_edges(magnitude, direction)
    top = magnitude.max()
    if top < 1e-6:
        return smoothed, magnitude, np.zeros(gray.shape, dtype=bool)
    edges = link_edges(thin, low_ratio * top, high_ratio * top)
    return smoothed, magnitude, edges


# 8. Save the edge image and a four-panel figure for comparison
def save_results(rgb, gray, magnitude, edges, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    edge_path = os.path.join(out_dir, "edges.png")
    Image.fromarray((edges * 255).astype(np.uint8)).save(edge_path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [
        (rgb.astype(np.uint8), "1. Original image", None),
        (gray, "2. Grayscale pixel array", "gray"),
        (magnitude, "3. Gradient strength (Sobel)", "magma"),
        (edges, f"4. Canny edges ({int(edges.sum()):,} pixels)", "gray"),
    ]
    for ax, (data, title, cmap) in zip(axes.ravel(), panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=13)
        ax.axis("off")
    fig.tight_layout()
    figure_path = os.path.join(out_dir, "edge_detection_result.png")
    fig.savefig(figure_path, dpi=200)
    return edge_path, figure_path


# 9. Main program
def main():
    parser = argparse.ArgumentParser(description="Canny edge detection written from scratch with NumPy")
    parser.add_argument("--image", default="sample_image.png", help="path to the input image")
    parser.add_argument("--out", default="output", help="folder for the results")
    parser.add_argument("--sigma", type=float, default=1.4, help="blur strength")
    parser.add_argument("--low", type=float, default=0.07, help="low threshold as a share of the strongest edge")
    parser.add_argument("--high", type=float, default=0.20, help="high threshold as a share of the strongest edge")
    args = parser.parse_args()

    rgb, gray = load_pixels(args.image)
    start = time.time()
    smoothed, magnitude, edges = canny(gray, args.sigma, args.low, args.high)
    seconds = time.time() - start
    share = 100 * edges.mean()
    print(f"Edge detection finished in {seconds:.2f} seconds")
    print(f"Edge pixels: {int(edges.sum()):,} of {edges.size:,} ({share:.1f}%)")
    edge_path, figure_path = save_results(rgb, gray, magnitude, edges, args.out)
    print(f"Saved {edge_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
