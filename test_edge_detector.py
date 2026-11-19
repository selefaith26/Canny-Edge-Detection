import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from edge_detector import canny, convolve, gaussian_kernel, load_pixels

SAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_image.png")


def make_image_file(array):
    handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    handle.close()
    Image.fromarray(array.astype(np.uint8)).save(handle.name)
    return handle.name


# 1. Converting an image file into a two-dimensional pixel array
class TestImageConversion(unittest.TestCase):
    def test_color_image_becomes_two_dimensional(self):
        path = make_image_file(np.zeros((30, 50, 3)))
        rgb, gray = load_pixels(path, show=False)
        os.remove(path)
        self.assertEqual(rgb.shape, (30, 50, 3))
        self.assertEqual(gray.shape, (30, 50))

    def test_gray_values_follow_the_luminance_formula(self):
        pixels = np.array([[[255, 255, 255], [255, 0, 0]], [[0, 255, 0], [0, 0, 255]]])
        path = make_image_file(pixels)
        _, gray = load_pixels(path, show=False)
        os.remove(path)
        expected = [[255.0, 76.245], [149.685, 29.07]]
        np.testing.assert_allclose(gray, expected, atol=1e-6)

    def test_sample_image_loads(self):
        _, gray = load_pixels(SAMPLE, show=False)
        self.assertEqual(gray.ndim, 2)
        self.assertGreaterEqual(gray.min(), 0)
        self.assertLessEqual(gray.max(), 255)


# 2. Convolution and the Gaussian kernel
class TestConvolution(unittest.TestCase):
    def test_identity_kernel_changes_nothing(self):
        array = np.random.default_rng(0).random((20, 25))
        identity = np.zeros((3, 3))
        identity[1, 1] = 1
        np.testing.assert_allclose(convolve(array, identity), array)

    def test_gaussian_kernel_sums_to_one_and_is_symmetric(self):
        kernel = gaussian_kernel(5, 1.4)
        self.assertAlmostEqual(kernel.sum(), 1.0)
        np.testing.assert_allclose(kernel, kernel.T)
        np.testing.assert_allclose(kernel, kernel[::-1, ::-1])

    def test_blurring_a_flat_image_keeps_it_flat(self):
        flat = np.full((15, 15), 90.0)
        np.testing.assert_allclose(convolve(flat, gaussian_kernel(5, 1.4)), flat)

    def test_matches_a_hand_calculation(self):
        array = np.arange(25, dtype=float).reshape(5, 5)
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        self.assertEqual(convolve(array, kernel)[2, 2], 7 + 17 + 11 + 13)


# 3. Edge detection on simple pictures with a known answer
class TestEdgeDetection(unittest.TestCase):
    def test_flat_image_has_no_edges(self):
        _, _, edges = canny(np.full((40, 40), 120.0))
        self.assertEqual(edges.sum(), 0)

    def test_vertical_step_gives_one_thin_edge_in_the_middle(self):
        step = np.zeros((40, 60))
        step[:, 30:] = 255
        _, _, edges = canny(step)
        columns = np.where(edges.any(axis=0))[0]
        self.assertTrue(((columns >= 28) & (columns <= 31)).all())
        self.assertTrue((edges.sum(axis=1)[5:-5] == 1).all())

    def test_horizontal_step_gives_one_thin_edge_in_the_middle(self):
        step = np.zeros((60, 40))
        step[30:, :] = 255
        _, _, edges = canny(step)
        rows = np.where(edges.any(axis=1))[0]
        self.assertTrue(((rows >= 28) & (rows <= 31)).all())
        self.assertTrue((edges.sum(axis=0)[5:-5] == 1).all())

    def test_square_is_outlined_and_its_inside_is_empty(self):
        square = np.zeros((60, 60))
        square[20:40, 20:40] = 255
        _, _, edges = canny(square)
        self.assertGreater(edges.sum(), 60)
        self.assertEqual(edges[24:36, 24:36].sum(), 0)
        self.assertEqual(edges[:12, :].sum(), 0)

    def test_result_is_a_boolean_array_the_size_of_the_input(self):
        _, gray = load_pixels(SAMPLE, show=False)
        _, _, edges = canny(gray)
        self.assertEqual(edges.shape, gray.shape)
        self.assertEqual(edges.dtype, bool)


# 4. Agreement with the OpenCV reference implementation
class TestAgainstOpenCV(unittest.TestCase):
    def test_edges_agree_with_opencv_within_one_pixel(self):
        try:
            import cv2
            from scipy.ndimage import binary_dilation
        except ImportError:
            self.skipTest("opencv-python and scipy are not installed")
        _, gray = load_pixels(SAMPLE, show=False)
        smoothed, magnitude, mine = canny(gray)
        top = magnitude.max()
        reference = cv2.Canny(np.round(smoothed).astype(np.uint8), 0.07 * top, 0.20 * top,
                              apertureSize=3, L2gradient=True) > 0
        grid = np.ones((3, 3), dtype=bool)
        precision = (mine & binary_dilation(reference, grid)).sum() / mine.sum()
        recall = (reference & binary_dilation(mine, grid)).sum() / reference.sum()
        print(f"\nAgreement with OpenCV: precision {precision:.1%}, recall {recall:.1%}")
        self.assertGreater(precision, 0.95)
        self.assertGreater(recall, 0.95)


if __name__ == "__main__":
    unittest.main(verbosity=2)
