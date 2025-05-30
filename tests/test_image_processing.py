"""
Tests for image processing utilities.
"""

import pytest
import numpy as np
import cv2
from PIL import Image
from unittest.mock import Mock, patch

from src.utils.image_processing import (
    apply_nms,
    calculate_iou,
    resize_image,
    enhance_image,
    pil_to_bytes
)


class TestImageProcessing:
    """Test image processing utilities."""

    def test_apply_nms_basic(self):
        """Test basic NMS functionality."""
        # Create sample boxes and scores
        boxes = np.array([
            [100, 100, 200, 200],
            [110, 110, 210, 210],
            [300, 300, 400, 400]
        ])
        scores = np.array([0.9, 0.8, 0.7])

        result = apply_nms(boxes, scores, nms_thresh=0.5)
        assert isinstance(result, list)
        assert len(result) <= len(boxes)

    def test_apply_nms_no_scores(self):
        """Test NMS without scores."""
        boxes = np.array([
            [100, 100, 200, 200],
            [300, 300, 400, 400]
        ])

        result = apply_nms(boxes, nms_thresh=0.5)
        assert isinstance(result, list)

    def test_calculate_iou_identical_boxes(self):
        """Test IoU calculation for identical boxes."""
        box1 = (100, 100, 200, 200)
        box2 = (100, 100, 200, 200)

        iou = calculate_iou(box1, box2)
        assert iou == 1.0

    def test_calculate_iou_no_overlap(self):
        """Test IoU calculation for non-overlapping boxes."""
        box1 = (100, 100, 200, 200)
        box2 = (300, 300, 400, 400)

        iou = calculate_iou(box1, box2)
        assert iou == 0.0

    def test_calculate_iou_partial_overlap(self):
        """Test IoU calculation for partially overlapping boxes."""
        box1 = (100, 100, 200, 200)
        box2 = (150, 150, 250, 250)

        iou = calculate_iou(box1, box2)
        assert 0 < iou < 1

    def test_resize_image_larger_than_max(self, sample_image_array):
        """Test image resizing when larger than maximum."""
        # Create a large image
        large_image = np.ones((2000, 3000, 3), dtype=np.uint8) * 255

        resized = resize_image(large_image, max_width=1920, max_height=1080)

        assert resized.shape[0] <= 1080
        assert resized.shape[1] <= 1920
        assert len(resized.shape) == 3

    def test_resize_image_smaller_than_max(self, sample_image_array):
        """Test image resizing when smaller than maximum."""
        # Use the sample image which is 600x800
        resized = resize_image(
            sample_image_array, max_width=1920, max_height=1080)

        # Should remain unchanged
        assert resized.shape == sample_image_array.shape

    def test_enhance_image(self, sample_image_array):
        """Test image enhancement."""
        enhanced = enhance_image(sample_image_array)

        assert enhanced.shape == sample_image_array.shape
        assert enhanced.dtype == sample_image_array.dtype

    def test_pil_to_bytes_png(self, sample_image):
        """Test PIL image to bytes conversion (PNG)."""
        image_bytes = pil_to_bytes(sample_image, format='PNG')

        assert isinstance(image_bytes, bytes)
        assert len(image_bytes) > 0

        # Verify it's a valid PNG by reading it back
        from io import BytesIO
        recovered_image = Image.open(BytesIO(image_bytes))
        assert recovered_image.size == sample_image.size

    def test_pil_to_bytes_jpeg(self, sample_image):
        """Test PIL image to bytes conversion (JPEG)."""
        image_bytes = pil_to_bytes(sample_image, format='JPEG')

        assert isinstance(image_bytes, bytes)
        assert len(image_bytes) > 0


class TestImageProcessingEdgeCases:
    """Test edge cases for image processing."""

    def test_empty_boxes_nms(self):
        """Test NMS with empty boxes array."""
        boxes = np.array([]).reshape(0, 4)
        result = apply_nms(boxes)
        assert len(result) == 0

    def test_single_box_nms(self):
        """Test NMS with single box."""
        boxes = np.array([[100, 100, 200, 200]])
        result = apply_nms(boxes)
        assert len(result) == 1

    def test_invalid_box_iou(self):
        """Test IoU calculation with invalid box dimensions."""
        # Box with zero area
        box1 = (100, 100, 100, 100)
        box2 = (100, 100, 200, 200)

        iou = calculate_iou(box1, box2)
        assert iou == 0.0

    def test_resize_very_small_image(self):
        """Test resizing very small image."""
        small_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
        resized = resize_image(small_image, max_width=1920, max_height=1080)

        # Should remain unchanged
        assert resized.shape == small_image.shape

    def test_enhance_grayscale_image(self):
        """Test enhancement on grayscale image."""
        gray_image = np.ones((600, 800), dtype=np.uint8) * 128
        enhanced = enhance_image(gray_image)

        assert enhanced.shape == gray_image.shape
        assert enhanced.dtype == gray_image.dtype


class TestImageProcessingIntegration:
    """Integration tests for image processing pipeline."""

    @patch('src.utils.image_processing.QReader')
    def test_qr_code_detection_mock(self, mock_qreader, sample_image_array):
        """Test QR code detection with mock."""
        # Mock QReader
        mock_qr_instance = Mock()
        mock_qr_instance.detect_and_decode.return_value = ["QR123456"]
        mock_qreader.return_value = mock_qr_instance

        # Test would go here - currently QR detection is not exposed
        # as a separate function but integrated in the main pipeline

    def test_corner_preprocessing_basic(self, sample_image_array):
        """Test corner preprocessing with basic input."""
        # This would test the corner_preprocess_image function
        # Currently it requires device parameter and torch models
        device = "cpu"

        # Mock the preprocessing (since we don't have the actual models)
        with patch('torch.cuda.is_available', return_value=False):
            # The actual function requires trained models
            # This is a placeholder for when we implement the test
            pass

    def test_warp_and_recognize_mock(self, sample_image_array):
        """Test image warping and recognition with mocks."""
        # Mock the warping and recognition process
        frame = sample_image_array
        corners = ((100, 100), (700, 100), (700, 500), (100, 500))

        with patch('cv2.getPerspectiveTransform') as mock_transform, \
                patch('cv2.warpPerspective') as mock_warp:

            mock_transform.return_value = np.eye(3)
            mock_warp.return_value = frame

            # Test would call warp_and_recognize function
            # Currently requires VietOCR model
            pass
