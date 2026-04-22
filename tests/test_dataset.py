"""Tests for dataset loading and augmentation."""

import numpy as np
import pytest
import torch

from src.dataset import (
    SyntheticTrayDataset,
    get_train_transforms,
    get_val_transforms,
)


class TestSyntheticDataset:
    """Test synthetic dataset generation."""
    
    def test_synthetic_dataset_shapes(self):
        """Synthetic dataset should return correct tensor shapes."""
        ds = SyntheticTrayDataset(n_samples=10, image_size=224, num_classes=43)
        sample = ds[0]
        
        assert sample["image"].shape == (3, 224, 224)
        assert sample["boxes"].shape[1] == 4  # xyxy format
        assert len(sample["labels"]) == len(sample["boxes"])
        assert len(sample["portions"]) == len(sample["boxes"])
    
    def test_synthetic_dataset_deterministic(self):
        """Same index should return same data (for testing)."""
        ds = SyntheticTrayDataset(n_samples=10, image_size=224, num_classes=43)
        
        sample1 = ds[0]
        sample2 = ds[0]
        
        assert torch.allclose(sample1["image"], sample2["image"])
        assert torch.allclose(sample1["boxes"], sample2["boxes"])
    
    def test_synthetic_dataset_batch(self):
        """Should be able to iterate over multiple samples."""
        ds = SyntheticTrayDataset(n_samples=5, image_size=224, num_classes=43)
        
        count = 0
        for sample in ds:
            assert sample["image"].shape == (3, 224, 224)
            count += 1
        
        assert count == 5


class TestTrainTransforms:
    """Test training augmentation transforms."""
    
    def test_train_augmentation_preserves_labels(self):
        """Augmented images should preserve label integrity."""
        transform = get_train_transforms(640)
        
        # Create dummy image and annotations
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = [[50, 50, 150, 150], [200, 200, 300, 300]]
        labels = [5, 10]
        portions = [150.0, 200.0]
        
        transformed = transform(
            image=img,
            bboxes=boxes,
            labels=labels,
            portions=portions
        )
        
        assert transformed["image"].shape[0] == 3  # CHW format
        assert transformed["image"].shape[1:] == (640, 640)
        assert len(transformed["labels"]) == len(labels)
        assert len(transformed["bboxes"]) == len(boxes)
    
    def test_train_augmentation_produces_valid_bboxes(self):
        """Augmentation should produce valid bounding boxes."""
        transform = get_train_transforms(640)
        
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = [[50, 50, 150, 150], [200, 200, 300, 300]]
        labels = [1, 2]
        portions = [100.0, 150.0]
        
        transformed = transform(
            image=img,
            bboxes=boxes,
            labels=labels,
            portions=portions
        )
        
        # Check bbox is still valid (x2 > x1, y2 > y1)
        for bbox in transformed["bboxes"]:
            assert bbox[2] > bbox[0], f"Invalid bbox: x2 <= x1"
            assert bbox[3] > bbox[1], f"Invalid bbox: y2 <= y1"
    
    def test_train_augmentation_randomness(self):
        """Training transforms should be random (different each call)."""
        transform = get_train_transforms(640)
        
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = [[50, 50, 150, 150]]
        labels = [5]
        portions = [150.0]
        
        # Run twice with same input
        t1 = transform(image=img, bboxes=boxes, labels=labels, portions=portions)
        t2 = transform(image=img, bboxes=boxes, labels=labels, portions=portions)
        
        # Images should be different (due to augmentation)
        assert not torch.allclose(t1["image"], t2["image"])


class TestValTransforms:
    """Test validation transforms (no augmentation)."""
    
    def test_val_augmentation_deterministic(self):
        """Validation transforms should be deterministic."""
        transform = get_val_transforms(640)
        
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = [[50, 50, 150, 150]]
        labels = [5]
        portions = [150.0]
        
        # Run twice with same input
        t1 = transform(image=img, bboxes=boxes, labels=labels, portions=portions)
        t2 = transform(image=img, bboxes=boxes, labels=labels, portions=portions)
        
        assert torch.allclose(t1["image"], t2["image"])
    
    def test_val_preserves_boxes(self):
        """Validation should preserve bounding boxes."""
        transform = get_val_transforms(640)
        
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = [[50, 50, 150, 150], [200, 200, 300, 300]]
        labels = [1, 2]
        portions = [100.0, 150.0]
        
        transformed = transform(
            image=img,
            bboxes=boxes,
            labels=labels,
            portions=portions
        )
        
        assert len(transformed["bboxes"]) == len(boxes)
        assert transformed["labels"] == labels
        assert transformed["portions"] == portions
    
    def test_val_output_size(self):
        """Validation should resize to target size."""
        transform = get_val_transforms(640)
        
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = [[50, 50, 150, 150]]
        labels = [1]
        portions = [100.0]
        
        transformed = transform(
            image=img,
            bboxes=boxes,
            labels=labels,
            portions=portions
        )
        
        assert transformed["image"].shape == (3, 640, 640)


class TestBboxFormat:
    """Test bounding box format handling."""
    
    def test_bbox_xyxy_format(self):
        """BBoxes should be in xyxy format."""
        transform = get_val_transforms(640)
        
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # xyxy format: [x1, y1, x2, y2]
        boxes = [[50, 50, 150, 150]]
        labels = [1]
        portions = [100.0]
        
        transformed = transform(
            image=img,
            bboxes=boxes,
            labels=labels,
            portions=portions
        )
        
        bbox = transformed["bboxes"][0]
        # After transform, should still be xyxy
        assert bbox[2] > bbox[0], "x2 should be > x1"
        assert bbox[3] > bbox[1], "y2 should be > y1"
    
    def test_bbox_clipping(self):
        """BBoxes should be clipped to image boundaries."""
        transform = get_val_transforms(640)
        
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Box partially outside image
        boxes = [[500, 400, 700, 600]]
        labels = [1]
        portions = [100.0]
        
        transformed = transform(
            image=img,
            bboxes=boxes,
            labels=labels,
            portions=portions
        )
        
        # Bbox should be clipped or dropped
        # Albumentations handles this automatically
        if len(transformed["bboxes"]) > 0:
            bbox = transformed["bboxes"][0]
            assert 0 <= bbox[0] <= 640
            assert 0 <= bbox[1] <= 640
            assert 0 <= bbox[2] <= 640
            assert 0 <= bbox[3] <= 640


class TestEmptyAnnotations:
    """Test handling of empty annotations."""
    
    def test_empty_boxes_handling(self):
        """Should handle images with no annotations."""
        transform = get_val_transforms(640)
        
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        boxes = []
        labels = []
        portions = []
        
        transformed = transform(
            image=img,
            bboxes=boxes,
            labels=labels,
            portions=portions
        )
        
        assert len(transformed["bboxes"]) == 0
        assert len(transformed["labels"]) == 0
        assert len(transformed["portions"]) == 0
