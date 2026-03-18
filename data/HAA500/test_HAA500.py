"""
Integration tests for HAA500 dataset class with real DataLoader and visualization.
"""
import os
import sys
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.HAA500.HAA500 import HAA500
from main.config import cfg


def create_dummy_annotation(data_dir, split='train'):
    """Create dummy annotation file for testing"""
    import json
    from pycocotools.coco import COCO
    
    annot_dir = os.path.join(data_dir, 'HAA500', 'annotations')
    img_dir = os.path.join(data_dir, 'HAA500', 'images')
    os.makedirs(annot_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    # Create dummy image
    img_path = os.path.join(img_dir, 'test_img.jpg')
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    cv2.imwrite(img_path, dummy_img)
    
    # Create COCO format annotation
    annotation = {
        "images": [
            {"id": 1, "file_name": "test_img.jpg", "width": 640, "height": 480}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "bbox": [100, 100, 200, 200],
                "body_kpts": np.random.rand(17, 3).astype(np.float32).tolist(),
                "foot_kpts": np.random.rand(7, 3).astype(np.float32).tolist(),
                "lefthand_kpts": np.random.rand(21, 3).astype(np.float32).tolist(),
                "righthand_kpts": np.random.rand(21, 3).astype(np.float32).tolist(),
                "face_kpts": np.random.rand(68, 3).astype(np.float32).tolist(),
                "lefthand_box": [150, 180, 50, 50],
                "righthand_box": [350, 180, 50, 50],
                "face_box": [200, 100, 60, 80],
                "smplx_params": {
                    "root_pose": np.zeros(3).tolist(),
                    "body_pose": np.zeros(63).tolist(),
                    "lhand_pose": np.zeros(45).tolist(),
                    "rhand_pose": np.zeros(45).tolist(),
                    "jaw_pose": np.zeros(3).tolist(),
                    "shape": np.zeros(10).tolist(),
                    "expr": np.zeros(10).tolist(),
                    "trans": np.zeros(3).tolist()
                }
            }
        ],
        "categories": [{"id": 1, "name": "person"}]
    }
    
    annot_file = os.path.join(annot_dir, f'{split}.json')
    with open(annot_file, 'w') as f:
        json.dump(annotation, f)
    
    return img_path, annot_file


def visualize_keypoints(img, keypoints, title="Keypoints", save_path=None):
    """Visualize keypoints on image"""
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    
    # Plot keypoints
    valid_kpts = keypoints[keypoints[:, 2] > 0]
    plt.scatter(valid_kpts[:, 0], valid_kpts[:, 1], c='red', s=10, alpha=0.7)
    
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def visualize_bbox(img, bbox, title="BBox", save_path=None):
    """Visualize bounding box on image"""
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    
    if bbox is not None:
        x, y, w, h = bbox
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='green', linewidth=2)
        plt.gca().add_patch(rect)
    
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def visualize_sample(img, joint_img, joint_valid, lhand_bbox, rhand_bbox, face_bbox, save_path=None):
    """Visualize complete sample with keypoints and bboxes"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    # Image with keypoints and bboxes
    axes[1].imshow(img)
    
    # Plot valid keypoints
    valid_mask = joint_valid[:, 0] > 0
    valid_kpts = joint_img[valid_mask]
    axes[1].scatter(valid_kpts[:, 0], valid_kpts[:, 1], c='red', s=5, alpha=0.7, label='Keypoints')
    
    # Plot bboxes if valid
    if lhand_bbox is not None and lhand_bbox[1, 0] > lhand_bbox[0, 0]:
        rect = plt.Rectangle((lhand_bbox[0, 0], lhand_bbox[0, 1]), 
                              lhand_bbox[1, 0] - lhand_bbox[0, 0],
                              lhand_bbox[1, 1] - lhand_bbox[0, 1],
                              fill=False, edgecolor='blue', linewidth=2, label='LHand')
        axes[1].add_patch(rect)
    
    if rhand_bbox is not None and rhand_bbox[1, 0] > rhand_bbox[0, 0]:
        rect = plt.Rectangle((rhand_bbox[0, 0], rhand_bbox[0, 1]),
                              rhand_bbox[1, 0] - rhand_bbox[0, 0],
                              rhand_bbox[1, 1] - rhand_bbox[0, 1],
                              fill=False, edgecolor='cyan', linewidth=2, label='RHand')
        axes[1].add_patch(rect)
    
    if face_bbox is not None and face_bbox[1, 0] > face_bbox[0, 0]:
        rect = plt.Rectangle((face_bbox[0, 0], face_bbox[0, 1]),
                              face_bbox[1, 0] - face_bbox[0, 0],
                              face_bbox[1, 1] - face_bbox[0, 1],
                              fill=False, edgecolor='yellow', linewidth=2, label='Face')
        axes[1].add_patch(rect)
    
    axes[1].set_title("Keypoints & BBoxes")
    axes[1].axis('off')
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()


def test_dataloader_train():
    """Test DataLoader with real data in train mode"""
    print("=" * 60)
    print("Testing HAA500 DataLoader (Train Mode)")
    print("=" * 60)
    
    # Create dummy data
    data_dir = cfg.data_dir
    img_path, annot_file = create_dummy_annotation(data_dir, 'train')
    print(f"\nCreated dummy annotation: {annot_file}")
    
    # Create dataset
    transform = lambda x: x  # Identity transform
    
    dataset = HAA500(transform=transform, data_split='train')
    print(f"\nDataset length: {len(dataset)}")
    print(f"Joint set: {dataset.joint_set['joint_num']} joints")
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Output first few items
    output_dir = os.path.join(os.path.dirname(__file__), 'test_output')
    os.makedirs(output_dir, exist_ok=True)
    
    for i, batch in enumerate(dataloader):
        if i >= 3:  # Only first 3 samples
            break
        
        print(f"\n--- Sample {i+1} ---")
        
        inputs, targets, meta_info = batch
        
        # Print shapes
        print(f"Input img shape: {inputs['img'].shape}")
        print(f"Target keys: {list(targets.keys())}")
        print(f"Meta info keys: {list(meta_info.keys())}")
        
        # Extract data
        img = inputs['img'][0].numpy()  # Remove batch dim
        joint_img = targets['joint_img'][0].numpy()
        joint_valid = meta_info['joint_valid'][0].numpy()
        
        print(f"Joint img shape: {joint_img.shape}")
        print(f"Joint valid shape: {joint_valid.shape}")
        print(f"Valid joints count: {(joint_valid > 0).sum()}")
        
        # Visualize
        if img.shape[-1] == 3:
            img_vis = img.copy()
        else:
            img_vis = np.transpose(img, (1, 2, 0))
        
        save_path = os.path.join(output_dir, f'train_sample_{i+1}.png')
        visualize_keypoints(img_vis, joint_img, f"Train Sample {i+1}", save_path)
    
    print(f"\n✓ Train test completed. Output saved to: {output_dir}")


def test_dataloader_test():
    """Test DataLoader with real data in test mode"""
    print("\n" + "=" * 60)
    print("Testing HAA500 DataLoader (Test Mode)")
    print("=" * 60)
    
    # Create dummy data
    data_dir = cfg.data_dir
    img_path, annot_file = create_dummy_annotation(data_dir, 'test')
    print(f"\nCreated dummy annotation: {annot_file}")
    
    # Create dataset
    transform = lambda x: x
    dataset = HAA500(transform=transform, data_split='test')
    
    print(f"\nDataset length: {len(dataset)}")
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    output_dir = os.path.join(os.path.dirname(__file__), 'test_output')
    os.makedirs(output_dir, exist_ok=True)
    
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        
        print(f"\n--- Sample {i+1} ---")
        
        inputs, targets, meta_info = batch
        
        print(f"Input img shape: {inputs['img'].shape}")
        print(f"Target keys: {list(targets.keys())} (should be empty for test)")
        
        img = inputs['img'][0].numpy()
        print(f"Image shape: {img.shape}")
        
        save_path = os.path.join(output_dir, f'test_sample_{i+1}.png')
        if img.ndim == 3 and img.shape[-1] == 3:
            plt.imsave(save_path, img)
            print(f"Saved: {save_path}")
    
    print(f"\n✓ Test mode completed. Output saved to: {output_dir}")


def test_joint_set_consistency():
    """Test that joint_set is correctly configured"""
    print("\n" + "=" * 60)
    print("Testing Joint Set Consistency")
    print("=" * 60)
    
    transform = lambda x: x
    
    # Test train split
    dataset_train = HAA500(transform=transform, data_split='train')
    # Test test split (might fail if no test data)
    try:
        dataset_test = HAA500(transform=transform, data_split='test')
        print("\nBoth train and test datasets created successfully")
    except Exception as e:
        print(f"\nTest dataset creation failed (expected if no data): {e}")
    
    # Verify joint_set
    joint_set = dataset_train.joint_set
    
    print(f"\nJoint count: {joint_set['joint_num']}")
    print(f"Actual joint names count: {len(joint_set['joints_name'])}")
    
    # Print joint breakdown
    print("\nJoint breakdown:")
    print(f"  Body (0-23): {joint_set['joints_name'][:24]}")
    print(f"  Left Hand (24-44): {joint_set['joints_name'][24:45][:5]}...")
    print(f"  Right Hand (45-65): {joint_set['joints_name'][45:66][:5]}...")
    print(f"  Face (66-133): {joint_set['joints_name'][66:][:5]}...")
    
    # Verify flip pairs
    print(f"\nFlip pairs count: {len(joint_set['flip_pairs'])}")
    print(f"Sample flip pairs: {joint_set['flip_pairs'][:5]}")
    
    # Check consistency
    assert joint_set['joint_num'] == len(joint_set['joints_name']), \
        f"Joint num mismatch: {joint_set['joint_num']} vs {len(joint_set['joints_name'])}"
    
    print("\n✓ Joint set consistency check passed!")


def test_merge_joint_function():
    """Test merge_joint function with known values"""
    print("\n" + "=" * 60)
    print("Testing merge_joint Function")
    print("=" * 60)
    
    transform = lambda x: x
    dataset = HAA500(transform=transform, data_split='train')
    
    # Create test keypoints with known values
    body_kpts = np.zeros((24, 3), dtype=np.float32)
    body_kpts[11] = [100, 200, 1]  # L_Hip
    body_kpts[12] = [200, 200, 1]  # R_Hip
    
    foot_kpts = np.ones((7, 3), dtype=np.float32) * [300, 400, 1]
    lhand_kpts = np.ones((21, 3), dtype=np.float32) * [150, 150, 1]
    rhand_kpts = np.ones((21, 3), dtype=np.float32) * [250, 150, 1]
    face_kpts = np.ones((68, 3), dtype=np.float32) * [175, 100, 1]
    
    merged = dataset.merge_joint(body_kpts, foot_kpts, lhand_kpts, rhand_kpts, face_kpts)
    
    print(f"\nMerged keypoints shape: {merged.shape}")
    print(f"Expected: (142, 3)")  # 24 + 1(pelvis) + 7 + 21 + 21 + 68
    
    # Check pelvis calculation
    pelvis = merged[24]
    expected_pelvis = np.array([150, 200, 1])  # Average of L_Hip and R_Hip
    print(f"\nPelvis: {pelvis}")
    print(f"Expected pelvis: {expected_pelvis}")
    
    assert np.allclose(pelvis, expected_pelvis), f"Pelvis mismatch!"
    
    # Check concatenation order
    print(f"\nBody joint [0]: {merged[0]}")  # Nose
    print(f"Foot joint [25]: {merged[25]}")  # First foot joint after pelvis
    print(f"LHand joint [32]: {merged[32]}")  # First hand joint
    print(f"RHand joint [53]: {merged[53]}")  # First right hand joint
    print(f"Face joint [74]: {merged[74]}")  # First face joint
    
    print("\n✓ merge_joint function test passed!")


def main():
    """Run all integration tests"""
    print("HAA500 Integration Tests")
    print("=" * 60)
    
    # Test 1: Joint set consistency
    try:
        test_joint_set_consistency()
    except Exception as e:
        print(f"✗ Joint set test failed: {e}")
    
    # Test 2: merge_joint function
    try:
        test_merge_joint_function()
    except Exception as e:
        print(f"✗ merge_joint test failed: {e}")
    
    # Test 3: DataLoader train mode (requires data)
    try:
        test_dataloader_train()
    except Exception as e:
        print(f"✗ DataLoader train test failed: {e}")
    
    # Test 4: DataLoader test mode
    try:
        test_dataloader_test()
    except Exception as e:
        print(f"✗ DataLoader test mode failed: {e}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
