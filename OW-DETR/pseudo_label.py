import cv2
import numpy as np
import os
import json
from util.box_ops import box_iou
import torch

def generate_proposals(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate to connect edges
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    h_img, w_img = img.shape[:2]
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter too small or too large boxes
        area = w * h
        if area < 500 or area > 0.8 * h_img * w_img:
            continue
            
        # [x, y, x2, y2]
        boxes.append([x, y, x + w, y + h])
        
    return boxes

def main():
    # Load COCO annotations
    ann_path = 'data/coco/annotations/instances_train2017.json'
    if not os.path.exists(ann_path):
        print("Run download_mini_coco.py first.")
        return
        
    with open(ann_path, 'r') as f:
        data = json.load(f)
        
    img_dir = 'data/coco/images/train2017'
    
    # Map image ID to known GT boxes (category_id != 81)
    img_to_gt = {}
    for ann in data['annotations']:
        if ann['category_id'] == 81: # unknown
            continue
        img_id = ann['image_id']
        x, y, w, h = ann['bbox']
        box = [x, y, x + w, y + h]
        
        if img_id not in img_to_gt:
            img_to_gt[img_id] = []
        img_to_gt[img_id].append(box)
        
    all_pseudo_labels = {}
    
    # Generate and filter proposals
    for img_info in data['images']:
        img_id = img_info['id']
        img_path = os.path.join(img_dir, img_info['file_name'])
        
        proposals = generate_proposals(img_path)
        if len(proposals) == 0:
            continue
            
        proposals_t = torch.tensor(proposals, dtype=torch.float32)
        
        if img_id in img_to_gt:
            gt_t = torch.tensor(img_to_gt[img_id], dtype=torch.float32)
            # Calculate IoU between proposals and GT
            iou, _ = box_iou(proposals_t, gt_t) # [N, M]
            
            # Max IoU with any known GT
            max_iou, _ = iou.max(dim=1)
            
            # Keep proposals with IoU < 0.3 (not overlapping with known objects)
            keep_idx = max_iou < 0.3
            pseudo_boxes = proposals_t[keep_idx].tolist()
        else:
            pseudo_boxes = proposals_t.tolist()
            
        all_pseudo_labels[img_id] = pseudo_boxes
        
        # Save visualization for sanity check
        img = cv2.imread(img_path)
        for box in pseudo_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        cv2.imwrite(f"pseudo_{img_info['file_name']}", img)
        
    with open('pseudo_labels.json', 'w') as f:
        json.dump(all_pseudo_labels, f)
        
    print(f"Generated pseudo-labels for {len(all_pseudo_labels)} images.")
    print("Saved sample visualizations (e.g. pseudo_*.jpg).")

if __name__ == '__main__':
    main()
