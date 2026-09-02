import os
import json
import urllib.request

def download_mini_coco():
    os.makedirs('data/coco/images/train2017', exist_ok=True)
    os.makedirs('data/coco/images/val2017', exist_ok=True)
    os.makedirs('data/coco/annotations', exist_ok=True)
    
    # Just a few COCO image URLs
    urls = [
        ("000000000139.jpg", "http://images.cocodataset.org/val2017/000000000139.jpg"),
        ("000000000285.jpg", "http://images.cocodataset.org/val2017/000000000285.jpg"),
        ("000000000632.jpg", "http://images.cocodataset.org/val2017/000000000632.jpg"),
        ("000000000724.jpg", "http://images.cocodataset.org/val2017/000000000724.jpg"),
        ("000000000776.jpg", "http://images.cocodataset.org/val2017/000000000776.jpg"),
    ]
    
    images = []
    annotations = []
    ann_id = 1
    
    for i, (fname, url) in enumerate(urls):
        try:
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(url, f"data/coco/images/train2017/{fname}")
            urllib.request.urlretrieve(url, f"data/coco/images/val2017/{fname}")
            
            # Add to dummy json
            images.append({
                "id": i + 1,
                "file_name": fname,
                "width": 640,
                "height": 480
            })
            
            # Dummy annotation for known class (e.g. 1 - person)
            annotations.append({
                "id": ann_id,
                "image_id": i + 1,
                "category_id": 1,
                "bbox": [100, 100, 200, 200],
                "area": 40000,
                "iscrowd": 0
            })
            ann_id += 1
            
            # Dummy annotation for unknown class (e.g. 81 - unknown in OWOD task 1)
            annotations.append({
                "id": ann_id,
                "image_id": i + 1,
                "category_id": 81,
                "bbox": [300, 300, 100, 100],
                "area": 10000,
                "iscrowd": 0
            })
            ann_id += 1
            
        except Exception as e:
            print(f"Failed to download {fname}: {e}")
            
    categories = [
        {"id": 1, "name": "person", "supercategory": "person"},
        {"id": 81, "name": "unknown", "supercategory": "unknown"}
    ]
    
    coco_json = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    with open('data/coco/annotations/instances_train2017.json', 'w') as f:
        json.dump(coco_json, f)
    with open('data/coco/annotations/instances_val2017.json', 'w') as f:
        json.dump(coco_json, f)
        
    print("Mini COCO dataset created.")

if __name__ == '__main__':
    download_mini_coco()
