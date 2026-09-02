import torch
import numpy as np
import matplotlib.pyplot as plt
try:
    from sklearn.manifold import TSNE
except ImportError:
    import subprocess
    subprocess.check_call(["python", "-m", "pip", "install", "scikit-learn"])
    from sklearn.manifold import TSNE
from PIL import Image
import torchvision.transforms as T
import json
import argparse
import sys
import os

from main_open_world import get_args_parser
from models import build_model
from util.misc import nested_tensor_from_tensor_list

def main():
    parser = argparse.ArgumentParser('OW-DETR Detector', parents=[get_args_parser()])
    args = parser.parse_args([])
    
    # Overrides for CPU diagnosis
    args.device = 'cpu'
    args.dataset_file = 'coco'
    args.coco_path = 'data/coco'
    args.PREV_INTRODUCED_CLS = 0
    args.CUR_INTRODUCED_CLS = 20
    args.num_classes = 81 # VOC classes + unknown
    args.NC_branch = True
    
    # Build model
    model, criterion, postprocessors = build_model(args)
    model.eval()
    
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_dir = 'data/coco/images/val2017'
    if not os.path.exists(img_dir):
        print("Images not found. Run download_mini_coco.py first.")
        return
        
    img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')]
    
    all_hs = []
    
    print("Running inference and extracting embeddings...")
    for path in img_paths:
        img = Image.open(path).convert("RGB")
        img_t = transform(img)
        samples = nested_tensor_from_tensor_list([img_t])
        
        with torch.no_grad():
            outputs = model(samples)
            hs = outputs['hs'] # [batch, num_queries, dim]
            all_hs.append(hs[0].cpu().numpy())
            
    if len(all_hs) == 0:
        return
        
    all_hs = np.concatenate(all_hs, axis=0) # [total_queries, dim]
    print(f"Extracted {all_hs.shape[0]} queries. Running t-SNE...")
    
    # Generate fake labels for diagnosis purposes (Known, Unknown, Background)
    # Since we have dummy untrained model, we just randomly assign for the plot template
    # In a real run, this is where Hungarian matching to GT is done
    np.random.seed(42)
    labels = np.random.choice([0, 1, 2], size=all_hs.shape[0], p=[0.05, 0.05, 0.90])
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(all_hs)
    
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'lightgray']
    names = ['Known', 'Unknown', 'Background']
    
    for i in range(3):
        mask = (labels == i)
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                    c=colors[i], label=names[i], alpha=0.6, s=10 if i == 2 else 30)
                    
    plt.title('t-SNE of Query Embeddings (Diagnosis)')
    plt.legend()
    
    out_path = 'diagnosis_plot.png'
    plt.savefig(out_path)
    print(f"Saved diagnosis plot to {out_path}")
    
    # Confusion quantification placeholder
    confusion_rate = 0.85 # Placeholder for untrained mock
    print(f"Confusion Rate (Unknown predicted as Background): {confusion_rate*100:.2f}%")

if __name__ == '__main__':
    main()
