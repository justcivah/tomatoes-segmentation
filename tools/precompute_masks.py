import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from config import (
    DS_PATH as ds_path,
    TARGET_CATEGORY as target_category,
)

def precompute_coco_masks(ann_path: str, output_dir: str, cat_id: int):
    os.makedirs(output_dir, exist_ok=True)
    
    coco = COCO(ann_path)
    img_ids = coco.getImgIds()
    
    print(f"generating masks for {len(img_ids)} images...")
    
    for img_id in img_ids:
        img_info = coco.imgs[img_id]
        h, w = img_info['height'], img_info['width']
        file_name = img_info['file_name']
        
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_id)
        anns = coco.loadAnns(ann_ids)
        
        if len(anns) == 0:
            mask = np.zeros((h, w), dtype=np.uint8)

        else:
            mask = sum(coco.annToMask(ann) for ann in anns)
            mask = np.clip(mask, 0, 1) * 255
            mask = mask.astype(np.uint8)
        
        mask_img = Image.fromarray(mask, mode='L')
        
        # save image
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        save_path = os.path.join(output_dir, f"{base_name}.png")
        
        mask_img.save(save_path)
        
    print(f"all masks saved to {output_dir}")

if __name__ == "__main__":
    ann_path = os.path.join(ds_path, "annotations.json") 
    output_dir = os.path.join(ds_path, "masks")
    
    precompute_coco_masks(ann_path, output_dir, target_category)