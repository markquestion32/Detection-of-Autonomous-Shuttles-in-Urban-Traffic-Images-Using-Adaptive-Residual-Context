#!/bin/bash
# setup_combined_dataset.sh - Download COCO train2017 and merge with shuttle data
# 
# WARNING: COCO train2017 is ~18GB to download and ~20GB unzipped
# Total storage needed: ~40GB

set -e

cd /home/mayounes/scratch/project-mtl3/data

echo "============================================"
echo "Setting up Combined COCO + Shuttle Dataset"
echo "============================================"

# 1. Download COCO train2017 images
if [ ! -d "coco_train2017" ]; then
    echo "Downloading COCO train2017 (18GB)..."
    wget http://images.cocodataset.org/zips/train2017.zip
    unzip train2017.zip
    mv train2017 coco_train2017
    rm train2017.zip
else
    echo "COCO train2017 already exists"
fi

# 2. Download COCO train2017 annotations
if [ ! -f "annotations/instances_train2017.json" ]; then
    echo "Downloading COCO annotations..."
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip annotations_trainval2017.zip
    rm annotations_trainval2017.zip
else
    echo "COCO annotations already exist"
fi

# 3. Create combined directory structure
echo "Creating combined dataset structure..."
mkdir -p data_combined/train/images
mkdir -p data_combined/train/labels
mkdir -p data_combined/val/images
mkdir -p data_combined/val/labels
mkdir -p data_combined/test/images
mkdir -p data_combined/test/labels

# 4. Link COCO images
echo "Linking COCO images..."
ln -sf $(pwd)/coco_train2017/* data_combined/train/images/ 2>/dev/null || true

# 5. Link Shuttle images (from data_81class which has class 80 labels)
echo "Linking shuttle images..."
for f in data_81class/train/images/*; do
    ln -sf $(pwd)/$f data_combined/train/images/ 2>/dev/null || true
done

# 6. Copy shuttle labels (class 80)
echo "Copying shuttle labels..."
cp data_81class/train/labels/* data_combined/train/labels/
cp data_81class/val/labels/* data_combined/val/labels/
cp data_81class/test/labels/* data_combined/test/labels/

# 7. Link val/test images
ln -sf $(pwd)/data_81class/val/images/* data_combined/val/images/ 2>/dev/null || true
ln -sf $(pwd)/data_81class/test/images/* data_combined/test/images/ 2>/dev/null || true

echo ""
echo "============================================"
echo "Done! Next step: Convert COCO annotations to YOLO format"
echo "Run: python src/coco_to_yolo.py"
echo "============================================"
