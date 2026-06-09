import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def analyze_dataset(labels_dir):
    print(f"Analyzing labels in: {labels_dir}")
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    if not label_files:
        print("No label files found!")
        return

    all_widths = []
    all_heights = []
    all_centers_x = []
    all_centers_y = []
    instances_per_image = []
    
    for fpath in label_files:
        with open(fpath, 'r') as f:
            lines = f.readlines()
            
        instances_per_image.append(len(lines))
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # cls, x, y, w, h
                x, y, w, h = map(float, parts[1:5])
                all_centers_x.append(x)
                all_centers_y.append(y)
                all_widths.append(w)
                all_heights.append(h)

    print(f"Total Images: {len(label_files)}")
    print(f"Total Instances: {len(all_widths)}")

    # Setup Plotting
    plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Dataset Characteristics (Autonomous Shuttle)', fontsize=16)

    # 1. Distribution of Instances per Image
    counts = Counter(instances_per_image)
    max_count = max(counts.keys()) if counts else 0
    x_vals = range(max_count + 2)
    y_vals = [counts.get(i, 0) for i in x_vals]
    
    axes[0, 0].bar(x_vals, y_vals, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Instances per Image Distribution')
    axes[0, 0].set_xlabel('Number of Shuttles')
    axes[0, 0].set_ylabel('Image Count')
    axes[0, 0].set_xticks(x_vals)

    # 2. Bounding Box Size Distribution (Area)
    areas = np.array(all_widths) * np.array(all_heights)
    axes[0, 1].hist(areas, bins=50, color='salmon', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('BBox Size (Normalized Area) Distribution')
    axes[0, 1].set_xlabel('Normalized Area (W * H)')
    axes[0, 1].set_ylabel('Frequency')

    # 3. Normalized Width vs Height Scatter
    axes[0, 2].scatter(all_widths, all_heights, alpha=0.3, s=10, c='purple')
    axes[0, 2].set_title('BBox Width vs Height (Normalized)')
    axes[0, 2].set_xlabel('Width')
    axes[0, 2].set_ylabel('Height')
    axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.3) # Diagonal reference
    axes[0, 2].set_xlim(0, 1)
    axes[0, 2].set_ylim(0, 1)

    # 4. Spatial Distribution (Heatmap style scatter)
    # Using hexbin for density
    hb = axes[1, 0].hexbin(all_centers_x, all_centers_y, gridsize=30, cmap='inferno', mincnt=1)
    axes[1, 0].set_title('Spatial Distribution of Centers')
    axes[1, 0].set_xlabel('Normalized Center X')
    axes[1, 0].set_ylabel('Normalized Center Y')
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(1, 0) # Invert Y to match image coordinates (top-left is 0,0 usually)
    cb = fig.colorbar(hb, ax=axes[1, 0])
    cb.set_label('Count')

    # 5. Width Distribution
    axes[1, 1].hist(all_widths, bins=50, color='green', edgecolor='black', alpha=0.6)
    axes[1, 1].set_title('BBox Normalized Width Distribution')
    axes[1, 1].set_xlabel('Width')

    # 6. Height Distribution
    axes[1, 2].hist(all_heights, bins=50, color='orange', edgecolor='black', alpha=0.6)
    axes[1, 2].set_title('BBox Normalized Height Distribution')
    axes[1, 2].set_xlabel('Height')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    output_path = 'dataset_analysis.pdf'
    plt.savefig(output_path, dpi=300)
    print(f"Saved analysis plots to {output_path}")

if __name__ == "__main__":
    labels_path = "/scratch/mayounes/project-mtl/data/train/labels"
    analyze_dataset(labels_path)
