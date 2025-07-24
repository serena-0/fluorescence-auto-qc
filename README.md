# Fluorescence Frame Auto-Centering & Quality Control (QC)

This script processes fluorescence microscopy image sequences by detecting circular targets, evaluating quality metrics, and cropping the image around the target region. Useful for automated data cleaning in high-throughput imaging.

## ✨ Features
- Auto-detect maximum inscribed circles from masks
- RANSAC-based edge fitting
- Multi-metric QC: edge coverage, circularity error, contrast
- Optional interactive review
- Center crop & mask the target region
- Save filtered images and locations for further analysis

## 📂 Input
- Multi-frame OME-TIFF image files

## 💾 Output
- `new_img1024org.npy`: saved cropped 1024×1024 images
- `new_location1024org.npy`: saved center x, y (normalized), and radius

## 🛠 Requirements
See [`requirements.txt`](./requirements.txt)

```bash
pip install -r requirements.txt
