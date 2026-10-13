"""
YOLOv12 Inference Script for Multi-Class Tomato Leaf Disease Detection
MCE 531 - Machine Vision Project (Group 8)
Federal University of Technology, Minna.
"""

import os
import sys
import tempfile
import urllib.request
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ==========================================
# CONFIGURATION SETUP
# ==========================================
MODEL_TYPE = "yolov12"
WEIGHTS_PATH = 'weights/best.pt'  # Change to your local path or weight URL
SOURCE = 'test_images/sample_leaf.jpg'  # Can be an image path, folder, or URL
SAVE_DIR = "results"
CONF_THRESHOLD = 0.25
IMGSZ = 640

def setup_environment():
    print('  Configuration Loaded:')
    print(f'   MODEL_TYPE    : {MODEL_TYPE}')
    print(f'   WEIGHTS_PATH  : {WEIGHTS_PATH}')
    print(f'   SOURCE        : {SOURCE}')
    print(f'   SAVE_DIR      : {SAVE_DIR}')
    print(f'   CONF          : {CONF_THRESHOLD}')
    print(f'   IMGSZ         : {IMGSZ}\n')

def load_model(weights_path):
    # If weights are hosted online, download them to a temporary file
    if weights_path.startswith('http://') or weights_path.startswith('https://'):
        print(f'[INFO] Downloading weights from URL ...')
        tmp = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
        tmp.close()
        try:
            urllib.request.urlretrieve(weights_path, tmp.name)
            weights_path = tmp.name
            print(f'[INFO] Saved temporary weights to: {weights_path}')
        except Exception as e:
            print(f'[-] Download failed: {e}')
            raise

    if not os.path.exists(weights_path):
        print(f'[-] Weights file not found at: "{weights_path}"')
        print('    -> Please check your configuration path.')
        raise FileNotFoundError(weights_path)

    model = YOLO(weights_path)
    print(f' {MODEL_TYPE.upper()} model loaded successfully!')
    print(f'   Classes ({len(model.names)}): {list(model.names.values())}\n')
    return model

def resolve_source(source):
    SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    if source.startswith('http://') or source.startswith('https://'):
        ext = Path(source).suffix.lower() or '.jpg'
        tmp_img = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp_img.close()
        print(f'[INFO] Downloading test image from URL ...')
        urllib.request.urlretrieve(source, tmp_img.name)
        resolved = tmp_img.name
        print(f' -> Image temporary saved to: {resolved}')
        
    elif os.path.isdir(source):
        images = [f for f in Path(source).iterdir() if f.suffix.lower() in SUPPORTED_EXTS]
        if not images:
            raise FileNotFoundError(f'No supported images found in directory: {source}')
        print(f'[INFO] Folder source detected: {len(images)} image(s) found.')
        resolved = source
        
    elif os.path.isfile(source):
        print(f'[INFO] Single local image source detected: {source}')
        resolved = source
    else:
        raise FileNotFoundError(f'Source not found: "{source}". Please check your configuration path.')
        
    print(f'   Resolved source path -> {resolved}\n')
    return resolved

def run_inference():
    setup_environment()
    
    # Load Model and Resolve Source
    model = load_model(WEIGHTS_PATH)
    resolved_source = resolve_source(SOURCE)
    
    # Ensure save directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f'[INFO] Running {MODEL_TYPE.upper()} inference ...\n')
    
    # Predict
    results = model.predict(
        source=resolved_source,
        imgsz=IMGSZ,
        conf=CONF_THRESHOLD,
        save=True,  # Automatically saves annotated configurations
        project=SAVE_DIR,
        name='predict',
        exist_ok=True,
        verbose=False,
    )
    
    print(f'\n Inference complete - {len(results)} image(s) processed.')
    print(f'   Annotated outputs saved to: {os.path.join(SAVE_DIR, "predict")}\n')
    
    # Plot results inline using matplotlib
    for result in results:
        annotated = result.plot()  # returns BGR numpy array
        rgb = annotated[:, :, ::-1]  # Convert BGR -> RGB for Matplotlib
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(rgb)
        ax.axis('off')
        ax.set_title(Path(result.path).name, fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_inference()
