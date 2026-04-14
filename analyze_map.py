from PIL import Image
import numpy as np
from collections import Counter
import os

def analyze_colors():
    base_dir = r'c:\Users\anura\OneDrive\Documents\Delhi Implementation\LandUseDataset'
    img_name = 'Screenshot 2026-02-14 235430.png'
    path = os.path.join(base_dir, img_name)
    
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    try:
        img = Image.open(path)
        img = img.resize((50, 50)) # Downscale to grid size
        img = img.convert('RGB')
        
        pixels = list(img.getdata())
        counts = Counter(pixels)
        
        print(f"Top 20 Colors in {img_name}:")
        for color, count in counts.most_common(20):
            print(f"RGB: {color} - Count: {count}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_colors()
