import numpy as np
import math
import os
from PIL import Image

class DataGenerator:
    def __init__(self, grid_shape=(50, 50)):
        self.shape = grid_shape
        self.rows, self.cols = grid_shape
        self.base_dir = r'C:\Users\anura\OneDrive\Documents\Delhi Implementation\LandUseDataset' 
        self.img_name = 'Screenshot 2026-02-14 235430.png'

    def generate_traffic(self):
        """
        Generate Traffic Demand based on ROAD NETWORKS (Ring Roads/Arterials)
        and MARKET HUBS.
        """
        traffic = np.zeros(self.shape)
        
        y, x = np.meshgrid(np.arange(self.rows), np.arange(self.cols), indexing='ij')
        
        # 1. Ring Road (Ellipse R~15)
        # Center (25, 25)
        # Dist from center
        dist = np.sqrt((x-25)**2 + (y-25)**2)
        mask_ring = (dist > 12) & (dist < 18)
        traffic[mask_ring] += 0.5
        
        # 2. Outer Ring Road (Ellipse R~25)
        mask_outer = (dist > 22) & (dist < 28)
        traffic[mask_outer] += 0.4
        
        # 3. Arterials (Cross Lines)
        # Horizontal
        traffic[25:27, :] += 0.3
        # Vertical
        traffic[:, 25:27] += 0.3
        
        # 4. Market Hubs (Specific Hotspots)
        markets = [
            (25, 25, 1.0, 10), # CP
            (35, 35, 0.9, 15), # Nehru Place
            (15, 15, 0.8, 15), # Rohini
            (38, 10, 0.8, 15), # Janakpuri
            (25, 40, 0.7, 15), # Laxmi Nagar
            (40, 20, 0.8, 20), # Airport Route
        ]
        
        for r, c, intensity, spread in markets:
            traffic += intensity * np.exp(-((x-c)**2 + (y-r)**2) / spread)
            
        traffic += np.random.normal(0, 0.05, self.shape)
        traffic = np.clip(traffic, 0.0, 1.0)
        return traffic

    def generate_land_use(self):
        """
        Generate Land Use from Image Overlay.
        0: Green Belt (Green)
        1: Water (Blue)
        2: Res (Yellow)
        3: Comm (Red)
        4: District (Dark Red) -> Mapped to Red
        5: Ind (Purple)
        6: Neutral (White/Grey)
        """
        land_use = np.full(self.shape, 6, dtype=int) # Default Neutral
        
        path = os.path.join(self.base_dir, self.img_name)
        if not os.path.exists(path):
            print(f"Warning: Image {path} not found. Using Fallback.")
            # Fallback procedural
            return self._fallback_procedural()
            
        try:
            img = Image.open(path).convert('RGB')
            # Resize to grid
            img = img.resize((self.cols, self.rows), Image.Resampling.NEAREST) # Correct (width, height) order
            pixels = np.array(img)
            
            # Map Pixels to Codes
            # r, g, b = pixels[y, x]
            # Vectorized approach or Loop
            for r in range(self.rows):
                for c in range(self.cols):
                    red, green, blue = pixels[r, c]
                    
                    # Logic Table
                    # Yellow (Res): High R, High G, Low B
                    if red > 180 and green > 180 and blue < 150:
                        land_use[r, c] = 2
                    
                    # Red (Comm): High R, Low G, Low B
                    elif red > 180 and green < 150 and blue < 150:
                        land_use[r, c] = 3
                        
                    # Green (Belt): Low R, High G, Low B
                    elif red < 150 and green > 150 and blue < 150:
                        land_use[r, c] = 0
                        
                    # Blue (Water): Low R, Low G, High B
                    elif red < 150 and green < 180 and blue > 180:
                        land_use[r, c] = 1
                        
                    # Purple (Ind): High R, Low G, High B
                    elif red > 120 and green < 120 and blue > 120:
                        land_use[r, c] = 5
                        
                    # Dark Grey/Lines (Roads/Infra?) -> Maybe treat as Neutral or Road
                    
            print("Successfully loaded Land Use from Image.")
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return self._fallback_procedural()
            
        return land_use
        
    def _fallback_procedural(self):
        # ... (Old procedural logic as backup) ...
        land_use = np.full(self.shape, 2, dtype=int)
        # Just return basic Res
        return land_use
