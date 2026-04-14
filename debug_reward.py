from reward_calculator import RewardCalculator

def test():
    calc = RewardCalculator()
    
    # Test Central Delhi (Connaught Place)
    lat, lon = 28.6139, 77.2090
    print(f"\n--- Testing Coordinate: {lat}, {lon} ---")
    
    # 1. Check Bounds
    print(f"Bounds: Min({calc.minx}, {calc.miny}) - Max({calc.maxx}, {calc.maxy})")
    inside_bbox = (calc.minx <= lon <= calc.maxx) and (calc.miny <= lat <= calc.maxy)
    print(f"Inside Bounding Box? {inside_bbox}")
    
    # 2. Check Grid Mapping
    row, col = calc.geo_to_grid(lat, lon)
    print(f"Mapped to Grid: Row={row}, Col={col}")
    print(f"Grid Shape: {calc.grid_shape}")
    
    # 3. Check Traffic Data
    try:
        val = calc.traffic[row, col]
        print(f"Traffic Value at ({row},{col}): {val}")
    except Exception as e:
        print(f"Traffic Lookup Failed: {e}")

    # 4. Full Assessment
    res = calc.assess_location(lat, lon)
    print("\n--- Assessment Result ---")
    for k, v in res.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    test()
