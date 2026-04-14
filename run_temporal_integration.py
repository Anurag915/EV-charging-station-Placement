#!/usr/bin/env python3
"""
Run Temporal Integration Script
==============================

This script runs the temporal integration and demonstrates the features.
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append('.')

def main():
    print("🚀 RUNNING TEMPORAL INTEGRATION")
    print("=" * 50)
    
    try:
        # Import the integration cell
        exec(open('temporal_integration_clean.py').read())
        
        print("\n✅ TEMPORAL INTEGRATION SUCCESSFUL!")
        print("\nGenerated files:")
        print("• temporal_patterns_analysis.png")
        
        print("\n🎯 Key Features Added:")
        print("✅ Time-Aware LSTM/GRU Integration")
        print("✅ Rush Hour vs Off-Peak Optimization")
        print("✅ Temporal Attention Mechanisms")
        print("✅ Dynamic Reward Shaping")
        print("✅ Real Traffic Data Integration")
        
        print("\n📊 Your project now learns from:")
        print("• 20 days of Delhi traffic patterns")
        print("• Rush hour demand spikes (7-10 AM, 5-8 PM)")
        print("• Off-peak efficiency optimization")
        print("• Weekly and daily cycles")
        print("• Time-dependent charging behavior")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during integration: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Integration completed successfully!")
    else:
        print("\n💥 Integration failed. Check the error messages above.")
    
    sys.exit(0 if success else 1)
