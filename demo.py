"""
Quick Demo Script for TCC Detection System
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime

from main import TCCPipeline


def run_quick_demo():
    """
    Run a quick demonstration of the TCC detection system
    """
    print("🛰️  Welcome to the TCC Detection System Demo!")
    print("=" * 50)
    
    # Initialize the pipeline
    pipeline = TCCPipeline(output_directory="demo_output")
    
    # Run demonstration with synthetic data
    print("\n🔍 Running TCC detection and tracking...")
    results = pipeline.run_detection_demo(
        use_synthetic=True, 
        num_time_steps=8  # Process 8 time steps (4 hours of data)
    )
    
    # Display results summary
    print("\n📊 RESULTS SUMMARY")
    print("=" * 30)
    
    metadata = results['metadata']
    print(f"✅ Tracks detected: {metadata['num_tracks']}")
    print(f"✅ TCCs identified: {metadata['num_tccs']}")
    print(f"✅ Time steps processed: {metadata['num_time_steps']}")
    print(f"✅ Time range: {metadata['time_range']['start'][:19]} to {metadata['time_range']['end'][:19]}")
    
    # Show track summary statistics
    if not results['track_summary'].empty:
        track_summary = results['track_summary']
        print(f"\n📈 TRACK STATISTICS")
        print(f"Average duration: {track_summary['duration_hours'].mean():.1f} hours")
        print(f"Average distance: {track_summary['total_distance_km'].mean():.1f} km")
        print(f"Average speed: {track_summary['avg_speed_kmh'].mean():.1f} km/h")
        print(f"Longest track: {track_summary['duration_hours'].max():.1f} hours")
        print(f"Fastest track: {track_summary['avg_speed_kmh'].max():.1f} km/h")
    
    # Show required TCC features that are extracted
    print(f"\n🎯 EXTRACTED TCC FEATURES")
    print("The system extracts all required features:")
    for i, feature in enumerate(metadata['required_features'], 1):
        print(f"  {i:2d}. {feature.replace('_', ' ').title()}")
    
    print(f"\n💾 All results and visualizations saved to: demo_output/")
    print("\nFiles created:")
    for file_type, file_path in results['files'].items():
        print(f"  📄 {file_type}: {file_path}")
    
    print("\n🎉 Demo completed successfully!")
    print("\nNext steps:")
    print("1. Check the generated visualizations in demo_output/")
    print("2. Review the CSV files for detailed TCC data")
    print("3. Adapt the code for your real INSAT-3D data")
    
    return results


if __name__ == "__main__":
    run_quick_demo() 