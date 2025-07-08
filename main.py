"""
Main pipeline for Tropical Cloud Cluster (TCC) Detection and Tracking
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import argparse
import json
import os

from src.data_loader import IRBRTDataLoader, SyntheticDataGenerator
from src.tcc_detector import TCCDetector
from src.tcc_tracker import TCCTracker
from src.visualization import TCCVisualizer
from src.config import REQUIRED_FEATURES


class TCCPipeline:
    """
    Main pipeline for TCC detection and tracking
    """
    
    def __init__(self, data_directory: str = None, output_directory: str = "output"):
        """
        Initialize the TCC pipeline
        
        Args:
            data_directory: Directory containing INSAT-3D data files
            output_directory: Directory for saving results
        """
        self.data_directory = data_directory
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        # Initialize components
        if data_directory:
            self.data_loader = IRBRTDataLoader(data_directory)
        self.detector = TCCDetector()
        self.tracker = TCCTracker()
        self.visualizer = TCCVisualizer()
        
        print("TCC Detection Pipeline initialized")
        print(f"Output directory: {self.output_directory}")
    
    def run_detection_demo(self, use_synthetic: bool = True, 
                          num_time_steps: int = 10) -> Dict:
        """
        Run a demonstration of TCC detection and tracking
        
        Args:
            use_synthetic: Whether to use synthetic data for demonstration
            num_time_steps: Number of time steps to process
            
        Returns:
            Dictionary containing results
        """
        print("\n=== TCC Detection and Tracking Demo ===")
        
        if use_synthetic:
            print("Generating synthetic INSAT-3D data...")
            time_series_data = self._generate_synthetic_time_series(num_time_steps)
        else:
            if not self.data_directory:
                raise ValueError("Data directory must be provided for real data processing")
            print("Loading real INSAT-3D data...")
            time_series_data = self._load_real_time_series()
        
        print(f"Processing {len(time_series_data)} time steps...")
        
        # Run TCC detection and tracking
        tracks = self.tracker.track_tccs(time_series_data)
        
        # Generate track summary
        track_summary = self.tracker.get_track_summary(tracks)
        
        # Create visualizations
        self._create_visualizations(time_series_data, tracks, track_summary)
        
        # Save results
        results = self._save_results(tracks, track_summary, time_series_data)
        
        print(f"\n=== Results Summary ===")
        print(f"Total tracks detected: {len(tracks)}")
        print(f"Total time steps processed: {len(time_series_data)}")
        print(f"Results saved to: {self.output_directory}")
        
        return results
    
    def _generate_synthetic_time_series(self, num_time_steps: int) -> List[Dict]:
        """
        Generate synthetic time series data for demonstration
        
        Args:
            num_time_steps: Number of time steps to generate
            
        Returns:
            List of data dictionaries
        """
        time_series_data = []
        start_time = datetime(2024, 1, 1, 0, 0)
        
        for i in range(num_time_steps):
            # Generate data with evolving cloud clusters
            num_clusters = max(1, 3 + int(np.sin(i * 0.5)))  # Varying number of clusters
            irbt_data, lats, lons = SyntheticDataGenerator.generate_sample_data(
                nlat=200, nlon=200, num_clusters=num_clusters
            )
            
            timestamp = start_time + timedelta(minutes=30 * i)  # Half-hourly data
            
            time_series_data.append({
                'irbt_data': irbt_data,
                'lats': lats,
                'lons': lons,
                'timestamp': pd.Timestamp(timestamp),
                'metadata': {
                    'source': 'synthetic',
                    'step': i
                }
            })
        
        return time_series_data
    
    def _load_real_time_series(self) -> List[Dict]:
        """
        Load real INSAT-3D time series data
        
        Returns:
            List of data dictionaries
        """
        # This would be implemented based on actual data availability
        print("Note: Real data loading would be implemented based on actual INSAT-3D data format")
        
        # For now, return synthetic data as placeholder
        return self._generate_synthetic_time_series(5)
    
    def _create_visualizations(self, time_series_data: List[Dict], 
                             tracks: Dict[int, List[Dict]], 
                             track_summary: pd.DataFrame):
        """
        Create and save visualizations
        
        Args:
            time_series_data: Time series data
            tracks: TCC tracks
            track_summary: Track summary statistics
        """
        print("Creating visualizations...")
        
        # Plot sample IRBT data
        sample_data = time_series_data[0]
        fig1 = self.visualizer.plot_irbt_data(
            sample_data['irbt_data'], 
            sample_data['lats'], 
            sample_data['lons'],
            f"Sample INSAT-3D IRBT Data - {sample_data['timestamp']}"
        )
        fig1.savefig(self.output_directory / "sample_irbt_data.png", dpi=150, bbox_inches='tight')
        plt.close(fig1)
        
        # Plot TCC detection for first time step
        first_step_tccs = []
        if tracks:
            # Get TCCs from first time step
            for track_tccs in tracks.values():
                if track_tccs:
                    first_step_tccs.extend([tcc for tcc in track_tccs if tcc['track_position'] == 1])
        
        if first_step_tccs:
            fig2 = self.visualizer.plot_tcc_detection(
                sample_data['irbt_data'],
                first_step_tccs,
                sample_data['lats'],
                sample_data['lons'],
                f"TCC Detection - {sample_data['timestamp']}"
            )
            fig2.savefig(self.output_directory / "tcc_detection_sample.png", dpi=150, bbox_inches='tight')
            plt.close(fig2)
        
        # Plot track paths
        if tracks:
            fig3 = self.visualizer.plot_track_paths(tracks)
            fig3.savefig(self.output_directory / "tcc_track_paths.png", dpi=150, bbox_inches='tight')
            plt.close(fig3)
        
        # Plot TCC statistics
        all_tccs = []
        for track_tccs in tracks.values():
            all_tccs.extend(track_tccs)
        
        if all_tccs:
            fig4 = self.visualizer.plot_tcc_statistics(all_tccs)
            fig4.savefig(self.output_directory / "tcc_statistics.png", dpi=150, bbox_inches='tight')
            plt.close(fig4)
        
        # Create summary report
        if not track_summary.empty:
            fig5 = self.visualizer.create_summary_report(tracks, track_summary)
            fig5.savefig(self.output_directory / "summary_report.png", dpi=150, bbox_inches='tight')
            plt.close(fig5)
        
        print(f"Visualizations saved to {self.output_directory}")
    
    def _save_results(self, tracks: Dict[int, List[Dict]], 
                     track_summary: pd.DataFrame, 
                     time_series_data: List[Dict]) -> Dict:
        """
        Save results to files
        
        Args:
            tracks: TCC tracks
            track_summary: Track summary
            time_series_data: Original time series data
            
        Returns:
            Dictionary containing file paths and summary
        """
        print("Saving results...")
        
        # Save track summary as CSV
        track_summary_file = self.output_directory / "track_summary.csv"
        track_summary.to_csv(track_summary_file, index=False)
        
        # Save detailed TCC data
        all_tccs = []
        for track_id, track_tccs in tracks.items():
            for tcc in track_tccs:
                # Remove complex objects for JSON serialization
                tcc_clean = {k: v for k, v in tcc.items() 
                           if k != 'cluster_mask' and not k.startswith('track_')}
                tcc_clean['track_id'] = track_id
                all_tccs.append(tcc_clean)
        
        tcc_data_df = pd.DataFrame(all_tccs)
        tcc_data_file = self.output_directory / "tcc_detections.csv"
        tcc_data_df.to_csv(tcc_data_file, index=False)
        
        # Save configuration and metadata
        metadata = {
            'processing_time': datetime.now().isoformat(),
            'num_time_steps': len(time_series_data),
            'num_tracks': len(tracks),
            'num_tccs': len(all_tccs),
            'time_range': {
                'start': time_series_data[0]['timestamp'].isoformat() if time_series_data else None,
                'end': time_series_data[-1]['timestamp'].isoformat() if time_series_data else None
            },
            'required_features': REQUIRED_FEATURES
        }
        
        metadata_file = self.output_directory / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        results = {
            'tracks': tracks,
            'track_summary': track_summary,
            'files': {
                'track_summary': str(track_summary_file),
                'tcc_detections': str(tcc_data_file),
                'metadata': str(metadata_file)
            },
            'metadata': metadata
        }
        
        print(f"Results saved:")
        print(f"  - Track summary: {track_summary_file}")
        print(f"  - TCC detections: {tcc_data_file}")
        print(f"  - Metadata: {metadata_file}")
        
        return results
    
    def process_real_data(self, start_time: datetime, end_time: datetime) -> Dict:
        """
        Process real INSAT-3D data for a specified time range
        
        Args:
            start_time: Start time for processing
            end_time: End time for processing
            
        Returns:
            Results dictionary
        """
        if not self.data_directory:
            raise ValueError("Data directory must be provided for real data processing")
        
        print(f"Processing real data from {start_time} to {end_time}")
        
        # Load data for specified time range
        time_series_data = self.data_loader.load_time_series(start_time, end_time)
        
        if not time_series_data:
            raise ValueError("No data found for specified time range")
        
        # Run detection and tracking
        tracks = self.tracker.track_tccs(time_series_data)
        track_summary = self.tracker.get_track_summary(tracks)
        
        # Create visualizations and save results
        self._create_visualizations(time_series_data, tracks, track_summary)
        results = self._save_results(tracks, track_summary, time_series_data)
        
        return results


def main():
    """
    Main function for command-line interface
    """
    parser = argparse.ArgumentParser(description='TCC Detection and Tracking Pipeline')
    parser.add_argument('--data-dir', type=str, help='Directory containing INSAT-3D data files')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
    parser.add_argument('--demo', action='store_true', help='Run demonstration with synthetic data')
    parser.add_argument('--time-steps', type=int, default=10, help='Number of time steps for demo')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TCCPipeline(data_directory=args.data_dir, output_directory=args.output_dir)
    
    if args.demo:
        # Run demonstration
        results = pipeline.run_detection_demo(use_synthetic=True, num_time_steps=args.time_steps)
    else:
        # Process real data (would need time range specification)
        print("Real data processing requires start and end times.")
        print("Use --demo flag to run demonstration with synthetic data.")
        return
    
    print("\nProcessing completed successfully!")


if __name__ == "__main__":
    main() 