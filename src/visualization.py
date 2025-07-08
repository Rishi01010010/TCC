"""
Visualization utilities for TCC detection and tracking results
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web servers
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

from .config import TEMP_COLORMAP, DEFAULT_FIGURE_SIZE, INDIAN_OCEAN_BOUNDS


class TCCVisualizer:
    """
    Class for creating various visualizations of TCC detection and tracking results
    """
    
    def __init__(self, figsize: Tuple[int, int] = DEFAULT_FIGURE_SIZE):
        """
        Initialize the visualizer
        
        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize
        
    def plot_irbt_data(self, irbt_data: np.ndarray, lats: np.ndarray, 
                      lons: np.ndarray, title: str = "INSAT-3D IRBT Data") -> plt.Figure:
        """
        Plot IRBT temperature data
        
        Args:
            irbt_data: 2D array of brightness temperatures
            lats: Latitude array
            lons: Longitude array
            title: Plot title
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize, 
                              subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Create temperature plot
        im = ax.contourf(lons, lats, irbt_data, levels=20, 
                        cmap=TEMP_COLORMAP, transform=ccrs.PlateCarree())
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.LAND, alpha=0.3)
        ax.add_feature(cfeature.OCEAN, alpha=0.3)
        
        # Set extent to Indian Ocean
        ax.set_extent([INDIAN_OCEAN_BOUNDS['lon_min'], INDIAN_OCEAN_BOUNDS['lon_max'],
                      INDIAN_OCEAN_BOUNDS['lat_min'], INDIAN_OCEAN_BOUNDS['lat_max']])
        
        # Add gridlines
        ax.gridlines(draw_labels=True, alpha=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Brightness Temperature (K)')
        
        ax.set_title(title)
        plt.tight_layout()
        
        return fig
    
    def plot_tcc_detection(self, irbt_data: np.ndarray, tccs: List[Dict],
                          lats: np.ndarray, lons: np.ndarray, 
                          title: str = "TCC Detection Results") -> plt.Figure:
        """
        Plot TCC detection results overlaid on IRBT data
        
        Args:
            irbt_data: Original IRBT data
            tccs: List of detected TCCs
            lats: Latitude array
            lons: Longitude array
            title: Plot title
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize,
                              subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Plot base IRBT data
        im = ax.contourf(lons, lats, irbt_data, levels=20, 
                        cmap=TEMP_COLORMAP, alpha=0.7, transform=ccrs.PlateCarree())
        
        # Overlay TCC detections
        colors = plt.cm.Set1(np.linspace(0, 1, len(tccs)))
        
        for i, tcc in enumerate(tccs):
            # Plot TCC center
            ax.plot(tcc['convective_lon'], tcc['convective_lat'], 
                   'o', color=colors[i], markersize=10, 
                   markeredgecolor='black', markeredgewidth=2,
                   transform=ccrs.PlateCarree(),
                   label=f"TCC {tcc.get('tcc_id', i+1)}")
            
            # Plot TCC boundary (approximate circle)
            circle = patches.Circle((tcc['convective_lon'], tcc['convective_lat']),
                                  tcc['max_radius_km'] / 111.0,  # Convert to degrees
                                  fill=False, color=colors[i], linewidth=2,
                                  transform=ccrs.PlateCarree())
            ax.add_patch(circle)
            
            # Add TCC information text
            info_text = (f"ID: {tcc.get('tcc_id', i+1)}\n"
                        f"Min T: {tcc['min_tb']:.1f}K\n"
                        f"Size: {tcc['pixel_count']} px")
            
            ax.text(tcc['convective_lon'], tcc['convective_lat'] + 1,
                   info_text, transform=ccrs.PlateCarree(),
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=colors[i], alpha=0.7),
                   fontsize=8, ha='center')
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.LAND, alpha=0.3)
        
        # Set extent
        ax.set_extent([INDIAN_OCEAN_BOUNDS['lon_min'], INDIAN_OCEAN_BOUNDS['lon_max'],
                      INDIAN_OCEAN_BOUNDS['lat_min'], INDIAN_OCEAN_BOUNDS['lat_max']])
        
        ax.gridlines(draw_labels=True, alpha=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Brightness Temperature (K)')
        
        ax.set_title(f"{title} - {len(tccs)} TCCs detected")
        if len(tccs) <= 10:  # Only show legend if not too many TCCs
            ax.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def plot_track_paths(self, tracks: Dict[int, List[Dict]], 
                        title: str = "TCC Track Paths") -> plt.Figure:
        """
        Plot TCC tracking paths over time
        
        Args:
            tracks: Dictionary of tracks
            title: Plot title
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize,
                              subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Generate colors for tracks
        colors = plt.cm.tab20(np.linspace(0, 1, len(tracks)))
        
        for i, (track_id, track_tccs) in enumerate(tracks.items()):
            if len(track_tccs) < 2:
                continue
            
            # Extract track coordinates
            lats = [tcc['convective_lat'] for tcc in track_tccs]
            lons = [tcc['convective_lon'] for tcc in track_tccs]
            
            # Plot track path
            ax.plot(lons, lats, 'o-', color=colors[i], 
                   linewidth=2, markersize=6, alpha=0.8,
                   transform=ccrs.PlateCarree(),
                   label=f'Track {track_id} ({len(track_tccs)} obs)')
            
            # Mark start and end points
            ax.plot(lons[0], lats[0], 's', color=colors[i], 
                   markersize=10, markeredgecolor='black',
                   transform=ccrs.PlateCarree())  # Start
            ax.plot(lons[-1], lats[-1], '^', color=colors[i], 
                   markersize=10, markeredgecolor='black',
                   transform=ccrs.PlateCarree())  # End
            
            # Add track ID label
            ax.text(lons[0], lats[0] + 1, f'T{track_id}',
                   transform=ccrs.PlateCarree(),
                   bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[i], alpha=0.7),
                   fontsize=8, ha='center')
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.LAND, alpha=0.4)
        ax.add_feature(cfeature.OCEAN, alpha=0.4)
        
        # Set extent
        ax.set_extent([INDIAN_OCEAN_BOUNDS['lon_min'], INDIAN_OCEAN_BOUNDS['lon_max'],
                      INDIAN_OCEAN_BOUNDS['lat_min'], INDIAN_OCEAN_BOUNDS['lat_max']])
        
        ax.gridlines(draw_labels=True, alpha=0.5)
        
        ax.set_title(f"{title} - {len(tracks)} tracks")
        if len(tracks) <= 10:
            ax.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def plot_tcc_statistics(self, tccs: List[Dict]) -> plt.Figure:
        """
        Plot statistics of detected TCCs
        
        Args:
            tccs: List of TCC dictionaries
            
        Returns:
            Figure object
        """
        if not tccs:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, 'No TCCs detected', ha='center', va='center')
            ax.set_title('TCC Statistics')
            return fig
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame(tccs)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot 1: Temperature distribution
        axes[0].hist(df['min_tb'], bins=20, alpha=0.7, color='blue')
        axes[0].set_xlabel('Minimum Temperature (K)')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('TCC Minimum Temperature Distribution')
        
        # Plot 2: Size distribution
        axes[1].hist(df['pixel_count'], bins=20, alpha=0.7, color='green')
        axes[1].set_xlabel('Pixel Count')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('TCC Size Distribution')
        
        # Plot 3: Radius distribution
        axes[2].hist(df['max_radius_km'], bins=20, alpha=0.7, color='red')
        axes[2].set_xlabel('Maximum Radius (km)')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('TCC Radius Distribution')
        
        # Plot 4: Temperature vs Size scatter
        axes[3].scatter(df['min_tb'], df['pixel_count'], alpha=0.6)
        axes[3].set_xlabel('Minimum Temperature (K)')
        axes[3].set_ylabel('Pixel Count')
        axes[3].set_title('Temperature vs Size')
        
        # Plot 5: Geographic distribution
        axes[4].scatter(df['convective_lon'], df['convective_lat'], 
                       c=df['min_tb'], cmap='coolwarm', alpha=0.7)
        axes[4].set_xlabel('Longitude')
        axes[4].set_ylabel('Latitude')
        axes[4].set_title('TCC Geographic Distribution')
        
        # Plot 6: Cloud height distribution
        if 'max_cloud_height' in df.columns:
            axes[5].hist(df['max_cloud_height'], bins=20, alpha=0.7, color='purple')
            axes[5].set_xlabel('Maximum Cloud Height (km)')
            axes[5].set_ylabel('Frequency')
            axes[5].set_title('TCC Cloud Height Distribution')
        else:
            axes[5].text(0.5, 0.5, 'Cloud height\ndata not available', 
                        ha='center', va='center')
            axes[5].set_title('Cloud Height Distribution')
        
        plt.tight_layout()
        return fig
    
    def plot_time_series(self, tracks: Dict[int, List[Dict]], 
                        variable: str = 'min_tb') -> plt.Figure:
        """
        Plot time series of TCC properties
        
        Args:
            tracks: Dictionary of tracks
            variable: Variable to plot ('min_tb', 'pixel_count', etc.)
            
        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(tracks)))
        
        for i, (track_id, track_tccs) in enumerate(tracks.items()):
            if len(track_tccs) < 2:
                continue
            
            timestamps = [tcc['timestamp'] for tcc in track_tccs]
            values = [tcc[variable] for tcc in track_tccs]
            
            ax.plot(timestamps, values, 'o-', color=colors[i], 
                   label=f'Track {track_id}', alpha=0.8)
        
        ax.set_xlabel('Time')
        ax.set_ylabel(variable.replace('_', ' ').title())
        ax.set_title(f'TCC {variable.replace("_", " ").title()} Time Series')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig
    
    def create_summary_report(self, tracks: Dict[int, List[Dict]], 
                            track_summary: pd.DataFrame) -> plt.Figure:
        """
        Create a comprehensive summary report
        
        Args:
            tracks: Dictionary of tracks
            track_summary: Summary DataFrame from tracker
            
        Returns:
            Figure object with multiple panels
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create a grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Panel 1: Track duration histogram
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(track_summary['duration_hours'], bins=15, alpha=0.7, color='blue')
        ax1.set_xlabel('Duration (hours)')
        ax1.set_ylabel('Number of Tracks')
        ax1.set_title('Track Duration Distribution')
        
        # Panel 2: Track distance histogram
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(track_summary['total_distance_km'], bins=15, alpha=0.7, color='green')
        ax2.set_xlabel('Total Distance (km)')
        ax2.set_ylabel('Number of Tracks')
        ax2.set_title('Track Distance Distribution')
        
        # Panel 3: Speed distribution
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(track_summary['avg_speed_kmh'], bins=15, alpha=0.7, color='red')
        ax3.set_xlabel('Average Speed (km/h)')
        ax3.set_ylabel('Number of Tracks')
        ax3.set_title('Track Speed Distribution')
        
        # Panel 4: Track start positions
        ax4 = fig.add_subplot(gs[1, :], projection=ccrs.PlateCarree())
        scatter = ax4.scatter(track_summary['start_lon'], track_summary['start_lat'],
                             c=track_summary['duration_hours'], cmap='viridis',
                             s=50, alpha=0.7, transform=ccrs.PlateCarree())
        ax4.add_feature(cfeature.COASTLINE)
        ax4.add_feature(cfeature.LAND, alpha=0.3)
        ax4.set_extent([INDIAN_OCEAN_BOUNDS['lon_min'], INDIAN_OCEAN_BOUNDS['lon_max'],
                       INDIAN_OCEAN_BOUNDS['lat_min'], INDIAN_OCEAN_BOUNDS['lat_max']])
        ax4.gridlines(draw_labels=True, alpha=0.5)
        ax4.set_title('TCC Track Starting Positions (colored by duration)')
        plt.colorbar(scatter, ax=ax4, label='Duration (hours)')
        
        # Panel 5: Summary statistics table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Calculate summary statistics
        stats_data = [
            ['Total Tracks', len(tracks)],
            ['Average Duration (hours)', f"{track_summary['duration_hours'].mean():.1f}"],
            ['Average Distance (km)', f"{track_summary['total_distance_km'].mean():.1f}"],
            ['Average Speed (km/h)', f"{track_summary['avg_speed_kmh'].mean():.1f}"],
            ['Longest Track (hours)', f"{track_summary['duration_hours'].max():.1f}"],
            ['Fastest Track (km/h)', f"{track_summary['avg_speed_kmh'].max():.1f}"]
        ]
        
        table = ax5.table(cellText=stats_data, 
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0.2, 0.2, 0.6, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        ax5.set_title('Track Summary Statistics', y=0.9)
        
        plt.suptitle('TCC Detection and Tracking Summary Report', fontsize=16, y=0.95)
        
        return fig 