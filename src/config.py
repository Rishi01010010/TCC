"""
Configuration file for Tropical Cloud Cluster (TCC) Detection Algorithm
"""

import numpy as np

# ====== TCC Detection Parameters ======

# INSAT-3D Data Configuration
INSAT_RESOLUTION_KM = 4  # 4km spatial resolution at nadir
HALF_HOURLY_MINUTES = 30  # Half-hourly data intervals

# TCC Detection Thresholds (Based on Project Brief)
TCC_IRBT_THRESHOLD = 230  # Kelvin - typical threshold for deep convection
MIN_TCC_RADIUS_DEG = 1.0  # Minimum 1° radius (~111 km)
MIN_TCC_RADIUS_KM = 111   # ~111 km equivalent
MIN_TCC_AREA_KM2 = 34800  # Minimum 34,800 km² (90% of 1° radius circle)

# Independence Criteria
MAX_PARENT_DISTANCE_KM = 1200  # Maximum distance for parent cluster association

# Tracking Search Radii (Based on Project Brief Table)
TRACKING_SEARCH_RADII = {
    3: 450,   # 3 hours: 450 km
    6: 550,   # 6 hours: 550 km
    9: 600,   # 9 hours: 600 km
    12: 650   # 12 hours: 650 km
}

# Geographic Bounds for Indian Ocean Basin
INDIAN_OCEAN_BOUNDS = {
    'lat_min': -40.0,   # Southern boundary
    'lat_max': 30.0,    # Northern boundary  
    'lon_min': 30.0,    # Western boundary (East Africa)
    'lon_max': 120.0    # Eastern boundary (Western Australia)
}

# Required TCC Features (as specified in brief)
REQUIRED_FEATURES = [
    'convective_lat',      # Center coordinates of coldest convection
    'convective_lon',      # Center coordinates of coldest convection
    'pixel_count',         # Number of pixels in TCC with Tb < threshold
    'mean_tb',            # Average temperature of all TCC pixels
    'min_tb',             # Temperature of coldest TCC pixel
    'median_tb',          # Median temperature of all TCC pixels
    'std_tb',             # Standard deviation of Tb for all pixels
    'max_radius_km',      # Largest distance from center to edge
    'min_radius_km',      # Smallest distance from center to edge
    'mean_radius_km',     # Average distance from center to edge
    'max_cloud_height',   # Tallest cloud-top height in TCC
    'mean_cloud_height'   # Average cloud-top height in TCC
]

# Cloud-top height calculation parameters
TB_TO_HEIGHT_FORMULA = {
    'method': 'moist_adiabatic',  # Method for Tb to height conversion
    'surface_temp': 300,          # Assumed surface temperature (K)
    'lapse_rate': 6.5            # Moist adiabatic lapse rate (K/km)
}

# Quality Control Parameters
MIN_PIXEL_COUNT = 100        # Minimum pixels for valid TCC
MAX_STD_TB = 15             # Maximum std dev for coherent structure
MIN_CIRCULARITY = 0.6       # Minimum circularity for TCC shape

# Data Processing
SMOOTHING_KERNEL_SIZE = 3    # Kernel size for spatial smoothing
EDGE_BUFFER_KM = 50         # Buffer from data edges

# ====== Geospatial Constants ======

# Earth radius in km
EARTH_RADIUS_KM = 6371.0

# Degrees to km conversion (approximate at equator)
DEG_TO_KM = 111.0

# ====== Data Processing Parameters ======

# Half-hourly data interval (minutes)
DATA_INTERVAL_MINUTES = 30

# Expected data frequency per day
DATA_POINTS_PER_DAY = 48

# ====== Cloud-top Height Conversion ======

# IRBT to cloud-top height conversion parameters
# Using standard atmospheric lapse rate
SURFACE_TEMP_K = 288.15  # Standard surface temperature (15°C)
LAPSE_RATE_K_PER_KM = 6.5  # Standard atmospheric lapse rate

def irbt_to_height(irbt_kelvin):
    """
    Convert IRBT temperature to approximate cloud-top height
    
    Args:
        irbt_kelvin (float): Infrared brightness temperature in Kelvin
        
    Returns:
        float: Approximate cloud-top height in km
    """
    height_km = (SURFACE_TEMP_K - irbt_kelvin) / LAPSE_RATE_K_PER_KM
    return max(0, height_km)  # Ensure non-negative height

# ====== Output Parameters ======

# Color map for temperature visualization
TEMP_COLORMAP = 'coolwarm'

# Figure size for plots
DEFAULT_FIGURE_SIZE = (12, 8) 