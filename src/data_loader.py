"""
Enhanced Data Loader for INSAT-3D IRBRT Data with Project Brief Specifications
"""

import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import glob
import warnings
import netCDF4 as nc

from .config import (
    INDIAN_OCEAN_BOUNDS, INSAT_RESOLUTION_KM, HALF_HOURLY_MINUTES,
    TCC_IRBT_THRESHOLD
)


class IRBRTDataLoader:
    """
    Enhanced INSAT-3D Infrared Brightness Temperature data loader
    Supporting NetCDF/HDF5 formats as specified in project brief
    """
    
    def __init__(self, data_directory: str):
        """
        Initialize INSAT-3D data loader
        
        Args:
            data_directory: Directory containing INSAT-3D NetCDF/HDF5 files
        """
        self.data_directory = Path(data_directory)
        self.resolution_km = INSAT_RESOLUTION_KM
        self.bounds = INDIAN_OCEAN_BOUNDS
        
        if not self.data_directory.exists():
            print(f"Warning: Data directory {data_directory} does not exist")
    
    def load_time_series(self, start_time: datetime, end_time: datetime,
                        half_hourly: bool = True) -> List[Dict]:
        """
        Load INSAT-3D time series data for specified time range
        
        Args:
            start_time: Start datetime
            end_time: End datetime  
            half_hourly: Whether to load half-hourly data (True) or hourly (False)
            
        Returns:
            List of time step dictionaries with IRBRT data
        """
        print(f"Loading INSAT-3D data from {start_time} to {end_time}")
        
        # Generate time steps
        interval_minutes = HALF_HOURLY_MINUTES if half_hourly else 60
        time_steps = self._generate_time_steps(start_time, end_time, interval_minutes)
        
        time_series_data = []
        for timestamp in time_steps:
            try:
                data_dict = self._load_single_timestep(timestamp)
                if data_dict:
                    time_series_data.append(data_dict)
            except Exception as e:
                print(f"Warning: Failed to load data for {timestamp}: {e}")
                continue
        
        print(f"Successfully loaded {len(time_series_data)} time steps")
        return time_series_data
    
    def _generate_time_steps(self, start_time: datetime, end_time: datetime,
                           interval_minutes: int) -> List[datetime]:
        """Generate list of time steps for data loading"""
        time_steps = []
        current_time = start_time
        
        while current_time <= end_time:
            time_steps.append(current_time)
            current_time += timedelta(minutes=interval_minutes)
        
        return time_steps
    
    def _load_single_timestep(self, timestamp: datetime) -> Optional[Dict]:
        """
        Load INSAT-3D data for a single timestamp
        
        Args:
            timestamp: Target datetime
            
        Returns:
            Dictionary with IRBRT data and coordinates
        """
        # Find matching file for timestamp
        file_path = self._find_data_file(timestamp)
        
        if not file_path:
            print(f"No data file found for {timestamp}")
            return None
        
        try:
            # Load NetCDF/HDF5 file
            if file_path.suffix.lower() in ['.nc', '.nc4']:
                return self._load_netcdf_file(file_path, timestamp)
            elif file_path.suffix.lower() in ['.h5', '.hdf5']:
                return self._load_hdf5_file(file_path, timestamp)
            else:
                print(f"Unsupported file format: {file_path.suffix}")
                return None
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def _find_data_file(self, timestamp: datetime) -> Optional[Path]:
        """
        Find INSAT-3D data file for given timestamp
        
        Args:
            timestamp: Target datetime
            
        Returns:
            Path to data file if found
        """
        # Common INSAT-3D file naming patterns
        patterns = [
            f"*{timestamp.strftime('%Y%m%d_%H%M')}*.nc",
            f"*{timestamp.strftime('%Y%m%d%H%M')}*.nc",
            f"*{timestamp.strftime('%Y%j_%H%M')}*.nc",  # Julian day format
            f"*{timestamp.strftime('%Y%m%d_%H%M')}*.h5",
            f"*{timestamp.strftime('%Y%m%d%H%M')}*.h5",
            f"*{timestamp.strftime('%Y%j_%H%M')}*.h5"
        ]
        
        for pattern in patterns:
            matches = list(self.data_directory.glob(pattern))
            if matches:
                return matches[0]  # Return first match
        
        # Try recursive search
        for pattern in patterns:
            matches = list(self.data_directory.rglob(pattern))
            if matches:
                return matches[0]
        
        return None
    
    def _load_netcdf_file(self, file_path: Path, timestamp: datetime) -> Dict:
        """
        Load NetCDF INSAT-3D file
        
        Args:
            file_path: Path to NetCDF file
            timestamp: Expected timestamp
            
        Returns:
            Data dictionary
        """
        with xr.open_dataset(file_path) as ds:
            # Common INSAT-3D variable names for IRBRT
            irbt_var_names = ['IRBT', 'irbt', 'BT', 'brightness_temperature', 
                             'TBB', 'IR_108', 'IR108', 'temp']
            
            irbt_data = None
            for var_name in irbt_var_names:
                if var_name in ds.variables:
                    irbt_data = ds[var_name].values
                    break
            
            if irbt_data is None:
                raise ValueError(f"No IRBT variable found in {file_path}")
            
            # Get coordinates
            if 'latitude' in ds.variables:
                lats = ds['latitude'].values
            elif 'lat' in ds.variables:
                lats = ds['lat'].values
            else:
                # Generate coordinate grid
                lats = self._generate_lat_grid(irbt_data.shape)
            
            if 'longitude' in ds.variables:
                lons = ds['longitude'].values
            elif 'lon' in ds.variables:
                lons = ds['lon'].values
            else:
                # Generate coordinate grid
                lons = self._generate_lon_grid(irbt_data.shape)
            
            # Ensure 2D coordinate arrays
            if lats.ndim == 1 and lons.ndim == 1:
                lons_2d, lats_2d = np.meshgrid(lons, lats)
            else:
                lats_2d, lons_2d = lats, lons
            
            # Apply quality control and geographic filtering
            irbt_data = self._apply_quality_control(irbt_data, lats_2d, lons_2d)
            
            return {
                'irbt_data': irbt_data,
                'lats': lats_2d,
                'lons': lons_2d,
                'timestamp': pd.Timestamp(timestamp),
                'metadata': {
                    'source': 'insat3d_netcdf',
                    'file_path': str(file_path),
                    'resolution_km': self.resolution_km
                }
            }
    
    def _load_hdf5_file(self, file_path: Path, timestamp: datetime) -> Dict:
        """
        Load HDF5 INSAT-3D file
        
        Args:
            file_path: Path to HDF5 file  
            timestamp: Expected timestamp
            
        Returns:
            Data dictionary
        """
        import h5py
        
        with h5py.File(file_path, 'r') as f:
            # Common HDF5 dataset paths for INSAT-3D
            irbt_paths = [
                'IMG_TIR1/IMG_TIR1',  # INSAT-3D TIR1 channel
                'IMG_TIR2/IMG_TIR2',  # INSAT-3D TIR2 channel
                'IRBT', 'irbt', 'BT', 'brightness_temperature'
            ]
            
            irbt_data = None
            for path in irbt_paths:
                if path in f:
                    irbt_data = f[path][:]
                    break
            
            if irbt_data is None:
                # Try to find any 2D dataset
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset) and len(f[key].shape) == 2:
                        irbt_data = f[key][:]
                        print(f"Using dataset '{key}' as IRBT data")
                        break
            
            if irbt_data is None:
                raise ValueError(f"No suitable IRBT dataset found in {file_path}")
            
            # Try to load coordinates
            lats = None
            lons = None
            
            lat_paths = ['Latitude', 'latitude', 'lat', 'LAT']
            lon_paths = ['Longitude', 'longitude', 'lon', 'LON']
            
            for path in lat_paths:
                if path in f:
                    lats = f[path][:]
                    break
            
            for path in lon_paths:
                if path in f:
                    lons = f[path][:]
                    break
            
            # Generate coordinates if not found
            if lats is None or lons is None:
                lats = self._generate_lat_grid(irbt_data.shape)
                lons = self._generate_lon_grid(irbt_data.shape)
            
            # Ensure 2D coordinate arrays
            if lats.ndim == 1 and lons.ndim == 1:
                lons_2d, lats_2d = np.meshgrid(lons, lats)
            else:
                lats_2d, lons_2d = lats, lons
            
            # Apply quality control
            irbt_data = self._apply_quality_control(irbt_data, lats_2d, lons_2d)
            
            return {
                'irbt_data': irbt_data,
                'lats': lats_2d,
                'lons': lons_2d,
                'timestamp': pd.Timestamp(timestamp),
                'metadata': {
                    'source': 'insat3d_hdf5',
                    'file_path': str(file_path),
                    'resolution_km': self.resolution_km
                }
            }
    
    def _generate_lat_grid(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate latitude grid for Indian Ocean region"""
        nlat, nlon = shape
        lat_min, lat_max = self.bounds['lat_min'], self.bounds['lat_max']
        return np.linspace(lat_max, lat_min, nlat)  # North to South
    
    def _generate_lon_grid(self, shape: Tuple[int, int]) -> np.ndarray:
        """Generate longitude grid for Indian Ocean region"""
        nlat, nlon = shape
        lon_min, lon_max = self.bounds['lon_min'], self.bounds['lon_max']
        return np.linspace(lon_min, lon_max, nlon)  # West to East
    
    def _apply_quality_control(self, irbt_data: np.ndarray, lats: np.ndarray, 
                              lons: np.ndarray) -> np.ndarray:
        """
        Apply quality control to INSAT-3D data
        
        Args:
            irbt_data: Raw IRBT data
            lats: Latitude coordinates
            lons: Longitude coordinates
            
        Returns:
            Quality controlled IRBT data
        """
        # Create copy to avoid modifying original
        qc_data = irbt_data.copy()
        
        # Remove invalid values (typical INSAT-3D fill values)
        invalid_mask = (
            (qc_data < 150) |      # Too cold (likely invalid)
            (qc_data > 350) |      # Too warm (likely invalid)
            (qc_data == -999) |    # Common fill value
            (qc_data == 0) |       # Zero values
            np.isnan(qc_data) |    # NaN values
            np.isinf(qc_data)      # Infinite values
        )
        
        qc_data[invalid_mask] = np.nan
        
        # Apply geographic bounds
        valid_region = (
            (lats >= self.bounds['lat_min']) &
            (lats <= self.bounds['lat_max']) &
            (lons >= self.bounds['lon_min']) &
            (lons <= self.bounds['lon_max'])
        )
        
        qc_data[~valid_region] = np.nan
        
        return qc_data
    
    def get_available_files(self) -> List[Dict]:
        """
        Get list of available INSAT-3D files in data directory
        
        Returns:
            List of file information dictionaries
        """
        file_patterns = ['*.nc', '*.nc4', '*.h5', '*.hdf5']
        available_files = []
        
        for pattern in file_patterns:
            files = list(self.data_directory.glob(pattern))
            files.extend(list(self.data_directory.rglob(pattern)))
            
            for file_path in files:
                try:
                    # Try to extract timestamp from filename
                    timestamp = self._extract_timestamp_from_filename(file_path.name)
                    
                    available_files.append({
                        'file_path': str(file_path),
                        'filename': file_path.name,
                        'timestamp': timestamp,
                        'size_mb': file_path.stat().st_size / (1024 * 1024),
                        'format': file_path.suffix.lower()
                    })
                except Exception as e:
                    print(f"Warning: Could not process {file_path}: {e}")
        
        # Sort by timestamp
        available_files.sort(key=lambda x: x['timestamp'] if x['timestamp'] else datetime.min)
        
        return available_files
    
    def _extract_timestamp_from_filename(self, filename: str) -> Optional[datetime]:
        """Extract timestamp from INSAT-3D filename"""
        import re
        
        # Common INSAT-3D timestamp patterns
        patterns = [
            r'(\d{8})_(\d{4})',      # YYYYMMDD_HHMM
            r'(\d{8})(\d{4})',       # YYYYMMDDHHMM
            r'(\d{7})_(\d{4})',      # YYYYDDD_HHMM (Julian day)
            r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})'  # YYYY-MM-DD_HH-MM
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                try:
                    if len(match.groups()) == 2:  # Date_Time format
                        date_str, time_str = match.groups()
                        if len(date_str) == 8:  # YYYYMMDD
                            year = int(date_str[:4])
                            month = int(date_str[4:6])
                            day = int(date_str[6:8])
                        elif len(date_str) == 7:  # YYYYDDD (Julian)
                            year = int(date_str[:4])
                            day_of_year = int(date_str[4:])
                            base_date = datetime(year, 1, 1)
                            actual_date = base_date + timedelta(days=day_of_year - 1)
                            month, day = actual_date.month, actual_date.day
                        
                        hour = int(time_str[:2])
                        minute = int(time_str[2:4])
                        
                        return datetime(year, month, day, hour, minute)
                    
                    elif len(match.groups()) == 5:  # Separate components
                        year, month, day, hour, minute = map(int, match.groups())
                        return datetime(year, month, day, hour, minute)
                        
                except ValueError:
                    continue
        
        return None


class SyntheticDataGenerator:
    """
    Enhanced synthetic data generator for testing TCC algorithms
    """
    
    @staticmethod
    def generate_sample_data(nlat: int = 200, nlon: int = 200, 
                           num_clusters: int = 3, 
                           background_temp: float = 280.0,
                           cluster_temp_range: Tuple[float, float] = (200.0, 240.0),
                           indian_ocean_region: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic INSAT-3D IRBRT data with realistic TCC patterns
        
        Args:
            nlat: Number of latitude points
            nlon: Number of longitude points
            num_clusters: Number of cloud clusters to generate
            background_temp: Background temperature (K)
            cluster_temp_range: Temperature range for clusters (min, max) in K
            indian_ocean_region: Whether to use Indian Ocean coordinates
            
        Returns:
            Tuple of (irbt_data, lats, lons)
        """
        # Generate coordinate grids
        if indian_ocean_region:
            bounds = INDIAN_OCEAN_BOUNDS
            lats_1d = np.linspace(bounds['lat_max'], bounds['lat_min'], nlat)
            lons_1d = np.linspace(bounds['lon_min'], bounds['lon_max'], nlon)
        else:
            lats_1d = np.linspace(30, -30, nlat)
            lons_1d = np.linspace(60, 120, nlon)
        
        lons, lats = np.meshgrid(lons_1d, lats_1d)
        
        # Initialize with background temperature
        irbt_data = np.full((nlat, nlon), background_temp, dtype=np.float32)
        
        # Add realistic noise
        noise = np.random.normal(0, 5, (nlat, nlon))
        irbt_data += noise
        
        # Generate cloud clusters
        for i in range(num_clusters):
            # Random cluster center
            center_lat = np.random.uniform(lats.min() + 5, lats.max() - 5)
            center_lon = np.random.uniform(lons.min() + 5, lons.max() - 5)
            
            # Random cluster size (ensure meets minimum criteria)
            radius_deg = np.random.uniform(1.5, 4.0)  # Larger than minimum 1°
            
            # Create cluster mask
            distances = np.sqrt((lats - center_lat)**2 + 
                              (lons - center_lon)**2 * np.cos(np.radians(center_lat))**2)
            
            # Gaussian-like cluster with some irregularity
            cluster_mask = distances < radius_deg
            
            # Add temperature structure within cluster
            min_temp, max_temp = cluster_temp_range
            cluster_center_temp = np.random.uniform(min_temp, min_temp + 20)
            
            # Create temperature gradient from center
            cluster_temps = cluster_center_temp + (distances / radius_deg) * 40
            cluster_temps = np.clip(cluster_temps, min_temp, max_temp)
            
            # Apply cluster temperatures
            irbt_data[cluster_mask] = cluster_temps[cluster_mask]
            
            # Add some structure within cluster
            inner_mask = distances < radius_deg * 0.5
            if np.any(inner_mask):
                irbt_data[inner_mask] -= np.random.uniform(10, 30)  # Colder core
        
        # Ensure no temperatures are too extreme
        irbt_data = np.clip(irbt_data, 180, 320)
        
        return irbt_data, lats, lons
    
    @staticmethod
    def generate_evolving_clusters(num_time_steps: int = 10, 
                                 num_initial_clusters: int = 2,
                                 **kwargs) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Generate time series of evolving cloud clusters
        
        Args:
            num_time_steps: Number of time steps to generate
            num_initial_clusters: Initial number of clusters
            **kwargs: Additional arguments for generate_sample_data
            
        Returns:
            List of (irbt_data, lats, lons) tuples
        """
        time_series = []
        
        # Track cluster evolution
        cluster_centers = []
        cluster_intensities = []
        
        for step in range(num_time_steps):
            # Evolve number of clusters
            if step == 0:
                num_clusters = num_initial_clusters
            else:
                # Randomly evolve cluster count
                evolution = np.random.choice([-1, 0, 1], p=[0.2, 0.6, 0.2])
                num_clusters = max(1, num_clusters + evolution)
            
            # Generate data for this time step
            irbt_data, lats, lons = SyntheticDataGenerator.generate_sample_data(
                num_clusters=num_clusters, **kwargs
            )
            
            time_series.append((irbt_data, lats, lons))
        
        return time_series 