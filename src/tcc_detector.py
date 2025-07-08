"""
Enhanced TCC Detection Module with Project Brief Specifications
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage import measure, morphology
try:
    from skimage.feature import peak_local_maxima
except ImportError:
    peak_local_maxima = None

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from typing import List, Dict, Tuple, Optional

from .config import (
    TCC_IRBT_THRESHOLD, MIN_TCC_RADIUS_KM, MIN_TCC_AREA_KM2,
    MAX_PARENT_DISTANCE_KM, MIN_PIXEL_COUNT, MIN_CIRCULARITY,
    INDIAN_OCEAN_BOUNDS, irbt_to_height, DEG_TO_KM
)


class TCCDetector:
    """
    Enhanced Tropical Cloud Cluster detector implementing project brief specifications
    """
    
    def __init__(self):
        self.threshold_k = TCC_IRBT_THRESHOLD
        self.min_radius_km = MIN_TCC_RADIUS_KM
        self.min_area_km2 = MIN_TCC_AREA_KM2
        self.max_parent_distance_km = MAX_PARENT_DISTANCE_KM
        self.min_circularity = MIN_CIRCULARITY
        
    def detect_tccs(self, irbt_data: np.ndarray, lats: np.ndarray, 
                   lons: np.ndarray, timestamp: pd.Timestamp = None) -> List[Dict]:
        """
        Detect TCCs using project brief specifications
        
        Args:
            irbt_data: IRBT temperature data (Kelvin)
            lats: Latitude coordinates
            lons: Longitude coordinates
            timestamp: Data timestamp
            
        Returns:
            List of TCC dictionaries with all required features
        """
        print(f"Detecting TCCs with threshold {self.threshold_k}K...")
        
        # Step 1: Apply Indian Ocean geographic constraints
        valid_mask = self._apply_geographic_constraints(lats, lons)
        
        # Step 2: Apply IRBT threshold filtering
        cold_pixels = (irbt_data < self.threshold_k) & valid_mask
        
        if not np.any(cold_pixels):
            print("No pixels below IRBT threshold found")
            return []
        
        # Step 3: Find connected components (candidate clusters)
        candidate_clusters = self._find_candidate_clusters(cold_pixels, irbt_data, lats, lons)
        print(f"Found {len(candidate_clusters)} candidate clusters")
        
        # Step 4: Apply size and area criteria
        size_filtered = self._apply_size_criteria(candidate_clusters, lats, lons)
        print(f"After size filtering: {len(size_filtered)} clusters")
        
        # Step 5: Apply circular structure filtering
        shape_filtered = self._apply_circular_filtering(size_filtered, irbt_data, lats, lons)
        print(f"After circularity filtering: {len(shape_filtered)} clusters")
        
        # Step 6: Apply independence algorithm
        independent_tccs = self._apply_independence_algorithm(shape_filtered)
        print(f"After independence filtering: {len(independent_tccs)} TCCs")
        
        # Step 7: Extract all required features
        tccs_with_features = []
        for i, tcc in enumerate(independent_tccs):
            tcc_features = self._extract_tcc_features(tcc, irbt_data, lats, lons, i+1)
            if tcc_features:
                tccs_with_features.append(tcc_features)
        
        print(f"Final TCCs with complete features: {len(tccs_with_features)}")
        return tccs_with_features
    
    def _apply_geographic_constraints(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Apply Indian Ocean basin geographic constraints"""
        bounds = INDIAN_OCEAN_BOUNDS
        valid_lat = (lats >= bounds['lat_min']) & (lats <= bounds['lat_max'])
        valid_lon = (lons >= bounds['lon_min']) & (lons <= bounds['lon_max'])
        
        # Create 2D mask
        valid_mask = np.zeros_like(lats, dtype=bool)
        for i in range(lats.shape[0]):
            for j in range(lats.shape[1]):
                valid_mask[i, j] = valid_lat[i, j] and valid_lon[i, j]
        
        return valid_mask
    
    def _find_candidate_clusters(self, cold_pixels: np.ndarray, irbt_data: np.ndarray,
                               lats: np.ndarray, lons: np.ndarray) -> List[Dict]:
        """Find connected components as candidate clusters"""
        # Label connected components
        labeled_clusters, num_clusters = ndimage.label(cold_pixels)
        
        candidates = []
        for cluster_id in range(1, num_clusters + 1):
            cluster_mask = labeled_clusters == cluster_id
            
            if np.sum(cluster_mask) < MIN_PIXEL_COUNT:
                continue
            
            # Find center of coldest convection
            cluster_temps = irbt_data[cluster_mask]
            min_temp_idx = np.unravel_index(
                np.argmin(irbt_data * cluster_mask + (1 - cluster_mask) * 1000),
                irbt_data.shape
            )
            
            candidates.append({
                'cluster_id': cluster_id,
                'mask': cluster_mask,
                'center_lat': lats[min_temp_idx],
                'center_lon': lons[min_temp_idx],
                'min_temp': np.min(cluster_temps),
                'pixel_count': np.sum(cluster_mask)
            })
        
        return candidates
    
    def _apply_size_criteria(self, candidates: List[Dict], lats: np.ndarray, 
                           lons: np.ndarray) -> List[Dict]:
        """Apply size criteria: radius ≥ 1° and area ≥ 34,800 km²"""
        filtered = []
        
        for candidate in candidates:
            # Calculate actual area in km²
            area_km2 = self._calculate_cluster_area(candidate['mask'], lats, lons)
            
            # Calculate maximum radius
            max_radius_km = self._calculate_max_radius(
                candidate['mask'], candidate['center_lat'], 
                candidate['center_lon'], lats, lons
            )
            
            # Apply criteria from project brief
            radius_ok = max_radius_km >= self.min_radius_km  # ≥ 111 km (1°)
            area_ok = area_km2 >= self.min_area_km2  # ≥ 34,800 km²
            
            if radius_ok and area_ok:
                candidate['area_km2'] = area_km2
                candidate['max_radius_km'] = max_radius_km
                filtered.append(candidate)
        
        return filtered
    
    def _apply_circular_filtering(self, candidates: List[Dict], irbt_data: np.ndarray,
                                lats: np.ndarray, lons: np.ndarray) -> List[Dict]:
        """Filter out non-circular convective structures"""
        filtered = []
        
        for candidate in candidates:
            circularity = self._calculate_circularity(candidate['mask'])
            
            # Apply circularity threshold from config
            if circularity >= self.min_circularity:
                candidate['circularity'] = circularity
                filtered.append(candidate)
        
        return filtered
    
    def _apply_independence_algorithm(self, candidates: List[Dict]) -> List[Dict]:
        """
        Apply independence algorithm from project brief:
        - TCCs within 1200 km are considered subsets of parent cluster
        - TCCs >1200 km apart are independent
        """
        if len(candidates) <= 1:
            return candidates
        
        # Calculate distances between all candidates
        centers = np.array([[c['center_lat'], c['center_lon']] for c in candidates])
        distances_km = self._calculate_distance_matrix(centers)
        
        # Find parent clusters (largest in each group)
        independent_tccs = []
        processed = set()
        
        for i, candidate in enumerate(candidates):
            if i in processed:
                continue
            
            # Find all candidates within 1200 km
            nearby_indices = np.where(distances_km[i] <= self.max_parent_distance_km)[0]
            
            # Select the largest (parent) cluster from the group
            group_candidates = [candidates[j] for j in nearby_indices]
            parent = max(group_candidates, key=lambda x: x['area_km2'])
            
            # Mark all in group as processed
            processed.update(nearby_indices)
            
            # Add parent cluster with group information
            parent['is_parent'] = True
            parent['group_size'] = len(group_candidates)
            independent_tccs.append(parent)
        
        return independent_tccs
    
    def _calculate_cluster_area(self, mask: np.ndarray, lats: np.ndarray, 
                              lons: np.ndarray) -> float:
        """Calculate actual area of cluster in km²"""
        # Get pixel coordinates
        pixel_lats = lats[mask]
        pixel_lons = lons[mask]
        
        # Estimate pixel size at cluster center
        center_lat = np.mean(pixel_lats)
        
        # Calculate area assuming rectangular pixels
        lat_spacing = np.abs(np.diff(lats[:, 0]).mean()) if lats.shape[0] > 1 else 0.04
        lon_spacing = np.abs(np.diff(lons[0, :]).mean()) if lats.shape[1] > 1 else 0.04
        
        # Convert to km
        lat_km = lat_spacing * 111.0  # 1° ≈ 111 km
        lon_km = lon_spacing * 111.0 * np.cos(np.radians(center_lat))
        
        pixel_area_km2 = lat_km * lon_km
        total_area_km2 = np.sum(mask) * pixel_area_km2
        
        return total_area_km2
    
    def _calculate_max_radius(self, mask: np.ndarray, center_lat: float, 
                            center_lon: float, lats: np.ndarray, lons: np.ndarray) -> float:
        """Calculate maximum radius from center to edge"""
        # Get all pixel coordinates in cluster
        pixel_lats = lats[mask]
        pixel_lons = lons[mask]
        
        # Calculate distances from center to all pixels
        distances = []
        for lat, lon in zip(pixel_lats, pixel_lons):
            dist_km = DEG_TO_KM * np.sqrt((lat - center_lat)**2 + 
                                         (lon - center_lon)**2 * np.cos(np.radians(center_lat))**2)
            distances.append(dist_km)
        
        return max(distances) if distances else 0.0
    
    def _calculate_circularity(self, mask: np.ndarray) -> float:
        """Calculate circularity of cluster shape"""
        if CV2_AVAILABLE:
            return self._calculate_circularity_cv2(mask)
        else:
            return self._calculate_circularity_simple(mask)
    
    def _calculate_circularity_cv2(self, mask: np.ndarray) -> float:
        """Calculate circularity using OpenCV"""
        # Find contours
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate circularity = 4π × Area / Perimeter²
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter == 0:
            return 0.0
        
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        return min(circularity, 1.0)  # Cap at 1.0 for perfect circle
    
    def _calculate_circularity_simple(self, mask: np.ndarray) -> float:
        """Simple circularity calculation without OpenCV"""
        # Calculate area
        area = np.sum(mask)
        
        if area == 0:
            return 0.0
        
        # Find perimeter using edge detection
        from scipy import ndimage
        edges = ndimage.binary_erosion(mask) ^ mask
        perimeter = np.sum(edges)
        
        if perimeter == 0:
            return 0.0
        
        # Calculate circularity = 4π × Area / Perimeter²
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        return min(circularity, 1.0)  # Cap at 1.0 for perfect circle
    
    def _calculate_distance_matrix(self, centers: np.ndarray) -> np.ndarray:
        """Calculate distance matrix between lat/lon centers in km"""
        n = len(centers)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    lat1, lon1 = centers[i]
                    lat2, lon2 = centers[j]
                    
                    # Simple distance calculation (good enough for regional analysis)
                    dlat = lat2 - lat1
                    dlon = (lon2 - lon1) * np.cos(np.radians((lat1 + lat2) / 2))
                    
                    distance_deg = np.sqrt(dlat**2 + dlon**2)
                    distances[i, j] = distance_deg * DEG_TO_KM
        
        return distances
    
    def _extract_tcc_features(self, tcc: Dict, irbt_data: np.ndarray, 
                            lats: np.ndarray, lons: np.ndarray, tcc_id: int) -> Dict:
        """Extract all required TCC features from project brief"""
        mask = tcc['mask']
        cluster_temps = irbt_data[mask]
        cluster_lats = lats[mask]
        cluster_lons = lons[mask]
        
        if len(cluster_temps) == 0:
            return None
        
        # Basic temperature statistics
        mean_tb = np.mean(cluster_temps)
        min_tb = np.min(cluster_temps)
        median_tb = np.median(cluster_temps)
        std_tb = np.std(cluster_temps)
        
        # Cloud heights using temperature-height conversion
        cloud_heights = [irbt_to_height(tb) for tb in cluster_temps]
        max_cloud_height = max(cloud_heights)
        mean_cloud_height = np.mean(cloud_heights)
        
        # Radius calculations
        center_lat, center_lon = tcc['center_lat'], tcc['center_lon']
        radii = []
        for lat, lon in zip(cluster_lats, cluster_lons):
            radius_km = DEG_TO_KM * np.sqrt((lat - center_lat)**2 + 
                                           (lon - center_lon)**2 * np.cos(np.radians(center_lat))**2)
            radii.append(radius_km)
        
        return {
            'tcc_id': tcc_id,
            'convective_lat': center_lat,
            'convective_lon': center_lon,
            'pixel_count': len(cluster_temps),
            'mean_tb': mean_tb,
            'min_tb': min_tb,
            'median_tb': median_tb,
            'std_tb': std_tb,
            'max_radius_km': max(radii) if radii else 0.0,
            'min_radius_km': min(radii) if radii else 0.0,
            'mean_radius_km': np.mean(radii) if radii else 0.0,
            'max_cloud_height': max_cloud_height,
            'mean_cloud_height': mean_cloud_height,
            'area_km2': tcc.get('area_km2', 0.0),
            'circularity': tcc.get('circularity', 0.0),
            'is_parent': tcc.get('is_parent', False),
            'group_size': tcc.get('group_size', 1),
            'cluster_mask': mask  # For tracking
        } 