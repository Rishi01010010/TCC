"""
Enhanced TCC Tracking Module with Project Brief Specifications
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy.spatial.distance import cdist

from .config import TRACKING_SEARCH_RADII, DEG_TO_KM
from .tcc_detector import TCCDetector


class TCCTracker:
    """
    Enhanced TCC tracker implementing project brief tracking specifications
    """
    
    def __init__(self):
        self.detector = TCCDetector()
        self.search_radii = TRACKING_SEARCH_RADII  # Time-based search radii from brief
        self.max_gap_hours = max(TRACKING_SEARCH_RADII.keys())  # 12 hours max
        self.tracks = {}  # Active tracks storage
        self.next_track_id = 1
        
    def track_tccs(self, time_series_data: List[Dict]) -> Dict[int, List[Dict]]:
        """
        Track TCCs across time using project brief specifications
        
        Args:
            time_series_data: List of time step data dictionaries
            
        Returns:
            Dictionary mapping track_id to list of TCC observations
        """
        print(f"Tracking TCCs across {len(time_series_data)} time steps...")
        
        all_tracks = {}
        self.tracks = {}  # Reset active tracks
        self.next_track_id = 1
        
        for step_idx, time_step in enumerate(time_series_data):
            print(f"Processing time step {step_idx + 1}/{len(time_series_data)}: {time_step['timestamp']}")
            
            # Detect TCCs in current time step
            current_tccs = self.detector.detect_tccs(
                time_step['irbt_data'],
                time_step['lats'], 
                time_step['lons'],
                time_step['timestamp']
            )
            
            # Add temporal information
            for tcc in current_tccs:
                tcc['timestamp'] = time_step['timestamp']
                tcc['time_step'] = step_idx + 1
            
            if step_idx == 0:
                # Initialize tracks for first time step
                self._initialize_tracks(current_tccs)
            else:
                # Match TCCs to existing tracks using time-based search radii
                self._update_tracks(current_tccs, time_step['timestamp'])
            
            # Clean up old tracks
            self._cleanup_old_tracks(time_step['timestamp'])
        
        # Convert active tracks to final format
        for track_id, track_data in self.tracks.items():
            if track_data['observations']:
                all_tracks[track_id] = track_data['observations']
        
        print(f"Final tracking result: {len(all_tracks)} tracks")
        return all_tracks
    
    def _initialize_tracks(self, tccs: List[Dict]):
        """Initialize tracks for first time step"""
        for tcc in tccs:
            track_id = self.next_track_id
            self.next_track_id += 1
            
            # Add tracking information
            tcc['track_id'] = track_id
            tcc['track_position'] = 1
            tcc['track_duration_hours'] = 0
            
            self.tracks[track_id] = {
                'last_timestamp': tcc['timestamp'],
                'last_position': (tcc['convective_lat'], tcc['convective_lon']),
                'observations': [tcc],
                'total_distance_km': 0.0
            }
        
        print(f"Initialized {len(tccs)} tracks")
    
    def _update_tracks(self, current_tccs: List[Dict], current_timestamp: pd.Timestamp):
        """Update tracks using time-based search radii from project brief"""
        if not current_tccs or not self.tracks:
            # Start new tracks for unmatched TCCs
            for tcc in current_tccs:
                self._start_new_track(tcc, current_timestamp)
            return
        
        # Calculate time gaps and corresponding search radii
        active_tracks = list(self.tracks.items())
        track_positions = []
        track_ids = []
        search_radii_km = []
        
        for track_id, track_data in active_tracks:
            last_timestamp = track_data['last_timestamp']
            time_gap_hours = (current_timestamp - last_timestamp).total_seconds() / 3600
            
            # Skip tracks that are too old
            if time_gap_hours > self.max_gap_hours:
                continue
            
            # Get appropriate search radius based on time gap
            search_radius_km = self._get_search_radius(time_gap_hours)
            
            track_positions.append(track_data['last_position'])
            track_ids.append(track_id)
            search_radii_km.append(search_radius_km)
        
        if not track_positions:
            # All tracks are too old, start new ones
            for tcc in current_tccs:
                self._start_new_track(tcc, current_timestamp)
            return
        
        # Match TCCs to tracks using variable search radii
        matches = self._match_tccs_to_tracks(
            current_tccs, track_positions, track_ids, search_radii_km
        )
        
        # Update matched tracks
        matched_tcc_indices = set()
        for tcc_idx, track_id in matches:
            tcc = current_tccs[tcc_idx]
            self._update_existing_track(track_id, tcc, current_timestamp)
            matched_tcc_indices.add(tcc_idx)
        
        # Start new tracks for unmatched TCCs
        for i, tcc in enumerate(current_tccs):
            if i not in matched_tcc_indices:
                self._start_new_track(tcc, current_timestamp)
    
    def _get_search_radius(self, time_gap_hours: float) -> float:
        """Get search radius based on time gap (from project brief table)"""
        # Find the appropriate radius from the table
        if time_gap_hours <= 3:
            return self.search_radii[3]   # 450 km
        elif time_gap_hours <= 6:
            return self.search_radii[6]   # 550 km
        elif time_gap_hours <= 9:
            return self.search_radii[9]   # 600 km
        elif time_gap_hours <= 12:
            return self.search_radii[12]  # 650 km
        else:
            return 0  # Too old, don't match
    
    def _match_tccs_to_tracks(self, current_tccs: List[Dict], track_positions: List[Tuple],
                            track_ids: List[int], search_radii_km: List[float]) -> List[Tuple[int, int]]:
        """Match TCCs to tracks using variable search radii"""
        matches = []
        
        if not current_tccs or not track_positions:
            return matches
        
        # Calculate current TCC positions
        current_positions = np.array([
            [tcc['convective_lat'], tcc['convective_lon']] for tcc in current_tccs
        ])
        track_positions = np.array(track_positions)
        
        # Calculate all distances
        distances_deg = cdist(current_positions, track_positions, metric='euclidean')
        
        # Convert to km (approximate)
        distances_km = distances_deg * 111.0  # 1° ≈ 111 km
        
        # Find matches within search radii
        used_tracks = set()
        for tcc_idx in range(len(current_tccs)):
            best_match = None
            min_distance = float('inf')
            
            for track_idx in range(len(track_ids)):
                if track_idx in used_tracks:
                    continue
                
                distance_km = distances_km[tcc_idx, track_idx]
                search_radius = search_radii_km[track_idx]
                
                if distance_km <= search_radius and distance_km < min_distance:
                    best_match = track_idx
                    min_distance = distance_km
            
            if best_match is not None:
                matches.append((tcc_idx, track_ids[best_match]))
                used_tracks.add(best_match)
        
        return matches
    
    def _update_existing_track(self, track_id: int, tcc: Dict, current_timestamp: pd.Timestamp):
        """Update an existing track with new TCC observation"""
        track_data = self.tracks[track_id]
        last_position = track_data['last_position']
        
        # Calculate movement distance
        new_position = (tcc['convective_lat'], tcc['convective_lon'])
        distance_km = self._calculate_distance_km(last_position, new_position)
        
        # Calculate track duration
        time_gap_hours = (current_timestamp - track_data['last_timestamp']).total_seconds() / 3600
        
        # Update TCC with track information
        tcc['track_id'] = track_id
        tcc['track_position'] = len(track_data['observations']) + 1
        tcc['track_duration_hours'] = time_gap_hours
        tcc['movement_distance_km'] = distance_km
        tcc['movement_speed_kmh'] = distance_km / time_gap_hours if time_gap_hours > 0 else 0
        
        # Update track data
        track_data['last_timestamp'] = current_timestamp
        track_data['last_position'] = new_position
        track_data['observations'].append(tcc)
        track_data['total_distance_km'] += distance_km
    
    def _start_new_track(self, tcc: Dict, current_timestamp: pd.Timestamp):
        """Start a new track for unmatched TCC"""
        track_id = self.next_track_id
        self.next_track_id += 1
        
        # Add tracking information
        tcc['track_id'] = track_id
        tcc['track_position'] = 1
        tcc['track_duration_hours'] = 0
        tcc['movement_distance_km'] = 0
        tcc['movement_speed_kmh'] = 0
        
        self.tracks[track_id] = {
            'last_timestamp': current_timestamp,
            'last_position': (tcc['convective_lat'], tcc['convective_lon']),
            'observations': [tcc],
            'total_distance_km': 0.0
        }
    
    def _cleanup_old_tracks(self, current_timestamp: pd.Timestamp):
        """Remove tracks that are too old (>12 hours)"""
        to_remove = []
        
        for track_id, track_data in self.tracks.items():
            time_gap_hours = (current_timestamp - track_data['last_timestamp']).total_seconds() / 3600
            if time_gap_hours > self.max_gap_hours:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
    
    def _calculate_distance_km(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Calculate distance between two lat/lon positions in km"""
        lat1, lon1 = pos1
        lat2, lon2 = pos2
        
        # Simple approximation for short distances
        dlat = lat2 - lat1
        dlon = (lon2 - lon1) * np.cos(np.radians((lat1 + lat2) / 2))
        
        distance_deg = np.sqrt(dlat**2 + dlon**2)
        return distance_deg * 111.0  # Convert to km
    
    def get_track_summary(self, tracks: Dict[int, List[Dict]]) -> pd.DataFrame:
        """
        Generate enhanced track summary with project brief metrics
        
        Args:
            tracks: Dictionary of track data
            
        Returns:
            DataFrame with track statistics
        """
        if not tracks:
            return pd.DataFrame()
        
        summary_data = []
        
        for track_id, observations in tracks.items():
            if len(observations) < 1:
                continue
            
            # Basic track information
            first_obs = observations[0]
            last_obs = observations[-1]
            
            # Calculate track duration
            if len(observations) > 1:
                duration_hours = (last_obs['timestamp'] - first_obs['timestamp']).total_seconds() / 3600
            else:
                duration_hours = 0
            
            # Calculate total distance
            total_distance = sum(obs.get('movement_distance_km', 0) for obs in observations[1:])
            
            # Calculate average properties
            avg_area = np.mean([obs.get('area_km2', 0) for obs in observations])
            avg_intensity = np.mean([obs['min_tb'] for obs in observations])  # Coldest temps = highest intensity
            avg_circularity = np.mean([obs.get('circularity', 0) for obs in observations])
            
            # Track evolution metrics
            intensity_change = last_obs['min_tb'] - first_obs['min_tb']  # Negative = intensification
            area_change = last_obs.get('area_km2', 0) - first_obs.get('area_km2', 0)
            
            summary_data.append({
                'track_id': track_id,
                'start_time': first_obs['timestamp'],
                'end_time': last_obs['timestamp'],
                'duration_hours': duration_hours,
                'num_observations': len(observations),
                'total_distance_km': total_distance,
                'avg_speed_kmh': total_distance / duration_hours if duration_hours > 0 else 0,
                'start_lat': first_obs['convective_lat'],
                'start_lon': first_obs['convective_lon'],
                'end_lat': last_obs['convective_lat'],
                'end_lon': last_obs['convective_lon'],
                'avg_area_km2': avg_area,
                'avg_min_tb': avg_intensity,
                'avg_circularity': avg_circularity,
                'intensity_change_k': intensity_change,
                'area_change_km2': area_change,
                'max_area_km2': max(obs.get('area_km2', 0) for obs in observations),
                'min_intensity_tb': min(obs['min_tb'] for obs in observations),  # Coldest = most intense
                'is_parent_cluster': any(obs.get('is_parent', False) for obs in observations)
            })
        
        return pd.DataFrame(summary_data)
    
    def get_tracking_statistics(self, tracks: Dict[int, List[Dict]]) -> Dict:
        """Get overall tracking statistics"""
        if not tracks:
            return {'total_tracks': 0}
        
        total_tracks = len(tracks)
        total_observations = sum(len(obs) for obs in tracks.values())
        
        # Duration statistics
        durations = []
        distances = []
        speeds = []
        
        for observations in tracks.values():
            if len(observations) > 1:
                first_obs = observations[0]
                last_obs = observations[-1]
                duration = (last_obs['timestamp'] - first_obs['timestamp']).total_seconds() / 3600
                durations.append(duration)
                
                total_dist = sum(obs.get('movement_distance_km', 0) for obs in observations[1:])
                distances.append(total_dist)
                
                if duration > 0:
                    speeds.append(total_dist / duration)
        
        return {
            'total_tracks': total_tracks,
            'total_observations': total_observations,
            'avg_duration_hours': np.mean(durations) if durations else 0,
            'max_duration_hours': max(durations) if durations else 0,
            'avg_distance_km': np.mean(distances) if distances else 0,
            'max_distance_km': max(distances) if distances else 0,
            'avg_speed_kmh': np.mean(speeds) if speeds else 0,
            'max_speed_kmh': max(speeds) if speeds else 0
        } 