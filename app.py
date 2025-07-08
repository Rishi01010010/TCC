"""
Flask Web Application for TCC Detection System
Interactive Dashboard for Tropical Cloud Cluster Analysis
"""

import os
import io
import base64
import json
import traceback
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'tcc_detection_secret_key_2024'

# Global variables to store results
current_results = None
current_pipeline = None

@app.route('/')
def index():
    """Main dashboard page"""
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index page: {e}")
        logger.error(traceback.format_exc())
        return f"Error: {str(e)}", 500

@app.route('/run_detection', methods=['POST'])
def run_detection():
    """Run TCC detection with parameters from web form"""
    try:
        logger.info("Starting TCC detection...")
        
        # Get parameters from form with validation
        use_synthetic = request.form.get('use_synthetic', 'true') == 'true'
        
        # Parse and validate num_time_steps
        try:
            num_time_steps_str = request.form.get('num_time_steps', '8')
            num_time_steps = int(num_time_steps_str)
            logger.info(f"Received num_time_steps: '{num_time_steps_str}' -> {num_time_steps}")
            
            # Validate range
            if num_time_steps < 2 or num_time_steps > 48:
                logger.warning(f"Invalid num_time_steps: {num_time_steps}, using default 8")
                num_time_steps = 8
                flash(f'Invalid time steps value ({num_time_steps_str}). Using default value of 8.', 'warning')
                
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing num_time_steps '{request.form.get('num_time_steps')}': {e}")
            num_time_steps = 8
            flash('Invalid time steps value. Using default value of 8.', 'warning')
        
        data_directory = request.form.get('data_directory', '').strip()
        output_directory = request.form.get('output_directory', 'web_output').strip()
        
        logger.info(f"Validated parameters: synthetic={use_synthetic}, steps={num_time_steps}, output={output_directory}")
        
        # Import the main pipeline (with error handling)
        try:
            from main import TCCPipeline
        except ImportError as e:
            logger.error(f"Failed to import TCCPipeline: {e}")
            flash('System error: Unable to load TCC detection pipeline', 'error')
            return redirect(url_for('index'))
        
        # Initialize pipeline
        pipeline = TCCPipeline(
            data_directory=data_directory if data_directory else None,
            output_directory=output_directory
        )
        
        logger.info("Pipeline initialized, running detection...")
        
        # Run detection
        if use_synthetic:
            try:
                logger.info(f"Running synthetic detection with {num_time_steps} time steps")
                results = pipeline.run_detection_demo(
                    use_synthetic=True,
                    num_time_steps=num_time_steps
                )
                
                if not results:
                    logger.error("Detection pipeline returned empty results")
                    flash('Detection completed but no results generated', 'error')
                    return redirect(url_for('index'))
                    
                logger.info(f"Detection successful: {len(results.get('tracks', {}))} tracks generated")
                
            except Exception as detection_error:
                logger.error(f"Detection pipeline error: {detection_error}")
                logger.error(traceback.format_exc())
                flash(f'Error during TCC detection: {str(detection_error)}', 'error')
                return redirect(url_for('index'))
        else:
            flash('Real data processing not implemented in this demo', 'warning')
            return redirect(url_for('index'))
        
        logger.info("Detection completed, processing results...")
        
        # Process and enhance results
        try:
            results = enhance_results_metadata(results)
            logger.info("Metadata enhancement completed")
        except Exception as e:
            logger.error(f"Metadata enhancement failed: {e}")
            # Continue with original results if enhancement fails
        
        # Store results globally
        global current_results, current_pipeline
        current_results = results
        current_pipeline = pipeline
        
        logger.info(f"Results stored: {results['metadata']['num_tccs']} TCCs, {results['metadata']['num_tracks']} tracks")
        
        flash(f'TCC detection completed successfully! Found {results["metadata"]["num_tccs"]} TCCs in {results["metadata"]["num_tracks"]} tracks.', 'success')
        return redirect(url_for('dashboard'))
        
    except Exception as e:
        logger.error(f"Error in run_detection: {e}")
        logger.error(traceback.format_exc())
        flash(f'Error running detection: {str(e)}', 'error')
        return redirect(url_for('index'))

def enhance_results_metadata(results):
    """
    Enhance results metadata with additional fields needed by dashboard
    """
    try:
        metadata = results.get('metadata', {})
        
        # Ensure basic fields exist with defaults
        metadata.setdefault('num_tccs', 0)
        metadata.setdefault('num_tracks', 0)
        metadata.setdefault('num_time_steps', 0)
        
        # Calculate time span in hours safely
        try:
            if (metadata.get('time_range') and 
                metadata['time_range'].get('start') and 
                metadata['time_range'].get('end')):
                start_time = pd.to_datetime(metadata['time_range']['start'])
                end_time = pd.to_datetime(metadata['time_range']['end'])
                time_span_hours = (end_time - start_time).total_seconds() / 3600
                metadata['time_span_hours'] = round(time_span_hours, 1)
            else:
                metadata['time_span_hours'] = 0
        except Exception as e:
            logger.error(f"Error calculating time span: {e}")
            metadata['time_span_hours'] = 0
        
        # Ensure time_range exists with defaults
        if 'time_range' not in metadata:
            metadata['time_range'] = {'start': None, 'end': None}
        
        # Ensure required features field exists
        if 'required_features' not in metadata:
            # Import from config if available
            try:
                from src.config import REQUIRED_FEATURES
                metadata['required_features'] = REQUIRED_FEATURES
            except ImportError:
                # Fallback list
                metadata['required_features'] = [
                    'convective_lat', 'convective_lon', 'pixel_count', 'min_tb', 'mean_tb',
                    'max_radius_km', 'mean_radius_km', 'area_km2', 'cloud_top_height',
                    'circularity', 'eccentricity'
                ]
        
        results['metadata'] = metadata
        logger.info(f"Enhanced metadata: time_span={metadata['time_span_hours']}h, features={len(metadata['required_features'])}")
        return results
        
    except Exception as e:
        logger.error(f"Error enhancing metadata: {e}")
        logger.error(traceback.format_exc())
        # Return results with minimal safe metadata if enhancement completely fails
        if 'metadata' not in results:
            results['metadata'] = {}
        results['metadata'].update({
            'time_span_hours': 0,
            'required_features': [],
            'time_range': {'start': None, 'end': None},
            'num_tccs': 0,
            'num_tracks': 0,
            'num_time_steps': 0
        })
        return results

@app.route('/dashboard')
def dashboard():
    """Main results dashboard"""
    try:
        if current_results is None:
            flash('No results available. Please run detection first.', 'warning')
            return redirect(url_for('index'))
        
        logger.info("Rendering dashboard...")
        
        # Prepare data for dashboard
        metadata = current_results['metadata']
        track_summary = current_results['track_summary']
        
        logger.info(f"Dashboard data: {len(track_summary)} tracks")
        
        # Generate plots for dashboard
        plots = generate_dashboard_plots()
        
        logger.info(f"Generated {len(plots)} plots for dashboard")
        
        return render_template('dashboard.html', 
                             metadata=metadata,
                             track_summary=track_summary.to_dict('records') if not track_summary.empty else [],
                             plots=plots)
    
    except Exception as e:
        logger.error(f"Error rendering dashboard: {e}")
        logger.error(traceback.format_exc())
        flash(f'Error loading dashboard: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/tcc_data')
def api_tcc_data():
    """API endpoint to get TCC data as JSON"""
    try:
        if current_results is None:
            logger.info("No TCC results available for training")
            return jsonify({'error': 'No TCC data available. Run detection first.'})
        
        # Ensure metadata exists with proper structure
        metadata = current_results.get('metadata', {})
        if not metadata:
            logger.warning("Empty metadata in current_results")
            return jsonify({'error': 'Invalid TCC data. Please run detection again.'})
        
        # Get number of TCCs for validation
        num_tccs = metadata.get('num_tccs', 0)
        logger.info(f"API returning TCC data: {num_tccs} TCCs available")
        
        # Convert track summary to JSON safely
        track_data = []
        try:
            track_summary = current_results.get('track_summary')
            if track_summary is not None and not track_summary.empty:
                track_data = track_summary.to_dict('records')
        except Exception as e:
            logger.warning(f"Error converting track summary: {e}")
            track_data = []
        
        # Ensure minimum required fields
        if 'num_tccs' not in metadata:
            metadata['num_tccs'] = 0
        if 'num_tracks' not in metadata:
            metadata['num_tracks'] = len(track_data)
        
        response_data = {
            'metadata': metadata,
            'tracks': track_data,
            'files': current_results.get('files', {}),
            'success': True
        }
        
        logger.info(f"API response: {metadata.get('num_tccs', 0)} TCCs, {len(track_data)} tracks")
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error in api_tcc_data: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'API error: {str(e)}'})

@app.route('/api/tcc_details')
def api_tcc_details():
    """API endpoint to get detailed TCC information"""
    try:
        if current_results is None:
            return jsonify({'error': 'No results available'})
        
        tracks = current_results['tracks']
        detailed_data = []
        
        for track_id, track_tccs in tracks.items():
            for tcc in track_tccs:
                # Remove non-serializable data safely
                tcc_data = {}
                for k, v in tcc.items():
                    if k != 'cluster_mask' and not isinstance(v, np.ndarray):
                        tcc_data[k] = v
                tcc_data['track_id'] = track_id
                detailed_data.append(tcc_data)
        
        return jsonify(detailed_data)
    
    except Exception as e:
        logger.error(f"Error in api_tcc_details: {e}")
        return jsonify({'error': str(e)})

@app.route('/visualization/<plot_type>')
def get_visualization(plot_type):
    """Generate and return specific visualizations"""
    try:
        if current_results is None:
            return jsonify({'error': 'No results available'})
        
        plot_data = generate_specific_plot(plot_type)
        return jsonify({'plot': plot_data})
    except Exception as e:
        logger.error(f"Error generating visualization {plot_type}: {e}")
        return jsonify({'error': str(e)})

@app.route('/download/<file_type>')
def download_file(file_type):
    """Download generated files"""
    try:
        if current_results is None:
            flash('No results available', 'error')
            return redirect(url_for('index'))
        
        files = current_results.get('files', {})
        
        if file_type in files and os.path.exists(files[file_type]):
            return send_file(files[file_type], as_attachment=True)
        else:
            flash(f'File type {file_type} not found', 'error')
            return redirect(url_for('dashboard'))
    
    except Exception as e:
        logger.error(f"Error downloading file {file_type}: {e}")
        flash(f'Error downloading file: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/ml_training')
def ml_training():
    """Machine Learning training interface"""
    try:
        return render_template('ml_training.html')
    except Exception as e:
        logger.error(f"Error rendering ML training page: {e}")
        flash(f'Error loading ML training page: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/run_ml_training', methods=['POST'])
def run_ml_training():
    """Run ML model training"""
    try:
        if current_results is None:
            return jsonify({'error': 'No TCC data available for training'})
        
        # Get ML parameters
        framework = request.form.get('framework', 'sklearn')
        epochs = int(request.form.get('epochs', 50))
        
        logger.info(f"Starting ML training with {framework}, {epochs} epochs")
        
        # Try to import ML framework (with fallback)
        try:
            from src.ml_framework import TCCMLFramework
            ml_framework = TCCMLFramework(framework=framework)
        except ImportError as e:
            logger.error(f"ML framework import failed: {e}")
            return jsonify({'error': 'ML framework not available. Please ensure all dependencies are installed.'})
        
        # Prepare training data from current results
        tracks = current_results['tracks']
        all_tccs = []
        for track_tccs in tracks.values():
            all_tccs.extend(track_tccs)
        
        if not all_tccs:
            return jsonify({'error': 'No TCC data available for training'})
        
        # Prepare features and labels (with error handling)
        try:
            logger.info(f"Preparing training data from {len(all_tccs)} TCCs")
            features, labels = ml_framework.prepare_training_data(all_tccs, [])
            logger.info(f"Training data prepared: features shape {features.shape}, labels shape {labels.shape}")
            
            # Validate training data
            if len(features) == 0:
                return jsonify({'error': 'No valid features extracted from TCC data'})
            
            if len(set(labels)) < 2:
                logger.warning(f"Only one class in labels: {set(labels)}")
                return jsonify({'error': 'Insufficient data diversity for training (need both positive and negative examples)'})
            
            # Train model
            logger.info(f"Starting model training with {framework}")
            training_results = ml_framework.train(features, labels, epochs=epochs)
            logger.info(f"Training completed: {training_results}")
            
            # Save model
            model_path = ml_framework.save_model('web_trained_model')
            logger.info(f"Model saved to: {model_path}")
            
            return jsonify({
                'success': True,
                'results': training_results,
                'message': f'Model trained successfully using {framework}!',
                'model_path': model_path,
                'data_info': {
                    'num_samples': len(features),
                    'num_features': features.shape[1] if len(features.shape) > 1 else 1,
                    'class_distribution': dict(zip(*np.unique(labels, return_counts=True)))
                }
            })
        
        except Exception as training_error:
            logger.error(f"ML training error: {training_error}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Training failed: {str(training_error)}'})
        
    except Exception as e:
        logger.error(f"Error in run_ml_training: {e}")
        return jsonify({'error': str(e)})

def generate_dashboard_plots():
    """Generate all plots for dashboard"""
    if current_results is None:
        return {}
    
    plots = {}
    
    try:
        # 1. TCC Statistics Overview
        tracks = current_results['tracks']
        all_tccs = []
        for track_tccs in tracks.values():
            all_tccs.extend(track_tccs)
        
        logger.info(f"Generating plots for {len(all_tccs)} TCCs")
        
        if all_tccs:
            # Temperature distribution plot
            try:
                temps = [tcc.get('min_tb', 200) for tcc in all_tccs]
                if temps:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(temps, bins=15, alpha=0.7, color='blue', edgecolor='black')
                    ax.set_xlabel('Minimum Temperature (K)')
                    ax.set_ylabel('Frequency')
                    ax.set_title('TCC Temperature Distribution')
                    ax.grid(True, alpha=0.3)
                    plots['temp_distribution'] = plot_to_base64(fig)
                    plt.close(fig)
                    logger.info("Temperature distribution plot generated")
            except Exception as e:
                logger.error(f"Error generating temperature plot: {e}")
            
            # Size distribution plot
            try:
                sizes = [tcc.get('pixel_count', 0) for tcc in all_tccs]
                if sizes:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(sizes, bins=15, alpha=0.7, color='green', edgecolor='black')
                    ax.set_xlabel('Pixel Count')
                    ax.set_ylabel('Frequency')
                    ax.set_title('TCC Size Distribution')
                    ax.grid(True, alpha=0.3)
                    plots['size_distribution'] = plot_to_base64(fig)
                    plt.close(fig)
                    logger.info("Size distribution plot generated")
            except Exception as e:
                logger.error(f"Error generating size plot: {e}")
            
            # Geographic distribution
            try:
                lats = [tcc.get('convective_lat', 0) for tcc in all_tccs]
                lons = [tcc.get('convective_lon', 0) for tcc in all_tccs]
                temps_for_color = [tcc.get('min_tb', 200) for tcc in all_tccs]
                
                if lats and lons:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    scatter = ax.scatter(lons, lats, c=temps_for_color, cmap='coolwarm', 
                                       s=60, alpha=0.7, edgecolors='black')
                    ax.set_xlabel('Longitude')
                    ax.set_ylabel('Latitude')
                    ax.set_title('TCC Geographic Distribution')
                    ax.grid(True, alpha=0.3)
                    plt.colorbar(scatter, ax=ax, label='Min Temperature (K)')
                    plots['geographic_distribution'] = plot_to_base64(fig)
                    plt.close(fig)
                    logger.info("Geographic distribution plot generated")
            except Exception as e:
                logger.error(f"Error generating geographic plot: {e}")
        
        # 2. Track Summary Statistics
        try:
            track_summary = current_results['track_summary']
            if not track_summary.empty:
                # Track duration histogram
                if 'duration_hours' in track_summary.columns:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(track_summary['duration_hours'], bins=10, alpha=0.7, 
                           color='purple', edgecolor='black')
                    ax.set_xlabel('Duration (hours)')
                    ax.set_ylabel('Number of Tracks')
                    ax.set_title('Track Duration Distribution')
                    ax.grid(True, alpha=0.3)
                    plots['track_duration'] = plot_to_base64(fig)
                    plt.close(fig)
                    logger.info("Track duration plot generated")
                
                # Track correlation matrix
                if len(track_summary) > 1:
                    numeric_cols = track_summary.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 1:
                        try:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            corr_matrix = track_summary[numeric_cols].corr()
                            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                            ax.set_title('Track Characteristics Correlation')
                            plots['track_correlation'] = plot_to_base64(fig)
                            plt.close(fig)
                            logger.info("Track correlation plot generated")
                        except Exception as e:
                            logger.error(f"Error generating correlation plot: {e}")
        except Exception as e:
            logger.error(f"Error generating track plots: {e}")
        
        # 3. Feature Importance (if available)
        try:
            if all_tccs:
                features = ['min_tb', 'pixel_count', 'max_radius_km', 'mean_radius_km', 
                           'area_km2', 'circularity']
                
                # Calculate feature statistics
                feature_stats = {}
                for feature in features:
                    values = [tcc.get(feature, 0) for tcc in all_tccs if tcc.get(feature) is not None]
                    if values:
                        feature_stats[feature] = np.std(values)
                
                if feature_stats:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    features_list = list(feature_stats.keys())
                    importance_values = list(feature_stats.values())
                    
                    bars = ax.bar(features_list, importance_values, alpha=0.7, color='orange', edgecolor='black')
                    ax.set_xlabel('Features')
                    ax.set_ylabel('Standard Deviation (Variability)')
                    ax.set_title('TCC Feature Variability Analysis')
                    ax.tick_params(axis='x', rotation=45)
                    plt.tight_layout()
                    plots['feature_importance'] = plot_to_base64(fig)
                    plt.close(fig)
                    logger.info("Feature importance plot generated")
        except Exception as e:
            logger.error(f"Error generating feature plot: {e}")
    
    except Exception as e:
        logger.error(f"Error in generate_dashboard_plots: {e}")
        logger.error(traceback.format_exc())
    
    logger.info(f"Generated {len(plots)} plots total")
    return plots

def generate_specific_plot(plot_type):
    """Generate specific plot based on type"""
    try:
        if current_results is None:
            return None
        
        tracks = current_results['tracks']
        all_tccs = []
        for track_tccs in tracks.values():
            all_tccs.extend(track_tccs)
        
        if plot_type == 'intensity_evolution' and tracks:
            # Plot intensity evolution over time
            fig, ax = plt.subplots(figsize=(12, 8))
            
            colors = plt.cm.tab10(np.linspace(0, 1, len(tracks)))
            
            for i, (track_id, track_tccs) in enumerate(tracks.items()):
                if len(track_tccs) > 1:
                    times = [tcc.get('timestamp', f'T{j}') for j, tcc in enumerate(track_tccs)]
                    intensities = [tcc.get('min_tb', 200) for tcc in track_tccs]
                    
                    ax.plot(range(len(times)), intensities, 'o-', color=colors[i], 
                           label=f'Track {track_id}', linewidth=2, markersize=6)
            
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Minimum Temperature (K)')
            ax.set_title('TCC Intensity Evolution Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return plot_to_base64(fig)
        
        elif plot_type == 'area_vs_intensity' and all_tccs:
            # Scatter plot of area vs intensity
            fig, ax = plt.subplots(figsize=(10, 8))
            
            areas = [tcc.get('area_km2', 0) for tcc in all_tccs]
            intensities = [tcc.get('min_tb', 200) for tcc in all_tccs]
            
            if areas and intensities:
                scatter = ax.scatter(areas, intensities, alpha=0.7, s=60, 
                                   c=intensities, cmap='coolwarm', edgecolors='black')
                ax.set_xlabel('Area (km²)')
                ax.set_ylabel('Minimum Temperature (K)')
                ax.set_title('TCC Area vs Intensity')
                ax.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax, label='Temperature (K)')
            
            return plot_to_base64(fig)
        
        return None
    
    except Exception as e:
        logger.error(f"Error generating specific plot {plot_type}: {e}")
        return None

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string for web display"""
    try:
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        img_buffer.close()
        return img_base64
    except Exception as e:
        logger.error(f"Error converting plot to base64: {e}")
        return None

@app.route('/about')
def about():
    """About page with system information"""
    try:
        return render_template('about.html')
    except Exception as e:
        logger.error(f"Error rendering about page: {e}")
        return f"Error: {str(e)}", 500

@app.route('/api/system_status')
def system_status():
    """API endpoint for system status"""
    try:
        status = {
            'current_results': current_results is not None,
            'pipeline_initialized': current_pipeline is not None,
            'available_frameworks': []
        }
        
        # Check available ML frameworks
        try:
            import tensorflow as tf
            status['available_frameworks'].append('tensorflow')
        except ImportError:
            pass
        
        try:
            import torch
            status['available_frameworks'].append('pytorch')
        except ImportError:
            pass
        
        status['available_frameworks'].append('sklearn')  # Always available
        
        return jsonify(status)
    
    except Exception as e:
        logger.error(f"Error in system_status: {e}")
        return jsonify({'error': str(e)})

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {error}")
    return render_template('error.html', error=str(error)), 500

@app.errorhandler(404)
def not_found(error):
    """Handle not found errors"""
    return render_template('error.html', error="Page not found"), 404

@app.route('/simple_dashboard')
def simple_dashboard():
    """Simplified dashboard for debugging"""
    if current_results is None:
        return """
        <h1>No Results Available</h1>
        <p>Please <a href="/">run detection</a> first.</p>
        """
    
    try:
        metadata = current_results['metadata']
        track_summary = current_results['track_summary']
        
        html = f"""
        <html>
        <head><title>Simple TCC Dashboard</title></head>
        <body>
            <h1>TCC Detection Results</h1>
            
            <h2>Summary</h2>
            <ul>
                <li>TCCs Detected: {metadata.get('num_tccs', 0)}</li>
                <li>Tracks: {metadata.get('num_tracks', 0)}</li>
                <li>Time Steps: {metadata.get('num_time_steps', 0)}</li>
                <li>Time Span: {metadata.get('time_span_hours', 0)}h</li>
            </ul>
            
            <h2>Track Summary</h2>
            <p>Number of tracks: {len(track_summary)}</p>
            
            <h2>Debug Info</h2>
            <pre>{str(metadata)}</pre>
            
            <p><a href="/dashboard">Try Full Dashboard</a> | <a href="/">Home</a></p>
        </body>
        </html>
        """
        
        return html
        
    except Exception as e:
        return f"""
        <html>
        <head><title>Error</title></head>
        <body>
            <h1>Simple Dashboard Error</h1>
            <p>Error: {str(e)}</p>
            <pre>{traceback.format_exc()}</pre>
            <p><a href="/">Home</a></p>
        </body>
        </html>
        """

@app.route('/test_minimal_detection', methods=['POST'])
def test_minimal_detection():
    """Minimal detection test that bypasses complex rendering"""
    try:
        logger.info("Starting minimal detection test...")
        
        from main import TCCPipeline
        pipeline = TCCPipeline(output_directory='test_minimal')
        
        results = pipeline.run_detection_demo(use_synthetic=True, num_time_steps=2)
        
        # Store results globally
        global current_results
        current_results = enhance_results_metadata(results)
        
        return jsonify({
            'success': True,
            'num_tccs': current_results['metadata']['num_tccs'],
            'num_tracks': current_results['metadata']['num_tracks'],
            'redirect_url': '/simple_dashboard'
        })
        
    except Exception as e:
        logger.error(f"Minimal detection failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Create directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('web_output', exist_ok=True)
    
    logger.info("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000) 