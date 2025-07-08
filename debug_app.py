#!/usr/bin/env python3
"""
Debug version of Flask app to isolate the issue
"""

import os
import sys
import logging
import traceback
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'debug_key'

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Debug TCC App</title></head>
    <body>
        <h1>Debug TCC Detection Test</h1>
        <form method="POST" action="/test_detection">
            <button type="submit">Test Detection</button>
        </form>
        <div id="result"></div>
        
        <script>
        setInterval(function() {
            fetch('/debug_status')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('result').innerHTML = 
                        '<pre>' + JSON.stringify(data, null, 2) + '</pre>';
                });
        }, 2000);
        </script>
    </body>
    </html>
    """

@app.route('/test_detection', methods=['POST'])
def test_detection():
    try:
        logger.info("=== STARTING DEBUG DETECTION ===")
        
        # Step 1: Test imports
        logger.info("Step 1: Testing imports...")
        try:
            from main import TCCPipeline
            logger.info("✓ TCCPipeline import successful")
        except Exception as e:
            logger.error(f"❌ Import failed: {e}")
            return jsonify({'error': f'Import failed: {e}', 'step': 'import'})
        
        # Step 2: Test pipeline initialization  
        logger.info("Step 2: Testing pipeline initialization...")
        try:
            pipeline = TCCPipeline(output_directory='debug_output')
            logger.info("✓ Pipeline initialization successful")
        except Exception as e:
            logger.error(f"❌ Pipeline init failed: {e}")
            return jsonify({'error': f'Pipeline init failed: {e}', 'step': 'init'})
        
        # Step 3: Test detection with minimal parameters
        logger.info("Step 3: Testing detection...")
        try:
            results = pipeline.run_detection_demo(use_synthetic=True, num_time_steps=2)
            logger.info("✓ Detection completed!")
            logger.info(f"Results keys: {list(results.keys())}")
        except Exception as e:
            logger.error(f"❌ Detection failed: {e}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Detection failed: {e}', 'step': 'detection', 'traceback': traceback.format_exc()})
        
        # Step 4: Test metadata enhancement
        logger.info("Step 4: Testing metadata enhancement...")
        try:
            from app import enhance_results_metadata
            enhanced_results = enhance_results_metadata(results)
            logger.info("✓ Metadata enhancement successful")
        except Exception as e:
            logger.error(f"❌ Metadata enhancement failed: {e}")
            return jsonify({'error': f'Metadata enhancement failed: {e}', 'step': 'metadata'})
        
        logger.info("=== ALL STEPS SUCCESSFUL ===")
        return jsonify({
            'success': True,
            'message': 'All detection steps completed successfully!',
            'num_tccs': enhanced_results['metadata']['num_tccs'],
            'num_tracks': enhanced_results['metadata']['num_tracks']
        })
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'step': 'unexpected',
            'traceback': traceback.format_exc()
        })

@app.route('/debug_status')
def debug_status():
    """Return debug status information"""
    try:
        import main
        import src.tcc_detector
        import src.tcc_tracker
        import src.visualization
        
        status = {
            'python_version': sys.version,
            'working_directory': os.getcwd(),
            'imports_available': {
                'main': True,
                'tcc_detector': True,
                'tcc_tracker': True,
                'visualization': True
            },
            'memory_info': 'Available',
            'directories': {
                'templates': os.path.exists('templates'),
                'src': os.path.exists('src'),
                'debug_output': os.path.exists('debug_output')
            }
        }
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        })

if __name__ == '__main__':
    # Create debug output directory
    os.makedirs('debug_output', exist_ok=True)
    
    logger.info("Starting debug Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port 