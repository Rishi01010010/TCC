#!/usr/bin/env python3
"""
Simple step-by-step test for TCC detection web interface
"""

from flask import Flask, jsonify, request
import os
import sys
import traceback

app = Flask(__name__)

@app.route('/')
def test_home():
    return """
    <h1>TCC Detection Step-by-Step Test</h1>
    <div>
        <button onclick="testStep1()">1. Test Imports</button><br><br>
        <button onclick="testStep2()">2. Test Pipeline Init</button><br><br>
        <button onclick="testStep3()">3. Test Detection</button><br><br>
        <button onclick="testStep4()">4. Test Template</button><br><br>
    </div>
    <div id="result" style="margin-top: 20px; padding: 10px; border: 1px solid #ccc;"></div>
    
    <script>
    function testStep1() {
        fetch('/test/imports')
            .then(r => r.json())
            .then(data => document.getElementById('result').innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>')
            .catch(e => document.getElementById('result').innerHTML = 'Error: ' + e);
    }
    
    function testStep2() {
        fetch('/test/pipeline')
            .then(r => r.json())
            .then(data => document.getElementById('result').innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>')
            .catch(e => document.getElementById('result').innerHTML = 'Error: ' + e);
    }
    
    function testStep3() {
        fetch('/test/detection')
            .then(r => r.json())
            .then(data => document.getElementById('result').innerHTML = '<pre>' + JSON.stringify(data, null, 2) + '</pre>')
            .catch(e => document.getElementById('result').innerHTML = 'Error: ' + e);
    }
    
    function testStep4() {
        fetch('/test/template')
            .then(r => r.text())
            .then(data => document.getElementById('result').innerHTML = data)
            .catch(e => document.getElementById('result').innerHTML = 'Error: ' + e);
    }
    </script>
    """

@app.route('/test/imports')
def test_imports():
    try:
        results = []
        
        # Test basic imports
        import numpy as np
        results.append("✓ numpy imported")
        
        import pandas as pd
        results.append("✓ pandas imported")
        
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        results.append("✓ matplotlib imported")
        
        # Test TCC imports
        from main import TCCPipeline
        results.append("✓ TCCPipeline imported")
        
        from src.tcc_detector import TCCDetector
        results.append("✓ TCCDetector imported")
        
        from src.tcc_tracker import TCCTracker
        results.append("✓ TCCTracker imported")
        
        return jsonify({"status": "success", "results": results})
        
    except Exception as e:
        return jsonify({"status": "error", "error": str(e), "traceback": traceback.format_exc()})

@app.route('/test/pipeline')
def test_pipeline():
    try:
        from main import TCCPipeline
        
        pipeline = TCCPipeline(output_directory='simple_test_output')
        
        return jsonify({
            "status": "success", 
            "message": "Pipeline initialized successfully",
            "output_dir": str(pipeline.output_directory)
        })
        
    except Exception as e:
        return jsonify({"status": "error", "error": str(e), "traceback": traceback.format_exc()})

@app.route('/test/detection')
def test_detection():
    try:
        from main import TCCPipeline
        
        pipeline = TCCPipeline(output_directory='simple_test_output')
        
        # Run with minimal parameters
        results = pipeline.run_detection_demo(use_synthetic=True, num_time_steps=2)
        
        # Convert results to JSON-safe format
        safe_results = {
            "metadata": results.get("metadata", {}),
            "track_count": len(results.get("tracks", {})),
            "track_summary_rows": len(results.get("track_summary", [])),
            "files": results.get("files", {})
        }
        
        return jsonify({"status": "success", "results": safe_results})
        
    except Exception as e:
        return jsonify({"status": "error", "error": str(e), "traceback": traceback.format_exc()})

@app.route('/test/template')
def test_template():
    try:
        # Test simple template rendering
        from flask import render_template_string
        
        test_template = """
        <div class="alert alert-success">
            <h4>Template Test Successful!</h4>
            <p>Metadata test: {{ metadata.num_tccs }}</p>
            <p>Time span test: {{ metadata.get('time_span_hours', 0) }}h</p>
        </div>
        """
        
        test_metadata = {
            "num_tccs": 5,
            "time_span_hours": 2.5
        }
        
        result = render_template_string(test_template, metadata=test_metadata)
        return result
        
    except Exception as e:
        return f"<div class='alert alert-danger'>Template Error: {str(e)}<br><pre>{traceback.format_exc()}</pre></div>"

if __name__ == '__main__':
    os.makedirs('simple_test_output', exist_ok=True)
    print("Starting simple test server on http://localhost:5002")
    app.run(debug=True, port=5002) 