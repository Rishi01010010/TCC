#!/usr/bin/env python3
"""
TCC Detection Web Application Launcher
Run this script to start the web interface for TCC detection and analysis.
"""

import sys
import os
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'flask', 'numpy', 'pandas', 'matplotlib', 
        'scikit-image', 'scipy', 'scikit-learn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies"""
    print("🔧 Installing required dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['templates', 'static/css', 'static/js', 'web_output']
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("📁 Created necessary directories")

def start_web_app():
    """Start the Flask web application"""
    print("🚀 Starting TCC Detection Web Application...")
    print("=" * 50)
    
    # Check if app.py exists
    if not os.path.exists('app.py'):
        print("❌ Error: app.py not found!")
        print("Please ensure you're running this script from the project directory.")
        return False
    
    # Create directories
    create_directories()
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"⚠️  Missing dependencies: {', '.join(missing)}")
        print("Installing dependencies...")
        if not install_dependencies():
            return False
    
    print("✅ All dependencies are available!")
    print("\n🌐 Starting web server...")
    print("📍 The application will be available at: http://localhost:5000")
    print("\n💡 Features available:")
    print("   • Interactive TCC Detection")
    print("   • Real-time Dashboard")
    print("   • Machine Learning Training")
    print("   • Data Export & Visualization")
    print("\n🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Wait a moment and open browser
    def open_browser():
        time.sleep(2)
        try:
            webbrowser.open('http://localhost:5000')
            print("\n🌐 Opening web browser...")
        except:
            print("\n💻 Please open http://localhost:5000 in your browser")
    
    # Start browser opening in background
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start Flask app
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print("\n\n🛑 Web application stopped by user")
        return True
    except Exception as e:
        print(f"\n❌ Error starting web application: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure all files are in place (app.py, templates/, etc.)")
        print("2. Check that port 5000 is not already in use")
        print("3. Verify Python version is 3.8+")
        return False

def main():
    """Main function"""
    print("🛰️  TCC Detection System - Web Interface Launcher")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        return
    
    # Check if we're in the right directory
    required_files = ['app.py', 'main.py', 'requirements.txt']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   • {file}")
        print("\nPlease run this script from the TCC project directory.")
        return
    
    # Start the web application
    if start_web_app():
        print("\n✅ Web application session ended successfully!")
    else:
        print("\n❌ Web application failed to start!")

if __name__ == "__main__":
    main() 