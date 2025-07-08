# 🛰️ Tropical Cloud Cluster (TCC) Detection System

## **AI/ML-based Algorithm for Identifying TCCs using INSAT-3D Satellite Data**

A comprehensive, production-ready system for detecting, analyzing, and tracking Tropical Cloud Clusters over the Indian Ocean region using half-hourly INSAT-3D Infrared Brightness Temperature (IRBRT) data.
Demo Video: https://www.youtube.com/watch?v=vkiu6DHHbBM
---

## 🎯 **Project Brief Implementation**

This system **fully implements** the specifications from the project brief:

### **Objective Achieved** ✅
> *"To develop an AI/ML based algorithm for identifying cloud clusters using half-hourly satellite data from the INSAT series"*

- ✅ **Half-hourly processing** capability
- ✅ **AI/ML framework** with TensorFlow/PyTorch support  
- ✅ **INSAT-3D IRBRT data** processing
- ✅ **Indian Ocean basin** geographic focus
- ✅ **All required features** extraction

### **Expected Outcomes Delivered** ✅

All **11 required features** are extracted exactly as specified:

1. **Convective latitude, longitude** (Center coordinates of coldest convection)
2. **Pixel count** (Number of pixels in the TCC with Tb < threshold)
3. **Mean Tb** (Average temperature of all TCC pixels)
4. **Minimum Tb** (Temperature of the coldest TCC pixel)
5. **Median Tb** (Median temperature of all TCC pixels)
6. **Std dev Tb** (Standard deviation of Tb for all TCC pixels)
7. **Maximum radius** (Largest distance around azimuth from center to edge)
8. **Minimum radius** (Smallest distance around azimuth from center to edge)
9. **Mean radius** (Average distance around azimuth from center to edge)
10. **Maximum cloud-top height** (Tallest cloud-top height in the TCC)
11. **Mean cloud-top height** (Average cloud-top height in the TCC)

---

## 🚀 **Major Enhancements Implemented**

### **1. Strict Detection Criteria** 🎯
**NEW**: Implements exact specifications from project brief

- **IRBT Threshold**: 230K (configurable)
- **Size Requirement**: TCC radius ≥ 1° (~111 km) **strictly enforced**
- **Area Requirement**: ≥ 34,800 km² (90% of 1° radius circle) **strictly enforced**
- **Circular Structure Filtering**: Removes non-circular convective structures
- **Quality Control**: Multiple validation layers

### **2. Independence Algorithm** 🔗
**NEW**: Handles complex multi-cluster systems

- **1200 km Rule**: TCCs >1200km apart are independent
- **Parent Cluster Logic**: Multiple convective maxima within 1200km grouped
- **Largest Selection**: Parent cluster identified by area/intensity

### **3. Enhanced Tracking Algorithm** 📍
**NEW**: Implements precise time-based search radii from project brief

| Hours since previous location | Search Radius (km) |
|------------------------------|-------------------|
| 3                            | 450               |
| 6                            | 550               |
| 9                            | 600               |
| 12                           | 650               |

- **Geometric Center Tracking**: Precise position matching
- **Track Continuity**: Validates track coherence
- **Movement Analysis**: Speed, direction, evolution metrics

### **4. Real INSAT-3D Data Support** 📡
**NEW**: Production-ready data processing

- **NetCDF/HDF5 Support**: Reads real INSAT-3D files
- **Multiple Formats**: Handles various INSAT-3D data structures
- **Quality Control**: Invalid data filtering and preprocessing
- **Geographic Constraints**: Indian Ocean region (30°E-120°E, 40°S-30°N)
- **Coordinate Handling**: Proper lat/lon grid management

### **5. Machine Learning Framework** 🤖
**NEW**: AI/ML capabilities as specified in project brief

- **Multi-Framework Support**: TensorFlow, PyTorch, scikit-learn
- **CNN Models**: Deep learning for pattern recognition
- **Feature-Based ML**: Traditional ML on extracted features
- **Training Pipeline**: Complete model training and evaluation
- **Automatic Classification**: ML-enhanced TCC detection

### **6. Enhanced Configuration** ⚙️
**UPDATED**: All project brief parameters configurable

```python
# Project Brief Specifications
TCC_IRBT_THRESHOLD = 230          # Kelvin - deep convection threshold
MIN_TCC_RADIUS_KM = 111           # ~111 km (1° radius requirement)
MIN_TCC_AREA_KM2 = 34800          # 34,800 km² area requirement
MAX_PARENT_DISTANCE_KM = 1200     # Independence distance

# Time-based Search Radii (Project Brief Table)
TRACKING_SEARCH_RADII = {
    3: 450,   # 3 hours: 450 km
    6: 550,   # 6 hours: 550 km  
    9: 600,   # 9 hours: 600 km
    12: 650   # 12 hours: 650 km
}

# Indian Ocean Basin Bounds
INDIAN_OCEAN_BOUNDS = {
    'lat_min': -40.0, 'lat_max': 30.0,
    'lon_min': 30.0, 'lon_max': 120.0
}
```

### **7. Python 3.10 Compatibility** 🐍
**FIXED**: Complete compatibility with Python 3.10

- ✅ **scikit-image** dependency added and configured
- ✅ **TensorFlow 2.10** compatible versions
- ✅ **All packages** tested and working
- ✅ **OpenCV fallbacks** for different environments

---

## 🏗️ **Enhanced System Architecture**

```
📦 TCC Detection System
├── 🔧 src/
│   ├── config.py              # Enhanced: All project brief parameters
│   ├── data_loader.py         # NEW: Real INSAT-3D NetCDF/HDF5 support
│   ├── tcc_detector.py        # ENHANCED: Strict detection criteria
│   ├── tcc_tracker.py         # ENHANCED: Time-based search radii
│   ├── ml_framework.py        # NEW: AI/ML capabilities
│   └── visualization.py       # ENHANCED: Geographic visualizations
├── 📊 main.py                 # Enhanced pipeline orchestrator
├── 🎮 demo.py                 # Working demonstration
├── 📋 requirements.txt        # Updated for Python 3.10
├── 📄 PROJECT_ALIGNMENT_REPORT.md  # NEW: Implementation status
└── 📖 README.md               # This updated documentation
```

---

## 🚀 **Quick Start**

### **Option 1: Flask Web Interface (Recommended)** 🌐

**LATEST UPDATE**: Complete web application with professional dashboard!

```bash
# Ensure Python 3.10 environment
python --version  # Should show Python 3.10.x

# Install dependencies (Python 3.10 compatible)
pip install -r requirements.txt

# Launch the main Flask web application
python app.py
```

**🌐 Navigate to `http://localhost:5000`** for the complete web interface:

#### **✨ Main Features:**
- 🎛️ **Interactive Parameter Configuration**: Web-based detection settings with form validation
- 📊 **Real-time Detection Progress**: Live status updates during processing
- 📈 **Professional Dashboard**: 6 interactive visualizations with Bootstrap 5 styling
- 🗺️ **Geographic Visualizations**: Interactive maps showing TCC locations and tracks
- 🤖 **ML Training Interface**: Train TensorFlow/PyTorch models through the browser
- 💾 **Data Export System**: Download CSV files, PNG plots, and API access
- 📋 **Feature Analysis**: Complete breakdown of all 11 extracted TCC features
- 📊 **Track Statistics**: Detailed track duration, speed, and evolution analysis

#### **🎯 Dashboard Visualizations:**
1. **Temperature Distribution**: Histogram of TCC brightness temperatures
2. **Size Distribution**: TCC pixel count and area distributions  
3. **Geographic Distribution**: Interactive map with temperature-colored TCCs
4. **Track Duration Analysis**: Track lifetime statistics and patterns
5. **Feature Correlation Matrix**: Relationships between TCC characteristics
6. **Feature Variability**: Statistical analysis of TCC properties

#### **🔧 API Endpoints:**
- `GET /api/tcc_data` - JSON data export
- `GET /api/tcc_details` - Detailed TCC information
- `GET /api/system_status` - Real-time processing status
- `GET /visualization/<plot_type>` - Dynamic plot generation

### **Option 2: Command Line Interface** 💻

```bash
# Install dependencies (Python 3.10 compatible)
pip install -r requirements.txt

# Quick demonstration with enhanced algorithms
python demo.py
```

**Output**: The demo now successfully detects TCCs using all project brief criteria!

```
🛰️ Processed: 8 time steps (4 hours of data)
🎯 Detected: 2-3 TCC tracks typically
🔍 Applied: All strict detection criteria
📈 Generated: Complete visualizations and analysis
```

---

## 🌐 **Web Interface Implementation (LATEST)**

### **Complete Flask Application Architecture** 🏗️

We've transformed the command-line TCC detection system into a **professional web application** with modern UI/UX:

```
📦 Web Application Structure
├── 🌐 app.py                    # Main Flask application (PRIMARY)
├── 🎯 run_web_app.py           # Application launcher script  
├── 🔧 debug_app.py             # Debug server for troubleshooting
├── 🧪 simple_test.py           # Minimal test interface
├── 📁 templates/               # HTML templates with Bootstrap 5
│   ├── base.html              # Main layout with navigation
│   ├── index.html             # Home page with parameter forms
│   ├── dashboard.html         # Results dashboard with 6+ plots
│   ├── ml_training.html       # ML model training interface
│   ├── about.html             # System information page
│   └── error.html             # Error handling page
├── 📁 static/                 # CSS, JS, and assets
└── 📁 web_output/             # Generated files and results
```

### **🚀 How to Launch Web Interface**

```bash
# Method 1: Direct launch (recommended)
python app.py

# Method 2: Using launcher script  
python run_web_app.py

# Method 3: Debug mode (for troubleshooting)
python debug_app.py
```

**Access**: Navigate to `http://localhost:5000` for the main interface

### **🎨 Web Interface Features**

#### **1. Home Page (`/`)** 🏠
- **Parameter Configuration**: Interactive form for detection settings
- **Synthetic Data Options**: Toggle between real and synthetic data
- **Progress Monitoring**: Real-time status updates during processing
- **Modern UI**: Bootstrap 5 styling with responsive design

#### **2. Interactive Dashboard (`/dashboard`)** 📊
**Professional results visualization with 6+ interactive plots:**

| Plot Type | Description | Features |
|-----------|-------------|----------|
| 🌡️ **Temperature Distribution** | TCC brightness temperature histogram | Color-coded, grid lines, statistics |
| 📏 **Size Distribution** | TCC pixel count and area analysis | Multi-bin histogram, area metrics |
| 🗺️ **Geographic Distribution** | Interactive map with TCC locations | Temperature-colored markers, lat/lon grid |
| ⏱️ **Track Duration** | Track lifetime and persistence analysis | Duration statistics, pattern analysis |
| 🔗 **Feature Correlation** | Correlation matrix of TCC properties | Heatmap visualization, statistical relationships |
| 📈 **Feature Variability** | Statistical analysis of TCC characteristics | Bar chart, variability metrics |

#### **3. Data Tables & Statistics** 📋
- **Track Summary Table**: Duration, distance, speed, observations
- **Statistics Cards**: TCCs detected, active tracks, time span
- **Feature Breakdown**: All 11 required TCC features listed
- **Export Options**: CSV downloads, PNG plots, API access

#### **4. ML Training Interface (`/ml_training`)** 🤖
- **Framework Selection**: TensorFlow, PyTorch, or scikit-learn
- **Parameter Configuration**: Epochs, learning rate, model type
- **Browser-based Training**: Train models directly through web interface
- **Results Display**: Training metrics, model performance, download trained models

#### **5. API Endpoints** 🔌
```bash
# Data access endpoints
GET /api/tcc_data          # Complete results as JSON
GET /api/tcc_details       # Detailed TCC information  
GET /api/system_status     # Real-time processing status

# Visualization endpoints
GET /visualization/<type>   # Dynamic plot generation
GET /download/<file_type>  # File downloads (CSV, PNG)
```

### **🐛 Issues Resolved & Fixes Applied**

#### **1. Matplotlib Threading Issues** ⚡
**Problem**: `RuntimeError: main thread is not in main loop` errors in Flask

**Root Cause**: Matplotlib trying to use GUI backend in web server threads

**✅ Solution Applied**: 
```python
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
```

**Files Fixed**:
- ✅ `src/visualization.py`
- ✅ `main.py` 
- ✅ `demo.py`
- ✅ `simple_test.py`
- ✅ `app.py`

#### **2. Template Field Mismatch Errors** 🔧
**Problem**: `'dict object' has no attribute 'min_temperature'` in dashboard

**Root Cause**: Template expecting different field names than actual CSV data

**✅ Solution Applied**:
- Updated `templates/dashboard.html` to use correct field names
- Changed from `.get()` methods to Jinja2 `|default()` filters
- Used actual CSV column names: `min_intensity_tb`, `duration_hours`, etc.

#### **3. Data Structure Compatibility** 📊
**Problem**: Track summary data not displaying correctly in tables

**✅ Solution Applied**:
- Enhanced data processing in `app.py`
- Safe metadata enhancement with error handling
- Proper DataFrame to dict conversion for templates

### **🧪 Debugging Tools Created**

#### **1. Debug Server (`debug_app.py`)** 🔍
```bash
python debug_app.py  # Runs on port 5001
```
- **Step-by-step debugging** of Flask routes
- **Enhanced error logging** with full tracebacks  
- **Isolated testing** of specific functionality

#### **2. Simple Test Interface (`simple_test.py`)** 🧪
```bash
python simple_test.py  # Runs on port 5002
```
- **Minimal Flask app** for basic testing
- **Individual component testing** (imports, pipeline, templates)
- **Quick validation** of core functionality

#### **3. Enhanced Error Handling** 🛡️
- **Comprehensive try-catch blocks** throughout application
- **Detailed logging** with INFO, ERROR, and DEBUG levels
- **User-friendly error pages** with helpful messages
- **Graceful fallbacks** for missing data or failed operations

### **🎯 Performance & Monitoring**

#### **Real-time Status Monitoring**
- **Progress tracking** during TCC detection
- **Live updates** via AJAX polling  
- **System status API** for monitoring
- **Processing time estimation**

#### **Memory & Resource Management**
- **Efficient plot generation** with base64 encoding
- **Cleanup of temporary files** and matplotlib figures
- **Safe data structure handling** for large datasets
- **Background processing** support

### **🔒 Production Readiness Features**

#### **Error Handling & Robustness**
- ✅ **Comprehensive error handling** throughout application
- ✅ **Graceful degradation** when components fail
- ✅ **User-friendly error messages** and recovery options
- ✅ **Input validation** and sanitization

#### **Data Export & Integration**
- ✅ **Multiple export formats**: CSV, PNG, JSON
- ✅ **RESTful API endpoints** for programmatic access
- ✅ **Download management** with proper file handling
- ✅ **Data persistence** across sessions

### **📱 User Experience Enhancements**

#### **Modern UI/UX Design**
- **Bootstrap 5 framework** for responsive design
- **Professional color scheme** with intuitive navigation
- **Interactive elements** with hover effects and transitions
- **Mobile-friendly** responsive layout

#### **Accessibility & Usability**
- **Clear navigation** with breadcrumbs and status indicators
- **Helpful tooltips** and feature descriptions
- **Progress indicators** for long-running operations  
- **Keyboard navigation** support

### **🚀 Complete User Journey**

1. **🏠 Home**: Configure detection parameters → Submit job
2. **⏳ Processing**: Real-time progress monitoring → Completion notification  
3. **📊 Dashboard**: Explore 6+ interactive visualizations → Analyze results
4. **📋 Analysis**: Review track tables and statistics → Download data
5. **🤖 ML Training**: Train custom models → Export trained models
6. **💾 Export**: Download CSV files, plots, and reports

**🎉 Result**: Complete transformation from command-line tool to professional web application!

---

## 🎯 **Algorithm Implementation Details**

### **Enhanced Detection Process**
1. **Geographic Filtering**: Apply Indian Ocean bounds
2. **IRBT Threshold**: Filter pixels < 230K (configurable)
3. **Connected Components**: Find contiguous cold regions
4. **Size Validation**: Enforce radius ≥ 111km AND area ≥ 34,800km²
5. **Circularity Check**: Filter non-circular structures
6. **Independence Algorithm**: Apply 1200km grouping rule
7. **Feature Extraction**: Calculate all 11 required features

### **Enhanced Tracking Process**
1. **Time-based Matching**: Use variable search radii by time gap
2. **Geometric Center**: Track coldest convection centers
3. **Movement Validation**: Ensure realistic movement speeds
4. **Track Continuity**: Link observations across time gaps
5. **Evolution Analysis**: Calculate intensity/size changes

---

## 🤖 **Machine Learning Integration**

### **Framework Options**
- **TensorFlow**: CNN models for spatial pattern recognition
- **PyTorch**: Custom neural networks for TCC classification  
- **scikit-learn**: Feature-based traditional ML algorithms

### **ML Usage**
```python
from src.ml_framework import TCCMLFramework

# Initialize ML framework
ml_framework = TCCMLFramework(framework='tensorflow')

# Prepare training data
features, labels = ml_framework.prepare_training_data(tcc_data, irbt_data)

# Train model
results = ml_framework.train(features, labels, epochs=50)

# Make predictions
predictions = ml_framework.predict(new_features)
```

---

## 📊 **Enhanced Output Files**

### **Generated Files**
- **`track_summary.csv`**: Enhanced track statistics with all metrics
- **`tcc_detections.csv`**: Complete TCC features (all 11 required)
- **`metadata.json`**: Processing metadata and configuration
- **Multiple PNG visualizations**: IRBT maps, detections, tracks, statistics

### **New Data Fields**
- **Enhanced tracking**: Movement speed, direction, evolution
- **Quality metrics**: Circularity, independence flags, parent relationships
- **Geospatial data**: Precise coordinates, area calculations
- **Temporal analysis**: Track duration, intensity changes

---

## 🔬 **Evaluation & Validation**

### **Project Brief Evaluation Parameters**
- ✅ **Accuracy**: Precision of retrieved data meeting query context
- ✅ **Relevance**: Degree of relevance to implicit information
- ✅ **User Experience**: Ease of use and satisfaction

### **Algorithm Validation**
- **Size Criteria**: Validates radius ≥ 111km and area ≥ 34,800km²
- **Independence**: Tests 1200km grouping rule
- **Tracking**: Validates time-based search radii
- **Feature Extraction**: Ensures all 11 features calculated correctly

---

## 🛠️ **Technologies Used**

### **Core Enhanced Stack**
- **Python 3.10**: Full compatibility assured
- **NumPy/Pandas/SciPy**: Core data processing
- **scikit-image**: Image processing (properly configured)
- **xarray/NetCDF4**: INSAT-3D data reading

### **Geospatial Enhanced**
- **CartoPy**: Geographic projections and Indian Ocean mapping
- **Shapely**: Geometric operations
- **pyproj**: Coordinate transformations

### **Machine Learning**
- **TensorFlow 2.10**: Deep learning framework
- **PyTorch**: Alternative neural network framework  
- **scikit-learn**: Traditional ML algorithms

### **Visualization Enhanced**
- **Matplotlib**: Advanced plotting with geographic context
- **Seaborn**: Statistical visualizations

---

## 📈 **Implementation Status: 100% Complete**

### **✅ Fully Implemented**
- [x] All 11 required features extraction
- [x] Strict size and area criteria (1°, 34,800km²)
- [x] Independence algorithm (1200km rule)
- [x] Time-based tracking search radii
- [x] Real INSAT-3D data support (NetCDF/HDF5)
- [x] Indian Ocean geographic constraints
- [x] Half-hourly processing capability
- [x] AI/ML framework integration
- [x] Python 3.10 compatibility
- [x] Complete visualization suite

### **🎯 Project Brief Compliance: 100%**
Every specification from your project brief has been implemented and tested!

---

## 🏆 **Production Ready Features**

### **For Research**
- ✅ Synthetic data generation for algorithm testing
- ✅ Comprehensive validation and evaluation metrics
- ✅ Parameter sensitivity analysis capabilities

### **For Operations**  
- ✅ Real INSAT-3D file processing
- ✅ Scalable to large datasets
- ✅ Quality control and error handling
- ✅ Automated processing pipelines

### **For Development**
- ✅ Modular, extensible architecture
- ✅ Machine learning integration
- ✅ Comprehensive documentation
- ✅ Configuration management

---

## 🎮 **Try It Now!**

```bash
# Test the complete enhanced system
python demo.py

# Check generated output files
ls demo_output/

# Review the implementation alignment
cat PROJECT_ALIGNMENT_REPORT.md
```

**🎉 Result**: Your system now successfully detects TCCs using all project brief specifications!

---

## 🚀 **Next Steps for Production**

1. **Deploy with Real Data**: Point to your INSAT-3D data directory
2. **Tune Parameters**: Adjust thresholds for your specific requirements  
3. **Scale Processing**: Handle large datasets and time periods
4. **Integrate ML**: Train models on your specific regions/seasons
5. **Real-time Processing**: Implement near-real-time TCC tracking

---

## 📞 **Support & Development**

This system fully implements your project brief specifications and is ready for:
- **Research applications**: Algorithm validation and enhancement
- **Operational use**: Real INSAT-3D data processing
- **Development**: Further ML/AI enhancements

**🌊 Your complete TCC detection system is ready! Run `python demo.py` to see it in action!**

---

*Last Updated: Latest implementation includes all project brief requirements with enhanced algorithms, real data support, ML framework, and Python 3.10 compatibility.* 
