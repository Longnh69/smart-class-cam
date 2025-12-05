#!/bin/bash

echo "============================================"
echo "Smart Classroom Attendance System"
echo "Setup Script v1.0"
echo "============================================"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8-3.11"
    exit 1
fi

echo "[1/6] Checking Python version..."
python3 --version
echo ""

# Create virtual environment
echo "[2/6] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment"
        exit 1
    fi
    echo "Done!"
fi
echo ""

# Activate virtual environment
echo "[3/6] Activating virtual environment..."
source venv/bin/activate
echo ""

# Upgrade pip
echo "[4/6] Upgrading pip..."
python -m pip install --upgrade pip --quiet
echo "Done!"
echo ""

# Install PyTorch with CUDA
echo "[5/6] Installing PyTorch with CUDA 12.1..."
echo "This may take 5-10 minutes depending on your internet speed..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if [ $? -ne 0 ]; then
    echo "WARNING: PyTorch installation failed. GPU may not work."
    echo "Continuing with CPU-only setup..."
fi
echo ""

# Install other requirements
echo "[6/6] Installing dependencies from requirements.txt..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install requirements"
    exit 1
fi
echo ""

# Test installation
echo "============================================"
echo "Testing Installation"
echo "============================================"
echo ""

echo "Testing PyTorch CUDA..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "WARNING: PyTorch test failed"
fi
echo ""

echo "Testing ONNX Runtime..."
python -c "import onnxruntime as ort; providers = ort.get_available_providers(); print('Providers:', providers); print('GPU Support:', 'CUDAExecutionProvider' in providers)" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "WARNING: ONNX Runtime test failed"
fi
echo ""

echo "Testing OpenCV..."
python -c "import cv2; print('OpenCV version:', cv2.__version__)" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "WARNING: OpenCV test failed"
fi
echo ""

echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "To activate the environment in the future, run:"
echo "    source venv/bin/activate"
echo ""
echo "To start the program:"
echo "    cd src"
echo "    python main.py"
echo ""
echo "See README.md for usage instructions."
echo ""