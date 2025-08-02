#!/bin/bash

# Remote Class Execution Demo Runner

echo "Remote Class Execution Demo"
echo "=========================="
echo ""

# Check if remotemedia is installed
if ! python -c "import remotemedia" 2>/dev/null; then
    echo "⚠️  remotemedia package not found. Installing..."
    cd ../..
    pip install -e .
    cd remote_media_processing_example/remote_class_execution_demo
fi

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Check if server is running
echo ""
echo "Checking if remote server is running..."
if ! nc -z localhost 50052 2>/dev/null; then
    echo "❌ Remote server not running on localhost:50052"
    echo ""
    echo "Please start the server in another terminal:"
    echo "  cd ../../remote_service"
    echo "  python src/server.py"
    echo ""
    echo "Or using Docker:"
    echo "  cd ../../remote_service"
    echo "  docker-compose up"
    exit 1
fi

echo "✅ Remote server is running"
echo ""

# Run the demo
python remote_execution_client.py