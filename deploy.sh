#!/bin/bash

# Production deployment script for bikeshare analytics assistant
set -e

echo "Starting production deployment..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found. Please copy env.example to .env and configure your values."
    exit 1
fi

# Load environment variables
source .env

# Validate required environment variables
required_vars=("POSTGRES_DB" "POSTGRES_USER" "POSTGRES_PASSWORD" "POSTGRES_HOST" "OPENAI_API_KEY")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "ERROR: $var is not set in .env file"
        exit 1
    fi
done

echo "Environment variables validated"

# Build the Docker image
echo "Building Docker image..."
docker build -t bikeshare-analytics:latest .

# Run security scan (if trivy is available)
if command -v trivy &> /dev/null; then
    echo "Running security scan..."
    trivy image bikeshare-analytics:latest --severity HIGH,CRITICAL
else
    echo "WARNING: Trivy not found. Skipping security scan."
fi

# Stop existing containers
echo "Stopping existing containers..."
docker compose down

# Start services
echo "Starting services..."
docker compose up -d

# Wait for database to be ready
echo "Waiting for database to be ready..."
timeout=60
counter=0
while [ $counter -lt $timeout ]; do
    if docker compose exec -T db pg_isready -U $POSTGRES_USER -d $POSTGRES_DB > /dev/null 2>&1; then
        echo "Database is ready"
        break
    fi
    sleep 1
    counter=$((counter + 1))
done

if [ $counter -eq $timeout ]; then
    echo "ERROR: Database failed to start within $timeout seconds"
    docker compose logs db
    exit 1
fi

# Wait for API to be ready
echo "Waiting for API to be ready..."
timeout=60
counter=0
while [ $counter -lt $timeout ]; do
    if curl -f http://localhost:8000/ready > /dev/null 2>&1; then
        echo "API is ready"
        break
    fi
    sleep 2
    counter=$((counter + 2))
done

if [ $counter -eq $timeout ]; then
    echo "ERROR: API failed to start within $timeout seconds"
    docker compose logs api
    exit 1
fi

# Test health endpoints
echo "Testing health endpoints..."
if curl -f http://localhost:8000/ping > /dev/null 2>&1; then
    echo "Health check passed"
else
    echo "ERROR: Health check failed"
    exit 1
fi

if curl -f http://localhost:8000/ready > /dev/null 2>&1; then
    echo "Readiness check passed"
else
    echo "ERROR: Readiness check failed"
    exit 1
fi

echo "Deployment completed successfully!"
echo "API is available at: http://localhost:8000"
echo "API documentation at: http://localhost:8000/docs"
echo ""
echo "Useful commands:"
echo "  docker compose logs -f api    # View API logs"
echo "  docker compose logs -f db     # View database logs"
echo "  docker compose down           # Stop all services"
echo "  docker compose restart api    # Restart API only"
