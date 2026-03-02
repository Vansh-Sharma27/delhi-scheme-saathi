#!/bin/bash
# Seed the local PostgreSQL database with scheme data
# Run this after docker compose up

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_DIR}/data"

# Wait for postgres to be ready
echo "Waiting for PostgreSQL to be ready..."
until docker exec dss-postgres pg_isready -U postgres > /dev/null 2>&1; do
    sleep 1
done
echo "PostgreSQL is ready!"

# Run the Python seed script
echo "Seeding database with scheme data..."
docker exec -i dss-app python -m scripts.seed_data

echo "Database seeding complete!"
echo ""
echo "Verification:"
docker exec dss-postgres psql -U postgres -d delhi_scheme_saathi -c "SELECT id, name FROM schemes;"
