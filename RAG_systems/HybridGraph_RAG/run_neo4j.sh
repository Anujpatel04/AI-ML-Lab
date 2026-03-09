#!/usr/bin/env bash
# Start Neo4j in Docker for Hybrid Graph RAG. Uses password from repo root .env (NEO4J_PASSWORD).
# Stop: docker stop neo4j
# Requires Docker Desktop (or Docker daemon) to be running.

set -e
if ! docker info &>/dev/null; then
  echo "Docker is not running. Start Docker Desktop (open the Docker app), then run this script again."
  exit 1
fi
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# Load NEO4J_PASSWORD from .env if present
if [ -f "$REPO_ROOT/.env" ]; then
  export $(grep -E '^NEO4J_PASSWORD=' "$REPO_ROOT/.env" | xargs)
fi
NEO4J_PASSWORD="${NEO4J_PASSWORD:-password}"

echo "Starting Neo4j (auth neo4j/$NEO4J_PASSWORD)..."
docker rm -f neo4j 2>/dev/null || true
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/"$NEO4J_PASSWORD" \
  neo4j:latest

echo "Waiting for Neo4j to be ready (bolt 7687)..."
for i in $(seq 1 30); do
  if nc -z localhost 7687 2>/dev/null || true; then
    echo "Neo4j is ready. Browser UI: http://localhost:7474  Bolt: bolt://localhost:7687"
    exit 0
  fi
  sleep 2
done
echo "Neo4j may still be starting. Check: docker logs neo4j"
