#!/bin/sh
# Per-deploy release command (deploy/fly.api.toml). Idempotent and safe to re-run.
# Creates the two dataset databases if missing, then applies migrations to each.
# Does NOT seed or embed (those are one-time manual steps — seeding wipes data).
set -e

echo "→ ensuring dataset databases exist"
uv run python deploy/createdbs.py

echo "→ migrating msmarco database"
uv run alembic -x dataset=msmarco upgrade head

echo "→ migrating fiqa database"
uv run alembic -x dataset=fiqa upgrade head

echo "✓ migrations applied to both databases"
