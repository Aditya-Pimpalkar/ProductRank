"""Create the per-dataset databases if they don't exist (idempotent).

Run before Alembic on a fresh ParadeDB instance. Connects to the maintenance `postgres`
database and issues CREATE DATABASE for each dataset's dbname (CREATE DATABASE cannot run
inside a transaction, hence autocommit). Safe to run on every deploy.
"""

from __future__ import annotations

import psycopg

from productrank.config import DATASETS, settings


def main() -> None:
    admin_dsn = settings._url("postgres").replace("postgresql+psycopg://", "postgresql://")
    for ds in DATASETS:
        name = settings.dataset_dbnames[ds]
        with psycopg.connect(admin_dsn, autocommit=True) as conn:
            exists = conn.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (name,)
            ).fetchone()
            if exists:
                print(f"  database {name!r} already exists")
            else:
                conn.execute(f'CREATE DATABASE "{name}"')
                print(f"  created database {name!r}")


if __name__ == "__main__":
    main()
