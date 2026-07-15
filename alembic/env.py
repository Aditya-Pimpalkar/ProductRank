"""Alembic environment.

Connection settings and target metadata both come from the application itself
(productrank.config + productrank.models), so migrations never drift from the app's
single source of truth.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Import models so their tables register on Base.metadata.
from productrank import models  # noqa: F401
from productrank.config import settings
from productrank.db import Base

config = context.config

# Target a specific dataset's database with `-x dataset=fiqa` (defaults to the app's
# default dataset). This applies migrations to each database in the
# two-databases-in-one-instance setup.
_x = context.get_x_argument(as_dictionary=True)
_dataset = _x.get("dataset")
DB_URL = settings.database_url_for(_dataset) if _dataset else settings.database_url
config.set_main_option("sqlalchemy.url", DB_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=DB_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = DB_URL
    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
