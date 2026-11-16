"""Read-only connection factory for bundled simulation facts."""

import sqlite3
from pathlib import Path

from agent.tools.common import configure_connection


def connection_factory(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Customer service database missing: {path}")
    def connect() -> sqlite3.Connection:
        uri = f"file:{path.resolve()}?mode=ro"
        return configure_connection(sqlite3.connect(uri, uri=True))
    return connect
