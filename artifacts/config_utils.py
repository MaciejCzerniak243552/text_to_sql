import os
import re
import urllib.parse
from typing import Optional

import streamlit as st


# Lightweight .env loader to avoid adding a dependency.
def load_dotenv_file(path: str = ".env", override: bool = False) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ, honoring overrides."""
    # Attempt to open the .env file; missing files are treated as optional.
    try:
        # Read the file in text mode so we can normalize whitespace and quotes.
        with open(path, "r", encoding="utf-8") as handle:
            # Iterate line-by-line to support comments and whitespace trimming.
            for raw_line in handle:
                # Normalize the line so we can parse key/value content.
                line = raw_line.strip()
                # Skip blank lines and comment-only entries.
                if not line or line.startswith("#"):
                    continue
                # Split on the first "=" so values can still include "=" characters.
                key, sep, value = line.partition("=")
                # Ignore malformed lines without a key=value separator.
                if not sep:
                    continue
                # Strip extra whitespace around the key/value.
                key = key.strip()
                value = value.strip()
                # Remove surrounding quotes when values are quoted in .env files.
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]
                # Preserve existing environment values unless override is requested.
                if not override and key in os.environ:
                    continue
                # Persist the parsed key/value into the process environment.
                os.environ[key] = value
    except FileNotFoundError:
        # Silently ignore missing .env files to keep local setup optional.
        pass


# Prefer Streamlit secrets; fall back to environment variables.
def get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    """Resolve config from st.secrets first, then environment variables."""
    try:
        # Use Streamlit secrets when available so we support deployed configs.
        if key in st.secrets:
            return st.secrets.get(key)
    except Exception:
        # If secrets are unavailable, fall back to the environment below.
        pass
    # Environment variables provide local defaults and container overrides.
    return os.getenv(key, default)


# Build a SQLAlchemy URL from DB_URL or individual settings.
def build_db_url() -> Optional[str]:
    """Return a SQLAlchemy connection URL or None if settings are incomplete."""
    # Prefer a fully specified DB_URL when provided.
    db_url = get_setting("DB_URL")
    if db_url:
        return db_url

    dialect = get_setting("DB_DIALECT", "sqlite")
    # SQLite uses a file path instead of host/user/password.
    if dialect.startswith("sqlite"):
        return get_setting("SQLITE_PATH", "sqlite:///testdb.sqlite")

    # Gather the discrete connection parts for non-SQLite engines.
    host = get_setting("DB_HOST")
    user = get_setting("DB_USER")
    password = get_setting("DB_PASSWORD", "")
    name = get_setting("DB_NAME")
    port = get_setting("DB_PORT", "")

    # Bail out if required settings are missing.
    if not host or not user or not name:
        return None

    # URL-encode credentials so special characters are safe.
    user_enc = urllib.parse.quote_plus(user)
    pass_enc = urllib.parse.quote_plus(password) if password else ""
    # Build the auth and host/port segments in SQLAlchemy URL format.
    auth = f"{user_enc}:{pass_enc}@" if password else f"{user_enc}@"
    hostport = f"{host}:{port}" if port else host
    return f"{dialect}://{auth}{hostport}/{name}"


# Redact passwords before printing.
def mask_db_url(url: str) -> str:
    """Hide DB password when showing the URL in the UI."""
    # Replace any password-like segment between ":" and "@".
    return re.sub(r":([^:@/]+)@", ":****@", url)
