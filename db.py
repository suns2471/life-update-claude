"""
db.py – Database helpers for the LifeUpdate app.

Supports both SQLite (local development) and PostgreSQL (production).
Set the DATABASE_URL environment variable to a postgres:// URL for
production; otherwise falls back to a local SQLite file.
"""

import json
import hashlib
import os
import re
from pathlib import Path

# ── Connection setup ─────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Detect which database engine to use
_USE_POSTGRES = DATABASE_URL.startswith("postgres")

if _USE_POSTGRES:
    import psycopg2
    import psycopg2.extras  # for RealDictCursor

    def get_connection():
        """Return a new PostgreSQL connection with dict-style rows."""
        url = DATABASE_URL
        # Railway/Render sometimes use postgres:// but psycopg2 needs postgresql://
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        conn = psycopg2.connect(url)
        return conn

    def _dict_cursor(conn):
        return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # PostgreSQL uses %s for placeholders
    _PH = "%s"
else:
    import sqlite3

    DB_PATH = Path(__file__).parent / "catchup.db"

    def get_connection():
        """Return a new SQLite connection with dict-style rows."""
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=MEMORY")
        conn.execute("PRAGMA synchronous=OFF")
        return conn

    def _dict_cursor(conn):
        return conn.cursor()

    # SQLite uses ? for placeholders
    _PH = "?"


def _ph(count=1):
    """Return comma-separated placeholders for the active database engine."""
    return ", ".join([_PH] * count)


def _row_to_dict(row):
    """Convert a database row to a plain dict regardless of engine."""
    if row is None:
        return None
    if isinstance(row, dict):
        return row
    return dict(row)


# ── Table creation ───────────────────────────────────────────────
def init_db():
    """Create the tables if they don't already exist."""
    conn = get_connection()
    c = _dict_cursor(conn) if _USE_POSTGRES else conn.cursor()

    if _USE_POSTGRES:
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id            SERIAL PRIMARY KEY,
                email         TEXT   UNIQUE NOT NULL,
                name          TEXT   NOT NULL,
                password_hash TEXT   NOT NULL,
                username      TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS contacts (
                id           SERIAL PRIMARY KEY,
                user_id      INTEGER NOT NULL REFERENCES users(id),
                "Name"       TEXT    NOT NULL,
                "Category"   TEXT    DEFAULT 'Uncategorized',
                extra_fields TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS contact_interactions (
                id         SERIAL PRIMARY KEY,
                contact_id INTEGER NOT NULL REFERENCES contacts(id) ON DELETE CASCADE,
                user_id    INTEGER NOT NULL REFERENCES users(id),
                date       TEXT    NOT NULL,
                note       TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS life_journal (
                id      SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                "Date"  TEXT    NOT NULL,
                entry1  TEXT,
                entry2  TEXT,
                entry3  TEXT,
                UNIQUE(user_id, "Date")
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS work_journal (
                id      SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id),
                "Date"  TEXT    NOT NULL,
                entry1  TEXT,
                entry2  TEXT,
                entry3  TEXT,
                UNIQUE(user_id, "Date")
            )
        """)
    else:
        # SQLite — check for old schema and migrate if needed
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if c.fetchone():
            c.execute("PRAGMA table_info(users)")
            cols = [row[1] if not isinstance(row, dict) else row.get("name", row.get("cid")) for row in c.fetchall()]
            if "email" not in cols:
                c.execute("ALTER TABLE users ADD COLUMN email TEXT")
                c.execute("ALTER TABLE users ADD COLUMN name TEXT")
                c.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
                c.execute("UPDATE users SET email = username || '@migrated.local', name = username, password_hash = 'migrated' WHERE email IS NULL")
                conn.commit()
        else:
            c.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    email         TEXT    UNIQUE NOT NULL,
                    name          TEXT    NOT NULL,
                    password_hash TEXT    NOT NULL,
                    username      TEXT
                )
            """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS contacts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL,
                Name         TEXT    NOT NULL,
                Category     TEXT    DEFAULT 'Uncategorized',
                extra_fields TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS contact_interactions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                contact_id INTEGER NOT NULL,
                user_id    INTEGER NOT NULL,
                date       TEXT    NOT NULL,
                note       TEXT,
                FOREIGN KEY(contact_id) REFERENCES contacts(id) ON DELETE CASCADE,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS life_journal (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                Date    TEXT NOT NULL,
                entry1  TEXT,
                entry2  TEXT,
                entry3  TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS work_journal (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                Date    TEXT NOT NULL,
                entry1  TEXT,
                entry2  TEXT,
                entry3  TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)

    conn.commit()
    conn.close()


# ── User helpers ─────────────────────────────────────────────────
def _hash_password(password, salt=None):
    """Hash a password with a random salt using SHA-256."""
    if salt is None:
        salt = os.urandom(16).hex()
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hashed}"


def _verify_password(password, password_hash):
    """Check a password against a stored salt:hash string."""
    salt, stored_hash = password_hash.split(":", 1)
    check = hashlib.sha256((salt + password).encode()).hexdigest()
    return check == stored_hash


def register_user(email, name, password):
    """Create a new user. Returns the user id. Raises ValueError if email exists."""
    conn = get_connection()
    c = _dict_cursor(conn) if _USE_POSTGRES else conn.cursor()
    c.execute(f"SELECT id FROM users WHERE email = {_PH}", (email,))
    if c.fetchone():
        conn.close()
        raise ValueError("An account with this email already exists")
    pw_hash = _hash_password(password)
    if _USE_POSTGRES:
        c.execute(
            f"INSERT INTO users (email, name, password_hash) VALUES ({_ph(3)}) RETURNING id",
            (email, name, pw_hash),
        )
        user_id = c.fetchone()["id"]
    else:
        c.execute(
            f"INSERT INTO users (email, name, password_hash) VALUES ({_ph(3)})",
            (email, name, pw_hash),
        )
        user_id = c.lastrowid
    conn.commit()
    conn.close()
    return user_id


def login_user(email, password):
    """Verify credentials. Returns user id on success, None on failure."""
    conn = get_connection()
    c = _dict_cursor(conn) if _USE_POSTGRES else conn.cursor()
    c.execute(f"SELECT id, password_hash FROM users WHERE email = {_PH}", (email,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    row = _row_to_dict(row)
    if not _verify_password(password, row["password_hash"]):
        return None
    return row["id"]


def get_user_by_id(user_id):
    """Return a dict with {id, name, email} or None if not found."""
    conn = get_connection()
    c = _dict_cursor(conn) if _USE_POSTGRES else conn.cursor()
    c.execute(f"SELECT id, name, email FROM users WHERE id = {_PH}", (user_id,))
    row = c.fetchone()
    conn.close()
    return _row_to_dict(row)


# ── Contacts helpers ─────────────────────────────────────────────
# Column quoting: PostgreSQL needs quotes around capitalized column names
_Q = '"' if _USE_POSTGRES else ''


def get_all_contacts(user_id):
    """Return every contact for this user as a plain dict."""
    conn = get_connection()
    c = _dict_cursor(conn) if _USE_POSTGRES else conn.cursor()
    c.execute(
        f'SELECT * FROM contacts WHERE user_id = {_PH} ORDER BY id',
        (user_id,),
    )
    rows = c.fetchall()
    conn.close()

    result = []
    for row in rows:
        contact = _row_to_dict(row)
        contact.pop("user_id", None)
        extra_json = contact.pop("extra_fields", None)
        if extra_json:
            contact.update(json.loads(extra_json))
        result.append(contact)
    return result


def get_existing_contact_phones(user_id):
    """Return a set of digits-only phone numbers across all contacts."""
    conn = get_connection()
    c = _dict_cursor(conn) if _USE_POSTGRES else conn.cursor()
    c.execute(
        f"SELECT extra_fields FROM contacts WHERE user_id = {_PH}",
        (user_id,),
    )
    rows = c.fetchall()
    conn.close()

    phones = set()
    for row in rows:
        row = _row_to_dict(row)
        extra_json = row["extra_fields"]
        if not extra_json:
            continue
        extra = json.loads(extra_json)
        for key, val in extra.items():
            if key.startswith("Phone") and val:
                digits = re.sub(r"[^\d]", "", val)
                if digits:
                    phones.add(digits)
    return phones


def get_all_categories(user_id):
    """Return a sorted list of unique category names for this user."""
    conn = get_connection()
    c = _dict_cursor(conn) if _USE_POSTGRES else conn.cursor()
    c.execute(
        f'SELECT DISTINCT {_Q}Category{_Q} FROM contacts '
        f'WHERE user_id = {_PH} AND {_Q}Category{_Q} IS NOT NULL ORDER BY {_Q}Category{_Q}',
        (user_id,),
    )
    rows = c.fetchall()
    conn.close()
    return [_row_to_dict(r)["Category"] for r in rows]


def insert_contact(user_id, name, category, extra_fields_dict=None):
    """Add a single contact. Returns the new contact's id."""
    conn = get_connection()
    c = _dict_cursor(conn) if _USE_POSTGRES else conn.cursor()
    extra_json = json.dumps(extra_fields_dict) if extra_fields_dict else None
    if _USE_POSTGRES:
        c.execute(
            f'INSERT INTO contacts (user_id, "Name", "Category", extra_fields) '
            f'VALUES ({_ph(4)}) RETURNING id',
            (user_id, name, category, extra_json),
        )
        contact_id = c.fetchone()["id"]
    else:
        c.execute(
            f"INSERT INTO contacts (user_id, Name, Category, extra_fields) "
            f"VALUES ({_ph(4)})",
            (user_id, name, category, extra_json),
        )
        contact_id = c.lastrowid
    conn.commit()
    conn.close()
    return contact_id


def update_contact_category(contact_id, new_category):
    """Change the category of a contact by its database id."""
    conn = get_connection()
    c = _dict_cursor(conn) if _USE_POSTGRES else conn.cursor()
    c.execute(
        f'UPDATE contacts SET {_Q}Category{_Q} = {_PH} WHERE id = {_PH}',
        (new_category, contact_id),
    )
    conn.commit()
    conn.close()


def get_contact_by_id(contact_id, user_id):
    """Return a single contact as a flat dict, or None."""
    conn = get_connection()
    c = _dict_cursor(conn) if _USE_POSTGRES else conn.cursor()
    c.execute(
        f"SELECT * FROM contacts WHERE id = {_PH} AND user_id = {_PH}",
        (contact_id, user_id),
    )
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    contact = _row_to_dict(row)
    contact.pop("user_id", None)
    extra_json = contact.pop("extra_fields", None)
    if extra_json:
        contact.update(json.loads(extra_json))
    return contact


def update_contact(contact_id, user_id, name, category, extra_fields_dict=None):
    """Update all fields of a contact."""
    conn = get_connection()
    c = _dict_cursor(conn) if _USE_POSTGRES else conn.cursor()
    extra_json = json.dumps(extra_fields_dict) if extra_fields_dict else None
    c.execute(
        f'UPDATE contacts SET {_Q}Name{_Q} = {_PH}, {_Q}Category{_Q} = {_PH}, extra_fields = {_PH} '
        f'WHERE id = {_PH} AND user_id = {_PH}',
        (name, category, extra_json, contact_id, user_id),
    )
    conn.commit()
    conn.close()


def delete_contact(contact_id, user_id):
    """Delete a single contact, only if it belongs to this user."""
    conn = get_connection()
    c = _dict_cursor(conn) if _USE_POSTGRES else conn.cursor()
    c.execute(
        f"DELETE FROM contacts WHERE id = {_PH} AND user_id = {_PH}",
        (contact_id, user_id),
    )
    conn.commit()
    conn.close()


def clear_all_contacts(user_id):
    """Delete every contact for this user."""
    conn = get_connection()
    c = _dict_cursor(conn) if _USE_POSTGRES else conn.cursor()
    c.execute(f"DELETE FROM contacts WHERE user_id = {_PH}", (user_id,))
    conn.commit()
    conn.close()


# ── Interaction helpers ──────────────────────────────────────────
def get_interactions(contact_id, user_id):
    """Return all interactions for a contact, newest first."""
    conn = get_connection()
    c = _dict_cursor(conn) if _USE_POSTGRES else conn.cursor()
    c.execute(
        f"SELECT id, date, note FROM contact_interactions "
        f"WHERE contact_id = {_PH} AND user_id = {_PH} ORDER BY date DESC, id DESC",
        (contact_id, user_id),
    )
    rows = c.fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def add_interaction(contact_id, user_id, date, note):
    """Add a new interaction entry for a contact."""
    conn = get_connection()
    c = _dict_cursor(conn) if _USE_POSTGRES else conn.cursor()
    c.execute(
        f"INSERT INTO contact_interactions (contact_id, user_id, date, note) "
        f"VALUES ({_ph(4)})",
        (contact_id, user_id, date, note),
    )
    conn.commit()
    conn.close()


def delete_interaction(interaction_id, user_id):
    """Delete a single interaction, only if it belongs to this user."""
    conn = get_connection()
    c = _dict_cursor(conn) if _USE_POSTGRES else conn.cursor()
    c.execute(
        f"DELETE FROM contact_interactions WHERE id = {_PH} AND user_id = {_PH}",
        (interaction_id, user_id),
    )
    conn.commit()
    conn.close()


# ── Journal helpers ──────────────────────────────────────────────
def get_journal_entries(user_id, table_name):
    """Return all journal entries for this user, newest first."""
    conn = get_connection()
    c = _dict_cursor(conn) if _USE_POSTGRES else conn.cursor()
    _d = f'{_Q}Date{_Q}'
    c.execute(
        f'SELECT id, {_d}, entry1, entry2, entry3 '
        f'FROM {table_name} WHERE user_id = {_PH} ORDER BY id DESC',
        (user_id,),
    )
    rows = c.fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def get_journal_entry_by_date(user_id, table_name, date):
    """Return a single journal entry for this user + date, or None."""
    conn = get_connection()
    c = _dict_cursor(conn) if _USE_POSTGRES else conn.cursor()
    _d = f'{_Q}Date{_Q}'
    c.execute(
        f'SELECT id, {_d}, entry1, entry2, entry3 '
        f'FROM {table_name} WHERE user_id = {_PH} AND {_d} = {_PH}',
        (user_id, date),
    )
    row = c.fetchone()
    conn.close()
    return _row_to_dict(row)


def save_journal_entry(user_id, table_name, date, entry1, entry2, entry3):
    """Upsert a journal entry: update if exists for this user+date, else insert."""
    conn = get_connection()
    c = _dict_cursor(conn) if _USE_POSTGRES else conn.cursor()
    _d = f'{_Q}Date{_Q}'

    c.execute(
        f'SELECT id FROM {table_name} WHERE user_id = {_PH} AND {_d} = {_PH}',
        (user_id, date),
    )
    existing = c.fetchone()

    if existing:
        existing = _row_to_dict(existing)
        c.execute(
            f"UPDATE {table_name} SET entry1 = {_PH}, entry2 = {_PH}, entry3 = {_PH} WHERE id = {_PH}",
            (entry1, entry2, entry3, existing["id"]),
        )
    else:
        c.execute(
            f'INSERT INTO {table_name} (user_id, {_d}, entry1, entry2, entry3) '
            f'VALUES ({_ph(5)})',
            (user_id, date, entry1, entry2, entry3),
        )
    conn.commit()
    conn.close()
