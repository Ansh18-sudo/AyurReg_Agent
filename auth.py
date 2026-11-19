"""Local user accounts with salted PBKDF2 password hashes."""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import uuid

import database

_ITER = 200_000
_USERNAME_RE = re.compile(r"^[a-z0-9_.-]{3,32}$")


def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _ITER).hex()


def register(username: str, password: str, display_name: str = ""):
    """Returns (ok, message, user_dict_or_None)."""
    username = (username or "").strip().lower()
    if not _USERNAME_RE.match(username):
        return False, "Username must be 3-32 characters: letters, digits, . _ -", None
    if len(password or "") < 8:
        return False, "Password must be at least 8 characters.", None
    salt = secrets.token_bytes(16)
    uid = uuid.uuid4().hex
    name = (display_name or "").strip() or username
    conn = database.get_conn()
    try:
        with conn:
            conn.execute(
                "INSERT INTO users(id, username, display_name, password_hash, salt, created_at) "
                "VALUES(?,?,?,?,?,?)",
                (uid, username, name, _hash(password, salt), salt.hex(), database.now()),
            )
    except sqlite3.IntegrityError:
        return False, "That username is already taken.", None
    return True, "Account created.", {"id": uid, "username": username, "display_name": name}


def login(username: str, password: str):
    row = database.get_conn().execute(
        "SELECT * FROM users WHERE username=?", ((username or "").strip().lower(),)
    ).fetchone()
    if not row:
        _hash(password or "", b"0" * 16)  # keep timing similar for unknown users
        return None
    expected = _hash(password or "", bytes.fromhex(row["salt"]))
    if hmac.compare_digest(expected, row["password_hash"]):
        return {"id": row["id"], "username": row["username"], "display_name": row["display_name"]}
    return None
