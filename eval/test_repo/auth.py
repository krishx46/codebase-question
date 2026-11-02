"""Authentication helpers used by the sample service."""

import hashlib
import hmac
import time


def hash_password(password: str, salt: str) -> str:
    """Return a salted digest for a password."""
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120000).hex()


def validate_credentials(username: str, password: str, users: dict) -> bool:
    """Check a username and password against stored user records."""
    user = users.get(username)
    if not user:
        return False
    return hmac.compare_digest(hash_password(password, user["salt"]), user["password_hash"])


def create_token(user_id: str, secret: str, now: int | None = None) -> str:
    """Create a signed token with a one hour expiration."""
    issued = now or int(time.time())
    payload = f"{user_id}:{issued + 3600}"
    signature = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}:{signature}"


def require_auth(headers: dict, secret: str) -> str | None:
    """Extract and verify the bearer token, returning its user ID."""
    token = headers.get("Authorization", "").removeprefix("Bearer ")
    pieces = token.split(":")
    if len(pieces) != 3 or int(pieces[1]) < int(time.time()):
        return None
    payload = ":".join(pieces[:2])
    expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return pieces[0] if hmac.compare_digest(expected, pieces[2]) else None
