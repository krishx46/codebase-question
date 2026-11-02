"""User account operations."""


def get_user(user_id: str, database):
    """Look up one user by primary key."""
    return database.users.find_one({"id": user_id})


def create_user(email: str, password_hash: str, database) -> dict:
    """Create a new active user account."""
    record = {"email": email.lower(), "password_hash": password_hash, "active": True}
    database.users.insert_one(record)
    return record


def deactivate_user(user_id: str, database) -> bool:
    """Mark an account as inactive without deleting its data."""
    result = database.users.update_one({"id": user_id}, {"$set": {"active": False}})
    return result.modified_count == 1
