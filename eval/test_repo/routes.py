"""HTTP route handlers for the sample service."""

from auth import create_token, require_auth, validate_credentials
from checkout import submit_checkout
from users import get_user


def login(request, users: dict, secret: str) -> dict:
    """Authenticate a login form and return a bearer token."""
    username = request.form["username"]
    password = request.form["password"]
    if not validate_credentials(username, password, users):
        return {"status": 401, "error": "invalid credentials"}
    return {"status": 200, "token": create_token(username, secret)}


def checkout_route(request, customer: dict, gateway, tax_rate: float) -> dict:
    """Submit the cart from a checkout request."""
    order = submit_checkout(request.json["cart"], customer, gateway, tax_rate)
    return {"status": 201, "order": order}


def profile(request, database, secret: str) -> dict:
    """Return the authenticated user's profile."""
    user_id = require_auth(request.headers, secret)
    if not user_id:
        return {"status": 401}
    return {"status": 200, "user": get_user(user_id, database)}


def health() -> dict:
    """Simple liveness endpoint."""
    return {"status": "ok"}
