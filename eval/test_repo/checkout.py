"""Checkout workflow for the sample service."""

from notifications import send_email


def calculate_total(items: list[dict], tax_rate: float) -> float:
    """Calculate line totals and tax for a basket."""
    subtotal = sum(item["price"] * item["quantity"] for item in items)
    return round(subtotal * (1 + tax_rate), 2)


def charge_card(payment_gateway, customer_id: str, amount: float) -> str:
    """Charge a customer and return the provider transaction ID."""
    result = payment_gateway.charge(customer_id=customer_id, amount=amount)
    if not result.get("approved"):
        raise ValueError("Card was declined")
    return result["transaction_id"]


def submit_checkout(cart: list[dict], customer: dict, gateway, tax_rate: float) -> dict:
    """Calculate the order, charge payment, then email the receipt."""
    total = calculate_total(cart, tax_rate)
    transaction_id = charge_card(gateway, customer["id"], total)
    order = {"customer_id": customer["id"], "total": total, "transaction_id": transaction_id}
    send_email(customer["email"], "Your receipt", str(order))
    return order
