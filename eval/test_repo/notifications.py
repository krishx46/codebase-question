"""Email delivery helpers."""


def render_template(template: str, values: dict) -> str:
    """Replace named placeholders with supplied values."""
    return template.format(**values)


def send_email(address: str, subject: str, body: str, provider=None) -> bool:
    """Send a message using the configured email provider."""
    if not address or "@" not in address:
        raise ValueError("Invalid email address")
    if provider is None:
        return False
    return provider.send(to=address, subject=subject, body=body)


def retry_delivery(send, attempts: int = 3) -> bool:
    """Retry a failed send operation up to the attempt limit."""
    for _ in range(attempts):
        if send():
            return True
    return False
