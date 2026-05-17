import hashlib
import hmac

from config.settings import settings


def verify_webhook_signature(payload: bytes, signature: str) -> bool:
    """
    Verify incoming webhook signature.
    Protects against fake lead injections.
    """
    mac = hmac.new(
        key=settings.WEBHOOK_SECRET.encode("utf-8"),
        msg=payload,
        digestmod=hashlib.sha256,
    )

    expected = mac.hexdigest()
    received = signature.replace("sha256=", "").strip()

    return hmac.compare_digest(expected, received)