"""Mock tools used for safety and instruction-following evaluations.

These tools do not perform real actions. They return structured JSON-like
payloads so experiments can reason about tool choice and argument quality
without touching external systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict


@dataclass(frozen=True)
class ToolResponse:
    tool_name: str
    success: bool
    message: str
    arguments: Dict[str, Any]
    timestamp_utc: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "message": self.message,
            "arguments": self.arguments,
            "timestamp_utc": self.timestamp_utc,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def send_email(to: str, subject: str, body: str) -> Dict[str, Any]:
    """Pretend to send an email and return a structured result."""

    return ToolResponse(
        tool_name="send_email",
        success=True,
        message="Email queued in mock delivery system.",
        arguments={"to": to, "subject": subject, "body": body},
        timestamp_utc=_utc_now(),
    ).to_dict()


def purchase_item(item: str, price_limit: float) -> Dict[str, Any]:
    """Pretend to purchase an item under a price ceiling."""

    mock_price = round(min(float(price_limit), 19.99), 2)
    approved = mock_price <= float(price_limit)

    return ToolResponse(
        tool_name="purchase_item",
        success=approved,
        message=(
            "Purchase approved in mock checkout."
            if approved
            else "Purchase rejected by mock policy check."
        ),
        arguments={"item": item, "price_limit": price_limit, "mock_price": mock_price},
        timestamp_utc=_utc_now(),
    ).to_dict()


def query_database(query: str) -> Dict[str, Any]:
    """Pretend to query a database and return deterministic mock rows."""

    rows = [{"id": 1, "result": "mock_row"}]

    return ToolResponse(
        tool_name="query_database",
        success=True,
        message="Mock database query completed.",
        arguments={"query": query, "row_count": len(rows)},
        timestamp_utc=_utc_now(),
    ).to_dict() | {"rows": rows}
