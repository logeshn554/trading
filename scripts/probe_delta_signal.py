"""Read-only probe of Delta fields needed for a causal ETH flow signal."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ethresearch.delta_mcp import DeltaMcpClient, DeltaMcpError


def describe(name: str, payload: object) -> None:
    value = payload.get("result", payload) if isinstance(payload, dict) else payload
    if isinstance(value, list):
        keys = sorted({key for row in value[:10] if isinstance(row, dict) for key in row})
        print(f"{name}: {len(value)} rows; sample fields: {', '.join(keys)}")
        if value:
            for label, row in (("first", value[0]), ("last", value[-1])):
                if isinstance(row, dict):
                    print(f"  {label} timestamp: {row.get('time', row.get('timestamp', row.get('created_at', 'unavailable')))}")
    elif isinstance(value, dict):
        print(f"{name}: fields: {', '.join(sorted(value))}")
    else:
        print(f"{name}: unexpected response type {type(value).__name__}")


def main() -> int:
    client = DeltaMcpClient()
    now = datetime.now(timezone.utc)
    try:
        for name, arguments in (
            ("get_product", {"symbol": "ETHUSD"}),
            ("get_candles", {"symbol": "ETHUSD", "resolution": "1h",
                             "start": int((now - timedelta(hours=8)).timestamp()),
                             "end": int(now.timestamp())}),
            ("get_recent_trades", {"symbol": "ETHUSD"}),
        ):
            describe(name, client.call(name, arguments))
        return 0
    except DeltaMcpError as exc:
        print(f"Delta public-data probe failed: {exc}")
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
