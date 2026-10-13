"""SentinelOne SDL V2 query client. Requires Python 3.10+ and requests."""

"""vibes were used in the creation of this"""

from __future__ import annotations

import random
import re
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote, urlsplit

import requests


FORWARD_HEADER = "X-Dataset-Query-Forward-Tag"


class SDLError(RuntimeError):
    """HTTP, protocol, or server-side query error."""

    def __init__(self, message, *, status_code=None, details=None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details


def parse_time(value: str | datetime, *, now: datetime | None = None) -> datetime:
    """Accept now, relative durations, or timezone-aware ISO timestamps.

    Units: ms, s, m, h, d, w, y, plus full names.
    Years are fixed 365-day durations. Calendar months are unsupported.
    """
    reference = now if now is not None else datetime.now(timezone.utc)
    if reference.utcoffset() is None:
        raise ValueError("now must be timezone-aware")

    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        text = value.strip()
        if text.lower() == "now":
            result = reference
        else:
            match = re.fullmatch(
                r"(\d+(?:\.\d+)?)\s*"
                r"(milliseconds?|ms|seconds?|secs?|s|minutes?|mins?|m|"
                r"hours?|hrs?|h|days?|d|weeks?|w|years?|yrs?|y)",
                text,
                re.IGNORECASE,
            )
            if match:
                amount = float(match.group(1))
                unit = match.group(2).lower()
                multiplier = (
                    0.001
                    if unit == "ms" or unit.startswith("millisecond")
                    else {
                        "s": 1,
                        "m": 60,
                        "h": 3600,
                        "d": 86400,
                        "w": 604800,
                        "y": 365 * 86400,
                    }[unit[0]]
                )
                result = reference - timedelta(seconds=amount * multiplier)
            else:
                try:
                    result = datetime.fromisoformat(
                        text[:-1] + "+00:00" if text.endswith("Z") else text
                    )
                except ValueError as exc:
                    raise ValueError(f"Invalid time: {value!r}") from exc
    else:
        raise TypeError("Time must be a string or datetime")

    if result.utcoffset() is None:
        raise ValueError("Absolute timestamps must include a timezone")
    return result.astimezone(timezone.utc)


def timestamp(value: str | datetime, *, now: datetime | None = None) -> str:
    """Convert a time input to an ISO-8601 UTC string."""
    return parse_time(value, now=now).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


@dataclass
class QueryHandle:
    id: str
    forward_tag: str = field(repr=False)
    response: dict[str, Any]
    query_type: str
    released: bool = False

    @property
    def complete(self) -> bool:
        total = self.response.get("stepsTotal", 0)
        completed = self.response.get("stepsCompleted", 0)
        return total > 0 and completed >= total


@dataclass
class QueryResult:
    """Preserves the full response alongside decoded table rows."""

    query_id: str
    query_type: str
    response: dict[str, Any]

    @property
    def data(self):
        return self.response.get("data")

    @property
    def columns(self) -> list:
        data = self.data
        return data.get("columns", []) if isinstance(data, dict) else []

    @property
    def values(self) -> list:
        data = self.data
        return data.get("values", []) if isinstance(data, dict) else []

    def rows(self) -> list[dict[str, Any]]:
        """Decode a PQ table. LOG results remain available through .data."""
        if self.query_type != "PQ":
            raise ValueError("rows() is for PQ tables; use .data for LOG results")

        names = []
        for column in self.columns:
            if isinstance(column, str):
                names.append(column)
            elif isinstance(column, dict) and "name" in column:
                names.append(column["name"])
            else:
                raise SDLError(
                    "Unexpected column format", details=self.columns
                )

        if len(set(names)) != len(names):
            raise SDLError("Duplicate column names; use distinct query aliases")

        rows = []
        for values in self.values:
            if len(values) != len(names):
                raise SDLError("Column count does not match row width")
            rows.append(dict(zip(names, values)))
        return rows


class SDLClient:
    """Synchronous client. Use one instance per thread."""

    def __init__(
        self,
        console_url: str,
        token: str,
        *,
        account_ids: list[str] | None = None,
        scope_header: str | None = None,
        request_timeout: float = 15,
        retries: int = 4,
    ):
        parsed = urlsplit(console_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "console_url must be an HTTPS origin, without an API path"
            )
        if not isinstance(token, str) or not token.strip():
            raise ValueError("A non-empty token is required")
        if account_ids is not None and (
            not isinstance(account_ids, list)
            or not account_ids
            or any(not isinstance(x, str) or not x.strip() for x in account_ids)
        ):
            raise ValueError("account_ids must be a non-empty list of strings")
        if request_timeout <= 0 or retries < 0:
            raise ValueError("Invalid timeout or retry count")

        self.url = console_url.rstrip("/") + "/sdl/v2/api/queries"
        self.account_ids = (
            list(account_ids) if account_ids is not None else None
        )
        self.request_timeout = request_timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        if scope_header:
            self.session.headers["S1-Scope"] = scope_header

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def _request(self, method, url, *, deadline=None, **kwargs):
        for attempt in range(self.retries + 1):
            timeout = self.request_timeout
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Query wait deadline exceeded")
                timeout = min(timeout, remaining)

            # POST transport errors and 5xx are deliberately not retried:
            # the server may have already created a query.
            response = self.session.request(
                method, url, timeout=timeout, allow_redirects=False, **kwargs
            )

            if response.status_code == 429 and attempt < self.retries:
                delay = self._retry_delay(response, attempt)
                response.close()
                if deadline is not None and time.monotonic() + delay >= deadline:
                    raise TimeoutError("Retry would exceed query deadline")
                time.sleep(delay)
                continue

            if not 200 <= response.status_code < 300:
                try:
                    details = response.json()
                except ValueError:
                    details = response.text[:2000]
                status = response.status_code
                response.close()
                raise SDLError(
                    f"SDL returned HTTP {status}: {details}",
                    status_code=status,
                    details=details,
                )
            return response

        raise SDLError("Retry attempts exhausted")

    @staticmethod
    def _retry_delay(response, attempt):
        value = response.headers.get("Retry-After")
        if value:
            try:
                return max(0, float(value))
            except ValueError:
                try:
                    date = parsedate_to_datetime(value)
                    return max(
                        0, (date - datetime.now(timezone.utc)).total_seconds()
                    )
                except (ValueError, TypeError, OverflowError):
                    pass
        return min(8, 0.5 * 2**attempt) + random.uniform(0, 0.25)

    @staticmethod
    def _decode(response):
        try:
            body = response.json()
        except ValueError as exc:
            raise SDLError("SDL returned a non-JSON response") from exc
        finally:
            response.close()

        if not isinstance(body, dict):
            raise SDLError("Expected a JSON object", details=body)
        if body.get("error") or body.get("errors"):
            raise SDLError("SDL reported a query error", details=body)
        return body

    def submit(self, body: dict[str, Any]) -> QueryHandle:
        """Submit an arbitrary documented V2 query body."""
        response = self._request("POST", self.url, json=body)
        forward_tag = response.headers.get(FORWARD_HEADER)
        result = self._decode(response)

        if not result.get("id"):
            raise SDLError("Launch response has no query ID", details=result)
        if not forward_tag:
            raise SDLError(
                "Launch response has no forwarding header; "
                f"query {result['id']} cannot be polled or cancelled",
                details=result,
            )
        return QueryHandle(
            id=str(result["id"]),
            forward_tag=forward_tag,
            response=result,
            query_type=body.get("queryType", ""),
        )

    def _body(self, query_type, start, end, priority, edr_strict):
        if priority not in ("LOW", "HIGH"):
            raise ValueError("priority must be LOW or HIGH")
        now = datetime.now(timezone.utc)
        start_dt = parse_time(start, now=now)
        end_dt = parse_time(end, now=now)
        if start_dt >= end_dt:
            raise ValueError("start must be earlier than end")

        body = {
            "queryType": query_type,
            "tenant": self.account_ids is None,
            "startTime": timestamp(start_dt),
            "endTime": timestamp(end_dt),
            "queryPriority": priority,
        }
        if self.account_ids is not None:
            body["accountIds"] = self.account_ids
        if edr_strict:
            body["scheme"] = "edr"
        return body

    def submit_pq(
        self, query, *, start="5h", end="now",
        priority="HIGH", edr_strict=False,
    ):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        body = self._body("PQ", start, end, priority, edr_strict)
        body["pq"] = {"query": query, "resultType": "TABLE"}
        return self.submit(body)

    def submit_logs(
        self, filter_expression, *, start="5h", end="now",
        limit=5000, priority="HIGH", edr_strict=False,
    ):
        if not isinstance(filter_expression, str) or not filter_expression.strip():
            raise ValueError("filter_expression must be a non-empty string")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 5000:
            raise ValueError("limit must be an integer between 1 and 5000")
        body = self._body("LOG", start, end, priority, edr_strict)
        body["log"] = {"filter": filter_expression, "limit": limit}
        return self.submit(body)

    def poll(self, handle: QueryHandle, *, deadline=None):
        if handle.released:
            raise ValueError("Query has already been released")
        response = self._request(
            "GET",
            f"{self.url}/{quote(handle.id, safe='')}",
            params={"lastStepSeen": handle.response.get("stepsCompleted", 0)},
            headers={FORWARD_HEADER: handle.forward_tag},
            deadline=deadline,
        )
        # Preserve prior fields if a poll omits them.
        handle.response.update(self._decode(response))
        return handle.response

    def cancel(self, handle: QueryHandle):
        """Release the query, including after successful completion."""
        if handle.released:
            return
        response = self._request(
            "DELETE",
            f"{self.url}/{quote(handle.id, safe='')}",
            headers={FORWARD_HEADER: handle.forward_tag},
        )
        response.close()
        handle.released = True

    def wait(
        self, handle: QueryHandle, *, timeout=120, poll_interval=1,
    ) -> QueryResult:
        """Wait for completion. Does not cancel; run helpers handle cleanup."""
        if timeout <= 0 or not 0 < poll_interval <= 2:
            raise ValueError("timeout must be positive; poll_interval must be <= 2")
        if handle.released:
            raise ValueError("Query has already been released")

        deadline = time.monotonic() + timeout
        while not handle.complete:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Query {handle.id} did not finish in time")
            time.sleep(min(poll_interval, remaining))
            self.poll(handle, deadline=deadline)

        return QueryResult(handle.id, handle.query_type, dict(handle.response))

    def _run(self, handle, timeout, poll_interval):
        try:
            return self.wait(
                handle, timeout=timeout, poll_interval=poll_interval
            )
        finally:
            # Cleanup must not hide the original error or discard results.
            try:
                self.cancel(handle)
            except Exception as exc:
                warnings.warn(
                    f"Could not release query {handle.id}: {exc}",
                    RuntimeWarning,
                )

    def power_query(
        self, query, *, start="5h", end="now", priority="HIGH",
        edr_strict=False, timeout=120, poll_interval=1,
    ) -> QueryResult:
        handle = self.submit_pq(
            query, start=start, end=end, priority=priority,
            edr_strict=edr_strict,
        )
        return self._run(handle, timeout, poll_interval)

    def logs(
        self, filter_expression, *, start="5h", end="now",
        limit=5000, priority="HIGH", edr_strict=False,
        timeout=120, poll_interval=1,
    ) -> QueryResult:
        """Run a LOG filter without pipeline commands.

        Raw LOG results can be capped at limit. This is not a paginated export.
        """
        handle = self.submit_logs(
            filter_expression, start=start, end=end, limit=limit,
            priority=priority, edr_strict=edr_strict,
        )
        return self._run(handle, timeout, poll_interval)
