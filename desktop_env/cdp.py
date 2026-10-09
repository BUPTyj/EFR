import logging
from typing import Any, Optional

import requests


def _normalize_cdp_endpoint(remote_debugging_url: str) -> str:
    return remote_debugging_url.rstrip("/")


def _is_websocket_endpoint(endpoint: str) -> bool:
    return endpoint.startswith(("ws://", "wss://"))


def resolve_cdp_websocket_url(
    remote_debugging_url: str,
    *,
    timeout: float = 10.0,
    logger: Optional[logging.Logger] = None,
) -> str:
    endpoint = _normalize_cdp_endpoint(remote_debugging_url)
    if _is_websocket_endpoint(endpoint):
        return endpoint

    version_url = f"{endpoint}/json/version"
    if logger:
        logger.debug("Resolving Chrome DevTools websocket URL from %s", version_url)

    response = requests.get(version_url, timeout=timeout)
    response.raise_for_status()

    payload = response.json()
    websocket_url = payload.get("webSocketDebuggerUrl")
    if not websocket_url:
        raise ValueError(
            f"Chrome DevTools endpoint {version_url} did not return a "
            "webSocketDebuggerUrl"
        )

    return websocket_url


def connect_over_cdp_with_discovery(
    browser_type: Any,
    remote_debugging_url: str,
    *,
    timeout: float = 10.0,
    logger: Optional[logging.Logger] = None,
):
    endpoint = _normalize_cdp_endpoint(remote_debugging_url)
    if _is_websocket_endpoint(endpoint):
        return browser_type.connect_over_cdp(endpoint)

    last_error: Optional[Exception] = None

    try:
        websocket_url = resolve_cdp_websocket_url(
            endpoint,
            timeout=timeout,
            logger=logger,
        )
        if logger:
            logger.debug(
                "Connecting to Chrome DevTools via websocket endpoint %s",
                websocket_url,
            )
        return browser_type.connect_over_cdp(websocket_url)
    except Exception as exc:
        last_error = exc
        if logger:
            logger.warning(
                "Failed to resolve or connect via websocket CDP endpoint from %s: %s. "
                "Falling back to direct HTTP CDP discovery.",
                endpoint,
                exc,
            )

    try:
        return browser_type.connect_over_cdp(endpoint)
    except Exception:
        if logger and last_error is not None:
            logger.debug(
                "Direct HTTP CDP fallback for %s also failed after websocket discovery "
                "error: %s",
                endpoint,
                last_error,
            )
        raise
