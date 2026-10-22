"""Google sign-in gateway in front of vtk-prompt-ui, with one UI process per user.

A trame app keeps a single global state shared by every connected browser, so
users cannot safely share a process. The gateway authenticates with Google,
creates ``<users dir>/<google id>/`` for each user and runs a dedicated
``vtk-prompt-ui`` child whose ``XDG_CONFIG_HOME`` points at that folder, so the
existing config file and ``sessions/`` end up in the user's own folder. HTTP and
websocket traffic is then proxied to that child.

Configuration (environment):
  GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET   OAuth client (required)
  VTK_PROMPT_SECRET_KEY    cookie signing key (random per start when unset)
  VTK_PROMPT_USERS_DIR     parent of the per-user folders
  VTK_PROMPT_BASE_URL      public URL, when behind a proxy (for the redirect URI)
  VTK_PROMPT_CONFIG        default config copied into each new user's folder
  VTK_PROMPT_MCP_URL       vtk-mcp server fixed for everyone (read-only in the settings)

Example:
    >>> vtk-prompt-gateway --port 8080
"""

import asyncio
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import aiohttp
from aiohttp import web

from . import get_logger
from .utils.env_config import APP_DIR_NAME, CONFIG_ENV_VAR, MCP_URL_ENV_VAR

logger = get_logger(__name__)

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"

HTTP = web.AppKey("http", aiohttp.ClientSession)
SESSION_COOKIE = "vtk_prompt_session"
STATE_COOKIE = "vtk_prompt_oauth_state"
SESSION_TTL = 7 * 24 * 3600
CHILD_STARTUP_TIMEOUT = 120.0

_USER_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
_HOP_BY_HOP = {"connection", "keep-alive", "transfer-encoding", "upgrade", "host", "content-length"}


def sign(secret: bytes, payload: dict) -> str:
    """Return ``payload`` as a tamper-proof cookie value."""
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    mac = hmac.new(secret, body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{mac}"


def unsign(secret: bytes, value: str | None) -> dict | None:
    """Return the payload of a cookie made by ``sign``, or None if invalid/expired."""
    try:
        body, mac = (value or "").rsplit(".", 1)
        expected = hmac.new(secret, body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(mac, expected):
            return None
        payload = json.loads(base64.urlsafe_b64decode(body))
    except ValueError:
        return None
    return payload if payload.get("exp", 0) > time.time() else None


def ensure_user_dir(users_dir: Path, user_id: str) -> Path:
    """Create (and seed) the folder holding one user's settings and data."""
    if not _USER_ID.fullmatch(user_id):
        raise ValueError(f"unsafe user id: {user_id!r}")
    config_dir = users_dir / user_id / APP_DIR_NAME
    config_dir.mkdir(parents=True, exist_ok=True)
    default = os.environ.get(CONFIG_ENV_VAR)
    target = config_dir / "config.yml"
    if default and Path(default).is_file() and not target.exists():
        shutil.copyfile(default, target)
    return users_dir / user_id


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Gateway:
    """Auth routes plus a per-user reverse proxy."""

    def __init__(self) -> None:
        """Read the configuration from the environment."""
        self.client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
        self.client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")
        if not (self.client_id and self.client_secret):
            raise SystemExit("GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET must be set")
        self.secret = (os.environ.get("VTK_PROMPT_SECRET_KEY") or secrets.token_hex(32)).encode()
        self.users_dir = Path(
            os.environ.get("VTK_PROMPT_USERS_DIR")
            or Path.home() / ".local" / "share" / APP_DIR_NAME / "users"
        )
        self.base_url = os.environ.get("VTK_PROMPT_BASE_URL", "").rstrip("/")
        self.children: dict[str, tuple[asyncio.subprocess.Process, int]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    # -- auth ---------------------------------------------------------------

    def _redirect_uri(self, request: web.Request) -> str:
        base = self.base_url or str(request.url.origin())
        return f"{base}/auth/callback"

    def _secure(self, request: web.Request) -> bool:
        return (self.base_url or str(request.url.origin())).startswith("https://")

    def _user(self, request: web.Request) -> dict | None:
        return unsign(self.secret, request.cookies.get(SESSION_COOKIE))

    async def login(self, request: web.Request) -> web.StreamResponse:
        """Send the browser to Google's sign-in page."""
        state = secrets.token_urlsafe(24)
        query = urlencode(
            {
                "client_id": self.client_id,
                "redirect_uri": self._redirect_uri(request),
                "response_type": "code",
                "scope": "openid email profile",
                "state": state,
            }
        )
        resp = web.HTTPFound(f"{GOOGLE_AUTH_URL}?{query}")
        resp.set_cookie(
            STATE_COOKIE,
            sign(self.secret, {"state": state, "exp": time.time() + 600}),
            max_age=600,
            httponly=True,
            samesite="Lax",
            secure=self._secure(request),
        )
        raise resp

    async def _google_user(self, request: web.Request, code: str) -> dict[str, Any]:
        """Exchange the authorization code for the signed-in Google profile."""
        async with aiohttp.ClientSession() as http:
            async with http.post(
                GOOGLE_TOKEN_URL,
                data={
                    "code": code,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "redirect_uri": self._redirect_uri(request),
                    "grant_type": "authorization_code",
                },
            ) as resp:
                resp.raise_for_status()
                token = await resp.json()
            headers = {"Authorization": f"Bearer {token['access_token']}"}
            async with http.get(GOOGLE_USERINFO_URL, headers=headers) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def callback(self, request: web.Request) -> web.StreamResponse:
        """Finish the Google sign-in and start the user's session."""
        saved = unsign(self.secret, request.cookies.get(STATE_COOKIE))
        code = request.query.get("code")
        if not (
            saved and code and hmac.compare_digest(saved["state"], request.query.get("state", ""))
        ):
            raise web.HTTPBadRequest(text="Invalid sign-in state, try again from /login")
        try:
            profile = await self._google_user(request, code)
            user_id = str(profile["sub"])
            ensure_user_dir(self.users_dir, user_id)
        except (aiohttp.ClientError, KeyError, ValueError) as exc:
            logger.error("Google sign-in failed: %s", exc)
            raise web.HTTPBadGateway(text="Google sign-in failed")
        resp = web.HTTPFound("/")
        resp.set_cookie(
            SESSION_COOKIE,
            sign(
                self.secret,
                {
                    "sub": user_id,
                    "email": profile.get("email", ""),
                    "exp": time.time() + SESSION_TTL,
                },
            ),
            max_age=SESSION_TTL,
            httponly=True,
            samesite="Lax",
            secure=self._secure(request),
        )
        resp.del_cookie(STATE_COOKIE)
        raise resp

    async def logout(self, request: web.Request) -> web.StreamResponse:
        """Forget the user's session."""
        resp = web.HTTPFound("/login")
        resp.del_cookie(SESSION_COOKIE)
        raise resp

    # -- per-user UI processes ------------------------------------------------

    async def _child_port(self, user_id: str) -> int:
        lock = self._locks.setdefault(user_id, asyncio.Lock())
        async with lock:
            running = self.children.get(user_id)
            if running and running[0].returncode is None:
                return running[1]
            user_dir = ensure_user_dir(self.users_dir, user_id)
            port = _free_port()
            # The child must use the user's own config, not a shared explicit one.
            env = {k: v for k, v in os.environ.items() if k != CONFIG_ENV_VAR}
            env["XDG_CONFIG_HOME"] = str(user_dir)
            env["VTK_PROMPT_AUTH"] = "true"  # tells the UI to show the logout button
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "vtk_prompt.vtk_prompt_ui",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--server",
                env=env,
            )
            logger.info("Started vtk-prompt-ui for user %s on port %d", user_id, port)
            deadline = time.monotonic() + CHILD_STARTUP_TIMEOUT
            while time.monotonic() < deadline and proc.returncode is None:
                try:
                    _, writer = await asyncio.open_connection("127.0.0.1", port)
                    writer.close()
                    self.children[user_id] = (proc, port)
                    return port
                except OSError:
                    await asyncio.sleep(0.5)
            proc.terminate()
            raise web.HTTPGatewayTimeout(text="vtk-prompt-ui did not start")

    # -- proxy ----------------------------------------------------------------

    @staticmethod
    def _forward_headers(request: web.Request) -> dict[str, str]:
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
        cookies = "; ".join(f"{k}={v}" for k, v in request.cookies.items() if k != SESSION_COOKIE)
        headers.pop("Cookie", None)
        if cookies:
            headers["Cookie"] = cookies
        return headers

    async def proxy(self, request: web.Request) -> web.StreamResponse:
        """Forward a signed-in user's request to their own UI process."""
        user = self._user(request)
        if user is None:
            raise web.HTTPFound("/login")
        port = await self._child_port(user["sub"])
        url = f"http://127.0.0.1:{port}{request.path_qs}"
        session: aiohttp.ClientSession = request.app[HTTP]
        headers = self._forward_headers(request)

        if request.headers.get("Upgrade", "").lower() == "websocket":
            return await self._proxy_ws(request, session, url, headers)

        async with session.request(
            request.method,
            url,
            headers=headers,
            data=await request.read(),
            allow_redirects=False,
        ) as upstream:
            resp = web.StreamResponse(status=upstream.status)
            for key, value in upstream.headers.items():
                if key.lower() not in _HOP_BY_HOP:
                    resp.headers.add(key, value)
            await resp.prepare(request)
            async for chunk in upstream.content.iter_any():
                await resp.write(chunk)
            await resp.write_eof()
            return resp

    async def _proxy_ws(
        self,
        request: web.Request,
        session: aiohttp.ClientSession,
        url: str,
        headers: dict[str, str],
    ) -> web.StreamResponse:
        protocols = [
            p.strip()
            for p in request.headers.get("Sec-WebSocket-Protocol", "").split(",")
            if p.strip()
        ]
        for name in (
            "Sec-WebSocket-Key",
            "Sec-WebSocket-Version",
            "Sec-WebSocket-Protocol",
            "Sec-WebSocket-Extensions",
        ):
            headers.pop(name, None)
        ws_client = web.WebSocketResponse(protocols=protocols, max_msg_size=0)
        await ws_client.prepare(request)
        async with session.ws_connect(
            url.replace("http://", "ws://", 1), headers=headers, protocols=protocols, max_msg_size=0
        ) as ws_server:

            async def pump(src: Any, dst: Any) -> None:
                async for msg in src:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        await dst.send_str(msg.data)
                    elif msg.type == aiohttp.WSMsgType.BINARY:
                        await dst.send_bytes(msg.data)
                    else:
                        break

            tasks = [
                asyncio.ensure_future(pump(ws_client, ws_server)),
                asyncio.ensure_future(pump(ws_server, ws_client)),
            ]
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in tasks:
                task.cancel()
        await ws_client.close()
        return ws_client

    # -- lifecycle --------------------------------------------------------------

    async def _on_startup(self, app: web.Application) -> None:
        app[HTTP] = aiohttp.ClientSession(auto_decompress=False)

    async def _on_cleanup(self, app: web.Application) -> None:
        await app[HTTP].close()
        for proc, _ in self.children.values():
            if proc.returncode is None:
                proc.terminate()

    def make_app(self) -> web.Application:
        """Build the aiohttp application."""
        app = web.Application()
        app.on_startup.append(self._on_startup)
        app.on_cleanup.append(self._on_cleanup)
        app.router.add_get("/login", self.login)
        app.router.add_get("/auth/callback", self.callback)
        app.router.add_get("/logout", self.logout)
        app.router.add_route("*", "/{tail:.*}", self.proxy)
        return app


def main() -> None:
    """Run the gateway."""
    import argparse

    parser = argparse.ArgumentParser(description="Google sign-in gateway for vtk-prompt-ui")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--mcp-url", help="vtk-mcp server URL, fixed for all users (read-only in their UI)"
    )
    args = parser.parse_args()
    if args.mcp_url:
        os.environ[MCP_URL_ENV_VAR] = args.mcp_url  # inherited by every user's UI process
    web.run_app(Gateway().make_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
