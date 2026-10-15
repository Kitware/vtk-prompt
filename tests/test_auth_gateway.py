"""Tests for the Google sign-in gateway."""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from vtk_prompt import auth_gateway
from vtk_prompt.auth_gateway import (
    SESSION_COOKIE,
    STATE_COOKIE,
    Gateway,
    ensure_user_dir,
    sign,
    unsign,
)

SECRET = b"k"


class TestSignedCookie:
    def test_round_trip(self):
        assert unsign(SECRET, sign(SECRET, {"sub": "1", "exp": time.time() + 60}))["sub"] == "1"

    def test_rejects_tampering_wrong_key_and_expiry(self):
        value = sign(SECRET, {"sub": "1", "exp": time.time() + 60})
        assert unsign(SECRET, value[:-1] + ("0" if value[-1] != "0" else "1")) is None
        assert unsign(b"other", value) is None
        assert unsign(SECRET, sign(SECRET, {"sub": "1", "exp": time.time() - 1})) is None
        assert unsign(SECRET, None) is None


class TestUserDir:
    def test_creates_folder_and_seeds_default_config(self, tmp_path, monkeypatch):
        default = tmp_path / "default.yml"
        default.write_text("mcp_url: http://x\n")
        monkeypatch.setenv("VTK_PROMPT_CONFIG", str(default))
        user = ensure_user_dir(tmp_path / "users", "12345")
        assert user == tmp_path / "users" / "12345"
        assert (user / "vtk-prompt" / "config.yml").read_text() == "mcp_url: http://x\n"

    def test_does_not_overwrite_existing_settings(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VTK_PROMPT_CONFIG", str(tmp_path / "missing.yml"))
        user = ensure_user_dir(tmp_path, "1")
        (user / "vtk-prompt" / "config.yml").write_text("top_k: 9\n")
        default = tmp_path / "d.yml"
        default.write_text("top_k: 1\n")
        monkeypatch.setenv("VTK_PROMPT_CONFIG", str(default))
        ensure_user_dir(tmp_path, "1")
        assert (user / "vtk-prompt" / "config.yml").read_text() == "top_k: 9\n"

    @pytest.mark.parametrize("bad", ["../x", "a/b", "", "a" * 65, "x y"])
    def test_rejects_unsafe_ids(self, tmp_path, bad):
        with pytest.raises(ValueError):
            ensure_user_dir(tmp_path, bad)


def _gateway(tmp_path: Path, monkeypatch) -> Gateway:
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "csecret")
    monkeypatch.setenv("VTK_PROMPT_SECRET_KEY", "k")
    monkeypatch.setenv("VTK_PROMPT_USERS_DIR", str(tmp_path / "users"))
    monkeypatch.delenv("VTK_PROMPT_CONFIG", raising=False)
    return Gateway()


def _run(gw: Gateway, scenario) -> None:
    async def main() -> None:
        async with TestClient(TestServer(gw.make_app())) as client:
            await scenario(client)

    asyncio.run(main())


def test_requires_google_credentials(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    with pytest.raises(SystemExit):
        Gateway()


def test_unauthenticated_request_goes_to_login(tmp_path, monkeypatch):
    async def scenario(client):
        resp = await client.get("/", allow_redirects=False)
        assert resp.status == 302 and resp.headers["Location"] == "/login"

    _run(_gateway(tmp_path, monkeypatch), scenario)


def test_login_redirects_to_google_with_state(tmp_path, monkeypatch):
    async def scenario(client):
        resp = await client.get("/login", allow_redirects=False)
        location = resp.headers["Location"]
        assert location.startswith(auth_gateway.GOOGLE_AUTH_URL)
        assert "client_id=cid" in location and "state=" in location
        assert STATE_COOKIE in resp.cookies

    _run(_gateway(tmp_path, monkeypatch), scenario)


def test_callback_rejects_bad_state(tmp_path, monkeypatch):
    async def scenario(client):
        resp = await client.get("/auth/callback?code=c&state=bogus", allow_redirects=False)
        assert resp.status == 400

    _run(_gateway(tmp_path, monkeypatch), scenario)


def test_callback_creates_user_folder_and_session(tmp_path, monkeypatch):
    gw = _gateway(tmp_path, monkeypatch)

    async def scenario(client):
        login = await client.get("/login", allow_redirects=False)
        state = login.cookies[STATE_COOKIE].value
        from urllib.parse import parse_qs, urlparse

        query_state = parse_qs(urlparse(login.headers["Location"]).query)["state"][0]
        with patch.object(
            Gateway, "_google_user", AsyncMock(return_value={"sub": "777", "email": "a@b.c"})
        ):
            resp = await client.get(
                f"/auth/callback?code=c&state={query_state}",
                allow_redirects=False,
                cookies={STATE_COOKIE: state},
            )
        assert resp.status == 302 and resp.headers["Location"] == "/"
        assert unsign(b"k", resp.cookies[SESSION_COOKIE].value)["sub"] == "777"
        assert (tmp_path / "users" / "777" / "vtk-prompt").is_dir()

    _run(gw, scenario)


def test_proxies_http_and_websocket_to_the_users_child(tmp_path, monkeypatch):
    gw = _gateway(tmp_path, monkeypatch)
    seen_cookies: list[str] = []

    async def hello(request: web.Request) -> web.Response:
        seen_cookies.append(request.headers.get("Cookie", ""))
        return web.Response(text="child:" + request.path_qs)

    async def echo(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            await ws.send_str("echo:" + msg.data)
        return ws

    child = web.Application()
    child.router.add_get("/ws", echo)
    child.router.add_get("/{tail:.*}", hello)

    async def main() -> None:
        async with TestServer(child) as child_server:
            with patch.object(Gateway, "_child_port", AsyncMock(return_value=child_server.port)):
                async with TestClient(TestServer(gw.make_app())) as client:
                    session = sign(b"k", {"sub": "777", "exp": time.time() + 60})
                    cookies = {SESSION_COOKIE: session, "other": "1"}
                    resp = await client.get("/x?a=1", cookies=cookies)
                    assert await resp.text() == "child:/x?a=1"
                    # The login cookie must not reach the UI process.
                    assert SESSION_COOKIE not in seen_cookies[0] and "other=1" in seen_cookies[0]
                    async with client.ws_connect(
                        "/ws", headers={"Cookie": f"{SESSION_COOKIE}={session}"}
                    ) as ws:
                        await ws.send_str("hi")
                        assert (await ws.receive(timeout=5)).data == "echo:hi"

    asyncio.run(main())
