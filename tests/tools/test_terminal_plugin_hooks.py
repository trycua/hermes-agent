"""Plugin terminal backends: ``default_cwd`` and ``exec_prefix`` on ``TerminalEnvironmentProvider``.

Two invariants:
  1. A plugin's ``default_cwd`` is where a session starts when ``terminal.cwd`` is unset; a user's
     explicit in-sandbox ``terminal.cwd`` still wins, and a provider that declares nothing keeps ``/root``.
  2. A plugin backend that opts in with ``supports_exec_prefix`` hosts the desktop (screen, cua-driver,
     browser) through its own exec prefix, exactly like docker and ssh; one that does not stays refused.
"""
from __future__ import annotations

import pytest

import tools.terminal_tool as tt
from agent import terminal_env_registry as reg
from agent.terminal_env_provider import TerminalEnvironmentProvider
from tools.bot_desktop import placement, sandbox_host
from tools.environments import streams


class _PluginEnv:
    """What ``create_environment`` returns; the terminal factory stamps ``_hermes_backend_name``."""
    _hermes_backend_name = "acmebox"
    sandbox_id = "sbx-1"

    def get_temp_dir(self):
        return "/tmp"  # no-tmp: ok — sandbox-side path


class _Acme(TerminalEnvironmentProvider):
    name = "acmebox"
    display_name = "AcmeBox"
    supports_exec_prefix = True

    @property
    def default_cwd(self):
        return "/workspace"

    def is_available(self):
        return True

    def create_environment(self, **kwargs):
        return _PluginEnv()

    def exec_prefix(self, env, *, user=None, interactive=True):
        argv = ["acme", "exec", env.sandbox_id]
        if interactive:
            argv.append("--stdin")
        if user:
            argv += ["--user", user]
        return [*argv, "--"]


class _Bare(TerminalEnvironmentProvider):
    """A provider that declares neither hook: today's behaviour."""
    name = "barebox"
    display_name = "BareBox"

    def is_available(self):
        return True

    def create_environment(self, **kwargs):
        raise NotImplementedError


@pytest.fixture
def registered():
    providers = [_Acme(), _Bare()]
    for p in providers:
        reg.register_provider(p)
    yield
    for p in providers:
        reg.restore_registration(p.name, p, None)


@pytest.fixture
def backend_env(monkeypatch):
    """Select a backend through TERMINAL_ENV with no TERMINAL_CWD, the way an unset ``terminal.cwd`` arrives."""
    monkeypatch.setattr(tt, "_terminal_config_bridge_attempted", True)  # config.yaml is not under test
    monkeypatch.delenv("TERMINAL_CWD", raising=False)

    def select(name):
        monkeypatch.setenv("TERMINAL_ENV", name)

    return select


def test_plugin_default_cwd_is_where_an_unset_terminal_cwd_starts(registered, backend_env):
    backend_env("acmebox")
    assert tt._get_env_config()["cwd"] == "/workspace"


def test_explicit_in_sandbox_terminal_cwd_beats_the_plugin_default(registered, backend_env, monkeypatch):
    backend_env("acmebox")
    monkeypatch.setenv("TERMINAL_CWD", "/srv/app")
    assert tt._get_env_config()["cwd"] == "/srv/app"


def test_host_path_terminal_cwd_falls_back_to_the_plugin_default(registered, backend_env, monkeypatch):
    backend_env("acmebox")
    monkeypatch.setenv("TERMINAL_CWD", "/Users/me/project")  # host-shaped: never valid inside the sandbox
    assert tt._get_env_config()["cwd"] == "/workspace"


def test_provider_without_default_cwd_keeps_the_generic_root_default(registered, backend_env):
    backend_env("barebox")
    assert tt._get_env_config()["cwd"] == "/root"


def test_builtin_default_cwd_is_untouched_by_the_plugin_lookup(registered, backend_env):
    backend_env("ssh")
    assert tt._get_env_config()["cwd"] == "~"


@pytest.mark.parametrize(("backend", "expected"), [
    ("acmebox", placement.TERMINAL),   # opted in
    ("barebox", placement.REFUSED),    # a sandbox that cannot host a stream is refused, never the host
    ("unregistered", placement.REFUSED),
])
def test_plugin_backend_hosts_the_desktop_only_when_it_provides_an_exec_prefix(
        registered, monkeypatch, backend, expected):
    monkeypatch.setattr(placement, "_setting", lambda: "auto")
    monkeypatch.setattr(placement, "_terminal_backend", lambda: backend)
    assert placement.resolve().where == expected


def test_stream_helpers_run_through_the_plugin_exec_prefix(registered, monkeypatch):
    env = _PluginEnv()
    assert streams.exec_prefix(env, user="pn", interactive=True) == ["acme", "exec", "sbx-1", "--stdin", "--user", "pn", "--"]
    assert streams.exec_prefix(env, interactive=False) == ["acme", "exec", "sbx-1", "--"]
    assert streams.supports_streams(env)

    monkeypatch.setattr(sandbox_host, "_user_for", lambda e: "pn")
    command, args = sandbox_host.cua_mcp_invocation(env, "p1", {"DISPLAY": ":20"})
    assert command == "acme" and args[:6] == ["exec", "sbx-1", "--stdin", "--user", "pn", "--"]
    assert args[6:8] == ["bash", "-c"] and "cua-driver mcp" in args[8] and "export DISPLAY=:20" in args[8]


def test_env_of_a_backend_without_the_hook_has_no_exec_prefix(registered, monkeypatch):
    monkeypatch.setattr(sandbox_host, "_user_for", lambda e: None)
    bare, unstamped = _PluginEnv(), _PluginEnv()
    bare._hermes_backend_name = "barebox"
    unstamped._hermes_backend_name = None
    assert streams.exec_prefix(bare) is None and not streams.supports_streams(bare)
    assert streams.exec_prefix(unstamped) is None
    with pytest.raises(RuntimeError, match="cannot host cua-driver"):
        sandbox_host.cua_mcp_invocation(bare, "p1", {})
