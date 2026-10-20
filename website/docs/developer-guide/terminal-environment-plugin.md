# Terminal Environment Provider Plugins

Hermes runs shell commands through a pluggable set of **terminal backends**.
The built-in backends (local, Docker, Singularity, Modal, Daytona, Vercel
Sandbox, SSH) live in the core repo under `tools/environments/`. Third-party
sandbox vendors integrate as **plugins** instead — a standalone plugin repo
installed under `~/.hermes/plugins/`, registering a backend the user selects
exactly like a built-in one via `terminal.backend` in `config.yaml`.

This page mirrors the [Browser Provider Plugins](./browser-provider-plugin.md)
guide — same registration flow, same scope semantics.

## What a provider controls

A registered backend automatically participates in every core surface:

| Surface | Driven by |
|---|---|
| Command dispatch (`terminal`, `execute_code`, file tools) | `create_environment()` |
| `hermes setup` backend picker | `display_name`, `description`, `setup_instructions()`, `post_setup()` |
| Dashboard terminal-backend picker (probe status) | `probe()` |
| `hermes status` / `hermes doctor` | `doctor_checks()` |
| System-prompt environment hints | `is_remote`, `env_description` |
| Dangerous-command approval skipping | `skip_container_guards` |
| Container path/cwd handling | `is_container` |
| Synced cache-file path translation | `cache_path_base` |
| Secret stripping from spawned subprocesses | `strip_env_keys` |
| Per-session sandbox isolation (`container_persistent: false`) | `session_isolated_when_nonpersistent` |
| Starting directory when `terminal.cwd` is unset | `default_cwd` |
| Bot Desktop, `computer_use` and the browser running inside the backend | `supports_exec_prefix`, `exec_prefix()` |

Declaring these flags on the provider closes the classic "new backend missed
classification site N" bug class — the core consults the registry at each
site instead of a hardcoded list of names.

## Minimal provider

```python title="~/.hermes/plugins/acmebox/__init__.py"
from agent.terminal_env_provider import TerminalEnvironmentProvider


class AcmeBoxEnvironment:
    """Must satisfy the BaseEnvironment duck-typed contract."""

    def __init__(self, cwd, timeout, task_id):
        self.cwd, self.timeout, self.task_id = cwd, timeout, task_id

    def execute(self, command, timeout=None, **kwargs):
        ...  # run the command in the sandbox
        return {"output": "...", "exit_code": 0}

    def cleanup(self):
        ...  # tear down / detach


class AcmeBoxProvider(TerminalEnvironmentProvider):
    name = "acmebox"
    display_name = "AcmeBox"
    is_remote = True          # commands don't run on the host
    is_container = True       # container-style path/cwd semantics

    @property
    def description(self):
        return "Run commands in an AcmeBox cloud sandbox."

    @property
    def cache_path_base(self):
        return "~/.hermes"    # where synced cache files land, or None

    @property
    def strip_env_keys(self):
        return frozenset({"ACMEBOX_TOKEN"})

    def is_available(self):
        import importlib.util, os
        return (
            importlib.util.find_spec("acmebox") is not None
            and bool(os.getenv("ACMEBOX_TOKEN"))
        )

    def create_environment(self, *, cwd, timeout, task_id="default",
                           image=None, container_config=None, **kwargs):
        return AcmeBoxEnvironment(cwd, timeout, task_id)


def register(ctx):
    ctx.register_terminal_environment_provider(AcmeBoxProvider())
```

```yaml title="~/.hermes/plugins/acmebox/plugin.yaml"
name: acmebox
version: 0.1.0
description: AcmeBox cloud sandbox terminal backend
kind: backend
```

Enable it, select it, run:

```bash
hermes plugins enable acmebox
hermes config set terminal.backend acmebox
```

## Rules

- **Reserved names.** Registrations that collide with a built-in backend name
  (`local`, `docker`, `singularity`, `modal`, `managed_modal`, `daytona`,
  `vercel_sandbox`, `ssh`) are rejected. Plugins extend the backend set; they
  never shadow in-tree backends.
- **`create_environment` must accept `**kwargs`** and ignore unknown keys —
  the forward-compat contract that lets the factory signature evolve without
  breaking older plugins.
- **`is_available()` / `probe()` must be cheap.** No network calls — they run
  during requirement checks and UI paints.
- **Fail-soft everywhere.** A provider attribute that raises is treated as
  its default by the core (e.g. a raising `skip_container_guards` keeps the
  approval layer ON). Don't rely on exceptions for control flow.
- **Secrets belong in `strip_env_keys`.** Your vendor token must never be
  readable by a model-authored shell command; listing it strips it from every
  spawned subprocess unconditionally, like the built-in `MODAL_*` /
  `DAYTONA_API_KEY` handling.

## Environment object contract

`create_environment()` returns an object satisfying the same duck-typed
interface as `tools.environments.base.BaseEnvironment`:

- `execute(command, timeout=None, ...)` → `{"output": str, "exit_code": int}`
- `cleanup()` — release resources; called on session teardown / idle reaping
- Optional: persistence hooks mirroring the built-in cloud backends

Subclassing `BaseEnvironment` is recommended (you inherit the shared file-sync
and background-process plumbing) but not required.

## Starting directory

A session starts in `terminal.cwd` when the user sets it. Otherwise the core
uses a per-backend default, and for a plugin backend that is `/root` unless you
override `default_cwd`:

```python
@property
def default_cwd(self):
    return "/workspace"   # e.g. the only path that survives a stop/start
```

Return `None` (the default) to keep `/root`. A user's `terminal.cwd` always
wins, and a host-looking path in it is still replaced by this value on
`is_container` backends.

## Hosting the desktop (`exec_prefix`)

The screen the bot works on (Xvnc and Xfce, the browser, and the `cua-driver`
process behind `computer_use`) follows the terminal backend. For `docker`,
`ssh` and `singularity` it runs inside the sandbox, started through the same
command channel the terminal uses. A plugin backend takes part by naming that
channel: set `supports_exec_prefix = True` and implement `exec_prefix()`.

```python
class AcmeBoxProvider(TerminalEnvironmentProvider):
    supports_exec_prefix = True

    def exec_prefix(self, env, *, user=None, interactive=True):
        # Local argv that runs whatever is appended to it inside the sandbox.
        argv = ["acmebox", "exec", env.sandbox_id]
        if interactive:
            argv.append("--stdin")
        if user:
            argv += ["--user", user]
        return argv + ["--"]
```

- `env` is an environment this provider created. Return `None` when it has no
  live sandbox to exec into.
- The client must keep argument boundaries the way `docker exec` does, and keep
  stdin open when `interactive` is true: the desktop holds long-lived stdio
  streams (the RFB relay, `cua-driver mcp`). A client that joins its arguments
  into one string, like `ssh`, needs a small wrapper that quotes them.
- Honor `user` when the client can switch accounts, otherwise ignore it.
- The sandbox image needs the desktop stack, exactly as for the built-in
  backends (see [Bot Screen](../user-guide/features/bot-screen.md)).

Without `supports_exec_prefix` nothing changes: `bot_desktop.placement: auto`
refuses a sandbox backend rather than run the screen on the gateway host beside
it, and the user can still opt in to the host with `placement: gateway`.

## Session isolation semantics

If your sandbox is **resumed by name** (a durable VM the backend re-attaches
to), set `session_isolated_when_nonpersistent = True`. With
`terminal.container_persistent: false`, each session then gets its own
sandbox identity instead of sharing one — without this, two independent
ephemeral runs could attach one live VM and delete it out from under each
other.
