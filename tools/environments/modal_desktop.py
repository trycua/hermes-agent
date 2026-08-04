"""Modal-backed desktop environment with a task-scoped CUA service."""

from __future__ import annotations

import base64
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tools.computer_use.backend import ActionResult, CaptureResult, ComputerUseBackend, UIElement
from tools.computer_use.transports.base import CuaToolTransport
from tools.computer_use.transports.modal_sandbox import ModalSandboxMcpTransport
from tools.environments.compute_provider import ComputeLease, EnvironmentCapabilities
from tools.environments.modal import ModalEnvironment


@dataclass(frozen=True)
class ModalDesktopConfig:
    image: str = "trycua/cua:latest"
    cwd: str = "/root"
    timeout: int = 60
    persistent_filesystem: bool = True
    cpu: float = 2
    memory: int = 8192
    cua_driver_runtime_command: tuple[str, ...] = ("/bin/cua-driver-mcp-runtime",)
    cua_driver_port: int = 8080
    cua_driver_path: str = "/mcp"
    sandbox_kwargs: Mapping[str, Any] = field(default_factory=dict)


# X window mapping lags launch_app (xdg-open returns before the browser maps
# its window), so a capture that names an expected window briefly waits for it.
_WINDOW_WAIT_ATTEMPTS = 8
_WINDOW_WAIT_INTERVAL_S = 0.5


class _TransportComputerBackend(ComputerUseBackend):
    """Adapt a CUA MCP transport to Hermes' synchronous backend contract."""

    def __init__(self, transport: CuaToolTransport):
        self.transport = transport

    def start(self) -> None:
        self.transport.start()

    def stop(self) -> None:
        self.transport.stop()

    def is_available(self) -> bool:
        return self.transport.is_alive()

    def capture(self, mode: str = "som", app: Optional[str] = None, pid: Optional[int] = None,
                window_id: Optional[int] = None) -> CaptureResult:
        """Capture through CUA's current window-state MCP contract.

        The image-configured Modal driver does not expose Hermes' legacy
        ``capture`` tool.  Resolve a target through ``list_windows`` and ask
        the same lease-bound MCP service for ``get_window_state`` instead.
        """
        if (pid is None) != (window_id is None):
            raise ValueError("capture targeting requires both pid and window_id")

        expects_window = app is not None or pid is not None
        attempts = _WINDOW_WAIT_ATTEMPTS if expects_window else 1
        target = None
        for attempt in range(attempts):
            if attempt:
                time.sleep(_WINDOW_WAIT_INTERVAL_S)
            windows = self.list_windows()
            target = _capture_target(windows, app=app, pid=pid, window_id=window_id)
            if target is not None:
                break
        if target is None:
            return self._capture_desktop(mode=mode, app=app)

        target_pid = _positive_int(target.get("pid"))
        target_window_id = _positive_int(target.get("window_id"))
        if target_pid is None or target_window_id is None:
            raise RuntimeError("CUA list_windows returned a target without pid/window_id")

        result = self.transport.call_tool(
            "get_window_state", {"pid": target_pid, "window_id": target_window_id},
        )
        data = _result_data(result)
        raw_elements = data.get("elements", [])
        elements = [
            _element_from(item, app=str(target.get("app_name", target.get("app", ""))),
                          pid=target_pid, window_id=target_window_id)
            for item in raw_elements if isinstance(item, Mapping)
        ]
        if mode == "vision":
            elements = []
        png_b64, image_mime_type = _image_from_result(result, data)
        width = _positive_int(data.get("width")) or 0
        height = _positive_int(data.get("height")) or 0
        if png_b64 and (not width or not height):
            width, height = _png_dimensions(png_b64)
        return CaptureResult(
            mode=mode, width=width, height=height, png_b64=png_b64,
            elements=elements,
            app=str(target.get("app_name", target.get("app", app or ""))),
            window_title=str(data.get("title", data.get("window_title", target.get("title", "")))),
            png_bytes_len=_base64_size(png_b64), image_mime_type=image_mime_type,
        )

    def _capture_desktop(self, *, mode: str, app: Optional[str]) -> CaptureResult:
        """Full-screen screenshot for captures that resolved no window.

        Reporting the real (possibly empty) screen beats the 0x0 result this
        used to return, which read as a dead desktop.
        """
        result = self.transport.call_tool("get_desktop_state", {})
        data = _result_data(result)
        png_b64, image_mime_type = _image_from_result(result, data)
        width = (
            _positive_int(data.get("screenshot_width"))
            or _positive_int(data.get("screen_width"))
            or 0
        )
        height = (
            _positive_int(data.get("screenshot_height"))
            or _positive_int(data.get("screen_height"))
            or 0
        )
        if png_b64 and (not width or not height):
            width, height = _png_dimensions(png_b64)
        return CaptureResult(
            mode=mode, width=width, height=height, png_b64=png_b64,
            app=app or "", png_bytes_len=_base64_size(png_b64),
            image_mime_type=image_mime_type,
        )

    def click(self, **kwargs: Any) -> ActionResult:
        return self._action("click", kwargs)

    def drag(self, **kwargs: Any) -> ActionResult:
        return self._action("drag", kwargs)

    def scroll(self, **kwargs: Any) -> ActionResult:
        return self._action("scroll", kwargs)

    def type_text(self, text: str, **kwargs: Any) -> ActionResult:
        return self._action("type_text", {"text": text, **kwargs})

    def key(self, keys: str, **kwargs: Any) -> ActionResult:
        return self._action("hotkey", {"keys": keys, **kwargs})

    def list_apps(self) -> List[Dict[str, Any]]:
        result = _result_data(self.transport.call_tool("list_apps", {}))
        apps = result.get("apps", result.get("result", []))
        return apps if isinstance(apps, list) else []

    def list_windows(self) -> List[Dict[str, Any]]:
        result = _result_data(self.transport.call_tool("list_windows", {}))
        windows = result.get("windows", result.get("result", []))
        return windows if isinstance(windows, list) else []

    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        return self._action("focus_app", {"app": app, "raise_window": raise_window})

    def launch_app(
        self,
        *,
        bundle_id: Optional[str] = None,
        name: Optional[str] = None,
        urls: Optional[List[str]] = None,
        additional_arguments: Optional[List[str]] = None,
        creates_new_application_instance: bool = False,
    ) -> Dict[str, Any]:
        if not bundle_id and not name:
            raise ValueError("launch_app requires either bundle_id or name")
        arguments: Dict[str, Any] = {}
        if bundle_id:
            arguments["bundle_id"] = bundle_id
        if name:
            arguments["name"] = name
        if urls:
            arguments["urls"] = list(urls)
        if additional_arguments:
            arguments["additional_arguments"] = list(additional_arguments)
        if creates_new_application_instance:
            arguments["creates_new_application_instance"] = True
        return dict(_result_data(self.transport.call_tool("launch_app", arguments)))

    def kill_app(self, *, pid: int) -> ActionResult:
        return self._action("kill_app", {"pid": int(pid)})

    def bring_to_front(self, *, pid: int, window_id: Optional[int] = None) -> ActionResult:
        arguments: Dict[str, Any] = {"pid": int(pid)}
        if window_id is not None:
            arguments["window_id"] = int(window_id)
        return self._action("bring_to_front", arguments)

    def set_value(self, value: str, element: Optional[int] = None) -> ActionResult:
        return self._action("set_value", {"value": value, "element": element})

    def _action(self, action: str, arguments: Mapping[str, Any]) -> ActionResult:
        data = _result_data(self.transport.call_tool(action, arguments))
        return ActionResult(
            ok=bool(data.get("ok", True)), action=action, message=str(data.get("message", "")),
            meta=dict(data.get("meta", {})) if isinstance(data.get("meta"), Mapping) else {},
        )


def _result_data(result: Mapping[str, Any]) -> Mapping[str, Any]:
    if result.get("isError") is True:
        raise RuntimeError(_mcp_error_message(result))
    structured = result.get("structuredContent")
    if isinstance(structured, Mapping):
        return structured
    return result


def _mcp_error_message(result: Mapping[str, Any]) -> str:
    error = result.get("error")
    if isinstance(error, Mapping):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
    data = result.get("data")
    if isinstance(data, str) and data:
        return data
    content = result.get("content")
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        messages = [
            item.get("text") for item in content
            if isinstance(item, Mapping) and isinstance(item.get("text"), str)
        ]
        if messages:
            return "\n".join(messages)
    return "CUA MCP tool call failed"


def _capture_target(
    windows: Sequence[Mapping[str, Any]], *, app: Optional[str], pid: Optional[int], window_id: Optional[int],
) -> Optional[Mapping[str, Any]]:
    candidates = [window for window in windows if window.get("is_on_screen", not window.get("off_screen", False))]
    if pid is not None and window_id is not None:
        return next(
            (window for window in candidates if window.get("pid") == pid and window.get("window_id") == window_id),
            None,
        )
    if app:
        needle = app.casefold()
        candidates = [
            window for window in candidates
            if needle in str(window.get("app_name", window.get("app", ""))).casefold()
        ]
    return candidates[0] if candidates else None


def _image_from_result(
    result: Mapping[str, Any], data: Mapping[str, Any],
) -> tuple[Optional[str], Optional[str]]:
    png_b64 = data.get("screenshot_png_b64") or data.get("png_b64") or data.get("image")
    image_mime_type = data.get("screenshot_mime_type")
    if not isinstance(png_b64, str):
        images = result.get("images")
        if isinstance(images, Sequence) and not isinstance(images, (str, bytes)) and images:
            png_b64 = images[0]
            mimes = result.get("image_mime_types")
            if isinstance(mimes, Sequence) and not isinstance(mimes, (str, bytes)) and mimes:
                image_mime_type = mimes[0]
    if not isinstance(png_b64, str):
        content = result.get("content")
        if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
            for item in content:
                if isinstance(item, Mapping) and item.get("type") == "image" and isinstance(item.get("data"), str):
                    png_b64 = item["data"]
                    image_mime_type = item.get("mimeType")
                    break
    if isinstance(png_b64, str) and png_b64.startswith("data:"):
        png_b64 = png_b64.split(",", 1)[-1]
    return (
        png_b64 if isinstance(png_b64, str) else None,
        image_mime_type if isinstance(image_mime_type, str) else None,
    )


def _png_dimensions(png_b64: str) -> tuple[int, int]:
    try:
        raw = base64.b64decode(png_b64, validate=False)
    except Exception:
        return 0, 0
    if raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) >= 24:
        return int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
    return 0, 0


def _base64_size(png_b64: Optional[str]) -> int:
    if not png_b64:
        return 0
    try:
        return len(base64.b64decode(png_b64, validate=False))
    except Exception:
        return len(png_b64) * 3 // 4


def _positive_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _element_from(value: Mapping[str, Any], *, app: str = "", pid: int = 0, window_id: int = 0) -> UIElement:
    bounds = value.get("bounds")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        frame = value.get("frame")
        bounds = (
            frame.get("x", 0), frame.get("y", 0), frame.get("w", 0), frame.get("h", 0)
        ) if isinstance(frame, Mapping) else (0, 0, 0, 0)
    return UIElement(
        index=int(value.get("element_index", value.get("index", 0))), role=str(value.get("role", "")),
        label=str(value.get("label", "")), bounds=tuple(int(part) for part in bounds),
        app=str(value.get("app", app)), pid=int(value.get("pid", pid)),
        window_id=int(value.get("window_id", window_id)),
        element_token=value.get("element_token") if isinstance(value.get("element_token"), str) else None,
    )


class ModalDesktopEnvironment(ModalEnvironment):
    """Modal terminal environment exposing the paired CUA driver backend."""

    def __init__(self, *, compute_lease: ComputeLease, config: ModalDesktopConfig):
        self._compute_lease = compute_lease
        self._desktop_config = config
        self._computer_backend: ComputerUseBackend | None = None
        sandbox_kwargs = {"cpu": config.cpu, "memory": config.memory, **dict(config.sandbox_kwargs)}
        configured_ports = sandbox_kwargs.pop("encrypted_ports", ())
        if isinstance(configured_ports, (str, bytes)) or not isinstance(configured_ports, Sequence):
            raise ValueError("modal.sandbox_kwargs.encrypted_ports must be a sequence")
        encrypted_ports = {config.cua_driver_port}
        for port in configured_ports:
            if not isinstance(port, int):
                raise ValueError("modal.sandbox_kwargs.encrypted_ports must contain integers")
            encrypted_ports.add(port)
        sandbox_kwargs["encrypted_ports"] = sorted(encrypted_ports)
        super().__init__(
            image=config.image, cwd=config.cwd, timeout=config.timeout,
            modal_sandbox_kwargs=sandbox_kwargs,
            persistent_filesystem=config.persistent_filesystem, task_id=compute_lease.task_id,
            sandbox_command=config.cua_driver_runtime_command,
        )

    @property
    def compute_lease(self) -> ComputeLease:
        return self._compute_lease

    def get_computer_backend(self) -> ComputerUseBackend:
        if self._computer_backend is None:
            self._computer_backend = _TransportComputerBackend(
                ModalSandboxMcpTransport(
                    self._sandbox, self._worker,
                    port=self._desktop_config.cua_driver_port,
                    path=self._desktop_config.cua_driver_path,
                )
            )
            self._computer_backend.start()
        return self._computer_backend

    def cleanup(self):
        if self._computer_backend is not None:
            self._computer_backend.stop()
        super().cleanup()


class ModalDesktopProvider:
    """Provision Modal desktop sandboxes and expose their shared lease."""

    name = "modal"

    def __init__(self, config: ModalDesktopConfig | None = None):
        self.config = config or ModalDesktopConfig()

    def acquire(self, task_id: str, *, image: str | None = None,
                capabilities: Sequence[str] | None = None) -> ComputeLease:
        enabled = EnvironmentCapabilities(computer_use=True)
        requested = frozenset(capabilities or enabled.to_capabilities())
        missing = requested - enabled.to_capabilities()
        if missing:
            raise ValueError(f"Modal desktop image lacks requested capabilities: {sorted(missing)}")
        return ComputeLease(
            task_id=task_id, lease_id=uuid.uuid4().hex, provider=self.name,
            image=image or self.config.image, capabilities=enabled,
        )

    def create_environment(self, lease: ComputeLease) -> ModalDesktopEnvironment:
        config = self.config if lease.image == self.config.image else ModalDesktopConfig(
            image=lease.image, cwd=self.config.cwd, timeout=self.config.timeout,
            persistent_filesystem=self.config.persistent_filesystem, cpu=self.config.cpu,
            memory=self.config.memory,
            cua_driver_runtime_command=self.config.cua_driver_runtime_command,
            cua_driver_port=self.config.cua_driver_port,
            cua_driver_path=self.config.cua_driver_path,
            sandbox_kwargs=self.config.sandbox_kwargs,
        )
        return ModalDesktopEnvironment(compute_lease=lease, config=config)

    def release(self, lease: ComputeLease) -> None:
        return None
