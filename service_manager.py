# ============================================================
# WebMind Service Manager
#
# Centralized process lifecycle and port orchestration controller.
# Manages starting, stopping, restarting, and monitoring the 4 AI bridges
# (ChatGPT, DeepSeek, Kimi, GLM) across Shared and Discrete port modes,
# with persistent custom port configuration.
# ============================================================

import os
import socket
import subprocess
import sys
import time
from typing import Dict, Any, Optional

import browser_manager
import profile_manager
import process_supervisor
import config_manager

SERVICES_SPEC: Dict[str, Dict[str, Any]] = {
    "chatgpt": {
        "name": "ChatGPT",
        "description": "OpenAI ChatGPT Web Bridge",
        "script": "server.py",
        "shared_port": 8001,
        "discrete_port": 8001,
        "cdp_port": 9222,
        "url": "https://chatgpt.com/",
    },
    "deepseek": {
        "name": "DeepSeek",
        "description": "DeepSeek Chat Web Bridge",
        "script": "deepseek_server.py",
        "shared_port": 8001,
        "discrete_port": 8002,
        "cdp_port": 9223,
        "url": "https://chat.deepseek.com/",
    },
    "kimi": {
        "name": "Kimi",
        "description": "Moonshot Kimi Web Bridge",
        "script": "kimi_server.py",
        "shared_port": 8001,
        "discrete_port": 8003,
        "cdp_port": 9224,
        "url": "https://kimi.ai/",
    },
    "glm": {
        "name": "GLM",
        "description": "Zhipu ChatGLM Web Bridge",
        "script": "glm_server.py",
        "shared_port": 8001,
        "discrete_port": 8004,
        "cdp_port": 9225,
        "url": "https://chatglm.cn/",
    },
}


class ServiceManager:
    def __init__(self):
        self._services: Dict[str, Dict[str, Any]] = {}
        self._active_mode: str = "shared"  # "shared" or "discrete"
        self._init_state()

    def _init_state(self):
        cfg = config_manager.load_config()
        self._active_mode = cfg.get("mode", "shared")
        shared_p = cfg.get("shared_port", 8001)
        ports = cfg.get("ports", {})

        for key, spec in SERVICES_SPEC.items():
            discrete_p = ports.get(key, spec["discrete_port"])
            active_p = shared_p if self._active_mode == "shared" else discrete_p
            self._services[key] = {
                **spec,
                "profile_dir": profile_manager.get_profile_dir(key, auto_migrate=False),
                "status": "STOPPED",  # STOPPED, STARTING, RUNNING, ERROR
                "shared_port": shared_p,
                "discrete_port": discrete_p,
                "port": active_p,
                "pid": None,
                "proc": None,
                "start_time": None,
                "error": None,
            }

    @property
    def shared_port(self) -> int:
        cfg = config_manager.load_config()
        return cfg.get("shared_port", 8001)

    @property
    def _shared_port(self) -> int:
        return self.shared_port

    def _is_port_open(self, port: int, host: str = "127.0.0.1") -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            return s.connect_ex((host, port)) == 0

    def _refresh_process_status(self, key: str):
        svc = self._services.get(key)
        if not svc:
            return

        proc = svc.get("proc")
        if proc is not None:
            ret = proc.poll()
            if ret is not None:
                # Process has terminated
                process_supervisor.unregister_process(proc)
                svc["proc"] = None
                svc["pid"] = None
                svc["status"] = "STOPPED" if ret == 0 else "ERROR"
                if ret != 0 and not svc.get("error"):
                    svc["error"] = f"Process exited with code {ret}"
            else:
                # Process is actively running
                port = svc["port"]
                if self._is_port_open(port):
                    svc["status"] = "RUNNING"
                elif svc["status"] != "STARTING":
                    svc["status"] = "RUNNING"
        else:
            if svc["status"] not in ("STOPPED", "ERROR"):
                svc["status"] = "STOPPED"

    def get_status(self) -> Dict[str, Any]:
        """Returns the full real-time operational status of WebMind."""
        cfg = config_manager.load_config()
        browser_exe, browser_name = browser_manager.find_chromium_executable()
        root_dir = profile_manager.get_profiles_root()

        services_info = {}
        for key in SERVICES_SPEC.keys():
            self._refresh_process_status(key)
            svc = self._services[key]
            uptime_sec = int(time.time() - svc["start_time"]) if svc.get("start_time") and svc["status"] == "RUNNING" else 0

            services_info[key] = {
                "name": svc["name"],
                "description": svc["description"],
                "status": svc["status"],
                "port": svc["port"],
                "shared_port": svc["shared_port"],
                "discrete_port": svc["discrete_port"],
                "cdp_port": svc["cdp_port"],
                "cdp_connected": browser_manager.is_cdp_available("127.0.0.1", svc["cdp_port"], timeout=0.2),
                "profile_dir": svc["profile_dir"],
                "pid": svc["pid"],
                "uptime": uptime_sec,
                "error": svc.get("error"),
                "endpoint": f"http://127.0.0.1:{svc['port']}/v1",
            }

        return {
            "mode": self._active_mode,
            "shared_port": cfg.get("shared_port", 8001),
            "api_key_required": cfg.get("api_key_required", False),
            "api_key": cfg.get("api_key", "sk-webmind"),
            "browser": {
                "name": browser_name,
                "executable": browser_exe,
            },
            "profiles_root": root_dir,
            "services": services_info,
        }

    def set_mode(self, mode: str):
        """Sets the mode to 'shared' or 'discrete' and persists to config."""
        if mode in ("shared", "discrete"):
            self._active_mode = mode
            cfg = config_manager.load_config()
            cfg["mode"] = mode
            config_manager.save_config(cfg)

            shared_p = cfg.get("shared_port", 8001)
            ports = cfg.get("ports", {})
            for key, svc in self._services.items():
                discrete_p = ports.get(key, svc["discrete_port"])
                svc["shared_port"] = shared_p
                svc["discrete_port"] = discrete_p
                svc["port"] = shared_p if mode == "shared" else discrete_p

    def update_service_port(self, name: str, port: int) -> bool:
        """Updates and persists a custom discrete port for an AI bridge."""
        key = name.lower().strip()
        if key not in self._services:
            return False
        if not config_manager.validate_port(port):
            return False

        config_manager.set_service_port(key, port)
        svc = self._services[key]
        svc["discrete_port"] = port
        if self._active_mode == "discrete":
            svc["port"] = port
            if svc["status"] == "RUNNING":
                self.restart_service(key)
        return True

    def update_shared_port(self, port: int) -> bool:
        """Updates and persists the Shared Mode port."""
        if not config_manager.validate_port(port):
            return False

        config_manager.set_shared_port(port)
        for key, svc in self._services.items():
            svc["shared_port"] = port
            if self._active_mode == "shared":
                svc["port"] = port
                if svc["status"] == "RUNNING":
                    self.restart_service(key)
        return True

    def start_service(self, name: str, mode: Optional[str] = None, port: Optional[int] = None) -> bool:
        """Starts a specific AI provider bridge."""
        key = name.lower().strip()
        if key not in self._services:
            return False

        if mode:
            self.set_mode(mode)

        svc = self._services[key]
        self._refresh_process_status(key)
        if svc["status"] == "RUNNING":
            return True

        target_port = port or (svc["shared_port"] if self._active_mode == "shared" else svc["discrete_port"])
        svc["port"] = target_port

        # In Shared Mode, stop any other provider currently bound to target_port
        if self._active_mode == "shared":
            for other_key, other_svc in self._services.items():
                if other_key != key and other_svc["status"] in ("RUNNING", "STARTING") and other_svc["port"] == target_port:
                    self.stop_service(other_key)

        env = os.environ.copy()
        env["PORT"] = str(target_port)

        script_path = os.path.join(os.getcwd(), svc["script"])
        cmd = [sys.executable, script_path, "--port", str(target_port)]

        try:
            svc["status"] = "STARTING"
            svc["error"] = None
            proc = subprocess.Popen(cmd, env=env)
            svc["proc"] = proc
            svc["pid"] = proc.pid
            svc["start_time"] = time.time()
            process_supervisor.register_process(proc)

            # Wait briefly for HTTP port to open
            for _ in range(15):
                time.sleep(0.2)
                if proc.poll() is not None:
                    break
                if self._is_port_open(target_port):
                    svc["status"] = "RUNNING"
                    return True

            if proc.poll() is None:
                svc["status"] = "RUNNING"
                return True
            else:
                svc["status"] = "ERROR"
                svc["error"] = f"Process terminated prematurely (code {proc.poll()})"
                return False
        except Exception as e:
            svc["status"] = "ERROR"
            svc["error"] = str(e)
            return False

    def stop_service(self, name: str) -> bool:
        """Stops a specific AI provider bridge cleanly."""
        key = name.lower().strip()
        if key not in self._services:
            return False

        svc = self._services[key]
        proc = svc.get("proc")
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=1.5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            process_supervisor.unregister_process(proc)

        svc["proc"] = None
        svc["pid"] = None
        svc["status"] = "STOPPED"
        svc["start_time"] = None
        return True

    def restart_service(self, name: str) -> bool:
        """Restarts a specific AI provider bridge."""
        self.stop_service(name)
        time.sleep(0.5)
        return self.start_service(name)

    def stop_all(self):
        """Stops all running AI provider bridges."""
        for key in list(self._services.keys()):
            self.stop_service(key)
        process_supervisor.cleanup_all()

    def open_browser(self, name: str) -> bool:
        """Launches the dedicated browser window for the specified AI so the user can log in."""
        key = name.lower().strip()
        if key not in self._services:
            return False

        svc = self._services[key]
        chrome_bin, bname = browser_manager.find_chromium_executable()
        profile_dir = svc["profile_dir"]
        cdp_port = svc["cdp_port"]
        target_url = svc["url"]

        args = browser_manager.get_chromium_launch_args(
            chrome_bin,
            cdp_port,
            profile_dir,
            target_url,
        )
        try:
            proc = subprocess.Popen(args)
            process_supervisor.register_process(proc)
            return True
        except Exception:
            return False


# Global singleton instance
manager = ServiceManager()