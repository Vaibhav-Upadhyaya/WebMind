# ============================================================
# WebMind Native System Tray Application
#
# Sits quietly in the Windows notification tray area with zero
# extra pip dependencies (uses Python's win32gui).
# Provides 1-click model toggling, dashboard opening, and clean
# process termination.
# ============================================================

import os
import sys
import threading
import time
import webbrowser
from typing import Dict, Any

import agent_config
import browser_manager
from service_manager import manager
import process_supervisor

DASHBOARD_PORT = 8000
DASHBOARD_URL = f"http://127.0.0.1:{DASHBOARD_PORT}/dashboard"


def start_dashboard_thread():
    """Runs the FastAPI Web Dashboard in a daemon background thread."""
    def _run():
        import uvicorn
        from dashboard import app
        uvicorn.run(app, host="127.0.0.1", port=DASHBOARD_PORT, log_level="warning")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


# ============================================================
# Windows Native System Tray Implementation (win32gui)
# ============================================================

class WebMindTrayApp:
    def __init__(self):
        import win32api
        import win32con
        import win32gui

        self.win32api = win32api
        self.win32con = win32con
        self.win32gui = win32gui

        self.class_name = "WebMindTrayWindowClass"
        self.hwnd = None
        self.hicon = None
        self.wm_notify = win32con.WM_USER + 20

        # Menu IDs
        self.ID_OPEN_DASHBOARD = 1000
        self.ID_TOGGLE_CHATGPT = 1001
        self.ID_TOGGLE_DEEPSEEK = 1002
        self.ID_TOGGLE_KIMI = 1003
        self.ID_TOGGLE_GLM = 1004
        self.ID_LAUNCH_ALL = 1005
        self.ID_CONFIG_OPENCODE = 1006
        self.ID_EXIT = 1007

        self.id_to_service = {
            self.ID_TOGGLE_CHATGPT: "chatgpt",
            self.ID_TOGGLE_DEEPSEEK: "deepseek",
            self.ID_TOGGLE_KIMI: "kimi",
            self.ID_TOGGLE_GLM: "glm",
        }

    def _create_window(self):
        wc = self.win32gui.WNDCLASS()
        wc.hInstance = self.win32gui.GetModuleHandle(None)
        wc.lpszClassName = self.class_name
        wc.style = self.win32con.CS_VREDRAW | self.win32con.CS_HREDRAW
        wc.hCursor = self.win32gui.LoadCursor(0, self.win32con.IDC_ARROW)
        wc.hbrBackground = self.win32con.COLOR_WINDOW
        wc.lpfnWndProc = self._wnd_proc

        try:
            self.win32gui.RegisterClass(wc)
        except Exception:
            pass  # Already registered

        self.hwnd = self.win32gui.CreateWindowEx(
            0,
            self.class_name,
            "WebMindTrayWindow",
            self.win32con.WS_OVERLAPPEDWINDOW,
            0, 0, self.win32con.CW_USEDEFAULT, self.win32con.CW_USEDEFAULT,
            0, 0, wc.hInstance, None
        )
        self.win32gui.UpdateWindow(self.hwnd)

    def _setup_icon(self):
        # Load standard application icon
        self.hicon = self.win32gui.LoadIcon(0, self.win32con.IDI_APPLICATION)
        flags = self.win32gui.NIF_ICON | self.win32gui.NIF_MESSAGE | self.win32gui.NIF_TIP
        nid = (self.hwnd, 0, flags, self.wm_notify, self.hicon, "WebMind AI Bridge")
        self.win32gui.Shell_NotifyIcon(self.win32gui.NIM_ADD, nid)

    def _show_menu(self):
        status = manager.get_status()
        services = status.get("services", {})

        menu = self.win32gui.CreatePopupMenu()
        self.win32gui.AppendMenu(menu, self.win32con.MF_STRING, self.ID_OPEN_DASHBOARD, "WebMind Dashboard")
        self.win32gui.AppendMenu(menu, self.win32con.MF_SEPARATOR, 0, "")

        # Service entries
        for mid, svc_key in self.id_to_service.items():
            s = services.get(svc_key, {})
            is_running = s.get("status") == "RUNNING"
            port = s.get("port", 8001)
            icon = "[ON]" if is_running else "[OFF]"
            action = "Stop" if is_running else "Start"
            label = f"{icon} {s.get('name', svc_key.capitalize())} ({port}) - Click to {action}"
            self.win32gui.AppendMenu(menu, self.win32con.MF_STRING, mid, label)

        self.win32gui.AppendMenu(menu, self.win32con.MF_SEPARATOR, 0, "")
        self.win32gui.AppendMenu(menu, self.win32con.MF_STRING, self.ID_LAUNCH_ALL, "Launch All Models (Ports 8001-8004)")
        self.win32gui.AppendMenu(menu, self.win32con.MF_STRING, self.ID_CONFIG_OPENCODE, "1-Click Configure OpenCode")
        self.win32gui.AppendMenu(menu, self.win32con.MF_SEPARATOR, 0, "")
        self.win32gui.AppendMenu(menu, self.win32con.MF_STRING, self.ID_EXIT, "Exit WebMind (Clean All)")

        pos = self.win32gui.GetCursorPos()
        self.win32gui.SetForegroundWindow(self.hwnd)
        self.win32gui.TrackPopupMenu(menu, self.win32con.TPM_LEFTALIGN, pos[0], pos[1], 0, self.hwnd, None)
        self.win32gui.PostMessage(self.hwnd, self.win32con.WM_NULL, 0, 0)

    def _wnd_proc(self, hwnd, msg, wparam, lparam):
        if msg == self.wm_notify:
            if lparam == self.win32con.WM_LBUTTONDBLCLK:
                # Double click -> open dashboard
                webbrowser.open(DASHBOARD_URL)
            elif lparam == self.win32con.WM_RBUTTONUP:
                # Right click -> context menu
                self._show_menu()
            return True

        if msg == self.win32con.WM_COMMAND:
            cmd_id = wparam & 0xFFFF
            if cmd_id == self.ID_OPEN_DASHBOARD:
                webbrowser.open(DASHBOARD_URL)
            elif cmd_id in self.id_to_service:
                svc_key = self.id_to_service[cmd_id]
                status = manager.get_status().get("services", {}).get(svc_key, {})
                if status.get("status") == "RUNNING":
                    manager.stop_service(svc_key)
                else:
                    manager.start_service(svc_key)
            elif cmd_id == self.ID_LAUNCH_ALL:
                manager.set_mode("discrete")
                for key in ["chatgpt", "deepseek", "kimi", "glm"]:
                    manager.start_service(key)
            elif cmd_id == self.ID_CONFIG_OPENCODE:
                saved = agent_config.write_opencode_config(mode=manager._active_mode)
                print(f"[WebMind Tray] Configured OpenCode at: {saved}")
            elif cmd_id == self.ID_EXIT:
                self.stop()
            return True

        if msg == self.win32con.WM_DESTROY:
            self.stop()
            return True

        return self.win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

    def start(self, open_dashboard: bool = True):
        """Starts the tray application and message loop."""
        print("[WebMind Tray] Initializing Windows System Tray...")
        start_dashboard_thread()
        time.sleep(0.5)

        if open_dashboard:
            webbrowser.open(DASHBOARD_URL)

        self._create_window()
        self._setup_icon()
        print(f"[WebMind Tray] Active in system tray. Dashboard at {DASHBOARD_URL}")
        self.win32gui.PumpMessages()

    def stop(self):
        """Cleans up the tray icon and terminates all running bridges."""
        print("[WebMind Tray] Exiting and terminating all child processes...")
        try:
            nid = (self.hwnd, 0)
            self.win32gui.Shell_NotifyIcon(self.win32gui.NIM_DELETE, nid)
        except Exception:
            pass
        manager.stop_all()
        process_supervisor.cleanup_all()
        self.win32gui.PostQuitMessage(0)
        sys.exit(0)


def run_tray(open_dashboard: bool = True):
    if os.name != "nt":
        print("System tray app is currently supported on Windows. Launching dashboard instead.")
        import dashboard
        dashboard.start_dashboard(open_browser_window=open_dashboard)
        return

    app = WebMindTrayApp()
    try:
        app.start(open_dashboard=open_dashboard)
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    run_tray(open_dashboard=True)
