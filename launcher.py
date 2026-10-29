# ============================================================
# WebMind Unified Launcher
#
# Supports two operational modes:
#
#   Mode 1: Shared Port Mode (Port 8001)
#     - Any selected model runs on http://127.0.0.1:8001
#     - Ideal for quick switching with coding agents (OpenCode)
#     - Independent Chrome profiles & CDP ports are preserved
#
#   Mode 2: Discrete Ports Mode (Ports 8001-8004)
#     - Each model runs on its dedicated port:
#         ChatGPT: 8001, DeepSeek: 8002, Kimi: 8003, GLM: 8004
#     - Supports running models concurrently or launching all 4
#
#   Phase 1 & 2 Production Enhancements:
#     - Universal browser detection (Chrome, Edge, Brave)
#     - Standardized profiles under %LOCALAPPDATA%\WebMind\profiles
#     - Clean process supervision (no zombie locks on exit)
#     - Web Control Panel (Dashboard) on http://127.0.0.1:8000/dashboard
#     - Native Windows System Tray integration
#     - 1-Click Coding Agent setup wizard (OpenCode, Cursor, Continue, Aider)
# ============================================================

import argparse
import os
import socket
import subprocess
import sys
import time

import browser_manager
import profile_manager
import process_supervisor
import agent_config
import config_manager
import updater

DEFAULT_SHARED_PORT = 8001


def get_providers_config():
    """Builds provider mappings dynamically from persisted config."""
    cfg = config_manager.load_config()
    shared_p = cfg.get("shared_port", DEFAULT_SHARED_PORT)
    discrete_ports = cfg.get("ports", {})

    providers = {
        "1": {
            "name": "ChatGPT",
            "description": "OpenAI ChatGPT Web",
            "script": "server.py",
            "port": discrete_ports.get("chatgpt", 8001),
            "cdp_port": 9222,
            "profile": profile_manager.get_profile_dir("chatgpt"),
            "key": "chatgpt",
        },
        "2": {
            "name": "DeepSeek",
            "description": "DeepSeek Chat Web",
            "script": "deepseek_server.py",
            "port": discrete_ports.get("deepseek", 8002),
            "cdp_port": 9223,
            "profile": profile_manager.get_profile_dir("deepseek"),
            "key": "deepseek",
        },
        "3": {
            "name": "Kimi",
            "description": "Moonshot Kimi Web",
            "script": "kimi_server.py",
            "port": discrete_ports.get("kimi", 8003),
            "cdp_port": 9224,
            "profile": profile_manager.get_profile_dir("kimi"),
            "key": "kimi",
        },
        "4": {
            "name": "GLM",
            "description": "Zhipu ChatGLM Web",
            "script": "glm_server.py",
            "port": discrete_ports.get("glm", 8004),
            "cdp_port": 9225,
            "profile": profile_manager.get_profile_dir("glm"),
            "key": "glm",
        },
    }

    shared = {
        "1": {**providers["1"], "port": shared_p, "mode": "shared"},
        "2": {**providers["2"], "port": shared_p, "mode": "shared"},
        "3": {**providers["3"], "port": shared_p, "mode": "shared"},
        "4": {**providers["4"], "port": shared_p, "mode": "shared"},
    }

    discrete = {
        "5": {**providers["1"], "mode": "discrete"},
        "6": {**providers["2"], "mode": "discrete"},
        "7": {**providers["3"], "mode": "discrete"},
        "8": {**providers["4"], "mode": "discrete"},
    }

    menu = {**shared, **discrete}
    return providers, shared, discrete, menu


# Base provider definitions (discrete port configuration)
PROVIDERS, SHARED_PROVIDERS, DISCRETE_PROVIDERS, MENU_OPTIONS = get_providers_config()

NAME_MAP = {
    "chatgpt": "1",
    "deepseek": "2",
    "kimi": "3",
    "glm": "4",
    "chatglm": "4",
    "all": "5",
    "all_discrete": "9",
}


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a TCP port is currently open and bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((host, port)) == 0


def check_and_warn_port(port: int):
    """Notify user if the target port is already in use by a lingering process."""
    if is_port_in_use(port):
        print(f"\n[!] Notice: Port {port} appears to be in use.")
        print(f"    If a previous WebMind instance is running on port {port}, please close it.")
        time.sleep(1)


def print_banner():
    exe, bname = browser_manager.find_chromium_executable()
    root = profile_manager.get_profiles_root()
    cfg = config_manager.load_config()
    sp = cfg.get("shared_port", DEFAULT_SHARED_PORT)
    ports = cfg.get("ports", {})
    cg_p = ports.get("chatgpt", 8001)
    ds_p = ports.get("deepseek", 8002)
    km_p = ports.get("kimi", 8003)
    gm_p = ports.get("glm", 8004)

    print()
    print("=" * 72)
    print(f"               WEBMIND MULTI-AI LAUNCHER (v{updater.CURRENT_VERSION})")
    print(f"  Browser Engine:  {bname} ({exe})")
    print(f"  Profiles Root:   {root}")
    print("=" * 72)
    print(f"  [MODE 1: SHARED PORT {sp}] (Best for OpenCode & quick testing)")
    print(f"    [1] ChatGPT   -> http://127.0.0.1:{sp}")
    print(f"    [2] DeepSeek  -> http://127.0.0.1:{sp}")
    print(f"    [3] Kimi      -> http://127.0.0.1:{sp}")
    print(f"    [4] GLM       -> http://127.0.0.1:{sp}")
    print()
    print("  [MODE 2: DISCRETE PORTS] (For concurrent / dedicated execution)")
    print(f"    [5] ChatGPT   -> http://127.0.0.1:{cg_p}  (Dedicated)")
    print(f"    [6] DeepSeek  -> http://127.0.0.1:{ds_p}  (Dedicated)")
    print(f"    [7] Kimi      -> http://127.0.0.1:{km_p}  (Dedicated)")
    print(f"    [8] GLM       -> http://127.0.0.1:{gm_p}  (Dedicated)")
    print(f"    [9] All       -> Launch ALL 4 models concurrently")
    print()
    print("  [WEB CONTROL PANEL & SYSTEM TRAY]")
    print("    [D] Web Dashboard    -> http://127.0.0.1:8000/dashboard")
    print("    [T] System Tray App  -> Background Taskbar Notification Icon")
    print()
    print("  [CONFIGURATION & SETUP]")
    print("    [C] Configure Coding Agents (OpenCode, Cursor, Continue, Aider)")
    print("    [P] Customize Ports (Shared & Dedicated)")
    print(f"    [U] Check for Updates (v{updater.CURRENT_VERSION})")
    print()
    print("    [0] Exit")
    print("=" * 72)


def handle_agent_config():
    cfg = config_manager.load_config()
    sp = cfg.get("shared_port", DEFAULT_SHARED_PORT)
    print("\n" + "=" * 72)
    print("                CODING AGENT INTEGRATION WIZARD")
    print("=" * 72)
    print("\n[1] OpenCode (opencode.json)")
    print("    WebMind models can be added directly to your OpenCode configuration.")
    default_path = agent_config.get_default_opencode_config_path()
    print(f"    Target path: {default_path}")
    print("\n[2] Cursor IDE")
    print("   " + agent_config.get_cursor_instructions(sp).replace("\n", "\n   "))
    print("\n[3] Continue.dev")
    print("   " + agent_config.get_continue_config_yaml(sp).replace("\n", "\n   "))
    print("\n[4] Aider")
    print("    " + agent_config.get_aider_command(sp))
    print("=" * 72)
    choice = input("\nOptions: [W] Write to opencode.json, [Enter] Back to menu: ").strip().lower()
    if choice == "w":
        saved = agent_config.write_opencode_config()
        print(f"\n[OK] Successfully updated OpenCode configuration at:\n     {saved}")
        time.sleep(1.5)


def handle_port_customization():
    cfg = config_manager.load_config()
    shared_p = cfg.get("shared_port", DEFAULT_SHARED_PORT)
    ports = cfg.get("ports", {})
    print("\n" + "=" * 72)
    print("                MANUAL PORT CONFIGURATION")
    print("=" * 72)
    print(f"Current Port Settings (stored in {config_manager.get_config_path()}):")
    print(f"  [S] Shared Mode Port : {shared_p}")
    print(f"  [1] ChatGPT Port     : {ports.get('chatgpt', 8001)}")
    print(f"  [2] DeepSeek Port    : {ports.get('deepseek', 8002)}")
    print(f"  [3] Kimi Port        : {ports.get('kimi', 8003)}")
    print(f"  [4] GLM Port         : {ports.get('glm', 8004)}")
    print(f"  [R] Reset all to defaults")
    print("=" * 72)
    choice = input("\nSelect setting to change [S, 1-4, R, or Enter to go back]: ").strip().lower()
    if choice == "s":
        val = input(f"Enter new Shared Port (1024-65535) [current {shared_p}]: ").strip()
        if val.isdigit() and config_manager.validate_port(int(val)):
            config_manager.set_shared_port(int(val))
            print(f"[OK] Shared port updated to {val}")
        else:
            print("[!] Invalid port number. Must be between 1024 and 65535.")
        time.sleep(1)
    elif choice in ("1", "2", "3", "4"):
        key_map = {"1": "chatgpt", "2": "deepseek", "3": "kimi", "4": "glm"}
        svc_key = key_map[choice]
        current = ports.get(svc_key, 8000 + int(choice))
        val = input(f"Enter new port for {svc_key.upper()} (1024-65535) [current {current}]: ").strip()
        if val.isdigit() and config_manager.validate_port(int(val)):
            config_manager.set_service_port(svc_key, int(val))
            print(f"[OK] {svc_key.upper()} port updated to {val}")
        else:
            print("[!] Invalid port number. Must be between 1024 and 65535.")
        time.sleep(1)
    elif choice == "r":
        config_manager.reset_defaults()
        print("[OK] All ports reset to factory defaults.")
        time.sleep(1)


def handle_check_updates():
    print("\n" + "=" * 72)
    print("                    CHECKING FOR UPDATES")
    print("=" * 72)
    print(f"Current Version: v{updater.CURRENT_VERSION}")
    print("Querying GitHub Releases API...")
    res = updater.check_for_updates()
    if res.get("update_available"):
        print(f"\n[!] A new version is available: v{res['latest_version']}!")
        if res.get("release_name"):
            print(f"    Release: {res['release_name']}")
        if res.get("download_url"):
            print(f"    Download: {res['download_url']}")
        if res.get("release_notes"):
            print("\nRelease Notes:")
            print("    " + res['release_notes'].replace("\n", "\n    "))
    else:
        print(f"\n[OK] WebMind is up to date (v{updater.CURRENT_VERSION}).")
    print("=" * 72)
    try:
        input("\nPress Enter to return to menu...")
    except (EOFError, KeyboardInterrupt):
        pass


def run_provider(provider_config: dict, port_override: int | None = None):
    p = provider_config
    target_port = port_override or p["port"]
    check_and_warn_port(target_port)

    mode_label = p.get("mode", "custom").upper()
    print(f"\n>>> Starting {p['name']} in [{mode_label} MODE]")
    print(f">>> API Endpoint: http://127.0.0.1:{target_port}/v1")
    print(f">>> Chrome CDP:   http://127.0.0.1:{p['cdp_port']}")
    print(f">>> Profile Dir:  {p['profile']}\n")

    env = os.environ.copy()
    env["PORT"] = str(target_port)

    cmd = [sys.executable, p["script"], "--port", str(target_port)]
    try:
        proc = subprocess.Popen(cmd, env=env)
        process_supervisor.register_process(proc)
        proc.wait()
    except KeyboardInterrupt:
        print(f"\nStopped {p['name']}.")
    finally:
        process_supervisor.cleanup_all()


def run_all():
    providers, _, _, _ = get_providers_config()
    print("\n>>> Launching ALL WebMind AI providers concurrently (Mode 2: Discrete Ports):")
    processes = []
    for key in sorted(providers.keys()):
        p = providers[key]
        print(f"    - {p['name']:<10}: http://127.0.0.1:{p['port']} (CDP: {p['cdp_port']})")
        env = os.environ.copy()
        env["PORT"] = str(p["port"])
        cmd = [sys.executable, p["script"], "--port", str(p["port"])]
        proc = subprocess.Popen(cmd, env=env)
        process_supervisor.register_process(proc)
        processes.append((p["name"], proc))
        time.sleep(0.5)

    print("\nAll providers are starting! Press Ctrl+C to stop all.\n")
    try:
        while True:
            time.sleep(1)
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"Notice: {name} process exited with code {proc.returncode}.")
    except KeyboardInterrupt:
        print("\nShutting down all WebMind providers...")
    finally:
        process_supervisor.cleanup_all()


def resolve_provider(ai_name: str, mode: str = "shared", port: int | None = None) -> tuple[dict, int]:
    providers, shared, discrete, menu = get_providers_config()
    cfg = config_manager.load_config()
    sp = cfg.get("shared_port", DEFAULT_SHARED_PORT)

    choice = ai_name.lower().strip()
    key = NAME_MAP.get(choice, choice)
    if key not in providers:
        raise ValueError(f"Unknown AI provider '{ai_name}'. Choose from: chatgpt, deepseek, kimi, glm, all.")

    base_p = providers[key]
    if port is not None:
        target_port = port
        target_mode = "custom"
    elif mode == "discrete":
        target_port = base_p["port"]
        target_mode = "discrete"
    else:
        target_port = sp
        target_mode = "shared"

    config = {**base_p, "port": target_port, "mode": target_mode}
    return config, target_port


def main():
    parser = argparse.ArgumentParser(description="WebMind Multi-AI Launcher")
    parser.add_argument(
        "--ai", "-a",
        type=str,
        help="AI provider to start directly (chatgpt, deepseek, kimi, glm, all)",
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["shared", "discrete"],
        default="shared",
        help="Port mode: 'shared' (port 8001 for all) or 'discrete' (distinct ports 8001-8004)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Explicit port override for the selected AI bridge",
    )
    parser.add_argument(
        "--dashboard", "-d",
        action="store_true",
        help="Start the WebMind Control Panel dashboard (http://127.0.0.1:8000/dashboard)",
    )
    parser.add_argument(
        "--tray", "-t",
        action="store_true",
        help="Start the native Windows system tray icon",
    )
    parser.add_argument(
        "--setup-opencode",
        action="store_true",
        help="Automatically configure OpenCode with WebMind models",
    )
    parser.add_argument(
        "--check-update",
        action="store_true",
        help="Check if a newer version of WebMind is available",
    )
    args = parser.parse_args()

    if args.check_update:
        handle_check_updates()
        return

    if args.dashboard:
        import dashboard
        dashboard.start_dashboard(open_browser_window=True)
        return

    if args.tray:
        import tray_app
        tray_app.run_tray(open_dashboard=True)
        return

    if args.setup_opencode:
        saved = agent_config.write_opencode_config(mode=args.mode)
        print(f"[OK] OpenCode configuration saved to: {saved}")
        return

    if args.ai:
        choice = args.ai.lower().strip()
        if choice in ("all", "5", "9"):
            run_all()
            return
        try:
            config, target_port = resolve_provider(choice, mode=args.mode, port=args.port)
            run_provider(config, port_override=target_port)
        except ValueError as err:
            print(f"Error: {err}")
            sys.exit(1)
        return

    # Interactive menu
    while True:
        providers, shared, discrete, menu = get_providers_config()
        print_banner()
        try:
            choice = input("Enter choice [1-9, D, T, C, P, U, 0 to exit]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if choice in ("0", "exit", "quit", "q"):
            print("Goodbye.")
            break

        if choice in ("d", "dashboard"):
            import dashboard
            dashboard.start_dashboard(open_browser_window=True)
            break

        if choice in ("t", "tray"):
            import tray_app
            tray_app.run_tray(open_dashboard=True)
            break

        if choice in ("c", "config"):
            handle_agent_config()
            continue

        if choice in ("p", "port", "ports"):
            handle_port_customization()
            continue

        if choice in ("u", "update", "updates"):
            handle_check_updates()
            continue

        if choice in ("9", "all"):
            run_all()
            break

        if choice in menu:
            run_provider(menu[choice])
            break

        # Check name fallback
        mapped = NAME_MAP.get(choice)
        if mapped == "5" or mapped == "9":
            run_all()
            break
        elif mapped and mapped in shared:
            run_provider(shared[mapped])
            break
        else:
            print(f"\n[!] Invalid selection '{choice}'. Please enter a valid menu option.")
            time.sleep(1)


if __name__ == "__main__":
    main()