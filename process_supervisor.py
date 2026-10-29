# ============================================================
# WebMind Process Supervisor
#
# Process lifecycle and clean shutdown engine.
# Guarantees that all child Chrome and server processes are cleanly
# terminated when WebMind exits, preventing orphaned zombie processes
# or lingering profile locks.
# ============================================================

import atexit
import ctypes
import os
import signal
import subprocess
import sys
import time
from typing import List, Set, Union


# Registered processes (Popen objects or integer PIDs)
_tracked_processes: List[subprocess.Popen] = []
_tracked_pids: Set[int] = set()
_is_cleaning_up = False


def register_process(proc: Union[subprocess.Popen, int]):
    """Registers a child process (Popen or PID) to be cleaned up on exit."""
    if isinstance(proc, subprocess.Popen):
        if proc not in _tracked_processes:
            _tracked_processes.append(proc)
        if proc.pid:
            _tracked_pids.add(proc.pid)
    elif isinstance(proc, int) and proc > 0:
        _tracked_pids.add(proc)


def unregister_process(proc: Union[subprocess.Popen, int]):
    """Unregisters a process that has already exited cleanly."""
    if isinstance(proc, subprocess.Popen):
        if proc in _tracked_processes:
            _tracked_processes.remove(proc)
        if proc.pid in _tracked_pids:
            _tracked_pids.remove(proc.pid)
    elif isinstance(proc, int):
        _tracked_pids.discard(proc)


def cleanup_all(verbose: bool = False):
    """
    Terminates all registered child processes and their process trees cleanly.
    Ensures that no orphaned Chrome instances remain to lock profile directories.
    """
    global _is_cleaning_up
    if _is_cleaning_up:
        return
    _is_cleaning_up = True

    # 1. Terminate tracked Popen objects
    for proc in list(_tracked_processes):
        try:
            if proc.poll() is None:
                if verbose:
                    print(f"[ProcessSupervisor] Terminating child PID {proc.pid}...")
                proc.terminate()
        except Exception:
            pass

    # Wait briefly for graceful shutdown
    time.sleep(0.3)

    # 2. Force kill any remaining processes and their process trees on Windows
    for pid in list(_tracked_pids):
        try:
            if os.name == "nt":
                # /T kills the entire process tree (all Chrome child renderer & gpu processes)
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=2.0,
                )
            else:
                os.kill(pid, signal.SIGKILL)
        except Exception:
            pass

    _tracked_processes.clear()
    _tracked_pids.clear()
    _is_cleaning_up = False


# ============================================================
# OS Signal & Console Control Handlers
# ============================================================

def _signal_handler(signum, frame):
    cleanup_all(verbose=False)
    sys.exit(0)


# Register Python exit handlers
atexit.register(cleanup_all)
try:
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
except Exception:
    pass

# Windows Console Control Handler (catches 'X' button, shutdown, logoff)
if os.name == "nt":
    try:
        HandlerRoutine = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_ulong)

        def _console_ctrl_handler(ctrl_type):
            cleanup_all(verbose=False)
            return False  # Pass to next handler / default exit

        _global_ctrl_handler = HandlerRoutine(_console_ctrl_handler)
        ctypes.windll.kernel32.SetConsoleCtrlHandler(_global_ctrl_handler, True)
    except Exception:
        pass
