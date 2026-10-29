# ============================================================
# WebMind Standalone Executable Builder (PyInstaller)
#
# Compiles WebMind into a standalone Windows binary (WebMind.exe)
# bundling Python runtime, FastAPI, Playwright, and all bridges.
# ============================================================

import os
import subprocess
import sys


def build_exe():
    print("=" * 60)
    print("WebMind PyInstaller Standalone Executable Builder")
    print("=" * 60)

    try:
        import PyInstaller
        print(f"[OK] Found PyInstaller {PyInstaller.__version__}")
    except ImportError:
        print("[!] PyInstaller is not installed in the current environment.")
        print("    To compile locally, run:")
        print("        pip install pyinstaller")
        print("        python build_exe.py")
        print("    (Note: Release binaries are also automatically built in CI/CD via GitHub Actions.)")
        return False

    spec_content = """# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('README.md', '.'),
    ],
    hiddenimports=[
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespans',
        'uvicorn.lifespans.on',
        'fastapi',
        'starlette',
        'pydantic',
        'playwright',
        'win32gui',
        'win32con',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='WebMind',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='WebMind',
)
"""

    spec_path = os.path.join(os.getcwd(), "WebMind.spec")
    with open(spec_path, "w", encoding="utf-8") as f:
        f.write(spec_content)

    print("Running PyInstaller build...")
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", spec_path]
    res = subprocess.run(cmd)

    if res.returncode == 0:
        print("\n[SUCCESS] Compiled standalone binary at: dist/WebMind/WebMind.exe")
        return True
    else:
        print(f"\n[ERROR] PyInstaller failed with code {res.returncode}")
        return False


if __name__ == "__main__":
    build_exe()
