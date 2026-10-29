# ============================================================
# Phase 1 Production Packaging Test Suite
#
# Validates browser detection, profile management, process
# supervision, and coding agent configuration generation.
# ============================================================

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import browser_manager
import profile_manager
import process_supervisor
import agent_config


class TestBrowserManager(unittest.TestCase):
    def test_find_chromium_executable(self):
        exe, name = browser_manager.find_chromium_executable()
        self.assertTrue(len(exe) > 0)
        self.assertTrue(len(name) > 0)
        # On Windows, either chrome.exe or msedge.exe should be resolved
        if os.name == "nt" and os.path.isabs(exe):
            self.assertTrue(os.path.exists(exe), f"Resolved browser binary does not exist: {exe}")

    def test_launch_args_construction(self):
        args = browser_manager.get_chromium_launch_args(
            "chrome.exe",
            9222,
            r"C:\Profiles\TestProfile",
            "https://chat.deepseek.com/",
            extra_flags=["--custom-flag=1"],
        )
        self.assertIn("chrome.exe", args)
        self.assertIn("--remote-debugging-port=9222", args)
        self.assertIn(r"--user-data-dir=C:\Profiles\TestProfile", args)
        self.assertIn("--custom-flag=1", args)
        self.assertEqual(args[-1], "https://chat.deepseek.com/")

    def test_custom_browser_override(self):
        with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as tf:
            custom_fake_bin = tf.name
        try:
            os.environ["WEBMIND_BROWSER_PATH"] = custom_fake_bin
            exe, name = browser_manager.find_chromium_executable()
            self.assertEqual(exe, custom_fake_bin)
            self.assertEqual(name, "Custom Browser")
        finally:
            os.environ.pop("WEBMIND_BROWSER_PATH", None)
            try:
                os.remove(custom_fake_bin)
            except Exception:
                pass


class TestProfileManager(unittest.TestCase):
    def test_profile_root_path(self):
        root = profile_manager.get_profiles_root()
        self.assertTrue(root.endswith("profiles"))
        self.assertIn("WebMind", root)

    def test_get_profile_dir_structure(self):
        for provider in ["chatgpt", "deepseek", "kimi", "glm"]:
            pdir = profile_manager.get_profile_dir(provider, auto_migrate=False)
            self.assertTrue(os.path.exists(pdir))

    def test_legacy_profile_migration(self):
        temp_dir = tempfile.mkdtemp()
        try:
            os.environ["WEBMIND_PROFILES_DIR"] = os.path.join(temp_dir, "profiles")
            # Create a mock legacy directory
            mock_legacy = os.path.join(temp_dir, "legacy_home", "WebMindChrome_TestMigrate")
            os.makedirs(mock_legacy, exist_ok=True)
            with open(os.path.join(mock_legacy, "session.txt"), "w") as f:
                f.write("test_active_session")

            # Temporarily register test provider
            profile_manager.PROVIDER_MAP["testmigrate"] = {
                "folder_name": "TestMigrate",
                "legacy_folder": "WebMindChrome_TestMigrate",
            }
            # Override _get_legacy_dir
            orig_legacy = profile_manager._get_legacy_dir
            profile_manager._get_legacy_dir = lambda name: os.path.join(temp_dir, "legacy_home", name)

            # Trigger migration
            migrated = profile_manager.migrate_legacy_profiles(verbose=False)
            self.assertEqual(migrated, 1)

            target = os.path.join(temp_dir, "profiles", "TestMigrate")
            self.assertTrue(os.path.exists(target))
            with open(os.path.join(target, "session.txt"), "r") as f:
                self.assertEqual(f.read(), "test_active_session")
        finally:
            profile_manager._get_legacy_dir = orig_legacy
            profile_manager.PROVIDER_MAP.pop("testmigrate", None)
            os.environ.pop("WEBMIND_PROFILES_DIR", None)
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestProcessSupervisor(unittest.TestCase):
    def test_register_and_cleanup(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
        process_supervisor.register_process(proc)
        self.assertIn(proc, process_supervisor._tracked_processes)
        self.assertIn(proc.pid, process_supervisor._tracked_pids)

        process_supervisor.cleanup_all(verbose=False)
        self.assertIsNotNone(proc.poll())
        self.assertEqual(len(process_supervisor._tracked_processes), 0)
        self.assertEqual(len(process_supervisor._tracked_pids), 0)


class TestAgentConfig(unittest.TestCase):
    def test_generate_opencode_config_shared(self):
        cfg = agent_config.generate_opencode_config(mode="shared", shared_port=8001)
        self.assertIn("$schema", cfg)
        models = cfg["models"]
        self.assertIn("webmind-chatgpt", models)
        self.assertIn("webmind-deepseek", models)
        self.assertIn("webmind-kimi", models)
        self.assertIn("webmind-glm", models)
        self.assertEqual(models["webmind-deepseek"]["baseUrl"], "http://127.0.0.1:8001/v1")

    def test_generate_opencode_config_discrete(self):
        cfg = agent_config.generate_opencode_config(mode="discrete")
        models = cfg["models"]
        self.assertEqual(models["webmind-chatgpt"]["baseUrl"], "http://127.0.0.1:8001/v1")
        self.assertEqual(models["webmind-deepseek"]["baseUrl"], "http://127.0.0.1:8002/v1")
        self.assertEqual(models["webmind-kimi"]["baseUrl"], "http://127.0.0.1:8003/v1")
        self.assertEqual(models["webmind-glm"]["baseUrl"], "http://127.0.0.1:8004/v1")

    def test_write_opencode_config_merging(self):
        temp_dir = tempfile.mkdtemp()
        cfg_file = os.path.join(temp_dir, "opencode.json")
        try:
            # Write an existing config with custom model
            with open(cfg_file, "w") as f:
                json.dump({"models": {"custom-model": {"type": "anthropic"}}}, f)

            # Merge WebMind models
            saved = agent_config.write_opencode_config(target_path=cfg_file, mode="shared")
            self.assertEqual(saved, cfg_file)

            with open(cfg_file, "r") as f:
                data = json.load(f)

            # Both the original custom model and WebMind models should exist
            self.assertIn("custom-model", data["models"])
            self.assertIn("webmind-deepseek", data["models"])
            self.assertIn("webmind-chatgpt", data["models"])
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_cursor_and_continue_instructions(self):
        cursor = agent_config.get_cursor_instructions(8001)
        self.assertIn("http://127.0.0.1:8001/v1", cursor)
        self.assertIn("sk-webmind", cursor)

        cont = agent_config.get_continue_config_yaml(8001)
        self.assertIn("apiBase: http://127.0.0.1:8001/v1", cont)

        aider = agent_config.get_aider_command(8001)
        self.assertIn("aider --openai-api-base", aider)


if __name__ == "__main__":
    unittest.main()
