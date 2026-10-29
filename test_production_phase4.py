# ============================================================
# Unit Tests for Production Phase 4: Standalone Release & Auto-Updates
# ============================================================

import io
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import URLError

from fastapi.testclient import TestClient

import updater
import dom_patches
import dashboard
import launcher


class TestUpdater(unittest.TestCase):
    def test_parse_version(self):
        self.assertEqual(updater.parse_version("1.0.0"), (1, 0, 0))
        self.assertEqual(updater.parse_version("v1.0.2"), (1, 0, 2))
        self.assertEqual(updater.parse_version("V2.1.4-rc1"), (2, 1, 4))
        self.assertEqual(updater.parse_version(""), (0, 0, 0))
        self.assertEqual(updater.parse_version("3.5"), (3, 5, 0))

    def test_is_newer_version(self):
        self.assertTrue(updater.is_newer_version("1.0.1", "1.0.0"))
        self.assertTrue(updater.is_newer_version("v2.0.0", "1.9.9"))
        self.assertTrue(updater.is_newer_version("1.1.0", "1.0.9"))
        self.assertFalse(updater.is_newer_version("1.0.0", "1.0.0"))
        self.assertFalse(updater.is_newer_version("v1.0.0", "1.0.0"))
        self.assertFalse(updater.is_newer_version("0.9.9", "1.0.0"))

    @patch("urllib.request.urlopen")
    def test_check_for_updates_mock_available(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({
            "tag_name": "v1.1.0",
            "name": "WebMind v1.1.0 Feature Release",
            "body": "Added awesome new capabilities",
            "published_at": "2026-10-20T12:00:00Z",
            "html_url": "https://github.com/webmind/webmind/releases/tag/v1.1.0",
            "assets": [
                {"name": "WebMind-v1.1.0.zip", "size": 1234567, "browser_download_url": "https://download/asset.zip"}
            ],
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        res = updater.check_for_updates(current_version="1.0.0")
        self.assertTrue(res["update_available"])
        self.assertEqual(res["latest_version"], "1.1.0")
        self.assertEqual(res["release_name"], "WebMind v1.1.0 Feature Release")
        self.assertEqual(len(res["assets"]), 1)

    @patch("urllib.request.urlopen")
    def test_check_for_updates_mock_up_to_date(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({
            "tag_name": "v1.0.0",
            "name": "WebMind v1.0.0",
            "body": "Initial release",
        }).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        res = updater.check_for_updates(current_version="1.0.0")
        self.assertFalse(res["update_available"])
        self.assertEqual(res["latest_version"], "1.0.0")

    @patch("urllib.request.urlopen", side_effect=URLError("Network unreachable"))
    def test_check_for_updates_timeout_handled(self, mock_urlopen):
        # Should gracefully return update_available=False without throwing
        res = updater.check_for_updates(current_version="1.0.0")
        self.assertFalse(res["update_available"])
        self.assertEqual(res["latest_version"], "1.0.0")

    def test_stage_update_file(self):
        temp_dir = tempfile.mkdtemp()
        try:
            orig_get_update_dir = updater.get_update_dir
            updater.get_update_dir = lambda: temp_dir

            test_file = os.path.join(temp_dir, "test_asset.txt")
            with open(test_file, "w") as f:
                f.write("test update payload")

            staged = updater.stage_update_file(test_file, filename="staged.txt")
            self.assertTrue(os.path.exists(staged))
            with open(staged, "r") as f:
                self.assertEqual(f.read(), "test update payload")
        finally:
            updater.get_update_dir = orig_get_update_dir
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestDomPatches(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.patches_file = os.path.join(self.temp_dir, "patches.json")
        self.orig_get_file = dom_patches.get_patches_file
        dom_patches.get_patches_file = lambda: self.patches_file

    def tearDown(self):
        dom_patches.get_patches_file = self.orig_get_file
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_selectors_present(self):
        for provider in ("chatgpt", "deepseek", "kimi", "glm"):
            sels = dom_patches.get_active_selectors(provider)
            self.assertIn("input", sels)
            self.assertIn("send", sels)
            self.assertTrue(len(sels["input"]) > 0)
            self.assertTrue(len(sels["send"]) > 0)

    def test_apply_patch_and_merge(self):
        # Apply custom selector override for deepseek send button
        custom_selector = 'button[data-custom="send"]:visible'
        success = dom_patches.apply_patch("deepseek", "send", custom_selector)
        self.assertTrue(success)

        # Active selectors should now have the custom selector at the front
        sels = dom_patches.get_active_selectors("deepseek")
        self.assertIn(custom_selector, sels["send"])
        self.assertEqual(sels["send"][0], custom_selector)
        # Built-in default selectors must be retained as fallbacks
        self.assertTrue(len(sels["send"]) > 1)
        self.assertIn('button[aria-label*="Send"]:visible', sels["send"])

        # Other providers remain untouched
        kimi_sels = dom_patches.get_active_selectors("kimi")
        self.assertNotIn(custom_selector, kimi_sels["send"])

    def test_reset_patches(self):
        dom_patches.apply_patch("chatgpt", "input", "#custom-input")
        self.assertTrue(os.path.exists(self.patches_file))

        dom_patches.reset_patches()
        self.assertFalse(os.path.exists(self.patches_file))
        sels = dom_patches.get_active_selectors("chatgpt")
        self.assertNotIn("#custom-input", sels["input"])


class TestDashboardPhase4(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.patches_file = os.path.join(self.temp_dir, "patches.json")
        self.orig_get_file = dom_patches.get_patches_file
        dom_patches.get_patches_file = lambda: self.patches_file
        self.client = TestClient(dashboard.app)

    def tearDown(self):
        dom_patches.get_patches_file = self.orig_get_file
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("updater.check_for_updates")
    def test_update_check_endpoint(self, mock_check):
        mock_check.return_value = {
            "update_available": True,
            "current_version": "1.0.0",
            "latest_version": "1.0.5",
            "download_url": "https://github.com/webmind/webmind/releases/tag/v1.0.5",
        }
        res = self.client.get("/api/update/check")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["update_available"])
        self.assertEqual(data["latest_version"], "1.0.5")

    def test_get_and_post_patches_endpoints(self):
        # Apply patch via POST
        res = self.client.post("/api/update/patches", json={
            "provider": "deepseek",
            "role": "input",
            "selectors": ["#new-chat-box"],
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")

        # Read back via GET
        res_get = self.client.get("/api/update/patches")
        self.assertEqual(res_get.status_code, 200)
        patches = res_get.json()
        self.assertIn("deepseek", patches)


class TestLauncherPhase4(unittest.TestCase):
    def test_banner_and_updater(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            launcher.print_banner()
        output = buf.getvalue()
        self.assertIn(f"v{updater.CURRENT_VERSION}", output)
        self.assertIn("[U] Check for Updates", output)


if __name__ == "__main__":
    unittest.main()
