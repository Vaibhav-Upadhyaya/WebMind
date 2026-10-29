# ============================================================
# Unit Tests for Production Phase 3: Hardening & Self-Healing
# ============================================================

import json
import logging
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

from fastapi.testclient import TestClient

import browser_guard
import config_manager
import diagnostics
import logger
import dashboard
import deepseek_server


class TestBrowserGuard(unittest.TestCase):
    def test_is_connection_error(self):
        self.assertTrue(browser_guard.is_connection_error(RuntimeError("Target page, context or browser has been closed")))
        self.assertTrue(browser_guard.is_connection_error(Exception("Connection closed")))
        self.assertTrue(browser_guard.is_connection_error(Exception("Protocol error (Runtime.callFunctionOn)")))
        self.assertFalse(browser_guard.is_connection_error(ValueError("Invalid argument")))
        self.assertFalse(browser_guard.is_connection_error(None))

    def test_is_browser_healthy(self):
        # Disconnected browser
        mock_browser = MagicMock()
        mock_browser.is_connected.return_value = False
        mock_page = MagicMock()
        mock_page.is_closed.return_value = False
        self.assertFalse(browser_guard.is_browser_healthy(mock_browser, mock_page))

        # Closed page
        mock_browser.is_connected.return_value = True
        mock_page.is_closed.return_value = True
        self.assertFalse(browser_guard.is_browser_healthy(mock_browser, mock_page))

        # Responsive page
        mock_page.is_closed.return_value = False
        mock_page.evaluate.return_value = 2
        self.assertTrue(browser_guard.is_browser_healthy(mock_browser, mock_page))

        # Page throwing on evaluate
        mock_page.evaluate.side_effect = RuntimeError("Dead")
        self.assertFalse(browser_guard.is_browser_healthy(mock_browser, mock_page))

    def test_with_recovery_direct_success(self):
        mock_instance = MagicMock()
        mock_instance.browser.is_connected.return_value = True
        mock_instance.page.is_closed.return_value = False
        mock_instance.page.evaluate.return_value = 2

        action = MagicMock(return_value="OK")
        res = browser_guard.with_recovery(
            mock_instance, "TestAI", 9999, "/tmp", "http://test", action, "arg1"
        )
        self.assertEqual(res, "OK")
        action.assert_called_once_with("arg1")

    @patch("browser_guard.relaunch_and_reconnect")
    def test_with_recovery_heals_on_disconnect(self, mock_reconnect):
        mock_reconnect.return_value = True
        mock_instance = MagicMock()
        mock_instance.browser.is_connected.return_value = True
        mock_instance.page.is_closed.return_value = False
        mock_instance.page.evaluate.return_value = 2

        # First call fails with Target closed, second call succeeds
        action = MagicMock(side_effect=[
            RuntimeError("Target page, context or browser has been closed"),
            "Recovered"
        ])

        res = browser_guard.with_recovery(
            mock_instance, "TestAI", 9999, "/tmp", "http://test", action, "prompt"
        )
        self.assertEqual(res, "Recovered")
        self.assertEqual(action.call_count, 2)
        mock_reconnect.assert_called_once()


class TestLogger(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.orig_log_dir_fn = logger.get_log_dir
        logger.get_log_dir = lambda: self.temp_dir

    def tearDown(self):
        logger.get_log_dir = self.orig_log_dir_fn
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sanitizing_filter(self):
        f = logger.SanitizingFilter()
        sanitized = f.sanitize("Authorization: Bearer my_secret_token_123456789")
        self.assertNotIn("my_secret_token_123456789", sanitized)
        self.assertIn("[REDACTED]", sanitized)

        sanitized_sk = f.sanitize("API Key sk-proj-1234567890abcdef")
        self.assertNotIn("sk-proj-1234567890abcdef", sanitized_sk)
        self.assertIn("sk-[REDACTED]", sanitized_sk)

    def test_rotating_logger(self):
        test_log_file = os.path.join(self.temp_dir, "test.log")
        log = logger.get_logger("test_rot", log_file=test_log_file, max_bytes=200, backup_count=2)
        
        # Write enough logs to trigger rotation
        for i in range(20):
            log.info(f"Log line number {i}: padding message to trigger byte threshold")

        files = os.listdir(self.temp_dir)
        self.assertIn("test.log", files)
        # Should have at least one backup rotation
        self.assertTrue(any(f.startswith("test.log.") for f in files))


class TestDiagnostics(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_diagnostics_summary(self):
        summary = diagnostics.get_diagnostics_summary()
        self.assertIn("timestamp", summary)
        self.assertIn("version", summary)
        self.assertIn("system", summary)
        self.assertIn("browser", summary)
        self.assertIn("config", summary)
        self.assertIn("cdp_reachability", summary)
        self.assertIn("profiles", summary)
        self.assertIn("recent_log_tail", summary)

    def test_export_diagnostic_zip(self):
        zip_path = diagnostics.export_diagnostic_zip(output_dir=self.temp_dir)
        self.assertTrue(os.path.exists(zip_path))
        self.assertTrue(zipfile.is_zipfile(zip_path))

        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()
            self.assertIn("diagnostics_summary.json", namelist)
            self.assertIn("webmind.log", namelist)

            data = json.loads(zf.read("diagnostics_summary.json").decode("utf-8"))
            self.assertIn("system", data)


class TestSecurityAndAuthentication(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "config.json")
        self.orig_config_path_fn = config_manager.get_config_path
        config_manager.get_config_path = lambda: self.config_file
        self.client = TestClient(deepseek_server.app)

    def tearDown(self):
        config_manager.get_config_path = self.orig_config_path_fn
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_auth_permissive_by_default(self):
        # Default: api_key_required is False
        res = self.client.get("/v1/models")
        self.assertEqual(res.status_code, 200)

        res2 = self.client.get("/v1/models", headers={"Authorization": "Bearer any-placeholder-key"})
        self.assertEqual(res2.status_code, 200)

    def test_auth_enforced_when_enabled(self):
        config_manager.set_api_key_required(True)
        config_manager.set_api_key("sk-custom-secret-key")

        # Missing token -> 401
        res = self.client.get("/v1/models")
        self.assertEqual(res.status_code, 401)
        data = res.json()
        self.assertIn("error", data)

        # Invalid token -> 401
        res2 = self.client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
        self.assertEqual(res2.status_code, 401)

        # Valid token -> 200
        res3 = self.client.get("/v1/models", headers={"Authorization": "Bearer sk-custom-secret-key"})
        self.assertEqual(res3.status_code, 200)


class TestDashboardPhase3(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "config.json")
        self.orig_config_path_fn = config_manager.get_config_path
        config_manager.get_config_path = lambda: self.config_file
        self.client = TestClient(dashboard.app)

    def tearDown(self):
        config_manager.get_config_path = self.orig_config_path_fn
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_diagnostics_summary_endpoint(self):
        res = self.client.get("/api/diagnostics/summary")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("system", data)
        self.assertIn("browser", data)

    def test_diagnostics_export_endpoint(self):
        res = self.client.get("/api/diagnostics/export")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("content-type"), "application/zip")
        self.assertTrue(len(res.content) > 0)

    def test_security_endpoint(self):
        res = self.client.post("/api/config/security", json={
            "api_key_required": True,
            "api_key": "sk-my-new-token",
        })
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertTrue(data["api_key_required"])
        self.assertEqual(data["api_key"], "sk-my-new-token")

        cfg = config_manager.load_config()
        self.assertTrue(cfg["api_key_required"])
        self.assertEqual(cfg["api_key"], "sk-my-new-token")


if __name__ == "__main__":
    unittest.main()
