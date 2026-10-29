# ============================================================
# Phase 2 Production Test Suite: UI & System Tray App
#
# Validates service lifecycle management, dashboard endpoints,
# mode switching, and tray app initialization.
# ============================================================

import os
import unittest
from fastapi.testclient import TestClient

import dashboard
from service_manager import ServiceManager
import tray_app


class TestServiceManager(unittest.TestCase):
    def setUp(self):
        self.mgr = ServiceManager()

    def test_initial_state(self):
        status = self.mgr.get_status()
        self.assertEqual(status["mode"], "shared")
        self.assertIn("browser", status)
        self.assertIn("profiles_root", status)
        services = status["services"]
        for key in ["chatgpt", "deepseek", "kimi", "glm"]:
            self.assertIn(key, services)
            self.assertEqual(services[key]["status"], "STOPPED")
            self.assertEqual(services[key]["port"], 8001)

    def test_mode_switching(self):
        self.mgr.set_mode("discrete")
        status = self.mgr.get_status()
        self.assertEqual(status["mode"], "discrete")
        services = status["services"]
        self.assertEqual(services["chatgpt"]["port"], 8001)
        self.assertEqual(services["deepseek"]["port"], 8002)
        self.assertEqual(services["kimi"]["port"], 8003)
        self.assertEqual(services["glm"]["port"], 8004)

        self.mgr.set_mode("shared")
        status2 = self.mgr.get_status()
        self.assertEqual(status2["mode"], "shared")
        self.assertEqual(status2["services"]["deepseek"]["port"], 8001)

    def test_unknown_service_handling(self):
        self.assertFalse(self.mgr.start_service("nonexistent_ai"))
        self.assertFalse(self.mgr.stop_service("nonexistent_ai"))
        self.assertFalse(self.mgr.open_browser("nonexistent_ai"))


class TestDashboardAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(dashboard.app)

    def test_get_dashboard_html(self):
        res = self.client.get("/dashboard")
        self.assertEqual(res.status_code, 200)
        self.assertIn("WebMind Control Panel", res.text)
        self.assertIn("services-grid", res.text)
        self.assertIn("badge-mode", res.text)

    def test_root_redirect(self):
        res = self.client.get("/", follow_redirects=False)
        self.assertIn(res.status_code, (301, 302, 307))
        self.assertEqual(res.headers.get("location"), "/dashboard")

    def test_api_status(self):
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("mode", data)
        self.assertIn("services", data)
        self.assertIn("deepseek", data["services"])

    def test_api_mode_change(self):
        res = self.client.post("/api/mode", json={"mode": "discrete"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["mode"], "discrete")

        res2 = self.client.post("/api/mode", json={"mode": "shared"})
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.json()["mode"], "shared")

    def test_api_snippets(self):
        res = self.client.get("/api/config/snippets")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("opencode", data)
        self.assertIn("cursor", data)
        self.assertIn("continue_yaml", data)
        self.assertIn("aider", data)

    def test_api_apply_opencode(self):
        res = self.client.post("/api/config/apply-opencode")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "ok")
        self.assertTrue(os.path.exists(res.json()["path"]))


class TestTrayApp(unittest.TestCase):
    def test_tray_app_structure(self):
        if os.name == "nt":
            app = tray_app.WebMindTrayApp()
            self.assertEqual(app.ID_OPEN_DASHBOARD, 1000)
            self.assertIn(app.ID_TOGGLE_CHATGPT, app.id_to_service)
            self.assertIn(app.ID_TOGGLE_DEEPSEEK, app.id_to_service)
            self.assertIn(app.ID_TOGGLE_KIMI, app.id_to_service)
            self.assertIn(app.ID_TOGGLE_GLM, app.id_to_service)


if __name__ == "__main__":
    unittest.main()
