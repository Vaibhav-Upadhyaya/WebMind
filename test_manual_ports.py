# ============================================================
# Unit Tests for Manual Port Configuration
# ============================================================

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

import config_manager
import service_manager
import agent_config
import dashboard
import launcher


class TestConfigManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "config.json")
        self.orig_config_path_fn = config_manager.get_config_path
        config_manager.get_config_path = lambda: self.config_file

    def tearDown(self):
        config_manager.get_config_path = self.orig_config_path_fn
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_config(self):
        cfg = config_manager.load_config()
        self.assertEqual(cfg["shared_port"], 8001)
        self.assertEqual(cfg["mode"], "shared")
        self.assertEqual(cfg["ports"]["chatgpt"], 8001)
        self.assertEqual(cfg["ports"]["deepseek"], 8002)
        self.assertEqual(cfg["ports"]["kimi"], 8003)
        self.assertEqual(cfg["ports"]["glm"], 8004)

    def test_validate_port(self):
        self.assertTrue(config_manager.validate_port(1024))
        self.assertTrue(config_manager.validate_port(8001))
        self.assertTrue(config_manager.validate_port(65535))
        self.assertFalse(config_manager.validate_port(80))
        self.assertFalse(config_manager.validate_port(1023))
        self.assertFalse(config_manager.validate_port(65536))
        self.assertFalse(config_manager.validate_port("8001"))
        self.assertFalse(config_manager.validate_port(-1))

    def test_set_service_port(self):
        success = config_manager.set_service_port("deepseek", 8999)
        self.assertTrue(success)
        cfg = config_manager.load_config()
        self.assertEqual(cfg["ports"]["deepseek"], 8999)

        # Invalid port
        fail = config_manager.set_service_port("deepseek", 80)
        self.assertFalse(fail)
        cfg2 = config_manager.load_config()
        self.assertEqual(cfg2["ports"]["deepseek"], 8999)

    def test_set_shared_port(self):
        success = config_manager.set_shared_port(9001)
        self.assertTrue(success)
        cfg = config_manager.load_config()
        self.assertEqual(cfg["shared_port"], 9001)

        # Invalid port
        fail = config_manager.set_shared_port(70000)
        self.assertFalse(fail)
        cfg2 = config_manager.load_config()
        self.assertEqual(cfg2["shared_port"], 9001)

    def test_reset_defaults(self):
        config_manager.set_shared_port(9999)
        config_manager.set_service_port("chatgpt", 9876)
        cfg = config_manager.reset_defaults()
        self.assertEqual(cfg["shared_port"], 8001)
        self.assertEqual(cfg["ports"]["chatgpt"], 8001)


class TestServiceManagerPorts(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "config.json")
        self.orig_config_path_fn = config_manager.get_config_path
        config_manager.get_config_path = lambda: self.config_file
        self.sm = service_manager.ServiceManager()

    def tearDown(self):
        config_manager.get_config_path = self.orig_config_path_fn
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_update_service_port(self):
        success = self.sm.update_service_port("kimi", 8333)
        self.assertTrue(success)
        self.assertEqual(self.sm._services["kimi"]["discrete_port"], 8333)
        status = self.sm.get_status()
        self.assertEqual(status["services"]["kimi"]["discrete_port"], 8333)

    def test_update_shared_port(self):
        success = self.sm.update_shared_port(8555)
        self.assertTrue(success)
        self.assertEqual(self.sm.shared_port, 8555)
        self.assertEqual(self.sm._services["chatgpt"]["shared_port"], 8555)
        status = self.sm.get_status()
        self.assertEqual(status["shared_port"], 8555)

    def test_invalid_ports_rejected(self):
        self.assertFalse(self.sm.update_service_port("kimi", 80))
        self.assertFalse(self.sm.update_service_port("unknown_ai", 8080))
        self.assertFalse(self.sm.update_shared_port(70000))


class TestDashboardPortAPI(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "config.json")
        self.orig_config_path_fn = config_manager.get_config_path
        config_manager.get_config_path = lambda: self.config_file
        self.client = TestClient(dashboard.app)

    def tearDown(self):
        config_manager.get_config_path = self.orig_config_path_fn
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_post_config_port_success(self):
        res = self.client.post("/api/config/port", json={"service": "glm", "port": 8888})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["port"], 8888)

    def test_post_config_port_invalid(self):
        res = self.client.post("/api/config/port", json={"service": "glm", "port": 22})
        self.assertEqual(res.status_code, 400)
        data = res.json()
        self.assertEqual(data["status"], "error")

    def test_post_config_shared_port_success(self):
        res = self.client.post("/api/config/shared-port", json={"shared_port": 9005})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["shared_port"], 9005)

    def test_post_config_shared_port_invalid(self):
        res = self.client.post("/api/config/shared-port", json={"shared_port": 99999})
        self.assertEqual(res.status_code, 400)
        data = res.json()
        self.assertEqual(data["status"], "error")

    def test_snippets_reflect_custom_port(self):
        self.client.post("/api/config/shared-port", json={"shared_port": 9100})
        res = self.client.get("/api/config/snippets")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("9100", data["cursor"])
        self.assertIn("9100", data["continue_yaml"])
        self.assertIn("9100", data["aider"])


class TestAgentConfigCustomPorts(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "config.json")
        self.orig_config_path_fn = config_manager.get_config_path
        config_manager.get_config_path = lambda: self.config_file

    def tearDown(self):
        config_manager.get_config_path = self.orig_config_path_fn
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_generate_opencode_reflects_custom_shared_port(self):
        config_manager.set_shared_port(9200)
        cfg = agent_config.generate_opencode_config(mode="shared")
        self.assertEqual(cfg["models"]["webmind-chatgpt"]["baseUrl"], "http://127.0.0.1:9200/v1")
        self.assertEqual(cfg["models"]["webmind-deepseek"]["baseUrl"], "http://127.0.0.1:9200/v1")

    def test_generate_opencode_reflects_custom_discrete_ports(self):
        config_manager.set_service_port("chatgpt", 8111)
        config_manager.set_service_port("deepseek", 8222)
        cfg = agent_config.generate_opencode_config(mode="discrete")
        self.assertEqual(cfg["models"]["webmind-chatgpt"]["baseUrl"], "http://127.0.0.1:8111/v1")
        self.assertEqual(cfg["models"]["webmind-deepseek"]["baseUrl"], "http://127.0.0.1:8222/v1")


class TestLauncherCustomPorts(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "config.json")
        self.orig_config_path_fn = config_manager.get_config_path
        config_manager.get_config_path = lambda: self.config_file

    def tearDown(self):
        config_manager.get_config_path = self.orig_config_path_fn
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_resolve_provider_custom_shared_port(self):
        config_manager.set_shared_port(9300)
        cfg, port = launcher.resolve_provider("chatgpt", mode="shared")
        self.assertEqual(port, 9300)

    def test_resolve_provider_custom_discrete_port(self):
        config_manager.set_service_port("deepseek", 8777)
        cfg, port = launcher.resolve_provider("deepseek", mode="discrete")
        self.assertEqual(port, 8777)


if __name__ == "__main__":
    unittest.main()
