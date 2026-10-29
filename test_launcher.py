import unittest
from launcher import (
    PROVIDERS,
    SHARED_PROVIDERS,
    DISCRETE_PROVIDERS,
    MENU_OPTIONS,
    NAME_MAP,
    DEFAULT_SHARED_PORT,
    resolve_provider,
)


class TestLauncher(unittest.TestCase):
    def test_providers_configured(self):
        expected_keys = {"1", "2", "3", "4"}
        self.assertEqual(set(PROVIDERS.keys()), expected_keys)
        self.assertEqual(PROVIDERS["1"]["name"], "ChatGPT")
        self.assertEqual(PROVIDERS["2"]["name"], "DeepSeek")
        self.assertEqual(PROVIDERS["3"]["name"], "Kimi")
        self.assertEqual(PROVIDERS["4"]["name"], "GLM")

    def test_distinct_ports_and_profiles(self):
        ports = [p["port"] for p in PROVIDERS.values()]
        cdp_ports = [p["cdp_port"] for p in PROVIDERS.values()]
        profiles = [p["profile"] for p in PROVIDERS.values()]

        # All API server ports, CDP ports, and profiles must be unique
        self.assertEqual(len(ports), len(set(ports)))
        self.assertEqual(len(cdp_ports), len(set(cdp_ports)))
        self.assertEqual(len(profiles), len(set(profiles)))

    def test_name_mapping(self):
        self.assertEqual(NAME_MAP["chatgpt"], "1")
        self.assertEqual(NAME_MAP["deepseek"], "2")
        self.assertEqual(NAME_MAP["kimi"], "3")
        self.assertEqual(NAME_MAP["glm"], "4")
        self.assertEqual(NAME_MAP["all"], "5")

    def test_shared_providers_all_use_port_8001(self):
        for key, p in SHARED_PROVIDERS.items():
            self.assertEqual(p["port"], DEFAULT_SHARED_PORT)
            self.assertEqual(p["mode"], "shared")

        # CDP and profiles remain completely isolated in shared mode
        cdp_ports = [p["cdp_port"] for p in SHARED_PROVIDERS.values()]
        profiles = [p["profile"] for p in SHARED_PROVIDERS.values()]
        self.assertEqual(len(cdp_ports), len(set(cdp_ports)))
        self.assertEqual(len(profiles), len(set(profiles)))

    def test_discrete_providers_use_dedicated_ports(self):
        self.assertEqual(DISCRETE_PROVIDERS["5"]["port"], 8001)
        self.assertEqual(DISCRETE_PROVIDERS["6"]["port"], 8002)
        self.assertEqual(DISCRETE_PROVIDERS["7"]["port"], 8003)
        self.assertEqual(DISCRETE_PROVIDERS["8"]["port"], 8004)

        for p in DISCRETE_PROVIDERS.values():
            self.assertEqual(p["mode"], "discrete")

    def test_menu_options_complete(self):
        expected_menu_keys = {"1", "2", "3", "4", "5", "6", "7", "8"}
        self.assertEqual(set(MENU_OPTIONS.keys()), expected_menu_keys)

    def test_resolve_provider_shared_mode(self):
        cfg, port = resolve_provider("deepseek", mode="shared")
        self.assertEqual(port, 8001)
        self.assertEqual(cfg["name"], "DeepSeek")
        self.assertEqual(cfg["cdp_port"], 9223)

        cfg, port = resolve_provider("kimi", mode="shared")
        self.assertEqual(port, 8001)
        self.assertEqual(cfg["name"], "Kimi")

        cfg, port = resolve_provider("glm", mode="shared")
        self.assertEqual(port, 8001)
        self.assertEqual(cfg["name"], "GLM")

        cfg, port = resolve_provider("chatgpt", mode="shared")
        self.assertEqual(port, 8001)
        self.assertEqual(cfg["name"], "ChatGPT")

    def test_resolve_provider_discrete_mode(self):
        cfg, port = resolve_provider("deepseek", mode="discrete")
        self.assertEqual(port, 8002)

        cfg, port = resolve_provider("kimi", mode="discrete")
        self.assertEqual(port, 8003)

        cfg, port = resolve_provider("glm", mode="discrete")
        self.assertEqual(port, 8004)

        cfg, port = resolve_provider("chatgpt", mode="discrete")
        self.assertEqual(port, 8001)

    def test_resolve_provider_custom_port_override(self):
        cfg, port = resolve_provider("deepseek", port=9999)
        self.assertEqual(port, 9999)
        self.assertEqual(cfg["port"], 9999)

    def test_resolve_provider_invalid_raises(self):
        with self.assertRaises(ValueError):
            resolve_provider("unknown_model")


if __name__ == "__main__":
    unittest.main()
