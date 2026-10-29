# ============================================================
# WebMind Web Control Panel & Dashboard API
#
# Exposes a local control panel at http://127.0.0.1:8000/dashboard
# with real-time service controls, mode toggles, login helpers,
# and 1-click coding agent configuration.
# ============================================================

import os
import webbrowser
from typing import Optional, List, Union
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from pydantic import BaseModel
import uvicorn

import agent_config
import config_manager
import diagnostics
import updater
import dom_patches
from service_manager import manager

app = FastAPI(title="WebMind Control Panel", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ServiceActionRequest(BaseModel):
    service: str
    mode: Optional[str] = None
    port: Optional[int] = None


class ModeRequest(BaseModel):
    mode: str


class UpdatePortRequest(BaseModel):
    service: str
    port: int


class UpdateSharedPortRequest(BaseModel):
    shared_port: int


class SecurityConfigRequest(BaseModel):
    api_key_required: bool
    api_key: Optional[str] = None


class PatchRequest(BaseModel):
    provider: str
    role: str
    selectors: Union[List[str], str]


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>WebMind Control Panel</title>
  <style>
    :root {
      --bg: #0d1117;
      --card-bg: #161b22;
      --border: #30363d;
      --text: #c9d1d9;
      --heading: #f0f6fc;
      --accent: #58a6ff;
      --success: #238636;
      --success-glow: #2ea043;
      --warning: #d29922;
      --danger: #da3633;
      --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: var(--bg); color: var(--text); font-family: var(--font); line-height: 1.5; padding: 24px; }
    .container { max-width: 1100px; margin: 0 auto; }
    
    /* Header */
    header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brand h1 { font-size: 24px; color: var(--heading); font-weight: 700; }
    .badge { padding: 4px 10px; font-size: 12px; border-radius: 12px; font-weight: 600; text-transform: uppercase; }
    .badge-mode { background: #1f6feb22; color: #58a6ff; border: 1px solid #1f6feb; }
    
    .meta-bar { display: flex; flex-wrap: wrap; gap: 16px; font-size: 13px; color: #8b949e; background: var(--card-bg); padding: 12px 16px; border-radius: 8px; border: 1px solid var(--border); margin-bottom: 24px; }
    .meta-item strong { color: var(--heading); }

    /* Mode Selector */
    .mode-switch-box { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center; }
    .mode-info h3 { color: var(--heading); font-size: 16px; margin-bottom: 4px; }
    .mode-info p { font-size: 13px; color: #8b949e; }
    .btn-group { display: flex; gap: 8px; }

    /* Service Grid */
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 18px; margin-bottom: 32px; }
    .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 18px; display: flex; flex-direction: column; justify-content: space-between; transition: border-color 0.2s; }
    .card:hover { border-color: #58a6ff66; }
    .card-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; }
    .card-title { font-size: 18px; font-weight: 700; color: var(--heading); }
    .status-pill { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; font-weight: 600; padding: 3px 8px; border-radius: 12px; }
    .status-pill.RUNNING { background: #23863622; color: #3fb950; border: 1px solid #238636; }
    .status-pill.STARTING { background: #d2992222; color: #d29922; border: 1px solid #d29922; }
    .status-pill.STOPPED { background: #30363d55; color: #8b949e; border: 1px solid #30363d; }
    .status-pill.ERROR { background: #da363322; color: #f85149; border: 1px solid #da3633; }
    
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
    
    .card-body { font-size: 13px; color: #8b949e; margin-bottom: 16px; }
    .card-row { display: flex; justify-content: space-between; margin-bottom: 6px; }
    .card-row strong { color: var(--text); }
    .endpoint-code { font-family: monospace; background: #0d1117; padding: 3px 6px; border-radius: 4px; border: 1px solid var(--border); color: #58a6ff; font-size: 12px; word-break: break-all; margin-top: 6px; display: block; }
    
    .card-actions { display: flex; flex-direction: column; gap: 8px; }
    .action-row { display: flex; gap: 8px; }

    /* Buttons */
    button { cursor: pointer; font-family: var(--font); font-size: 13px; font-weight: 600; padding: 8px 12px; border-radius: 6px; border: 1px solid transparent; transition: all 0.15s; text-align: center; }
    .btn-primary { background: var(--success); color: #fff; }
    .btn-primary:hover { background: var(--success-glow); }
    .btn-danger { background: var(--danger); color: #fff; }
    .btn-danger:hover { background: #b62324; }
    .btn-secondary { background: #21262d; color: var(--text); border-color: var(--border); }
    .btn-secondary:hover { background: #30363d; color: var(--heading); }
    .btn-active { background: #1f6feb; color: #fff; border-color: #58a6ff; }
    .btn-sm { padding: 5px 8px; font-size: 12px; flex: 1; }

    /* Agent Config Section */
    .section-title { font-size: 20px; color: var(--heading); margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; }
    .agent-panel { background: var(--card-bg); border: 1px solid var(--border); border-radius: 10px; padding: 20px; }
    .agent-tabs { display: flex; gap: 8px; border-bottom: 1px solid var(--border); padding-bottom: 12px; margin-bottom: 16px; }
    .tab-btn { background: none; border: none; color: #8b949e; font-size: 14px; padding: 6px 12px; border-radius: 6px; }
    .tab-btn.active { color: var(--heading); background: #21262d; font-weight: 700; }
    .tab-content { display: none; }
    .tab-content.active { display: block; }
    pre { background: #0d1117; border: 1px solid var(--border); border-radius: 6px; padding: 14px; font-size: 13px; color: #e6edf3; overflow-x: auto; margin-bottom: 12px; }
    
    .toast { position: fixed; bottom: 20px; right: 20px; background: #238636; color: #fff; padding: 12px 20px; border-radius: 8px; font-weight: 600; font-size: 14px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); opacity: 0; transition: opacity 0.3s; pointer-events: none; }
    .toast.show { opacity: 1; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="brand">
        <h1>WebMind <span style="font-size: 13px; font-weight: 500; color: #8b949e;">v1.0.0</span></h1>
        <span class="badge badge-mode" id="badge-mode">Mode 1: Shared 8001</span>
      </div>
      <div style="display: flex; gap: 8px;">
        <button class="btn-secondary" onclick="checkUpdates(false)">Check Updates</button>
        <button class="btn-secondary" onclick="exportDiagnostics()">Export Diagnostics</button>
        <button class="btn-secondary" onclick="fetchStatus()">Refresh Status</button>
      </div>
    </header>

    <div id="update-banner" style="display: none; background: #1f6feb22; border: 1px solid #1f6feb; border-radius: 8px; padding: 12px 16px; margin-bottom: 20px; align-items: center; justify-content: space-between;">
      <div>
        <strong style="color: var(--heading);" id="update-banner-title">Update Available!</strong>
        <p style="font-size: 13px; color: #8b949e;" id="update-banner-desc">A new version of WebMind is available.</p>
      </div>
      <div style="display: flex; gap: 8px; align-items: center;">
        <a id="update-banner-link" href="#" target="_blank" class="btn-primary" style="text-decoration: none; padding: 6px 12px; font-size: 12px; display: inline-flex; align-items: center;">Download Update</a>
        <button class="btn-secondary btn-sm" onclick="dismissUpdateBanner()">Dismiss</button>
      </div>
    </div>

    <div class="meta-bar">
      <div class="meta-item">Browser Engine: <strong id="meta-browser">Detecting...</strong></div>
      <div class="meta-item">Profiles Root: <strong id="meta-profiles">...</strong></div>
      <div class="meta-item">API Server: <strong id="meta-apiserver">http://127.0.0.1:8001/v1</strong></div>
    </div>

    <div class="mode-switch-box" style="flex-wrap: wrap; gap: 12px;">
      <div class="mode-info">
        <h3 id="mode-title">Operational Port Mode</h3>
        <p id="mode-desc">Shared Mode allows your coding agents to point to one URL and switch models on the fly.</p>
      </div>
      <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
        <div id="shared-port-group" style="display: flex; align-items: center; gap: 6px;">
          <label style="font-size: 13px; color: var(--text);">Shared Port:</label>
          <input type="number" id="input-shared-port" style="width: 75px; background: #0d1117; border: 1px solid var(--border); color: #fff; padding: 4px 6px; border-radius: 4px; font-size: 12px;" min="1024" max="65535">
          <button class="btn-secondary btn-sm" onclick="saveSharedPort()">Save</button>
        </div>
        <div class="btn-group">
          <button id="btn-mode-shared" class="btn-secondary btn-active" onclick="setMode('shared')">Shared</button>
          <button id="btn-mode-discrete" class="btn-secondary" onclick="setMode('discrete')">Discrete</button>
        </div>
      </div>
    </div>

    <div class="section-title">
      <h2>AI Provider Bridges</h2>
      <button class="btn-secondary btn-sm" style="max-width: 140px;" onclick="stopAll()">Stop All</button>
    </div>

    <div class="grid" id="services-grid">
      <!-- Injected by JavaScript -->
    </div>

    <div class="section-title">
      <h2>Security & API Key Authentication</h2>
    </div>

    <div class="agent-panel" style="margin-bottom: 32px;">
      <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
          <h4 style="color: var(--heading); margin-bottom: 4px;">Localhost Token Verification</h4>
          <p style="font-size: 13px; color: #8b949e;">When enabled, requests must include <code>Authorization: Bearer &lt;key&gt;</code>. Disabled by default for zero-setup local access.</p>
        </div>
        <div style="display: flex; align-items: center; gap: 10px;">
          <label style="font-size: 13px; display: flex; align-items: center; gap: 6px; cursor: pointer;">
            <input type="checkbox" id="chk-api-key-required" onchange="toggleApiKeyRequired()"> Require API Key
          </label>
          <input type="text" id="input-api-key" style="background: #0d1117; border: 1px solid var(--border); color: #fff; padding: 4px 8px; border-radius: 4px; font-size: 13px; width: 140px;" placeholder="sk-webmind">
          <button class="btn-secondary btn-sm" onclick="saveApiKey()">Save Key</button>
        </div>
      </div>
    </div>

    <div class="section-title">
      <h2>Coding Agent Integration</h2>
    </div>

    <div class="agent-panel">
      <div class="agent-tabs">
        <button class="tab-btn active" onclick="switchTab('tab-opencode')">OpenCode</button>
        <button class="tab-btn" onclick="switchTab('tab-cursor')">Cursor IDE</button>
        <button class="tab-btn" onclick="switchTab('tab-continue')">Continue.dev</button>
        <button class="tab-btn" onclick="switchTab('tab-aider')">Aider</button>
      </div>

      <div id="tab-opencode" class="tab-content active">
        <p style="font-size: 13px; color: #8b949e; margin-bottom: 10px;">
          Automatically configure OpenCode models or copy the <code>opencode.json</code> block:
        </p>
        <pre id="code-opencode"></pre>
        <div style="display: flex; gap: 10px;">
          <button class="btn-primary" onclick="applyOpenCode()">1-Click Apply to opencode.json</button>
          <button class="btn-secondary" onclick="copySnippet('code-opencode')">Copy JSON</button>
        </div>
      </div>

      <div id="tab-cursor" class="tab-content">
        <p style="font-size: 13px; color: #8b949e; margin-bottom: 10px;">
          Enter in Cursor Settings -> Models -> OpenAI API Key & Base URL:
        </p>
        <pre id="code-cursor"></pre>
        <button class="btn-secondary" onclick="copySnippet('code-cursor')">Copy Setup</button>
      </div>

      <div id="tab-continue" class="tab-content">
        <p style="font-size: 13px; color: #8b949e; margin-bottom: 10px;">
          Add to your <code>~/.continue/config.yaml</code>:
        </p>
        <pre id="code-continue"></pre>
        <button class="btn-secondary" onclick="copySnippet('code-continue')">Copy YAML</button>
      </div>

      <div id="tab-aider" class="tab-content">
        <p style="font-size: 13px; color: #8b949e; margin-bottom: 10px;">
          Run in your terminal:
        </p>
        <pre id="code-aider"></pre>
        <button class="btn-secondary" onclick="copySnippet('code-aider')">Copy Command</button>
      </div>
    </div>
  </div>

  <div id="toast" class="toast">Configuration saved successfully!</div>

  <script>
    let currentStatus = null;

    function showToast(msg) {
      const t = document.getElementById('toast');
      t.innerText = msg;
      t.classList.add('show');
      setTimeout(() => t.classList.remove('show'), 2500);
    }

    async function fetchStatus() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        currentStatus = data;
        render(data);
      } catch (e) {
        console.error('Failed to fetch status:', e);
      }
    }

    function render(data) {
      // Header & Meta
      document.getElementById('meta-browser').innerText = `${data.browser.name} (${data.browser.executable})`;
      document.getElementById('meta-profiles').innerText = data.profiles_root;
      
      const badge = document.getElementById('badge-mode');
      const btnShared = document.getElementById('btn-mode-shared');
      const btnDiscrete = document.getElementById('btn-mode-discrete');
      const sharedPortGroup = document.getElementById('shared-port-group');
      const sharedPortInput = document.getElementById('input-shared-port');
      
      if (sharedPortInput && document.activeElement !== sharedPortInput) {
        sharedPortInput.value = data.shared_port;
      }
      
      if (data.mode === 'shared') {
        badge.innerText = `Mode 1: Shared (${data.shared_port})`;
        btnShared.className = 'btn-secondary btn-active';
        btnDiscrete.className = 'btn-secondary';
        sharedPortGroup.style.display = 'flex';
        document.getElementById('mode-desc').innerText = `Shared Mode: All selected AI models answer on http://127.0.0.1:${data.shared_port}/v1.`;
        document.getElementById('meta-apiserver').innerText = `http://127.0.0.1:${data.shared_port}/v1`;
      } else {
        badge.innerText = 'Mode 2: Discrete Ports';
        btnShared.className = 'btn-secondary';
        btnDiscrete.className = 'btn-secondary btn-active';
        sharedPortGroup.style.display = 'none';
        const portSummary = Object.entries(data.services).map(([k, s]) => `${s.name}: ${s.discrete_port}`).join(', ');
        document.getElementById('mode-desc').innerText = `Discrete Mode: Dedicated ports (${portSummary}).`;
        document.getElementById('meta-apiserver').innerText = 'Discrete Ports Active';
      }

      // Security & API Key
      const chkApiKey = document.getElementById('chk-api-key-required');
      const inputApiKey = document.getElementById('input-api-key');
      if (chkApiKey) chkApiKey.checked = Boolean(data.api_key_required);
      if (inputApiKey && document.activeElement !== inputApiKey) {
        inputApiKey.value = data.api_key || 'sk-webmind';
      }

      // Services Grid
      const grid = document.getElementById('services-grid');
      const activeId = document.activeElement ? document.activeElement.id : null;
      const activeVal = activeId && document.getElementById(activeId) ? document.getElementById(activeId).value : null;

      grid.innerHTML = '';

      for (const [key, svc] of Object.entries(data.services)) {
        const isRunning = svc.status === 'RUNNING';
        const isStarting = svc.status === 'STARTING';
        
        const portSection = data.mode === 'shared' ? `
          <div class="card-row">
            <span>Active Port:</span>
            <strong>${svc.port} (Shared)</strong>
          </div>
          <div class="card-row" style="align-items: center;">
            <span>Discrete Port:</span>
            <div style="display: flex; gap: 4px; align-items: center;">
              <input type="number" id="port-input-${key}" value="${svc.discrete_port}" style="width: 75px; background: #0d1117; border: 1px solid var(--border); color: #fff; padding: 2px 6px; border-radius: 4px; font-size: 12px;" min="1024" max="65535">
              <button class="btn-secondary" style="padding: 2px 8px; font-size: 11px;" onclick="saveServicePort('${key}')">Save</button>
            </div>
          </div>
        ` : `
          <div class="card-row" style="align-items: center;">
            <span>Port:</span>
            <div style="display: flex; gap: 4px; align-items: center;">
              <input type="number" id="port-input-${key}" value="${svc.discrete_port}" style="width: 75px; background: #0d1117; border: 1px solid var(--border); color: #fff; padding: 2px 6px; border-radius: 4px; font-size: 12px;" min="1024" max="65535">
              <button class="btn-secondary" style="padding: 2px 8px; font-size: 11px;" onclick="saveServicePort('${key}')">Save</button>
            </div>
          </div>
        `;

        const card = document.createElement('div');
        card.className = 'card';
        card.innerHTML = `
          <div>
            <div class="card-header">
              <div class="card-title">${svc.name}</div>
              <span class="status-pill ${svc.status}">
                <span class="status-dot"></span> ${svc.status}
              </span>
            </div>
            <div class="card-body">
              ${portSection}
              <div class="card-row">
                <span>Chrome CDP:</span>
                <span>
                  <strong>${svc.cdp_port}</strong>
                  ${svc.cdp_connected 
                    ? '<span class="status-pill RUNNING" style="font-size:10px; margin-left:6px; padding: 2px 6px;"><span class="status-dot"></span> Online</span>' 
                    : '<span class="status-pill STOPPED" style="font-size:10px; margin-left:6px; padding: 2px 6px;"><span class="status-dot"></span> Closed</span>'
                  }
                </span>
              </div>
              ${isRunning ? `<div class="card-row"><span>Uptime:</span><strong>${svc.uptime}s</strong></div>` : ''}
              ${svc.error ? `<div style="color: #f85149; font-size: 11px; margin-top: 4px;">${svc.error}</div>` : ''}
              <span class="endpoint-code">http://127.0.0.1:${svc.port}/v1</span>
            </div>
          </div>
          <div class="card-actions">
            ${isRunning 
              ? `<button class="btn-danger" onclick="stopService('${key}')">Stop</button>`
              : `<button class="btn-primary" onclick="startService('${key}')" ${isStarting ? 'disabled' : ''}>${isStarting ? 'Starting...' : 'Start Service'}</button>`
            }
            <div class="action-row">
              <button class="btn-secondary btn-sm" onclick="restartService('${key}')">Restart</button>
              <button class="btn-secondary btn-sm" onclick="openBrowser('${key}')">Open Browser / Login</button>
            </div>
          </div>
        `;
        grid.appendChild(card);
      }

      if (activeId && document.getElementById(activeId)) {
        const el = document.getElementById(activeId);
        el.value = activeVal;
        el.focus();
      }

      updateSnippets();
    }

    async function setMode(mode) {
      await fetch('/api/mode', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ mode })
      });
      fetchStatus();
    }

    async function saveServicePort(key) {
      const input = document.getElementById(`port-input-${key}`);
      const port = parseInt(input.value, 10);
      if (isNaN(port) || port < 1024 || port > 65535) {
        showToast('Invalid port! Must be between 1024 and 65535.');
        return;
      }
      try {
        const res = await fetch('/api/config/port', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ service: key, port: port })
        });
        const data = await res.json();
        if (res.ok && data.status === 'ok') {
          showToast(`Port for ${key} set to ${port}`);
          fetchStatus();
        } else {
          showToast(data.message || 'Error updating port');
        }
      } catch (e) {
        showToast('Failed to connect to server');
      }
    }

    async function saveSharedPort() {
      const input = document.getElementById('input-shared-port');
      const port = parseInt(input.value, 10);
      if (isNaN(port) || port < 1024 || port > 65535) {
        showToast('Invalid port! Must be between 1024 and 65535.');
        return;
      }
      try {
        const res = await fetch('/api/config/shared-port', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ shared_port: port })
        });
        const data = await res.json();
        if (res.ok && data.status === 'ok') {
          showToast(`Shared port set to ${port}`);
          fetchStatus();
        } else {
          showToast(data.message || 'Error updating shared port');
        }
      } catch (e) {
        showToast('Failed to connect to server');
      }
    }

    function exportDiagnostics() {
      showToast('Generating diagnostic report...');
      window.location.href = '/api/diagnostics/export';
    }

    async function toggleApiKeyRequired() {
      const chk = document.getElementById('chk-api-key-required');
      const keyInput = document.getElementById('input-api-key');
      try {
        const res = await fetch('/api/config/security', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            api_key_required: chk ? chk.checked : false,
            api_key: keyInput ? keyInput.value : undefined
          })
        });
        showToast(chk.checked ? 'API Key verification enabled' : 'API Key verification disabled');
        fetchStatus();
      } catch (e) {
        showToast('Failed to update security settings');
      }
    }

    async function saveApiKey() {
      const chk = document.getElementById('chk-api-key-required');
      const keyInput = document.getElementById('input-api-key');
      const key = keyInput ? keyInput.value.trim() : '';
      if (!key) {
        showToast('API key cannot be empty');
        return;
      }
      try {
        const res = await fetch('/api/config/security', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            api_key_required: chk ? chk.checked : false,
            api_key: key
          })
        });
        showToast('API Key saved successfully');
        fetchStatus();
      } catch (e) {
        showToast('Failed to save API key');
      }
    }

    async function checkUpdates(silent = false) {
      if (!silent) showToast('Checking for updates...');
      try {
        const res = await fetch('/api/update/check');
        const data = await res.json();
        if (data.update_available) {
          const banner = document.getElementById('update-banner');
          document.getElementById('update-banner-title').innerText = `Update Available: v${data.latest_version}`;
          document.getElementById('update-banner-desc').innerText = data.release_name || 'A newer version of WebMind is available on GitHub.';
          const link = document.getElementById('update-banner-link');
          link.href = data.download_url || 'https://github.com/webmind/webmind/releases';
          banner.style.display = 'flex';
          if (!silent) showToast(`New version v${data.latest_version} available!`);
        } else {
          if (!silent) showToast(`WebMind is up to date (v${data.current_version})`);
        }
      } catch (e) {
        if (!silent) showToast('Failed to connect to update server');
      }
    }

    function dismissUpdateBanner() {
      document.getElementById('update-banner').style.display = 'none';
    }

    async function startService(key) {
      await fetch('/api/service/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ service: key })
      });
      fetchStatus();
    }

    async function stopService(key) {
      await fetch('/api/service/stop', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ service: key })
      });
      fetchStatus();
    }

    async function restartService(key) {
      await fetch('/api/service/restart', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ service: key })
      });
      fetchStatus();
    }

    async function stopAll() {
      await fetch('/api/service/stop-all', { method: 'POST' });
      fetchStatus();
    }

    async function openBrowser(key) {
      await fetch('/api/service/open-browser', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ service: key })
      });
      showToast('Opened dedicated browser for ' + key);
    }

    async function applyOpenCode() {
      const res = await fetch('/api/config/apply-opencode', { method: 'POST' });
      const data = await res.json();
      showToast('Config saved to: ' + data.path);
    }

    async function updateSnippets() {
      const res = await fetch('/api/config/snippets');
      const data = await res.json();
      document.getElementById('code-opencode').innerText = JSON.stringify(data.opencode, null, 2);
      document.getElementById('code-cursor').innerText = data.cursor;
      document.getElementById('code-continue').innerText = data.continue_yaml;
      document.getElementById('code-aider').innerText = data.aider;
    }

    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      event.target.classList.add('active');
      document.getElementById(tabId).classList.add('active');
    }

    function copySnippet(elementId) {
      const text = document.getElementById(elementId).innerText;
      navigator.clipboard.writeText(text);
      showToast('Copied to clipboard!');
    }

    // Auto-polling every 2.5 seconds
    fetchStatus();
    checkUpdates(true);
    setInterval(fetchStatus, 2500);
  </script>
</body>
</html>
"""


@app.get("/", response_class=RedirectResponse)
def root():
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)


@app.get("/api/status")
def get_status():
    return manager.get_status()


@app.post("/api/mode")
def set_mode(req: ModeRequest):
    manager.set_mode(req.mode)
    return {"status": "ok", "mode": manager._active_mode}


@app.post("/api/service/start")
def start_service(req: ServiceActionRequest):
    success = manager.start_service(req.service, mode=req.mode, port=req.port)
    return {"status": "ok" if success else "error", "service": req.service}


@app.post("/api/service/stop")
def stop_service(req: ServiceActionRequest):
    success = manager.stop_service(req.service)
    return {"status": "ok" if success else "error", "service": req.service}


@app.post("/api/service/restart")
def restart_service(req: ServiceActionRequest):
    success = manager.restart_service(req.service)
    return {"status": "ok" if success else "error", "service": req.service}


@app.post("/api/service/stop-all")
def stop_all():
    manager.stop_all()
    return {"status": "ok"}


@app.post("/api/service/open-browser")
def open_browser(req: ServiceActionRequest):
    success = manager.open_browser(req.service)
    return {"status": "ok" if success else "error", "service": req.service}


@app.post("/api/config/port")
def update_port(req: UpdatePortRequest):
    success = manager.update_service_port(req.service, req.port)
    if not success:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Invalid port {req.port} (must be 1024-65535) or invalid service"}
        )
    return {"status": "ok", "service": req.service, "port": req.port}


@app.post("/api/config/shared-port")
def update_shared_port(req: UpdateSharedPortRequest):
    success = manager.update_shared_port(req.shared_port)
    if not success:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Invalid shared port {req.shared_port} (must be 1024-65535)"}
        )
    return {"status": "ok", "shared_port": req.shared_port}


@app.post("/api/config/apply-opencode")
def apply_opencode():
    path = agent_config.write_opencode_config(mode=manager._active_mode)
    return {"status": "ok", "path": path}


@app.get("/api/config/snippets")
def get_snippets():
    mode = manager._active_mode
    sp = manager._shared_port
    opencode_cfg = agent_config.generate_opencode_config(mode=mode, shared_port=sp)
    cursor_info = agent_config.get_cursor_instructions(sp)
    continue_cfg = agent_config.get_continue_config_yaml(sp)
    aider_cmd = agent_config.get_aider_command(sp)
    return {
        "opencode": opencode_cfg,
        "cursor": cursor_info,
        "continue_yaml": continue_cfg,
        "aider": aider_cmd,
    }


@app.get("/api/diagnostics/summary")
def get_diagnostics_summary():
    return diagnostics.get_diagnostics_summary()


@app.get("/api/diagnostics/export")
def export_diagnostics():
    zip_path = diagnostics.export_diagnostic_zip()
    filename = os.path.basename(zip_path)
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=filename,
    )


@app.post("/api/config/security")
def update_security(req: SecurityConfigRequest):
    config_manager.set_api_key_required(req.api_key_required)
    if req.api_key:
        config_manager.set_api_key(req.api_key)
    cfg = config_manager.load_config()
    return {
        "status": "ok",
        "api_key_required": cfg.get("api_key_required", False),
        "api_key": cfg.get("api_key", "sk-webmind"),
    }


@app.get("/api/update/check")
def check_updates():
    return updater.check_for_updates()


@app.get("/api/update/patches")
def get_patches():
    return dom_patches.load_patches()


@app.post("/api/update/patches")
def apply_patch(req: PatchRequest):
    success = dom_patches.apply_patch(req.provider, req.role, req.selectors)
    return {
        "status": "ok" if success else "error",
        "provider": req.provider,
        "role": req.role,
        "selectors": req.selectors,
    }


def start_dashboard(port: int = 8000, open_browser_window: bool = True):
    """Starts the WebMind Control Panel server."""
    if open_browser_window:
        webbrowser.open(f"http://127.0.0.1:{port}/dashboard")
    print(f"\n[WebMind Dashboard] Running at http://127.0.0.1:{port}/dashboard\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    start_dashboard(port=8000, open_browser_window=True)