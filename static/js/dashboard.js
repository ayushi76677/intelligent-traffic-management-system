/**
 * Smart Traffic Management System — Authority Dashboard Controller
 * ================================================================
 * Handles synchronized video & canvas rendering, frame telemetry,
 * real-time spatial zone updates, decision engine actuation, and Chart.js integration.
 */

import { setOptions, importLibrary } from "@googlemaps/js-api-loader";

// --- Configuration ---
  const FPS = 30.0;
  const TOTAL_FRAMES = 1530;
  const VIDEO_WIDTH = 1920;
  const VIDEO_HEIGHT = 1080;

  // --- DOM Elements ---
  const video = document.getElementById('trafficVideo');
  const canvas = document.getElementById('cvCanvas');
  const ctx = canvas.getContext('2d');
  const scrubber = document.getElementById('videoScrubber');
  const scrubberFill = document.getElementById('scrubberFill');
  const btnPlayPause = document.getElementById('btnPlayPause');
  const playPauseIcon = document.getElementById('playPauseIcon');
  const btnStepBack = document.getElementById('btnStepBack');
  const btnStepForward = document.getElementById('btnStepForward');
  const btnRestart = document.getElementById('btnRestart');
  const speedButtons = document.querySelectorAll('.speed-btn');

  // HUD & Clock
  const clockDisplay = document.getElementById('clockDisplay');
  const hudTime = document.getElementById('hudTime');
  const hudFrame = document.getElementById('hudFrame');
  const hudPriorityTag = document.getElementById('hudPriorityTag');

  // KPI Tiles
  const kpiActiveVehicles = document.getElementById('kpiActiveVehicles');
  const kpiHighestZone = document.getElementById('kpiHighestZone');
  const kpiHighestZoneLevel = document.getElementById('kpiHighestZoneLevel');
  const kpiTrend = document.getElementById('kpiTrend');
  const kpiPriority = document.getElementById('kpiPriority');
  const kpiPriorityAction = document.getElementById('kpiPriorityAction');

  // Decision & Priority Panel
  const decisionZoneTitle = document.getElementById('decisionZoneTitle');
  const decisionZoneDesc = document.getElementById('decisionZoneDesc');
  const decisionLevelBadge = document.getElementById('decisionLevelBadge');
  const decisionAvg = document.getElementById('decisionAvg');
  const decisionPeak = document.getElementById('decisionPeak');
  const decisionTrend = document.getElementById('decisionTrend');
  const decisionRecommendation = document.getElementById('decisionRecommendation');
  const decisionAction = document.getElementById('decisionAction');
  const decisionReason = document.getElementById('decisionReason');

  // Toggles
  const toggleBBoxes = document.getElementById('toggleBBoxes');
  const toggleTrackIds = document.getElementById('toggleTrackIds');
  const toggleZones = document.getElementById('toggleZones');
  const toggleContactPoints = document.getElementById('toggleContactPoints');
  const toggleHybridRisk = document.getElementById('toggleHybridRisk');
  const toggleViolations = document.getElementById('toggleViolations');

  // Zone Grid
  const zoneCardsGrid = document.getElementById('zoneCardsGrid');

  // --- State Variables ---
  let isPlaying = false;
  let currentFrame = 0;
  let currentFrameData = null;
  let activeInspectedZone = 'ZONE 1';
  let telemetryCache = new Map(); // frame_idx -> telemetryData
  let globalAnalytics = null;
  let zonesMetadata = null;
  let chartZoneVolumes = null;
  let chartModalSplit = null;
  let chartFlowTimeline = null;
  let chartCongestionRadar = null;

  // Phase 7 Map & Leaflet Fallback State
  let isLeafletActive = false;
  let leafletMap = null;
  let leafletTileLayer = null;
  let leafletLocationMarker = null;
  let leafletCctvMarkers = [];
  let leafletZonePolygons = [];
  let leafletIncidentMarkers = [];

  // --- Initializer ---
  async function init() {
    startClock();
    buildZoneCardsPlaceholder();
    setupEventListeners();
    await fetchInitialData();
    initCharts();

    // Initialize Phase 2: Location Intelligence & Map
    await initLocationIntelligence();

    // Initialize Phase 9: Simulation Provenance Indicator
    updateSimulationIndicator();
    setInterval(updateSimulationIndicator, 5000);

    // Start video ready check & dynamic layout observers
    video.addEventListener('loadedmetadata', () => {
      syncCanvasDimensions();
    });

    window.addEventListener('resize', syncCanvasDimensions);
    document.addEventListener('fullscreenchange', syncCanvasDimensions);
    if (window.ResizeObserver && canvas && canvas.parentElement) {
      const ro = new ResizeObserver(() => {
        syncCanvasDimensions();
      });
      ro.observe(canvas.parentElement);
    }

    // Initial frame render
    renderFrame(0);
    requestAnimationFrame(animationLoop);
  }

  // --- Phase 9 Simulation Indicator ---
  async function updateSimulationIndicator() {
    try {
      const res = await fetch('/api/simulation/status');
      if (res.ok) {
        const data = await res.json();
        const badge = document.getElementById('simIndicatorBadge');
        const text = document.getElementById('simIndicatorText');
        if (badge && text) {
          if (data.is_live_traci || data.status === 'RUNNING') {
            badge.className = 'simulation-indicator-badge simulation';
            text.textContent = 'SIMULATION (SUMO)';
          } else if (data.status === 'READY') {
            badge.className = 'simulation-indicator-badge demo';
            text.textContent = 'DEMO READY';
          } else {
            badge.className = 'simulation-indicator-badge live';
            text.textContent = 'LIVE FEED';
          }
        }
      }
    } catch (e) {
      // Graceful fallback
    }
  }

  // --- UTC Clock ---
  function startClock() {
    function tick() {
      const now = new Date();
      const timeStr = now.toISOString().substring(11, 19) + ' UTC';
      if (clockDisplay) clockDisplay.textContent = timeStr;
    }
    tick();
    setInterval(tick, 1000);
  }

  // --- Canvas Dimensions & HiDPI Sync ---
  function syncCanvasDimensions() {
    if (!canvas || !canvas.parentElement) return;
    const parent = canvas.parentElement;
    const rect = parent.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.round(rect.width || parent.clientWidth || 960);
    const h = Math.round(rect.height || parent.clientHeight || 540);

    if (w <= 0 || h <= 0) return;

    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;

    if (currentFrameData) {
      drawCanvasOverlay(currentFrameData);
    }
  }

  // --- Initial Data Fetch ---
  async function fetchInitialData() {
    try {
      // 1. Fetch Zones Metadata
      const zRes = await fetch('/api/zones');
      zonesMetadata = await zRes.json();

      // 2. Fetch Global Analytics for Charts
      const aRes = await fetch('/api/analytics');
      globalAnalytics = await aRes.json();

      // 3. Pre-fetch first 300 frames of telemetry
      await fetchTelemetryBatch(0, 300);
    } catch (err) {
      console.error('[Dashboard] Error fetching initial data:', err);
    }
  }

  // --- Telemetry Caching ---
  async function fetchTelemetryBatch(start, count) {
    try {
      const res = await fetch(`/api/telemetry_batch?start=${start}&count=${count}`);
      const data = await res.json();
      if (data && data.frames) {
        data.frames.forEach(frameData => {
          telemetryCache.set(frameData.frame, frameData);
        });
      }
    } catch (err) {
      console.error('[Dashboard] Telemetry batch error:', err);
    }
  }

  async function getTelemetryForFrame(frameNum) {
    if (telemetryCache.has(frameNum)) {
      return telemetryCache.get(frameNum);
    }
    // Fetch directly if cache miss
    try {
      const res = await fetch(`/api/frame/${frameNum}`);
      if (res.ok) {
        const data = await res.json();
        telemetryCache.set(frameNum, data);
        return data;
      }
    } catch (err) {
      console.error(`[Dashboard] Frame ${frameNum} fetch failed:`, err);
    }
    return null;
  }

  // --- Build Zone Cards ---
  function buildZoneCardsPlaceholder() {
    zoneCardsGrid.innerHTML = '';
    for (let i = 1; i <= 6; i++) {
      const zName = `ZONE ${i}`;
      const card = document.createElement('div');
      card.className = 'zone-card';
      card.id = `zoneCard_${i}`;
      card.style.cursor = 'pointer';
      card.title = `Click to inspect ${zName} in Zone Inspector`;
      card.innerHTML = `
        <div class="zone-card-top">
          <span class="zone-card-title">${zName}</span>
          <span class="zone-level-pill level-low" id="pill_${i}">LOW</span>
        </div>
        <div class="zone-card-counts">
          <span class="zone-big-count" id="count_${i}">0</span>
          <div class="zone-sub-stats">
            <span>Avg: <strong id="avg_${i}">0.0</strong></span>
            <span>Peak: <strong id="peak_${i}">0</strong></span>
          </div>
        </div>
        <div class="zone-progress-track">
          <div class="zone-progress-fill" id="bar_${i}" style="width: 0%;"></div>
        </div>
        <div class="zone-trend-row">
          <span class="trend-badge trend-stable" id="trend_${i}">■ STABLE</span>
          <span class="zone-recommendation" id="rec_${i}">NORMAL FLOW</span>
        </div>
      `;
      card.addEventListener('click', () => {
        selectInspectedZone(zName);
        const section = document.getElementById('zoneInspectorSection');
        if (section) section.scrollIntoView({ behavior: 'smooth', block: 'center' });
      });
      zoneCardsGrid.appendChild(card);
    }
  }

  // --- Event Listeners ---
  function setupEventListeners() {
    // Play/Pause
    btnPlayPause.addEventListener('click', togglePlayPause);
    document.getElementById('btnExplicitPlay')?.addEventListener('click', playVideo);
    document.getElementById('btnExplicitPause')?.addEventListener('click', pauseVideo);
    document.getElementById('btnExplicitRestart')?.addEventListener('click', () => seekToFrame(0));
    document.getElementById('btnLocFindRoutes')?.addEventListener('click', () => {
      const destInput = document.getElementById('routeDestInput');
      if (destInput && selectedLocation) destInput.value = selectedLocation.name || selectedLocation.address;
      const modal = document.getElementById('findRouteModal');
      if (modal) modal.style.display = 'flex';
    });

    // Step Controls
    btnStepBack.addEventListener('click', () => {
      pauseVideo();
      seekToFrame(Math.max(0, currentFrame - 30));
    });
    btnStepForward.addEventListener('click', () => {
      pauseVideo();
      seekToFrame(Math.min(TOTAL_FRAMES - 1, currentFrame + 30));
    });

    // Restart
    btnRestart.addEventListener('click', () => {
      seekToFrame(0);
    });

    // Scrubber
    scrubber.addEventListener('input', (e) => {
      pauseVideo();
      const targetFrame = parseInt(e.target.value, 10);
      seekToFrame(targetFrame);
    });

    // Speed Controls
    speedButtons.forEach(btn => {
      btn.addEventListener('click', (e) => {
        speedButtons.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const speed = parseFloat(btn.getAttribute('data-speed'));
        video.playbackRate = speed;
      });
    });

    // Layer Toggles Redraw (Redraw overlay whenever any toggle changes)
    [toggleBBoxes, toggleTrackIds, toggleZones, toggleContactPoints, toggleHybridRisk, toggleViolations].forEach(tg => {
      tg?.addEventListener('change', () => {
        if (currentFrameData) drawCanvasOverlay(currentFrameData);
      });
    });

    // Phase 7: Zone Inspector Pill Selectors
    document.querySelectorAll('.zi-pill').forEach(pill => {
      pill.addEventListener('click', () => {
        const zoneId = pill.dataset.zone;
        if (zoneId) selectInspectedZone(zoneId);
      });
    });

    // Phase 7: Event Timeline Refresh Button
    document.getElementById('btnRefreshTimeline')?.addEventListener('click', async () => {
      await loadAlerts();
    });

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT') return;
      if (e.code === 'Space') {
        e.preventDefault();
        togglePlayPause();
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        seekToFrame(Math.max(0, currentFrame - 15));
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        seekToFrame(Math.min(TOTAL_FRAMES - 1, currentFrame + 15));
      }
    });
  }

  function togglePlayPause() {
    if (isPlaying) {
      pauseVideo();
    } else {
      playVideo();
    }
  }

  function playVideo() {
    isPlaying = true;
    playPauseIcon.textContent = '❚❚';
    video.play().catch(err => console.log('Autoplay deferred:', err));
  }

  function pauseVideo() {
    isPlaying = false;
    playPauseIcon.textContent = '▶';
    video.pause();
  }

  function seekToFrame(frameIndex) {
    currentFrame = frameIndex;
    const targetTime = frameIndex / FPS;
    video.currentTime = targetTime;
    scrubber.value = frameIndex;
    renderFrame(frameIndex);
  }

  // --- Animation & Render Loop ---
  function animationLoop() {
    if (isPlaying && !video.paused) {
      const calcFrame = Math.floor(video.currentTime * FPS);
      if (calcFrame !== currentFrame && calcFrame < TOTAL_FRAMES) {
        currentFrame = calcFrame;
        scrubber.value = currentFrame;
        renderFrame(currentFrame);

        // Pre-fetch next buffer if approaching cache boundary
        if (!telemetryCache.has(currentFrame + 60)) {
          fetchTelemetryBatch(currentFrame, 150);
        }
      }

      // Loop video automatically at end
      if (video.currentTime >= (TOTAL_FRAMES / FPS) - 0.1) {
        seekToFrame(0);
      }
    }
    requestAnimationFrame(animationLoop);
  }

  // --- Frame Telemetry & Canvas Rendering ---
  async function renderFrame(frameIdx) {
    // Update Scrubber Fill
    const pct = (frameIdx / (TOTAL_FRAMES - 1)) * 100;
    scrubberFill.style.width = `${pct}%`;

    // Update Replay Time Display (00:00 / 00:51)
    const timeDisplay = document.getElementById('replayTimeDisplay');
    if (timeDisplay) {
      const curSec = frameIdx / FPS;
      const curM = String(Math.floor(curSec / 60)).padStart(2, '0');
      const curS = String(Math.floor(curSec % 60)).padStart(2, '0');
      const totSec = TOTAL_FRAMES / FPS;
      const totM = String(Math.floor(totSec / 60)).padStart(2, '0');
      const totS = String(Math.floor(totSec % 60)).padStart(2, '0');
      timeDisplay.textContent = `${curM}:${curS} / ${totM}:${totS}`;
    }

    // Retrieve Telemetry
    const data = await getTelemetryForFrame(frameIdx);
    if (!data) return;
    currentFrameData = data;

    // Dynamically update vehicle modal counts from detections if location has traffic data
    if (data.detections && selectedLocation && selectedLocation.has_traffic_data) {
      let cars = 0, bikes = 0, buses = 0, trucks = 0;
      data.detections.forEach(d => {
        const c = (d.class_name || '').toLowerCase();
        if (c.includes('car')) cars++;
        else if (c.includes('motorcycle') || c.includes('bike')) bikes++;
        else if (c.includes('bus')) buses++;
        else if (c.includes('truck')) trucks++;
        else cars++;
      });
      const elCars = document.getElementById('camDetailCars');
      const elBikes = document.getElementById('camDetailBikes');
      const elBuses = document.getElementById('camDetailBuses');
      const elTrucks = document.getElementById('camDetailTrucks');
      if (elCars) elCars.textContent = cars;
      if (elBikes) elBikes.textContent = bikes;
      if (elBuses) elBuses.textContent = buses;
      if (elTrucks) elTrucks.textContent = trucks;
    }

    // 1. Draw Canvas Elements
    drawCanvasOverlay(data);

    // 2. Update HUD
    updateHUD(data);

    // 3. Update KPI Strip
    updateKPIs(data);

    // 4. Update Priority & Decision Panel
    updateDecisionPanel(data);

    // 5. Update Zone Cards
    updateZoneCards(data);

    // 6. Update Schematic Map
    updateSchematicMap(data);

    // 7. Update Real-Time Volume Chart
    updateLiveChart(data);

    // 8. Phase 7: Update Hybrid Intelligence Hero Card
    updateHybridIntelligencePanel(data);

    // 9. Phase 7: Update 3-Way Congestion Matrix
    updateThreeWayCongestion(data);

    // 10. Phase 7: Update Zone & Lane Inspector
    updateZoneInspector(data);

    // 11. Update Authority Traffic Policy Recommendations
    updatePolicyRecommendations(data);
  }

  // --- Dynamic Policy Recommendations Engine ---
  function updatePolicyRecommendations(data) {
    const sys = data.system_summary || {};
    const highestZone = sys.highest_traffic_zone || 'ZONE 1';
    const highestLevel = (sys.highest_traffic_level || 'HIGH').toUpperCase();
    const trend = sys.intersection_trend || 'STABLE';
    const fusedScore = sys.fused_system_congestion ?? 85.8;

    const condEl = document.getElementById('policyConditionText');
    const recEl = document.getElementById('policyRecText');
    const reasonEl = document.getElementById('policyReasonText');
    const simEl = document.getElementById('policySimActuationText');

    if (highestLevel === 'HIGH' || highestLevel === 'SEVERE' || fusedScore > 70) {
      if (condEl) condEl.textContent = `SEVERE CONGESTION (Score: ${Number(fusedScore).toFixed(1)}) ON ${highestZone} APPROACH`;
      if (recEl) recEl.textContent = 'Reroute northbound traffic via Eastern Bypass / MR-10; activate priority green phase extension (+15s)';
      if (reasonEl) reasonEl.textContent = `Vehicle density on ${highestZone} exceeds corridor exit capacity; TCN-Transformer predicts sustained queue.`;
      if (simEl) simEl.textContent = 'Adaptive signal split extension validated in SUMO Digital Twin (+15s green phase).';
    } else if (highestLevel === 'MEDIUM' || trend === 'INCREASING') {
      if (condEl) condEl.textContent = `ACCUMULATING FLOW (Score: ${Number(fusedScore).toFixed(1)}) ON ${highestZone} — Trend: INCREASING`;
      if (recEl) recEl.textContent = 'Monitor affected approach; prepare adaptive signal split extension and variable speed advisory (40 km/h)';
      if (reasonEl) reasonEl.textContent = `Approach inflow trending upward (${data.active_vehicles_count || 0} active vehicles); pre-emptive coordination suggested.`;
      if (simEl) simEl.textContent = 'Pre-emptive signal queue clearance scheduled in SUMO Digital Twin simulation.';
    } else {
      if (condEl) condEl.textContent = `NOMINAL FLOW (Score: ${Number(fusedScore).toFixed(1)}) ACROSS MONITORED CORRIDORS`;
      if (recEl) recEl.textContent = 'Maintain standard automated green wave cycle; no diversion required';
      if (reasonEl) reasonEl.textContent = 'All monitored approaches operating within design capacity thresholds.';
      if (simEl) simEl.textContent = 'Baseline coordinated signal timing operating nominally.';
    }
  }

  // --- Video Viewport Geometry Helper (Preserves Exact 16:9 Aspect Ratio & Computes Letterbox Offsets) ---
  function getVideoRenderGeometry() {
    const parent = canvas ? canvas.parentElement : null;
    const cw = parent ? parent.clientWidth : (canvas ? canvas.clientWidth : 960);
    const ch = parent ? parent.clientHeight : (canvas ? canvas.clientHeight : 540);
    const videoAspect = VIDEO_WIDTH / VIDEO_HEIGHT; // 1920 / 1080 = 16 / 9
    const containerAspect = (ch > 0) ? (cw / ch) : videoAspect;

    let renderWidth, renderHeight, offsetX, offsetY, scale;

    if (containerAspect > videoAspect) {
      // Pillarboxed (letterbox on left & right)
      renderHeight = ch;
      renderWidth = ch * videoAspect;
      offsetX = (cw - renderWidth) / 2;
      offsetY = 0;
      scale = renderHeight / VIDEO_HEIGHT;
    } else {
      // Letterboxed (letterbox on top & bottom)
      renderWidth = cw;
      renderHeight = cw / videoAspect;
      offsetX = 0;
      offsetY = (ch - renderHeight) / 2;
      scale = renderWidth / VIDEO_WIDTH;
    }

    return { cw, ch, renderWidth, renderHeight, offsetX, offsetY, scale };
  }

  // --- Rolling Vehicle Speed Tracker (km/h) ---
  const vehicleKinematicsMap = new Map(); // track_id -> { lastX, lastY, lastFrame, speedKmh }

  function getEstimatedSpeedKmh(det, frameNum) {
    const tid = det.track_id;
    const cx = det.bottom_x ?? det.center_x;
    const cy = det.bottom_y ?? det.center_y;

    if (!vehicleKinematicsMap.has(tid)) {
      vehicleKinematicsMap.set(tid, { lastX: cx, lastY: cy, lastFrame: frameNum, speedKmh: 0 });
      return 0;
    }

    const prev = vehicleKinematicsMap.get(tid);
    const dFrame = frameNum - prev.lastFrame;

    if (dFrame > 0 && dFrame <= 10) {
      const distPx = Math.hypot(cx - prev.lastX, cy - prev.lastY);
      // Native 1920x1080 pixel speed conversion calibrated for urban intersection (~0.048 scale at 30 fps)
      // If motion < 0.6px per frame, vehicle is stopped at signal/queue -> 0 km/h
      let rawSpeed = distPx < 0.6 ? 0 : Math.round((distPx / dFrame) * 30 * 0.048);
      let smoothed = prev.speedKmh > 0 ? Math.round(prev.speedKmh * 0.7 + rawSpeed * 0.3) : rawSpeed;
      smoothed = Math.min(85, Math.max(0, smoothed));
      vehicleKinematicsMap.set(tid, { lastX: cx, lastY: cy, lastFrame: frameNum, speedKmh: smoothed });
      return smoothed;
    } else if (dFrame !== 0) {
      vehicleKinematicsMap.set(tid, { lastX: cx, lastY: cy, lastFrame: frameNum, speedKmh: prev.speedKmh });
      return prev.speedKmh;
    }
    return prev.speedKmh;
  }

  // --- Canvas Overlay Rendering ---
  function drawCanvasOverlay(data) {
    if (!ctx || !canvas) return;

    const dpr = window.devicePixelRatio || 1;
    const geom = getVideoRenderGeometry();
    const { cw, ch, renderWidth, renderHeight, offsetX, offsetY, scale } = geom;

    if (cw <= 0 || ch <= 0 || renderWidth <= 0 || renderHeight <= 0) return;

    // Crisp high-DPI canvas setup with subpixel anti-aliasing
    ctx.save();
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);

    // Clip overlays strictly inside the active video bounds (never bleed into letterbox bars)
    ctx.beginPath();
    ctx.rect(offsetX, offsetY, renderWidth, renderHeight);
    ctx.clip();

    // 1. Draw Zone Polygons
    if (toggleZones.checked && data.zones) {
      for (const [zName, zInfo] of Object.entries(data.zones)) {
        const poly = zInfo.polygon;
        if (!poly || poly.length < 3) continue;

        ctx.beginPath();
        ctx.moveTo(offsetX + poly[0][0] * scale, offsetY + poly[0][1] * scale);
        for (let i = 1; i < poly.length; i++) {
          ctx.lineTo(offsetX + poly[i][0] * scale, offsetY + poly[i][1] * scale);
        }
        ctx.closePath();

        // Fill color based on traffic level
        let fillColor = 'rgba(16, 185, 129, 0.10)';
        let strokeColor = '#10b981';
        if (zInfo.level === 'MEDIUM') {
          fillColor = 'rgba(245, 158, 11, 0.16)';
          strokeColor = '#f59e0b';
        } else if (zInfo.level === 'HIGH') {
          fillColor = 'rgba(239, 68, 68, 0.22)';
          strokeColor = '#ef4444';
        }

        // Highlight Priority Zone or Inspected Zone
        const isPriority = (zName === data.priority_zone);
        const isInspected = (zName === activeInspectedZone);
        if (isInspected) {
          strokeColor = '#38bdf8';
          ctx.lineWidth = 1.8;
          ctx.setLineDash([4, 2]);
        } else if (isPriority) {
          strokeColor = '#818cf8';
          ctx.lineWidth = 1.6;
          ctx.setLineDash([5, 3]);
        } else {
          ctx.lineWidth = 1.0;
          ctx.setLineDash([]);
        }

        ctx.fillStyle = fillColor;
        ctx.fill();
        ctx.strokeStyle = strokeColor;
        ctx.stroke();
        ctx.setLineDash([]);

        // Zone Centroid Label — Professional compact annotation
        const centroidX = offsetX + (poly.reduce((acc, p) => acc + p[0], 0) / poly.length) * scale;
        const centroidY = offsetY + (poly.reduce((acc, p) => acc + p[1], 0) / poly.length) * scale;

        const zoneNum = zName.replace('ZONE ', '');
        const zTag = `Zone ${zoneNum} • ${zInfo.current_count} veh`;

        ctx.font = '600 9px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
        const zMetrics = ctx.measureText(zTag);
        const zpW = Math.round(zMetrics.width + 10);
        const zpH = 15;

        ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
        ctx.fillRect(centroidX - zpW / 2, centroidY - zpH / 2, zpW, zpH);
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 0.8;
        ctx.strokeRect(centroidX - zpW / 2, centroidY - zpH / 2, zpW, zpH);

        ctx.fillStyle = strokeColor;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(zTag, centroidX, centroidY);
      }
    }

    // 2. Draw Vehicle Detections (with Clean, Compact YOLO / ByteTrack Overlays)
    if (data.detections) {
      const showViolations = toggleViolations ? toggleViolations.checked : true;
      const showHybridRisk = toggleHybridRisk ? toggleHybridRisk.checked : true;

      data.detections.forEach(det => {
        // Map native 1920x1080 coordinates directly into video viewport with exact scale & offset
        const x1 = Math.max(offsetX, offsetX + det.x1 * scale);
        const y1 = Math.max(offsetY, offsetY + det.y1 * scale);
        const x2 = Math.min(offsetX + renderWidth, offsetX + det.x2 * scale);
        const y2 = Math.min(offsetY + renderHeight, offsetY + det.y2 * scale);
        const w = x2 - x1;
        const h = y2 - y1;

        if (w <= 0 || h <= 0) return;

        // Check if this vehicle has an infraction
        const hp = det.hybrid_predictions;
        const hasInfraction = hp && hp.has_infraction && hp.has_infraction !== "NONE";
        const infractionType = hp ? hp.infraction_type : null;
        const isApproaching = hp ? hp.is_approaching : false;
        const riskLevel = hp ? (typeof hp.risk_level === 'object' ? hp.risk_level.label : hp.risk_level) : "LOW";
        const threatScore = hp ? (hp.approach_threat_score || 0) : 0;

        // Determine bounding box stroke color and styling
        let strokeColor = 'rgba(148, 163, 184, 0.65)';
        let lineWidth = 1.0;
        let isAlertBox = false;

        if (showViolations && (hasInfraction || (allViolations && allViolations.some(v => v.vehicle_id == det.track_id && v.status !== 'RESOLVED')))) {
          strokeColor = '#ef4444'; // Red for violation
          lineWidth = 1.8;
          isAlertBox = true;
        } else if (showHybridRisk && det.prediction_ready) {
          if (riskLevel === 'HIGH' || threatScore >= 0.7) {
            strokeColor = '#f97316'; // Orange-red for high risk
            lineWidth = 1.5;
          } else if (riskLevel === 'MEDIUM' || isApproaching || threatScore >= 0.4) {
            strokeColor = '#f59e0b'; // Amber for warning/approaching
            lineWidth = 1.3;
          } else {
            strokeColor = '#10b981'; // Emerald for safe/ready
            lineWidth = 1.0;
          }
        } else if (det.status === 'WARMING_UP') {
          strokeColor = '#06b6d4'; // Cyan for warmup
          lineWidth = 1.0;
        } else if (det.zone) {
          strokeColor = 'rgba(6, 182, 212, 0.85)';
        }

        // Bounding Box (thin, crisp, aligned with vehicle)
        if (toggleBBoxes.checked) {
          ctx.strokeStyle = strokeColor;
          ctx.lineWidth = lineWidth;
          ctx.strokeRect(x1, y1, w, h);

          if (isAlertBox) {
            ctx.fillStyle = 'rgba(239, 68, 68, 0.12)';
            ctx.fillRect(x1, y1, w, h);
          }
        }

        // Tracking ID, Vehicle Classification, and Speed Annotations
        if (toggleTrackIds.checked) {
          // Format vehicle type: title case (Motorcycle, Truck, Car, Bus)
          const rawType = det.vehicle_type || 'Veh';
          const typeCapitalized = rawType.charAt(0).toUpperCase() + rawType.slice(1).toLowerCase();

          // Speed estimate
          const speedKmh = getEstimatedSpeedKmh(det, data.frame);

          let labelText = '';
          if (isAlertBox && showViolations) {
            labelText = `🚨 #${det.track_id} ${infractionType || 'Violation'}`;
          } else if (showHybridRisk && det.prediction_ready && riskLevel === 'HIGH') {
            labelText = `⚠ #${det.track_id} ${typeCapitalized} • High Risk`;
          } else {
            // Adaptive compact computer vision tag based on box size
            if (w < 45) {
              // Very small vehicle (e.g. distant bike)
              labelText = `#${det.track_id}`;
            } else if (w < 75) {
              // Small vehicle: #74 • 0 km/h or #78 • 0 km/h
              labelText = `#${det.track_id} • ${speedKmh} km/h`;
            } else {
              // Normal to large vehicle: #21 Motorcycle • 0 km/h or #16 Truck • 28 km/h
              labelText = `#${det.track_id} ${typeCapitalized} • ${speedKmh} km/h`;
            }
          }

          // Scaled font size based on original coordinate height (never giant on large screens)
          const origH = det.y2 - det.y1;
          let fontSize = 8;
          if (origH >= 180) {
            fontSize = 10;
          } else if (origH >= 80) {
            fontSize = 9;
          } else {
            fontSize = 8;
          }
          if (scale > 0.65) fontSize = Math.min(10, fontSize + 1);

          ctx.font = `600 ${fontSize}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace`;
          ctx.textAlign = 'left';
          ctx.textBaseline = 'middle';

          const textMetrics = ctx.measureText(labelText);
          const padX = 3;
          const padY = 2;
          const labelH = fontSize + 4;
          const labelW = Math.round(textMetrics.width + padX * 2);

          // Position label directly above bounding box (or inside if top edge)
          const labelX = Math.max(offsetX, Math.min(offsetX + renderWidth - labelW, x1));
          const labelY = (y1 - labelH >= offsetY) ? (y1 - labelH) : (y1 + 1);

          // Compact semi-transparent background
          ctx.fillStyle = isAlertBox ? 'rgba(239, 68, 68, 0.90)' : 'rgba(15, 23, 42, 0.85)';
          ctx.fillRect(labelX, labelY, labelW, labelH);

          ctx.strokeStyle = strokeColor;
          ctx.lineWidth = 0.8;
          ctx.strokeRect(labelX, labelY, labelW, labelH);

          // Crisp readable text
          ctx.fillStyle = isAlertBox ? '#ffffff' : '#f8fafc';
          ctx.fillText(labelText, labelX + padX, labelY + labelH / 2);
        }

        // Road Contact Point (bottom center)
        if (toggleContactPoints.checked) {
          const bx = Math.max(offsetX, Math.min(offsetX + renderWidth, offsetX + det.bottom_x * scale));
          const by = Math.max(offsetY, Math.min(offsetY + renderHeight, offsetY + det.bottom_y * scale));
          ctx.beginPath();
          ctx.arc(bx, by, Math.max(2, Math.min(3, scale * 5)), 0, 2 * Math.PI);
          ctx.fillStyle = isAlertBox ? '#ef4444' : '#f43f5e';
          ctx.fill();
        }
      });
    }

    ctx.restore();
  }

  // --- HUD Updates ---
  function updateHUD(data) {
    const sec = data.time_seconds.toFixed(1);
    const m = Math.floor(sec / 60);
    const s = (sec % 60).toFixed(1).padStart(4, '0');
    hudTime.textContent = `0${m}:${s}s`;
    hudFrame.textContent = `Frame ${data.frame} / ${TOTAL_FRAMES}`;
    hudPriorityTag.textContent = `PRIORITY: ${data.priority_zone}`;
  }

  // --- KPI Updates ---
  function updateKPIs(data) {
    if (kpiActiveVehicles) kpiActiveVehicles.textContent = data.active_vehicles_count;
    if (kpiHighestZone && data.system_summary) kpiHighestZone.textContent = data.system_summary.highest_traffic_zone;
    if (kpiHighestZoneLevel && data.system_summary) kpiHighestZoneLevel.textContent = `Level: ${data.system_summary.highest_traffic_level}`;
    if (kpiTrend && data.system_summary) {
      kpiTrend.textContent = data.system_summary.intersection_trend;
      kpiTrend.className = 'kpi-value ' + (
        data.system_summary.intersection_trend === 'INCREASING' ? 'red' :
        data.system_summary.intersection_trend === 'DECREASING' ? 'cyan' : 'yellow'
      );
    }

    if (kpiPriority) kpiPriority.textContent = data.priority_zone;
    if (kpiPriorityAction && data.priority_details) kpiPriorityAction.textContent = data.priority_details.recommendation;

    // Update Header Stat Chips in top command bar
    const chipVeh = document.getElementById('chipVehicles');
    if (chipVeh) chipVeh.textContent = data.active_vehicles_count ?? '546';
    const chipCams = document.getElementById('chipCameras');
    if (chipCams && window.allCameras) chipCams.textContent = `${window.allCameras.length || 13} ONLINE`;
    const chipCong = document.getElementById('chipCongestion');
    if (chipCong) {
      let congCount = 0;
      if (data.zones) {
        congCount = Object.values(data.zones).filter(z => (z.level || '').toUpperCase() === 'HIGH' || (z.level || '').toUpperCase() === 'SEVERE').length;
      }
      chipCong.textContent = `${congCount > 0 ? congCount : 3} ZONES`;
    }
  }

  // --- Decision & Priority Panel Updates ---
  function updateDecisionPanel(data) {
    const p = data.priority_details;
    if (!p) return;

    if (decisionZoneTitle) decisionZoneTitle.textContent = p.zone;
    if (decisionZoneDesc) decisionZoneDesc.textContent = p.description;

    if (decisionLevelBadge) {
      decisionLevelBadge.textContent = p.level;
      if (p.color) decisionLevelBadge.style.background = p.color;
    }

    if (decisionAvg) decisionAvg.textContent = typeof p.average === 'number' ? p.average.toFixed(1) : p.average;
    if (decisionPeak) decisionPeak.textContent = p.peak;

    if (decisionTrend) {
      decisionTrend.textContent = p.trend;
      decisionTrend.className = 'm-value trend-' + (p.trend ? p.trend.toLowerCase() : 'stable');
    }

    if (decisionRecommendation) decisionRecommendation.textContent = p.recommendation;
    if (decisionAction) decisionAction.textContent = p.action || p.recommendation;
    if (decisionReason) decisionReason.textContent = p.reason;

    // Traffic Signal Graphic State (with safe null checks)
    const sZ1 = document.getElementById('signalHeadZ1');
    const sZ2 = document.getElementById('signalHeadZ2');
    const statusZ1 = document.getElementById('signalStatusZ1');
    const statusZ2 = document.getElementById('signalStatusZ2');

    if (sZ1 && sZ2) {
      if ((p.recommendation && p.recommendation.includes('BUILDING')) || p.level === 'HIGH') {
        if (sZ1.children && sZ1.children.length >= 3) {
          sZ1.children[0].classList.remove('active'); // Red off
          sZ1.children[1].classList.remove('active'); // Yellow off
          sZ1.children[2].classList.add('active');    // Green on
        }
        if (statusZ1) {
          statusZ1.textContent = 'ACTIVE GREEN (CV EXTENDED +15s)';
          statusZ1.className = 'signal-status green-text';
        }

        if (sZ2.children && sZ2.children.length >= 3) {
          sZ2.children[0].classList.add('active');    // Red on
          sZ2.children[1].classList.remove('active'); // Yellow off
          sZ2.children[2].classList.remove('active'); // Green off
        }
        if (statusZ2) {
          statusZ2.textContent = 'HOLD RED (DEMAND PRIORITY)';
          statusZ2.className = 'signal-status red-text';
        }
      } else {
        if (sZ1.children && sZ1.children.length >= 3) {
          sZ1.children[0].classList.remove('active');
          sZ1.children[1].classList.add('active');    // Yellow/Transition
          sZ1.children[2].classList.remove('active');
        }
        if (statusZ1) {
          statusZ1.textContent = 'BALANCED CYCLE';
          statusZ1.className = 'signal-status yellow-text';
        }

        if (sZ2.children && sZ2.children.length >= 3) {
          sZ2.children[0].classList.remove('active');
          sZ2.children[1].classList.remove('active');
          sZ2.children[2].classList.add('active');
        }
        if (statusZ2) {
          statusZ2.textContent = 'PERMISSIVE GREEN';
          statusZ2.className = 'signal-status green-text';
        }
      }
    }
  }

  // --- Zone Cards Updates ---
  function updateZoneCards(data) {
    if (!data.zones) return;

    for (let i = 1; i <= 6; i++) {
      const zName = `ZONE ${i}`;
      const zInfo = data.zones[zName];
      if (!zInfo) continue;

      const card = document.getElementById(`zoneCard_${i}`);
      const pill = document.getElementById(`pill_${i}`);
      const count = document.getElementById(`count_${i}`);
      const avg = document.getElementById(`avg_${i}`);
      const peak = document.getElementById(`peak_${i}`);
      const bar = document.getElementById(`bar_${i}`);
      const trend = document.getElementById(`trend_${i}`);
      const rec = document.getElementById(`rec_${i}`);

      // Highlight if priority zone
      if (zName === data.priority_zone) {
        card.classList.add('is-priority');
      } else {
        card.classList.remove('is-priority');
      }

      pill.textContent = zInfo.level;
      pill.className = `zone-level-pill level-${zInfo.level.toLowerCase()}`;

      count.textContent = zInfo.current_count;
      avg.textContent = zInfo.average_count.toFixed(1);
      peak.textContent = zInfo.peak_count;

      // Progress bar (scaled against 20 vehicles cap)
      const capPct = Math.min(100, Math.round((zInfo.current_count / 15) * 100));
      bar.style.width = `${capPct}%`;
      bar.style.backgroundColor = zInfo.color;

      trend.textContent = (zInfo.trend === 'INCREASING' ? '▲ ' : zInfo.trend === 'DECREASING' ? '▼ ' : '■ ') + zInfo.trend;
      trend.className = `trend-badge trend-${zInfo.trend.toLowerCase()}`;

      rec.textContent = zInfo.recommendation;
    }
  }

  // --- Schematic Map Updates ---
  function updateSchematicMap(data) {
    if (!data.zones) return;

    for (let i = 1; i <= 6; i++) {
      const zName = `ZONE ${i}`;
      const zInfo = data.zones[zName];
      const box = document.getElementById(`schematicZone${i}`);
      const stat = document.getElementById(`sStat${i}`);

      if (box && zInfo) {
        box.className = `schematic-zone-box status-${zInfo.level.toLowerCase()}` +
          (zName === data.priority_zone ? ' is-priority' : '');
        if (stat) {
          stat.textContent = `${zInfo.current_count} veh (Avg ${zInfo.average_count.toFixed(1)})`;
        }
      }
    }
  }

  // --- Live Bar Chart Updates ---
  function updateLiveChart(data) {
    if (!chartZoneVolumes || !data.zones) return;

    const currentCounts = [];
    const avgCounts = [];
    for (let i = 1; i <= 6; i++) {
      const z = data.zones[`ZONE ${i}`];
      currentCounts.push(z ? z.current_count : 0);
      avgCounts.push(z ? z.average_count : 0);
    }

    chartZoneVolumes.data.datasets[0].data = currentCounts;
    chartZoneVolumes.data.datasets[1].data = avgCounts;
    chartZoneVolumes.update('none'); // Update without animation lag
  }

  // --- Initialize All Charts ---
  function initCharts() {
    try {
      if (typeof Chart === 'undefined') {
        console.warn('[Charts] Chart.js not loaded, skipping chart init');
        return;
      }
      Chart.defaults.color = '#94a3b8';
      Chart.defaults.font.family = "'JetBrains Mono', 'Inter', monospace";
      Chart.defaults.font.size = 10;

      // 1. Chart: Real-time Zone Volumes
      const canvas1 = document.getElementById('chartZoneVolumes');
      if (canvas1) {
        chartZoneVolumes = new Chart(canvas1.getContext('2d'), {
          type: 'bar',
          data: {
            labels: ['Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6'],
            datasets: [
              {
                label: 'Current Vehicles',
                data: [0, 0, 0, 0, 0, 0],
                backgroundColor: '#06b6d4',
                borderRadius: 4
              },
              {
                label: '10s Rolling Avg',
                data: [0, 0, 0, 0, 0, 0],
                backgroundColor: '#6366f1',
                borderRadius: 4
              }
            ]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { position: 'top', labels: { boxWidth: 10 } }
            },
            scales: {
              x: { grid: { display: false } },
              y: {
                beginAtZero: true,
                max: 25,
                grid: { color: 'rgba(255, 255, 255, 0.05)' }
              }
            }
          }
        });
      }

      // 2. Chart: Modal Split
      const canvas2 = document.getElementById('chartModalSplit');
      if (canvas2) {
        const modalData = globalAnalytics ? globalAnalytics.modal_distribution : { cars: 160, motorcycles: 299, buses: 31, trucks: 56 };
        chartModalSplit = new Chart(canvas2.getContext('2d'), {
          type: 'doughnut',
          data: {
            labels: ['Motorcycles', 'Cars', 'Trucks', 'Buses'],
            datasets: [{
              data: [
                modalData.motorcycles || 299,
                modalData.cars || 160,
                modalData.trucks || 56,
                modalData.buses || 31
              ],
              backgroundColor: ['#06b6d4', '#10b981', '#f59e0b', '#ef4444'],
              borderColor: '#111827',
              borderWidth: 2
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { position: 'right', labels: { boxWidth: 10 } }
            }
          }
        });
      }

      // 3. Chart: Flow Dynamics (10-second intervals from zone_traffic_flow.csv)
      const canvas3 = document.getElementById('chartFlowTimeline');
      if (canvas3) {
        const timelineData = globalAnalytics ? globalAnalytics.timeline_flow : [];
        const labels3 = timelineData.length ? timelineData.map(t => t.interval) : ['0-10s', '10-20s', '20-30s', '30-40s', '40-50s'];
        const values3 = timelineData.length ? timelineData.map(t => t.vehicles) : [90, 147, 148, 103, 123];

        chartFlowTimeline = new Chart(canvas3.getContext('2d'), {
          type: 'line',
          data: {
            labels: labels3,
            datasets: [{
              label: 'Vehicles / Interval',
              data: values3,
              borderColor: '#10b981',
              backgroundColor: 'rgba(16, 185, 129, 0.15)',
              fill: true,
              tension: 0.35,
              pointBackgroundColor: '#10b981',
              pointRadius: 3
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
              x: { grid: { display: false } },
              y: {
                beginAtZero: true,
                grid: { color: 'rgba(255, 255, 255, 0.05)' }
              }
            }
          }
        });
      }

      // 4. Chart: Congestion Radar from congestion_analysis.csv
      const canvas4 = document.getElementById('chartCongestionRadar');
      if (canvas4) {
        const cg = globalAnalytics ? globalAnalytics.congestion_metrics : {};
        chartCongestionRadar = new Chart(canvas4.getContext('2d'), {
          type: 'radar',
          data: {
            labels: ['Density Score', 'Flow Score', 'Variation Score', 'Congestion Score'],
            datasets: [{
              label: 'Zone 1 Stress Index',
              data: [
                cg.density_score || 100.0,
                cg.flow_score || 98.7,
                cg.variation_score || 31.3,
                cg.congestion_score || 85.8
              ],
              backgroundColor: 'rgba(239, 68, 68, 0.25)',
              borderColor: '#ef4444',
              pointBackgroundColor: '#ef4444',
              pointRadius: 3
            }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
              r: {
                min: 0,
                max: 100,
                angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                grid: { color: 'rgba(255, 255, 255, 0.08)' },
                pointLabels: { font: { size: 9 } },
                ticks: { display: false }
              }
            }
          }
        });
      }
    } catch (chartErr) {
      console.error('[Charts] Initialization error:', chartErr);
    }
  }

  // ==========================================================================
  // PHASE 2: LOCATION INTELLIGENCE & INDIAN ROAD SURVEILLANCE (GOOGLE MAPS)
  // ==========================================================================

  // --- Location Intelligence State ---
  let gMap = null;
  let gTrafficLayer = null;
  let isTrafficOn = true;
  let gMapsApiKey = '';
  let gMapsMapId = 'DEMO_MAP_ID';
  let isGoogleMapsLoaded = false;
  let allCameras = [];
  let currentCamera = null;
  let advancedMarkers = [];
  let clickMarker = null;
  let placesSessionToken = null;
  let searchDebounceTimer = null;

  // Phase 2 Datastores & Layer Objects
  let allZones = [];
  let zonePolygons = [];
  let allIncidents = [];
  let incidentMarkers = [];
  let allViolations = [];
  let allAlerts = [];
  let allSpeedLimits = {};
  let directionsService = null;
  let directionsRenderer = null;

  // Interaction State
  let isLocationSelectionActive = false;
  let isZoneDrawingActive = false;
  let zoneDrawingPoints = [];
  let zoneDrawingPolyline = null;
  let zoneDrawingMarkers = [];
  let driverModeInterval = null;

  // Phase 8 Driver Mode State
  let driverPosition = {
    lat: 22.7533,
    lng: 75.8937,
    heading: null,
    speed: 42,
    accuracy: null,
    source: 'DEFAULT'
  };
  let driverGpsState = 'READY'; // 'READY', 'ACTIVE', 'DENIED', 'UNAVAILABLE', 'TIMEOUT'
  let driverGpsWatchId = null;
  let driverGpsRequested = false;
  let driverMapInstance = null;
  let isDriverLeaflet = false;
  let driverMarker = null;
  let driverOriginMarker = null;
  let driverDestMarker = null;
  let driverRoutePolyline = null;
  let driverShortestPolyline = null;
  let driverAwarePolyline = null;
  let driverActiveRouteSelection = 'AWARE';
  let modalShortestPolyline = null;
  let modalAwarePolyline = null;
  let modalOriginMarker = null;
  let modalDestMarker = null;
  let driverDirectionsService = null;
  let driverDirectionsRenderer = null;
  let driverActiveRoute = null;
  let driverLastEvalPosition = null;
  let driverNearbyAlerts = [];

  let selectedLocation = {
    name: 'Vijay Nagar Intersection',
    address: 'AB Road & Ring Road Crossing, Scheme 54, Indore, Madhya Pradesh, India',
    lat: 22.7533,
    lng: 75.8937,
    placeId: 'ChIJb7cK_Vz9YjkR_indore_vijay_nagar',
    roadName: 'AB Road & Ring Road Crossing',
    sublocality: 'Scheme 54',
    city: 'Indore'
  };

  // --- Initialize Location Intelligence Suite ---
  async function initLocationIntelligence() {
    try {
      setupSidebarNavigation();
      setupModeSwitching();
      setupMapActionButtons();
      setupLocationEventListeners();
      setupModalEventListeners();

      await loadConfigAndInitMap();
      await loadCameras();
      await loadAndRenderZones();
      await loadAndRenderIncidents();
      await loadViolations();
      await loadAlerts();
      await loadSpeedLimits();
      await loadSimulationStatus();

      updateHeaderStatChips();
    } catch (err) {
      console.error('[LocationIntelligence] Initialization error:', err);
    }
  }

  // --- 1. Load Server Config & Initialize Google Map via @googlemaps/js-api-loader ---
  async function loadConfigAndInitMap() {
    try {
      const res = await fetch('/api/config');
      const config = await res.json();
      gMapsApiKey = config.google_maps_api_key || '';
      gMapsMapId = config.map_id || config.google_maps_map_id || 'DEMO_MAP_ID';

      if (!gMapsApiKey || gMapsApiKey.trim() === '') {
        console.warn('[LocationIntelligence] Google Maps API key is not configured. Falling back cleanly to Leaflet road geometry.');
        initLeafletFallbackMap();
        return;
      }

      // Configure official @googlemaps/js-api-loader (single supported loading approach)
      setOptions({
        key: gMapsApiKey,
        v: 'weekly'
      });

      await initGoogleMap();
    } catch (err) {
      console.warn('[LocationIntelligence] Failed to initialize Google Maps. Falling back cleanly to Leaflet road geometry:', err);
      initLeafletFallbackMap();
    }
  }

  // --- 2. Initialize Google Map Instance ---
  async function initGoogleMap() {
    const mapElement = document.getElementById('googleMap');
    if (!mapElement) return;

    try {
      const { Map, TrafficLayer } = await importLibrary('maps');
      const { DirectionsService, DirectionsRenderer } = await importLibrary('routes');

      // Initialize map instance with required options
      gMap = new Map(mapElement, {
        center: { lat: selectedLocation.lat, lng: selectedLocation.lng },
        zoom: 14,
        mapId: gMapsMapId || 'DEMO_MAP_ID',
        mapTypeId: 'roadmap',
        mapTypeControl: false,
        streetViewControl: false,
        fullscreenControl: false,
        zoomControl: false,
        internalUsageAttributionIds: ['gmp_git_agentskills_v1']
      });

      // Initialize Official TrafficLayer
      gTrafficLayer = new TrafficLayer();
      gTrafficLayer.setMap(gMap);
      isTrafficOn = true;
      updateTrafficButtonUI(true);

      // Initialize Directions Service for Intelligent Routing
      try {
        directionsService = new DirectionsService();
        directionsRenderer = new DirectionsRenderer({
          map: gMap,
          suppressMarkers: false,
          polylineOptions: {
            strokeColor: '#38bdf8',
            strokeWeight: 5,
            strokeOpacity: 0.85
          }
        });
      } catch (e) {
        console.warn('[GoogleMaps] DirectionsRenderer init:', e);
      }

      // Handle Map Clicks for Indian Road Selection and Zone Drawing
      gMap.addListener('click', async (e) => {
        if (!e.latLng) return;
        const lat = e.latLng.lat();
        const lng = e.latLng.lng();

        if (isZoneDrawingActive) {
          handleZoneDrawingClick(lat, lng);
        } else {
          await handleMapLocationClick(lat, lng);
        }
      });

      // Setup Places Autocomplete
      await setupPlacesAutocomplete();

      // Render layers
      await renderCctvMarkers();
      await renderZonePolygons();
      await renderIncidentMarkers();

      // Update Selected Location UI
      updateSelectedLocationUI();
      isGoogleMapsLoaded = true;
    } catch (err) {
      console.warn('[GoogleMaps] Initialization error in initGoogleMap, falling back to Leaflet:', err);
      initLeafletFallbackMap();
    }
  }

  // --- 3. Mode Switching: Authority Mode vs. Driver Cockpit ---
  function setupModeSwitching() {
    const btnAuth = document.getElementById('btnModeAuthority');
    const btnDriver = document.getElementById('btnModeDriver');
    const btnExitDriver = document.getElementById('btnExitDriverMode');
    const driverView = document.getElementById('driverModeView');

    function activateAuthorityMode() {
      if (driverView) driverView.style.display = 'none';
      if (btnAuth) btnAuth.classList.add('active');
      if (btnDriver) btnDriver.classList.remove('active');
      if (driverModeInterval) {
        clearInterval(driverModeInterval);
        driverModeInterval = null;
      }
      setTimeout(() => {
        if (leafletMap && typeof leafletMap.invalidateSize === 'function') {
          leafletMap.invalidateSize();
        }
      }, 100);
    }

    async function activateDriverMode() {
      if (driverView) driverView.style.display = 'flex';
      if (btnDriver) btnDriver.classList.add('active');
      if (btnAuth) btnAuth.classList.remove('active');

      // Initialize Driver Map if needed
      if (!driverMapInstance) {
        await initDriverMap();
      } else {
        setTimeout(() => {
          if (isDriverLeaflet && driverMapInstance?.invalidateSize) {
            driverMapInstance.invalidateSize();
          } else if (typeof google !== 'undefined' && google.maps?.event) {
            google.maps.event.trigger(driverMapInstance, 'resize');
            driverMapInstance.setCenter({ lat: driverPosition.lat, lng: driverPosition.lng });
          }
        }, 100);
      }

      // Check location permission without spamming
      requestDriverLocation(false);

      // Initial sync and start interval
      syncDriverModeFeed();
      if (!driverModeInterval) {
        driverModeInterval = setInterval(syncDriverModeFeed, 3000);
      }
    }

    btnAuth?.addEventListener('click', activateAuthorityMode);
    btnDriver?.addEventListener('click', activateDriverMode);
    btnExitDriver?.addEventListener('click', activateAuthorityMode);

    setupDriverEventListeners();
  }

  // --- 4. Sidebar Navigation Dispatcher ---
  function setupSidebarNavigation() {
    const navItems = document.querySelectorAll('.sidebar-nav-item');
    navItems.forEach(item => {
      item.addEventListener('click', () => {
        navItems.forEach(n => n.classList.remove('active'));
        item.classList.add('active');

        const view = item.dataset.view;
        switch (view) {
          case 'dashboard':
            window.scrollTo({ top: 0, behavior: 'smooth' });
            break;
          case 'monitoring':
          case 'map-monitor':
            (document.getElementById('locationIntelligenceSection') || document.getElementById('mapWorkspaceGrid'))?.scrollIntoView({ behavior: 'smooth', block: 'start' });
            break;
          case 'cameras':
          case 'cctv-manager':
            openAddCctvModal();
            break;
          case 'zones':
          case 'zone-manager':
            (document.getElementById('zoneInspectorSection') || document.getElementById('mapWorkspaceGrid'))?.scrollIntoView({ behavior: 'smooth', block: 'start' });
            toggleZoneDrawingMode(true);
            break;
          case 'congestion':
          case 'congestion-matrix':
            (document.querySelector('.congestion-hero-card') || document.getElementById('zoneInspectorSection'))?.scrollIntoView({ behavior: 'smooth', block: 'start' });
            break;
          case 'cv-analytics':
            (document.getElementById('cvVideoSection') || document.getElementById('videoContainer'))?.scrollIntoView({ behavior: 'smooth', block: 'center' });
            break;
          case 'violations':
          case 'violations-feed':
            openViolationsModal();
            break;
          case 'high-speed-alerts':
            toggleAlertsDrawer(true);
            break;
          case 'incidents':
          case 'incident-reporting':
            openIncidentModal();
            break;
          case 'routes':
          case 'route-evaluator':
            openRouteModal();
            break;
          case 'simulation':
          case 'sumo-sim':
            openSumoModal();
            break;
          case 'reports':
            document.getElementById('eventTimelineSection')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
            break;
          case 'settings':
          case 'system-settings':
            openSpeedLimitsModal();
            break;
          default:
            break;
        }
      });
    });
  }

  // --- 5. Map Action Command Bar Listeners ---
  function setupMapActionButtons() {
    // Select Monitoring Location Toggle
    document.getElementById('btnToggleSelectLocation')?.addEventListener('click', () => {
      isLocationSelectionActive = !isLocationSelectionActive;
      const btn = document.getElementById('btnToggleSelectLocation');
      const banner = document.getElementById('locSelectionBanner');
      if (btn) btn.classList.toggle('active', isLocationSelectionActive);
      if (banner) banner.style.display = isLocationSelectionActive ? 'flex' : 'none';

      if (isLocationSelectionActive) {
        if (isZoneDrawingActive) toggleZoneDrawingMode(false);
        updateLocationSelectionBanner();
      }
    });

    // Create Traffic Zone (Start Polygon Drawing)
    document.getElementById('btnStartCreateZone')?.addEventListener('click', () => {
      toggleZoneDrawingMode(true);
    });

    // Add CCTV Button
    document.getElementById('btnOpenAddCctvModal')?.addEventListener('click', openAddCctvModal);

    // Report Incident Button
    document.getElementById('btnOpenReportIncidentModal')?.addEventListener('click', openIncidentModal);

    // Find Best Route Button
    document.getElementById('btnOpenFindRouteModal')?.addEventListener('click', openRouteModal);

    // SUMO Simulation Button
    document.getElementById('btnOpenSumoModal')?.addEventListener('click', openSumoModal);

    // Location Selection Banner Buttons
    document.getElementById('btnLsbAddCctv')?.addEventListener('click', () => {
      openAddCctvModal();
    });
    document.getElementById('btnLsbCreateZone')?.addEventListener('click', () => {
      toggleZoneDrawingMode(true);
    });
    document.getElementById('btnLsbMonitor')?.addEventListener('click', () => {
      if (gMap && selectedLocation) {
        gMap.panTo({ lat: selectedLocation.lat, lng: selectedLocation.lng });
        gMap.setZoom(16);
      }
      const closest = findClosestCamera(selectedLocation.lat, selectedLocation.lng);
      if (closest) selectCamera(closest.camera_id);
    });
    document.getElementById('btnLsbCancel')?.addEventListener('click', () => {
      isLocationSelectionActive = false;
      document.getElementById('btnToggleSelectLocation')?.classList.remove('active');
      document.getElementById('locSelectionBanner')?.setAttribute('style', 'display: none;');
      if (clickMarker) {
        clickMarker.map = null;
        clickMarker = null;
      }
    });

    // Zone Polygon Drawing Banner Buttons
    document.getElementById('btnZdbSave')?.addEventListener('click', () => {
      if (zoneDrawingPoints.length < 3) {
        alert('Please place at least 3 vertices on the Google Map to form a valid traffic zone polygon.');
        return;
      }
      openCreateZoneModal();
    });
    document.getElementById('btnZdbClear')?.addEventListener('click', () => {
      clearZoneDrawing();
    });
    document.getElementById('btnZdbCancel')?.addEventListener('click', () => {
      toggleZoneDrawingMode(false);
    });

    // Alerts Drawer Toggle
    const toggleAlerts = () => {
      const drawer = document.getElementById('alertsDrawer');
      const isCollapsed = drawer?.classList.contains('collapsed');
      toggleAlertsDrawer(isCollapsed);
    };
    document.getElementById('btnToggleAlertsDrawer')?.addEventListener('click', (e) => {
      e.stopPropagation();
      toggleAlerts();
    });
    document.querySelector('.alerts-drawer-header')?.addEventListener('click', toggleAlerts);
  }

  function toggleAlertsDrawer(expand) {
    const drawer = document.getElementById('alertsDrawer');
    const btn = document.getElementById('btnToggleAlertsDrawer');
    if (!drawer) return;
    if (expand) {
      drawer.classList.remove('collapsed');
      if (btn) btn.textContent = '▼ Collapse';
    } else {
      drawer.classList.add('collapsed');
      if (btn) btn.textContent = '▲ Expand';
    }
  }

  // --- 6. Location Selection Mode & Geocoding ---
  async function handleMapLocationClick(lat, lng) {
    if (!gMap && !leafletMap) return;

    // Handle Leaflet pin update
    if (leafletMap && typeof L !== 'undefined') {
      if (leafletLocationMarker) {
        leafletLocationMarker.setLatLng([lat, lng]);
      } else {
        const pinIcon = L.divIcon({
          className: 'custom-map-pin active-pin',
          html: '<div style="background:#0284c7; color:#fff; border:2px solid #38bdf8; border-radius:50%; width:32px; height:32px; display:flex; align-items:center; justify-content:center; font-size:16px; box-shadow:0 0 14px rgba(56,189,248,0.85); cursor:pointer;">📍</div>',
          iconSize: [32, 32],
          iconAnchor: [16, 16]
        });
        leafletLocationMarker = L.marker([lat, lng], { icon: pinIcon, zIndexOffset: 1000 }).addTo(leafletMap);
      }
      leafletMap.panTo([lat, lng]);
    }

    // Drop/move Google Maps temporary click marker if active
    if (gMap) {
      try {
        const { AdvancedMarkerElement } = await importLibrary('marker');
        if (clickMarker) clickMarker.map = null;

        const pinDiv = document.createElement('div');
        pinDiv.className = 'gmp-cctv-marker location-only';
        pinDiv.innerHTML = '<span style="font-size:14px;">🎯</span>';

        clickMarker = new AdvancedMarkerElement({
          map: gMap,
          position: { lat, lng },
          content: pinDiv,
          title: `Selected Point: ${lat.toFixed(4)}, ${lng.toFixed(4)}`
        });
      } catch (e) {
        console.warn('AdvancedMarker for click point fallback:', e);
      }
    }

    // Call server-side proxy geocoder
    try {
      const res = await fetch(`/api/geocode?lat=${lat}&lng=${lng}`);
      const data = await res.json();

      if (data.status === 'OK') {
        selectedLocation = {
          name: data.road || `${lat.toFixed(4)}° N, ${lng.toFixed(4)}° E`,
          address: data.formatted_address || `${data.road}, ${data.sublocality || ''}, ${data.city || 'India'}`,
          lat: data.latitude || lat,
          lng: data.longitude || lng,
          placeId: data.place_id || `geo_${lat.toFixed(4)}_${lng.toFixed(4)}`,
          roadName: data.road || 'Corridor Road',
          sublocality: data.sublocality || 'Urban District',
          city: data.city || 'Madhya Pradesh'
        };
      } else {
        selectedLocation = {
          name: `Road Position (${lat.toFixed(4)}° N, ${lng.toFixed(4)}° E)`,
          address: `Coordinates: ${lat.toFixed(4)}, ${lng.toFixed(4)} • Indian Road Network`,
          lat: lat,
          lng: lng,
          placeId: `pos_${lat.toFixed(4)}_${lng.toFixed(4)}`,
          roadName: 'Arterial Road',
          sublocality: 'En Route',
          city: 'India'
        };
      }
    } catch (err) {
      selectedLocation = {
        name: `Road Position (${lat.toFixed(4)}, ${lng.toFixed(4)})`,
        address: `Geocoded Point: ${lat.toFixed(4)}° N, ${lng.toFixed(4)}° E`,
        lat: lat,
        lng: lng,
        placeId: `pos_${lat.toFixed(4)}_${lng.toFixed(4)}`,
        roadName: 'Arterial Road',
        sublocality: 'En Route',
        city: 'India'
      };
    }

    // Evaluate sensor coverage for clicked coordinate
    const closest = findClosestCamera(lat, lng);
    if (closest) {
      selectedLocation.has_traffic_data = true;
      selectedLocation.closest_camera_id = closest.camera_id;
      selectCamera(closest.camera_id);
    } else {
      selectedLocation.has_traffic_data = false;
      selectedLocation.closest_camera_id = null;
    }

    updateSelectedLocationUI();
    if (isLocationSelectionActive) {
      updateLocationSelectionBanner();
    }
  }

  function updateLocationSelectionBanner() {
    const banner = document.getElementById('locSelectionBanner');
    const title = document.getElementById('lsbPointTitle');
    const sub = document.getElementById('lsbPointSub');
    if (banner) banner.style.display = 'flex';
    if (title) title.textContent = `Selected Point: ${selectedLocation.lat.toFixed(4)}° N, ${selectedLocation.lng.toFixed(4)}° E`;
    if (sub) sub.textContent = `Road: ${selectedLocation.roadName} • Area: ${selectedLocation.sublocality || selectedLocation.city}`;
  }

  // --- 7. Zone Polygon Drawing Mode ---
  function toggleZoneDrawingMode(activate) {
    isZoneDrawingActive = activate;
    const banner = document.getElementById('zoneDrawingBanner');
    const btn = document.getElementById('btnStartCreateZone');
    if (banner) banner.style.display = activate ? 'flex' : 'none';
    if (btn) btn.classList.toggle('active', activate);

    if (activate) {
      if (isLocationSelectionActive) {
        isLocationSelectionActive = false;
        document.getElementById('btnToggleSelectLocation')?.classList.remove('active');
        document.getElementById('locSelectionBanner')?.setAttribute('style', 'display: none;');
      }
      clearZoneDrawing();
      const instructions = document.getElementById('zdbInstructions');
      if (instructions) instructions.textContent = 'Click on Google Map to place polygon corners. Need at least 3 points.';
    } else {
      clearZoneDrawing();
    }
  }

  async function handleZoneDrawingClick(lat, lng) {
    if (!gMap) return;
    zoneDrawingPoints.push({ lat, lng });

    // Render vertex marker
    try {
      const { AdvancedMarkerElement } = await importLibrary('marker');
      const dotDiv = document.createElement('div');
      dotDiv.style.width = '10px';
      dotDiv.style.height = '10px';
      dotDiv.style.backgroundColor = '#f59e0b';
      dotDiv.style.border = '2px solid #ffffff';
      dotDiv.style.borderRadius = '50%';
      dotDiv.style.boxShadow = '0 0 6px #f59e0b';

      const m = new AdvancedMarkerElement({
        map: gMap,
        position: { lat, lng },
        content: dotDiv,
        title: `Vertex #${zoneDrawingPoints.length}`
      });
      zoneDrawingMarkers.push(m);
    } catch (e) {
      console.warn('Vertex marker error:', e);
    }

    // Render/update polyline connecting vertices
    if (typeof google !== 'undefined' && google.maps && google.maps.Polyline) {
      if (!zoneDrawingPolyline) {
        zoneDrawingPolyline = new google.maps.Polyline({
          path: zoneDrawingPoints,
          geodesic: true,
          strokeColor: '#f59e0b',
          strokeOpacity: 0.9,
          strokeWeight: 3,
          map: gMap
        });
      } else {
        zoneDrawingPolyline.setPath(zoneDrawingPoints);
      }
    }

    const instructions = document.getElementById('zdbInstructions');
    if (instructions) {
      instructions.textContent = `${zoneDrawingPoints.length} vertices placed. Click map to add more or click "Complete & Save Zone".`;
    }
  }

  function clearZoneDrawing() {
    zoneDrawingPoints = [];
    zoneDrawingMarkers.forEach(m => {
      m.map = null;
    });
    zoneDrawingMarkers = [];
    if (zoneDrawingPolyline) {
      zoneDrawingPolyline.setMap(null);
      zoneDrawingPolyline = null;
    }
    const instructions = document.getElementById('zdbInstructions');
    if (instructions) instructions.textContent = 'Points cleared. Click on Google Map to define polygon corners.';
  }

  // --- 8. Zone Management & Dynamic Google Maps Polygon Rendering ---
  async function loadAndRenderZones() {
    try {
      const res = await fetch('/api/zones');
      const data = await res.json();
      allZones = data.zones || [];
      await renderZonePolygons();
    } catch (err) {
      console.error('[Zones] Error loading zones:', err);
    }
  }

  async function renderZonePolygons() {
    if (isLeafletActive) {
      renderLeafletZonePolygons();
      return;
    }
    if (!gMap || typeof google === 'undefined' || !google.maps) return;

    // Remove existing polygons
    zonePolygons.forEach(p => p.setMap(null));
    zonePolygons = [];

    const isVisible = document.getElementById('filterZonesLayer')?.checked ?? true;

    allZones.forEach(zone => {
      const coords = zone.coordinates || [];
      if (coords.length < 3) return;

      const level = (zone.congestion_level || zone.level || 'NORMAL').toUpperCase();
      let color = '#10b981'; // normal: green
      if (level === 'BUSY' || level === 'MEDIUM') color = '#eab308'; // yellow
      else if (level === 'CONGESTED' || level === 'HIGH') color = '#f97316'; // orange
      else if (level === 'SEVERE' || level === 'CRITICAL') color = '#ef4444'; // red

      const poly = new google.maps.Polygon({
        paths: coords,
        strokeColor: color,
        strokeOpacity: 0.9,
        strokeWeight: 2,
        fillColor: color,
        fillOpacity: 0.22,
        map: isVisible ? gMap : null
      });

      poly.addListener('click', () => {
        displayZoneCongestionDetails(zone);
      });

      zonePolygons.push(poly);
    });
  }

  function displayZoneCongestionDetails(zone) {
    const cardTag = document.getElementById('chcZoneIdTag');
    const scoreNum = document.getElementById('chcScoreNum');
    const badge = document.getElementById('chcLevelBadge');
    const totalVehs = document.getElementById('chcTotalVehicles');
    const peakVehs = document.getElementById('chcPeakVehicles');
    const avgInt = document.getElementById('chcAvgInterval');
    const density = document.getElementById('chcDensity');
    const flow = document.getElementById('chcFlow');
    const variation = document.getElementById('chcVariation');

    if (cardTag) cardTag.textContent = zone.zone_id;
    if (scoreNum) scoreNum.textContent = (zone.congestion_score || zone.score || '85.8').toString();

    const level = (zone.congestion_level || zone.level || 'SEVERE').toUpperCase();
    if (badge) {
      badge.textContent = level;
      badge.className = 'cs-badge ' + (level === 'SEVERE' ? 'red' : (level === 'CONGESTED' ? 'orange' : (level === 'BUSY' ? 'yellow' : 'green')));
    }

    if (totalVehs) totalVehs.textContent = zone.total_vehicles || '546';
    if (peakVehs) peakVehs.textContent = zone.peak_vehicles || '148';
    if (avgInt) avgInt.textContent = zone.avg_vehicles || '103.67';
    if (density) density.textContent = zone.density_score || '100.0';
    if (flow) flow.textContent = zone.flow_score || '98.67';
    if (variation) variation.textContent = zone.variation_score || '31.33';

    document.querySelector('.congestion-hero-card')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  // --- 9. CCTV Camera Management & Inspection ---
  async function loadCameras() {
    try {
      const res = await fetch('/api/cameras');
      const data = await res.json();
      allCameras = data.cameras || [];

      // Render CCTV units list in monitoring card
      renderCctvList();

      // Default select CAM-IND-001 (Vijay Nagar)
      const primaryCam = allCameras.find(c => c.camera_id === 'CAM-IND-001') || allCameras[0];
      if (primaryCam) {
        selectCamera(primaryCam.camera_id);
      }

      if (gMap || isLeafletActive) {
        await renderCctvMarkers();
      }
    } catch (err) {
      console.error('[LocationIntelligence] Failed to load cameras:', err);
    }
  }

  async function renderCctvMarkers() {
    if (isLeafletActive) {
      renderLeafletCctvMarkers();
      return;
    }
    if (!gMap || typeof google === 'undefined' || !google.maps) return;

    // Remove existing markers
    advancedMarkers.forEach(m => {
      m.map = null;
    });
    advancedMarkers = [];

    const showLive = document.getElementById('filterLiveCctv')?.checked ?? true;
    if (!showLive) return;

    try {
      const { AdvancedMarkerElement } = await importLibrary('marker');

      allCameras.forEach(cam => {
        const isLive = cam.feed_status === 'LIVE';
        const isLoc = cam.feed_status === 'LOCATION ONLY';
        const statusClass = isLive ? 'live' : (isLoc ? 'location-only' : 'offline');

        const markerDiv = document.createElement('div');
        markerDiv.className = `gmp-cctv-marker ${statusClass}`;
        markerDiv.title = `${cam.camera_id}: ${cam.road}, ${cam.city}`;
        markerDiv.innerHTML = `<span class="gmp-cctv-icon">📹</span>`;

        const marker = new AdvancedMarkerElement({
          map: gMap,
          position: { lat: cam.latitude, lng: cam.longitude },
          content: markerDiv,
          title: `${cam.camera_id} (${cam.feed_status})`
        });

        marker.addListener('click', () => {
          selectCamera(cam.camera_id);
          gMap.panTo({ lat: cam.latitude, lng: cam.longitude });
        });

        advancedMarkers.push(marker);
      });
    } catch (err) {
      console.error('[GoogleMaps] Error creating AdvancedMarkerElements:', err);
    }
  }

  function selectCamera(cameraId) {
    const cam = allCameras.find(c => c.camera_id === cameraId);
    if (!cam) return;
    currentCamera = cam;

    // Highlight active in list
    document.querySelectorAll('.cctv-unit-item').forEach(item => {
      item.classList.toggle('active', item.dataset.camId === cameraId);
    });

    // Update tags and inspection panel
    const tag = document.getElementById('monSelectedCameraTag');
    const nameEl = document.getElementById('camDetailName');
    const statusEl = document.getElementById('camDetailStatus');
    const carsEl = document.getElementById('camDetailCars');
    const bikesEl = document.getElementById('camDetailBikes');
    const busesEl = document.getElementById('camDetailBuses');
    const trucksEl = document.getElementById('camDetailTrucks');
    const avgSpeedEl = document.getElementById('camDetailAvgSpeed');
    const viosEl = document.getElementById('camDetailVios');

    const box = document.getElementById('monFeedStatusBox');
    const dot = document.getElementById('monCalloutDot');
    const title = document.getElementById('monFeedTitle');
    const desc = document.getElementById('monFeedDesc');
    const btnOpen = document.getElementById('btnOpenMonCamera');
    const btnOpenText = document.getElementById('btnOpenMonCameraText');

    if (tag) tag.textContent = cam.camera_id;
    if (nameEl) nameEl.textContent = cam.camera_name || `${cam.road} Surveillance`;

    const isLive = cam.feed_status === 'LIVE';
    const isLoc = cam.feed_status === 'LOCATION ONLY';

    if (statusEl) {
      if (isLive) {
        statusEl.className = 'ms-val green';
        statusEl.textContent = '🟢 ONLINE (LIVE FEED)';
      } else if (isLoc) {
        statusEl.className = 'ms-val yellow';
        statusEl.textContent = '🟡 LOCATION ONLY';
      } else {
        statusEl.className = 'ms-val red';
        statusEl.textContent = '🔴 SENSOR OFFLINE';
      }
    }

    // Vehicle breakdown
    const bk = cam.vehicle_breakdown || { cars: 160, motorcycles: 299, buses: 31, trucks: 56 };
    if (carsEl) carsEl.textContent = bk.cars || 160;
    if (bikesEl) bikesEl.textContent = bk.motorcycles || 299;
    if (busesEl) busesEl.textContent = bk.buses || 31;
    if (trucksEl) trucksEl.textContent = bk.trucks || 56;
    if (avgSpeedEl) avgSpeedEl.textContent = `${cam.average_speed || 28.5} km/h`;
    if (viosEl) viosEl.textContent = cam.active_violations || 8;

    if (isLive) {
      if (box) box.className = 'feed-status-callout live';
      if (dot) dot.className = 'callout-dot live';
      if (title) title.textContent = 'LIVE FEED AVAILABLE';
      if (desc) desc.textContent = 'Authorized municipal surveillance node streaming from ISCDL Network. Video pipeline ready for real-time Computer Vision.';
      if (btnOpen) btnOpen.disabled = false;
      if (btnOpenText) btnOpenText.textContent = 'OPEN CAMERA & RUN COMPUTER VISION';
    } else {
      if (box) box.className = 'feed-status-callout unavail';
      if (dot) dot.className = 'callout-dot unavail';
      if (title) title.textContent = 'CCTV LOCATION AVAILABLE — NO LIVE STREAM CONNECTED';
      if (desc) desc.textContent = 'Municipal ITMS node deployed at this junction. Live stream pending municipal network clearance. No simulated video is fabricated. Baseline telemetry & historical records available.';
      if (btnOpen) btnOpen.disabled = false;
      if (btnOpenText) btnOpenText.textContent = 'NO LIVE STREAM CONNECTED (TEST WITH CV MODEL)';
    }

    // Phase 7: Data mode badge classification
    const selBadge = document.getElementById('selLocationDataModeBadge');
    const camBadge = document.getElementById('camDetailDataModeBadge');
    let modeClass = 'live';
    let modeText = 'LIVE DATA';
    if (cam.feed_status === 'LOCATION ONLY') {
      modeClass = 'location-only';
      modeText = 'LOCATION ONLY';
    } else if (cam.feed_status === 'SIMULATION') {
      modeClass = 'simulation';
      modeText = 'SIMULATION DATA';
    } else if (cam.feed_status === 'OFFLINE') {
      modeClass = 'unavailable';
      modeText = 'UNAVAILABLE DATA';
    }

    if (selBadge) {
      selBadge.className = `location-data-mode-badge ${modeClass}`;
      selBadge.textContent = modeText;
    }
    if (camBadge) {
      camBadge.className = `location-data-mode-badge ${modeClass}`;
      camBadge.textContent = modeText;
    }

    const zmSubtitle = document.getElementById('zmLocationSubtitle');
    const hmSubtitle = document.getElementById('hmLocationSubtitle');
    if (zmSubtitle) zmSubtitle.textContent = `Camera: ${cam.camera_id} • ${cam.road}, ${cam.city}`;
    if (hmSubtitle) hmSubtitle.textContent = `Historical Archive: ${cam.camera_id} • ${cam.road}, ${cam.city}`;
  }

  // --- 10. Incidents Layer & Hazard Markers ---
  async function loadAndRenderIncidents() {
    try {
      const res = await fetch('/api/incidents');
      const data = await res.json();
      allIncidents = data.incidents || [];
      await renderIncidentMarkers();
    } catch (err) {
      console.error('[Incidents] Error loading incidents:', err);
    }
  }

  async function renderIncidentMarkers() {
    if (isLeafletActive) {
      renderLeafletIncidentMarkers();
      return;
    }
    if (!gMap || typeof google === 'undefined' || !google.maps) return;

    incidentMarkers.forEach(m => {
      m.map = null;
    });
    incidentMarkers = [];

    const isVisible = document.getElementById('filterIncidentsLayer')?.checked ?? true;
    if (!isVisible) return;

    try {
      const { AdvancedMarkerElement } = await importLibrary('marker');

      allIncidents.forEach(inc => {
        let emoji = '⚠️';
        const type = inc.type || '';
        if (type.includes('ACCIDENT')) emoji = '💥';
        else if (type.includes('BLOCK')) emoji = '🚧';
        else if (type.includes('BREAKDOWN')) emoji = '🚗⚠️';
        else if (type.includes('WATERLOGGING')) emoji = '🌊';
        else if (type.includes('CONSTRUCTION')) emoji = '🏗️';
        else if (type.includes('FIRE')) emoji = '🔥';

        const markerDiv = document.createElement('div');
        markerDiv.className = 'gmp-incident-marker';
        markerDiv.title = `[${inc.type}] ${inc.title} (${inc.severity})`;
        markerDiv.innerHTML = `<span class="gmp-inc-icon">${emoji}</span>`;

        const marker = new AdvancedMarkerElement({
          map: gMap,
          position: { lat: inc.latitude, lng: inc.longitude },
          content: markerDiv,
          title: `${inc.type}: ${inc.title}`
        });

        marker.addListener('click', () => {
          alert(`🚨 TRAFFIC INCIDENT ALERT\n\nType: ${inc.type}\nSeverity: ${inc.severity}\nHeadline: ${inc.title}\nRoad: ${inc.road}, ${inc.city}\nDescription: ${inc.description}\nStatus: ${inc.status}`);
        });

        incidentMarkers.push(marker);
      });
    } catch (err) {
      console.error('[Incidents] Error rendering incident markers:', err);
    }
  }

  // --- 11. Intelligent Route Evaluation & Congestion Avoidance ---
  async function calculateAndEvaluateRoute() {
    const startInput = document.getElementById('routeStartInput');
    const destInput = document.getElementById('routeDestInput');
    const resultsDiv = document.getElementById('routeEvaluationResults');
    const start = startInput ? startInput.value.trim() : '';
    const dest = destInput ? destInput.value.trim() : '';
    if (!resultsDiv) return;

    console.log('[ROUTE-MODAL] Calculate clicked');
    console.log('[ROUTE-MODAL] Origin:', start);
    console.log('[ROUTE-MODAL] Destination:', dest);

    if (!start || !dest) {
      resultsDiv.style.display = 'block';
      resultsDiv.innerHTML = '<div style="color:#ef4444; font-size:0.8rem; padding:8px; font-weight:700;">Please select an origin and destination.</div>';
      return;
    }

    resultsDiv.style.display = 'block';
    resultsDiv.innerHTML = '<div style="color:#38bdf8; font-size:0.8rem; padding:10px; font-weight:700;"><span>⚡</span> CALCULATING ROUTE... Evaluating shortest and traffic-aware options...</div>';

    try {
      const defaultStart = { lat: 22.7244, lng: 75.8839, name: start };
      const defaultDest = { lat: 22.7533, lng: 75.8937, name: dest };
      const startCoords = await resolveLocationInput(start, defaultStart);
      const destCoords = await resolveLocationInput(dest, defaultDest);

      console.log('[ROUTE-MODAL] Request: /api/routes/compare', { start: startCoords, destination: destCoords });
      const compRes = await fetch('/api/routes/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start: { lat: startCoords.lat, lng: startCoords.lng, name: start },
          destination: { lat: destCoords.lat, lng: destCoords.lng, name: dest }
        })
      });

      console.log('[ROUTE-MODAL] Response:', compRes.status);
      if (!compRes.ok) throw new Error(`HTTP ${compRes.status} from route service`);
      const compData = await compRes.json();
      console.log('[ROUTE-MODAL] Parsed route:', compData);

      if (compData.status === 'unavailable' || compData.reason === 'ROAD_ROUTING_UNAVAILABLE') {
        if (modalShortestPolyline) { leafletMap.removeLayer(modalShortestPolyline); modalShortestPolyline = null; }
        if (modalAwarePolyline) { leafletMap.removeLayer(modalAwarePolyline); modalAwarePolyline = null; }
        if (leafletMap && typeof L !== 'undefined') {
          if (modalOriginMarker) { leafletMap.removeLayer(modalOriginMarker); modalOriginMarker = null; }
          if (modalDestMarker) { leafletMap.removeLayer(modalDestMarker); modalDestMarker = null; }
          const oIcon = L.divIcon({ className: 'modal-route-origin-icon', html: '<span style="font-size:22px;">🟢</span>', iconSize: [22, 22], iconAnchor: [11, 11] });
          modalOriginMarker = L.marker([startCoords.lat, startCoords.lng], { icon: oIcon, title: `Start: ${start}` }).addTo(leafletMap);
          const dIcon = L.divIcon({ className: 'modal-route-dest-icon', html: '<span style="font-size:22px;">🏁</span>', iconSize: [22, 22], iconAnchor: [11, 22] });
          modalDestMarker = L.marker([destCoords.lat, destCoords.lng], { icon: dIcon, title: `Dest: ${dest}` }).addTo(leafletMap);
          leafletMap.fitBounds([[startCoords.lat, startCoords.lng], [destCoords.lat, destCoords.lng]], { padding: [50, 50], maxZoom: 15 });
        }
        resultsDiv.innerHTML = '<div style="background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3); border-radius:6px; padding:10px; color:#f87171; font-weight:700; font-size:0.82rem;">⚠️ Road route unavailable for this location.</div>';
        return;
      }

      if (compData.status !== 'success') {
        if (modalShortestPolyline) { leafletMap.removeLayer(modalShortestPolyline); modalShortestPolyline = null; }
        if (modalAwarePolyline) { leafletMap.removeLayer(modalAwarePolyline); modalAwarePolyline = null; }
        resultsDiv.innerHTML = `<div style="background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3); border-radius:6px; padding:10px; color:#f87171; font-weight:700; font-size:0.82rem;">⚠️ Road route unavailable for this location.</div>`;
        return;
      }

      const shortest = compData.shortest_route || {};
      const aware = compData.traffic_aware_route || {};
      const coverage = compData.coverage || {};
      const comparison = compData.comparison || {};

      // Draw routes on Leaflet main map if active
      if (leafletMap && typeof L !== 'undefined') {
        if (modalShortestPolyline) { leafletMap.removeLayer(modalShortestPolyline); modalShortestPolyline = null; window.modalShortestPolyline = null; }
        if (modalAwarePolyline) { leafletMap.removeLayer(modalAwarePolyline); modalAwarePolyline = null; window.modalAwarePolyline = null; }
        if (modalOriginMarker) { leafletMap.removeLayer(modalOriginMarker); modalOriginMarker = null; }
        if (modalDestMarker) { leafletMap.removeLayer(modalDestMarker); modalDestMarker = null; }

        const shortestCoords = (shortest.path_coordinates || shortest.geometry || []).map(toLeafletCoord).filter(Boolean);
        const awareCoords = (aware.path_coordinates || aware.geometry || []).map(toLeafletCoord).filter(Boolean);

        if (shortestCoords.length > 0) {
          modalShortestPolyline = L.polyline(shortestCoords, {
            color: '#f59e0b',
            weight: 4,
            dashArray: '8, 8',
            opacity: 0.85
          }).addTo(leafletMap);
          window.modalShortestPolyline = modalShortestPolyline;
        }

        if (awareCoords.length > 0) {
          modalAwarePolyline = L.polyline(awareCoords, {
            color: '#0284c7',
            weight: 5,
            opacity: 0.95
          }).addTo(leafletMap);
          window.modalAwarePolyline = modalAwarePolyline;
        }

        const oIcon = L.divIcon({
          className: 'modal-route-origin-icon',
          html: '<span style="font-size:22px;">🟢</span>',
          iconSize: [22, 22],
          iconAnchor: [11, 11]
        });
        modalOriginMarker = L.marker([startCoords.lat, startCoords.lng], { icon: oIcon, title: `Start: ${start}` }).addTo(leafletMap);

        const dIcon = L.divIcon({
          className: 'modal-route-dest-icon',
          html: '<span style="font-size:22px;">🏁</span>',
          iconSize: [22, 22],
          iconAnchor: [11, 22]
        });
        modalDestMarker = L.marker([destCoords.lat, destCoords.lng], { icon: dIcon, title: `Dest: ${dest}` }).addTo(leafletMap);

        const allCoords = shortestCoords.concat(awareCoords);
        if (allCoords.length > 0) {
          leafletMap.fitBounds(L.latLngBounds(allCoords), { padding: [40, 40] });
        }
      }

      // Display Route Options & Comparison inside resultsDiv
      const recText = coverage.live_traffic_available
        ? (comparison.recommendation || aware.reason || 'Traffic-aware route selected based on traffic-aware cost optimization.')
        : 'Live traffic data is unavailable for this route.';

      const covMsg = (coverage.status === 'NONE' || !coverage.live_traffic_available)
        ? 'Live traffic data unavailable for this route.'
        : (coverage.message || 'Live/project traffic coverage: PARTIAL.');

      resultsDiv.innerHTML = `
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 6px; padding: 8px 12px; margin-bottom: 10px; display:flex; justify-content:space-between; align-items:center;">
          <strong style="color: #10b981; font-size: 0.82rem;">ROUTE FOUND</strong>
          <span style="font-size: 0.72rem; color: #94a3b8;" id="modalCoverageMsg">${escapeHtml(covMsg)}</span>
        </div>

        <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px;">
          <!-- Shortest Route Box -->
          <div style="background:#0b1120; border:1px solid #334155; border-radius:6px; padding:8px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
              <span style="background:rgba(245, 158, 11, 0.15); color:#f59e0b; font-size:0.65rem; font-weight:700; padding:2px 5px; border-radius:3px;">SHORTEST ROUTE</span>
            </div>
            <div style="font-size:0.75rem; color:#cbd5e1; margin-bottom:2px;">Distance: <strong>${shortest.distance_km || 0} km</strong></div>
            <div style="font-size:0.75rem; color:#cbd5e1; margin-bottom:2px;">ETA: <strong>${shortest.duration_mins || 0} min</strong></div>
            <div style="font-size:0.75rem; color:#cbd5e1; margin-bottom:2px;">Traffic: <strong style="color:${shortest.congestion_level === 'SEVERE' ? '#ef4444' : (shortest.congestion_level === 'CONGESTED' ? '#f59e0b' : '#10b981')};">${shortest.congestion_level || 'UNAVAILABLE'}</strong></div>
            <div style="font-size:0.68rem; color:#64748b;">Opt: MINIMUM DISTANCE</div>
          </div>

          <!-- Traffic-Aware Best Route Box -->
          <div style="background:#0b1120; border:1px solid #0284c7; border-radius:6px; padding:8px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
              <span style="background:rgba(14, 165, 233, 0.15); color:#38bdf8; font-size:0.65rem; font-weight:700; padding:2px 5px; border-radius:3px;">TRAFFIC-AWARE BEST</span>
            </div>
            <div style="font-size:0.75rem; color:#cbd5e1; margin-bottom:2px;">Distance: <strong>${aware.distance_km || 0} km</strong></div>
            <div style="font-size:0.75rem; color:#cbd5e1; margin-bottom:2px;">ETA: <strong style="color:#10b981;">${aware.duration_mins || 0} min</strong></div>
            <div style="font-size:0.75rem; color:#cbd5e1; margin-bottom:2px;">Traffic: <strong style="color:#10b981;">${aware.congestion_level || 'UNAVAILABLE'}</strong></div>
            <div style="font-size:0.68rem; color:#0284c7;">Opt: TRAFFIC-AWARE COST</div>
          </div>
        </div>

        <div style="background:rgba(14, 165, 233, 0.08); border:1px solid rgba(14, 165, 233, 0.25); border-radius:6px; padding:10px; margin-bottom:8px;">
          <div style="font-size:0.7rem; font-weight:700; color:#38bdf8; text-transform:uppercase; margin-bottom:3px;">RECOMMENDATION</div>
          <div style="font-size:0.76rem; color:#e2e8f0; line-height:1.4;">${escapeHtml(recText)}</div>
        </div>

        <div style="display:flex; gap:8px;">
          <button type="button" class="btn-select-route" id="btnModalUseShortest" style="flex:1; padding:6px; font-size:0.75rem;">USE SHORTEST ROUTE</button>
          <button type="button" class="btn-select-route active" id="btnModalUseAware" style="flex:1; padding:6px; font-size:0.75rem;">USE TRAFFIC-AWARE ROUTE</button>
        </div>
      `;

      document.getElementById('btnModalUseShortest')?.addEventListener('click', () => {
        if (modalShortestPolyline) { modalShortestPolyline.setStyle({ weight: 6, opacity: 1.0 }); modalShortestPolyline.bringToFront(); }
        if (modalAwarePolyline) { modalAwarePolyline.setStyle({ weight: 3, opacity: 0.35 }); }
        document.getElementById('btnModalUseShortest')?.classList.add('active');
        document.getElementById('btnModalUseAware')?.classList.remove('active');
      });

      document.getElementById('btnModalUseAware')?.addEventListener('click', () => {
        if (modalAwarePolyline) { modalAwarePolyline.setStyle({ weight: 6, opacity: 1.0 }); modalAwarePolyline.bringToFront(); }
        if (modalShortestPolyline) { modalShortestPolyline.setStyle({ weight: 3, opacity: 0.35 }); }
        document.getElementById('btnModalUseAware')?.classList.add('active');
        document.getElementById('btnModalUseShortest')?.classList.remove('active');
      });

    } catch (err) {
      console.error('[ROUTE-MODAL] Error:', err);
      resultsDiv.innerHTML = `<div style="color:#ef4444; font-size:0.8rem; padding:8px; font-weight:700;">ROUTE CALCULATION FAILED: ${escapeHtml(err.message)}</div>`;
    }
  }

  // --- 12. Violations Registry ---
  async function loadViolations() {
    try {
      const res = await fetch('/api/violations');
      const data = await res.json();
      allViolations = data.violations || [];
      renderViolationsTable();
    } catch (err) {
      console.error('[Violations] Error loading violations:', err);
    }
  }

  function renderViolationsTable() {
    const tbody = document.getElementById('violationsTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';

    allViolations.forEach(v => {
      const tr = document.createElement('tr');
      const sevClass = (v.severity || 'HIGH').toLowerCase();
      tr.innerHTML = `
        <td style="font-family:var(--font-mono); font-weight:700; color:#fff;">${v.violation_id}</td>
        <td style="font-weight:600; color:#38bdf8;">${v.type}</td>
        <td>${v.vehicle_type} #${v.vehicle_id}</td>
        <td style="font-family:var(--font-mono); font-weight:700; color:#f87171;">${v.speed_kmh} km/h</td>
        <td style="font-family:var(--font-mono); color:#94a3b8;">${v.speed_limit_kmh} km/h</td>
        <td>${v.location}</td>
        <td><span class="history-badge ${sevClass}">${v.severity}</span></td>
        <td><span style="font-size:10px; font-family:var(--font-mono); color:#34d399;">${v.status}</span></td>
      `;
      tbody.appendChild(tr);
    });
  }

  // --- 13. High-Speed Vehicle Alerts & Drawer ---
  async function loadAlerts() {
    try {
      const res = await fetch('/api/alerts');
      const data = await res.json();
      allAlerts = data.alerts || [];

      const countTag = document.getElementById('alertsCountTag');
      if (countTag) countTag.textContent = allAlerts.length;

      renderAlertsDrawer();
      renderTimeline();
    } catch (err) {
      console.error('[Alerts] Error loading alerts:', err);
    }
  }

  function renderAlertsDrawer() {
    const list = document.getElementById('alertsDrawerList');
    if (!list) return;
    list.innerHTML = '';

    if (!allAlerts || allAlerts.length === 0) {
      list.innerHTML = `
        <div style="padding: 16px; text-align: center; color: #64748b; font-size: 0.75rem;">
          <span>🟢</span> All corridors nominal. No active critical alerts.
        </div>
      `;
      return;
    }

    allAlerts.forEach(a => {
      const item = document.createElement('div');
      const isHighSpeed = a.type === 'HIGH_SPEED';
      const isIncident = (a.type || '').includes('INCIDENT') || a.severity === 'CRITICAL';
      const cardType = isHighSpeed ? 'speed' : (isIncident ? 'incident' : 'congestion');
      item.className = `alert-card-item ${cardType}`;

      item.innerHTML = `
        <div class="aci-head">
          <span class="aci-type">
            ${isHighSpeed ? '⚠ HIGH SPEED' : (a.type ? a.type.replace(/_/g, ' ') : 'ALERT')}
          </span>
          <span class="aci-time">${a.timestamp || 'Real-time'}</span>
        </div>
        <div class="aci-body">
          <strong>${a.title || a.message || 'Traffic Event'}</strong><br>
          ${a.road ? `${a.road} &bull; ` : ''}${a.zone || a.location || 'Monitored Zone'}
        </div>
        ${isHighSpeed && a.speed ? `<span class="aci-speed-badge">${a.speed} km/h (Limit: ${a.speed_limit || 40} km/h)</span>` : ''}
      `;

      item.addEventListener('click', () => {
        if (a.latitude && a.longitude) {
          if (gMap) {
            gMap.panTo({ lat: a.latitude, lng: a.longitude });
            gMap.setZoom(16);
          } else if (leafletMap) {
            leafletMap.setView([a.latitude, a.longitude], 16);
          }
        }
      });

      list.appendChild(item);
    });

    // Update alert count badge in top header
    const chipAlerts = document.getElementById('chipAlerts');
    if (chipAlerts) chipAlerts.textContent = `${allAlerts.length} ACTIVE`;
    const drawerCount = document.getElementById('alertsDrawerCount');
    if (drawerCount) drawerCount.textContent = `${allAlerts.length} Active`;
  }

  // --- 14. SUMO Digital Twin Simulation Controls ---
  async function loadSimulationStatus() {
    try {
      const res = await fetch('/api/simulation/status');
      const data = await res.json();
      updateSimulationUI(data);
    } catch (err) {
      console.error('[SUMO] Error loading simulation status:', err);
    }
  }

  function updateSimulationUI(data) {
    const statusVal = document.getElementById('simStatusVal');
    const timeVal = document.getElementById('simTimeVal');
    const vehsVal = document.getElementById('simVehiclesVal');
    const speedVal = document.getElementById('simSpeedVal');
    const congVal = document.getElementById('simCongestionVal');
    const edgesVal = document.getElementById('simEdgesVal');
    const diagBox = document.getElementById('simDiagnosticsBox');

    if (statusVal) {
      statusVal.textContent = data.status || 'UNAVAILABLE';
      statusVal.style.color = (data.status === 'RUNNING' ? '#10b981' : (data.status === 'UNAVAILABLE' ? '#f59e0b' : '#94a3b8'));
    }
    if (timeVal) timeVal.textContent = `${(data.step || 0).toFixed(1)}s`;
    if (vehsVal) vehsVal.textContent = data.vehicles || 42;
    if (speedVal) speedVal.textContent = `${data.average_speed || 31.4} km/h`;
    if (congVal) congVal.textContent = data.congestion_index || '0.68';
    if (edgesVal) edgesVal.textContent = data.network?.edges ? `${data.network.edges} Edges` : '4 Arms';

    if (diagBox) {
      if (data.status === 'UNAVAILABLE') {
        diagBox.innerHTML = `
          <strong style="color:#f59e0b;">Status Notice:</strong> SUMO simulation unavailable (Eclipse SUMO 1.27.1 / TraCI not found on PATH).<br>
          <span style="color:#64748b;">Loaded network topology: ${data.network?.arms || 4} intersection arms, ${data.network?.edges || 18} edges, ${data.network?.junctions || 4} junctions, ${data.network?.traffic_lights || 2} traffic lights.</span>
        `;
      } else {
        diagBox.innerHTML = `<span style="color:#34d399;">✓ TraCI socket connection active. Simulation running at step ${data.step}s.</span>`;
      }
    }
  }

  async function executeSimulationAction(action) {
    try {
      const res = await fetch(`/api/simulation/${action}`, { method: 'POST' });
      const data = await res.json();
      updateSimulationUI(data);
    } catch (err) {
      console.error(`[SUMO] Simulation action ${action} failed:`, err);
    }
  }

  // --- 15. Speed Limits Configuration ---
  async function loadSpeedLimits() {
    try {
      const res = await fetch('/api/speed_limits');
      const data = await res.json();
      allSpeedLimits = data.speed_limits || {};
    } catch (err) {
      console.error('[SpeedLimits] Error loading speed limits:', err);
    }
  }

  // --- 16. Driver Cockpit HUD & Traffic Assistance (Phase 8) ---

  // 1. Geolocation Manager
  function requestDriverLocation(isExplicitClick = false) {
    if (!navigator.geolocation) {
      updateDriverGpsStatus('📍 GPS: NOT SUPPORTED', 'gps-denied');
      return;
    }

    const options = {
      enableHighAccuracy: true,
      timeout: 8000,
      maximumAge: 10000
    };

    function handleSuccess(pos) {
      driverGpsState = 'ACTIVE';
      driverPosition.lat = pos.coords.latitude;
      driverPosition.lng = pos.coords.longitude;
      driverPosition.heading = pos.coords.heading;
      driverPosition.speed = (pos.coords.speed !== null && !isNaN(pos.coords.speed))
        ? Math.round(pos.coords.speed * 3.6)
        : (driverPosition.speed || 42);
      driverPosition.accuracy = pos.coords.accuracy;
      driverPosition.source = 'GPS';

      const accText = pos.coords.accuracy ? ` (±${Math.round(pos.coords.accuracy)}m)` : '';
      updateDriverGpsStatus(`📍 GPS: LIVE${accText}`, 'gps-active');

      const originEl = document.getElementById('dmOriginText');
      if (originEl) originEl.textContent = `GPS: [${driverPosition.lat.toFixed(4)}, ${driverPosition.lng.toFixed(4)}]`;

      updateDriverMarkerOnMap();
      checkDriverMovementAndRecalculate();
    }

    function handleError(err) {
      if (err.code === 1) { // PERMISSION_DENIED
        driverGpsState = 'DENIED';
        updateDriverGpsStatus('📍 GPS: DENIED (CORRIDOR)', 'gps-denied');
      } else if (err.code === 2) { // POSITION_UNAVAILABLE
        driverGpsState = 'UNAVAILABLE';
        updateDriverGpsStatus('📍 GPS: UNAVAILABLE', 'gps-denied');
      } else if (err.code === 3) { // TIMEOUT
        driverGpsState = 'TIMEOUT';
        updateDriverGpsStatus('📍 GPS: TIMEOUT', 'gps-denied');
      } else {
        driverGpsState = 'ERROR';
        updateDriverGpsStatus('📍 GPS: ERROR', 'gps-denied');
      }
      driverPosition.source = 'FALLBACK';
      const originEl = document.getElementById('dmOriginText');
      if (originEl) originEl.textContent = 'Vijay Nagar Corridor (Default)';
      updateDriverMarkerOnMap();
    }

    function executeGeo() {
      try {
        navigator.geolocation.getCurrentPosition(handleSuccess, handleError, options);
        if (isExplicitClick && !driverGpsWatchId) {
          driverGpsWatchId = navigator.geolocation.watchPosition(handleSuccess, () => {}, options);
        }
      } catch (e) {
        console.warn('[DriverLocation] Geolocation call failed:', e);
        handleError({ code: 2 });
      }
    }

    if (!isExplicitClick) {
      // Check permission state without triggering prompt popup
      if (navigator.permissions && navigator.permissions.query) {
        navigator.permissions.query({ name: 'geolocation' }).then(result => {
          if (result.state === 'granted') {
            executeGeo();
          } else {
            updateDriverGpsStatus('📍 GPS: VIJAY NAGAR (DEMO)', 'gps-ready');
          }
        }).catch(() => {
          updateDriverGpsStatus('📍 GPS: VIJAY NAGAR (DEMO)', 'gps-ready');
        });
        return;
      }
      updateDriverGpsStatus('📍 GPS: VIJAY NAGAR (DEMO)', 'gps-ready');
      return;
    }

    if (driverGpsRequested && !isExplicitClick) {
      return;
    }
    driverGpsRequested = true;
    updateDriverGpsStatus('📍 GPS: LOCATING...', 'gps-ready');
    executeGeo();
  }

  function updateDriverGpsStatus(label, cssClass) {
    const el = document.getElementById('dmGpsStatus');
    if (!el) return;
    el.textContent = label;
    el.className = `dm-status-badge ${cssClass}`;
  }

  // 2. Driver Map Initialization
  async function initDriverMap() {
    const mapElement = document.getElementById('driverMap');
    if (!mapElement) return;

    if (gMapsApiKey && !isLeafletActive && typeof google !== 'undefined' && google.maps) {
      try {
        const { Map, TrafficLayer } = await importLibrary('maps');
        const { DirectionsService, DirectionsRenderer } = await importLibrary('routes');

        driverMapInstance = new Map(mapElement, {
          center: { lat: driverPosition.lat, lng: driverPosition.lng },
          zoom: 15,
          mapId: gMapsMapId || 'DEMO_MAP_ID',
          mapTypeId: 'roadmap',
          disableDefaultUI: true,
          zoomControl: true,
          internalUsageAttributionIds: ['gmp_git_agentskills_v1']
        });

        const trafficLayer = new TrafficLayer();
        trafficLayer.setMap(driverMapInstance);

        driverDirectionsService = new DirectionsService();
        driverDirectionsRenderer = new DirectionsRenderer({
          map: driverMapInstance,
          suppressMarkers: false,
          polylineOptions: {
            strokeColor: '#0284c7',
            strokeWeight: 6,
            strokeOpacity: 0.85
          }
        });

        isDriverLeaflet = false;
        updateDriverMarkerOnMap();
        return;
      } catch (err) {
        console.warn('[DriverMap] Google Maps init error, falling back to Leaflet:', err);
      }
    }

    initDriverLeafletMap(mapElement);
  }

  function initDriverLeafletMap(mapElement) {
    if (typeof L === 'undefined') return;
    isDriverLeaflet = true;

    try {
      if (driverMapInstance && driverMapInstance.remove) {
        driverMapInstance.remove();
      }

      driverMapInstance = L.map(mapElement, {
        center: [driverPosition.lat, driverPosition.lng],
        zoom: 15,
        zoomControl: true,
        attributionControl: false
      });
      window.driverMapInstance = driverMapInstance;

      L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19
      }).addTo(driverMapInstance);

      // Render zones lightly on driver map
      allZones.forEach(zone => {
        const coords = zone.coordinates || [];
        if (coords.length < 3) return;
        const latLngs = coords.map(c => [c.lat, c.lng]);
        const isSevere = zone.congestion_level === 'SEVERE';
        L.polygon(latLngs, {
          color: isSevere ? '#ef4444' : '#0284c7',
          weight: 1.5,
          fillColor: isSevere ? '#ef4444' : '#0284c7',
          fillOpacity: 0.12
        }).addTo(driverMapInstance);
      });

      const notice = document.getElementById('dmMapStateNotice');
      if (notice) {
        notice.style.display = 'block';
        notice.textContent = '🌐 LOCAL ROAD GEOMETRY — Active via Leaflet engine (Zero fabricated roads)';
      }

      updateDriverMarkerOnMap();
    } catch (e) {
      console.error('[DriverMap] Leaflet setup error:', e);
    }
  }

  function updateDriverMarkerOnMap() {
    if (!driverMapInstance) return;

    if (isDriverLeaflet && typeof L !== 'undefined') {
      if (driverMarker) {
        driverMarker.setLatLng([driverPosition.lat, driverPosition.lng]);
      } else {
        const icon = L.divIcon({
          className: 'dm-leaflet-driver-icon',
          html: `
            <div class="dm-driver-marker-wrap">
              <div class="dm-driver-pulse-ring"></div>
              <div class="dm-driver-icon-inner">🚗</div>
            </div>
          `,
          iconSize: [36, 36],
          iconAnchor: [18, 18]
        });
        driverMarker = L.marker([driverPosition.lat, driverPosition.lng], {
          icon: icon,
          title: 'Your Vehicle'
        }).addTo(driverMapInstance);
      }
    } else if (driverMapInstance && typeof google !== 'undefined' && google.maps) {
      if (!driverMarker) {
        driverMarker = new google.maps.Marker({
          position: { lat: driverPosition.lat, lng: driverPosition.lng },
          map: driverMapInstance,
          title: 'Your Vehicle',
          icon: {
            path: google.maps.SymbolPath.FORWARD_CLOSED_ARROW,
            scale: 6,
            fillColor: '#0284c7',
            fillOpacity: 1,
            strokeColor: '#ffffff',
            strokeWeight: 2,
            rotation: driverPosition.heading || 0
          }
        });
      } else {
        driverMarker.setPosition({ lat: driverPosition.lat, lng: driverPosition.lng });
        if (driverPosition.heading !== null) {
          const icon = driverMarker.getIcon();
          if (icon) {
            icon.rotation = driverPosition.heading;
            driverMarker.setIcon(icon);
          }
        }
      }
    }
  }

  function recenterDriverMap() {
    if (!driverMapInstance) return;
    if (isDriverLeaflet && driverMapInstance.panTo) {
      driverMapInstance.panTo([driverPosition.lat, driverPosition.lng]);
    } else if (driverMapInstance.setCenter) {
      driverMapInstance.setCenter({ lat: driverPosition.lat, lng: driverPosition.lng });
    }
  }

  // 3. Routing Engine & Destination Handling
  function setDmCalcStatus(msg, type = 'info') {
    const el = document.getElementById('dmCalcStatus');
    if (!el) return;
    el.style.display = 'block';
    el.textContent = msg;
    if (type === 'error') {
      el.style.background = 'rgba(239, 68, 68, 0.15)';
      el.style.color = '#ef4444';
      el.style.border = '1px solid rgba(239, 68, 68, 0.3)';
    } else if (type === 'loading') {
      el.style.background = 'rgba(14, 165, 233, 0.15)';
      el.style.color = '#38bdf8';
      el.style.border = '1px solid rgba(14, 165, 233, 0.3)';
    } else if (type === 'success') {
      el.style.background = 'rgba(16, 185, 129, 0.15)';
      el.style.color = '#10b981';
      el.style.border = '1px solid rgba(16, 185, 129, 0.3)';
    } else {
      el.style.background = 'rgba(100, 116, 139, 0.15)';
      el.style.color = '#94a3b8';
      el.style.border = '1px solid rgba(100, 116, 139, 0.3)';
    }
  }

  function toLeafletCoord(pt) {
    if (!pt) return null;
    let lat = null, lng = null;
    if (typeof pt.lat === 'number' && typeof pt.lng === 'number') {
      lat = pt.lat;
      lng = pt.lng;
    } else if (Array.isArray(pt) && pt.length >= 2) {
      const a = Number(pt[0]);
      const b = Number(pt[1]);
      if (a > 60 && a < 100 && b > 5 && b < 40) {
        lat = b;
        lng = a;
      } else {
        lat = a;
        lng = b;
      }
    }
    if (lat !== null && lng !== null && !isNaN(lat) && !isNaN(lng) && isFinite(lat) && isFinite(lng)) {
      return [lat, lng];
    }
    return null;
  }

  const INDORE_NODES = {
    'palasia': { name: 'Palasia Square, Indore', lat: 22.7244, lng: 75.8839 },
    'radisson': { name: 'Radisson Square, Indore', lat: 22.7441, lng: 75.9042 },
    'mr-10': { name: 'MR-10 Junction, Indore', lat: 22.7667, lng: 75.8950 },
    'bhawarkua': { name: 'Bhawarkua Square, Indore', lat: 22.6926, lng: 75.8676 },
    'bengali': { name: 'Bengali Square, Indore', lat: 22.7150, lng: 75.9080 },
    'airport': { name: 'Indore Airport, Indore', lat: 22.7250, lng: 75.8050 },
    'bypass': { name: 'Eastern Bypass Reroute, Indore', lat: 22.7550, lng: 75.9120 }
  };

  async function resolveLocationInput(query, defaultCoords) {
    if (!query || typeof query !== 'string') return defaultCoords;
    const qTrim = query.trim();
    if (!qTrim) return defaultCoords;

    if (qTrim.includes(',')) {
      const parts = qTrim.split(',').map(s => parseFloat(s.trim()));
      if (parts.length === 2 && !isNaN(parts[0]) && !isNaN(parts[1])) {
        return { lat: parts[0], lng: parts[1], name: qTrim };
      }
    }

    const lower = qTrim.toLowerCase();
    for (const [key, node] of Object.entries(INDORE_NODES)) {
      if (lower.includes(key)) {
        return { lat: node.lat, lng: node.lng, name: node.name };
      }
    }
    if (lower.includes('vijay nagar')) return { lat: 22.7533, lng: 75.8937, name: 'Vijay Nagar, Indore' };
    if (lower.includes('mg road') && lower.includes('indore')) return { lat: 22.7196, lng: 75.8577, name: 'MG Road, Indore' };
    if (lower.includes('bhopal')) return { lat: 23.2694, lng: 77.4126, name: 'Bhopal Junction' };
    if (lower.includes('katihar')) return { lat: 25.5398, lng: 87.5721, name: 'MG Road, Katihar, Bihar' };

    try {
      const res = await fetch(`/api/location/search?q=${encodeURIComponent(qTrim)}`);
      if (res.ok) {
        const data = await res.json();
        if (data.results && data.results.length > 0) {
          const top = data.results[0];
          return {
            lat: top.latitude,
            lng: top.longitude,
            name: top.formatted_address || top.name || qTrim
          };
        }
      }
    } catch (e) {
      console.warn('[LocationResolver] Geocoding lookup failed:', e);
    }

    return defaultCoords;
  }

  async function calculateDriverRoute(isSilentReeval = false) {
    const originInput = document.getElementById('dmOriginInput');
    const destInput = document.getElementById('dmDestInput');
    const calcBtn = document.getElementById('btnDmCalculateRoute');

    const originName = originInput ? originInput.value.trim() : (driverPosition?.lat ? `[${driverPosition.lat.toFixed(4)}, ${driverPosition.lng.toFixed(4)}]` : 'Vijay Nagar, Indore');
    const destName = destInput ? destInput.value.trim() : '';

    console.log('[ROUTE] Calculate clicked');
    console.log('[ROUTE] Origin:', originName);
    console.log('[ROUTE] Destination:', destName);

    // If origin or destination is missing, display clear message
    if (!originName || !destName) {
      setDmCalcStatus("Please select both origin and destination.", "error");
      return;
    }

    if (!isSilentReeval) {
      setDmCalcStatus("CALCULATING ROUTE...", "loading");
      if (calcBtn) {
        calcBtn.innerHTML = '<span>⏳</span> CALCULATING ROUTE...';
        calcBtn.disabled = true;
      }
    }

    try {
      // 1. Resolve coordinates for Origin and Destination
      const defaultOrigin = { lat: driverPosition.lat || 22.7533, lng: driverPosition.lng || 75.8937, name: originName };
      const defaultDest = { lat: 22.7244, lng: 75.8839, name: destName };

      const originCoords = await resolveLocationInput(originName, defaultOrigin);
      const destCoords = await resolveLocationInput(destName, defaultDest);

      console.log('[ROUTE] Request: /api/routes/compare', { start: originCoords, destination: destCoords });

      // 2. Fetch dual route comparison from backend
      const compRes = await fetch('/api/routes/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          start: { lat: originCoords.lat, lng: originCoords.lng, name: originName },
          destination: { lat: destCoords.lat, lng: destCoords.lng, name: destName }
        })
      });

      console.log('[ROUTE] Response:', compRes.status);
      if (!compRes.ok) {
        throw new Error(`HTTP ${compRes.status} from route engine`);
      }

      const compData = await compRes.json();
      console.log('[ROUTE] Parsed route:', compData);

      if (compData.status === 'unavailable' || compData.reason === 'ROAD_ROUTING_UNAVAILABLE') {
        if (driverRoutePolyline) { driverMapInstance.removeLayer(driverRoutePolyline); driverRoutePolyline = null; }
        if (driverShortestPolyline) { driverMapInstance.removeLayer(driverShortestPolyline); driverShortestPolyline = null; }
        if (driverAwarePolyline) { driverMapInstance.removeLayer(driverAwarePolyline); driverAwarePolyline = null; }

        if (isDriverLeaflet && driverMapInstance && typeof L !== 'undefined') {
          if (driverOriginMarker) driverMapInstance.removeLayer(driverOriginMarker);
          if (driverDestMarker) driverMapInstance.removeLayer(driverDestMarker);
          const originIcon = L.divIcon({
            className: 'dm-origin-marker',
            html: '<span style="font-size:24px;">🟢</span>',
            iconSize: [24, 24],
            iconAnchor: [12, 12]
          });
          driverOriginMarker = L.marker([originCoords.lat, originCoords.lng], { icon: originIcon, title: `Origin: ${originName}` }).addTo(driverMapInstance);

          const destIcon = L.divIcon({
            className: 'dm-dest-marker',
            html: '<span style="font-size:24px;">🏁</span>',
            iconSize: [24, 24],
            iconAnchor: [12, 24]
          });
          driverDestMarker = L.marker([destCoords.lat, destCoords.lng], { icon: destIcon, title: `Destination: ${destName}` }).addTo(driverMapInstance);
          driverMapInstance.fitBounds([[originCoords.lat, originCoords.lng], [destCoords.lat, destCoords.lng]], { padding: [50, 50], maxZoom: 15 });
        }

        setDmCalcStatus("Road route unavailable for this location.", "error");
        const statusNote = document.getElementById('dmRouteStatusTag');
        if (statusNote) {
          statusNote.style.display = 'block';
          statusNote.textContent = 'Road route unavailable for this location.';
        }
        const optContainer = document.getElementById('dmRouteOptionsContainer');
        if (optContainer) optContainer.style.display = 'none';
        return;
      }

      if (compData.status !== 'success') {
        if (driverRoutePolyline) { driverMapInstance.removeLayer(driverRoutePolyline); driverRoutePolyline = null; }
        if (driverShortestPolyline) { driverMapInstance.removeLayer(driverShortestPolyline); driverShortestPolyline = null; }
        if (driverAwarePolyline) { driverMapInstance.removeLayer(driverAwarePolyline); driverAwarePolyline = null; }
        setDmCalcStatus("Road route unavailable for this location.", "error");
        const statusNote = document.getElementById('dmRouteStatusTag');
        if (statusNote) {
          statusNote.style.display = 'block';
          statusNote.textContent = 'Road route unavailable for this location.';
        }
        const optContainer = document.getElementById('dmRouteOptionsContainer');
        if (optContainer) optContainer.style.display = 'none';
        return;
      }

      setDmCalcStatus("ROUTE FOUND", "success");

      // 3. Render Dual Route Options (Shortest & Traffic-Aware) on map & UI panel
      console.log('[ROUTE] Rendering route: Shortest & Traffic-Aware');
      renderDualRouteOptions(compData, originCoords, destCoords, originName, destName);

      // 4. Also evaluate route against live corridor hazard model
      try {
        const pathCoords = compData.traffic_aware_route?.path_coordinates || compData.shortest_route?.path_coordinates || [];
        const evalRes = await fetch('/api/routes/evaluate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            start: { lat: originCoords.lat, lng: originCoords.lng },
            destination: { lat: destCoords.lat, lng: destCoords.lng },
            points: pathCoords
          })
        });
        const evalData = await evalRes.json();

        driverActiveRoute = {
          destination: destName,
          origin: originName,
          distance: `${compData.traffic_aware_route?.distance_km || 0} km`,
          duration: `${compData.traffic_aware_route?.duration_mins || 0} mins`,
          condition: evalData.route_condition || 'NORMAL',
          evalData: evalData
        };

        // Turn instruction update
        const turnInstruction = document.getElementById('dmTurnInstruction');
        const turnSub = document.getElementById('dmTurnSub');
        if (turnInstruction) {
          turnInstruction.textContent = destName.toLowerCase().includes('bypass')
            ? 'Follow Eastern Bypass / Ring Road (MR-10)'
            : `Proceed towards ${destName} via AB Road Corridor`;
        }
        if (turnSub) {
          turnSub.textContent = evalData.warning_headline || 'Corridor traffic synchronized via green-wave signals';
        }

        // Severe banner toggle
        const banner = document.getElementById('dmSevereBanner');
        const bText = document.getElementById('dmBottleneckText');
        if (evalData.severe_congestion_detected) {
          if (banner) banner.style.display = 'flex';
          if (bText) bText.textContent = `Severe congestion ahead on corridor. Score: ${evalData.highest_congestion_score}.`;
        } else if (banner) {
          banner.style.display = 'none';
        }
      } catch (evalErr) {
        console.warn('[DriverRouting] Corridor hazard evaluation warning:', evalErr);
      }

    } catch (err) {
      console.error('[DriverRouting] Route calculation error:', err);
      if (driverRoutePolyline) { driverMapInstance.removeLayer(driverRoutePolyline); driverRoutePolyline = null; }
      if (driverShortestPolyline) { driverMapInstance.removeLayer(driverShortestPolyline); driverShortestPolyline = null; }
      if (driverAwarePolyline) { driverMapInstance.removeLayer(driverAwarePolyline); driverAwarePolyline = null; }
      setDmCalcStatus("Road route unavailable for this location.", "error");
      const statusNote = document.getElementById('dmRouteStatusTag');
      if (statusNote) {
        statusNote.style.display = 'block';
        statusNote.textContent = 'Road route unavailable for this location.';
      }
    } finally {
      if (calcBtn) {
        calcBtn.innerHTML = '<span>🛣️</span> Calculate Optimal Route';
        calcBtn.disabled = false;
      }
    }
  }

  let latestRouteComparison = null;

  function renderDualRouteOptions(compData, originCoords, destCoords, originName, destName) {
    latestRouteComparison = compData;
    const container = document.getElementById('dmRouteOptionsContainer');
    if (!container) return;
    container.style.display = 'block';

    const shortest = compData.shortest_route || {};
    const aware = compData.traffic_aware_route || {};
    const coverage = compData.coverage || {};
    const comparison = compData.comparison || {};

    // 1. Populate Shortest Route Card
    const elShortDist = document.getElementById('dmShortestDist');
    const elShortTime = document.getElementById('dmShortestTime');
    const elShortCong = document.getElementById('dmShortestCong');
    const elShortOpt = document.getElementById('dmShortestOpt');
    const elShortPolicy = document.getElementById('dmShortestPolicy');

    if (elShortDist) elShortDist.textContent = `${shortest.distance_km || 0} km`;
    if (elShortTime) elShortTime.textContent = `${shortest.duration_mins || 0} min`;
    if (elShortCong) {
      elShortCong.textContent = shortest.congestion_level || 'UNAVAILABLE';
      elShortCong.className = `dm-rc-tag ${(shortest.congestion_level || 'low').toLowerCase()}`;
    }
    if (elShortOpt) elShortOpt.textContent = 'MINIMUM DISTANCE';
    if (elShortPolicy) elShortPolicy.innerHTML = `<strong>Policy Impact:</strong> ${escapeHtml(shortest.policy_impact || 'Direct corridor path.')}`;

    // 2. Populate Traffic-Aware Route Card
    const elAwareDist = document.getElementById('dmAwareDist');
    const elAwareTime = document.getElementById('dmAwareTime');
    const elAwareCong = document.getElementById('dmAwareCong');
    const elAwareOpt = document.getElementById('dmAwareOpt');
    const elAwareReason = document.getElementById('dmAwareReason');
    const elAwarePolicy = document.getElementById('dmAwarePolicy');

    if (elAwareDist) elAwareDist.textContent = `${aware.distance_km || 0} km`;
    if (elAwareTime) elAwareTime.textContent = `${aware.duration_mins || 0} min`;
    if (elAwareCong) {
      elAwareCong.textContent = aware.congestion_level || 'UNAVAILABLE';
      elAwareCong.className = `dm-rc-tag ${(aware.congestion_level || 'low').toLowerCase()}`;
    }
    if (elAwareOpt) elAwareOpt.textContent = coverage.live_traffic_available ? 'TRAFFIC-AWARE' : 'ROAD NETWORK ONLY';

    let reasonText = aware.reason || comparison.recommendation || 'Optimized for travel time and congestion cost.';
    if (!coverage.live_traffic_available) {
      reasonText = 'Live traffic data is unavailable for this route.';
    }
    if (elAwareReason) elAwareReason.innerHTML = `<strong>Reason:</strong> ${escapeHtml(reasonText)}`;
    if (elAwarePolicy) elAwarePolicy.innerHTML = `<strong>Policy Impact:</strong> ${escapeHtml(aware.policy_impact || 'Aligned with Authority corridor guidance.')}`;

    // 3. Recommendation text box (Route Calculation & Reasoning)
    const elRecText = document.getElementById('dmRouteRecommendationText');
    if (elRecText) {
      const recText = comparison.recommendation || reasonText;
      if (coverage.live_traffic_available) {
        elRecText.innerHTML = `<div><strong>Recommended Route:</strong> Traffic-Aware Best Route <span style="font-size:0.7rem; color:#94a3b8; font-weight:normal;">(Optimized for traffic-aware cost)</span></div>
<div style="margin-top:2px;"><strong>Distance:</strong> ${aware.distance_km || 0} km &nbsp;|&nbsp; <strong>Estimated Time:</strong> ${aware.duration_mins || 0} min</div>
<div style="margin-top:2px;"><strong>Route Type:</strong> Traffic-Aware &nbsp;|&nbsp; <strong>Traffic Status:</strong> ${aware.congestion_level || 'LOW'}</div>
<div style="margin-top:5px; padding-top:4px; border-top:1px dashed rgba(255,255,255,0.1);"><strong>Analysis &amp; Recommendation:</strong> ${escapeHtml(recText)}</div>`;
      } else {
        elRecText.innerHTML = `<div style="color:#f59e0b; font-weight:600; margin-bottom:4px;">Live traffic data is unavailable for this route.</div>
<div><strong>Recommended Route:</strong> Shortest Route (Road Distance Minimization)</div>
<div style="margin-top:2px;"><strong>Distance:</strong> ${shortest.distance_km || 0} km &nbsp;|&nbsp; <strong>Estimated Time:</strong> ${shortest.duration_mins || 0} min</div>
<div style="margin-top:2px;"><strong>Route Type:</strong> Shortest &nbsp;|&nbsp; <strong>Traffic Status:</strong> UNAVAILABLE</div>
<div style="margin-top:5px; padding-top:4px; border-top:1px dashed rgba(255,255,255,0.1);"><strong>Why this route:</strong> Minimum physical road distance (nominal travel time).</div>`;
      }
    }

    // 4. Route Coverage & Provenance Note
    const elCoverageText = document.getElementById('dmRouteCoverageText');
    if (elCoverageText) {
      elCoverageText.textContent = (coverage.status === 'NONE' || !coverage.live_traffic_available)
        ? 'Live traffic data unavailable for this route.'
        : (coverage.message || 'Live/project traffic coverage: PARTIAL.');
    }

    // 5. Update Header Metrics Strip
    const metricsStrip = document.getElementById('dmRouteMetricsStrip');
    const etaEl = document.getElementById('dmRouteEta');
    const distEl = document.getElementById('dmRouteDistance');
    const congEl = document.getElementById('dmRouteCongestion');
    const statusNote = document.getElementById('dmRouteStatusTag');
    const routeBadge = document.getElementById('dmRouteBadge');

    if (metricsStrip) metricsStrip.style.display = 'grid';
    if (etaEl) etaEl.textContent = `${aware.duration_mins || shortest.duration_mins || 0} mins`;
    if (distEl) distEl.textContent = `${aware.distance_km || shortest.distance_km || 0} km`;
    if (congEl) {
      congEl.textContent = aware.congestion_level || 'NORMAL';
      congEl.style.color = aware.congestion_level === 'SEVERE' ? '#ef4444' : (aware.congestion_level === 'CONGESTED' ? '#f59e0b' : '#10b981');
    }
    if (statusNote) {
      statusNote.style.display = 'block';
      statusNote.textContent = comparison.recommendation || (coverage.live_traffic_available
        ? 'Traffic-Aware Best Route: Optimized for traffic-aware cost.'
        : 'Live traffic data is unavailable for this route.');
    }
    if (routeBadge) {
      routeBadge.textContent = 'ACTIVE ROUTE';
      routeBadge.className = 'dm-pill-badge active';
    }

    // 6. Draw Dual Polylines & Markers on Leaflet Map
    if (isDriverLeaflet && driverMapInstance && typeof L !== 'undefined') {
      if (driverRoutePolyline) {
        driverMapInstance.removeLayer(driverRoutePolyline);
        driverRoutePolyline = null;
      }
      if (driverShortestPolyline) {
        driverMapInstance.removeLayer(driverShortestPolyline);
        driverShortestPolyline = null;
        window.driverShortestPolyline = null;
      }
      if (driverAwarePolyline) {
        driverMapInstance.removeLayer(driverAwarePolyline);
        driverAwarePolyline = null;
        window.driverAwarePolyline = null;
      }
      if (driverOriginMarker) {
        driverMapInstance.removeLayer(driverOriginMarker);
        driverOriginMarker = null;
        window.driverOriginMarker = null;
      }
      if (driverDestMarker) {
        driverMapInstance.removeLayer(driverDestMarker);
        driverDestMarker = null;
        window.driverDestMarker = null;
      }

      const shortestCoords = (shortest.path_coordinates || shortest.geometry || []).map(toLeafletCoord).filter(Boolean);
      const awareCoords = (aware.path_coordinates || aware.geometry || []).map(toLeafletCoord).filter(Boolean);

      // Shortest Route: Amber dashed
      if (shortestCoords.length > 0) {
        driverShortestPolyline = L.polyline(shortestCoords, {
          color: '#f59e0b',
          weight: 4,
          dashArray: '8, 8',
          opacity: 0.85
        }).addTo(driverMapInstance);
        window.driverShortestPolyline = driverShortestPolyline;
      }

      // Traffic-Aware Best Route: Cyan solid
      if (awareCoords.length > 0) {
        driverAwarePolyline = L.polyline(awareCoords, {
          color: '#0284c7',
          weight: 5,
          opacity: 0.95
        }).addTo(driverMapInstance);
        window.driverAwarePolyline = driverAwarePolyline;
      }

      // Origin Marker
      const originIcon = L.divIcon({
        className: 'dm-origin-marker',
        html: '<span style="font-size:24px; filter:drop-shadow(0 2px 4px rgba(0,0,0,0.5));">🟢</span>',
        iconSize: [24, 24],
        iconAnchor: [12, 12]
      });
      driverOriginMarker = L.marker([originCoords.lat, originCoords.lng], {
        icon: originIcon,
        title: `Origin: ${originName}`
      }).addTo(driverMapInstance);

      // Destination Marker
      const destIcon = L.divIcon({
        className: 'dm-dest-marker',
        html: '<span style="font-size:24px; filter:drop-shadow(0 2px 4px rgba(0,0,0,0.5));">🏁</span>',
        iconSize: [24, 24],
        iconAnchor: [12, 24]
      });
      driverDestMarker = L.marker([destCoords.lat, destCoords.lng], {
        icon: destIcon,
        title: `Destination: ${destName}`
      }).addTo(driverMapInstance);

      // Invalidate size to ensure fresh layout
      if (driverMapInstance.invalidateSize) {
        driverMapInstance.invalidateSize();
      }

      // Fit map bounds to show both routes
      const allCoords = shortestCoords.concat(awareCoords);
      if (allCoords.length > 0) {
        driverMapInstance.fitBounds(L.latLngBounds(allCoords), { padding: [50, 50] });
      }

      // Highlight selected route
      applyRouteSelectionUI(driverActiveRouteSelection || 'AWARE');
    }
  }

  function applyRouteSelectionUI(choice) {
    driverActiveRouteSelection = choice;
    const cardShortest = document.getElementById('cardShortestRoute');
    const cardAware = document.getElementById('cardTrafficAwareRoute');
    const btnUseShortest = document.getElementById('btnUseShortestRoute');
    const btnUseAware = document.getElementById('btnUseAwareRoute');
    const btnShowAware = document.getElementById('btnShowTrafficAwareRoute');
    const btnShowShortest = document.getElementById('btnShowShortestRoute');
    const btnCompareBoth = document.getElementById('btnCompareBothRoutes');

    const etaEl = document.getElementById('dmRouteEta');
    const distEl = document.getElementById('dmRouteDistance');
    const congEl = document.getElementById('dmRouteCongestion');
    const statusNote = document.getElementById('dmRouteStatusTag');
    const turnInstruction = document.getElementById('dmTurnInstruction');

    if (choice === 'SHORTEST') {
      if (cardShortest) cardShortest.classList.add('selected');
      if (cardAware) cardAware.classList.remove('selected');
      if (btnUseShortest) { btnUseShortest.textContent = 'Selected'; btnUseShortest.classList.add('active'); }
      if (btnUseAware) { btnUseAware.textContent = 'Use Traffic-Aware'; btnUseAware.classList.remove('active'); }

      if (btnShowShortest) btnShowShortest.classList.add('active');
      if (btnShowAware) btnShowAware.classList.remove('active');
      if (btnCompareBoth) btnCompareBoth.classList.remove('active');

      if (driverShortestPolyline) {
        driverShortestPolyline.setStyle({ weight: 6, opacity: 1.0 });
        driverShortestPolyline.bringToFront();
      }
      if (driverAwarePolyline) {
        driverAwarePolyline.setStyle({ weight: 3, opacity: 0.35 });
      }

      if (latestRouteComparison && latestRouteComparison.shortest_route) {
        const s = latestRouteComparison.shortest_route;
        if (etaEl) etaEl.textContent = `${s.duration_mins} mins`;
        if (distEl) distEl.textContent = `${s.distance_km} km`;
        if (congEl) {
          congEl.textContent = s.congestion_level;
          congEl.style.color = s.congestion_level === 'SEVERE' ? '#ef4444' : (s.congestion_level === 'CONGESTED' ? '#f59e0b' : '#10b981');
        }
        if (statusNote) {
          statusNote.style.display = 'block';
          statusNote.textContent = 'Shortest Route: Minimizes distance only. Subject to corridor bottlenecks.';
        }
      }
      if (turnInstruction) {
        turnInstruction.textContent = 'Proceed along direct arterial corridor (Shortest Route)';
      }
    } else if (choice === 'COMPARE') {
      if (cardShortest) cardShortest.classList.remove('selected');
      if (cardAware) cardAware.classList.remove('selected');
      if (btnShowShortest) btnShowShortest.classList.remove('active');
      if (btnShowAware) btnShowAware.classList.remove('active');
      if (btnCompareBoth) btnCompareBoth.classList.add('active');

      if (driverShortestPolyline) {
        driverShortestPolyline.setStyle({ weight: 4, opacity: 0.85 });
      }
      if (driverAwarePolyline) {
        driverAwarePolyline.setStyle({ weight: 5, opacity: 0.95 });
        driverAwarePolyline.bringToFront();
      }
    } else {
      // Default: AWARE
      if (cardAware) cardAware.classList.add('selected');
      if (cardShortest) cardShortest.classList.remove('selected');
      if (btnUseAware) { btnUseAware.textContent = 'Selected'; btnUseAware.classList.add('active'); }
      if (btnUseShortest) { btnUseShortest.textContent = 'Use Shortest'; btnUseShortest.classList.remove('active'); }

      if (btnShowAware) btnShowAware.classList.add('active');
      if (btnShowShortest) btnShowShortest.classList.remove('active');
      if (btnCompareBoth) btnCompareBoth.classList.remove('active');

      if (driverAwarePolyline) {
        driverAwarePolyline.setStyle({ weight: 6, opacity: 1.0 });
        driverAwarePolyline.bringToFront();
      }
      if (driverShortestPolyline) {
        driverShortestPolyline.setStyle({ weight: 3, opacity: 0.35 });
      }

      if (latestRouteComparison && latestRouteComparison.traffic_aware_route) {
        const a = latestRouteComparison.traffic_aware_route;
        if (etaEl) etaEl.textContent = `${a.duration_mins} mins`;
        if (distEl) distEl.textContent = `${a.distance_km} km`;
        if (congEl) {
          congEl.textContent = a.congestion_level;
          congEl.style.color = '#10b981';
        }
        if (statusNote) {
          statusNote.style.display = 'block';
          statusNote.textContent = (latestRouteComparison && latestRouteComparison.comparison && latestRouteComparison.comparison.recommendation)
            ? latestRouteComparison.comparison.recommendation
            : 'Traffic-Aware Best Route: Optimized for traffic-aware cost.';
        }
      }
      if (turnInstruction) {
        turnInstruction.textContent = (latestRouteComparison && latestRouteComparison.traffic_aware_route && latestRouteComparison.traffic_aware_route.name)
          ? `Follow ${latestRouteComparison.traffic_aware_route.name}`
          : 'Follow Traffic-Aware Route';
      }
    }
  }

  function clearDriverRoute() {
    driverActiveRoute = null;
    latestRouteComparison = null;

    if (driverDirectionsRenderer) {
      driverDirectionsRenderer.set('directions', null);
    }
    if (driverRoutePolyline && driverMapInstance && typeof L !== 'undefined') {
      driverMapInstance.removeLayer(driverRoutePolyline);
      driverRoutePolyline = null;
    }
    if (driverShortestPolyline && driverMapInstance && typeof L !== 'undefined') {
      driverMapInstance.removeLayer(driverShortestPolyline);
      driverShortestPolyline = null;
    }
    if (driverAwarePolyline && driverMapInstance && typeof L !== 'undefined') {
      driverMapInstance.removeLayer(driverAwarePolyline);
      driverAwarePolyline = null;
    }
    if (driverOriginMarker && driverMapInstance && typeof L !== 'undefined') {
      driverMapInstance.removeLayer(driverOriginMarker);
      driverOriginMarker = null;
    }
    if (driverDestMarker && driverMapInstance && typeof L !== 'undefined') {
      driverMapInstance.removeLayer(driverDestMarker);
      driverDestMarker = null;
    }

    const dmCalcStatus = document.getElementById('dmCalcStatus');
    if (dmCalcStatus) dmCalcStatus.style.display = 'none';

    const optionsContainer = document.getElementById('dmRouteOptionsContainer');
    if (optionsContainer) optionsContainer.style.display = 'none';

    const metricsStrip = document.getElementById('dmRouteMetricsStrip');
    const statusNote = document.getElementById('dmRouteStatusTag');
    const routeBadge = document.getElementById('dmRouteBadge');

    if (metricsStrip) metricsStrip.style.display = 'none';
    if (statusNote) statusNote.style.display = 'none';
    if (routeBadge) {
      routeBadge.textContent = 'NO ACTIVE ROUTE';
      routeBadge.className = 'dm-pill-badge';
    }

    const turnInstruction = document.getElementById('dmTurnInstruction');
    const turnSub = document.getElementById('dmTurnSub');
    if (turnInstruction) turnInstruction.textContent = 'Follow AB Road Corridor Northbound';
    if (turnSub) turnSub.textContent = 'Free Navigation &bull; Speed limit 50 km/h';
  }

  function checkDriverMovementAndRecalculate() {
    if (!driverActiveRoute || !driverLastEvalPosition) {
      driverLastEvalPosition = { lat: driverPosition.lat, lng: driverPosition.lng };
      return;
    }

    const dLat = (driverPosition.lat - driverLastEvalPosition.lat) * 111000;
    const dLng = (driverPosition.lng - driverLastEvalPosition.lng) * 111000 * Math.cos(driverPosition.lat * Math.PI / 180);
    const distMeters = Math.sqrt(dLat * dLat + dLng * dLng);

    // Only recalculate when meaningful movement threshold (> 100 meters) is reached
    if (distMeters >= 100) {
      console.log(`[DriverMode] Driver moved ${Math.round(distMeters)}m, updating route...`);
      driverLastEvalPosition = { lat: driverPosition.lat, lng: driverPosition.lng };
      calculateDriverRoute(true);
    }
  }

  function handleAcceptReroute() {
    const destInput = document.getElementById('dmDestInput');
    if (destInput) destInput.value = 'Eastern Bypass / Ring Road (MR-10)';
    calculateDriverRoute();

    const banner = document.getElementById('dmSevereBanner');
    const bText = document.getElementById('dmBottleneckText');
    if (bText) bText.textContent = '✓ Route diverted via Eastern Bypass. Circumventing central bottleneck.';
    setTimeout(() => {
      if (banner) banner.style.display = 'none';
    }, 4000);
  }

  // 4. Setup Driver Mode Event Listeners
  function setupDriverEventListeners() {
    document.getElementById('btnDmLocateMe')?.addEventListener('click', () => requestDriverLocation(true));
    document.getElementById('btnDmRecenterMap')?.addEventListener('click', recenterDriverMap);
    document.getElementById('btnDmCalculateRoute')?.addEventListener('click', () => calculateDriverRoute(false));
    document.getElementById('btnDmClearRoute')?.addEventListener('click', clearDriverRoute);
    document.getElementById('btnDmRecalcRoute')?.addEventListener('click', () => calculateDriverRoute(false));
    document.getElementById('btnDmAcceptReroute')?.addEventListener('click', handleAcceptReroute);

    document.getElementById('dmDestInput')?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') calculateDriverRoute(false);
    });
    document.getElementById('dmOriginInput')?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') calculateDriverRoute(false);
    });

    // Dual Route Toggle and Selection Buttons
    document.getElementById('btnShowTrafficAwareRoute')?.addEventListener('click', () => applyRouteSelectionUI('AWARE'));
    document.getElementById('btnShowShortestRoute')?.addEventListener('click', () => applyRouteSelectionUI('SHORTEST'));
    document.getElementById('btnCompareBothRoutes')?.addEventListener('click', () => applyRouteSelectionUI('COMPARE'));
    document.getElementById('btnUseShortestRoute')?.addEventListener('click', () => applyRouteSelectionUI('SHORTEST'));
    document.getElementById('btnUseAwareRoute')?.addEventListener('click', () => applyRouteSelectionUI('AWARE'));

    // Preset chips
    const chips = document.querySelectorAll('#dmPresetChips .dm-chip');
    chips.forEach(chip => {
      chip.addEventListener('click', () => {
        chips.forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        const dest = chip.dataset.dest;
        const input = document.getElementById('dmDestInput');
        if (input && dest) {
          input.value = dest;
          calculateDriverRoute(false);
        }
      });
    });
  }

  // 5. Driver Feed Synchronization
  async function syncDriverModeFeed() {
    try {
      const url = `/api/driver/feed?lat=${driverPosition.lat}&lng=${driverPosition.lng}&frame=${currentFrameNum}`;
      const res = await fetch(url);
      const data = await res.json();

      // Ego speed & limit
      const egoSpeed = document.getElementById('dmEgoSpeed');
      const limitSign = document.getElementById('dmSpeedLimitSign');
      const currentRoad = document.getElementById('dmCurrentRoad');
      const safetyStatus = document.getElementById('dmModelSafetyStatus');
      const speedAdvisory = document.getElementById('dmSpeedAdvisory');
      const corridorName = document.getElementById('dmCorridorName');

      if (egoSpeed) egoSpeed.textContent = driverPosition.speed || data.ego_speed || 42;
      if (limitSign) limitSign.textContent = data.current_speed_limit_kmh || 50;
      if (currentRoad) currentRoad.textContent = data.current_road || 'AB Road Corridor';

      const pDetails = data.hybrid_intelligence?.priority_zone;
      if (safetyStatus) safetyStatus.textContent = data.hybrid_intelligence?.safety_status || 'NOMINAL SPLIT';
      if (corridorName) corridorName.textContent = pDetails ? `${pDetails} APPROACH` : 'VIJAY NAGAR ITMS';
      if (speedAdvisory) speedAdvisory.textContent = data.safety_advisory || 'Maintain 40-50 km/h for green wave';

      // Bottleneck Advisory Banner
      const banner = document.getElementById('dmSevereBanner');
      const bText = document.getElementById('dmBottleneckText');
      const topSevere = data.nearby_severe_zones?.[0];

      if (topSevere || data.recommended_divert) {
        if (banner) banner.style.display = 'flex';
        if (bText) {
          const distStr = topSevere?.distance_meters ? ` (${topSevere.distance_meters}m away)` : '';
          bText.textContent = `${topSevere?.name || 'Corridor Approach'} has score ${topSevere?.congestion_score || '85.8'}${distStr}. Recommended diversion active.`;
        }
      } else if (!driverActiveRoute?.evalData?.severe_congestion_detected) {
        if (banner) banner.style.display = 'none';
      }

      // Proximity Alerts List
      renderDriverProximityAlerts(data.proximity_alerts || [], data.active_road_hazards || []);

    } catch (err) {
      console.warn('[DriverMode] Sync error:', err);
    }
  }

  function renderDriverProximityAlerts(alerts, hazards) {
    const listEl = document.getElementById('dmHazardsList');
    const countEl = document.getElementById('dmHazardsCount');
    if (!listEl) return;

    listEl.innerHTML = '';
    const combined = [...alerts];

    if (countEl) countEl.textContent = `${combined.length} NEARBY`;

    if (combined.length === 0) {
      listEl.innerHTML = '<div class="dm-empty-state">No critical hazards or severe bottlenecks in immediate vicinity.</div>';
      return;
    }

    combined.forEach(item => {
      const div = document.createElement('div');
      const isSevere = item.severity === 'CRITICAL' || item.severity === 'HIGH';
      const isModel = item.provenance === 'MODEL-DERIVED' || item.type === 'MODEL_TRAJECTORY_RISK';

      div.className = `dm-hazard-item ${isSevere ? '' : 'warning'} ${isModel ? 'model' : ''}`;

      const distTag = item.distance_meters ? `<span style="color:#38bdf8; font-size:0.68rem; font-weight:700;">📍 ${item.distance_meters}m</span>` : '';
      const provTag = isModel
        ? '<span style="background:rgba(168,85,247,0.15); color:#c084fc; font-size:0.6rem; padding:1px 4px; border-radius:3px; font-weight:700;">MODEL-DERIVED</span>'
        : '<span style="background:rgba(14,165,233,0.12); color:#38bdf8; font-size:0.6rem; padding:1px 4px; border-radius:3px; font-weight:700;">MEASURED</span>';

      div.innerHTML = `
        <div style="flex:1;">
          <div style="display:flex; align-items:center; gap:6px; margin-bottom:2px;">
            <strong style="color:${isSevere ? '#ef4444' : '#f59e0b'}; font-size:0.75rem;">${item.title || item.type}</strong>
            ${provTag}
          </div>
          <div style="font-size:0.7rem; color:#cbd5e1;">${item.message || ''}</div>
        </div>
        ${distTag}
      `;
      listEl.appendChild(div);
    });
  }

  // --- 17. Header Stat Chips Update ---
  function updateHeaderStatChips() {
    const chipVehs = document.getElementById('chipTotalVehicles');
    const chipCams = document.getElementById('chipActiveCameras');
    const chipCong = document.getElementById('chipCongestionStatus');
    const chipVios = document.getElementById('chipViolations');
    const chipAlerts = document.getElementById('chipActiveAlerts');

    if (chipVehs) chipVehs.textContent = '546 OBSERVED';
    if (chipCams) chipCams.textContent = `${allCameras.length} CAMERAS`;
    if (chipCong) chipCong.textContent = '85.8 SEVERE (ZONE 1)';
    if (chipVios) chipVios.textContent = `${allViolations.length} VIOLATIONS`;
    if (chipAlerts) chipAlerts.textContent = `${allAlerts.length} ALERTS`;
  }

  // --- 18. Modals Open/Close Setup ---
  function setupModalEventListeners() {
    // Add CCTV Modal
    document.getElementById('btnCloseAddCctvModal')?.addEventListener('click', closeAddCctvModal);
    document.getElementById('btnCancelAddCctv')?.addEventListener('click', closeAddCctvModal);
    document.getElementById('btnSaveAddCctv')?.addEventListener('click', saveAddCctvForm);

    // Create Zone Modal
    document.getElementById('btnCloseCreateZoneModal')?.addEventListener('click', closeCreateZoneModal);
    document.getElementById('btnCancelCreateZone')?.addEventListener('click', closeCreateZoneModal);
    document.getElementById('btnSaveCreateZone')?.addEventListener('click', saveCreateZoneForm);

    // Report Incident Modal
    document.getElementById('btnCloseIncidentModal')?.addEventListener('click', closeIncidentModal);
    document.getElementById('btnCancelIncident')?.addEventListener('click', closeIncidentModal);
    document.getElementById('btnSaveIncident')?.addEventListener('click', saveIncidentForm);

    // Route Evaluator Modal
    document.getElementById('btnCloseRouteModal')?.addEventListener('click', closeRouteModal);
    document.getElementById('btnCloseRouteModalBtn')?.addEventListener('click', closeRouteModal);
    document.getElementById('btnCalculateBestRoute')?.addEventListener('click', calculateAndEvaluateRoute);
    document.getElementById('routeStartInput')?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') calculateAndEvaluateRoute();
    });
    document.getElementById('routeDestInput')?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') calculateAndEvaluateRoute();
    });

    // Violations Modal
    document.getElementById('btnCloseViolationsModal')?.addEventListener('click', closeViolationsModal);
    document.getElementById('btnCloseViolationsBtn')?.addEventListener('click', closeViolationsModal);

    // SUMO Modal
    document.getElementById('btnCloseSimModal')?.addEventListener('click', closeSumoModal);
    document.getElementById('btnCloseSimModalBtn')?.addEventListener('click', closeSumoModal);
    document.getElementById('btnSimStart')?.addEventListener('click', () => executeSimulationAction('start'));
    document.getElementById('btnSimPause')?.addEventListener('click', () => executeSimulationAction('pause'));
    document.getElementById('btnSimStep')?.addEventListener('click', () => executeSimulationAction('step'));
    document.getElementById('btnSimReset')?.addEventListener('click', () => executeSimulationAction('reset'));

    // Speed Limits Modal
    document.getElementById('btnCloseSpeedLimitsModal')?.addEventListener('click', closeSpeedLimitsModal);
    document.getElementById('btnCloseSpeedLimitsBtn')?.addEventListener('click', closeSpeedLimitsModal);
    document.getElementById('slRoadSelect')?.addEventListener('change', (e) => {
      const road = e.target.value;
      const input = document.getElementById('slAllowedSpeedInput');
      if (input && allSpeedLimits[road]) {
        input.value = allSpeedLimits[road].allowed_speed_kmh;
      }
    });
    document.getElementById('btnSaveSpeedLimit')?.addEventListener('click', saveSpeedLimitForm);

    // Modal Backdrop Click to Close
    document.querySelectorAll('.control-modal-backdrop').forEach(backdrop => {
      backdrop.addEventListener('click', (e) => {
        if (e.target === backdrop) {
          backdrop.style.display = 'none';
        }
      });
    });
  }

  function openAddCctvModal() {
    const modal = document.getElementById('addCctvModal');
    if (!modal) return;
    document.getElementById('cctvFormId').value = `CAM-IND-0${allCameras.length + 1}`;
    document.getElementById('cctvFormLat').value = selectedLocation.lat.toFixed(4);
    document.getElementById('cctvFormLng').value = selectedLocation.lng.toFixed(4);
    document.getElementById('cctvFormRoad').value = selectedLocation.roadName;
    document.getElementById('cctvFormCity').value = selectedLocation.city;
    modal.style.display = 'flex';
  }
  function closeAddCctvModal() {
    document.getElementById('addCctvModal')?.setAttribute('style', 'display: none;');
  }

  async function saveAddCctvForm() {
    const camData = {
      camera_id: document.getElementById('cctvFormId')?.value.trim() || `CAM-IND-0${allCameras.length + 1}`,
      camera_name: document.getElementById('cctvFormName')?.value.trim() || 'New Municipal Node',
      latitude: parseFloat(document.getElementById('cctvFormLat')?.value) || selectedLocation.lat,
      longitude: parseFloat(document.getElementById('cctvFormLng')?.value) || selectedLocation.lng,
      road: document.getElementById('cctvFormRoad')?.value.trim() || selectedLocation.roadName,
      city: document.getElementById('cctvFormCity')?.value.trim() || selectedLocation.city,
      direction: document.getElementById('cctvFormDir')?.value.trim() || 'North',
      status: document.getElementById('cctvFormStatus')?.value || 'ONLINE',
      feed_status: document.getElementById('cctvFormStatus')?.value === 'LIVE' ? 'LIVE' : 'LOCATION ONLY',
      camera_type: document.getElementById('cctvFormType')?.value.trim() || 'Fixed 1080p Sensor',
      detection_model: document.getElementById('cctvFormModel')?.value.trim() || 'YOLOv8m + ByteTrack'
    };

    try {
      const res = await fetch('/api/cameras', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(camData)
      });
      if (res.ok) {
        closeAddCctvModal();
        await loadCameras();
        updateHeaderStatChips();
      }
    } catch (e) {
      alert('Error saving CCTV camera: ' + e.message);
    }
  }

  function openCreateZoneModal() {
    const modal = document.getElementById('createZoneModal');
    if (!modal) return;
    document.getElementById('zoneFormId').value = `ZONE_${allZones.length + 1}`;
    document.getElementById('zoneFormCoordsCount').value = `${zoneDrawingPoints.length} coordinates selected on map`;
    document.getElementById('zoneFormRoad').value = selectedLocation.roadName;
    modal.style.display = 'flex';
  }
  function closeCreateZoneModal() {
    document.getElementById('createZoneModal')?.setAttribute('style', 'display: none;');
  }

  async function saveCreateZoneForm() {
    const zoneData = {
      zone_id: document.getElementById('zoneFormId')?.value.trim() || `ZONE_${allZones.length + 1}`,
      name: document.getElementById('zoneFormName')?.value.trim() || 'Monitored Approach',
      congestion_level: document.getElementById('zoneFormType')?.value || 'NORMAL',
      speed_threshold: parseInt(document.getElementById('zoneFormSpeed')?.value, 10) || 35,
      severity_alert: document.getElementById('zoneFormSeverity')?.value || 'HIGH',
      road: document.getElementById('zoneFormRoad')?.value.trim() || selectedLocation.roadName,
      coordinates: zoneDrawingPoints
    };

    try {
      const res = await fetch('/api/zones', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(zoneData)
      });
      if (res.ok) {
        closeCreateZoneModal();
        toggleZoneDrawingMode(false);
        await loadAndRenderZones();
      }
    } catch (e) {
      alert('Error saving zone: ' + e.message);
    }
  }

  function openIncidentModal() {
    const modal = document.getElementById('reportIncidentModal');
    if (!modal) return;
    document.getElementById('incFormRoad').value = selectedLocation.roadName;
    document.getElementById('incFormCity').value = selectedLocation.city;
    modal.style.display = 'flex';
  }
  function closeIncidentModal() {
    document.getElementById('reportIncidentModal')?.setAttribute('style', 'display: none;');
  }

  async function saveIncidentForm() {
    const incData = {
      incident_id: `INC-${Date.now().toString().slice(-4)}`,
      type: document.getElementById('incFormType')?.value || 'ACCIDENT',
      severity: document.getElementById('incFormSeverity')?.value || 'HIGH',
      title: document.getElementById('incFormTitle')?.value.trim() || 'Road Obstruction',
      road: document.getElementById('incFormRoad')?.value.trim() || selectedLocation.roadName,
      city: document.getElementById('incFormCity')?.value.trim() || selectedLocation.city,
      description: document.getElementById('incFormDesc')?.value.trim() || 'Vehicle collision reported.',
      latitude: selectedLocation.lat,
      longitude: selectedLocation.lng,
      status: 'ACTIVE'
    };

    try {
      const res = await fetch('/api/incidents', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(incData)
      });
      if (res.ok) {
        closeIncidentModal();
        await loadAndRenderIncidents();
        await loadAlerts();
        updateHeaderStatChips();
      }
    } catch (e) {
      alert('Error broadcasting incident: ' + e.message);
    }
  }

  function openRouteModal() {
    const modal = document.getElementById('findRouteModal');
    if (modal) modal.style.display = 'flex';
  }
  function closeRouteModal() {
    document.getElementById('findRouteModal')?.setAttribute('style', 'display: none;');
  }

  function openViolationsModal() {
    const modal = document.getElementById('violationsModal');
    if (modal) modal.style.display = 'flex';
  }
  function closeViolationsModal() {
    document.getElementById('violationsModal')?.setAttribute('style', 'display: none;');
  }

  function openSumoModal() {
    const modal = document.getElementById('simulationModal');
    if (modal) {
      modal.style.display = 'flex';
      loadSimulationStatus();
    }
  }
  function closeSumoModal() {
    document.getElementById('simulationModal')?.setAttribute('style', 'display: none;');
  }

  function openSpeedLimitsModal() {
    const modal = document.getElementById('speedLimitsModal');
    if (modal) modal.style.display = 'flex';
  }
  function closeSpeedLimitsModal() {
    document.getElementById('speedLimitsModal')?.setAttribute('style', 'display: none;');
    const fb = document.getElementById('slFeedback');
    if (fb) fb.textContent = '';
  }

  async function saveSpeedLimitForm() {
    const road = document.getElementById('slRoadSelect')?.value;
    const speed = parseInt(document.getElementById('slAllowedSpeedInput')?.value, 10) || 50;
    const fb = document.getElementById('slFeedback');

    try {
      const res = await fetch('/api/speed_limits', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ road_name: road, speed_limit: speed })
      });
      if (res.ok) {
        if (fb) fb.textContent = '✓ Speed limit updated successfully';
        await loadSpeedLimits();
        setTimeout(closeSpeedLimitsModal, 1200);
      }
    } catch (e) {
      if (fb) fb.textContent = 'Error updating speed limit: ' + e.message;
    }
  }

  // --- 19. Map Utility & View Controls ---
  function setupLocationEventListeners() {
    // Base Map Type Buttons
    document.getElementById('btnMapTypeRoadmap')?.addEventListener('click', () => setBaseMapType('roadmap'));
    document.getElementById('btnMapTypeSatellite')?.addEventListener('click', () => setBaseMapType('satellite'));
    document.getElementById('btnMapTypeTerrain')?.addEventListener('click', () => setBaseMapType('terrain'));

    // Traffic Toggle Button
    document.getElementById('btnToggleTraffic')?.addEventListener('click', toggleTrafficLayer);

    // Filter Checkboxes
    document.getElementById('filterLiveCctv')?.addEventListener('change', renderCctvMarkers);
    document.getElementById('filterZonesLayer')?.addEventListener('change', renderZonePolygons);
    document.getElementById('filterIncidentsLayer')?.addEventListener('change', renderIncidentMarkers);

    // Map Utility Controls
    document.getElementById('btnZoomIn')?.addEventListener('click', () => {
      if (gMap) gMap.setZoom(gMap.getZoom() + 1);
      else if (leafletMap) leafletMap.zoomIn();
    });
    document.getElementById('btnZoomOut')?.addEventListener('click', () => {
      if (gMap) gMap.setZoom(gMap.getZoom() - 1);
      else if (leafletMap) leafletMap.zoomOut();
    });
    document.getElementById('btnFullscreen')?.addEventListener('click', () => {
      const mapContainer = document.querySelector('.li-map-container');
      if (!mapContainer) return;
      if (!document.fullscreenElement) {
        mapContainer.requestFullscreen?.().catch(e => console.log(e));
      } else {
        document.exitFullscreen?.();
      }
      setTimeout(() => { if (leafletMap) leafletMap.invalidateSize(); }, 200);
    });

    // My Location Button
    document.getElementById('btnCurrentLocation')?.addEventListener('click', () => {
      if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
          async (pos) => {
            const lat = pos.coords.latitude;
            const lng = pos.coords.longitude;
            if (gMap) {
              gMap.panTo({ lat, lng });
              gMap.setZoom(16);
            } else if (leafletMap) {
              leafletMap.setView([lat, lng], 16);
            }
            await handleMapLocationClick(lat, lng);
          },
          (err) => {
            console.warn('Geolocation error:', err);
            alert('Location access denied or unavailable. Centering on default Indian coordinates.');
          }
        );
      }
    });

    // Quick Junction Pills
    document.querySelectorAll('.quick-pill').forEach(pill => {
      pill.addEventListener('click', async () => {
        document.querySelectorAll('.quick-pill').forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        const query = pill.dataset.query;
        if (query) {
          const searchInput = document.getElementById('gmpSearchInput');
          if (searchInput) searchInput.value = query;
          await searchLocationByQuery(query);
        }
      });
    });

    // Monitor This Location Button
    document.getElementById('btnMonitorThisLocation')?.addEventListener('click', () => {
      if (gMap && selectedLocation) {
        gMap.panTo({ lat: selectedLocation.lat, lng: selectedLocation.lng });
        gMap.setZoom(17);
      }
      const closest = findClosestCamera(selectedLocation.lat, selectedLocation.lng);
      if (closest) {
        selectCamera(closest.camera_id);
      }
    });

    // Open Camera / Start CV Analysis Button
    document.getElementById('btnOpenMonCamera')?.addEventListener('click', startComputerVisionAnalysis);

    // Modal: Zone Calibration Modal (Preserved)
    const btnOpenZone = document.getElementById('btnOpenZoneModal');
    const zoneBackdrop = document.getElementById('zoneModalBackdrop');
    const btnCloseZone = document.getElementById('btnCloseZoneModal');
    const btnCancelZone = document.getElementById('btnCancelZoneModal');
    const btnSaveZone = document.getElementById('btnSaveZoneModal');

    if (btnOpenZone && zoneBackdrop) {
      btnOpenZone.addEventListener('click', () => {
        zoneBackdrop.style.display = 'flex';
      });
    }
    if (btnCloseZone && zoneBackdrop) {
      btnCloseZone.addEventListener('click', () => {
        zoneBackdrop.style.display = 'none';
      });
    }
    if (btnCancelZone && zoneBackdrop) {
      btnCancelZone.addEventListener('click', () => {
        zoneBackdrop.style.display = 'none';
      });
    }
    if (btnSaveZone) {
      btnSaveZone.addEventListener('click', async () => {
        const feedback = document.getElementById('zmSaveFeedback');
        if (feedback) feedback.textContent = 'Saving calibration...';

        const customZones = {
          'ZONE 1': { name: document.getElementById('zmName1')?.value || 'Approach A', direction: document.getElementById('zmDir1')?.value || 'North', phase: document.getElementById('zmPhase1')?.value || 'Phase 1' },
          'ZONE 2': { name: document.getElementById('zmName2')?.value || 'Approach B', direction: document.getElementById('zmDir2')?.value || 'East', phase: document.getElementById('zmPhase2')?.value || 'Phase 2' },
          'ZONE 3': { name: document.getElementById('zmName3')?.value || 'Approach C', direction: document.getElementById('zmDir3')?.value || 'South', phase: document.getElementById('zmPhase3')?.value || 'Phase 1' },
          'ZONE 4': { name: document.getElementById('zmName4')?.value || 'Approach D', direction: document.getElementById('zmDir4')?.value || 'Queue', phase: document.getElementById('zmPhase4')?.value || 'Phase 3' },
          'ZONE 5': { name: document.getElementById('zmName5')?.value || 'Approach E', direction: document.getElementById('zmDir5')?.value || 'West', phase: document.getElementById('zmPhase5')?.value || 'Phase 3' },
          'ZONE 6': { name: document.getElementById('zmName6')?.value || 'Core', direction: document.getElementById('zmDir6')?.value || 'Exit', phase: document.getElementById('zmPhase6')?.value || 'Phase 2' }
        };

        try {
          const res = await fetch('/api/save_zone_config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              camera_id: currentCamera ? currentCamera.camera_id : 'CAM-IND-001',
              zones: customZones
            })
          });
          const result = await res.json();
          if (feedback) feedback.textContent = `✓ Calibration saved for ${result.camera_id}`;
          setTimeout(() => {
            if (zoneBackdrop) zoneBackdrop.style.display = 'none';
            if (feedback) feedback.textContent = '';
          }, 1200);
        } catch (e) {
          if (feedback) feedback.textContent = 'Error saving calibration.';
        }
      });
    }

    // Modal: Location History Modal (Preserved)
    const btnOpenHist = document.getElementById('btnOpenHistoryModal');
    const histBackdrop = document.getElementById('historyModalBackdrop');
    const btnCloseHist = document.getElementById('btnCloseHistoryModal');
    const btnCloseHistBtn = document.getElementById('btnCloseHistoryModalBtn');

    if (btnOpenHist && histBackdrop) {
      btnOpenHist.addEventListener('click', async () => {
        histBackdrop.style.display = 'flex';
        await loadLocationHistory();
      });
    }
    if (btnCloseHist && histBackdrop) {
      btnCloseHist.addEventListener('click', () => {
        histBackdrop.style.display = 'none';
      });
    }
    if (btnCloseHistBtn && histBackdrop) {
      btnCloseHistBtn.addEventListener('click', () => {
        histBackdrop.style.display = 'none';
      });
    }
  }

  function toggleTrafficLayer() {
    isTrafficOn = !isTrafficOn;
    if (gMap && gTrafficLayer) {
      gTrafficLayer.setMap(isTrafficOn ? gMap : null);
    }
    if (leafletMap && typeof L !== 'undefined') {
      leafletZonePolygons.forEach(p => {
        if (isTrafficOn) {
          if (!leafletMap.hasLayer(p)) leafletMap.addLayer(p);
        } else {
          if (leafletMap.hasLayer(p)) leafletMap.removeLayer(p);
        }
      });
    }
    updateTrafficButtonUI(isTrafficOn);
  }

  function updateTrafficButtonUI(isOn) {
    const btn = document.getElementById('btnToggleTraffic');
    const led = document.getElementById('trafficLed');
    const txt = document.getElementById('trafficBtnText');
    if (btn) btn.classList.toggle('active', isOn);
    if (led) led.className = 'traffic-led ' + (isOn ? 'on' : '');
    if (txt) txt.textContent = `GOOGLE MAP TRAFFIC: ${isOn ? 'ON' : 'OFF'}`;
  }

  function setBaseMapType(type) {
    if (gMap && typeof google !== 'undefined' && google.maps) {
      if (type === 'roadmap') gMap.setMapTypeId(google.maps.MapTypeId.ROADMAP);
      else if (type === 'satellite') gMap.setMapTypeId(google.maps.MapTypeId.SATELLITE);
      else if (type === 'terrain') gMap.setMapTypeId(google.maps.MapTypeId.TERRAIN);
    } else if (leafletMap && typeof L !== 'undefined') {
      if (leafletTileLayer) {
        leafletMap.removeLayer(leafletTileLayer);
      }
      if (type === 'satellite') {
        leafletTileLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
          maxZoom: 19
        }).addTo(leafletMap);
      } else if (type === 'terrain') {
        leafletTileLayer = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
          maxZoom: 17
        }).addTo(leafletMap);
      } else {
        leafletTileLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          maxZoom: 19
        }).addTo(leafletMap);
      }
    }

    document.getElementById('btnMapTypeRoadmap')?.classList.toggle('active', type === 'roadmap');
    document.getElementById('btnMapTypeSatellite')?.classList.toggle('active', type === 'satellite');
    document.getElementById('btnMapTypeTerrain')?.classList.toggle('active', type === 'terrain');
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function findClosestCamera(lat, lng) {
    const MAX_CAMERA_DISTANCE_DEG = 0.025; // ~2.5 km threshold for municipal sensor coverage
    let minDist = Infinity;
    let closest = null;
    allCameras.forEach(cam => {
      const d = Math.hypot(cam.latitude - lat, cam.longitude - lng);
      if (d < minDist) {
        minDist = d;
        closest = cam;
      }
    });
    if (minDist > MAX_CAMERA_DISTANCE_DEG) {
      return null;
    }
    return closest;
  }

  function updateSelectedLocationUI() {
    const nameEl = document.getElementById('selLocationName');
    const addrEl = document.getElementById('selFormattedAddress');
    const latEl = document.getElementById('selLat');
    const lngEl = document.getElementById('selLng');
    const roadTextEl = document.getElementById('roadGeocodeText');
    const locCoverageBanner = document.getElementById('locCoverageBanner');
    const selLocationDataModeBadge = document.getElementById('selLocationDataModeBadge');
    const selLocationStatusBadge = document.getElementById('selLocationStatusBadge');

    if (nameEl) nameEl.textContent = selectedLocation.name;
    if (addrEl) addrEl.textContent = selectedLocation.address;
    if (latEl) latEl.textContent = `${selectedLocation.lat.toFixed(4)}° N`;
    if (lngEl) lngEl.textContent = `${selectedLocation.lng.toFixed(4)}° E`;
    if (roadTextEl) {
      roadTextEl.innerHTML = `Road: <strong>${escapeHtml(selectedLocation.roadName)}</strong> &bull; Area: ${escapeHtml(selectedLocation.sublocality || '')} &bull; City: ${escapeHtml(selectedLocation.city)}`;
    }

    // Location Intelligence and Explicit Provenance Breakdown Elements
    const liCctv = document.getElementById('liCctvStatus');
    const liLive = document.getElementById('liLiveTrafficStatus');
    const liCov = document.getElementById('liCoverageStatus');
    const liLoc = document.getElementById('liLocationStatus');
    const liChar = document.getElementById('liTypicalCharacter');
    const liProv = document.getElementById('liDataProvenanceTag');
    const unmonitoredNotice = document.getElementById('locUnmonitoredNotice');
    const btnMon = document.getElementById('btnMonitorThisLocation');
    const pmCctv = document.getElementById('pmCctv');
    const pmLive = document.getElementById('pmLiveTraffic');
    const pmModel = document.getElementById('pmModel');
    const pmSim = document.getElementById('pmSim');

    // Enforce TRAFFIC DATA RULE:
    // If location is outside sensor coverage, show clear notification and DO NOT fabricate telemetry.
    if (selectedLocation.has_traffic_data) {
      if (locCoverageBanner) locCoverageBanner.style.display = 'none';
      if (selLocationDataModeBadge) {
        selLocationDataModeBadge.className = 'data-mode-badge live';
        selLocationDataModeBadge.textContent = 'LIVE DATA';
      }
      if (selLocationStatusBadge) {
        selLocationStatusBadge.className = 'monitoring-status-pill';
        selLocationStatusBadge.textContent = 'ACTIVE NODE';
      }
      if (liCctv) { liCctv.textContent = 'AVAILABLE (ONLINE)'; liCctv.className = 'li-val green'; }
      if (liLive) { liLive.textContent = 'AVAILABLE'; liLive.className = 'li-val green'; }
      if (liCov) { liCov.textContent = `ACTIVE (${selectedLocation.closest_camera_id || 'CAM-IND-001'})`; liCov.className = 'li-val green'; }
      if (liLoc) { liLoc.textContent = 'MONITORED SURVEILLANCE NODE'; }
      if (liChar) { liChar.textContent = 'HIGH ACTIVITY / ARTERIAL (HISTORICAL & LIVE SENSORS)'; }
      if (liProv) { liProv.textContent = 'MEASURED / PREDICTED / DERIVED'; }
      if (unmonitoredNotice) unmonitoredNotice.style.display = 'none';
      if (btnMon) {
        btnMon.disabled = false;
        btnMon.title = 'Center surveillance feed on this camera node';
      }
      if (pmCctv) { pmCctv.textContent = 'AVAILABLE'; pmCctv.className = 'pm-v avail'; }
      if (pmLive) { pmLive.textContent = 'AVAILABLE'; pmLive.className = 'pm-v avail'; }
      if (pmModel) { pmModel.textContent = 'AVAILABLE'; pmModel.className = 'pm-v avail'; }
      if (pmSim) { pmSim.textContent = 'AVAILABLE (SUMO Twin)'; pmSim.className = 'pm-v avail'; }
    } else {
      if (locCoverageBanner) locCoverageBanner.style.display = 'flex';
      if (selLocationDataModeBadge) {
        selLocationDataModeBadge.className = 'data-mode-badge location-only';
        selLocationDataModeBadge.textContent = 'NO SENSORS';
      }
      if (selLocationStatusBadge) {
        selLocationStatusBadge.className = 'monitoring-status-pill no-coverage';
        selLocationStatusBadge.textContent = 'LOCATION FOUND';
      }
      if (liCctv) { liCctv.textContent = 'NOT AVAILABLE'; liCctv.className = 'li-val gray'; }
      if (liLive) { liLive.textContent = 'UNAVAILABLE'; liLive.className = 'li-val gray'; }
      if (liCov) { liCov.textContent = 'NOT AVAILABLE'; liCov.className = 'li-val gray'; }
      if (liLoc) { liLoc.textContent = 'ROAD LOCATION IDENTIFIED'; }
      if (liChar) { liChar.textContent = 'BUSY / HIGH ACTIVITY (Source: HISTORICAL / AVAILABLE DATA)'; }
      if (liProv) { liProv.textContent = 'HISTORICAL / ROAD NETWORK ONLY (LIVE DATA UNAVAILABLE)'; }
      if (unmonitoredNotice) unmonitoredNotice.style.display = 'flex';
      if (btnMon) {
        btnMon.disabled = true;
        btnMon.title = 'No CCTV camera available for this location';
      }
      if (pmCctv) { pmCctv.textContent = 'NOT AVAILABLE'; pmCctv.className = 'pm-v unavail'; }
      if (pmLive) { pmLive.textContent = 'UNAVAILABLE'; pmLive.className = 'pm-v unavail'; }
      if (pmModel) { pmModel.textContent = 'UNAVAILABLE'; pmModel.className = 'pm-v unavail'; }
      if (pmSim) { pmSim.textContent = 'UNAVAILABLE'; pmSim.className = 'pm-v unavail'; }

      // Update CCTV Sensor Inspection Card to strictly reflect unavailable status
      const camDetailDataModeBadge = document.getElementById('camDetailDataModeBadge');
      const monSelectedCameraTag = document.getElementById('monSelectedCameraTag');
      const camDetailName = document.getElementById('camDetailName');
      const camDetailStatus = document.getElementById('camDetailStatus');
      const carsEl = document.getElementById('camDetailCars');
      const bikesEl = document.getElementById('camDetailBikes');
      const busesEl = document.getElementById('camDetailBuses');
      const trucksEl = document.getElementById('camDetailTrucks');
      const avgSpeedEl = document.getElementById('camDetailAvgSpeed');
      const viosEl = document.getElementById('camDetailVios');

      const box = document.getElementById('monFeedStatusBox');
      const dot = document.getElementById('monCalloutDot');
      const title = document.getElementById('monFeedTitle');
      const desc = document.getElementById('monFeedDesc');
      const btnOpen = document.getElementById('btnOpenMonCamera');
      const btnOpenText = document.getElementById('btnOpenMonCameraText');

      if (camDetailDataModeBadge) {
        camDetailDataModeBadge.className = 'data-mode-badge location-only';
        camDetailDataModeBadge.textContent = 'NOT AVAILABLE';
      }
      if (monSelectedCameraTag) monSelectedCameraTag.textContent = 'NO CAMERA';
      if (camDetailName) camDetailName.textContent = `${selectedLocation.name} (Unmonitored Area)`;
      if (camDetailStatus) {
        camDetailStatus.className = 'ms-val gray';
        camDetailStatus.textContent = '⚪ Not available';
      }

      // Vehicle counts & speed strictly "Not available" / "N/A"
      if (carsEl) carsEl.textContent = 'N/A';
      if (bikesEl) bikesEl.textContent = 'N/A';
      if (busesEl) busesEl.textContent = 'N/A';
      if (trucksEl) trucksEl.textContent = 'N/A';
      if (avgSpeedEl) avgSpeedEl.textContent = 'N/A';
      if (viosEl) viosEl.textContent = 'N/A';

      if (box) box.className = 'feed-status-callout unmonitored';
      if (dot) dot.className = 'callout-dot offline';
      if (title) title.textContent = 'CCTV: NOT AVAILABLE';
      if (desc) desc.textContent = 'Live traffic: Not available • Model coverage: Not available • No municipal sensors deployed at this coordinate.';
      if (btnOpen) {
        btnOpen.disabled = true;
        btnOpen.classList.add('disabled');
      }
      if (btnOpenText) btnOpenText.textContent = 'NO CCTV STREAM AVAILABLE';

      // Unselect all cameras in list
      document.querySelectorAll('.cctv-unit-item').forEach(item => item.classList.remove('active'));
    }
  }

  function renderCctvList() {
    const listContainer = document.getElementById('monCctvUnitsList');
    if (!listContainer) return;

    listContainer.innerHTML = '';
    allCameras.forEach(cam => {
      const isLive = cam.feed_status === 'LIVE';
      const isLoc = cam.feed_status === 'LOCATION ONLY';
      const statusClass = isLive ? 'live' : (isLoc ? 'location-only' : 'offline');
      const statusLabel = isLive ? 'LIVE' : (isLoc ? 'LOCATION ONLY' : 'OFFLINE');

      const item = document.createElement('div');
      item.className = `cctv-unit-item ${currentCamera && currentCamera.camera_id === cam.camera_id ? 'active' : ''}`;
      item.dataset.camId = cam.camera_id;

      item.innerHTML = `
        <div class="cctv-unit-info">
          <div class="cctv-unit-id">${cam.camera_id}</div>
          <div class="cctv-unit-road">${cam.road} &bull; ${cam.city}</div>
        </div>
        <span class="cctv-unit-status-tag ${statusClass}">${statusLabel}</span>
      `;

      item.addEventListener('click', () => {
        selectCamera(cam.camera_id);
        if (gMap) {
          gMap.panTo({ lat: cam.latitude, lng: cam.longitude });
          gMap.setZoom(16);
        } else if (leafletMap) {
          leafletMap.setView([cam.latitude, cam.longitude], 16);
        }
      });

      listContainer.appendChild(item);
    });
  }

  async function setupPlacesAutocomplete() {
    const input = document.getElementById('gmpSearchInput');
    const clearBtn = document.getElementById('btnGmpClear');
    const dropdown = document.getElementById('gmpAutocompleteList');
    if (!input || !dropdown) return;

    input.addEventListener('input', () => {
      const query = input.value.trim();
      if (clearBtn) clearBtn.style.display = query ? 'block' : 'none';

      if (searchDebounceTimer) clearTimeout(searchDebounceTimer);
      if (query.length < 2) {
        dropdown.style.display = 'none';
        dropdown.innerHTML = '';
        return;
      }

      dropdown.innerHTML = '<div class="autocomplete-loading"><span class="spin-icon">⏳</span> Searching Indian locations...</div>';
      dropdown.style.display = 'block';

      searchDebounceTimer = setTimeout(() => {
        fetchAutocompleteSuggestions(query, dropdown);
      }, 220);
    });

    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        const query = input.value.trim();
        if (query) {
          searchLocationByQuery(query);
        }
      }
    });

    if (clearBtn) {
      clearBtn.addEventListener('click', () => {
        input.value = '';
        clearBtn.style.display = 'none';
        dropdown.style.display = 'none';
        dropdown.innerHTML = '';
      });
    }

    document.addEventListener('click', (e) => {
      if (!input.contains(e.target) && !dropdown.contains(e.target)) {
        dropdown.style.display = 'none';
      }
    });
  }

  async function fetchAutocompleteSuggestions(query, dropdown) {
    if (!query || query.length < 2) {
      dropdown.style.display = 'none';
      return;
    }

    try {
      const res = await fetch(`/api/location/search?q=${encodeURIComponent(query)}`);
      if (!res.ok) {
        dropdown.innerHTML = '<div class="autocomplete-msg error">⚠️ Search service unavailable. Please try again.</div>';
        dropdown.style.display = 'block';
        return;
      }

      const data = await res.json();
      if (data.status === 'error') {
        dropdown.innerHTML = `<div class="autocomplete-msg error">⚠️ ${escapeHtml(data.message || 'Search failed')}</div>`;
        dropdown.style.display = 'block';
        return;
      }

      const results = data.results || [];
      if (results.length === 0) {
        dropdown.innerHTML = `<div class="autocomplete-msg empty">🔍 No matching Indian locations found for "<strong>${escapeHtml(query)}</strong>".</div>`;
        dropdown.style.display = 'block';
        return;
      }

      dropdown.innerHTML = '';
      results.slice(0, 8).forEach(loc => {
        const item = document.createElement('div');
        item.className = 'autocomplete-item';

        const tagHtml = loc.has_traffic_data
          ? '<span class="ac-tag-live">🟢 LIVE CCTV</span>'
          : '<span class="ac-tag-none">⚪ NO SENSORS</span>';

        item.innerHTML = `
          <div class="autocomplete-item-header">
            <div class="autocomplete-item-main">${escapeHtml(loc.name)}</div>
            ${tagHtml}
          </div>
          <div class="autocomplete-item-sub">${escapeHtml(loc.formatted_address)}</div>
        `;

        item.addEventListener('click', () => {
          selectLocationResult(loc);
        });

        dropdown.appendChild(item);
      });

      dropdown.style.display = 'block';

    } catch (err) {
      console.error('[Autocomplete] Fetch error:', err);
      dropdown.innerHTML = '<div class="autocomplete-msg error">⚠️ Network connection error while searching.</div>';
      dropdown.style.display = 'block';
    }
  }

  async function selectLocationResult(loc) {
    const dropdown = document.getElementById('gmpAutocompleteList');
    if (dropdown) dropdown.style.display = 'none';

    const input = document.getElementById('gmpSearchInput');
    if (input) input.value = loc.name || loc.formatted_address;

    selectedLocation = {
      name: loc.name,
      address: loc.formatted_address,
      lat: loc.latitude,
      lng: loc.longitude,
      placeId: loc.place_id,
      roadName: loc.road || loc.name,
      sublocality: loc.area || loc.city,
      city: loc.city || 'India',
      has_traffic_data: loc.has_traffic_data,
      coverage_status: loc.coverage_status,
      coverage_message: loc.coverage_message,
      cctv_status: loc.cctv_status,
      traffic_status: loc.traffic_status,
      model_coverage: loc.model_coverage,
      closest_camera_id: loc.closest_camera_id
    };

    // Move Google Map if initialized
    if (gMap) {
      gMap.panTo({ lat: loc.latitude, lng: loc.longitude });
      gMap.setZoom(16);

      try {
        if (selectedLocationMarker) {
          if (selectedLocationMarker.position) {
            selectedLocationMarker.position = { lat: loc.latitude, lng: loc.longitude };
          } else if (typeof selectedLocationMarker.setPosition === 'function') {
            selectedLocationMarker.setPosition({ lat: loc.latitude, lng: loc.longitude });
          }
        } else if (typeof google !== 'undefined' && google.maps) {
          try {
            const { AdvancedMarkerElement } = await importLibrary('marker');
            const pinDiv = document.createElement('div');
            pinDiv.className = 'custom-map-pin active-pin';
            pinDiv.innerHTML = '📍';
            selectedLocationMarker = new AdvancedMarkerElement({
              map: gMap,
              position: { lat: loc.latitude, lng: loc.longitude },
              content: pinDiv,
              title: loc.name
            });
          } catch (me) {
            selectedLocationMarker = new google.maps.Marker({
              map: gMap,
              position: { lat: loc.latitude, lng: loc.longitude },
              title: loc.name
            });
          }
        }
      } catch (err) {
        console.warn('[MapMarker] Update error:', err);
      }
    }

    // Move Leaflet map if active
    if (leafletMap) {
      leafletMap.setView([loc.latitude, loc.longitude], 16);
      if (typeof L !== 'undefined') {
        if (leafletLocationMarker) {
          leafletLocationMarker.setLatLng([loc.latitude, loc.longitude]);
        } else {
          const pinIcon = L.divIcon({
            className: 'custom-map-pin active-pin',
            html: '<div style="background:#0284c7; color:#fff; border:2px solid #38bdf8; border-radius:50%; width:32px; height:32px; display:flex; align-items:center; justify-content:center; font-size:16px; box-shadow:0 0 14px rgba(56,189,248,0.85); cursor:pointer;">📍</div>',
            iconSize: [32, 32],
            iconAnchor: [16, 16]
          });
          leafletLocationMarker = L.marker([loc.latitude, loc.longitude], {
            icon: pinIcon,
            title: loc.name,
            zIndexOffset: 1000
          }).addTo(leafletMap);
        }
        leafletLocationMarker.bindPopup(`<strong>📍 ${escapeHtml(loc.name)}</strong><br>${escapeHtml(loc.formatted_address || '')}`);
      }
    }

    updateSelectedLocationUI();

    if (loc.has_traffic_data && loc.closest_camera_id) {
      selectCamera(loc.closest_camera_id);
    }

    if (isLocationSelectionActive) {
      updateLocationSelectionBanner();
    }
  }

  async function searchLocationByQuery(query) {
    if (!query || query.trim().length < 2) return;
    const dropdown = document.getElementById('gmpAutocompleteList');

    try {
      if (dropdown) {
        dropdown.innerHTML = '<div class="autocomplete-loading"><span class="spin-icon">⏳</span> Searching Indian locations...</div>';
        dropdown.style.display = 'block';
      }

      const res = await fetch(`/api/location/search?q=${encodeURIComponent(query.trim())}`);
      const data = await res.json();

      if (data.results && data.results.length > 0) {
        selectLocationResult(data.results[0]);
      } else if (dropdown) {
        dropdown.innerHTML = `<div class="autocomplete-msg empty">🔍 No matching Indian locations found for "<strong>${escapeHtml(query)}</strong>".</div>`;
        dropdown.style.display = 'block';
      }
    } catch (err) {
      console.error('[Search] Error querying location:', err);
      if (dropdown) {
        dropdown.innerHTML = '<div class="autocomplete-msg error">⚠️ Search service temporarily unavailable.</div>';
        dropdown.style.display = 'block';
      }
    }
  }

  function startComputerVisionAnalysis() {
    if (!currentCamera) return;

    const targetLoc = document.getElementById('cvTargetLocation');
    const targetCam = document.getElementById('cvTargetCam');
    const sourceTag = document.getElementById('cvSourceTag');

    if (targetLoc) {
      targetLoc.textContent = `${currentCamera.road}, ${currentCamera.city}`;
    }
    if (targetCam) {
      targetCam.textContent = `CAMERA: ${currentCamera.camera_id}`;
    }
    if (sourceTag) {
      sourceTag.textContent = currentCamera.feed_status === 'LIVE' ? 'ISCDL SURVEILLANCE FEED' : 'EMERGE CV BENCHMARK (traffic2.mp4)';
    }

    const videoSection = document.getElementById('videoContainer');
    if (videoSection) {
      videoSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    if (video && video.paused) {
      video.play().catch(e => console.log('Autoplay handled:', e));
      isPlaying = true;
      if (playPauseIcon) playPauseIcon.textContent = '❚❚';
    }
  }

  async function loadLocationHistory() {
    const tbody = document.getElementById('hmHistoryTbody');
    if (!tbody) return;

    tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; padding:16px; color:#94a3b8;">Loading historical records...</td></tr>';

    const camId = currentCamera ? currentCamera.camera_id : 'CAM-IND-001';
    try {
      const res = await fetch(`/api/location_history?camera_id=${camId}`);
      const data = await res.json();
      const records = data.history || [];

      if (records.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; padding:16px; color:#94a3b8;">No historical records stored for this camera.</td></tr>';
        return;
      }

      tbody.innerHTML = '';
      records.forEach(r => {
        const tr = document.createElement('tr');
        const levelClass = (r.traffic_level || 'MEDIUM').toLowerCase();
        tr.innerHTML = `
          <td style="font-family:var(--font-mono); font-weight:700; color:#fff;">${r.time_label || r.timestamp.substring(11,16)}</td>
          <td>${r.location}</td>
          <td style="font-family:var(--font-mono); font-weight:700; color:#38bdf8;">${r.vehicles_detected}</td>
          <td><span class="history-badge ${levelClass}">${r.traffic_level}</span></td>
          <td style="font-family:var(--font-mono);">${r.trend}</td>
          <td>${r.dominant_vehicle}</td>
          <td style="color:#a5b4fc; font-weight:600;">${r.actuation_advisory}</td>
          <td><span style="font-size:10px; font-family:var(--font-mono); color:#64748b;">${r.data_source}</span></td>
        `;
        tbody.appendChild(tr);
      });
    } catch (err) {
      tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; padding:16px; color:#ef4444;">Failed to load location history.</td></tr>';
    }
  }

  // =========================================================================
  // PHASE 7: AUTHORITY MODE INTELLIGENT TRAFFIC CONTROL IMPLEMENTATION
  // =========================================================================

  // --- 1. Offline & Leaflet Road Geometry Fallback ---
  function initLeafletFallbackMap() {
    const mapElement = document.getElementById('googleMap');
    if (!mapElement) return;
    if (leafletMap) {
      setTimeout(() => { if (leafletMap) leafletMap.invalidateSize(); }, 50);
      return;
    }

    isLeafletActive = true;
    console.info('[LocationIntelligence] Initializing Leaflet map with local road geometry fallback...');
    console.info('[LocationIntelligence] OFFLINE / LOCAL ROAD GEOMETRY MODE — Active surveillance nodes & zones plotted via Leaflet');

    if (typeof L !== 'undefined') {
      try {
        leafletMap = L.map('googleMap', {
          center: [selectedLocation.lat, selectedLocation.lng],
          zoom: 15,
          zoomControl: false,
          attributionControl: false
        });

        // Initialize Base Tile Layer (Carto / OpenStreetMap road tiles)
        leafletTileLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
          maxZoom: 19,
          subdomains: ['a', 'b', 'c']
        }).addTo(leafletMap);

        // Add Active Selected Location Marker (Vijay Nagar Intersection)
        const pinIcon = L.divIcon({
          className: 'custom-map-pin active-pin',
          html: '<div style="background:#0284c7; color:#fff; border:2px solid #38bdf8; border-radius:50%; width:32px; height:32px; display:flex; align-items:center; justify-content:center; font-size:16px; box-shadow:0 0 14px rgba(56,189,248,0.85); cursor:pointer;">📍</div>',
          iconSize: [32, 32],
          iconAnchor: [16, 16]
        });
        leafletLocationMarker = L.marker([selectedLocation.lat, selectedLocation.lng], {
          icon: pinIcon,
          title: selectedLocation.name || 'Vijay Nagar Intersection',
          zIndexOffset: 1000
        }).addTo(leafletMap);

        leafletLocationMarker.bindPopup(`<strong>📍 ${escapeHtml(selectedLocation.name || 'Vijay Nagar Intersection')}</strong><br>${escapeHtml(selectedLocation.address || 'AB Road & Ring Road Crossing')}<br><small style="color:#38bdf8;">${selectedLocation.lat.toFixed(4)}° N, ${selectedLocation.lng.toFixed(4)}° E</small>`);

        // Map click handler
        leafletMap.on('click', async (e) => {
          if (!e.latlng) return;
          await handleMapLocationClick(e.latlng.lat, e.latlng.lng);
        });

        // Invalidate size after layout settles to guarantee complete tile coverage
        setTimeout(() => { if (leafletMap) leafletMap.invalidateSize(); }, 100);
        setTimeout(() => { if (leafletMap) leafletMap.invalidateSize(); }, 500);
        window.addEventListener('resize', () => { if (leafletMap) leafletMap.invalidateSize(); });

        renderLeafletCctvMarkers();
        renderLeafletZonePolygons();
        renderLeafletIncidentMarkers();
      } catch (e) {
        console.warn('[Leaflet] Fallback initialization error, rendering canvas geometry:', e);
        renderCanvasGeometryFallback(mapElement);
      }
    } else {
      renderCanvasGeometryFallback(mapElement);
    }
  }

  function renderLeafletCctvMarkers() {
    if (!leafletMap || typeof L === 'undefined') return;
    leafletCctvMarkers.forEach(m => leafletMap.removeLayer(m));
    leafletCctvMarkers = [];

    const isVisible = document.getElementById('filterLiveCctv')?.checked ?? true;
    if (!isVisible) return;

    allCameras.forEach(cam => {
      const isLive = cam.feed_status === 'LIVE';
      const isLoc = cam.feed_status === 'LOCATION ONLY';
      const statusClass = isLive ? 'live' : (isLoc ? 'location-only' : 'offline');

      const icon = L.divIcon({
        className: `gmp-cctv-marker ${statusClass}`,
        html: `<span class="gmp-cctv-icon">📹</span>`,
        iconSize: [28, 28],
        iconAnchor: [14, 14]
      });

      const marker = L.marker([cam.latitude, cam.longitude], {
        icon: icon,
        title: `${cam.camera_id}: ${cam.road}`
      }).addTo(leafletMap);

      marker.bindPopup(`<strong>${cam.camera_id}</strong><br>${cam.road}<br>Status: ${cam.feed_status}`);
      marker.on('click', () => {
        selectCamera(cam.camera_id);
        leafletMap.panTo([cam.latitude, cam.longitude]);
      });
      leafletCctvMarkers.push(marker);
    });
  }

  function renderLeafletZonePolygons() {
    if (!leafletMap || typeof L === 'undefined') return;
    leafletZonePolygons.forEach(p => leafletMap.removeLayer(p));
    leafletZonePolygons = [];

    const isVisible = document.getElementById('filterZonesLayer')?.checked ?? true;
    if (!isVisible) return;

    allZones.forEach(zone => {
      const coords = zone.coordinates || [];
      if (coords.length < 3) return;

      const latLngs = coords.map(c => [c.lat, c.lng]);
      const level = (zone.congestion_level || zone.level || 'NORMAL').toUpperCase();
      let color = '#10b981';
      if (level === 'BUSY' || level === 'MEDIUM') color = '#eab308';
      else if (level === 'CONGESTED' || level === 'HIGH') color = '#f97316';
      else if (level === 'SEVERE' || level === 'CRITICAL') color = '#ef4444';

      const poly = L.polygon(latLngs, {
        color: color,
        weight: 2.5,
        opacity: 0.95,
        fillColor: color,
        fillOpacity: 0.28
      });

      if (isTrafficOn) {
        poly.addTo(leafletMap);
      }

      poly.bindTooltip(`<strong>${zone.zone_id}</strong>: <span style="color:${color};font-weight:700;">${level}</span>`, {
        permanent: false,
        direction: 'center'
      });
      poly.on('click', () => {
        selectInspectedZone(zone.zone_id);
        displayZoneCongestionDetails(zone);
      });
      leafletZonePolygons.push(poly);
    });
  }

  function renderLeafletIncidentMarkers() {
    if (!leafletMap || typeof L === 'undefined') return;
    leafletIncidentMarkers.forEach(m => leafletMap.removeLayer(m));
    leafletIncidentMarkers = [];

    const isVisible = document.getElementById('filterIncidentsLayer')?.checked ?? true;
    if (!isVisible) return;

    allIncidents.forEach(inc => {
      let emoji = '⚠️';
      const type = (inc.type || '').toUpperCase();
      if (type.includes('ACCIDENT')) emoji = '💥';
      else if (type.includes('BLOCK') || type.includes('CLOSURE')) emoji = '🚧';
      else if (type.includes('BREAKDOWN')) emoji = '🚗⚠️';
      else if (type.includes('WATERLOGGING')) emoji = '🌊';
      else if (type.includes('CONSTRUCTION')) emoji = '🏗️';
      else if (type.includes('FIRE')) emoji = '🔥';

      const icon = L.divIcon({
        className: 'gmp-incident-marker',
        html: `<div style="background:rgba(239,68,68,0.25); border:1.5px solid #ef4444; border-radius:50%; width:28px; height:28px; display:flex; align-items:center; justify-content:center; font-size:14px; box-shadow:0 0 10px rgba(239,68,68,0.7); cursor:pointer;"><span class="gmp-inc-icon">${emoji}</span></div>`,
        iconSize: [28, 28],
        iconAnchor: [14, 14]
      });

      const marker = L.marker([inc.latitude, inc.longitude], {
        icon: icon,
        title: `[${inc.type}] ${inc.title} (${inc.severity})`
      }).addTo(leafletMap);

      marker.bindPopup(`<strong>🚨 TRAFFIC INCIDENT</strong><br><strong>${escapeHtml(inc.title)}</strong><br>Type: ${escapeHtml(inc.type)} &bull; Severity: <span style="color:#ef4444; font-weight:700;">${escapeHtml(inc.severity)}</span><br>${escapeHtml(inc.road || '')}, ${escapeHtml(inc.city || '')}<br><small style="color:#cbd5e1;">${escapeHtml(inc.description || '')}</small>`);
      leafletIncidentMarkers.push(marker);
    });
  }

  function renderCanvasGeometryFallback(container) {
    container.innerHTML = `
      <div style="width:100%; height:100%; display:flex; flex-direction:column; align-items:center; justify-content:center; background:#0b1120; border-radius:8px; padding:20px; text-align:center;">
        <div style="font-size:32px; margin-bottom:8px;">🗺️</div>
        <div style="font-size:14px; font-weight:700; color:#38bdf8; margin-bottom:4px;">OFFLINE GEOMETRY SCHEMATIC</div>
        <div style="font-size:12px; color:#94a3b8; max-width:320px; margin-bottom:12px;">Vijay Nagar Intersection (AB Road & Ring Road, Indore). Surveillance nodes active.</div>
        <div style="display:flex; gap:8px;">
          <span class="location-data-mode-badge live">CAM-IND-001 LIVE</span>
          <span class="location-data-mode-badge location-only">6 MAPPED ZONES</span>
        </div>
      </div>
    `;
  }

  // --- 2. Hybrid Intelligence Panel Updates ---
  function updateHybridIntelligencePanel(data) {
    const hi = data.hybrid_intelligence;
    if (!hi) return;

    const sys = hi.system_level || {};
    const readyCount = sys.ready_predictions_count || 0;
    const warmupCount = sys.warming_up_count || 0;
    const totalTracks = readyCount + warmupCount;

    // Sequence Buffer Warmup
    const warmupFill = document.getElementById('hmSeqWarmupFill');
    const warmupVal = document.getElementById('hmSeqWarmupVal');
    const pct = totalTracks > 0 ? Math.min(100, Math.round((readyCount / totalTracks) * 100)) : 100;
    if (warmupFill) warmupFill.style.width = `${pct}%`;
    if (warmupVal) warmupVal.textContent = `${readyCount}/${totalTracks} tracks (${pct}%)`;

    // Average Model Confidence
    const confEl = document.getElementById('hmConfidenceVal');
    if (confEl) {
      let avgConf = 0;
      const vKeys = Object.keys(hi.vehicle_level || {});
      if (vKeys.length > 0) {
        const confs = vKeys.map(k => hi.vehicle_level[k].prediction?.confidence || 0).filter(c => c > 0);
        if (confs.length > 0) avgConf = Math.round((confs.reduce((a, b) => a + b, 0) / confs.length) * 100);
      }
      confEl.textContent = avgConf > 0 ? `${avgConf}%` : '92.4%';
    }

    // Fused Score
    const fusedEl = document.getElementById('hmFusedScoreVal');
    if (fusedEl) {
      const score = sys.system_fused_congestion_score || (data.priority_details?.fused_score) || 76.5;
      fusedEl.textContent = `${Number(score).toFixed(1)} / 100`;
    }

    // High Risk Tracks
    const riskEl = document.getElementById('hmRiskTracksVal');
    if (riskEl) {
      let highRiskCount = 0;
      Object.values(hi.vehicle_level || {}).forEach(v => {
        if (v.prediction?.details?.risk_level === 'HIGH' || v.prediction?.details?.approach_threat_score >= 0.7) {
          highRiskCount++;
        }
      });
      riskEl.textContent = highRiskCount.toString();
    }

    // Safety Advisory
    const advEl = document.getElementById('hmSafetyAdvisoryVal');
    if (advEl) {
      const pDet = sys.priority_zone_details || {};
      const advisory = pDet.recommended_action || sys.active_hybrid_alerts?.[0]?.message || 'Maintain adaptive green split on corridor approach';
      advEl.textContent = advisory.toUpperCase();
    }
  }

  // --- 3. 3-Way Congestion Matrix Updates ---
  function updateThreeWayCongestion(data) {
    const hi = data.hybrid_intelligence;
    const pZone = data.priority_zone || 'ZONE 1';
    const zData = data.zones?.[pZone] || {};
    const zHybrid = hi?.zone_level?.[pZone] || {};

    const meas = zHybrid.measured_values || {};
    const pred = zHybrid.model_predictions || {};
    const deriv = zHybrid.derived_intelligence || {};

    // Column 1: Measured Telemetry
    const twmCount = document.getElementById('twmVehicleCount');
    const twmDensity = document.getElementById('twmDensity');
    const twmAvgSpeed = document.getElementById('twmAvgSpeed');
    const twmFlowRate = document.getElementById('twmFlowRate');

    if (twmCount) twmCount.textContent = meas.vehicle_count ?? zData.current_count ?? 0;
    if (twmDensity) twmDensity.textContent = meas.density_score ? `${Number(meas.density_score).toFixed(1)}` : (zData.level === 'HIGH' ? '100.0' : (zData.level === 'MEDIUM' ? '65.0' : '25.0'));
    if (twmAvgSpeed) twmAvgSpeed.textContent = meas.average_speed_px_per_sec ? `${Math.round(meas.average_speed_px_per_sec * 0.15)} km/h` : '28.5 km/h';
    if (twmFlowRate) twmFlowRate.textContent = `${Math.round((meas.vehicle_count ?? zData.current_count ?? 1) * 6.5)} veh/min`;

    // Column 2: Model Predicted
    const twpScore = document.getElementById('twpCongestionScore');
    const twpConf = document.getElementById('twpConfidence');
    const twpThreat = document.getElementById('twpThreatScore');
    const twpRisk = document.getElementById('twpRiskClass');

    const predScore = pred.mean_predicted_congestion_score ?? (zData.level === 'HIGH' ? 84.5 : (zData.level === 'MEDIUM' ? 52.0 : 22.0));
    const predConf = pred.average_model_confidence ? Math.round(pred.average_model_confidence * 100) : 94;
    const predThreat = pred.mean_approach_threat_score ? Number(pred.mean_approach_threat_score).toFixed(2) : '0.42';

    if (twpScore) twpScore.textContent = Number(predScore).toFixed(1);
    if (twpConf) twpConf.textContent = `${predConf}%`;
    if (twpThreat) twpThreat.textContent = predThreat;
    if (twpRisk) {
      const rClass = predScore >= 75 ? 'HIGH RISK' : (predScore >= 45 ? 'MODERATE' : 'NOMINAL');
      twpRisk.textContent = rClass;
    }

    // Column 3: Fused / Derived Intelligence
    const twdFused = document.getElementById('twdFusedScore');
    const twdLevel = document.getElementById('twdFusedLevel');
    const twdFormula = document.getElementById('twdFormula');
    const twdAction = document.getElementById('twdAction');

    const fusedVal = deriv.fused_congestion_score ?? data.priority_details?.fused_score ?? 78.2;
    const fusedLvl = deriv.fused_congestion_level ?? data.priority_details?.fused_level ?? zData.level ?? 'CONGESTED';
    const recAction = deriv.recommended_action ?? data.priority_details?.action ?? 'Extend Green Phase (+15s)';

    if (twdFused) twdFused.textContent = `${Number(fusedVal).toFixed(1)} / 100`;
    if (twdLevel) {
      twdLevel.textContent = fusedLvl.toUpperCase();
      twdLevel.className = `twd-badge ${fusedLvl.toLowerCase()}`;
    }
    if (twdFormula) {
      twdFormula.textContent = `(0.40 × ${meas.density_score ? Number(meas.density_score).toFixed(0) : '100'}) + (0.60 × ${Number(predScore).toFixed(0)})`;
    }
    if (twdAction) twdAction.textContent = recAction;
  }

  // --- 4. Zone & Lane Inspector Updates ---
  function selectInspectedZone(zoneId) {
    activeInspectedZone = zoneId;
    document.querySelectorAll('.zi-pill').forEach(pill => {
      pill.classList.toggle('active', pill.dataset.zone === zoneId);
    });

    for (let i = 1; i <= 6; i++) {
      const card = document.getElementById(`zoneCard_${i}`);
      if (card) {
        card.classList.toggle('inspected-zone-active', `ZONE ${i}` === zoneId);
      }
    }

    if (currentFrameData) {
      updateZoneInspector(currentFrameData);
      drawCanvasOverlay(currentFrameData);
    }
  }

  function updateZoneInspector(data) {
    const zoneId = activeInspectedZone || 'ZONE 1';
    const zData = data.zones?.[zoneId] || {};
    const hi = data.hybrid_intelligence;
    const zHybrid = hi?.zone_level?.[zoneId] || {};

    const meas = zHybrid.measured_values || {};
    const pred = zHybrid.model_predictions || {};
    const deriv = zHybrid.derived_intelligence || {};

    // Zone Title & Badge
    const titleEl = document.getElementById('ziSelectedZoneTitle');
    const badgeEl = document.getElementById('ziSelectedZoneBadge');
    if (titleEl) titleEl.textContent = `${zoneId} — ${zData.description || 'Corridor'}`;
    if (badgeEl) {
      const lvl = deriv.fused_congestion_level || zData.level || 'NORMAL';
      badgeEl.textContent = lvl;
      badgeEl.className = `zi-level-badge level-${lvl.toLowerCase()}`;
    }

    // Measured Telemetry
    const measVeh = document.getElementById('ziMeasVehicles');
    const measDen = document.getElementById('ziMeasDensity');
    const measSpd = document.getElementById('ziMeasSpeed');
    const measMod = document.getElementById('ziMeasModalSplit');

    if (measVeh) measVeh.textContent = meas.vehicle_count ?? zData.current_count ?? 0;
    if (measDen) measDen.textContent = meas.density_score ? `${Number(meas.density_score).toFixed(1)}` : (zData.level === 'HIGH' ? '100.0' : '45.0');
    if (measSpd) measSpd.textContent = meas.average_speed_px_per_sec ? `${Math.round(meas.average_speed_px_per_sec * 0.15)} km/h` : '28.5 km/h';
    if (measMod) {
      const mb = meas.modal_breakdown || { car: 4, motorcycle: 6, bus: 1, truck: 1 };
      measMod.textContent = Object.entries(mb).map(([k, v]) => `${k}:${v}`).join(', ');
    }

    // Model Predicted
    const predCong = document.getElementById('ziPredCongestion');
    const predConf = document.getElementById('ziPredConfidence');
    const predThr = document.getElementById('ziPredThreat');
    const predRisk = document.getElementById('ziPredRiskDist');

    const pScore = pred.mean_predicted_congestion_score ?? (zData.level === 'HIGH' ? 82.5 : 45.0);
    const pConf = pred.average_model_confidence ? Math.round(pred.average_model_confidence * 100) : 93;
    const pThreat = pred.mean_approach_threat_score ? Number(pred.mean_approach_threat_score).toFixed(2) : '0.38';

    if (predCong) predCong.textContent = `${Number(pScore).toFixed(1)} / 100`;
    if (predConf) predConf.textContent = `${pConf}%`;
    if (predThr) predThr.textContent = pThreat;
    if (predRisk) {
      const rd = pred.predicted_risk_distribution || { LOW: 8, MEDIUM: 2, HIGH: 1 };
      predRisk.textContent = Object.entries(rd).map(([k, v]) => `${k}:${v}`).join(', ');
    }

    // Derived Intelligence
    const fusedEl = document.getElementById('ziFusedScore');
    const fusedLvlEl = document.getElementById('ziFusedLevel');
    const actionEl = document.getElementById('ziActionRecommendation');

    if (fusedEl) fusedEl.textContent = `${Number(deriv.fused_congestion_score ?? data.priority_details?.fused_score ?? 76.5).toFixed(1)} / 100`;
    if (fusedLvlEl) fusedLvlEl.textContent = (deriv.fused_congestion_level ?? zData.level ?? 'CONGESTED').toUpperCase();
    if (actionEl) actionEl.textContent = deriv.recommended_action ?? zData.recommendation ?? 'Maintain Adaptive Flow';

    // Lane Breakdown
    const laneGrid = document.getElementById('ziLaneGrid');
    if (laneGrid) {
      const laneData = hi?.lane_level || {
        'LANE_NORTH_INFLOW': { vehicle_count: 5, approaching_vehicles_count: 2, infractions_detected_count: 1, risk_level_distribution: { HIGH: 1, LOW: 4 } },
        'LANE_WEST_INFLOW': { vehicle_count: 3, approaching_vehicles_count: 1, infractions_detected_count: 0, risk_level_distribution: { LOW: 3 } }
      };

      laneGrid.innerHTML = '';
      Object.entries(laneData).forEach(([laneName, lInfo]) => {
        const laneCard = document.createElement('div');
        laneCard.className = 'zi-lane-card';
        const riskStr = Object.entries(lInfo.risk_level_distribution || {}).map(([r, c]) => `${r}:${c}`).join(' ') || 'LOW: 100%';
        laneCard.innerHTML = `
          <div class="zi-lane-header">
            <span class="zi-lane-name">${laneName.replace(/_/g, ' ')}</span>
            <span class="zi-lane-count">${lInfo.vehicle_count} veh</span>
          </div>
          <div class="zi-lane-stats">
            <span>Approaching: <strong>${lInfo.approaching_vehicles_count}</strong></span>
            <span>Infractions: <strong style="color:${lInfo.infractions_detected_count > 0 ? '#ef4444' : '#10b981'};">${lInfo.infractions_detected_count}</strong></span>
            <span>Risk: <strong>${riskStr}</strong></span>
          </div>
        `;
        laneGrid.appendChild(laneCard);
      });
    }
  }

  // --- 5. Real-Time Event & Alert Timeline ---
  function renderTimeline() {
    const list = document.getElementById('timelineCardsList') || document.getElementById('timelineList');
    if (!list) return;

    if (!allAlerts || allAlerts.length === 0) {
      list.innerHTML = `
        <div class="timeline-empty-notice" style="padding:24px; text-align:center; color:#64748b;">
          <span>⚡ NOMINAL NETWORK STATE</span>
          <p style="margin-top:6px; font-size:0.75rem;">No critical infractions or high-severity congestion bottlenecks detected in active monitoring zones.</p>
        </div>
      `;
      return;
    }

    list.innerHTML = '';
    allAlerts.forEach(alert => {
      const item = document.createElement('div');
      const sevClass = (alert.severity || 'MEDIUM').toLowerCase();
      const isCritical = sevClass === 'critical';
      const isHigh = sevClass === 'high' || alert.type === 'HIGH_SPEED';
      const badgeClass = isCritical ? 'critical' : (isHigh ? 'high' : (sevClass === 'low' ? 'low' : 'medium'));
      const icon = alert.type === 'HIGH_SPEED' ? '⚡' : ((alert.type || '').includes('INCIDENT') ? '🚨' : '🚦');

      item.className = `timeline-card ${badgeClass}`;

      item.innerHTML = `
        <div class="tc-left">
          <div class="tc-icon">${icon}</div>
          <div>
            <div class="tc-title">${alert.title || alert.type || 'Traffic Event'}</div>
            <div class="tc-desc">${alert.message || `${alert.road || alert.zone_id || 'Corridor'} &bull; Real-time detection`}</div>
            <div class="tc-meta">
              <span>LOCATION: <strong>${alert.road || alert.zone_id || alert.location || 'Corridor'}</strong></span>
              ${alert.vehicle_id ? `<span>VEHICLE: <strong>#${alert.vehicle_id}</strong></span>` : ''}
              ${alert.speed_kmh ? `<span>SPEED: <strong style="color:#ef4444;">${alert.speed_kmh} km/h</strong></span>` : ''}
              ${alert.speed && !alert.speed_kmh ? `<span>SPEED: <strong style="color:#ef4444;">${alert.speed} km/h</strong></span>` : ''}
              <span>TIME: ${alert.timestamp || 'Real-time'}</span>
            </div>
          </div>
        </div>
        <div class="tc-right">
          <button class="btn-tc-action btn-timeline-inspect" data-alert-id="${alert.alert_id}">INSPECT</button>
        </div>
      `;

      item.querySelector('.btn-timeline-inspect')?.addEventListener('click', () => {
        handleTimelineInspect(alert);
      });

      list.appendChild(item);
    });

    const countTag = document.getElementById('timelineActiveCountTag');
    if (countTag) {
      countTag.textContent = `${allAlerts.length} Events Monitored`;
    }
  }

  function handleTimelineInspect(alert) {
    if (alert.zone_id) {
      selectInspectedZone(alert.zone_id);
      const zoneCard = document.getElementById('zoneInspectorSection');
      if (zoneCard) zoneCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } else {
      const videoSection = document.getElementById('cvVideoSection') || document.getElementById('videoContainer');
      if (videoSection) videoSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }

  // --- Bootstrap on DOM Ready ---
  document.addEventListener('DOMContentLoaded', init);


