import init, { WasmDetector } from "./pkg/agua_web.js";

const PROCESS_INTERVAL_MS = 100;
const MAX_SAMPLES_PER_TICK = 12800; // ~267ms at 48kHz — keeps each process() fast
const DEFAULT_KEY = "agua-demo";

let detector = null;
let audioContext = null;
let workletNode = null;
let sampleChunks = [];
let sampleChunksLength = 0;
let processTimer = null;
let listening = false;

// DOM elements
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const statusEl = document.getElementById("status");
const statusDot = document.getElementById("status-dot");
const payloadEl = document.getElementById("payload");
const confidenceEl = document.getElementById("confidence");
const confidenceBar = document.getElementById("confidence-bar");
const bufferBar = document.getElementById("buffer-bar");
const bufferText = document.getElementById("buffer-text");
const sampleRateEl = document.getElementById("sample-rate");
const keyInput = document.getElementById("key-input");
const deviceSelect = document.getElementById("device-select");
const detectionLog = document.getElementById("detection-log");

function setStatus(text, state) {
  statusEl.textContent = text;
  statusDot.className = "status-dot " + state;
}

function addDetection(payload, confidence) {
  const time = new Date().toLocaleTimeString();
  const entry = document.createElement("div");
  entry.className = "log-entry";
  entry.innerHTML =
    `<span class="log-time">${time}</span> ` +
    `<code class="log-payload">${payload}</code> ` +
    `<span class="log-confidence">(${(confidence * 100).toFixed(1)}%)</span>`;
  detectionLog.prepend(entry);

  // Keep last 50 entries
  while (detectionLog.children.length > 50) {
    detectionLog.removeChild(detectionLog.lastChild);
  }
}

function processSamples() {
  if (!detector || sampleChunksLength === 0) return;

  // Drain at most MAX_SAMPLES_PER_TICK to keep the main thread responsive.
  let toProcess = 0;
  let chunksToTake = 0;
  for (let i = 0; i < sampleChunks.length; i++) {
    if (toProcess + sampleChunks[i].length > MAX_SAMPLES_PER_TICK && toProcess > 0) break;
    toProcess += sampleChunks[i].length;
    chunksToTake = i + 1;
  }

  const samples = new Float32Array(toProcess);
  let offset = 0;
  for (let i = 0; i < chunksToTake; i++) {
    samples.set(sampleChunks[i], offset);
    offset += sampleChunks[i].length;
  }
  sampleChunks = sampleChunks.slice(chunksToTake);
  sampleChunksLength -= toProcess;

  const result = detector.process(samples);

  // Update buffer fill
  const fill = detector.get_buffer_fill();
  bufferBar.style.width = (fill * 100).toFixed(1) + "%";
  bufferText.textContent = (fill * 100).toFixed(0) + "%";

  if (result) {
    payloadEl.textContent = result;
    const confidence = detector.get_confidence();
    confidenceEl.textContent = (confidence * 100).toFixed(1) + "%";
    confidenceBar.style.width = Math.min(confidence * 100, 100).toFixed(1) + "%";
    setStatus("Watermark detected", "detected");
    addDetection(result, confidence);

    // Return to listening state after a moment
    setTimeout(() => {
      if (listening) setStatus("Listening...", "listening");
    }, 2000);
  }
}

async function start() {
  try {
    setStatus("Initializing...", "idle");

    await init();

    const key = keyInput.value.trim() || DEFAULT_KEY;
    const audioConstraints = {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    };
    const selectedDevice = deviceSelect.value;
    if (selectedDevice) {
      audioConstraints.deviceId = { exact: selectedDevice };
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: audioConstraints,
    });

    audioContext = new AudioContext();
    const sampleRate = audioContext.sampleRate;
    sampleRateEl.textContent = sampleRate + " Hz";

    detector = new WasmDetector(key, sampleRate);

    await audioContext.audioWorklet.addModule("processor.js");
    const source = audioContext.createMediaStreamSource(stream);
    workletNode = new AudioWorkletNode(audioContext, "sample-forwarder");

    workletNode.port.onmessage = (event) => {
      const chunk = event.data;
      sampleChunks.push(chunk);
      sampleChunksLength += chunk.length;
    };

    source.connect(workletNode);
    // Don't connect to destination (no feedback loop)

    processTimer = setInterval(processSamples, PROCESS_INTERVAL_MS);
    listening = true;

    startBtn.disabled = true;
    stopBtn.disabled = false;
    keyInput.disabled = true;
    deviceSelect.disabled = true;
    setStatus("Listening...", "listening");
  } catch (err) {
    setStatus("Error: " + err.message, "error");
    console.error(err);
  }
}

function stop() {
  listening = false;

  if (processTimer) {
    clearInterval(processTimer);
    processTimer = null;
  }
  if (workletNode) {
    workletNode.disconnect();
    workletNode = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }

  sampleChunks = [];
  sampleChunksLength = 0;
  detector = null;

  startBtn.disabled = false;
  stopBtn.disabled = true;
  keyInput.disabled = false;
  deviceSelect.disabled = false;
  setStatus("Stopped", "idle");
}

async function enumerateDevices() {
  try {
    // Request permission first so device labels are available
    const tempStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    tempStream.getTracks().forEach((t) => t.stop());

    const devices = await navigator.mediaDevices.enumerateDevices();
    const audioInputs = devices.filter((d) => d.kind === "audioinput");

    deviceSelect.innerHTML = '<option value="">Default</option>';
    for (const device of audioInputs) {
      const option = document.createElement("option");
      option.value = device.deviceId;
      option.textContent = device.label || `Microphone ${deviceSelect.length}`;
      deviceSelect.appendChild(option);
    }
  } catch (err) {
    console.warn("Could not enumerate devices:", err);
  }
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);
enumerateDevices();
