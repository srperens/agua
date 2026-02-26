

const VERSION = "0.8.0";
console.log("[app.js] loaded, VERSION=" + VERSION);
const PROCESS_INTERVAL_MS = 50;
const MAX_SAMPLES_PER_TICK = 48000; // ~1s at 48kHz — allows catching up if backlog grows
const MAX_QUEUE_SAMPLES = 48000 * 5; // cap backlog to ~5s to avoid UI slowdown
const DEFAULT_KEY = "agua-default-key";

let detectorWorker = null;
let detectorReady = false;
let audioContext = null;
let workletNode = null;
let analyser = null;
let freqData = null;
let sampleChunks = [];
let sampleChunksLength = 0;
let sampleChunksReadIndex = 0;
let processTimer = null;
let listening = false;
let processedSamplesTotal = 0;
let workerProcessedTotal = 0;

// --- Recording state ---
let recording = false;
let recordedChunks = [];
let recordedSampleCount = 0;
let recordedBlob = null;

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
const offlineFile = document.getElementById("offline-file");
const offlineRun = document.getElementById("offline-run");
const offlineStatus = document.getElementById("offline-status");
const offlineDemo = document.getElementById("offline-demo");
const offlineDownload = document.getElementById("offline-download");
const playerToggle = document.getElementById("player-toggle");
const playerStatus = document.getElementById("player-status");
const playerSelect = document.getElementById("player-select");
const debugToggle = document.getElementById("debug-toggle");
const debugPanel = document.getElementById("debug-panel");
const dbgRms = document.getElementById("dbg-rms");
const dbgPeak = document.getElementById("dbg-peak");
const dbgBand = document.getElementById("dbg-band");
const dbgQueued = document.getElementById("dbg-queued");
const dbgProcessed = document.getElementById("dbg-processed");
const dbgBatch = document.getElementById("dbg-batch");
const dbgDetectAttempts = document.getElementById("dbg-detect-attempts");
const dbgSyncCorr = document.getElementById("dbg-sync-corr");
const dbgSyncCandidates = document.getElementById("dbg-sync-candidates");
const dbgCombineCount = document.getElementById("dbg-combine-count");
const vuBar = document.getElementById("vu-bar");
const clearLogBtn = document.getElementById("clear-log-btn");
const recBtn = document.getElementById("rec-btn");
const recDownload = document.getElementById("rec-download");
const recStatus = document.getElementById("rec-status");

document.getElementById("version-tag").textContent = `v${VERSION}`;

clearLogBtn.addEventListener("click", () => {
  detectionLog.innerHTML = "";
  payloadEl.textContent = "\u2014";
  confidenceEl.textContent = "\u2014";
  confidenceBar.style.width = "0%";
});

debugToggle.checked = localStorage.getItem("agua-debug") === "1";
debugPanel.style.display = debugToggle.checked ? "flex" : "none";
debugToggle.addEventListener("change", () => {
  debugPanel.style.display = debugToggle.checked ? "flex" : "none";
  localStorage.setItem("agua-debug", debugToggle.checked ? "1" : "0");
});

offlineFile.addEventListener("change", () => {
  offlineRun.disabled = !offlineFile.files || offlineFile.files.length === 0;
});

async function runOfflineDetect(arrayBuf) {
  offlineStatus.textContent = "Decoding WAV...";
  console.log("[offline] starting, arrayBuf size:", arrayBuf.byteLength);
  const ctx = new AudioContext({ sampleRate: 48000 });
  try {
    const audioBuf = await ctx.decodeAudioData(arrayBuf.slice(0));
    console.log("[offline] decoded:", audioBuf.length, "frames @", audioBuf.sampleRate, "Hz, channels:", audioBuf.numberOfChannels);
    offlineStatus.textContent = `Decoded: ${audioBuf.length} frames @ ${audioBuf.sampleRate} Hz`;
    const channelData = audioBuf.getChannelData(0);

    // Log signal stats
    let sumSq = 0, peak = 0, minVal = 0, maxVal = 0;
    for (let i = 0; i < channelData.length; i++) {
      const v = channelData[i];
      sumSq += v * v;
      if (v > maxVal) maxVal = v;
      if (v < minVal) minVal = v;
      const av = Math.abs(v);
      if (av > peak) peak = av;
    }
    const rms = Math.sqrt(sumSq / channelData.length);
    console.log("[offline] signal stats: rms=", rms.toFixed(6), "peak=", peak.toFixed(6), "min=", minVal.toFixed(6), "max=", maxVal.toFixed(6));
    console.log("[offline] first 20 samples:", Array.from(channelData.slice(0, 20)).map(v => v.toFixed(6)));

    const chunkSize = 4096;
    const offlineWorker = new Worker(`worker.js?v=${VERSION}`, { type: "module" });
    const key = keyInput.value.trim() || DEFAULT_KEY;
    console.log("[offline] init worker: key='" + key + "', sampleRate=", audioBuf.sampleRate, ", preprocess=false");
    await new Promise((resolve) => {
      offlineWorker.onmessage = (event) => {
        const msg = event.data || {};
        if (msg.type === "ready") {
          console.log("[offline] worker ready");
          resolve();
        }
        if (msg.type === "info") console.log("[offline worker]", msg.message);
      };
      offlineWorker.postMessage({
        type: "init",
        key,
        sampleRate: audioBuf.sampleRate,
        preprocess: false,
      });
    });

    let detected = null;
    let confidence = 0;
    const totalChunks = Math.ceil(channelData.length / chunkSize);
    console.log("[offline] processing", totalChunks, "chunks of", chunkSize, "samples");
    for (let i = 0; i < channelData.length; i += chunkSize) {
      const slice = channelData.subarray(i, Math.min(i + chunkSize, channelData.length));
      const chunk = new Float32Array(slice);
      const res = await new Promise((resolve) => {
        const handler = (event) => {
          const msg = event.data || {};
          if (msg.type === "result") {
            offlineWorker.removeEventListener("message", handler);
            resolve(msg);
          }
        };
        offlineWorker.addEventListener("message", handler);
        offlineWorker.postMessage({ type: "process", samples: chunk }, [chunk.buffer]);
      });
      const idx = Math.floor(i / chunkSize) + 1;
      if (idx % 50 === 0 || res.payload) {
        console.log("[offline] chunk", idx + "/" + totalChunks, "bufferFill=", (res.bufferFill || 0).toFixed(3), "payload=", res.payload, "confidence=", res.confidence);
      }
      if (idx % 200 === 0) {
        const pct = ((idx / totalChunks) * 100).toFixed(1);
        offlineStatus.textContent = `Processing... ${pct}%`;
      }
      if (res.payload) {
        detected = res.payload;
        confidence = res.confidence || 0;
        console.log("[offline] DETECTED:", detected, "confidence:", confidence);
        break;
      }
    }
    console.log("[offline] loop done, detected=", detected);
    offlineWorker.terminate();
    if (detected) {
      offlineStatus.textContent = `Detected payload ${detected} (confidence ${(confidence * 100).toFixed(1)}%)`;
      payloadEl.textContent = detected;
      confidenceEl.textContent = (confidence * 100).toFixed(1) + "%";
      confidenceBar.style.width = Math.min(confidence * 100, 100).toFixed(1) + "%";
      setStatus("Watermark detected (offline)", "detected");
      addDetection(detected, confidence);
    } else {
      offlineStatus.textContent = "No watermark detected (offline)";
    }
  } catch (err) {
    offlineStatus.textContent = "Decode error: " + err.message;
  } finally {
    ctx.close();
  }
}

offlineRun.addEventListener("click", async () => {
  if (!offlineFile.files || offlineFile.files.length === 0) return;
  const file = offlineFile.files[0];
  const arrayBuf = await file.arrayBuffer();
  await runOfflineDetect(arrayBuf);
});

offlineDemo.addEventListener("click", async () => {
  const demoFile = playerSelect.value;
  offlineStatus.textContent = "Downloading demo WAV...";
  const url = new URL(`${demoFile}?v=${VERSION}`, window.location.href).toString();
  const res = await fetch(url);
  if (!res.ok) {
    offlineStatus.textContent = `Download failed: ${res.status}`;
    return;
  }
  const arrayBuf = await res.arrayBuffer();
  await runOfflineDetect(arrayBuf);
});

offlineDownload.addEventListener("click", () => {
  const demoFile = playerSelect.value;
  const a = document.createElement("a");
  a.href = `${demoFile}?v=${VERSION}`;
  a.download = demoFile;
  a.click();
});

// --- Demo player (loop playback through speakers) ---
let playerAudioBuf = null;
let playerCtx = null;
let playerSource = null;
let playerPlaying = false;

playerToggle.addEventListener("click", async () => {
  if (playerPlaying) {
    // Stop
    if (playerSource) {
      playerSource.stop();
      playerSource = null;
    }
    if (playerCtx) {
      playerCtx.close();
      playerCtx = null;
    }
    playerPlaying = false;
    playerToggle.textContent = "Play demo";
    playerStatus.textContent = "Stopped";
    return;
  }

  try {
    playerStatus.textContent = "Loading...";
    if (!playerAudioBuf) {
      const res = await fetch(new URL(`${playerSelect.value}?v=${VERSION}`, window.location.href).toString());
      if (!res.ok) {
        playerStatus.textContent = "Download failed: " + res.status;
        return;
      }
      const arrayBuf = await res.arrayBuffer();
      const tmpCtx = new AudioContext({ sampleRate: 48000 });
      playerAudioBuf = await tmpCtx.decodeAudioData(arrayBuf);
      tmpCtx.close();
    }

    playerCtx = new AudioContext({ sampleRate: 48000 });
    playerSource = playerCtx.createBufferSource();
    playerSource.buffer = playerAudioBuf;
    playerSource.loop = true;
    playerSource.connect(playerCtx.destination);
    playerSource.start();
    playerPlaying = true;
    playerToggle.textContent = "Stop demo";
    playerStatus.textContent = "Playing (loop)";

    playerSource.onended = () => {
      // Only fires if stop() wasn't called manually (shouldn't happen with loop)
      playerPlaying = false;
      playerToggle.textContent = "Play demo";
      playerStatus.textContent = "Stopped";
    };
  } catch (err) {
    playerStatus.textContent = "Error: " + err.message;
    console.error("[player]", err);
  }
});

playerSelect.addEventListener("change", () => {
  playerAudioBuf = null;
  if (playerPlaying) {
    // Stop current playback, then restart with new file
    if (playerSource) { playerSource.stop(); playerSource = null; }
    if (playerCtx) { playerCtx.close(); playerCtx = null; }
    playerPlaying = false;
    playerToggle.textContent = "Play demo";
    playerStatus.textContent = "Switching...";
    playerToggle.click();
  }
});

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
  // VU meter — always active, even before worker is ready
  if (analyser && sampleChunksLength > 0) {
    const td = new Float32Array(analyser.fftSize);
    analyser.getFloatTimeDomainData(td);
    let sumSq = 0;
    for (let i = 0; i < td.length; i++) sumSq += td[i] * td[i];
    const vu = Math.min(Math.sqrt(sumSq / td.length) * 10, 1);
    vuBar.style.width = (vu * 100).toFixed(1) + "%";
  }

  if (!detectorReady || sampleChunksLength === 0) return;

  // Drain at most MAX_SAMPLES_PER_TICK to keep the main thread responsive.
  let toProcess = 0;
  let chunksToTake = 0;
  for (let i = sampleChunksReadIndex; i < sampleChunks.length; i++) {
    if (toProcess + sampleChunks[i].length > MAX_SAMPLES_PER_TICK && toProcess > 0) break;
    toProcess += sampleChunks[i].length;
    chunksToTake = i + 1;
  }

  const samples = new Float32Array(toProcess);
  let offset = 0;
  for (let i = sampleChunksReadIndex; i < chunksToTake; i++) {
    samples.set(sampleChunks[i], offset);
    offset += sampleChunks[i].length;
  }

  sampleChunksReadIndex = chunksToTake;
  if (sampleChunksReadIndex > 64) {
    sampleChunks = sampleChunks.slice(sampleChunksReadIndex);
    sampleChunksReadIndex = 0;
  }
  sampleChunksLength -= toProcess;

  processedSamplesTotal += samples.length;

  if (debugToggle.checked) {
    let sumSq = 0;
    let peak = 0;
    for (let i = 0; i < samples.length; i++) {
      const v = samples[i];
      sumSq += v * v;
      const av = Math.abs(v);
      if (av > peak) peak = av;
    }
    const rms = Math.sqrt(sumSq / samples.length);
    dbgRms.textContent = rms.toFixed(5);
    dbgPeak.textContent = peak.toFixed(5);
    dbgQueued.textContent = sampleChunksLength.toString();
    dbgProcessed.textContent = processedSamplesTotal.toString();
    dbgBatch.textContent = samples.length.toString();

    if (analyser && freqData) {
      analyser.getFloatFrequencyData(freqData);
      const binFreq = audioContext.sampleRate / analyser.fftSize;
      const minBin = Math.max(0, Math.ceil(860 / binFreq));
      const maxBin = Math.min(freqData.length - 1, Math.floor(4300 / binFreq));
      let sum = 0;
      let count = 0;
      for (let i = minBin; i <= maxBin; i++) {
        const db = freqData[i];
        const power = Math.pow(10, db / 10);
        sum += power;
        count += 1;
      }
      const avgPower = count > 0 ? sum / count : 0;
      dbgBand.textContent = avgPower.toExponential(3);
    }
  }

  detectorWorker.postMessage(
    { type: "process", samples },
    [samples.buffer],
  );
}

async function start() {
  try {
    setStatus("Initializing...", "idle");

    const key = keyInput.value.trim() || DEFAULT_KEY;
    const audioConstraints = {
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      channelCount: 1,
      sampleRate: 48000,
      sampleSize: 32,
      latency: 0,
      voiceIsolation: false,
    };
    const selectedDevice = deviceSelect.value;
    if (selectedDevice) {
      audioConstraints.deviceId = { exact: selectedDevice };
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: audioConstraints,
    });

    audioContext = new AudioContext({ sampleRate: 48000 });
    const sampleRate = audioContext.sampleRate;
    sampleRateEl.textContent = sampleRate + " Hz";

    const track = stream.getAudioTracks()[0];
    if (track && track.getSettings) {
      const settings = track.getSettings();
      const parts = [];
      if (settings.sampleRate) parts.push(`sr=${settings.sampleRate}`);
      if (settings.channelCount) parts.push(`ch=${settings.channelCount}`);
      if (settings.sampleSize) parts.push(`bits=${settings.sampleSize}`);
      if (settings.echoCancellation !== undefined)
        parts.push(`aec=${settings.echoCancellation}`);
      if (settings.noiseSuppression !== undefined)
        parts.push(`ns=${settings.noiseSuppression}`);
      if (settings.autoGainControl !== undefined)
        parts.push(`agc=${settings.autoGainControl}`);
      if (settings.voiceIsolation !== undefined)
        parts.push(`vi=${settings.voiceIsolation}`);
      console.log("[audio] input settings:", parts.join(", "));
      if (settings.sampleRate && settings.sampleRate < 44100) {
        console.warn("[audio] device sample rate is", settings.sampleRate, "Hz (need 44100+)");
      }
      if (settings.noiseSuppression === true || settings.echoCancellation === true) {
        console.warn("[audio] browser is applying DSP (NS/AEC) despite request to disable");
      }
    }

    detectorWorker = new Worker(`worker.js?v=${VERSION}`, { type: "module" });
    detectorWorker.onmessage = (event) => {
      const msg = event.data || {};
      if (msg.type === "info" && msg.message) {
        console.info(msg.message);
        return;
      }
      if (msg.type === "debug" && msg.message) {
        console.debug(msg.message);
        return;
      }
      if (msg.type === "ready") {
        detectorReady = true;
        return;
      }
      if (msg.type === "result") {
        workerProcessedTotal = msg.processedTotal || workerProcessedTotal;
        const fill = msg.bufferFill ?? 0;
        bufferBar.style.width = (fill * 100).toFixed(1) + "%";
        bufferText.textContent = (fill * 100).toFixed(0) + "%";
        if (debugToggle.checked) {
          dbgProcessed.textContent = workerProcessedTotal.toString();
          if (msg.detectAttempts !== undefined) {
            dbgDetectAttempts.textContent = msg.detectAttempts;
            dbgSyncCorr.textContent = (msg.bestSyncCorr ?? 0).toFixed(4);
            dbgSyncCandidates.textContent = msg.syncCandidates ?? 0;
            dbgCombineCount.textContent = msg.combineCount ?? 0;
          }
        }
        if (msg.payload) {
          payloadEl.textContent = msg.payload;
          const confidence = msg.confidence ?? 0;
          confidenceEl.textContent = (confidence * 100).toFixed(1) + "%";
          confidenceBar.style.width = Math.min(confidence * 100, 100).toFixed(1) + "%";
          setStatus("Watermark detected", "detected");
          addDetection(msg.payload, confidence);
          setTimeout(() => {
            if (listening) setStatus("Listening...", "listening");
          }, 2000);
        }
      }
    };
    detectorWorker.postMessage({ type: "init", key, sampleRate });

    await audioContext.audioWorklet.addModule(`processor.js?v=${VERSION}`);
    const source = audioContext.createMediaStreamSource(stream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 4096;
    freqData = new Float32Array(analyser.frequencyBinCount);
    workletNode = new AudioWorkletNode(audioContext, "sample-forwarder");

    workletNode.port.onmessage = (event) => {
      const chunk = event.data;

      // Capture raw mic samples for recording (before any processing)
      if (recording) {
        recordedChunks.push(new Float32Array(chunk));
        recordedSampleCount += chunk.length;
        if (recordedSampleCount % 48000 < chunk.length) {
          const secs = (recordedSampleCount / (audioContext ? audioContext.sampleRate : 48000)).toFixed(1);
          recStatus.textContent = `Recording... ${secs}s`;
        }
      }

      sampleChunks.push(chunk);
      sampleChunksLength += chunk.length;
      if (sampleChunksLength > MAX_QUEUE_SAMPLES) {
        let dropped = 0;
        while (sampleChunksReadIndex < sampleChunks.length && sampleChunksLength > MAX_QUEUE_SAMPLES) {
          const first = sampleChunks[sampleChunksReadIndex];
          sampleChunksReadIndex += 1;
          sampleChunksLength -= first.length;
          dropped += first.length;
        }
        if (sampleChunksReadIndex > 64) {
          sampleChunks = sampleChunks.slice(sampleChunksReadIndex);
          sampleChunksReadIndex = 0;
        }
        if (debugToggle.checked) {
          dbgQueued.textContent = sampleChunksLength.toString();
          dbgBatch.textContent = `dropped ${dropped}`;
        }
      }
    };

    source.connect(workletNode);
    source.connect(analyser);
    // Don't connect to destination (no feedback loop)

    processTimer = setInterval(processSamples, PROCESS_INTERVAL_MS);
    listening = true;

    startBtn.disabled = true;
    stopBtn.disabled = false;
    keyInput.disabled = true;
    deviceSelect.disabled = true;
    recBtn.disabled = false;
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
  sampleChunksReadIndex = 0;
  detectorReady = false;
  if (detectorWorker) {
    detectorWorker.terminate();
    detectorWorker = null;
  }
  analyser = null;
  freqData = null;
  processedSamplesTotal = 0;
  workerProcessedTotal = 0;

  // Stop any active recording
  if (recording) {
    recording = false;
    recBtn.textContent = "Record";
    recBtn.style.background = "";
    recBtn.style.color = "";
  }
  recBtn.disabled = true;

  startBtn.disabled = false;
  stopBtn.disabled = true;
  keyInput.disabled = false;
  deviceSelect.disabled = false;
  setStatus("Stopped", "idle");
}

// --- Recording functions ---

function encodeWav(samples, sampleRate) {
  // 32-bit float WAV to preserve exact mic samples
  const numChannels = 1;
  const bytesPerSample = 4;
  const dataSize = samples.length * bytesPerSample;
  const headerSize = 44;
  const buffer = new ArrayBuffer(headerSize + dataSize);
  const view = new DataView(buffer);

  function writeString(offset, str) {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);            // fmt chunk size
  view.setUint16(20, 3, true);             // format = IEEE float
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * bytesPerSample, true);
  view.setUint16(32, numChannels * bytesPerSample, true);
  view.setUint16(34, bytesPerSample * 8, true);
  writeString(36, "data");
  view.setUint32(40, dataSize, true);

  const output = new Float32Array(buffer, headerSize);
  output.set(samples);

  return new Blob([buffer], { type: "audio/wav" });
}

recBtn.addEventListener("click", () => {
  if (!listening) return;

  if (!recording) {
    // Start recording
    recording = true;
    recordedChunks = [];
    recordedSampleCount = 0;
    recordedBlob = null;
    recBtn.textContent = "Stop rec";
    recBtn.style.background = "var(--red)";
    recBtn.style.color = "#fff";
    recDownload.disabled = true;
    recStatus.textContent = "Recording...";
    console.log("[rec] started");
  } else {
    // Stop recording
    recording = false;
    recBtn.textContent = "Record";
    recBtn.style.background = "";
    recBtn.style.color = "";

    if (recordedSampleCount === 0) {
      recStatus.textContent = "No samples recorded";
      return;
    }

    // Merge chunks into single Float32Array
    const merged = new Float32Array(recordedSampleCount);
    let off = 0;
    for (const chunk of recordedChunks) {
      merged.set(chunk, off);
      off += chunk.length;
    }

    const sr = audioContext ? audioContext.sampleRate : 48000;
    recordedBlob = encodeWav(merged, sr);
    const durSec = (recordedSampleCount / sr).toFixed(1);
    recStatus.textContent = `${durSec}s recorded (${recordedSampleCount} samples @ ${sr} Hz)`;
    recDownload.disabled = false;
    recordedChunks = [];
    console.log("[rec] stopped:", recordedSampleCount, "samples,", durSec, "s");
  }
});

recDownload.addEventListener("click", () => {
  if (!recordedBlob) return;
  const url = URL.createObjectURL(recordedBlob);
  const a = document.createElement("a");
  a.href = url;
  const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  a.download = `agua-mic-${ts}.wav`;
  a.click();
  URL.revokeObjectURL(url);
});

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

    // Restore saved device selection
    const saved = localStorage.getItem("agua-device");
    if (saved && [...deviceSelect.options].some((o) => o.value === saved)) {
      deviceSelect.value = saved;
    }
  } catch (err) {
    console.warn("Could not enumerate devices:", err);
  }
}

deviceSelect.addEventListener("change", () => {
  if (deviceSelect.value) {
    localStorage.setItem("agua-device", deviceSelect.value);
  } else {
    localStorage.removeItem("agua-device");
  }
});

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);
enumerateDevices();
