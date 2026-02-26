

const VERSION = "0.4.0";
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
const debugToggle = document.getElementById("debug-toggle");
const debugPanel = document.getElementById("debug-panel");
const dbgRms = document.getElementById("dbg-rms");
const dbgPeak = document.getElementById("dbg-peak");
const dbgRmsPre = document.getElementById("dbg-rms-pre");
const dbgPeakPre = document.getElementById("dbg-peak-pre");
const dbgBand = document.getElementById("dbg-band");
const dbgBandPost = document.getElementById("dbg-band-post");
const dbgQueued = document.getElementById("dbg-queued");
const dbgProcessed = document.getElementById("dbg-processed");
const dbgBatch = document.getElementById("dbg-batch");
const signalWarning = document.getElementById("signal-warning");
const vuBar = document.getElementById("vu-bar");
const constraintsInfo = document.getElementById("constraints-info");

document.getElementById("version-tag").textContent = `v${VERSION}`;
const gainSlider = document.getElementById("gain-slider");
const gainValue = document.getElementById("gain-value");
const autoGainToggle = document.getElementById("auto-gain-toggle");
const autoGainValue = document.getElementById("auto-gain-value");

debugToggle.addEventListener("change", () => {
  debugPanel.style.display = debugToggle.checked ? "flex" : "none";
});

gainSlider.addEventListener("input", () => {
  gainValue.textContent = `${Number(gainSlider.value).toFixed(2)}x`;
});

offlineFile.addEventListener("change", () => {
  offlineRun.disabled = !offlineFile.files || offlineFile.files.length === 0;
});

async function runOfflineDetect(arrayBuf) {
  offlineStatus.textContent = "Decoding WAV...";
  const ctx = new AudioContext({ sampleRate: 48000 });
  try {
    const audioBuf = await ctx.decodeAudioData(arrayBuf.slice(0));
    offlineStatus.textContent = `Decoded: ${audioBuf.length} frames @ ${audioBuf.sampleRate} Hz`;
    const channelData = audioBuf.getChannelData(0);
    const chunkSize = 4096;
    const offlineWorker = new Worker(`worker.js?v=${VERSION}`, { type: "module" });
    await new Promise((resolve) => {
      offlineWorker.onmessage = (event) => {
        const msg = event.data || {};
        if (msg.type === "ready") resolve();
      };
      offlineWorker.postMessage({
        type: "init",
        key: keyInput.value.trim() || DEFAULT_KEY,
        sampleRate: audioBuf.sampleRate,
      });
    });

    let detected = null;
    let confidence = 0;
    const totalChunks = Math.ceil(channelData.length / chunkSize);
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
      if (idx % 200 === 0) {
        const pct = ((idx / totalChunks) * 100).toFixed(1);
        offlineStatus.textContent = `Processing... ${pct}%`;
      }
      if (res.payload) {
        detected = res.payload;
        confidence = res.confidence || 0;
        break;
      }
    }
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
  offlineStatus.textContent = "Downloading demo WAV...";
  const url = new URL("demo.wav", window.location.href).toString();
  const res = await fetch(url);
  if (!res.ok) {
    offlineStatus.textContent = `Download failed: ${res.status}`;
    return;
  }
  const arrayBuf = await res.arrayBuffer();
  await runOfflineDetect(arrayBuf);
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

  let gain = Number(gainSlider.value);
  if (autoGainToggle.checked) {
    let sumSq = 0;
    for (let i = 0; i < samples.length; i++) {
      const v = samples[i];
      sumSq += v * v;
    }
    const rms = Math.sqrt(sumSq / samples.length);
    const target = 0.05;
    autoGainValue.textContent = target.toFixed(2);
    if (rms > 0) {
      gain *= Math.min(target / rms, 20);
    }
  }
  if (gain !== 1) {
    for (let i = 0; i < samples.length; i++) {
      samples[i] = Math.max(-1, Math.min(1, samples[i] * gain));
    }
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

    // Post-gain band energy using the same data as detector
    if (detectorWorker) {
      detectorWorker.postMessage({ type: "debug_band", samples: samples.slice(0, Math.min(samples.length, 4096)) });
    }

    if (analyser && freqData) {
      const td = new Float32Array(analyser.fftSize);
      analyser.getFloatTimeDomainData(td);
      let preSumSq = 0;
      let prePeak = 0;
      for (let i = 0; i < td.length; i++) {
        const v = td[i];
        preSumSq += v * v;
        const av = Math.abs(v);
        if (av > prePeak) prePeak = av;
      }
      const preRms = Math.sqrt(preSumSq / td.length);
      dbgRmsPre.textContent = preRms.toFixed(5);
      dbgPeakPre.textContent = prePeak.toFixed(5);

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
      signalWarning.style.display = avgPower < 1e-8 ? "flex" : "none";
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
      constraintsInfo.textContent = "Input settings: " + parts.join(", ");
      constraintsInfo.style.display = "block";
    }

    detectorWorker = new Worker(`worker.js?v=${VERSION}`, { type: "module" });
    detectorWorker.onmessage = (event) => {
      const msg = event.data || {};
      if (msg.type === "debug_band") {
        dbgBandPost.textContent = msg.value.toExponential(3);
        return;
      }
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
