const WORKER_VERSION = "0.6.5";
console.log("[worker.js] loaded, VERSION=" + WORKER_VERSION);

let wasmInit = null;
let WasmDetector = null;
let detector = null;
let processedTotal = 0;
let batchCount = 0;

self.onmessage = async (event) => {
  const msg = event.data || {};
  if (msg.type === "init") {
    try {
      console.log("[worker] init: key='" + msg.key + "' sampleRate=" + msg.sampleRate + " preprocess=" + msg.preprocess);
      const mod = await import("./pkg/agua_web.js?v=" + WORKER_VERSION);
      wasmInit = mod.default;
      WasmDetector = mod.WasmDetector;
      await wasmInit({ module_or_path: "./pkg/agua_web_bg.wasm?v=" + WORKER_VERSION });
      detector = new WasmDetector(msg.key, msg.sampleRate);
      console.log("[worker] WasmDetector created");
      if (msg.preprocess === false) {
        detector.set_preprocess(false);
        console.log("[worker] preprocessing disabled");
      }
      self.postMessage({
        type: "info",
        message: `worker init: sampleRate=${msg.sampleRate} preprocess=${msg.preprocess}`,
      });
      self.postMessage({ type: "ready" });
    } catch (err) {
      console.error("[worker] init FAILED:", err);
      self.postMessage({ type: "info", message: `worker init FAILED: ${err}` });
    }
    return;
  }
  if (msg.type === "process" && detector) {
    const samples = msg.samples;
    processedTotal += samples.length;
    batchCount += 1;
    if (batchCount === 1) {
      console.log("[worker] first chunk: length=" + samples.length + " first5=[" + Array.from(samples.slice(0, 5)).map(v => v.toFixed(6)) + "]");
    }
    const payload = detector.process(samples);
    const confidence = detector.get_confidence();
    const bufferFill = detector.get_buffer_fill();
    const bestSyncCorr = typeof detector.get_best_sync_corr === "function" ? detector.get_best_sync_corr() : 0;
    const detectAttempts = typeof detector.get_detect_attempts === "function" ? detector.get_detect_attempts() : 0;
    const syncCandidates = typeof detector.get_sync_candidates === "function" ? detector.get_sync_candidates() : 0;

    if (batchCount % 50 === 0) {
      console.log("[worker] batch=" + batchCount + " total=" + processedTotal + " fill=" + bufferFill.toFixed(3) + " syncCorr=" + bestSyncCorr.toFixed(4) + " candidates=" + syncCandidates + " attempts=" + detectAttempts);
    }
    if (payload) {
      console.log("[worker] DETECTED: payload=" + payload + " confidence=" + confidence.toFixed(4));
      self.postMessage({
        type: "info",
        message: `DETECTED: payload=${payload} confidence=${confidence.toFixed(4)}`,
      });
    }
    self.postMessage({
      type: "result",
      payload: payload || null,
      confidence,
      bufferFill,
      processedTotal,
      bestSyncCorr,
      detectAttempts,
      syncCandidates,
    });
  }
};
