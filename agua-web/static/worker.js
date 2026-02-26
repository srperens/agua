const v = new URL(import.meta.url).searchParams.get("v") || "";
const { default: init, WasmDetector } = await import(`./pkg/agua_web.js${v ? `?v=${v}` : ""}`);

let detector = null;
let processedTotal = 0;
let batchCount = 0;
let sampleRate = 48000;

self.onmessage = async (event) => {
  const msg = event.data || {};
  if (msg.type === "init") {
    await init();
    detector = new WasmDetector(msg.key, msg.sampleRate);
    sampleRate = msg.sampleRate;
    self.postMessage({
      type: "info",
      message: `worker init: sampleRate=${msg.sampleRate}`,
    });
    self.postMessage({ type: "ready" });
    return;
  }
  if (msg.type === "process" && detector) {
    const samples = msg.samples;
    processedTotal += samples.length;
    batchCount += 1;
    const payload = detector.process(samples);
    const confidence = detector.get_confidence();
    const bufferFill = detector.get_buffer_fill();
    if (batchCount % 50 === 0) {
      self.postMessage({
        type: "info",
        message: `worker process: batches=${batchCount} totalSamples=${processedTotal} bufferFill=${bufferFill.toFixed(
          3,
        )}`,
      });
    }
    if (payload) {
      self.postMessage({
        type: "info",
        message: `worker detect: payload=${payload} confidence=${confidence.toFixed(4)}`,
      });
    }
    self.postMessage({
      type: "result",
      payload: payload || null,
      confidence,
      bufferFill,
      processedTotal,
    });
  }
  if (msg.type === "debug_band" && detector) {
    const samples = msg.samples;
    if (samples && samples.length >= 1024) {
      const frame = 1024;
      const hop = 512;
      const binFreq = sampleRate / frame;
      const minBin = Math.max(0, Math.ceil(860 / binFreq));
      const maxBin = Math.min(frame / 2, Math.floor(4300 / binFreq));
      let sumPower = 0;
      let count = 0;
      const cosTable = new Float32Array(frame);
      const sinTable = new Float32Array(frame);
      for (let n = 0; n < frame; n++) {
        const a = (2 * Math.PI * n) / frame;
        cosTable[n] = Math.cos(a);
        sinTable[n] = Math.sin(a);
      }
      for (let offset = 0; offset + frame <= samples.length; offset += hop) {
        const re = new Float32Array(frame);
        for (let i = 0; i < frame; i++) {
          const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (frame - 1));
          re[i] = samples[offset + i] * w;
        }
        for (let k = minBin; k <= maxBin; k++) {
          let sumRe = 0;
          let sumIm = 0;
          for (let n = 0; n < frame; n++) {
            const idx = (k * n) % frame;
            const c = cosTable[idx];
            const s = sinTable[idx];
            sumRe += re[n] * c;
            sumIm += re[n] * s;
          }
          sumPower += sumRe * sumRe + sumIm * sumIm;
        }
        count += 1;
      }
      const avgPower = count > 0 ? sumPower / count : 0;
      self.postMessage({ type: "debug_band", value: avgPower });
    }
  }
};
