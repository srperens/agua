/**
 * AudioWorkletProcessor that forwards raw f32 samples to the main thread.
 *
 * Each process() call receives 128 samples (one render quantum).
 * We copy and post them to avoid transferring the shared buffer.
 */
const PROCESSOR_VERSION = "0.5.1";
console.log("[processor.js] loaded, VERSION=" + PROCESSOR_VERSION);

class SampleForwarder extends AudioWorkletProcessor {
  process(inputs, _outputs, _parameters) {
    const input = inputs[0];
    if (input && input.length > 0) {
      // Take first channel (mono)
      const samples = input[0];
      if (samples && samples.length > 0) {
        this.port.postMessage(new Float32Array(samples));
      }
    }
    return true;
  }
}

registerProcessor("sample-forwarder", SampleForwarder);
