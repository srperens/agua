use agua_core::{PreProcessor, StreamDetector, WatermarkConfig, WatermarkKey};
use wasm_bindgen::prelude::*;

macro_rules! console_log {
    ($($arg:tt)*) => {
        web_sys::console::log_1(&format!($($arg)*).into())
    };
}

/// WASM wrapper around agua-core's StreamDetector with mic preprocessing.
#[wasm_bindgen]
pub struct WasmDetector {
    preprocessor: PreProcessor,
    preprocess_enabled: bool,
    detector: StreamDetector,
    last_confidence: f32,
    last_payload: Option<String>,
    process_calls: u32,
    total_samples: u64,
}

#[wasm_bindgen]
impl WasmDetector {
    /// Create a new detector with the given key passphrase and sample rate.
    #[wasm_bindgen(constructor)]
    pub fn new(key: &str, sample_rate: u32) -> Result<WasmDetector, JsError> {
        let wm_key = WatermarkKey::from_passphrase(key);
        let config = WatermarkConfig {
            sample_rate,
            ..WatermarkConfig::default()
        };

        let preprocessor = PreProcessor::new(sample_rate);
        let detector =
            StreamDetector::new(&wm_key, &config).map_err(|e| JsError::new(&format!("{e}")))?;

        console_log!(
            "[wasm] WasmDetector created: key='{}' sr={} preprocess=true",
            key,
            sample_rate
        );

        Ok(WasmDetector {
            preprocessor,
            preprocess_enabled: true,
            detector,
            last_confidence: 0.0,
            last_payload: None,
            process_calls: 0,
            total_samples: 0,
        })
    }

    /// Enable or disable mic preprocessing (bandpass + RMS normalization).
    ///
    /// Enabled by default. Disable for clean file input (e.g. offline WAV detection).
    pub fn set_preprocess(&mut self, enabled: bool) {
        console_log!("[wasm] set_preprocess({})", enabled);
        self.preprocess_enabled = enabled;
    }

    /// Feed audio samples to the detector.
    ///
    /// Applies bandpass filtering and RMS normalization (if enabled),
    /// then feeds to the streaming detector. Returns the detected payload
    /// as a hex string, or null if nothing detected yet.
    pub fn process(&mut self, samples: &[f32]) -> Option<String> {
        self.process_calls += 1;
        self.total_samples += samples.len() as u64;

        // Compute input signal stats
        let (in_rms, in_peak) = signal_stats(samples);

        let results = if self.preprocess_enabled {
            let mut buf = samples.to_vec();
            self.preprocessor.process(&mut buf);

            let (post_rms, post_peak) = signal_stats(&buf);

            // Log every 100th call to avoid flooding console
            if self.process_calls % 100 == 1 {
                console_log!(
                    "[wasm] process #{}: {} samples | in: rms={:.5} peak={:.5} | post-preproc: rms={:.5} peak={:.5} | buf_fill={:.3} buf_samples={}",
                    self.process_calls,
                    samples.len(),
                    in_rms,
                    in_peak,
                    post_rms,
                    post_peak,
                    self.detector.buffer_fill(),
                    self.detector.diag_buffer_samples(),
                );
            }

            self.detector.process(&buf)
        } else {
            if self.process_calls % 100 == 1 {
                console_log!(
                    "[wasm] process #{}: {} samples (no preproc) | rms={:.5} peak={:.5} | buf_fill={:.3}",
                    self.process_calls,
                    samples.len(),
                    in_rms,
                    in_peak,
                    self.detector.buffer_fill(),
                );
            }
            self.detector.process(samples)
        };

        // Log detection attempts and sync search diagnostics
        let attempts = self.detector.diag_detect_attempts();
        let best_corr = self.detector.diag_best_sync_corr();
        let candidates = self.detector.diag_sync_candidates();
        if self.process_calls % 100 == 1 && attempts > 0 {
            console_log!(
                "[wasm] detect stats: attempts={} best_sync_corr={:.4} sync_candidates={} total_samples={}",
                attempts,
                best_corr,
                candidates,
                self.total_samples,
            );
        }

        if let Some(result) = results.last() {
            self.last_confidence = result.confidence;
            let hex = result.payload.to_hex();
            console_log!(
                "[wasm] DETECTED: payload={} confidence={:.4}",
                hex,
                result.confidence
            );
            self.last_payload = Some(hex.clone());
            Some(hex)
        } else {
            None
        }
    }

    /// Confidence of the last successful detection (0.0 if none yet).
    pub fn get_confidence(&self) -> f32 {
        self.last_confidence
    }

    /// How full the detection buffer is (0.0 to 1.0).
    pub fn get_buffer_fill(&self) -> f32 {
        self.detector.buffer_fill()
    }

    /// The last detected payload hex string, or null.
    pub fn get_last_payload(&self) -> Option<String> {
        self.last_payload.clone()
    }

    /// Best sync correlation found in the last detection attempt.
    pub fn get_best_sync_corr(&self) -> f32 {
        self.detector.diag_best_sync_corr()
    }

    /// Number of detection attempts so far.
    pub fn get_detect_attempts(&self) -> u32 {
        self.detector.diag_detect_attempts()
    }

    /// Number of sync candidates above threshold in last detection attempt.
    pub fn get_sync_candidates(&self) -> u32 {
        self.detector.diag_sync_candidates()
    }

    /// Reset the detector and preprocessor state.
    pub fn reset(&mut self) {
        console_log!("[wasm] reset()");
        self.preprocessor.reset();
        self.detector.reset();
        self.last_confidence = 0.0;
        self.last_payload = None;
        self.process_calls = 0;
        self.total_samples = 0;
    }
}

fn signal_stats(samples: &[f32]) -> (f32, f32) {
    let mut sum_sq = 0.0f32;
    let mut peak = 0.0f32;
    for &s in samples {
        sum_sq += s * s;
        let a = s.abs();
        if a > peak {
            peak = a;
        }
    }
    let rms = if samples.is_empty() {
        0.0
    } else {
        (sum_sq / samples.len() as f32).sqrt()
    };
    (rms, peak)
}
