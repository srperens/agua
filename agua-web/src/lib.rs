use agua_core::{StreamDetector, WatermarkConfig, WatermarkKey};
use wasm_bindgen::prelude::*;

/// Reset the detector after this many blocks without a detection to prevent
/// unbounded memory growth in StreamDetector's internal soft_values buffer.
const MAX_BLOCKS_WITHOUT_DETECTION: usize = 3;

/// WASM wrapper around agua-core's StreamDetector.
#[wasm_bindgen]
pub struct WasmDetector {
    key: WatermarkKey,
    config: WatermarkConfig,
    detector: StreamDetector,
    hop_size: usize,
    last_confidence: f32,
    last_payload: Option<String>,
    frames_since_detection: usize,
    frames_per_block: usize,
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

        let hop_size = config.hop_size();
        let detector =
            StreamDetector::new(&wm_key, &config).map_err(|e| JsError::new(&format!("{e}")))?;

        // sync (128 bits) + coded payload ((128 + 32) data bits * 6 rate = 960) = 1088
        let frames_per_block = 128 + (128 + 32) * 6;

        Ok(WasmDetector {
            key: wm_key,
            config,
            detector,
            hop_size,
            last_confidence: 0.0,
            last_payload: None,
            frames_since_detection: 0,
            frames_per_block,
        })
    }

    /// Feed audio samples to the detector.
    ///
    /// Returns the detected payload as a hex string, or null if nothing detected yet.
    pub fn process(&mut self, samples: &[f32]) -> Option<String> {
        let results = self.detector.process(samples);

        self.frames_since_detection += samples.len() / self.hop_size;

        if let Some(result) = results.last() {
            self.last_confidence = result.confidence;
            let hex = result.payload.to_hex();
            self.last_payload = Some(hex.clone());
            self.frames_since_detection = 0;
            Some(hex)
        } else {
            // Reset detector periodically to prevent unbounded memory growth
            // when no watermark is present in the audio.
            let reset_threshold = self.frames_per_block * MAX_BLOCKS_WITHOUT_DETECTION;
            if self.frames_since_detection >= reset_threshold {
                if let Ok(fresh) = StreamDetector::new(&self.key, &self.config) {
                    self.detector = fresh;
                }
                self.frames_since_detection = 0;
            }
            None
        }
    }

    /// Confidence of the last successful detection (0.0 if none yet).
    pub fn get_confidence(&self) -> f32 {
        self.last_confidence
    }

    /// How full the detection buffer is (0.0 to 1.0).
    pub fn get_buffer_fill(&self) -> f32 {
        (self.frames_since_detection as f32 / self.frames_per_block as f32).min(1.0)
    }

    /// The last detected payload hex string, or null.
    pub fn get_last_payload(&self) -> Option<String> {
        self.last_payload.clone()
    }
}
