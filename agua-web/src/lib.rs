use agua_core::{PreProcessor, StreamDetector, WatermarkConfig, WatermarkKey};
use wasm_bindgen::prelude::*;

/// WASM wrapper around agua-core's StreamDetector with mic preprocessing.
#[wasm_bindgen]
pub struct WasmDetector {
    preprocessor: PreProcessor,
    detector: StreamDetector,
    last_confidence: f32,
    last_payload: Option<String>,
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

        Ok(WasmDetector {
            preprocessor,
            detector,
            last_confidence: 0.0,
            last_payload: None,
        })
    }

    /// Feed audio samples to the detector.
    ///
    /// Applies bandpass filtering and RMS normalization (to handle mic AGC),
    /// then feeds to the streaming detector. Returns the detected payload
    /// as a hex string, or null if nothing detected yet.
    pub fn process(&mut self, samples: &[f32]) -> Option<String> {
        // Preprocess: bandpass filter + RMS normalization
        let mut buf = samples.to_vec();
        self.preprocessor.process(&mut buf);

        let results = self.detector.process(&buf);

        if let Some(result) = results.last() {
            self.last_confidence = result.confidence;
            let hex = result.payload.to_hex();
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

    /// Reset the detector and preprocessor state.
    pub fn reset(&mut self) {
        self.preprocessor.reset();
        self.detector.reset();
        self.last_confidence = 0.0;
        self.last_payload = None;
    }
}
