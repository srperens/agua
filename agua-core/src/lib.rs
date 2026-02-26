pub mod codec;
pub mod config;
pub mod detect;
pub mod embed;
pub mod error;
pub mod fft;
pub mod frame;
pub mod key;
pub mod patchwork;
pub mod payload;
pub mod preprocess;
pub mod stream;
pub mod sync;

#[cfg(feature = "parallel")]
pub mod parallel;

// Re-export primary API types
pub use config::WatermarkConfig;
pub use detect::DetectionDiagnostics;
pub use embed::DetectionResult;
pub use error::Error;
pub use key::WatermarkKey;
pub use payload::Payload;
pub use preprocess::PreProcessor;
pub use stream::{StreamDetector, StreamEmbedder};

#[cfg(feature = "parallel")]
pub use parallel::{detect_parallel, embed_parallel};

/// Embed a watermark into audio samples (in-place).
///
/// This is the one-shot API for file-based workflows.
/// For streaming/real-time use, see [`StreamEmbedder`].
pub fn embed(
    samples: &mut [f32],
    payload: &Payload,
    key: &WatermarkKey,
    config: &WatermarkConfig,
) -> error::Result<()> {
    embed::embed(samples, payload, key, config)
}

/// Embed a watermark into audio samples (in-place) starting at a frame offset.
pub fn embed_with_offset(
    samples: &mut [f32],
    payload: &Payload,
    key: &WatermarkKey,
    config: &WatermarkConfig,
    frame_offset: u32,
) -> error::Result<()> {
    embed::embed_with_offset(samples, payload, key, config, frame_offset)
}

/// Detect watermarks in audio samples.
///
/// Returns all successfully detected watermark payloads.
/// For streaming/real-time use, see [`StreamDetector`].
pub fn detect(
    samples: &[f32],
    key: &WatermarkKey,
    config: &WatermarkConfig,
) -> error::Result<Vec<DetectionResult>> {
    detect::detect(samples, key, config)
}

/// Detect watermarks with diagnostics (sync correlation, candidate counts, etc.).
pub fn detect_with_diagnostics(
    samples: &[f32],
    key: &WatermarkKey,
    config: &WatermarkConfig,
) -> (error::Result<Vec<DetectionResult>>, DetectionDiagnostics) {
    detect::detect_with_diagnostics(samples, key, config)
}

/// Detect watermarks in audio samples starting at a frame offset.
pub fn detect_with_offset(
    samples: &[f32],
    key: &WatermarkKey,
    config: &WatermarkConfig,
    frame_offset: u32,
) -> error::Result<Vec<DetectionResult>> {
    detect::detect_with_offset(samples, key, config, frame_offset)
}
