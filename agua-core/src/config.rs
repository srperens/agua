/// Configuration for watermark embedding and detection.
#[derive(Debug, Clone)]
pub struct WatermarkConfig {
    /// Sample rate in Hz (e.g. 44100, 48000).
    pub sample_rate: u32,
    /// Watermark embedding strength. Higher = more robust but more audible.
    /// Typical range: 0.005 to 0.05. Default: 0.01.
    pub strength: f32,
    /// FFT frame size in samples. Must be power of 2. Default: 1024.
    pub frame_size: usize,
    /// Number of frequency bin pairs per frame. Default: 30.
    pub num_bin_pairs: usize,
    /// Minimum FFT bin index (skip DC and very low frequencies). Default: 5.
    pub min_bin: usize,
    /// Maximum FFT bin index. Default: 500.
    pub max_bin: usize,
}

impl Default for WatermarkConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            strength: 0.01,
            frame_size: 1024,
            num_bin_pairs: 200,
            min_bin: 5,
            max_bin: 500,
        }
    }
}

impl WatermarkConfig {
    /// Number of complex frequency bins (frame_size / 2 + 1).
    pub fn num_bins(&self) -> usize {
        self.frame_size / 2 + 1
    }

    /// Hop size for overlap-add (50% overlap).
    pub fn hop_size(&self) -> usize {
        self.frame_size / 2
    }
}
