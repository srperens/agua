/// Configuration for watermark embedding and detection.
#[derive(Debug, Clone)]
pub struct WatermarkConfig {
    /// Sample rate in Hz (e.g. 44100, 48000).
    pub sample_rate: u32,
    /// Watermark embedding strength (power-law exponent delta).
    /// Higher = more robust but more audible. Default: 0.1.
    pub strength: f32,
    /// FFT frame size in samples. Must be power of 2. Default: 1024.
    pub frame_size: usize,
    /// Number of frequency bin pairs per frame. Default: 60.
    pub num_bin_pairs: usize,
    /// Minimum frequency in Hz for watermark embedding. Default: 860.0.
    pub min_freq_hz: f32,
    /// Maximum frequency in Hz for watermark embedding. Default: 4300.0.
    pub max_freq_hz: f32,
    /// Spacing between bins in each pair. Default: 8 (~375 Hz at 48 kHz).
    /// Higher values spread pairs across wider frequency gaps, improving
    /// resilience to acoustic channel degradation and comb filtering.
    pub bin_spacing: usize,
}

impl Default for WatermarkConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            strength: 0.1,
            frame_size: 1024,
            num_bin_pairs: 60,
            min_freq_hz: 860.0,
            max_freq_hz: 4300.0,
            bin_spacing: 8,
        }
    }
}

impl WatermarkConfig {
    /// Configuration tuned for robustness against lossy codecs (MP3, AAC, Opus).
    ///
    /// Uses higher embedding strength (`0.05`) to survive lossy compression.
    /// The watermark will be slightly more audible but significantly more robust.
    pub fn robust() -> Self {
        Self {
            strength: 0.05,
            ..Self::default()
        }
    }

    /// Configuration tuned for the acoustic channel (speaker → air → mic).
    ///
    /// Uses strength `0.08` to overcome the severe degradation of the
    /// acoustic path: room reverberation, speaker/mic frequency response,
    /// AGC, and ambient noise. More audible than `robust()` but necessary
    /// for reliable detection through air.
    pub fn acoustic() -> Self {
        Self {
            strength: 0.08,
            ..Self::default()
        }
    }

    /// Number of complex frequency bins (frame_size / 2 + 1).
    pub fn num_bins(&self) -> usize {
        self.frame_size / 2 + 1
    }

    /// Hop size for overlap-add (50% overlap).
    pub fn hop_size(&self) -> usize {
        self.frame_size / 2
    }

    /// Compute the effective FFT bin range from frequency bounds.
    ///
    /// Returns `(min_bin, max_bin)` where `max_bin` is exclusive.
    /// At 48 kHz / 1024 FFT with default frequencies: bins 19..92.
    pub fn effective_bin_range(&self) -> (usize, usize) {
        let bin_freq = self.sample_rate as f32 / self.frame_size as f32;
        let min_bin = (self.min_freq_hz / bin_freq).ceil() as usize;
        let max_bin = (self.max_freq_hz / bin_freq).floor() as usize + 1;
        (min_bin, max_bin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_bin_range_48khz() {
        let config = WatermarkConfig::default();
        let (min_bin, max_bin) = config.effective_bin_range();
        assert_eq!(min_bin, 19);
        assert_eq!(max_bin, 92);
    }

    #[test]
    fn bin_range_44100hz() {
        let config = WatermarkConfig {
            sample_rate: 44100,
            ..WatermarkConfig::default()
        };
        let (min_bin, max_bin) = config.effective_bin_range();
        // 44100/1024 = 43.066 Hz per bin
        // min: ceil(860/43.066) = 20, max: floor(4300/43.066)+1 = 100
        assert_eq!(min_bin, 20);
        assert_eq!(max_bin, 100);
    }

    #[test]
    fn hop_size_is_half_frame() {
        let config = WatermarkConfig::default();
        assert_eq!(config.hop_size(), 512);
    }
}
