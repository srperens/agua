use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;

use crate::error::{Error, Result};

/// Pre-allocated FFT processor for a fixed frame size.
pub struct FftProcessor {
    frame_size: usize,
    forward: Arc<dyn RealToComplex<f32>>,
    inverse: Arc<dyn ComplexToReal<f32>>,
    // Pre-allocated scratch buffers
    freq_buf: Vec<Complex32>,
    scratch_fwd: Vec<Complex32>,
    scratch_inv: Vec<Complex32>,
}

impl FftProcessor {
    /// Create a new FFT processor for the given frame size.
    /// Frame size must be even and > 0.
    pub fn new(frame_size: usize) -> Result<Self> {
        let mut planner = RealFftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(frame_size);
        let inverse = planner.plan_fft_inverse(frame_size);

        let freq_buf = forward.make_output_vec();
        let scratch_fwd = forward.make_scratch_vec();
        let scratch_inv = inverse.make_scratch_vec();

        Ok(Self {
            frame_size,
            forward,
            inverse,
            freq_buf,
            scratch_fwd,
            scratch_inv,
        })
    }

    /// Number of complex frequency bins (frame_size/2 + 1).
    pub fn num_bins(&self) -> usize {
        self.frame_size / 2 + 1
    }

    /// Frame size this processor was created for.
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Perform forward FFT: time domain -> frequency domain.
    ///
    /// `time_buf` must have exactly `frame_size` elements and will be modified in place.
    /// Returns a reference to the internal frequency buffer.
    pub fn forward(&mut self, time_buf: &mut [f32]) -> Result<&mut [Complex32]> {
        if time_buf.len() != self.frame_size {
            return Err(Error::Fft(format!(
                "expected {} samples, got {}",
                self.frame_size,
                time_buf.len()
            )));
        }
        self.forward
            .process_with_scratch(time_buf, &mut self.freq_buf, &mut self.scratch_fwd)
            .map_err(|e| Error::Fft(e.to_string()))?;
        Ok(&mut self.freq_buf)
    }

    /// Perform inverse FFT: frequency domain -> time domain.
    ///
    /// Reads from the internal frequency buffer and writes to `time_buf`.
    /// The output is scaled by `frame_size` (realfft convention); call `normalize`
    /// after if you need unit-scale output.
    pub fn inverse(&mut self, time_buf: &mut [f32]) -> Result<()> {
        if time_buf.len() != self.frame_size {
            return Err(Error::Fft(format!(
                "expected {} samples, got {}",
                self.frame_size,
                time_buf.len()
            )));
        }
        self.inverse
            .process_with_scratch(&mut self.freq_buf, time_buf, &mut self.scratch_inv)
            .map_err(|e| Error::Fft(e.to_string()))?;
        Ok(())
    }

    /// Get a mutable reference to the frequency buffer for modification.
    pub fn freq_bins_mut(&mut self) -> &mut [Complex32] {
        &mut self.freq_buf
    }

    /// Get an immutable reference to the frequency buffer.
    pub fn freq_bins(&self) -> &[Complex32] {
        &self.freq_buf
    }

    /// Normalize time-domain samples after inverse FFT.
    /// realfft's inverse scales by frame_size, so divide by it.
    pub fn normalize(&self, time_buf: &mut [f32]) {
        let scale = 1.0 / self.frame_size as f32;
        for s in time_buf.iter_mut() {
            *s *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let size = 1024;
        let mut proc = FftProcessor::new(size).unwrap();

        // Generate a test signal (sum of sinusoids)
        let mut original = vec![0.0f32; size];
        for (i, sample) in original.iter_mut().enumerate() {
            let t = i as f32 / size as f32;
            *sample = (2.0 * std::f32::consts::PI * 100.0 * t).sin()
                + 0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        }
        let original_copy = original.clone();

        // Forward FFT
        let mut time_buf = original;
        proc.forward(&mut time_buf).unwrap();

        // Inverse FFT
        proc.inverse(&mut time_buf).unwrap();
        proc.normalize(&mut time_buf);

        // Check round-trip accuracy
        for (i, (a, b)) in original_copy.iter().zip(time_buf.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "sample {i}: {a} vs {b}, diff={}",
                (a - b).abs()
            );
        }
    }

    #[test]
    fn num_bins_correct() {
        let proc = FftProcessor::new(1024).unwrap();
        assert_eq!(proc.num_bins(), 513);
    }

    #[test]
    fn wrong_buffer_size() {
        let mut proc = FftProcessor::new(1024).unwrap();
        let mut buf = vec![0.0f32; 512];
        assert!(proc.forward(&mut buf).is_err());
    }
}
