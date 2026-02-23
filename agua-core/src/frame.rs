/// Apply a Hann window to a frame of samples.
pub fn apply_hann_window(samples: &mut [f32]) {
    let n = samples.len() as f32;
    for (i, s) in samples.iter_mut().enumerate() {
        let w = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n).cos());
        *s *= w;
    }
}

/// Generate a Hann window of the given size.
pub fn hann_window(size: usize) -> Vec<f32> {
    let n = size as f32;
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / n).cos()))
        .collect()
}

/// Overlap-add reconstruction helper.
///
/// Adds `frame` into `output` starting at `offset`, applying the given window.
/// The window should be the synthesis window (Hann for WOLA).
pub fn overlap_add(output: &mut [f32], frame: &[f32], window: &[f32], offset: usize) {
    for (i, (&sample, &w)) in frame.iter().zip(window.iter()).enumerate() {
        let pos = offset + i;
        if pos < output.len() {
            output[pos] += sample * w;
        }
    }
}

/// Extract a frame from the input signal at the given offset, applying a window.
///
/// If the frame extends past the end of the input, it is zero-padded.
pub fn extract_frame(input: &[f32], offset: usize, frame_size: usize, window: &[f32]) -> Vec<f32> {
    let mut frame = vec![0.0f32; frame_size];
    for i in 0..frame_size {
        let pos = offset + i;
        if pos < input.len() {
            frame[i] = input[pos] * window[i];
        }
    }
    frame
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_window_endpoints() {
        let w = hann_window(1024);
        assert_eq!(w.len(), 1024);
        // Hann window is zero at endpoints
        assert!(w[0].abs() < 1e-6);
        // And peaks at the center
        assert!((w[512] - 1.0).abs() < 0.01);
    }

    #[test]
    fn overlap_add_reconstruction() {
        // With 50% overlap and Hann window, the sum of overlapping windows
        // should be approximately constant (COLA condition).
        let size = 1024;
        let hop = size / 2;
        let window = hann_window(size);
        let num_frames = 10;
        let total_len = hop * (num_frames + 1);

        let mut sum = vec![0.0f32; total_len];
        for f in 0..num_frames {
            let frame = vec![1.0f32; size]; // constant signal
            overlap_add(&mut sum, &frame, &window, f * hop);
        }

        // In the steady-state region (after the first frame, before the last),
        // the sum should be close to 1.0
        for &s in &sum[size..total_len - size] {
            assert!((s - 1.0).abs() < 0.01, "COLA violation: sum = {s}");
        }
    }

    #[test]
    fn extract_frame_with_padding() {
        let input = vec![1.0f32; 100];
        let window = vec![1.0f32; 256];
        let frame = extract_frame(&input, 50, 256, &window);
        assert_eq!(frame.len(), 256);
        // First 50 samples should be 1.0
        for &s in &frame[..50] {
            assert!((s - 1.0).abs() < 1e-6);
        }
        // After input ends, should be zero-padded
        for &s in &frame[50..] {
            assert!(s.abs() < 1e-6);
        }
    }
}
