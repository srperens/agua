use crate::key::WatermarkKey;

/// Length of the sync pattern in bits.
pub const SYNC_PATTERN_BITS: usize = 64;

/// Generate a deterministic sync pattern from the watermark key.
/// Returns a fixed 64-bit pattern used to mark the start of each watermark block.
pub fn generate_sync_pattern(key: &WatermarkKey) -> Vec<bool> {
    // Use the key to generate a pseudo-random 64-bit sync pattern
    // by encrypting a known block with a specific "sync" marker
    use aes::Aes128;
    use aes::cipher::{BlockEncrypt, KeyInit};

    let cipher = Aes128::new_from_slice(key.as_bytes()).expect("key is always 16 bytes");
    let mut block = aes::Block::from([
        0x53, 0x59, 0x4E, 0x43, // "SYNC"
        0x50, 0x41, 0x54, 0x54, // "PATT"
        0x45, 0x52, 0x4E, 0x00, // "ERN\0"
        0x00, 0x00, 0x00, 0x00,
    ]);
    cipher.encrypt_block(&mut block);

    let bytes: [u8; 16] = block.into();
    let mut pattern = Vec::with_capacity(SYNC_PATTERN_BITS);
    for &byte in &bytes[..8] {
        for j in (0..8).rev() {
            pattern.push((byte >> j) & 1 == 1);
        }
    }
    pattern
}

/// Compute correlation between detected soft values and the expected sync pattern.
///
/// `soft_values` should contain soft-decision values for the sync region.
/// Returns the normalized correlation (-1.0 to 1.0).
pub fn correlate_sync(soft_values: &[f32], sync_pattern: &[bool]) -> f32 {
    if soft_values.len() != sync_pattern.len() || soft_values.is_empty() {
        return 0.0;
    }

    let mut correlation = 0.0f32;
    for (&soft, &expected) in soft_values.iter().zip(sync_pattern.iter()) {
        if expected {
            correlation += soft;
        } else {
            correlation -= soft;
        }
    }

    correlation / soft_values.len() as f32
}

/// Structure of a watermark block:
/// [SYNC_PATTERN_BITS sync frames] [CODED_BITS data frames]
/// Total frames per block = SYNC_PATTERN_BITS + CODED_BITS
pub fn frames_per_block() -> usize {
    use crate::codec::CODED_BITS;
    SYNC_PATTERN_BITS + CODED_BITS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sync_pattern_deterministic() {
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let p1 = generate_sync_pattern(&key);
        let p2 = generate_sync_pattern(&key);
        assert_eq!(p1, p2);
        assert_eq!(p1.len(), SYNC_PATTERN_BITS);
    }

    #[test]
    fn sync_pattern_differs_by_key() {
        let k1 = WatermarkKey::new(&[1u8; 16]).unwrap();
        let k2 = WatermarkKey::new(&[2u8; 16]).unwrap();
        let p1 = generate_sync_pattern(&k1);
        let p2 = generate_sync_pattern(&k2);
        assert_ne!(p1, p2);
    }

    #[test]
    fn correlation_perfect_match() {
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let pattern = generate_sync_pattern(&key);
        let soft: Vec<f32> = pattern
            .iter()
            .map(|&b| if b { 1.0 } else { -1.0 })
            .collect();
        let corr = correlate_sync(&soft, &pattern);
        assert!((corr - 1.0).abs() < 1e-6);
    }

    #[test]
    fn correlation_perfect_mismatch() {
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let pattern = generate_sync_pattern(&key);
        // Invert all soft values
        let soft: Vec<f32> = pattern
            .iter()
            .map(|&b| if b { -1.0 } else { 1.0 })
            .collect();
        let corr = correlate_sync(&soft, &pattern);
        assert!((corr - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn correlation_random_near_zero() {
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let pattern = generate_sync_pattern(&key);
        // Use a different key's pattern as "random" soft values
        let other_key = WatermarkKey::new(&[99u8; 16]).unwrap();
        let other_pattern = generate_sync_pattern(&other_key);
        let soft: Vec<f32> = other_pattern
            .iter()
            .map(|&b| if b { 1.0 } else { -1.0 })
            .collect();
        let corr = correlate_sync(&soft, &pattern);
        // Random correlation should be close to zero
        assert!(corr.abs() < 0.5, "random correlation too high: {corr}");
    }
}
