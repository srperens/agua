use aes::Aes128;
use aes::cipher::{BlockEncrypt, KeyInit};

use crate::error::{Error, Result};

/// A watermark key used to deterministically select frequency bin pairs.
///
/// Wraps an AES-128 key used as a PRNG to generate pseudo-random bin pair
/// selections for the patchwork algorithm.
#[derive(Clone)]
pub struct WatermarkKey {
    cipher: Aes128,
    raw: [u8; 16],
}

impl WatermarkKey {
    /// Create a new watermark key from a 16-byte slice.
    pub fn new(key_bytes: &[u8]) -> Result<Self> {
        if key_bytes.len() != 16 {
            return Err(Error::InvalidKeyLength(key_bytes.len()));
        }
        let mut raw = [0u8; 16];
        raw.copy_from_slice(key_bytes);
        let cipher = Aes128::new_from_slice(key_bytes).expect("key length already validated");
        Ok(Self { cipher, raw })
    }

    /// Create a key from a passphrase by hashing it with a simple mixing function.
    pub fn from_passphrase(passphrase: &str) -> Self {
        let mut key = [0u8; 16];
        // Simple deterministic mixing: iterate bytes, XOR + rotate
        for (i, b) in passphrase.bytes().enumerate() {
            key[i % 16] ^= b;
            // Additional mixing: rotate the target byte
            key[(i + 7) % 16] = key[(i + 7) % 16].wrapping_add(b.wrapping_mul(0x9E));
        }
        // Final pass: avalanche mixing using AES itself
        let cipher = Aes128::new_from_slice(&key).expect("key is 16 bytes");
        let mut block = aes::Block::from(key);
        cipher.encrypt_block(&mut block);
        let raw: [u8; 16] = block.into();
        let cipher = Aes128::new_from_slice(&raw).expect("key is 16 bytes");
        Self { cipher, raw }
    }

    /// Returns the raw 16-byte key.
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.raw
    }

    /// Generate pseudo-random bin pairs for a given frame index.
    ///
    /// Each pair consists of bins `(k, k + bin_spacing)` where `k` is randomly
    /// selected. With `bin_spacing = 1` (default), this gives adjacent-bin
    /// pairing for maximum SNR. Larger spacing improves resilience to comb
    /// filtering at the cost of slightly noisier soft values.
    /// The key determines which bin is "a" vs "b" in each pair.
    pub fn generate_bin_pairs(
        &self,
        frame_index: u32,
        num_pairs: usize,
        min_bin: usize,
        max_bin: usize,
        bin_spacing: usize,
    ) -> Vec<(usize, usize)> {
        let bin_range = max_bin - min_bin;
        if bin_range <= bin_spacing {
            return Vec::new();
        }

        let mut pairs = Vec::with_capacity(num_pairs);
        let mut counter: u32 = 0;
        // Range for center bin: [min_bin, max_bin - spacing) so that center+spacing < max_bin
        let center_range = bin_range - bin_spacing;

        while pairs.len() < num_pairs {
            // Build a 16-byte input block: [frame_index(4) | counter(4) | padding(8)]
            let mut input = [0u8; 16];
            input[0..4].copy_from_slice(&frame_index.to_le_bytes());
            input[4..8].copy_from_slice(&counter.to_le_bytes());

            let mut block = aes::Block::from(input);
            self.cipher.encrypt_block(&mut block);
            let output: [u8; 16] = block.into();

            // Extract pairs from output bytes: 2 bytes for center bin + 1 byte for direction
            // Use 3 bytes per pair → 5 pairs per 16-byte block
            for chunk in output.chunks_exact(3) {
                if pairs.len() >= num_pairs {
                    break;
                }
                let center =
                    (u16::from_le_bytes([chunk[0], chunk[1]]) as usize) % center_range + min_bin;
                let swap = chunk[2] & 1 == 1;
                if swap {
                    pairs.push((center + bin_spacing, center));
                } else {
                    pairs.push((center, center + bin_spacing));
                }
            }
            counter += 1;
        }

        pairs
    }
}

impl std::fmt::Debug for WatermarkKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WatermarkKey")
            .field("raw", &"[REDACTED]")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_valid_key() {
        let key = WatermarkKey::new(&[0u8; 16]).unwrap();
        assert_eq!(key.as_bytes(), &[0u8; 16]);
    }

    #[test]
    fn new_invalid_length() {
        assert!(WatermarkKey::new(&[0u8; 15]).is_err());
        assert!(WatermarkKey::new(&[0u8; 17]).is_err());
    }

    #[test]
    fn from_passphrase_deterministic() {
        let k1 = WatermarkKey::from_passphrase("test-key");
        let k2 = WatermarkKey::from_passphrase("test-key");
        assert_eq!(k1.as_bytes(), k2.as_bytes());
    }

    #[test]
    fn from_passphrase_different_inputs() {
        let k1 = WatermarkKey::from_passphrase("key-a");
        let k2 = WatermarkKey::from_passphrase("key-b");
        assert_ne!(k1.as_bytes(), k2.as_bytes());
    }

    #[test]
    fn bin_pairs_deterministic() {
        let key = WatermarkKey::new(&[42u8; 16]).unwrap();
        let pairs1 = key.generate_bin_pairs(0, 30, 5, 500, 1);
        let pairs2 = key.generate_bin_pairs(0, 30, 5, 500, 1);
        assert_eq!(pairs1, pairs2);
    }

    #[test]
    fn bin_pairs_in_range() {
        let key = WatermarkKey::new(&[1u8; 16]).unwrap();
        let pairs = key.generate_bin_pairs(7, 30, 5, 500, 1);
        assert_eq!(pairs.len(), 30);
        for (a, b) in &pairs {
            assert!(*a >= 5 && *a < 500);
            assert!(*b >= 5 && *b < 500);
            assert_ne!(a, b);
            // Adjacent-bin pairing: difference should be exactly 1
            assert_eq!((*a as isize - *b as isize).unsigned_abs(), 1);
        }
    }

    #[test]
    fn bin_pairs_with_spacing() {
        let key = WatermarkKey::new(&[1u8; 16]).unwrap();
        let pairs = key.generate_bin_pairs(7, 30, 5, 500, 4);
        assert_eq!(pairs.len(), 30);
        for (a, b) in &pairs {
            assert!(*a >= 5 && *a < 500);
            assert!(*b >= 5 && *b < 500);
            assert_ne!(a, b);
            // Bin spacing of 4: difference should be exactly 4
            assert_eq!((*a as isize - *b as isize).unsigned_abs(), 4);
        }
    }

    #[test]
    fn bin_pairs_differ_across_frames() {
        let key = WatermarkKey::new(&[99u8; 16]).unwrap();
        let p0 = key.generate_bin_pairs(0, 30, 5, 500, 1);
        let p1 = key.generate_bin_pairs(1, 30, 5, 500, 1);
        assert_ne!(p0, p1);
    }
}
