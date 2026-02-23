use crate::error::{Error, Result};

/// Maximum payload size in bits.
pub const PAYLOAD_BITS: usize = 128;
/// CRC-32 size in bits.
pub const CRC_BITS: usize = 32;
/// Total data bits (payload + CRC).
pub const TOTAL_DATA_BITS: usize = PAYLOAD_BITS + CRC_BITS;

/// A 128-bit watermark payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Payload {
    /// The raw 128-bit payload as 16 bytes.
    data: [u8; 16],
}

impl Payload {
    /// Create a payload from a 16-byte array.
    pub fn new(data: [u8; 16]) -> Self {
        Self { data }
    }

    /// Create a payload from a byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 16 {
            return Err(Error::InvalidPayloadLength {
                expected: 128,
                got: bytes.len() * 8,
            });
        }
        let mut data = [0u8; 16];
        data.copy_from_slice(bytes);
        Ok(Self { data })
    }

    /// Create a payload from a hex string.
    pub fn from_hex(hex: &str) -> Result<Self> {
        let hex = hex.trim();
        if hex.len() != 32 {
            return Err(Error::InvalidPayloadLength {
                expected: 128,
                got: hex.len() * 4,
            });
        }
        let mut data = [0u8; 16];
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).map_err(|_| {
                Error::InvalidPayloadLength {
                    expected: 128,
                    got: 0,
                }
            })?;
        }
        Ok(Self { data })
    }

    /// Get the raw bytes.
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.data
    }

    /// Convert to hex string.
    pub fn to_hex(&self) -> String {
        self.data.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Get individual bits as a vector of bools (MSB first).
    pub fn to_bits(&self) -> Vec<bool> {
        let mut bits = Vec::with_capacity(PAYLOAD_BITS);
        for byte in &self.data {
            for j in (0..8).rev() {
                bits.push((byte >> j) & 1 == 1);
            }
        }
        bits
    }

    /// Reconstruct payload from bits (MSB first).
    pub fn from_bits(bits: &[bool]) -> Result<Self> {
        if bits.len() != PAYLOAD_BITS {
            return Err(Error::InvalidPayloadLength {
                expected: PAYLOAD_BITS,
                got: bits.len(),
            });
        }
        let mut data = [0u8; 16];
        for (i, &bit) in bits.iter().enumerate() {
            if bit {
                data[i / 8] |= 1 << (7 - (i % 8));
            }
        }
        Ok(Self { data })
    }

    /// Append CRC-32 to payload bits, returning TOTAL_DATA_BITS bits.
    pub fn encode_with_crc(&self) -> Vec<bool> {
        let mut bits = self.to_bits();
        let crc = crc32(&self.data);
        for j in (0..32).rev() {
            bits.push((crc >> j) & 1 == 1);
        }
        debug_assert_eq!(bits.len(), TOTAL_DATA_BITS);
        bits
    }

    /// Decode payload from TOTAL_DATA_BITS bits, verifying CRC.
    pub fn decode_with_crc(bits: &[bool]) -> Result<Self> {
        if bits.len() != TOTAL_DATA_BITS {
            return Err(Error::InvalidPayloadLength {
                expected: TOTAL_DATA_BITS,
                got: bits.len(),
            });
        }
        let payload = Payload::from_bits(&bits[..PAYLOAD_BITS])?;
        let expected_crc = crc32(&payload.data);

        let mut got_crc: u32 = 0;
        for &bit in &bits[PAYLOAD_BITS..] {
            got_crc = (got_crc << 1) | (bit as u32);
        }

        if expected_crc != got_crc {
            return Err(Error::CrcMismatch {
                expected: expected_crc,
                got: got_crc,
            });
        }
        Ok(payload)
    }
}

/// CRC-32 (ISO 3309 / ITU-T V.42, same as used in PNG, gzip, etc.)
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payload_round_trip_bits() {
        let data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let payload = Payload::new(data);
        let bits = payload.to_bits();
        assert_eq!(bits.len(), 128);
        let recovered = Payload::from_bits(&bits).unwrap();
        assert_eq!(payload, recovered);
    }

    #[test]
    fn payload_crc_round_trip() {
        let payload = Payload::new([0xDE, 0xAD, 0xBE, 0xEF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let encoded = payload.encode_with_crc();
        assert_eq!(encoded.len(), TOTAL_DATA_BITS);
        let decoded = Payload::decode_with_crc(&encoded).unwrap();
        assert_eq!(payload, decoded);
    }

    #[test]
    fn payload_crc_detects_error() {
        let payload = Payload::new([0xFF; 16]);
        let mut encoded = payload.encode_with_crc();
        // Flip a bit
        encoded[50] = !encoded[50];
        assert!(Payload::decode_with_crc(&encoded).is_err());
    }

    #[test]
    fn payload_hex_round_trip() {
        let payload = Payload::new([
            0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45,
            0x67, 0x89,
        ]);
        let hex = payload.to_hex();
        let recovered = Payload::from_hex(&hex).unwrap();
        assert_eq!(payload, recovered);
    }

    #[test]
    fn crc32_known_value() {
        // CRC-32 of "123456789" is 0xCBF43926
        let data = b"123456789";
        assert_eq!(crc32(data), 0xCBF4_3926);
    }
}
