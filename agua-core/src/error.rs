use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("invalid key length: expected 16 bytes, got {0}")]
    InvalidKeyLength(usize),

    #[error("invalid payload length: expected {expected} bits, got {got}")]
    InvalidPayloadLength { expected: usize, got: usize },

    #[error("CRC mismatch: expected {expected:#010x}, got {got:#010x}")]
    CrcMismatch { expected: u32, got: u32 },

    #[error("audio too short: need at least {needed} samples, got {got}")]
    AudioTooShort { needed: usize, got: usize },

    #[error("no watermark detected")]
    NotDetected,

    #[error("FFT error: {0}")]
    Fft(String),
}

pub type Result<T> = std::result::Result<T, Error>;
