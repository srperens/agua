use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "agua", about = "Audio watermarking tool", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Embed a watermark into a WAV file
    Embed {
        /// Input WAV file
        #[arg(short, long)]
        input: PathBuf,

        /// Output WAV file
        #[arg(short, long)]
        output: PathBuf,

        /// Payload as 32-char hex string (128 bits)
        #[arg(short, long)]
        payload: String,

        /// Key passphrase
        #[arg(short, long, default_value = "agua-default-key")]
        key: String,

        /// Embedding strength (0.001 - 0.1)
        #[arg(short, long, default_value = "0.01")]
        strength: f32,

        /// Parameter profile
        #[arg(long, value_enum)]
        profile: Option<Profile>,

        /// Delay before embedding starts (seconds)
        #[arg(long, default_value = "0")]
        offset_seconds: f32,

        /// FFT frame size (power of 2)
        #[arg(long, default_value = "1024")]
        frame_size: usize,

        /// Number of bin pairs per frame
        #[arg(long, default_value = "200")]
        num_bin_pairs: usize,

        /// Minimum FFT bin index
        #[arg(long, default_value = "5")]
        min_bin: usize,

        /// Maximum FFT bin index
        #[arg(long, default_value = "500")]
        max_bin: usize,
    },
    /// Detect a watermark in a WAV file
    Detect {
        /// Input WAV file
        #[arg(short, long)]
        input: PathBuf,

        /// Key passphrase
        #[arg(short, long, default_value = "agua-default-key")]
        key: String,

        /// Parameter profile
        #[arg(long, value_enum)]
        profile: Option<Profile>,

        /// Start offset for detection (seconds)
        #[arg(long, default_value = "0")]
        offset_seconds: f32,

        /// FFT frame size (power of 2)
        #[arg(long, default_value = "1024")]
        frame_size: usize,

        /// Number of bin pairs per frame
        #[arg(long, default_value = "200")]
        num_bin_pairs: usize,

        /// Minimum FFT bin index
        #[arg(long, default_value = "5")]
        min_bin: usize,

        /// Maximum FFT bin index
        #[arg(long, default_value = "500")]
        max_bin: usize,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Profile {
    Music,
}

fn apply_profile(
    profile: Option<Profile>,
    strength: &mut f32,
    num_bin_pairs: &mut usize,
    min_bin: &mut usize,
    max_bin: &mut usize,
) {
    match profile {
        Some(Profile::Music) => {
            *strength = 0.05;
            *num_bin_pairs = 50;
            *min_bin = 5;
            *max_bin = 300;
        }
        None => {}
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Command::Embed {
            input,
            output,
            payload,
            key,
            mut strength,
            profile,
            offset_seconds,
            frame_size,
            mut num_bin_pairs,
            mut min_bin,
            mut max_bin,
        } => {
            if !frame_size.is_power_of_two() {
                return Err(format!("frame_size must be power of 2, got {frame_size}").into());
            }
            apply_profile(profile, &mut strength, &mut num_bin_pairs, &mut min_bin, &mut max_bin);
            let reader = hound::WavReader::open(&input)?;
            let spec = reader.spec();

            if spec.channels != 1 {
                eprintln!(
                    "Warning: input has {} channels, only the first channel will be used.",
                    spec.channels
                );
            }

            let mut samples: Vec<f32> = match spec.sample_format {
                hound::SampleFormat::Float => reader
                    .into_samples::<f32>()
                    .collect::<Result<Vec<f32>, _>>()?,
                hound::SampleFormat::Int => {
                    let bits = spec.bits_per_sample;
                    let max = (1i32 << (bits - 1)) as f32;
                    reader
                        .into_samples::<i32>()
                        .collect::<Result<Vec<i32>, _>>()?
                        .into_iter()
                        .map(|s| s as f32 / max)
                        .collect()
                }
            };

            // If multi-channel, take only first channel
            if spec.channels > 1 {
                samples = samples
                    .chunks(spec.channels as usize)
                    .map(|c| c[0])
                    .collect();
            }

            let wm_key = agua_core::WatermarkKey::from_passphrase(&key);
            let wm_payload = agua_core::Payload::from_hex(&payload)?;
            let config = agua_core::WatermarkConfig {
                sample_rate: spec.sample_rate,
                strength,
                frame_size,
                num_bin_pairs,
                min_bin,
                max_bin,
                ..Default::default()
            };
            let num_bins = config.num_bins();
            if config.min_bin >= num_bins || config.max_bin >= num_bins {
                eprintln!(
                    "Warning: bin range [{}, {}] exceeds available bins (0..{}). Consider lowering min/max bin.",
                    config.min_bin,
                    config.max_bin,
                    num_bins - 1
                );
            }

            eprintln!(
                "Embedding watermark into {} ({} samples, {}Hz)...",
                input.display(),
                samples.len(),
                spec.sample_rate
            );

            let frames_per_block = agua_core::sync::frames_per_block();
            let total_frames = samples.len() / config.frame_size;
            let offset_samples = (offset_seconds * config.sample_rate as f32).round() as usize;
            let offset_frames = offset_samples / config.frame_size;
            let effective_offset = offset_frames * config.frame_size;
            if effective_offset >= samples.len() {
                return Err(format!(
                    "offset_seconds too large: {}s exceeds audio length",
                    offset_seconds
                )
                .into());
            }
            if effective_offset != offset_samples {
                eprintln!(
                    "Warning: offset aligned to frame boundary ({} samples ≈ {:.3}s).",
                    effective_offset,
                    effective_offset as f32 / config.sample_rate as f32
                );
            }
            let num_frames = total_frames.saturating_sub(offset_frames);
            if num_frames < frames_per_block {
                let needed_samples = frames_per_block * config.frame_size;
                let needed_seconds = needed_samples as f32 / config.sample_rate as f32;
                eprintln!(
                    "Warning: audio is too short for reliable detection ({} frames, need {}). Minimum duration ≈ {:.2}s at {}Hz.",
                    num_frames,
                    frames_per_block,
                    needed_seconds,
                    config.sample_rate
                );
            }

            let start = effective_offset;
            let frame_offset = offset_frames as u32;
            agua_core::embed_with_offset(&mut samples[start..], &wm_payload, &wm_key, &config, frame_offset)?;

            // Write output WAV
            let out_spec = hound::WavSpec {
                channels: 1,
                sample_rate: spec.sample_rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            let mut writer = hound::WavWriter::create(&output, out_spec)?;
            for &s in &samples {
                writer.write_sample(s)?;
            }
            writer.finalize()?;

            eprintln!("Watermarked audio written to {}", output.display());
            println!("Payload: {}", wm_payload.to_hex());
        }
        Command::Detect {
            input,
            key,
            profile,
            offset_seconds,
            frame_size,
            mut num_bin_pairs,
            mut min_bin,
            mut max_bin,
        } => {
            if !frame_size.is_power_of_two() {
                return Err(format!("frame_size must be power of 2, got {frame_size}").into());
            }
            let mut strength = 0.01;
            apply_profile(profile, &mut strength, &mut num_bin_pairs, &mut min_bin, &mut max_bin);
            let reader = hound::WavReader::open(&input)?;
            let spec = reader.spec();

            let mut samples: Vec<f32> = match spec.sample_format {
                hound::SampleFormat::Float => reader
                    .into_samples::<f32>()
                    .collect::<Result<Vec<f32>, _>>()?,
                hound::SampleFormat::Int => {
                    let bits = spec.bits_per_sample;
                    let max = (1i32 << (bits - 1)) as f32;
                    reader
                        .into_samples::<i32>()
                        .collect::<Result<Vec<i32>, _>>()?
                        .into_iter()
                        .map(|s| s as f32 / max)
                        .collect()
                }
            };

            if spec.channels > 1 {
                samples = samples
                    .chunks(spec.channels as usize)
                    .map(|c| c[0])
                    .collect();
            }

            let wm_key = agua_core::WatermarkKey::from_passphrase(&key);
            let config = agua_core::WatermarkConfig {
                sample_rate: spec.sample_rate,
                strength,
                frame_size,
                num_bin_pairs,
                min_bin,
                max_bin,
                ..Default::default()
            };
            let num_bins = config.num_bins();
            if config.min_bin >= num_bins || config.max_bin >= num_bins {
                eprintln!(
                    "Warning: bin range [{}, {}] exceeds available bins (0..{}). Consider lowering min/max bin.",
                    config.min_bin,
                    config.max_bin,
                    num_bins - 1
                );
            }

            eprintln!(
                "Detecting watermark in {} ({} samples, {}Hz)...",
                input.display(),
                samples.len(),
                spec.sample_rate
            );

            let frames_per_block = agua_core::sync::frames_per_block();
            let total_frames = samples.len() / config.frame_size;
            let offset_samples = (offset_seconds * config.sample_rate as f32).round() as usize;
            let offset_frames = offset_samples / config.frame_size;
            let effective_offset = offset_frames * config.frame_size;
            if effective_offset >= samples.len() {
                return Err(format!(
                    "offset_seconds too large: {}s exceeds audio length",
                    offset_seconds
                )
                .into());
            }
            if effective_offset != offset_samples {
                eprintln!(
                    "Warning: offset aligned to frame boundary ({} samples ≈ {:.3}s).",
                    effective_offset,
                    effective_offset as f32 / config.sample_rate as f32
                );
            }
            let num_frames = total_frames.saturating_sub(offset_frames);
            if num_frames < frames_per_block {
                let needed_samples = frames_per_block * config.frame_size;
                let needed_seconds = needed_samples as f32 / config.sample_rate as f32;
                eprintln!(
                    "Warning: audio is too short for reliable detection ({} frames, need {}). Minimum duration ≈ {:.2}s at {}Hz.",
                    num_frames,
                    frames_per_block,
                    needed_seconds,
                    config.sample_rate
                );
            }

            let start = effective_offset;
            let frame_offset = offset_frames as u32;
            match agua_core::detect_with_offset(&samples[start..], &wm_key, &config, frame_offset) {
                Ok(results) => {
                    for (i, r) in results.iter().enumerate() {
                        println!("Detection #{}", i + 1);
                        println!("  Payload:    {}", r.payload.to_hex());
                        println!("  Confidence: {:.4}", r.confidence);
                        println!("  Offset:     frame {}", r.offset);
                        let offset_seconds =
                            r.offset as f32 * config.frame_size as f32 / config.sample_rate as f32;
                        println!("  Offset:     {:.3} s", offset_seconds);
                    }
                }
                Err(agua_core::Error::NotDetected) => {
                    eprintln!("No watermark detected.");
                    std::process::exit(1);
                }
                Err(e) => return Err(e.into()),
            }
        }
    }

    Ok(())
}
