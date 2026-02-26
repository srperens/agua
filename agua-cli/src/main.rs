use clap::{Parser, Subcommand};
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

        /// Embedding strength (power-law exponent delta)
        #[arg(short, long, default_value = "0.1")]
        strength: f32,

        /// Delay before embedding starts (seconds)
        #[arg(long, default_value = "0")]
        offset_seconds: f32,

        /// FFT frame size (power of 2)
        #[arg(long, default_value = "1024")]
        frame_size: usize,

        /// Number of bin pairs per frame
        #[arg(long, default_value = "60")]
        num_bin_pairs: usize,

        /// Minimum frequency in Hz for watermark embedding
        #[arg(long, default_value = "860.0")]
        min_freq: f32,

        /// Maximum frequency in Hz for watermark embedding
        #[arg(long, default_value = "4300.0")]
        max_freq: f32,

        /// Bin spacing within each frequency pair (1 = adjacent)
        #[arg(long, default_value = "8")]
        bin_spacing: usize,
    },
    /// Detect a watermark in a WAV file
    Detect {
        /// Input WAV file
        #[arg(short, long)]
        input: PathBuf,

        /// Key passphrase
        #[arg(short, long, default_value = "agua-default-key")]
        key: String,

        /// Start offset for detection (seconds)
        #[arg(long, default_value = "0")]
        offset_seconds: f32,

        /// FFT frame size (power of 2)
        #[arg(long, default_value = "1024")]
        frame_size: usize,

        /// Number of bin pairs per frame
        #[arg(long, default_value = "60")]
        num_bin_pairs: usize,

        /// Minimum frequency in Hz for watermark embedding
        #[arg(long, default_value = "860.0")]
        min_freq: f32,

        /// Maximum frequency in Hz for watermark embedding
        #[arg(long, default_value = "4300.0")]
        max_freq: f32,

        /// Bin spacing within each frequency pair (1 = adjacent)
        #[arg(long, default_value = "8")]
        bin_spacing: usize,

        /// Show debug diagnostics (sync correlations, soft values) on failure
        #[arg(short, long)]
        verbose: bool,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Command::Embed {
            input,
            output,
            payload,
            key,
            strength,
            offset_seconds,
            frame_size,
            num_bin_pairs,
            min_freq,
            max_freq,
            bin_spacing,
        } => {
            if !frame_size.is_power_of_two() {
                return Err(format!("frame_size must be power of 2, got {frame_size}").into());
            }
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
                min_freq_hz: min_freq,
                max_freq_hz: max_freq,
                bin_spacing,
            };

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
                    "Warning: offset aligned to frame boundary ({} samples = {:.3}s).",
                    effective_offset,
                    effective_offset as f32 / config.sample_rate as f32
                );
            }
            let num_frames = total_frames.saturating_sub(offset_frames);
            if num_frames < frames_per_block {
                let needed_samples = frames_per_block * config.frame_size;
                let needed_seconds = needed_samples as f32 / config.sample_rate as f32;
                eprintln!(
                    "Warning: audio is too short for reliable detection ({} frames, need {}). \
                     Minimum duration = {:.2}s at {}Hz.",
                    num_frames, frames_per_block, needed_seconds, config.sample_rate
                );
            }

            let start = effective_offset;
            let frame_offset = offset_frames as u32;
            agua_core::embed_with_offset(
                &mut samples[start..],
                &wm_payload,
                &wm_key,
                &config,
                frame_offset,
            )?;

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
            offset_seconds,
            frame_size,
            num_bin_pairs,
            min_freq,
            max_freq,
            bin_spacing,
            verbose,
        } => {
            if !frame_size.is_power_of_two() {
                return Err(format!("frame_size must be power of 2, got {frame_size}").into());
            }
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
                strength: 0.1,
                frame_size,
                num_bin_pairs,
                min_freq_hz: min_freq,
                max_freq_hz: max_freq,
                bin_spacing,
            };

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
                    "Warning: offset aligned to frame boundary ({} samples = {:.3}s).",
                    effective_offset,
                    effective_offset as f32 / config.sample_rate as f32
                );
            }
            let num_frames = total_frames.saturating_sub(offset_frames);
            if num_frames < frames_per_block {
                let needed_samples = frames_per_block * config.frame_size;
                let needed_seconds = needed_samples as f32 / config.sample_rate as f32;
                eprintln!(
                    "Warning: audio is too short for reliable detection ({} frames, need {}). \
                     Minimum duration = {:.2}s at {}Hz.",
                    num_frames, frames_per_block, needed_seconds, config.sample_rate
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
                    if verbose {
                        // Debug: compute best sync correlations to see if any peaks exist
                        use agua_core::{fft::FftProcessor, frame::hann_window, sync};
                        let hop_size = config.hop_size();
                        let frame_size = config.frame_size;
                        if samples[start..].len() >= frame_size {
                            let num_frames = (samples[start..].len() - frame_size) / hop_size + 1;
                            let window = hann_window(frame_size);
                            let mut fft = FftProcessor::new(frame_size)?;
                            let mut soft_values = Vec::with_capacity(num_frames);
                            for frame_idx in 0..num_frames {
                                let offset = frame_idx * hop_size;
                                let mut buf = vec![0.0f32; frame_size];
                                let end = (offset + frame_size).min(samples[start..].len());
                                buf[..end - offset].copy_from_slice(&samples[start..][offset..end]);
                                for (i, s) in buf.iter_mut().enumerate() {
                                    *s *= window[i];
                                }
                                fft.forward(&mut buf)?;
                                let global_frame = frame_offset as usize + frame_idx;
                                let soft = agua_core::patchwork::detect_frame(
                                    fft.freq_bins(),
                                    &wm_key,
                                    global_frame as u32,
                                    &config,
                                );
                                soft_values.push(soft);
                            }

                            let sync_pattern = sync::generate_sync_pattern(&wm_key);
                            let frames_per_block = sync::frames_per_block();
                            let scan_end = num_frames.saturating_sub(frames_per_block);
                            let mut best: Vec<(usize, f32)> = (0..=scan_end)
                                .map(|start| {
                                    let sync_soft =
                                        &soft_values[start..start + sync::SYNC_PATTERN_BITS];
                                    let corr = sync::correlate_sync(sync_soft, &sync_pattern);
                                    (start, corr)
                                })
                                .collect();
                            best.sort_by(|a, b| {
                                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            let top = best.iter().take(5).cloned().collect::<Vec<_>>();
                            let max = soft_values
                                .iter()
                                .cloned()
                                .fold(f32::NEG_INFINITY, f32::max);
                            let min = soft_values.iter().cloned().fold(f32::INFINITY, f32::min);
                            let mean = if soft_values.is_empty() {
                                0.0
                            } else {
                                soft_values.iter().sum::<f32>() / soft_values.len() as f32
                            };
                            eprintln!(
                                "Debug: soft_values count={}, min={:.5}, max={:.5}, mean={:.5}",
                                soft_values.len(),
                                min,
                                max,
                                mean
                            );
                            eprintln!("Debug: top sync correlations (start_frame, corr):");
                            for (i, (s, c)) in top.iter().enumerate() {
                                eprintln!("  {}: start={} corr={:.5}", i + 1, s, c);
                            }
                        }
                    }
                    eprintln!("No watermark detected.");
                    std::process::exit(1);
                }
                Err(e) => return Err(e.into()),
            }
        }
    }

    Ok(())
}
