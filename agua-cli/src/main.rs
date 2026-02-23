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

        /// Embedding strength (0.001 - 0.1)
        #[arg(short, long, default_value = "0.01")]
        strength: f32,
    },
    /// Detect a watermark in a WAV file
    Detect {
        /// Input WAV file
        #[arg(short, long)]
        input: PathBuf,

        /// Key passphrase
        #[arg(short, long, default_value = "agua-default-key")]
        key: String,
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
        } => {
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
                ..Default::default()
            };

            eprintln!(
                "Embedding watermark into {} ({} samples, {}Hz)...",
                input.display(),
                samples.len(),
                spec.sample_rate
            );

            agua_core::embed(&mut samples, &wm_payload, &wm_key, &config)?;

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
        Command::Detect { input, key } => {
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
                ..Default::default()
            };

            eprintln!(
                "Detecting watermark in {} ({} samples, {}Hz)...",
                input.display(),
                samples.len(),
                spec.sample_rate
            );

            match agua_core::detect(&samples, &wm_key, &config) {
                Ok(results) => {
                    for (i, r) in results.iter().enumerate() {
                        println!("Detection #{}", i + 1);
                        println!("  Payload:    {}", r.payload.to_hex());
                        println!("  Confidence: {:.4}", r.confidence);
                        println!("  Offset:     frame {}", r.offset);
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
