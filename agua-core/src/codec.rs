use crate::payload::TOTAL_DATA_BITS;

/// Code rate: 1/6 (each input bit produces 6 output bits).
pub const CODE_RATE_INV: usize = 6;
/// Constraint length K=7, meaning 64 states.
pub const CONSTRAINT_LENGTH: usize = 7;
/// Number of states in the trellis (2^(K-1)).
pub const NUM_STATES: usize = 1 << (CONSTRAINT_LENGTH - 1);
/// Total coded bits output.
pub const CODED_BITS: usize = TOTAL_DATA_BITS * CODE_RATE_INV;

/// Generator polynomials for rate 1/6, K=7 convolutional code.
/// These are standard polynomials in octal notation: 171, 133, 165, 117, 155, 127.
const GENERATORS: [u8; CODE_RATE_INV] = [
    0o171, // 0b1111001
    0o133, // 0b1011011
    0o165, // 0b1110101
    0o117, // 0b1001111
    0o155, // 0b1101101
    0o127, // 0b1010111
];

/// Convolutional encoder (rate 1/6, K=7).
pub fn encode(input_bits: &[bool]) -> Vec<bool> {
    let mut output = Vec::with_capacity(input_bits.len() * CODE_RATE_INV);
    let mut state: u8 = 0; // 6-bit shift register

    for &bit in input_bits {
        let input_val = if bit { 1u8 } else { 0u8 };
        // Current state + input bit forms K bits: [input, state[0], state[1], ..., state[K-2]]
        let reg = (input_val << (CONSTRAINT_LENGTH - 1)) | state;

        for &poly in &GENERATORS {
            let masked = reg & poly;
            let parity = masked.count_ones() % 2;
            output.push(parity == 1);
        }

        // Shift state: new state = [input, state[0], ..., state[K-3]]
        state = ((input_val << (CONSTRAINT_LENGTH - 2)) | (state >> 1)) & (NUM_STATES as u8 - 1);
    }

    output
}

/// Soft-decision Viterbi decoder (rate 1/6, K=7).
///
/// `soft_bits` contains soft values where positive = more likely 1, negative = more likely 0.
/// The magnitude indicates confidence.
///
/// Returns the decoded bits.
pub fn decode(soft_bits: &[f32]) -> Vec<bool> {
    let num_input_bits = soft_bits.len() / CODE_RATE_INV;
    if num_input_bits == 0 {
        return Vec::new();
    }

    // Path metrics for each state. Use f32 for soft decisions.
    let mut path_metric = vec![f32::NEG_INFINITY; NUM_STATES];
    path_metric[0] = 0.0; // Start in state 0

    // Survivor paths: for each timestep and state, store the previous state
    let mut survivors = vec![vec![0u8; NUM_STATES]; num_input_bits];

    // Pre-compute expected output for each (state, input_bit) pair
    let mut expected_output = vec![vec![[false; CODE_RATE_INV]; 2]; NUM_STATES];
    for (state, state_outputs) in expected_output.iter_mut().enumerate() {
        for input_bit in 0..2u8 {
            let reg = (input_bit << (CONSTRAINT_LENGTH - 1)) | (state as u8);
            for (g, &poly) in GENERATORS.iter().enumerate() {
                let masked = reg & poly;
                state_outputs[input_bit as usize][g] = masked.count_ones() % 2 == 1;
            }
        }
    }

    // Trellis traversal
    for t in 0..num_input_bits {
        let mut new_metric = vec![f32::NEG_INFINITY; NUM_STATES];
        let mut new_survivor = vec![0u8; NUM_STATES];

        let soft_slice = &soft_bits[t * CODE_RATE_INV..(t + 1) * CODE_RATE_INV];

        for state in 0..NUM_STATES {
            if path_metric[state] == f32::NEG_INFINITY {
                continue;
            }

            for input_bit in 0..2u8 {
                // Compute branch metric (correlation with expected output)
                let expected = &expected_output[state][input_bit as usize];
                let mut branch_metric = 0.0f32;
                for (g, &soft) in soft_slice.iter().enumerate() {
                    if expected[g] {
                        branch_metric += soft; // soft > 0 agrees with expected 1
                    } else {
                        branch_metric -= soft; // soft < 0 agrees with expected 0
                    }
                }

                // Next state
                let next_state = ((input_bit as usize) << (CONSTRAINT_LENGTH - 2)) | (state >> 1);

                let candidate = path_metric[state] + branch_metric;
                if candidate > new_metric[next_state] {
                    new_metric[next_state] = candidate;
                    new_survivor[next_state] = state as u8;
                }
            }
        }

        path_metric = new_metric;
        survivors[t] = new_survivor;
    }

    // Traceback: find the best final state
    let mut best_state = 0;
    let mut best_metric = f32::NEG_INFINITY;
    for (state, &metric) in path_metric.iter().enumerate() {
        if metric > best_metric {
            best_metric = metric;
            best_state = state;
        }
    }

    // Trace back through survivors to recover bits
    let mut decoded = vec![false; num_input_bits];
    let mut state = best_state;
    for t in (0..num_input_bits).rev() {
        let prev_state = survivors[t][state] as usize;
        // The input bit that caused the transition prev_state -> state
        // state = (input_bit << (K-2)) | (prev_state >> 1)
        let input_bit = (state >> (CONSTRAINT_LENGTH - 2)) & 1;
        decoded[t] = input_bit == 1;
        state = prev_state;
    }

    decoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_round_trip() {
        // Test with a known pattern
        let input: Vec<bool> = (0..TOTAL_DATA_BITS).map(|i| (i * 7 + 3) % 3 == 0).collect();

        let coded = encode(&input);
        assert_eq!(coded.len(), CODED_BITS);

        // Convert to hard soft-decision values
        let soft: Vec<f32> = coded.iter().map(|&b| if b { 1.0 } else { -1.0 }).collect();
        let decoded = decode(&soft);

        assert_eq!(decoded.len(), input.len());
        assert_eq!(decoded, input);
    }

    #[test]
    fn decode_with_noise() {
        let input: Vec<bool> = (0..TOTAL_DATA_BITS)
            .map(|i| (i * 13 + 5) % 2 == 0)
            .collect();

        let coded = encode(&input);
        let mut soft: Vec<f32> = coded.iter().map(|&b| if b { 1.0 } else { -1.0 }).collect();

        // Add some noise / flip a few soft values partially
        // With rate 1/6 there's a lot of redundancy, so it should still decode
        for i in (0..soft.len()).step_by(13) {
            soft[i] *= 0.1; // Reduce confidence significantly
        }

        let decoded = decode(&soft);
        assert_eq!(decoded, input);
    }

    #[test]
    fn decode_with_bit_errors() {
        let input: Vec<bool> = (0..TOTAL_DATA_BITS)
            .map(|i| (i * 11 + 2) % 3 == 0)
            .collect();

        let coded = encode(&input);
        let mut soft: Vec<f32> = coded.iter().map(|&b| if b { 1.0 } else { -1.0 }).collect();

        // Flip about 5% of bits (hard errors)
        let mut flipped = 0;
        for i in (0..soft.len()).step_by(20) {
            soft[i] = -soft[i];
            flipped += 1;
        }

        let decoded = decode(&soft);
        // With rate 1/6, K=7, should handle ~5% bit error rate
        let errors: usize = decoded
            .iter()
            .zip(input.iter())
            .filter(|(a, b)| a != b)
            .count();

        assert!(
            errors == 0,
            "had {errors} errors after decoding ({flipped} coded bits flipped)"
        );
    }

    #[test]
    fn encode_length() {
        let input = vec![false; TOTAL_DATA_BITS];
        let coded = encode(&input);
        assert_eq!(coded.len(), CODED_BITS);
    }

    #[test]
    fn all_zeros() {
        let input = vec![false; TOTAL_DATA_BITS];
        let coded = encode(&input);
        let soft: Vec<f32> = coded.iter().map(|&b| if b { 1.0 } else { -1.0 }).collect();
        let decoded = decode(&soft);
        assert_eq!(decoded, input);
    }

    #[test]
    fn all_ones() {
        let input = vec![true; TOTAL_DATA_BITS];
        let coded = encode(&input);
        let soft: Vec<f32> = coded.iter().map(|&b| if b { 1.0 } else { -1.0 }).collect();
        let decoded = decode(&soft);
        assert_eq!(decoded, input);
    }
}
