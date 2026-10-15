//! Exact scalar-F64 word access for selected WGSL kernels. These helpers need
//! SHADER_INT64; callers must validate the enabled device before any effect.

const HELPERS: &str = r#"
fn runmat_f64_low(value: f64) -> u32 {
    return u32(bitcast<u64>(value) & u64(0xffffffffu));
}
fn runmat_f64_high(value: f64) -> u32 {
    return u32(bitcast<u64>(value) >> 32u);
}
fn runmat_f64_from_words(low: u32, high: u32) -> f64 {
    return bitcast<f64>(u64(low) | (u64(high) << 32u));
}
"#;

pub(crate) fn inject(source: &str) -> String {
    if source.contains("fn runmat_f64_from_words(") {
        return source.to_owned();
    }
    format!("{source}\n{HELPERS}")
}

/// Exact widening of a binary32 storage word, without a floating binary32
/// intermediate (which could flush subnormals). Same enabled F64/INT64 gate
/// as the word helpers above. NaN payloads are not a language promise.
pub(crate) fn binary32_widening() -> &'static str {
    BINARY32_WIDENING
}

pub(crate) fn binary32_widening_implementation(
    output: u32,
) -> runmat_accelerate_api::scalar_program::EmittedScalarImplementation {
    use runmat_accelerate_api::scalar_program::{
        EmittedScalarImplementation, ScalarImplementation,
    };
    // Maximum named algorithmic stages: classify, CLZ, normalize, two word
    // packs, and assemble. Not a native-instruction or elapsed-time bound.
    EmittedScalarImplementation {
        output,
        implementation: ScalarImplementation::Binary32WidenToBinary64WordsV1,
        invocations_per_element: 1,
        algorithmic_units_per_invocation: 1 + 1 + 1 + 2 + 1,
    }
}

const BINARY32_WIDENING: &str = r#"
fn runmat_f32_words_to_f64(bits: u32) -> f64 {
    let sign = bits & 0x80000000u;
    let encoded = (bits >> 23u) & 255u;
    var fraction = bits & 0x7fffffu;
    if encoded == 255u {
        return runmat_f64_from_words(0u, sign | 0x7ff00000u | select(0u, 0x80000u, fraction != 0u));
    }
    if encoded == 0u && fraction == 0u { return runmat_f64_from_words(0u, sign); }
    var exponent = i32(encoded) + 896;
    if encoded == 0u {
        let highest = 31u - countLeadingZeros(fraction);
        exponent = i32(highest) + 874;
        fraction = (fraction << (23u - highest)) & 0x7fffffu;
    }
    return runmat_f64_from_words(fraction << 29u, sign | (u32(exponent) << 20u) | (fraction >> 3u));
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_word_helper_is_idempotent_and_has_no_wide_unsuffixed_literal() {
        let once = inject("fn main() {}");
        assert_eq!(inject(&once), once);
        assert!(once.contains("bitcast<u64>(value)"));
        assert!(once.contains("u64(high) << 32u"));
        assert!(!once.contains("bitcast<vec2<u32>>(value)"));
    }
}
