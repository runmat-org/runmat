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

/// Canonical declarations for the typed shader compiler. Callers identify
/// declarations through Naga spans rather than textual include markers.
pub(crate) const fn helpers() -> &'static str {
    HELPERS
}

pub(crate) fn inject(source: &str) -> String {
    if source.contains("fn runmat_f64_from_words(") {
        return source.to_owned();
    }
    format!("{source}\n{HELPERS}")
}

/// Append the same helper bytes to the owned shader without a second full
/// source allocation. `generated_from` preserves the old injection decision
/// when a caller has already prefixed a separate semantic preamble.
pub(crate) fn append_if_missing(source: &mut String, generated_from: usize) {
    if !source[generated_from..].contains("fn runmat_f64_from_words(") {
        source.push('\n');
        source.push_str(HELPERS);
    }
}

pub(crate) const fn appended_bytes() -> usize {
    1 + HELPERS.len()
}

/// Exact widening of a binary32 storage word, without a floating binary32
/// intermediate (which could flush subnormals). Same enabled F64/INT64 gate
/// as the word helpers above. NaN payloads are not a language promise.
pub(crate) fn write_binary32_widening(writer: &mut dyn std::fmt::Write) -> std::fmt::Result {
    writer.write_str(runmat_accelerate_api::wgpu_ieee_words::BINARY32_TO_BINARY64_WORDS_WGSL)?;
    writer.write_str("\n")?;
    writer.write_str(BINARY32_WIDENING)
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
    let words = runmat_f32_to_binary64_words(bits);
    return runmat_f64_from_words(words.x, words.y);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_word_helper_is_idempotent_and_has_no_wide_unsuffixed_literal() {
        let once = inject("fn main() {}");
        let mut owned = "fn main() {}".to_owned();
        append_if_missing(&mut owned, 0);
        assert_eq!(owned, once);
        let mut prefixed = format!("{HELPERS}fn main() {{}}");
        append_if_missing(&mut prefixed, HELPERS.len());
        assert_eq!(prefixed, format!("{HELPERS}{once}"));
        append_if_missing(&mut owned, 0);
        assert_eq!(owned, once);
        assert_eq!(inject(&once), once);
        assert!(once.contains("bitcast<u64>(value)"));
        assert!(once.contains("u64(high) << 32u"));
        assert!(!once.contains("bitcast<vec2<u32>>(value)"));
    }
}
