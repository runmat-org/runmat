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
