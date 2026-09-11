use super::parse::parse_float;

#[test]
fn rejects_float_indices_at_or_beyond_the_platform_limit() {
    assert!(parse_float(usize::MAX as f64).is_err());
    assert!(parse_float(usize::MAX as f64 + 1.0).is_err());
}
