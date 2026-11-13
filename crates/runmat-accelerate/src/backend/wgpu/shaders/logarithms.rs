//! Real logarithm endpoints, shared by direct kernels and eligible fusion bodies.
use crate::backend::wgpu::types::NumericPrecision;

pub(crate) fn helpers(precision: NumericPrecision) -> String {
    let ty = precision.as_str();
    let (classification, infinity, nan, cutoff) = match precision {
        NumericPrecision::F32 => (
            "let bits = bitcast<u32>(a);\nlet magnitude = bits & 0x7fffffffu;\nlet is_nan = magnitude > 0x7f800000u;\nlet is_infinite = magnitude == 0x7f800000u;\nlet is_zero = magnitude == 0u;\nlet negative = (bits & 0x80000000u) != 0u;",
            "bitcast<f32>(infinity_bits)", "bitcast<f32>(nan_bits)", "1.0e-4",
        ),
        NumericPrecision::F64 => (
            "let low = runmat_f64_low(a);\nlet high = runmat_f64_high(a);\nlet magnitude = high & 0x7fffffffu;\nlet is_nan = magnitude > 0x7ff00000u || (magnitude == 0x7ff00000u && low != 0u);\nlet is_infinite = magnitude == 0x7ff00000u && low == 0u;\nlet is_zero = magnitude == 0u && low == 0u;\nlet negative = (high & 0x80000000u) != 0u;",
            "runmat_f64_from_words(0u, 0x7ff00000u)", "runmat_f64_from_words(0u, 0x7ff80000u)", "1.0e-6",
        ),
    };
    format!(
        r#"
fn runmat_log(a: {ty}) -> {ty} {{
    let infinity_bits: u32 = 0x7f800000u;
    let nan_bits: u32 = 0x7fc00000u;
    {classification}
    if is_nan {{ return a; }}
    if is_zero {{ return -{infinity}; }}
    if negative {{ return {nan}; }}
    if is_infinite {{ return a; }}
    return log(a);
}}
fn runmat_log1p(a: {ty}) -> {ty} {{
    let nan_bits: u32 = 0x7fc00000u;
    {classification}
    if is_nan {{ return a; }}
    if is_infinite {{
        if negative {{ return {nan}; }}
        return a;
    }}
    if abs(a) < {ty}({cutoff}) {{
        let a2 = a * a;
        let a3 = a2 * a;
        let a4 = a2 * a2;
        let a5 = a4 * a;
        let correction = a5 * {ty}(0.2) - a4 * {ty}(0.25) +
            a3 / {ty}(3.0) - a2 * {ty}(0.5);
        return a + correction;
    }}
    return runmat_log({ty}(1.0) + a);
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logarithms_classify_encodings_before_finite_math() {
        for precision in [NumericPrecision::F32, NumericPrecision::F64] {
            let source = helpers(precision);
            assert!(source.contains(match precision {
                NumericPrecision::F32 => "bitcast<u32>",
                NumericPrecision::F64 => "runmat_f64_high",
            }));
            assert!(source.find("if is_nan").unwrap() < source.find("return log(a)").unwrap());
            assert!(source.contains("if is_zero { return -"));
            assert!(source.contains("if negative { return "));
            assert!(source.contains("return runmat_log("));
        }
    }
}
