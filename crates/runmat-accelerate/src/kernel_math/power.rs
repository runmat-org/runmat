//! One WGSL numerical policy shared by standalone power and fused expressions.
use runmat_accelerate_api::ProviderPrecision;
use runmat_runtime::builtins::common::spec::{FusionMathHelper, ScalarType};
use std::fmt::Write as _;

#[path = "power/stream_replace.rs"]
mod stream_replace;
use stream_replace::ReplaceWriter;

pub(crate) fn entrypoint(precision: ProviderPrecision) -> &'static str {
    let scalar = match precision {
        ProviderPrecision::F32 => ScalarType::F32,
        ProviderPrecision::F64 => ScalarType::F64,
    };
    FusionMathHelper::RealPower
        .entrypoint(scalar)
        .expect("floating power precision")
}

#[cfg(any(feature = "wgpu", test))]
pub(crate) fn functions(precision: ProviderPrecision) -> String {
    let mut source = String::new();
    write_functions(precision, &mut source).expect("String writer is infallible");
    source
}

pub(crate) fn write_functions(
    precision: ProviderPrecision,
    writer: &mut impl std::fmt::Write,
) -> std::fmt::Result {
    let ty = match precision {
        ProviderPrecision::F32 => "f32",
        ProviderPrecision::F64 => "f64",
    };
    let bits = match precision {
        ProviderPrecision::F32 => {
            r#"
fn finite(v: f32) -> bool { return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u; }
fn nan(v: f32) -> bool { return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u; }
fn negative(v: f32) -> bool { return (bitcast<u32>(v) & 0x80000000u) != 0u; }
"#
        }
        ProviderPrecision::F64 => {
            r#"
fn finite(v: f64) -> bool { return (runmat_f64_high(v) & 0x7ff00000u) != 0x7ff00000u; }
fn nan(v: f64) -> bool {
    let low = runmat_f64_low(v);
    let high = runmat_f64_high(v);
    return (high & 0x7ff00000u) == 0x7ff00000u && ((high & 0x000fffffu) != 0u || low != 0u);
}

fn negative(v: f64) -> bool { return (runmat_f64_high(v) & 0x80000000u) != 0u; }
"#
        }
    };
    // Construct nonfinite values from runtime integer expressions. A literal
    // bitcast is a WGSL const-expression and browser validators reject its
    // nonfinite f32 result while creating the shader module.
    let (infinity, not_number) = match precision {
        ProviderPrecision::F32 => ("bitcast<f32>(infinity_bits)", "bitcast<f32>(nan_bits)"),
        ProviderPrecision::F64 => (
            "runmat_f64_from_words(0u, infinity_bits)",
            "runmat_f64_from_words(0u, nan_bits)",
        ),
    };
    let (infinity_bits, nan_bits) = match precision {
        ProviderPrecision::F32 => ("0x7f800000u", "0x7fc00000u"),
        ProviderPrecision::F64 => ("0x7ff00000u", "0x7ff80000u"),
    };
    let name = entrypoint(precision);
    let mut real_power = ReplaceWriter::new(writer, "real_power(", [name, "("]);
    let mut needs_complex =
        ReplaceWriter::new(&mut real_power, "needs_complex(", [name, "_needs_complex("]);
    let mut negative = ReplaceWriter::new(&mut needs_complex, "negative(", [name, "_negative("]);
    let mut nan = ReplaceWriter::new(&mut negative, "nan(", [name, "_nan("]);
    let mut finite = ReplaceWriter::new(&mut nan, "finite(", [name, "_finite("]);
    let mut types = ReplaceWriter::new(&mut finite, "T", [ty, ""]);
    write!(
        &mut types,
        r#"{bits}
fn needs_complex(a: T, b: T) -> bool {{
    return finite(a) && a < T(0.0) && finite(b) && trunc(b) != b;
}}
fn real_power(a: T, b: T) -> T {{
    let infinity_bits: u32 = {infinity_bits};
    let nan_bits: u32 = {nan_bits};
    let infinity = {infinity};
    let not_number = {not_number};
    if nan(a) || nan(b) {{ return not_number; }}
    if b == T(0.0) {{ return T(1.0); }}
    let magnitude = abs(a);
    if !finite(b) {{
        if magnitude == T(1.0) {{ return T(1.0); }}
        return select(T(0.0), infinity, (magnitude > T(1.0)) == (b > T(0.0)));
    }}
    let odd = trunc(b) == b && abs(b - T(2.0) * trunc(b / T(2.0))) == T(1.0);
    var result: T;
    if magnitude == T(0.0) {{ result = select(T(0.0), infinity, b < T(0.0)); }}
    else if !finite(a) {{ result = select(T(0.0), infinity, b > T(0.0)); }}
    else {{ result = pow(magnitude, b); }}
    if negative(a) && odd {{ return -result; }}
    if needs_complex(a, b) {{ return not_number; }}
    return result;
}}
"#
    )?;
    types.finish()?;
    finite.finish()?;
    nan.finish()?;
    negative.finish()?;
    needs_complex.finish()?;
    real_power.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_precisions_have_distinct_helpers_and_identical_exception_order() {
        for precision in [ProviderPrecision::F32, ProviderPrecision::F64] {
            let source = functions(precision);
            let name = entrypoint(precision);
            assert!(source.contains(&format!("fn {name}(")));
            assert!(!source.contains("#[cfg(test)]"));
            assert!(!source.contains("mod tests"));
            assert!(!source.contains("fn real_power("));
            assert!(!source.contains(&format!("{name}_{name}")));
            assert!(
                source.find(&format!("if {name}_nan(a)")).unwrap()
                    < source.find("if b ==").unwrap()
            );
            assert!(source.contains(&format!("if {name}_negative(a) && odd")));
            assert!(source.contains("let infinity_bits: u32 ="));
            assert!(source.contains("let nan_bits: u32 ="));
            assert!(!source.contains("bitcast<f32>(0x"));
            match precision {
                ProviderPrecision::F32 => {
                    assert!(source.contains("bitcast<f32>(infinity_bits)"));
                    assert!(source.contains("bitcast<f32>(nan_bits)"));
                }
                ProviderPrecision::F64 => {
                    assert!(source.contains("runmat_f64_from_words(0u, infinity_bits)"));
                    assert!(source.contains("runmat_f64_from_words(0u, nan_bits)"));
                }
            }
        }
    }
}
