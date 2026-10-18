//! One ties-away rounding implementation for direct and fused WGSL execution.
use runmat_accelerate_api::ProviderPrecision;
use runmat_runtime::builtins::common::spec::{FusionMathHelper, ScalarType};

#[cfg(feature = "wgpu")]
mod digits;

#[cfg(feature = "wgpu")]
pub(crate) fn digit_functions(precision: ProviderPrecision) -> String {
    functions(precision) + &digits::functions(precision)
}

pub(crate) fn entrypoint(precision: ProviderPrecision) -> &'static str {
    FusionMathHelper::Round
        .entrypoint(match precision {
            ProviderPrecision::F32 => ScalarType::F32,
            ProviderPrecision::F64 => ScalarType::F64,
        })
        .expect("floating rounding precision")
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
    let (ty, classification) = match precision {
        ProviderPrecision::F32 => ("f32", "let bits = bitcast<u32>(a);\nlet special = (bits & 0x7f800000u) == 0x7f800000u;\nlet zero = (bits & 0x7fffffffu) == 0u;\nlet negative = (bits & 0x80000000u) != 0u;"),
        ProviderPrecision::F64 => ("f64", "let low = runmat_f64_low(a);\nlet high = runmat_f64_high(a);\nlet special = (high & 0x7ff00000u) == 0x7ff00000u;\nlet zero = (high & 0x7fffffffu) == 0u && low == 0u;\nlet negative = (high & 0x80000000u) != 0u;"),
    };
    let name = entrypoint(precision);
    write!(
        writer,
        r#"
fn {name}(a: {ty}) -> {ty} {{
    {classification}
    if special || zero {{ return a; }}
    let magnitude = abs(a);
    let integral = trunc(magnitude);
    // Test the represented fraction rather than adding 0.5 to the input:
    // that addition changes already-integral values near the precision limit.
    let rounded = integral + select({ty}(0.0), {ty}(1.0), magnitude - integral >= {ty}(0.5));
    if negative {{ return -rounded; }}
    return rounded;
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn both_precisions_share_fraction_test_and_preserve_encoded_specials() {
        for precision in [ProviderPrecision::F32, ProviderPrecision::F64] {
            let source = functions(precision);
            assert!(source.contains(&format!("fn {}(", entrypoint(precision))));
            assert!(source.contains("if special || zero { return a; }"));
            assert!(source.contains("magnitude - integral >="));
            assert!(!source.contains("a + 0.5"));
        }
    }
}
