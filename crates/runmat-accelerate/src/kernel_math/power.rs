//! One WGSL numerical policy shared by standalone power and fused expressions.
use runmat_accelerate_api::ProviderPrecision;
use runmat_runtime::builtins::common::spec::{FusionMathHelper, ScalarType};

pub(crate) fn entrypoint(precision: ProviderPrecision) -> &'static str {
    let scalar = match precision {
        ProviderPrecision::F32 => ScalarType::F32,
        ProviderPrecision::F64 => ScalarType::F64,
    };
    FusionMathHelper::RealPower
        .entrypoint(scalar)
        .expect("floating power precision")
}

pub(crate) fn functions(precision: ProviderPrecision) -> String {
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
fn finite(v: f64) -> bool { return (bitcast<vec2<u32>>(v).y & 0x7ff00000u) != 0x7ff00000u; }
fn nan(v: f64) -> bool {
    let b = bitcast<vec2<u32>>(v);
    return (b.y & 0x7ff00000u) == 0x7ff00000u && ((b.y & 0x000fffffu) != 0u || b.x != 0u);
}

fn negative(v: f64) -> bool { return (bitcast<vec2<u32>>(v).y & 0x80000000u) != 0u; }
"#
        }
    };
    let source = format!(
        r#"{bits}
fn needs_complex(a: T, b: T) -> bool {{
    return finite(a) && a < T(0.0) && finite(b) && trunc(b) != b;
}}
fn real_power(a: T, b: T) -> T {{
    let infinity = T(bitcast<f32>(0x7f800000u));
    let not_number = T(bitcast<f32>(0x7fc00000u));
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
    );
    let name = entrypoint(precision);
    source
        .replace("T", ty)
        .replace("finite(", &format!("{name}_finite("))
        .replace("nan(", &format!("{name}_nan("))
        .replace("negative(", &format!("{name}_negative("))
        .replace("needs_complex(", &format!("{name}_needs_complex("))
        .replace("real_power(", &format!("{name}("))
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
        }
    }
}
