//! Certified Double-scalar POWER on binary32 storage, without float arithmetic.
//! Not a binary64 emulator. A binary32 square has at most 48 significant bits;
//! the exact product is also the binary64 product, then rounded once to F32.
use runmat_accelerate_api::ExactSingleBasePower;

pub(crate) fn shader(operation: ExactSingleBasePower) -> String {
    let body = match operation {
        ExactSingleBasePower::Zero => "return 0x3f800000u;",
        ExactSingleBasePower::Identity => "return bits;",
        ExactSingleBasePower::Square => "return square_bits(bits);",
    };
    format!(
        "{HELPERS}\nfn apply(bits: u32) -> u32 {{\n\
        if (bits & 0x7fffffffu) > 0x7f800000u {{ return 0x7fc00000u; }}\n\
        {body}\n}}\n{ENTRY}"
    )
}

const ENTRY: &str = r#"
struct Tensor { data: array<u32> };
struct Params { len: u32, offset: u32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> A: Tensor;
@group(0) @binding(1) var<storage, read_write> Out: Tensor;
@group(0) @binding(2) var<uniform> params: Params;
@compute @workgroup_size(@WG@)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.len { return; }
    let idx = params.offset + gid.x;
    Out.data[idx] = apply(A.data[idx]);
}
"#;

const HELPERS: &str = r#"
// Shift the 48-bit integer right and round nearest, ties to even. All callers
// use shifts in 1..=48; split cases avoid undefined/modulo-32 shifts.
fn round48(low: u32, high: u32, shift: u32) -> u32 {
    var quotient: u32;
    var guard: u32;
    var sticky: bool;
    if shift < 32u {
        quotient = (low >> shift) | (high << (32u - shift));
        guard = (low >> (shift - 1u)) & 1u;
        sticky = (low & ((1u << (shift - 1u)) - 1u)) != 0u;
    } else if shift == 32u {
        quotient = high;
        guard = low >> 31u;
        sticky = (low & 0x7fffffffu) != 0u;
    } else {
        quotient = high >> (shift - 32u);
        guard = (high >> (shift - 33u)) & 1u;
        sticky = low != 0u || (high & ((1u << (shift - 33u)) - 1u)) != 0u;
    }
    return quotient + select(0u, 1u, guard != 0u && (sticky || (quotient & 1u) != 0u));
}

fn square_bits(bits: u32) -> u32 {
    let absolute = bits & 0x7fffffffu;
    if absolute == 0u { return 0u; }
    if absolute >= 0x7f800000u { return select(0x7f800000u, 0x7fc00000u, absolute > 0x7f800000u); }
    let encoded = (absolute >> 23u) & 255u;
    let significand = (absolute & 0x7fffffu) | select(0u, 0x800000u, encoded != 0u);
    let bottom = significand & 0xffffu;
    let top = significand >> 16u;
    let first = bottom * bottom;
    let cross = 2u * bottom * top;
    let low = first + (cross << 16u);
    let high = top * top + (cross >> 16u) + select(0u, 1u, low < first);
    var highest: u32;
    if high != 0u { highest = 63u - countLeadingZeros(high); }
    else { highest = 31u - countLeadingZeros(low); }
    let scale = 2 * (i32(max(encoded, 1u)) - 150);
    var exponent = i32(highest) + scale;
    if exponent > 127 { return 0x7f800000u; }
    if exponent < -126 {
        let shift = -149 - scale;
        if shift > 48 { return 0u; }
        // Rounded subnormal significand; 0x800000 naturally encodes min-normal.
        return round48(low, high, u32(shift));
    }
    var rounded = round48(low, high, highest - 23u);
    if rounded == 0x1000000u { rounded = rounded >> 1u; exponent = exponent + 1; }
    if exponent > 127 { return 0x7f800000u; }
    return (u32(exponent + 127) << 23u) | (rounded & 0x7fffffu);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::wgpu::shaders::validation_tests::{validate_generated, RequiredCaps};

    #[test]
    fn exact_scalar_power_uses_only_baseline_integer_words() {
        for operation in [
            ExactSingleBasePower::Zero,
            ExactSingleBasePower::Identity,
            ExactSingleBasePower::Square,
        ] {
            let source = shader(operation);
            assert!(!source.contains("f32"));
            assert!(!source.contains("f64"));
            assert!(!source.contains("u64"));
            validate_generated(
                "exact-single-base-power",
                &source,
                64,
                8,
                RequiredCaps::None,
            );
        }
    }
}
