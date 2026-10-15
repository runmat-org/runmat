//! Shared word-only certified POWER math, independent of indexing/dispatch.
//! A finite binary32 square is exact in binary64, then rounded once to F32.
//! This is not general binary64 emulation.
use runmat_accelerate_api::ExactSingleBasePower;

pub(crate) fn implementation(
    operation: ExactSingleBasePower,
    output: u32,
) -> runmat_accelerate_api::scalar_program::EmittedScalarImplementation {
    use runmat_accelerate_api::scalar_program::{
        EmittedScalarImplementation, ScalarImplementation,
    };
    let (implementation, units) = match operation {
        // Same four-limb/CLZ/round48/classify/assemble/normalize/pack stages as
        // the general product. This is a structural prior, not measured time.
        ExactSingleBasePower::Square => (
            ScalarImplementation::Binary32Square48RneV1,
            super::product_exact::work_prior().algorithmic_units(),
        ),
        // Classify and select/pack; no native pow or floating conversion.
        ExactSingleBasePower::Identity => (ScalarImplementation::Binary32IdentityWordsV1, 1 + 1),
        ExactSingleBasePower::Zero => (ScalarImplementation::Binary32ZeroPowerWordsV1, 1 + 1),
    };
    EmittedScalarImplementation {
        output,
        implementation,
        invocations_per_element: 1,
        algorithmic_units_per_invocation: units,
    }
}

pub(crate) fn entrypoint(operation: ExactSingleBasePower) -> &'static str {
    match operation {
        ExactSingleBasePower::Zero => "rm_exact_power_zero_bits",
        ExactSingleBasePower::Identity => "rm_exact_power_identity_bits",
        ExactSingleBasePower::Square => "rm_exact_power_square_bits",
    }
}

pub(crate) fn functions() -> &'static str {
    FUNCTIONS
}

const FUNCTIONS: &str = r#"
// Shift the 48-bit integer right and round nearest, ties to even. All callers
// use shifts in 1..=48; split cases avoid undefined/modulo-32 shifts.
fn rm_exact_power_round48(low: u32, high: u32, shift: u32) -> u32 {
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

fn rm_exact_power_square_raw_bits(bits: u32) -> u32 {
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
        return rm_exact_power_round48(low, high, u32(shift));
    }
    var rounded = rm_exact_power_round48(low, high, highest - 23u);
    if rounded == 0x1000000u { rounded = rounded >> 1u; exponent = exponent + 1; }
    if exponent > 127 { return 0x7f800000u; }
    return (u32(exponent + 127) << 23u) | (rounded & 0x7fffffu);
}
fn rm_exact_power_zero_bits(bits: u32) -> u32 {
    return select(0x3f800000u, 0x7fc00000u, (bits & 0x7fffffffu) > 0x7f800000u);
}
fn rm_exact_power_identity_bits(bits: u32) -> u32 {
    return select(bits, 0x7fc00000u, (bits & 0x7fffffffu) > 0x7f800000u);
}
fn rm_exact_power_square_bits(bits: u32) -> u32 {
    return rm_exact_power_square_raw_bits(bits);
}
"#;
