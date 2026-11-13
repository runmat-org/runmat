//! Fixed ordinary real elementwise Max policy, not a caller-selected fast mode.
use crate::backend::wgpu::types::NumericPrecision;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RealMaxPolicy {
    IncludeNaNPreferPositiveZero,
}

pub(crate) fn scalar_shader(precision: NumericPrecision, policy: RealMaxPolicy) -> String {
    let (source, old, replacement, helper) = match (precision, policy) {
        (NumericPrecision::F32, RealMaxPolicy::IncludeNaNPreferPositiveZero) => (
            super::elementwise::SCALAR_SHADER_F32,
            "case 6u: { result = max(a, s); }",
            "case 6u: { result = runmat_real_max(a, s); }",
            F32_HELPER,
        ),
        (NumericPrecision::F64, RealMaxPolicy::IncludeNaNPreferPositiveZero) => (
            super::elementwise::SCALAR_SHADER_F64,
            "case 6u: { result = max(a, scalar); }",
            "case 6u: { result = runmat_real_max(a, scalar); }",
            F64_HELPER,
        ),
    };
    assert_eq!(
        source.matches(old).count(),
        1,
        "canonical scalar Max ABI changed"
    );
    let shader = format!("{}\n{helper}", source.replace(old, replacement));
    match precision {
        NumericPrecision::F32 => shader,
        NumericPrecision::F64 => super::ieee_words::inject(&shader),
    }
}

/// Specialize the actual binary and broadcast templates without duplicating
/// the scalar owner's IEEE word-order theorem or changing any other op case.
pub(crate) fn binary_shader(
    precision: NumericPrecision,
    broadcast: bool,
    policy: RealMaxPolicy,
) -> String {
    let source = match (precision, broadcast) {
        (NumericPrecision::F32, false) => super::elementwise::BINARY_SHADER_F32.to_owned(),
        (NumericPrecision::F32, true) => super::elementwise::BINARY_BROADCAST_SHADER_F32.to_owned(),
        (NumericPrecision::F64, broadcast) => super::elementwise::binary_shader_f64(
            super::elementwise::RealBinaryShaderKind::General,
            broadcast,
        ),
    };
    let (old, replacement, helper) = match (precision, policy) {
        (NumericPrecision::F32, RealMaxPolicy::IncludeNaNPreferPositiveZero) => (
            "case 7u: { return max(a, b); }",
            "case 7u: { return runmat_real_max(a, b); }",
            F32_HELPER,
        ),
        (NumericPrecision::F64, RealMaxPolicy::IncludeNaNPreferPositiveZero) => (
            "case 7u: { return max(a, b); }",
            "case 7u: { return runmat_real_max(a, b); }",
            F64_HELPER,
        ),
    };
    assert_eq!(
        source.matches(old).count(),
        1,
        "canonical binary Max ABI changed"
    );
    let shader = format!("{}\n{helper}", source.replace(old, replacement));
    match precision {
        NumericPrecision::F32 => shader,
        NumericPrecision::F64 => super::ieee_words::inject(&shader),
    }
}

pub(super) const F32_HELPER: &str = r#"
fn runmat_real_max(a: f32, b: f32) -> f32 {
    let aw = bitcast<u32>(a);
    let bw = bitcast<u32>(b);
    let am = aw & 0x7fffffffu;
    let bm = bw & 0x7fffffffu;
    if (am > 0x7f800000u) { return a; }
    if (bm > 0x7f800000u) { return b; }
    if ((am | bm) == 0u) {
        return bitcast<f32>((aw & bw) & 0x80000000u);
    }
    // Order non-NaN IEEE words, including subnormals, without a floating
    // comparison that can inherit the device's denormal arithmetic mode.
    let an = (aw & 0x80000000u) != 0u;
    let bn = (bw & 0x80000000u) != 0u;
    if (an != bn) {
        if (!bn) { return b; }
        return a;
    }
    if ((!an && (bm > am)) || (an && (bm < am))) { return b; }
    return a;
}
"#;

pub(super) const F64_HELPER: &str = r#"
fn runmat_real_max(a: f64, b: f64) -> f64 {
    let al = runmat_f64_low(a);
    let ah = runmat_f64_high(a);
    let bl = runmat_f64_low(b);
    let bh = runmat_f64_high(b);
    let am = ah & 0x7fffffffu;
    let bm = bh & 0x7fffffffu;
    if ((am > 0x7ff00000u) || ((am == 0x7ff00000u) && (al != 0u))) { return a; }
    if ((bm > 0x7ff00000u) || ((bm == 0x7ff00000u) && (bl != 0u))) { return b; }
    if ((am | al | bm | bl) == 0u) {
        return runmat_f64_from_words(0u, (ah & bh) & 0x80000000u);
    }
    let an = (ah & 0x80000000u) != 0u;
    let bn = (bh & 0x80000000u) != 0u;
    if (an != bn) {
        if (!bn) { return b; }
        return a;
    }
    let larger = (bm > am) || ((bm == am) && (bl > al));
    let smaller = (bm < am) || ((bm == am) && (bl < al));
    if ((!an && larger) || (an && smaller)) { return b; }
    return a;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::wgpu::shaders::validation_tests::{validate_generated, RequiredCaps};

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn generated_scalar_max_uses_exact_ieee_policy_and_existing_abi() {
        for (precision, caps) in [
            (NumericPrecision::F32, RequiredCaps::None),
            (NumericPrecision::F64, RequiredCaps::Float64Int64),
        ] {
            let shader = scalar_shader(precision, RealMaxPolicy::IncludeNaNPreferPositiveZero);
            assert!(shader.contains("fn runmat_real_max("));
            assert!(!shader.contains("case 6u: { result = max("));
            assert!(shader.contains("params.offset + local"));
            assert!(shader.contains("idx >= params.total"));
            assert!(shader.contains("case 0u: { result = a +"));
            validate_generated("ordinary scalar Max", &shader, 512, 8, caps);
        }
        // The unchanged arithmetic bundle remains valid without INT64.
        validate_generated(
            "ordinary Double scalar arithmetic",
            super::super::elementwise::SCALAR_SHADER_F64,
            512,
            8,
            RequiredCaps::Float64,
        );
    }

    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn generated_binary_max_same_and_broadcast_use_shared_ieee_policy() {
        for (precision, caps) in [
            (NumericPrecision::F32, RequiredCaps::None),
            (NumericPrecision::F64, RequiredCaps::Float64Int64),
        ] {
            for broadcast in [false, true] {
                let shader = binary_shader(
                    precision,
                    broadcast,
                    RealMaxPolicy::IncludeNaNPreferPositiveZero,
                );
                assert_eq!(shader.matches("fn runmat_real_max(").count(), 1);
                assert!(!shader.contains("case 7u: { return max(a, b); }"));
                assert!(shader.contains("case 7u: { return runmat_real_max(a, b); }"));
                assert!(shader.contains("case 0u: { return a + b; }"));
                assert!(shader.contains("params.offset + local"));
                assert!(shader.contains("@compute @workgroup_size(512)"));
                validate_generated(
                    &format!("ordinary binary Max {precision:?} broadcast={broadcast}"),
                    &shader,
                    512,
                    8,
                    caps,
                );
            }
        }
    }
}
