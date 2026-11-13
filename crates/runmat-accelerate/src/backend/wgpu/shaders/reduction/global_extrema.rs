//! The global Min/Max tree shares ordinary real Max's IEEE ordering. Sum,
//! product, and count-nonzero retain their existing arithmetic branches.

use crate::backend::wgpu::types::NumericPrecision;

pub(super) fn source(precision: NumericPrecision) -> String {
    let (template, max_helper, min_helper) = match precision {
        NumericPrecision::F32 => (
            super::REDUCE_GLOBAL_SHADER_F32,
            super::super::real_max::F32_HELPER,
            F32_MIN_HELPER,
        ),
        NumericPrecision::F64 => (
            super::REDUCE_GLOBAL_SHADER_F64,
            super::super::real_max::F64_HELPER,
            F64_MIN_HELPER,
        ),
    };
    let selected = format!("{template}\n{max_helper}\n{min_helper}");
    // This single injector supplies both the NaN predicate and, for Double,
    // the same enabled-feature-gated IEEE-word ABI used by ordinary Max.
    super::super::nan_predicate::inject(&selected, precision)
}

const F32_MIN_HELPER: &str = r#"
fn runmat_real_min(a: f32, b: f32) -> f32 {
    let aw = bitcast<u32>(a);
    let bw = bitcast<u32>(b);
    let am = aw & 0x7fffffffu;
    let bm = bw & 0x7fffffffu;
    if (am > 0x7f800000u) { return a; }
    if (bm > 0x7f800000u) { return b; }
    if ((am | bm) == 0u) {
        return bitcast<f32>((aw | bw) & 0x80000000u);
    }
    let an = (aw & 0x80000000u) != 0u;
    let bn = (bw & 0x80000000u) != 0u;
    if (an != bn) {
        if (bn) { return b; }
        return a;
    }
    if ((!an && (bm < am)) || (an && (bm > am))) { return b; }
    return a;
}
"#;

const F64_MIN_HELPER: &str = r#"
fn runmat_real_min(a: f64, b: f64) -> f64 {
    let al = runmat_f64_low(a);
    let ah = runmat_f64_high(a);
    let bl = runmat_f64_low(b);
    let bh = runmat_f64_high(b);
    let am = ah & 0x7fffffffu;
    let bm = bh & 0x7fffffffu;
    if ((am > 0x7ff00000u) || ((am == 0x7ff00000u) && (al != 0u))) { return a; }
    if ((bm > 0x7ff00000u) || ((bm == 0x7ff00000u) && (bl != 0u))) { return b; }
    if ((am | al | bm | bl) == 0u) {
        return runmat_f64_from_words(0u, (ah | bh) & 0x80000000u);
    }
    let an = (ah & 0x80000000u) != 0u;
    let bn = (bh & 0x80000000u) != 0u;
    if (an != bn) {
        if (bn) { return b; }
        return a;
    }
    let smaller = (bm < am) || ((bm == am) && (bl < al));
    let larger = (bm > am) || ((bm == am) && (bl > al));
    if ((!an && smaller) || (an && larger)) { return b; }
    return a;
}
"#;

#[cfg(test)]
#[path = "global_extrema/validation_tests.rs"]
mod validation_tests;
