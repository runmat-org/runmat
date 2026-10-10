//! Solve and stochastic kernels not included in the static matrix cohort.

use super::{validate_generated, RequiredCaps};
use crate::backend::wgpu::shaders::{qr_power_iter, solve, stochastic_evolution};
use crate::backend::wgpu::types::NumericPrecision;
use std::collections::BTreeSet;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn triangular_solve_selected_source_forms_validate() {
    let mut labels = BTreeSet::new();
    for (precision, caps) in [
        (NumericPrecision::F32, RequiredCaps::None),
        (NumericPrecision::F64, RequiredCaps::Float64),
    ] {
        for workgroup in [128, 256, 512] {
            for transpose in [false, true] {
                for lower in [false, true] {
                    let label =
                    format!("triangular-solve-{precision:?}-wg{workgroup}-transpose{transpose}-lower{lower}");
                    assert!(labels.insert(label.clone()));
                    validate_generated(
                        &label,
                        &solve::triangular_linsolve_shader(precision, transpose, lower, workgroup),
                        workgroup,
                        8,
                        caps,
                    );
                }
            }
        }
    }
    assert_eq!(labels.len(), 24);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn qr_cholesky_and_stochastic_evolution_validate() {
    validate_generated(
        "qr-power-iter-cholesky-f32",
        qr_power_iter::QR_POWER_ITER_CHOL_SHADER,
        1,
        8,
        RequiredCaps::None,
    );
    for workgroup in [64, 128, 256] {
        for (precision, source, caps) in [
            (
                "f32",
                stochastic_evolution::STOCHASTIC_EVOLUTION_SHADER_F32,
                RequiredCaps::None,
            ),
            (
                "f64",
                stochastic_evolution::STOCHASTIC_EVOLUTION_SHADER_F64,
                RequiredCaps::Float64,
            ),
        ] {
            validate_generated(
                &format!("stochastic-evolution-{precision}-wg{workgroup}"),
                source,
                workgroup,
                8,
                caps,
            );
        }
    }
}
