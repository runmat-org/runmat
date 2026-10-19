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
    use runmat_accelerate_api::scalar_program::ScalarLiteral;
    use runmat_accelerate_api::stochastic_evolution::StochasticEvolutionArithmetic;
    use runmat_types::NumericClass;
    for workgroup in [64, 128, 256] {
        for state in [NumericClass::Single, NumericClass::Double] {
            for noise in [NumericClass::Single, NumericClass::Double] {
                for drift in [NumericClass::Single, NumericClass::Double] {
                    for scale in [NumericClass::Single, NumericClass::Double] {
                        let literal = |class| match class {
                            NumericClass::Single => ScalarLiteral::F32Bits(0),
                            NumericClass::Double => ScalarLiteral::F64Bits(0),
                            _ => unreachable!(),
                        };
                        let arithmetic = StochasticEvolutionArithmetic::new(
                            state,
                            noise,
                            literal(drift),
                            literal(scale),
                        )
                        .unwrap();
                        let source =
                            stochastic_evolution::source(arithmetic.class_identity()).unwrap();
                        let caps = if arithmetic.requires_f64() {
                            RequiredCaps::Float64
                        } else {
                            RequiredCaps::None
                        };
                        validate_generated(
                &format!("stochastic-evolution-{state:?}-{noise:?}-{drift:?}-{scale:?}-wg{workgroup}"),
                &source,
                workgroup,
                8,
                caps,
            );
                    }
                }
            }
        }
    }
}
