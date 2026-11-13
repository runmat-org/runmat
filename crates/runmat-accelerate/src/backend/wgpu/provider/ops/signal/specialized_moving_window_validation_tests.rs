//! Validate the specialized production builder selected by the moving-window hook.

use super::*;
use crate::backend::wgpu::shaders::validation_tests::{validate_generated, RequiredCaps};
use std::collections::BTreeSet;

const OPERATIONS: [ProviderMovingWindowOp; 8] = [
    ProviderMovingWindowOp::Sum,
    ProviderMovingWindowOp::Mean,
    ProviderMovingWindowOp::Prod,
    ProviderMovingWindowOp::Min,
    ProviderMovingWindowOp::Max,
    ProviderMovingWindowOp::Std,
    ProviderMovingWindowOp::Var,
    ProviderMovingWindowOp::Median,
];

fn default_fill(op: ProviderMovingWindowOp) -> f64 {
    match op {
        ProviderMovingWindowOp::Min => f64::INFINITY,
        ProviderMovingWindowOp::Max => f64::NEG_INFINITY,
        _ => f64::NAN,
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn specialized_moving_window_source_variants_validate_with_exact_caps() {
    let mut cases = BTreeSet::new();
    for precision in [NumericPrecision::F32, NumericPrecision::F64] {
        let caps = match precision {
            NumericPrecision::F32 => RequiredCaps::None,
            NumericPrecision::F64 => RequiredCaps::Float64Int64,
        };
        for op in OPERATIONS {
            let normalizations: &[ProviderStdNormalization] = if matches!(
                op,
                ProviderMovingWindowOp::Std | ProviderMovingWindowOp::Var
            ) {
                &[
                    ProviderStdNormalization::Sample,
                    ProviderStdNormalization::Population,
                ]
            } else {
                &[ProviderStdNormalization::Sample]
            };
            for nan_mode in [ProviderNanMode::Include, ProviderNanMode::Omit] {
                for (endpoint_name, endpoints) in [
                    ("shrink", ProviderMovingWindowEndpoints::Shrink),
                    ("discard", ProviderMovingWindowEndpoints::Discard),
                    ("fill-finite", ProviderMovingWindowEndpoints::Fill(-0.5)),
                    (
                        "fill-default",
                        ProviderMovingWindowEndpoints::Fill(default_fill(op)),
                    ),
                ] {
                    for &normalization in normalizations {
                        let out_axis_len =
                            if matches!(endpoints, ProviderMovingWindowEndpoints::Discard) {
                                2
                            } else {
                                4
                            };
                        let fill_value = moving_window_endpoint_code(endpoints).1;
                        let spec = MovingWindowShaderSpec {
                            precision,
                            output_len: 3 * out_axis_len * 2,
                            axis_len: 4,
                            out_axis_len,
                            stride_before: 3,
                            before: 1,
                            after: 1,
                            op,
                            endpoints,
                            nan_mode,
                            normalization,
                            fill_value,
                        };
                        let name = format!(
                            "moving-specialized-{precision:?}-{op:?}-{nan_mode:?}-{endpoint_name}-{normalization:?}"
                        );
                        assert!(cases.insert(name.clone()), "duplicate {name}");
                        validate_generated(
                            &name,
                            &build_specialized_moving_window_shader(&spec),
                            crate::backend::wgpu::config::WORKGROUP_SIZE,
                            8,
                            caps,
                        );
                    }
                }
            }
        }
    }
    assert_eq!(
        cases.len(),
        160,
        "specialized moving-window case inventory changed"
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn specialized_fill_preserves_signed_zero_and_subnormal_bits() {
    for (precision, values, caps) in [
        (
            NumericPrecision::F32,
            [-0.0, f32::from_bits(1) as f64],
            RequiredCaps::None,
        ),
        (
            NumericPrecision::F64,
            [-0.0, f64::from_bits(1)],
            RequiredCaps::Float64Int64,
        ),
    ] {
        for (index, fill_value) in values.into_iter().enumerate() {
            let spec = MovingWindowShaderSpec {
                precision,
                output_len: 24,
                axis_len: 4,
                out_axis_len: 4,
                stride_before: 3,
                before: 1,
                after: 1,
                op: ProviderMovingWindowOp::Sum,
                endpoints: ProviderMovingWindowEndpoints::Fill(fill_value),
                nan_mode: ProviderNanMode::Include,
                normalization: ProviderStdNormalization::Sample,
                fill_value,
            };
            let source = build_specialized_moving_window_shader(&spec);
            let expected_bits = match (precision, index) {
                (NumericPrecision::F32, 0) => "var bits: u32 = 0x80000000u",
                (NumericPrecision::F32, 1) => "var bits: u32 = 0x00000001u",
                (NumericPrecision::F64, 0) => {
                    "var low: u32 = 0x00000000u; var high: u32 = 0x80000000u"
                }
                (NumericPrecision::F64, 1) => {
                    "var low: u32 = 0x00000001u; var high: u32 = 0x00000000u"
                }
                _ => unreachable!("two fill-bit cases per precision"),
            };
            assert!(
                source.contains(expected_bits),
                "fill bits changed for {precision:?} case {index}"
            );
            validate_generated(
                &format!("moving-specialized-fill-bits-{precision:?}-{index}"),
                &source,
                crate::backend::wgpu::config::WORKGROUP_SIZE,
                8,
                caps,
            );
        }
    }
}
