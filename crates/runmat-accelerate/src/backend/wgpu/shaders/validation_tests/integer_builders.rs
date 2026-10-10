//! Production integer builders, including their selected floating index classes.

use super::{validate_generated, RequiredCaps};
use crate::backend::wgpu::shaders::integer;
use crate::backend::wgpu::types::NumericPrecision;
use std::collections::BTreeSet;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn integer_builders_validate_selected_workgroups_and_output_classes() {
    let mut labels = BTreeSet::new();
    for workgroup in [64, 128, 256] {
        let basic = [
            (
                "minmax",
                integer::minmax_shader(workgroup),
                RequiredCaps::None,
            ),
            (
                "arithmetic",
                integer::arithmetic_shader(workgroup),
                RequiredCaps::None,
            ),
            (
                "reduce-dim",
                integer::reduce_dim_shader(workgroup),
                RequiredCaps::None,
            ),
            (
                "mean-dim",
                integer::mean_dim_shader(workgroup),
                RequiredCaps::None,
            ),
            (
                "floating-sum-cast",
                integer::floating_sum_cast_shader(workgroup),
                RequiredCaps::Float64,
            ),
        ];
        for (name, source, caps) in basic {
            let label = format!("integer-{name}-wg{workgroup}");
            assert!(labels.insert(label.clone()));
            validate_generated(&label, &source, workgroup, 8, caps);
        }
        for (precision, scalar, plain_caps, cast_caps) in [
            (
                NumericPrecision::F32,
                "f32",
                RequiredCaps::None,
                RequiredCaps::None,
            ),
            (
                NumericPrecision::F64,
                "f64",
                RequiredCaps::Float64,
                RequiredCaps::Float64Int64,
            ),
        ] {
            let typed = [
                (
                    "comparison",
                    integer::comparison_shader(scalar, workgroup),
                    plain_caps,
                ),
                (
                    "extrema-dim",
                    integer::extrema_dim_shader(scalar, workgroup),
                    plain_caps,
                ),
                (
                    "cumulative-scan",
                    integer::cumulative_scan_shader(scalar, workgroup),
                    plain_caps,
                ),
                (
                    "cast",
                    integer::cast_shader(precision, workgroup),
                    cast_caps,
                ),
            ];
            for (name, source, caps) in typed {
                let label = format!("integer-{name}-{scalar}-wg{workgroup}");
                assert!(labels.insert(label.clone()));
                validate_generated(&label, &source, workgroup, 8, caps);
            }
        }
    }
    assert_eq!(labels.len(), 3 * (5 + 2 * 4));
}
