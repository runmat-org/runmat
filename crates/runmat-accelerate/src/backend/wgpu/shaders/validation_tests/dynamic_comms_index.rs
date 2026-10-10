//! Validate the production communication and index shader builders.

use std::collections::HashSet;

use super::{validate_generated, RequiredCaps};
use crate::backend::wgpu::shaders::{comms, ind2sub, sub2ind};
use crate::backend::wgpu::types::NumericPrecision;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn modulation_variants_validate_with_finite_word_capability() {
    let mut labels = HashSet::new();
    for precision in [NumericPrecision::F32, NumericPrecision::F64] {
        let caps = match precision {
            NumericPrecision::F32 => RequiredCaps::None,
            NumericPrecision::F64 => RequiredCaps::Float64Int64,
        };
        for order in [4, 16] {
            for workgroup in [64, 128] {
                for (family, source) in [
                    (
                        "symbols",
                        comms::modulate_constellation_shader(precision, order, workgroup),
                    ),
                    (
                        "bits",
                        comms::modulate_bits_constellation_shader(precision, order, workgroup),
                    ),
                ] {
                    let label = format!("comms-{family}-{precision:?}-order{order}-wg{workgroup}");
                    assert!(
                        labels.insert(label.clone()),
                        "duplicate shader case {label}"
                    );
                    validate_generated(&label, &source, workgroup, 8, caps);
                }
            }
        }
    }
    assert_eq!(labels.len(), 2 * 2 * 2 * 2);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn index_builder_variants_validate_with_finite_word_capability() {
    let mut labels = HashSet::new();
    for precision in [NumericPrecision::F32, NumericPrecision::F64] {
        let caps = match precision {
            NumericPrecision::F32 => RequiredCaps::None,
            NumericPrecision::F64 => RequiredCaps::Float64Int64,
        };
        for (case, dims, strides, scalar_mask) in [
            ("vector", vec![7], vec![1], vec![0]),
            ("matrix-scalar-row", vec![3, 5], vec![1, 3], vec![1, 0]),
        ] {
            for workgroup in [64, 128] {
                for (family, source) in [
                    (
                        "sub2ind",
                        sub2ind::build_sub2ind_shader(
                            precision,
                            &dims,
                            &strides,
                            &scalar_mask,
                            workgroup,
                        ),
                    ),
                    (
                        "ind2sub",
                        ind2sub::build_ind2sub_shader(
                            precision,
                            &dims,
                            &strides,
                            dims.iter().product(),
                            workgroup,
                        ),
                    ),
                ] {
                    let label = format!("{family}-{precision:?}-{case}-wg{workgroup}");
                    assert!(
                        labels.insert(label.clone()),
                        "duplicate shader case {label}"
                    );
                    validate_generated(&label, &source, workgroup, 8, caps);
                }
            }
        }
    }
    assert_eq!(labels.len(), 2 * 2 * 2 * 2);
}
