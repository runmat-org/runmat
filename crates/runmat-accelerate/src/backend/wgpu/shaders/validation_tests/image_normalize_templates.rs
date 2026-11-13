//! Validate the non-Windows production image-normalize template after the same
//! substitutions used by bootstrap and lazily tuned pipeline construction.
//! This is shader syntax/typing coverage, not staged ImageNormalize admission.

use super::{validate_generated, RequiredCaps};
use crate::backend::wgpu::shaders::{image_normalize, image_normalize_stub};
use crate::backend::wgpu::types::NumericPrecision;
use std::collections::BTreeSet;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn image_normalize_production_templates_validate_bootstrap_and_tuned_geometries() {
    let mut labels = BTreeSet::new();
    // The first row is the bootstrap requested by provider::init. The other
    // rows exercise the lane/spatial/batch/value axes of selected tuning.
    let geometries = [(1, 1, 32, 1), (4, 2, 64, 2), (16, 4, 128, 4)];
    for (batch_tile, values_per_thread, lane_count, spatial_tile) in geometries {
        let geometry = image_normalize::ImageNormalizeShaderGeometry {
            batch_tile,
            values_per_thread,
            lane_count,
            spatial_tile,
        };
        for (precision, vector_width, caps) in [
            (NumericPrecision::F32, 4, RequiredCaps::None),
            (NumericPrecision::F64, 2, RequiredCaps::Float64),
        ] {
            let label = format!(
                "image-normalize-{precision:?}-bt{batch_tile}-vp{values_per_thread}-wg{lane_count}-st{spatial_tile}"
            );
            assert!(labels.insert(label.clone()));
            let source = image_normalize::materialize(precision, geometry);
            assert!(source.contains(&format!("const BATCH_TILE: u32 = {batch_tile}u;")));
            assert!(source.contains(&format!(
                "const VALUES_PER_THREAD: u32 = {values_per_thread}u;"
            )));
            assert!(source.contains(&format!("const LANE_COUNT: u32 = {lane_count}u;")));
            assert!(source.contains(&format!("const SPATIAL_TILE: u32 = {spatial_tile}u;")));
            assert!(source.contains(&format!("const BATCH_VEC_WIDTH: u32 = {vector_width}u;")));
            assert!(!source.contains("@BT@"));
            assert!(!source.contains("@VP@"));
            assert!(!source.contains("@BV@"));
            assert!(!source.contains("@ST@"));
            validate_generated(&label, &source, lane_count, 8, caps);
            // Syntax validation alone cannot prove the pipeline layout admits
            // the shader. Reflect the exact production source against the same
            // entries used by bootstrap and reused by tuned pipelines.
            let module = naga::front::wgsl::parse_str(&source).unwrap();
            let entries = crate::backend::wgpu::bindings::image_normalize_entries();
            let mut reflected = BTreeSet::new();
            for (_, global) in module.global_variables.iter() {
                let Some(binding) = &global.binding else {
                    continue;
                };
                assert_eq!(binding.group, 0, "{label}");
                assert!(
                    reflected.insert(binding.binding),
                    "{label}: duplicate binding"
                );
                let entry = entries
                    .iter()
                    .find(|entry| entry.binding == binding.binding)
                    .unwrap_or_else(|| panic!("{label}: missing binding {}", binding.binding));
                let expected = match global.space {
                    naga::AddressSpace::Uniform => wgpu::BufferBindingType::Uniform,
                    naga::AddressSpace::Storage { access } => wgpu::BufferBindingType::Storage {
                        read_only: !access.contains(naga::StorageAccess::STORE),
                    },
                    other => panic!("{label}: unexpected resource space {other:?}"),
                };
                assert_eq!(
                    entry.ty,
                    wgpu::BindingType::Buffer {
                        ty: expected,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    "{label}: binding {}",
                    binding.binding
                );
                assert_eq!(entry.visibility, wgpu::ShaderStages::COMPUTE);
            }
            assert_eq!(reflected, BTreeSet::from([0, 1, 2, 3, 4]), "{label}");
            assert_eq!(entries.len(), reflected.len(), "{label}");
        }
    }
    assert_eq!(labels.len(), 6);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn windows_image_normalize_bootstrap_stubs_validate_without_execution_credit() {
    // Windows provider initialization compiles these one-workgroup bootstrap
    // modules before the adapter-tuned numerical pipeline is available. They
    // do not write output and cannot certify the numerical runtime route.
    for (label, source, caps) in [
        (
            "image-normalize-windows-bootstrap-f32",
            image_normalize_stub::IMAGE_NORMALIZE_STUB_SHADER_F32,
            RequiredCaps::None,
        ),
        (
            "image-normalize-windows-bootstrap-f64",
            image_normalize_stub::IMAGE_NORMALIZE_STUB_SHADER_F64,
            RequiredCaps::Float64,
        ),
    ] {
        validate_generated(label, source, 1, 8, caps);
    }
}
