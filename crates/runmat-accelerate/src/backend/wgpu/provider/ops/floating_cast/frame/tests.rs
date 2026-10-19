use super::*;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn physical_class_and_logical_producer_are_admitted_together() {
    assert!(source_allowed(
        GpuTensorStorage::Real,
        NumericElementType::F32,
        NumericPrecision::F32,
        Some(GpuTensorStorage::Real),
        false,
        false
    ));
    assert!(source_allowed(
        GpuTensorStorage::Real,
        NumericElementType::F32,
        NumericPrecision::F32,
        Some(GpuTensorStorage::Real),
        true,
        true
    ));
    assert!(!source_allowed(
        GpuTensorStorage::Real,
        NumericElementType::F32,
        NumericPrecision::F32,
        Some(GpuTensorStorage::Real),
        true,
        false
    ));
    assert!(!source_allowed(
        GpuTensorStorage::Real,
        NumericElementType::F32,
        NumericPrecision::F64,
        Some(GpuTensorStorage::Real),
        false,
        false
    ));
    assert!(!source_allowed(
        GpuTensorStorage::Real,
        NumericElementType::I32,
        NumericPrecision::F32,
        Some(GpuTensorStorage::Real),
        false,
        false
    ));
    assert!(!source_allowed(
        GpuTensorStorage::ComplexInterleaved,
        NumericElementType::F64,
        NumericPrecision::F64,
        Some(GpuTensorStorage::ComplexInterleaved),
        true,
        true
    ));
    assert!(!source_allowed(
        GpuTensorStorage::Real,
        NumericElementType::F32,
        NumericPrecision::F32,
        Some(GpuTensorStorage::ComplexInterleaved),
        false,
        false
    ));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn non_dispatch_alias_copy_and_empty_routes_ignore_disabled_compute_geometry() {
    assert!(selected_launch(false, 0, 0).unwrap().is_none());
    assert!(selected_launch(true, 0, 0).is_err());
    assert!(selected_launch(true, 64, 0).is_err());
    let (launch, capacity) = selected_launch(true, 64, 1).unwrap().unwrap();
    assert_eq!(capacity, 64);
    let mut limits = wgpu::Limits::default();
    assert!(limits_allow(3, 12, 24, launch, capacity, &limits));
    limits.max_uniform_buffers_per_shader_stage = 0;
    assert!(!limits_allow(3, 12, 24, launch, capacity, &limits));
    limits.max_uniform_buffers_per_shader_stage = 1;
    limits.max_storage_buffer_binding_size = 16;
    assert!(!limits_allow(3, 12, 24, launch, capacity, &limits));
    assert_eq!(
        crate::backend::wgpu_storage::raw_word_requested_bytes(0),
        Some(4)
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn valid_floating_cast_resource_denials_are_typed_unsupported() {
    let mut limits = wgpu::Limits::default();
    let (launch, capacity) = selected_launch(true, 64, limits.max_compute_workgroups_per_dimension)
        .unwrap()
        .unwrap();
    assert!(require_pipeline_limits(3, 12, 24, launch, capacity, &limits).is_ok());
    limits.max_bindings_per_bind_group = 2;
    let binding = require_pipeline_limits(3, 12, 24, launch, capacity, &limits).unwrap_err();
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &binding
    ));
    limits.max_bindings_per_bind_group = 3;
    limits.max_uniform_buffers_per_shader_stage = 0;
    let uniform = require_pipeline_limits(3, 12, 24, launch, capacity, &limits).unwrap_err();
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &uniform
    ));
    limits.max_uniform_buffers_per_shader_stage = 1;
    limits.max_storage_buffer_binding_size = 16;
    let storage = require_pipeline_limits(3, 12, 24, launch, capacity, &limits).unwrap_err();
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &storage
    ));
    limits.max_buffer_size = 3;
    let empty_copy = require_output_extent(4, &limits).unwrap_err();
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &empty_copy
    ));
}
