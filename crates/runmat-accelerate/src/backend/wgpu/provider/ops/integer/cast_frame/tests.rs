use super::*;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn all_integer_classes_have_exact_word_extents_and_logical_authority() {
    for (kind, words) in [
        (IntegerElementType::I8, 3),
        (IntegerElementType::I16, 3),
        (IntegerElementType::I32, 3),
        (IntegerElementType::I64, 6),
        (IntegerElementType::U8, 3),
        (IntegerElementType::U16, 3),
        (IntegerElementType::U32, 3),
        (IntegerElementType::U64, 6),
    ] {
        assert_eq!(integer_word_count(kind, 3).unwrap(), words);
        assert_eq!(
            crate::backend::wgpu_storage::raw_word_requested_bytes(
                integer_word_count(kind, 0).unwrap() as u64,
            ),
            Some(4),
        );
    }
    assert!(logical_source_allowed(false, false));
    assert!(logical_source_allowed(true, true));
    assert!(!logical_source_allowed(true, false));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn frozen_integer_cast_limits_reject_each_physical_shortfall_before_allocation() {
    let mut limits = wgpu::Limits::default();
    let launch = FrozenLinearLaunch::new(64, limits.max_compute_workgroups_per_dimension).unwrap();
    assert!(limits_allow(3, 12, 24, Some(launch), &limits));
    limits.max_storage_buffer_binding_size = 16;
    assert!(!limits_allow(3, 12, 24, Some(launch), &limits));
    limits.max_storage_buffer_binding_size = 1024;
    limits.max_uniform_buffers_per_shader_stage = 0;
    assert!(!limits_allow(3, 12, 24, Some(launch), &limits));
    limits.max_uniform_buffers_per_shader_stage = 1;
    limits.max_compute_workgroup_size_x = 32;
    assert!(!limits_allow(3, 12, 24, Some(launch), &limits));
    assert!(limits_allow(0, 0, 4, None, &limits));
    limits.max_buffer_size = 3;
    assert!(!limits_allow(0, 0, 4, None, &limits));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn valid_integer_cast_resource_denials_are_typed_unsupported() {
    let mut limits = wgpu::Limits::default();
    let launch = FrozenLinearLaunch::new(64, limits.max_compute_workgroups_per_dimension).unwrap();
    assert!(require_selected_limits(3, 12, 24, Some(launch), &limits).is_ok());
    limits.max_storage_buffers_per_shader_stage = 1;
    let binding = require_selected_limits(3, 12, 24, Some(launch), &limits).unwrap_err();
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &binding
    ));
    limits.max_storage_buffers_per_shader_stage = 2;
    limits.max_buffer_size = 3;
    let empty = require_selected_limits(0, 0, 4, None, &limits).unwrap_err();
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &empty
    ));
    limits.max_buffer_size = 1024;
    limits.max_compute_workgroup_size_x = 32;
    let workgroup = require_selected_limits(3, 12, 24, Some(launch), &limits).unwrap_err();
    assert!(runmat_accelerate_api::is_unsupported_provider_operation(
        &workgroup
    ));
    #[cfg(target_pointer_width = "64")]
    {
        limits.max_compute_workgroup_size_x = 64;
        let length = require_selected_limits(u32::MAX as usize + 1, 12, 24, Some(launch), &limits)
            .unwrap_err();
        assert!(runmat_accelerate_api::is_unsupported_provider_operation(
            &length
        ));
    }
}
