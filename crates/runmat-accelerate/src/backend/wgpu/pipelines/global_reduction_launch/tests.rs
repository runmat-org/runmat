use super::*;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn global_reduction_geometry_is_one_frozen_shader_dispatch_contract() {
    let limits = wgpu::Limits {
        max_compute_workgroup_size_x: 128,
        max_compute_invocations_per_workgroup: 128,
        max_compute_workgroup_storage_size: 1024,
        max_compute_workgroups_per_dimension: 3,
        ..wgpu::Limits::default()
    };
    let small = FrozenGlobalReductionLaunch::new(64, NumericPrecision::F64, &limits).unwrap();
    let large = FrozenGlobalReductionLaunch::new(128, NumericPrecision::F64, &limits).unwrap();
    assert_ne!(small, large);
    assert_eq!(small.elements_per_group(), 128);
    assert_eq!(large.elements_per_group(), 256);
    assert_eq!(small.max_input_per_pass(), Some(384));
    assert_eq!(large.max_input_per_pass(), Some(768));
    for (input, output, groups) in [(0, 0, 0), (1, 1, 1), (128, 1, 1), (129, 2, 2), (384, 3, 3)] {
        assert_eq!(small.output_len(input), output);
        assert_eq!(small.dispatch_groups(input), Some(groups));
    }
    assert_eq!(small.dispatch_groups(385), None);
    assert_eq!(large.dispatch_groups(385), Some(2));
    for (precision, workgroup) in [
        (NumericPrecision::F32, small),
        (NumericPrecision::F64, large),
    ] {
        let source = crate::backend::wgpu::shaders::reduction::global_source(precision)
            .replace("@WG@", &workgroup.workgroup().to_string());
        assert!(!source.contains("@WG@"));
        assert!(source.contains(&format!("@workgroup_size({})", workgroup.workgroup())));
    }

    let storage_too_small = wgpu::Limits {
        max_compute_workgroup_storage_size: 511,
        ..limits.clone()
    };
    assert!(
        FrozenGlobalReductionLaunch::new(64, NumericPrecision::F64, &storage_too_small).is_none()
    );
    assert!(
        FrozenGlobalReductionLaunch::new(64, NumericPrecision::F32, &storage_too_small).is_some()
    );
    let no_dispatch = wgpu::Limits {
        max_compute_workgroups_per_dimension: 0,
        ..limits.clone()
    };
    assert!(FrozenGlobalReductionLaunch::new(64, NumericPrecision::F32, &no_dispatch).is_none());
    assert!(FrozenGlobalReductionLaunch::new(0, NumericPrecision::F32, &limits).is_none());
    assert!(FrozenGlobalReductionLaunch::new(96, NumericPrecision::F32, &limits).is_none());
    assert!(FrozenGlobalReductionLaunch::new(256, NumericPrecision::F32, &limits).is_none());
}
