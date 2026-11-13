use super::*;

fn plan(
    len: usize,
    launch: FrozenLinearLaunch,
    limits: &wgpu::Limits,
) -> Result<FrozenFusionRecipe> {
    FrozenFusionRecipe::plan(
        "@compute @workgroup_size(@WG@) fn main() {}",
        2,
        &FusedElementwiseWork {
            output_shape: &[len, 1],
            len,
            invocation_geometry: ElementwiseInvocationGeometry::OutputLanes,
            uniform_abi: Some(FusionUniformAbi::Simple),
            representation: FusedElementwiseRepresentation {
                input_precision: NumericPrecision::F32,
                output_precision: NumericPrecision::F32,
                output_storage: GpuTensorStorage::Real,
                logical_output: false,
            },
        },
        launch,
        limits,
    )
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn frozen_fusion_counts_actual_ordinary_uniform_and_noop_at_chunk_tail() {
    let limits = wgpu::Limits::default();
    let launch = FrozenLinearLaunch::new(64, 2).unwrap();
    for (len, dispatches) in [(0, 0), (1, 1), (64, 1), (65, 1), (128, 1), (129, 2)] {
        let recipe = plan(len, launch, &limits).unwrap();
        let observed = recipe.expected();
        assert_eq!(observed.output_buffer_bytes, Some((len.max(1) * 4) as u64));
        assert_eq!(observed.uniform_allocations, 1);
        assert_eq!(
            observed.uniform_buffer_bytes,
            std::mem::size_of::<crate::backend::wgpu::params::FusionParams>() as u64
        );
        assert_eq!(observed.dispatches, dispatches);
        assert_eq!(observed.queue_submissions, dispatches + 1);
        recipe.require(&observed).unwrap();
    }
    let first = plan(129, launch, &limits).unwrap();
    let second = plan(129, FrozenLinearLaunch::new(128, 2).unwrap(), &limits).unwrap();
    assert!(!first.same_work(&second));
    assert_eq!(first.launch().workgroup(), 64);
    assert_eq!(second.expected().dispatches, 1);
    assert!(first.pipeline().shader.contains("workgroup_size(64)"));
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn reduced_enabled_limits_and_each_wrong_observation_fail_closed() {
    let launch = FrozenLinearLaunch::new(64, 2).unwrap();
    let limits = wgpu::Limits::default();
    let recipe = plan(129, launch, &limits).unwrap();
    for mutation in 0..5 {
        let mut wrong = recipe.expected();
        match mutation {
            0 => wrong.output_buffer_bytes = Some(4),
            1 => wrong.uniform_allocations += 1,
            2 => wrong.uniform_buffer_bytes += 16,
            3 => wrong.dispatches += 1,
            4 => wrong.queue_submissions -= 1,
            _ => unreachable!(),
        }
        assert!(recipe.require(&wrong).is_err());
    }
    let mut missing_uniform = limits.clone();
    missing_uniform.max_uniform_buffers_per_shader_stage = 0;
    assert!(plan(129, launch, &missing_uniform).is_err());
    let mut missing_storage = limits.clone();
    missing_storage.max_storage_buffers_per_shader_stage = 2;
    assert!(plan(129, launch, &missing_storage).is_err());
    let mut wrong_workgroup = limits.clone();
    wrong_workgroup.max_compute_workgroup_size_x = 32;
    assert!(plan(129, launch, &wrong_workgroup).is_err());
    let mut wrong_extent = limits;
    wrong_extent.max_storage_buffer_binding_size = 128;
    assert!(plan(129, launch, &wrong_extent).is_err());
    let wrong_dispatch = wgpu::Limits {
        max_compute_workgroups_per_dimension: 1,
        ..wgpu::Limits::default()
    };
    assert!(plan(129, launch, &wrong_dispatch).is_err());
}
