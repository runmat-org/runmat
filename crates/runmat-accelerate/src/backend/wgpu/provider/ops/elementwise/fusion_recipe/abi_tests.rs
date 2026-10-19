use super::*;

fn representation(storage: GpuTensorStorage) -> FusedElementwiseRepresentation {
    FusedElementwiseRepresentation {
        input_precision: NumericPrecision::F32,
        output_precision: NumericPrecision::F32,
        output_storage: storage,
        logical_output: false,
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn typed_uniform_abi_not_shader_identifiers_controls_the_physical_recipe() {
    let work = |abi| FusedElementwiseWork {
        output_shape: &[2, 3],
        len: 6,
        invocation_geometry: ElementwiseInvocationGeometry::OutputLanes,
        representation: representation(GpuTensorStorage::Real),
        uniform_abi: abi,
    };
    let launch = FrozenLinearLaunch::new(64, 2).unwrap();
    let limits = wgpu::Limits::default();
    // This pure descriptor witness cannot authorize execution of arbitrary WGSL.
    let text = "// out_shape a_shape\n@compute @workgroup_size(@WG@) fn main() {}";
    let simple = FrozenFusionRecipe::plan(
        text,
        2,
        &work(Some(FusionUniformAbi::Simple)),
        launch,
        &limits,
    )
    .unwrap();
    let broadcast = FrozenFusionRecipe::plan(
        text,
        2,
        &work(Some(FusionUniformAbi::Broadcast)),
        launch,
        &limits,
    )
    .unwrap();
    assert_eq!(simple.expected().uniform_buffer_bytes, 16);
    assert_eq!(
        broadcast.expected().uniform_buffer_bytes,
        crate::backend::wgpu_geometry::fusion_broadcast_uniform_bytes(
            2,
            crate::backend::wgpu::params::BCAST_MAX_RANK
        )
        .unwrap()
    );
    assert!(!simple.same_work(&broadcast));
    assert!(FrozenFusionRecipe::plan(text, 2, &work(None), launch, &limits).is_err());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn shape_words_are_exact_for_real_complex_scalar_and_empty_outputs() {
    let real = representation(GpuTensorStorage::Real);
    let complex = representation(GpuTensorStorage::ComplexInterleaved);
    validate_output_words(&[2, 3], 6, &real).unwrap();
    validate_output_words(&[2, 3], 12, &complex).unwrap();
    validate_output_words(&[], 1, &real).unwrap();
    validate_output_words(&[], 2, &complex).unwrap();
    for storage in [GpuTensorStorage::Real, GpuTensorStorage::ComplexInterleaved] {
        let represented = representation(storage);
        validate_output_words(&[2, 0, 5], 0, &represented).unwrap();
        validate_output_words(&[usize::MAX, 0], 0, &represented).unwrap();
    }
    assert!(validate_output_words(&[2, 3], 12, &real).is_err());
    assert!(validate_output_words(&[2, 3], 6, &complex).is_err());
    assert!(validate_output_words(&[usize::MAX, 2], 0, &real).is_err());
    assert!(validate_output_words(&[usize::MAX], 0, &complex).is_err());
    let mut invalid_logical = complex;
    invalid_logical.logical_output = true;
    assert!(validate_output_words(&[2, 3], 12, &invalid_logical).is_err());
    let launch = FrozenLinearLaunch::new(64, 2).unwrap();
    let empty = FrozenFusionRecipe::plan(
        "",
        1,
        &FusedElementwiseWork {
            output_shape: &[2, 0, 5],
            len: 0,
            invocation_geometry: ElementwiseInvocationGeometry::OutputLanes,
            representation: complex,
            uniform_abi: Some(FusionUniformAbi::Simple),
        },
        launch,
        &wgpu::Limits::default(),
    )
    .unwrap();
    assert_eq!(empty.expected().output_buffer_bytes, Some(4));
    assert_eq!(empty.expected().dispatches, 0);
    assert_eq!(empty.expected().queue_submissions, 1);
}
