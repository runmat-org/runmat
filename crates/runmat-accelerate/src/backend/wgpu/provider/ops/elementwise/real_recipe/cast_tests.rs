//! Actual canonical owner qualification; F64 witnesses require capable hardware.
use super::*;
use runmat_accelerate_api::{ProviderPrecision, ProviderRealElementwiseOutputRoute as Route};

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn reduced_kernel_limits_reject_conversion_but_not_zero_work_alias() {
    use super::super::cast::limits_allow;
    let mut limits = wgpu::Limits::default();
    let allow = |route, limits: &wgpu::Limits| {
        let output = match route {
            Route::FreshAllocation => 16,
            Route::SameClassFloatingCastAlias => 8,
        };
        limits_allow(route, limits, 8, output, 64, 64, 16)
    };
    assert!(allow(Route::FreshAllocation, &limits));
    limits.max_uniform_buffers_per_shader_stage = 0;
    assert!(!allow(Route::FreshAllocation, &limits));
    assert!(allow(Route::SameClassFloatingCastAlias, &limits));
    limits.max_storage_buffers_per_shader_stage = 0;
    limits.max_compute_workgroup_size_x = 0;
    limits.max_compute_invocations_per_workgroup = 0;
    limits.max_compute_workgroups_per_dimension = 0;
    limits.max_uniform_buffer_binding_size = 0;
    assert!(!allow(Route::FreshAllocation, &limits));
    assert!(allow(Route::SameClassFloatingCastAlias, &limits));
    limits.max_buffer_size = 4;
    assert!(!allow(Route::SameClassFloatingCastAlias, &limits));
    limits.max_buffer_size = 32;
    limits.max_storage_buffer_binding_size = 4;
    assert!(!allow(Route::SameClassFloatingCastAlias, &limits));
}

fn request(input: &GpuTensorHandle, target: ProviderPrecision) -> ProviderRealElementwiseRequest {
    ProviderRealElementwiseRequest {
        operation: Operation::FloatingCast { target },
        source: source_representation(input).unwrap(),
        source_buffer_bytes: input.physical().requested_bytes(),
    }
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires actual WGPU adapter; run explicitly"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn same_class_cast_keeps_exact_source_extent_and_zero_execution_work() {
    let provider = owner().await;
    let input = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[1.25, -0.0, f32::INFINITY]),
            shape: &[3, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let recipe =
        ProviderRealElementwiseRecipe::plan(provider, request(&input, ProviderPrecision::F32))
            .unwrap()
            .unwrap();
    assert_eq!(
        recipe.physical().output_route,
        Route::SameClassFloatingCastAlias
    );
    assert_eq!(recipe.physical().pipeline_key, None);
    assert_eq!(
        recipe.physical().requested_incremental_peak_bytes(),
        Some(0)
    );
    let records_before = provider.allocations.record_count().unwrap();
    let cold = recipe.execute(&input).await.unwrap();
    let warm = recipe.execute(&input).await.unwrap();
    assert_eq!(cold.observed, warm.observed);
    assert_eq!(
        cold.observed,
        ProviderRealElementwiseObserved {
            output_buffer_bytes: Some(input.physical().requested_bytes()),
            ..Default::default()
        }
    );
    assert_eq!(provider.allocations.record_count().unwrap(), records_before);
    for output in [&cold.output, &warm.output] {
        assert!(output.shares_allocation_with(&input));
        assert_eq!(output.view(), input.view());
        assert_eq!(output.physical(), input.physical());
    }
    drop(cold);
    drop(warm);
    input.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires actual F32 WGPU adapter; RUNMAT_WGPU_FORCE_PRECISION=f32"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn unavailable_f64_cast_is_rejected_before_any_selected_effect() {
    let provider = owner().await;
    assert_eq!(provider.precision(), ProviderPrecision::F32);
    let input = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&[4.0]),
            shape: &[1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let records_before = provider.allocations.record_count().unwrap();
    assert!(
        ProviderRealElementwiseRecipe::plan(provider, request(&input, ProviderPrecision::F64))
            .unwrap()
            .is_none()
    );
    assert_eq!(provider.allocations.record_count().unwrap(), records_before);
    input.allocation_lease().validate_live().unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires actual shader-f64 adapter; RUNMAT_WGPU_FORCE_PRECISION=f64 and RUNMAT_WG=64"]
async fn actual_class_changing_casts_multichunk_tail_and_warm_receipts_match() {
    let provider = owner().await;
    assert_eq!(provider.precision(), ProviderPrecision::F64);
    assert_eq!(
        provider.workgroup_config.scalar, 64,
        "bounded multichunk witness requires constructor RUNMAT_WG=64"
    );
    let capacity =
        crate::backend::wgpu::config::elementwise_chunk_capacity(provider.workgroup_config.scalar)
            .unwrap();
    let len = usize::try_from(capacity + 1).unwrap();
    let mut values = vec![1.25_f32; len];
    values[len - 1] = -3.5;
    let input = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&values),
            shape: &[len, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let up = ProviderRealElementwiseRecipe::plan(provider, request(&input, ProviderPrecision::F64))
        .unwrap()
        .unwrap();
    let cold = up.execute(&input).await.unwrap();
    let warm = up.execute(&input).await.unwrap();
    assert_eq!(cold.observed, warm.observed);
    assert_eq!(cold.observed.uniform_allocations, 2);
    assert_eq!(cold.observed.uniform_buffer_bytes, 32);
    assert_eq!(cold.observed.dispatches, 2);
    assert_eq!(cold.observed.queue_submissions, 2);
    assert_eq!(
        cold.observed.output_buffer_bytes,
        Some(cold.output.physical().requested_bytes())
    );
    assert_eq!(up.output().element_type, ProviderElementType::F64);
    let expected: Vec<f64> = values.iter().map(|value| f64::from(*value)).collect();
    assert_eq!(
        provider.download_numeric(&cold.output).await.unwrap().data,
        HostNumericDataOwned::F64(expected)
    );
    let down = ProviderRealElementwiseRecipe::plan(
        provider,
        request(&cold.output, ProviderPrecision::F32),
    )
    .unwrap()
    .unwrap();
    let published = down.execute(&cold.output).await.unwrap();
    let down_warm = down.execute(&cold.output).await.unwrap();
    assert_eq!(published.observed, down_warm.observed);
    assert_eq!(published.observed.uniform_buffer_bytes, 32);
    assert_eq!(published.observed.dispatches, 2);
    assert_eq!(
        published.output.physical().element_type(),
        runmat_accelerate_api::NumericElementType::F32
    );
    assert_eq!(
        provider
            .download_numeric(&published.output)
            .await
            .unwrap()
            .data,
        HostNumericDataOwned::F32(values)
    );
    assert!(!cold.output.shares_allocation_with(&input));
    assert!(!published.output.shares_allocation_with(&cold.output));
    input.allocation_lease().validate_live().unwrap();
}
