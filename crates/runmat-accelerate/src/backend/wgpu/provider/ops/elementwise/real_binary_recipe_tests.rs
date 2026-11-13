//! Real adapter evidence: ordinary and recipe routes share outputs and the
//! actual output/uniform/submission transcript. Native runs are opt-in.
use super::*;
use crate::backend::wgpu::provider::WgpuProviderOptions;
use runmat_accelerate_api::{
    AccelProvider, HostNumericDataOwned, HostNumericDataView, HostNumericTensorView,
};

async fn owner() -> &'static WgpuProvider {
    #[cfg(not(target_arch = "wasm32"))]
    let provider =
        WgpuProvider::new(WgpuProviderOptions::default()).expect("actual WGPU adapter required");
    #[cfg(target_arch = "wasm32")]
    let provider = WgpuProvider::new_async(WgpuProviderOptions::default())
        .await
        .expect("actual WebGPU adapter required");
    let provider = Box::leak(Box::new(provider));
    // SAFETY: process-lifetime facade and uniquely minted actual WGPU device.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    provider
}

fn input(provider: &WgpuProvider, values: &[f32], shape: &[usize]) -> GpuTensorHandle {
    provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(values),
            shape,
            storage: GpuTensorStorage::Real,
        })
        .unwrap()
}

fn request(
    operation: ProviderRealBinaryOperation,
    lhs: &GpuTensorHandle,
    rhs: &GpuTensorHandle,
) -> ProviderRealBinaryRequest {
    ProviderRealBinaryRequest {
        operation,
        lhs: ProviderRealBinaryInput {
            representation: real_recipe::source_representation(lhs).unwrap(),
            buffer_bytes: lhs.physical().requested_bytes(),
        },
        rhs: ProviderRealBinaryInput {
            representation: real_recipe::source_representation(rhs).unwrap(),
            buffer_bytes: rhs.physical().requested_bytes(),
        },
    }
}

async fn qualify(provider: &'static WgpuProvider) {
    let lhs = input(provider, &[8.0; 40], &[2, 4, 5]);
    let same = input(provider, &[2.0; 40], &[2, 4, 5]);
    let broadcast = input(provider, &[2.0, 4.0], &[2, 1, 1]);
    let same_recipe = ProviderRealBinaryRecipe::plan(
        provider,
        request(ProviderRealBinaryOperation::Sub, &lhs, &same),
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        same_recipe.physical().route,
        ProviderRealBinaryRoute::SameShape
    );
    assert_eq!(same_recipe.physical().pre_output_submissions, 2);
    let same_cold = same_recipe.execute(&lhs, &same).await.unwrap();
    let same_warm = same_recipe.execute(&lhs, &same).await.unwrap();
    assert_eq!(same_cold.observed, same_warm.observed);
    assert_eq!(same_cold.observed.queue_submissions, 3);
    assert_eq!(same_cold.observed.uniform_allocations, 1);
    assert_eq!(same_cold.observed.uniform_buffer_bytes, 256);
    let ordinary_same = provider.elem_sub(&lhs, &same).await.unwrap();
    for output in [&same_cold.output, &same_warm.output, &ordinary_same] {
        assert_eq!(
            provider.download_numeric(output).await.unwrap().data,
            HostNumericDataOwned::F32(vec![6.0; 40])
        );
    }
    let bcast_recipe = ProviderRealBinaryRecipe::plan(
        provider,
        request(ProviderRealBinaryOperation::Div, &lhs, &broadcast),
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        bcast_recipe.physical().route,
        ProviderRealBinaryRoute::Broadcast
    );
    assert_eq!(bcast_recipe.physical().output.shape, [2, 4, 5]);
    assert_eq!(bcast_recipe.physical().pre_output_submissions, 0);
    let bcast_cold = bcast_recipe.execute(&lhs, &broadcast).await.unwrap();
    let bcast_warm = bcast_recipe.execute(&lhs, &broadcast).await.unwrap();
    assert_eq!(bcast_cold.observed, bcast_warm.observed);
    assert_eq!(bcast_cold.observed.uniform_allocations, 1);
    assert_eq!(
        bcast_cold.observed.uniform_buffer_bytes,
        bcast_recipe.physical().uniform_buffer_bytes
    );
    assert_eq!(bcast_cold.observed.queue_submissions, 1);
    let ordinary_bcast = provider.elem_div(&lhs, &broadcast).await.unwrap();
    let expected = (0..40)
        .map(|index| if index % 2 == 0 { 4.0 } else { 2.0 })
        .collect::<Vec<f32>>();
    for output in [&bcast_cold.output, &bcast_warm.output, &ordinary_bcast] {
        assert_eq!(
            provider.download_numeric(output).await.unwrap().data,
            HostNumericDataOwned::F32(expected.clone())
        );
    }
    let mut forged = request(ProviderRealBinaryOperation::Div, &lhs, &broadcast);
    forged.rhs.representation.shape = vec![3, 1, 1];
    assert!(ProviderRealBinaryRecipe::plan(provider, forged)
        .unwrap()
        .is_none());
    let mut forged = request(ProviderRealBinaryOperation::Div, &lhs, &broadcast);
    forged.rhs.buffer_bytes = 1;
    assert!(ProviderRealBinaryRecipe::plan(provider, forged)
        .unwrap()
        .is_none());
    lhs.allocation_lease().validate_live().unwrap();
    same.allocation_lease().validate_live().unwrap();
    broadcast.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires an actual WGPU adapter; run explicitly"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn actual_binary_sub_div_same_and_broadcast_match_ordinary_cold_warm() {
    qualify(owner().await).await;
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(
    not(target_arch = "wasm32"),
    ignore = "requires an actual WGPU adapter; run explicitly"
)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn actual_binary_same_and_broadcast_cover_workgroup_tail() {
    let provider = owner().await;
    let mut values = vec![8.0_f32; 514];
    values[512] = 20.0;
    values[513] = 40.0;
    let lhs = input(provider, &values, &[2, 257, 1]);
    let same = input(provider, &vec![2.0; 514], &[2, 257, 1]);
    let broadcast = input(provider, &[2.0, 4.0], &[2, 1, 1]);
    let sub = ProviderRealBinaryRecipe::plan(
        provider,
        request(ProviderRealBinaryOperation::Sub, &lhs, &same),
    )
    .unwrap()
    .unwrap();
    let div = ProviderRealBinaryRecipe::plan(
        provider,
        request(ProviderRealBinaryOperation::Div, &lhs, &broadcast),
    )
    .unwrap()
    .unwrap();
    let selected_sub = sub.execute(&lhs, &same).await.unwrap();
    let selected_div = div.execute(&lhs, &broadcast).await.unwrap();
    let ordinary_sub = provider.elem_sub(&lhs, &same).await.unwrap();
    let ordinary_div = provider.elem_div(&lhs, &broadcast).await.unwrap();
    let expected_sub = values.iter().map(|value| *value - 2.0).collect::<Vec<_>>();
    let expected_div = values
        .iter()
        .enumerate()
        .map(|(index, value)| *value / if index % 2 == 0 { 2.0 } else { 4.0 })
        .collect::<Vec<_>>();
    for handle in [&selected_sub.output, &ordinary_sub] {
        assert_eq!(
            provider.download_numeric(handle).await.unwrap().data,
            HostNumericDataOwned::F32(expected_sub.clone())
        );
    }
    for handle in [&selected_div.output, &ordinary_div] {
        assert_eq!(
            provider.download_numeric(handle).await.unwrap().data,
            HostNumericDataOwned::F32(expected_div.clone())
        );
    }
    assert_eq!(selected_sub.observed.queue_submissions, 3);
    assert_eq!(selected_div.observed.queue_submissions, 1);
    assert_eq!(selected_sub.observed.output_buffer_bytes, Some(514 * 4));
    assert_eq!(selected_div.observed.output_buffer_bytes, Some(514 * 4));
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires an actual adapter, RUNMAT_WGPU_FORCE_PRECISION=f32 and RUNMAT_WG=256"]
async fn actual_f32_binary_device_launch_preserves_outputs_and_receipts() {
    let provider = owner().await;
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F32
    );
    assert_eq!(provider.binary_linear_launch().unwrap().workgroup(), 256);
    qualify(provider).await;
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires actual shader-f64 adapter and RUNMAT_WGPU_FORCE_PRECISION=f64"]
async fn actual_f64_owner_supports_f32_real_binary_without_class_widening() {
    let provider = owner().await;
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F64
    );
    qualify(provider).await;
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires actual shader-f64 adapter and RUNMAT_WGPU_FORCE_PRECISION=f64"]
async fn actual_f64_binary_route_retains_double_output_and_receipt() {
    let provider = owner().await;
    assert_eq!(
        provider.precision(),
        runmat_accelerate_api::ProviderPrecision::F64
    );
    let lhs = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[8.0, 16.0]),
            shape: &[2, 1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let rhs = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F64(&[2.0, 4.0]),
            shape: &[2, 1, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let recipe = ProviderRealBinaryRecipe::plan(
        provider,
        request(ProviderRealBinaryOperation::Div, &lhs, &rhs),
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        recipe.physical().output.element_type,
        ProviderElementType::F64
    );
    let result = recipe.execute(&lhs, &rhs).await.unwrap();
    assert_eq!(result.observed.output_buffer_bytes, Some(16));
    assert_eq!(result.observed.queue_submissions, 3);
    assert_eq!(
        provider
            .download_numeric(&result.output)
            .await
            .unwrap()
            .data,
        HostNumericDataOwned::F64(vec![4.0, 4.0])
    );
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn enabled_dispatch_limit_and_incompatible_broadcast_are_rejected() {
    let route = broadcast_plan::BroadcastRoute::with_enabled_limit(
        &[2, 513],
        &[2, 1],
        crate::backend::wgpu::params::BCAST_MAX_RANK,
        4,
        1,
    )
    .unwrap();
    assert_eq!(route.chunk_capacity, 512);
    assert_eq!(route.submissions, 3);
    assert!(broadcast_plan::BroadcastRoute::with_enabled_limit(
        &[2, 513],
        &[2, 1],
        crate::backend::wgpu::params::BCAST_MAX_RANK,
        4,
        0,
    )
    .is_err());
    assert!(broadcast_plan::BroadcastRoute::with_enabled_limit(
        &[2, 513],
        &[3, 1],
        crate::backend::wgpu::params::BCAST_MAX_RANK,
        4,
        1,
    )
    .is_err());
}
