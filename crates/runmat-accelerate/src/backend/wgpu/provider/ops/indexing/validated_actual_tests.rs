use super::*;
use runmat_accelerate_api::observation::{observe_provider_future, ProviderOperationReceipt};
use runmat_accelerate_api::{
    AccelProvider, HostIntegerDataView, HostIntegerTensorView, HostTensorView,
};
#[cfg(target_arch = "wasm32")]
use runmat_accelerate_api::{HostNumericDataView, HostNumericTensorView};

fn receipt(provider: &WgpuProvider) -> ProviderOperationReceipt {
    #[cfg(not(target_arch = "wasm32"))]
    {
        ProviderOperationReceipt::with_wgpu_progress(
            provider.observation_budget.clone(),
            provider
                .observation_progress
                .as_ref()
                .expect("actual WGPU provider has native progress")
                .clone(),
        )
    }
    #[cfg(target_arch = "wasm32")]
    {
        ProviderOperationReceipt::with_budget(provider.observation_budget.clone())
    }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn mapped_error_words_require_the_exact_backing_extent() {
    let words = [3u32, 2, 65535, 0];
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_ne_bytes()).collect();
    assert_eq!(decode_indexing_error_words(&bytes).unwrap(), words);
    assert!(decode_indexing_error_words(&bytes[..15]).is_err());
    assert!(decode_indexing_error_words(&[0u8; 17]).is_err());
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn indexing_shader_admission_uses_host_integrality_and_unsigned_dimensions() {
    let sub = crate::backend::wgpu::shaders::sub2ind::build_sub2ind_shader(
        NumericPrecision::F64,
        &[i32::MAX as u32 + 1],
        &[1],
        &[0],
        64,
    );
    assert!(sub.contains("const DIM_0: u32 = 2147483648u"));
    assert!(sub.contains("rounded > f64(DIM_0)"));
    assert!(!sub.contains("i32(DIM_0)"));
    assert!(sub.contains("2.220446049250313e-16"));
    let ind = crate::backend::wgpu::shaders::ind2sub::build_ind2sub_shader(
        NumericPrecision::F64,
        &[2, 2],
        &[1, 2],
        4,
        64,
    );
    assert!(ind.contains("let rounded_index: u64 = u64(rounded)"));
    assert!(ind.contains("zero_based / u64(STRIDE_1)"));
    assert!(ind.contains("2.220446049250313e-16"));
}

async fn actual_owner() -> &'static WgpuProvider {
    #[cfg(not(target_arch = "wasm32"))]
    let provider = WgpuProvider::new(Default::default()).expect("actual WGPU adapter");
    #[cfg(target_arch = "wasm32")]
    let provider = WgpuProvider::new_async(Default::default())
        .await
        .expect("actual WebGPU adapter");
    let provider = Box::leak(Box::new(provider));
    // SAFETY: this actual provider lives for the process.
    unsafe { runmat_accelerate_api::register_device_provider(provider) };
    provider
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(not(target_arch = "wasm32"), ignore = "requires actual WGPU adapter")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn subscript_kernels_cover_selected_workgroup_tail_and_async_error_readback() {
    let provider = actual_owner().await;
    let len = provider.workgroup_config.scalar as usize + 1;
    let rows = 2usize;
    let dims = [rows, len];
    let strides = [1usize, rows];
    let shape = [len, 1];
    let row: Vec<f64> = (0..len).map(|index| (index % rows + 1) as f64).collect();
    let col: Vec<f64> = (1..=len).map(|index| index as f64).collect();
    let expected: Vec<f64> = row
        .iter()
        .zip(&col)
        .map(|(&row, &col)| row + (col - 1.0) * rows as f64)
        .collect();
    let row_handle = provider
        .upload(&HostTensorView {
            data: &row,
            shape: &shape,
        })
        .unwrap();
    let col_handle = provider
        .upload(&HostTensorView {
            data: &col,
            shape: &shape,
        })
        .unwrap();
    let inputs = [&row_handle, &col_handle];
    if provider.precision != NumericPrecision::F64
        || !provider.ieee_words_available(provider.precision)
    {
        let cache = provider.fused_cache_counters_exec();
        assert!(provider
            .sub2ind_exec(&dims, &strides, &inputs, &[false, false], len, &shape)
            .await
            .unwrap_err()
            .is_unsupported());
        assert_eq!(provider.fused_cache_counters_exec(), cache);
        row_handle.allocation_lease().validate_live().unwrap();
        col_handle.allocation_lease().validate_live().unwrap();
        return;
    }
    let linear = provider
        .sub2ind_exec(&dims, &strides, &inputs, &[false, false], len, &shape)
        .await
        .expect("sub2ind has no device error");
    let host = provider.download(&linear).await.unwrap();
    assert_eq!(host.shape, shape);
    assert_eq!(host.data, expected);
    let outputs = provider
        .ind2sub_exec(&dims, &strides, &linear, rows * len, len, &shape)
        .await
        .expect("ind2sub has no device error");
    assert_eq!(outputs.len(), 2);
    assert_eq!(provider.download(&outputs[0]).await.unwrap().data, row);
    assert_eq!(provider.download(&outputs[1]).await.unwrap().data, col);
    row_handle.allocation_lease().validate_live().unwrap();
    col_handle.allocation_lease().validate_live().unwrap();
    linear.allocation_lease().validate_live().unwrap();
}

#[cfg(all(not(target_arch = "wasm32"), target_pointer_width = "64"))]
#[tokio::test]
#[ignore = "requires actual F64 WGPU adapter"]
async fn wide_positive_ind2sub_and_fractional_indices_follow_host_domain() {
    let provider = actual_owner().await;
    assert_eq!(provider.precision, NumericPrecision::F64);
    let wide = provider
        .upload(&HostTensorView {
            data: &[u32::MAX as f64 + 1.0],
            shape: &[1, 1],
        })
        .unwrap();
    let coords = provider
        .ind2sub_exec(&[2, 2], &[1, 2], &wide, 4, 1, &[1, 1])
        .await
        .expect("positive wide index expands the final coordinate");
    assert_eq!(provider.download(&coords[0]).await.unwrap().data, [2.0]);
    assert_eq!(
        provider.download(&coords[1]).await.unwrap().data,
        [2_147_483_648.0]
    );
    let high_dimension_subscript = provider
        .upload(&HostTensorView {
            data: &[2_147_483_648.0],
            shape: &[1, 1],
        })
        .unwrap();
    let high_dimension_linear = provider
        .sub2ind_exec(
            &[2_147_483_648],
            &[1],
            &[&high_dimension_subscript],
            &[false],
            1,
            &[1, 1],
        )
        .await
        .expect("unsigned dimension remains valid above signed i32 maximum");
    assert_eq!(
        provider
            .download(&high_dimension_linear)
            .await
            .unwrap()
            .data,
        [2_147_483_648.0]
    );
    let fractional = provider
        .upload(&HostTensorView {
            data: &[1.0 + 1e-12],
            shape: &[1, 1],
        })
        .unwrap();
    assert!(provider
        .ind2sub_exec(&[2, 2], &[1, 2], &fractional, 4, 1, &[1, 1])
        .await
        .is_err());
    assert!(provider
        .sub2ind_exec(&[2], &[1], &[&fractional], &[false], 1, &[1, 1])
        .await
        .is_err());
    wide.allocation_lease().validate_live().unwrap();
    high_dimension_subscript
        .allocation_lease()
        .validate_live()
        .unwrap();
    fractional.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(not(target_arch = "wasm32"), ignore = "requires actual WGPU adapter")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn floating_scatter_covers_both_selected_workgroup_tails_without_mutating_source() {
    let provider = actual_owner().await;
    let extent = provider.workgroup_config.scalar as usize + 1;
    let shape = [extent, 2usize];
    let source: Vec<f64> = (0..extent * 2).map(|index| index as f64).collect();
    let replacement: Vec<f64> = (0..extent).map(|index| 1000.0 + index as f64).collect();
    let matrix = provider
        .upload(&HostTensorView {
            data: &source,
            shape: &shape,
        })
        .unwrap();
    let values = provider
        .upload(&HostTensorView {
            data: &replacement,
            shape: &[extent, 1],
        })
        .unwrap();
    let scalar_replacement = [9001.0, 9002.0];
    let scalar_values = provider
        .upload(&HostTensorView {
            data: &scalar_replacement,
            shape: &[2, 1],
        })
        .unwrap();
    let functional = provider
        .scatter_linear_functional_exec(&matrix, &[1, (extent + 1) as u32], &scalar_values)
        .unwrap();
    let mut functional_expected = source.clone();
    functional_expected[1] = scalar_replacement[0];
    functional_expected[extent + 1] = scalar_replacement[1];
    assert!(!functional.shares_allocation_with(&matrix));
    assert!(!functional.shares_allocation_with(&scalar_values));
    assert_eq!(
        provider.download(&functional).await.unwrap().data,
        functional_expected
    );
    assert_eq!(provider.download(&matrix).await.unwrap().data, source);
    assert_eq!(
        provider.download(&scalar_values).await.unwrap().data,
        scalar_replacement
    );
    if provider.precision() == runmat_accelerate_api::ProviderPrecision::F32 {
        let output = provider
            .scatter_column_exec(&matrix, 1, &values)
            .await
            .unwrap();
        let mut expected = source.clone();
        expected[extent..].copy_from_slice(&replacement);
        assert_eq!(provider.download(&output).await.unwrap().data, expected);
        assert_eq!(provider.download(&matrix).await.unwrap().data, source);
        let row_shape = [2usize, extent];
        let row_source: Vec<f64> = (0..extent * 2).map(|index| index as f64).collect();
        let row_values: Vec<f64> = (0..extent).map(|index| 2000.0 + index as f64).collect();
        let row_matrix = provider
            .upload(&HostTensorView {
                data: &row_source,
                shape: &row_shape,
            })
            .unwrap();
        let row_replacement = provider
            .upload(&HostTensorView {
                data: &row_values,
                shape: &[1, extent],
            })
            .unwrap();
        let row_output = provider
            .scatter_row_exec(&row_matrix, 1, &row_replacement)
            .await
            .unwrap();
        let mut row_expected = row_source.clone();
        for (column, &value) in row_values.iter().enumerate() {
            row_expected[1 + column * 2] = value;
        }
        assert_eq!(
            provider.download(&row_output).await.unwrap().data,
            row_expected
        );
        assert_eq!(
            provider.download(&row_matrix).await.unwrap().data,
            row_source
        );
        row_matrix.allocation_lease().validate_live().unwrap();
        row_replacement.allocation_lease().validate_live().unwrap();
    } else {
        assert!(provider
            .scatter_column_exec(&matrix, 1, &values)
            .await
            .unwrap_err()
            .is_unsupported());
    }
    matrix.allocation_lease().validate_live().unwrap();
    values.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(not(target_arch = "wasm32"), ignore = "requires actual WGPU adapter")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn invalid_subscript_class_has_no_pipeline_cache_or_submission_effect() {
    let provider = actual_owner().await;
    let logical = provider
        .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
            data: &[1, 0],
            shape: &[2, 1],
        })
        .unwrap();
    let (hits, misses) = provider.fused_cache_counters_exec();
    let receipt = receipt(provider);
    let checkpoint = receipt.recorder().submission_checkpoint();
    let result = observe_provider_future(
        Some(receipt.recorder()),
        provider.sub2ind_exec(&[2], &[1], &[&logical], &[false], 2, &[2, 1]),
    )
    .await;
    assert!(result.unwrap_err().is_unsupported());
    assert_eq!(provider.fused_cache_counters_exec(), (hits, misses));
    assert_eq!(receipt.recorder().submission_checkpoint(), checkpoint);
    logical.allocation_lease().validate_live().unwrap();
}

#[cfg_attr(not(target_arch = "wasm32"), tokio::test)]
#[cfg_attr(not(target_arch = "wasm32"), ignore = "requires actual WGPU adapter")]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test(async))]
async fn scatter_rejects_extra_vector_axis_before_integer_gather_or_floating_copy() {
    let provider = actual_owner().await;
    let integer_matrix = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&[1, 2, 3, 4]),
            shape: &[2, 2],
        })
        .unwrap();
    let integer_column = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&[5, 6, 7, 8]),
            shape: &[2, 2],
        })
        .unwrap();
    let integer_row = provider
        .upload_integer(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&[5, 6, 7, 8]),
            shape: &[2, 2],
        })
        .unwrap();
    let records = provider.allocations.record_count().unwrap();
    let counters = provider.fused_cache_counters_exec();
    let operation = receipt(provider);
    let checkpoint = operation.recorder().submission_checkpoint();
    assert!(observe_provider_future(
        Some(operation.recorder()),
        provider.scatter_column_exec(&integer_matrix, 0, &integer_column),
    )
    .await
    .is_err());
    assert!(observe_provider_future(
        Some(operation.recorder()),
        provider.scatter_row_exec(&integer_matrix, 0, &integer_row),
    )
    .await
    .is_err());
    assert_eq!(provider.allocations.record_count().unwrap(), records);
    assert_eq!(provider.fused_cache_counters_exec(), counters);
    assert_eq!(operation.recorder().submission_checkpoint(), checkpoint);
    integer_matrix.allocation_lease().validate_live().unwrap();
    integer_column.allocation_lease().validate_live().unwrap();

    if provider.precision == NumericPrecision::F32 {
        let matrix = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0, 3.0, 4.0],
                shape: &[2, 2],
            })
            .unwrap();
        let extra_column = provider
            .upload(&HostTensorView {
                data: &[5.0, 6.0, 7.0, 8.0],
                shape: &[2, 2],
            })
            .unwrap();
        let records = provider.allocations.record_count().unwrap();
        let counters = provider.fused_cache_counters_exec();
        let checkpoint = operation.recorder().submission_checkpoint();
        assert!(observe_provider_future(
            Some(operation.recorder()),
            provider.scatter_column_exec(&matrix, 0, &extra_column),
        )
        .await
        .is_err());
        assert!(observe_provider_future(
            Some(operation.recorder()),
            provider.scatter_row_exec(&matrix, 0, &extra_column),
        )
        .await
        .is_err());
        assert_eq!(provider.allocations.record_count().unwrap(), records);
        assert_eq!(provider.fused_cache_counters_exec(), counters);
        assert_eq!(operation.recorder().submission_checkpoint(), checkpoint);
        matrix.allocation_lease().validate_live().unwrap();
        extra_column.allocation_lease().validate_live().unwrap();
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn dropping_unpolled_indexing_future_has_no_effect_and_preserves_borrowed_input() {
    let provider = actual_owner().await;
    let input = provider
        .upload(&HostTensorView {
            data: &[1.0, 2.0],
            shape: &[2, 1],
        })
        .unwrap();
    let dims = [2usize];
    let strides = [1usize];
    let inputs = [&input];
    let mask = [false];
    let shape = [2usize, 1];
    let records = provider.allocations.record_count().unwrap();
    let cache = provider.validated_fusion_cache.lock().unwrap().len();
    let future = provider.sub2ind_exec(&dims, &strides, &inputs, &mask, 2, &shape);
    drop(future);
    assert_eq!(provider.allocations.record_count().unwrap(), records);
    assert_eq!(provider.validated_fusion_cache.lock().unwrap().len(), cache);
    input.allocation_lease().validate_live().unwrap();
    if provider.precision == NumericPrecision::F64 {
        let output = provider
            .sub2ind_exec(&dims, &strides, &inputs, &mask, 2, &shape)
            .await
            .unwrap();
        assert_eq!(provider.download(&output).await.unwrap().data, [1.0, 2.0]);
    } else {
        assert!(provider
            .sub2ind_exec(&dims, &strides, &inputs, &mask, 2, &shape)
            .await
            .unwrap_err()
            .is_unsupported());
    }
    input.allocation_lease().validate_live().unwrap();
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn pending_cold_f32_scatter_cancel_retains_borrowed_inputs_and_retries() {
    use std::future::Future;
    let provider = actual_owner().await;
    assert_eq!(provider.precision, NumericPrecision::F32);
    let rows = provider.workgroup_config.scalar as usize + 1;
    let source: Vec<f32> = (0..rows * 2).map(|index| index as f32).collect();
    let replacement: Vec<f32> = (0..rows).map(|index| index as f32 + 1000.0).collect();
    let matrix = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&source),
            shape: &[rows, 2],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let values = provider
        .upload_numeric_raw(&HostNumericTensorView {
            data: HostNumericDataView::F32(&replacement),
            shape: &[rows, 1],
            storage: GpuTensorStorage::Real,
        })
        .unwrap();
    let records = provider.allocations.record_count().unwrap();
    let cache = provider.validated_fusion_cache.lock().unwrap().len();
    let operation = receipt(provider);
    let checkpoint = operation.recorder().submission_checkpoint();
    let mut future = Box::pin(observe_provider_future(
        Some(operation.recorder()),
        provider.scatter_column_exec(&matrix, 1, &values),
    ));
    let pending = std::future::poll_fn(|context| {
        std::task::Poll::Ready(future.as_mut().poll(context).is_pending())
    })
    .await;
    assert!(
        pending,
        "web pop_error_scope Promise must suspend cold preparation"
    );
    drop(future);
    assert_eq!(provider.allocations.record_count().unwrap(), records);
    assert_eq!(provider.validated_fusion_cache.lock().unwrap().len(), cache);
    assert_eq!(operation.recorder().submission_checkpoint(), checkpoint);
    matrix.allocation_lease().validate_live().unwrap();
    values.allocation_lease().validate_live().unwrap();

    let retry = provider
        .scatter_column_exec(&matrix, 1, &values)
        .await
        .unwrap();
    let mut expected: Vec<f64> = source.iter().map(|&value| f64::from(value)).collect();
    expected[rows..].copy_from_slice(
        &replacement
            .iter()
            .map(|&value| f64::from(value))
            .collect::<Vec<_>>(),
    );
    assert_eq!(provider.download(&retry).await.unwrap().data, expected);
    assert_eq!(
        provider.download(&matrix).await.unwrap().data,
        source
            .iter()
            .map(|&value| f64::from(value))
            .collect::<Vec<_>>()
    );
    matrix.allocation_lease().validate_live().unwrap();
    values.allocation_lease().validate_live().unwrap();
}
