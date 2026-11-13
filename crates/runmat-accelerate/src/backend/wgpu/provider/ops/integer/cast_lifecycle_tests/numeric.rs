use super::*;
async fn all_classes_and_typed_postallocation_failure() {
    let provider = owner().await;
    let input = provider
        .upload_integer_exec(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&[-1, 0, 127, 255, 65_535, i64::MAX]),
            shape: &[1, 2, 3],
        })
        .unwrap();
    let cases = [
        (
            IntegerElementType::I8,
            HostIntegerDataOwned::I8(vec![-1, 0, 127, 127, 127, 127]),
        ),
        (
            IntegerElementType::I16,
            HostIntegerDataOwned::I16(vec![-1, 0, 127, 255, 32_767, 32_767]),
        ),
        (
            IntegerElementType::I32,
            HostIntegerDataOwned::I32(vec![-1, 0, 127, 255, 65_535, i32::MAX]),
        ),
        (
            IntegerElementType::I64,
            HostIntegerDataOwned::I64(vec![-1, 0, 127, 255, 65_535, i64::MAX]),
        ),
        (
            IntegerElementType::U8,
            HostIntegerDataOwned::U8(vec![0, 0, 127, 255, 255, 255]),
        ),
        (
            IntegerElementType::U16,
            HostIntegerDataOwned::U16(vec![0, 0, 127, 255, 65_535, 65_535]),
        ),
        (
            IntegerElementType::U32,
            HostIntegerDataOwned::U32(vec![0, 0, 127, 255, 65_535, u32::MAX]),
        ),
        (
            IntegerElementType::U64,
            HostIntegerDataOwned::U64(vec![0, 0, 127, 255, 65_535, i64::MAX as u64]),
        ),
    ];
    for (target, expected) in cases {
        let output = provider.cast_to_integer_exec(&input, target).await.unwrap();
        assert_eq!(output.shape(), &[1, 2, 3]);
        assert_eq!(
            runmat_accelerate_api::handle_integer_type(&output),
            Some(target)
        );
        assert_eq!(
            provider.download_integer_exec(&output).await.unwrap().data,
            expected
        );
        output.release().unwrap();
        input.allocation_lease().validate_live().unwrap();
    }
    let logical = provider
        .upload_logical(&HostLogicalTensorView {
            data: &[0, 1],
            shape: &[1, 2],
        })
        .unwrap();
    let numeric = provider
        .cast_to_integer_exec(&logical, IntegerElementType::U8)
        .await
        .unwrap();
    assert!(!numeric.shares_allocation_with(&logical));
    assert_eq!(
        provider.download_integer_exec(&numeric).await.unwrap().data,
        HostIntegerDataOwned::U8(vec![0, 1])
    );
    logical.allocation_lease().validate_live().unwrap();
    numeric.release().unwrap();
    logical.release().unwrap();
    let count = provider.allocations.record_count().unwrap();
    let mut physical = None;
    let error = provider
        .cast_to_integer_exec_with_checkpoint(&input, IntegerElementType::U8, |buffer| {
            physical = Some(weak_buffer(buffer));
            Err(runmat_accelerate_api::unsupported_provider_operation(
                "injected after integer cast allocation",
            ))
        })
        .await
        .expect_err("postallocation failure is terminal");
    assert!(error.is_after_effect());
    assert!(!error.is_unsupported());
    assert!(error
        .source()
        .is_some_and(|source| source.is::<runmat_accelerate_api::UnsupportedProviderOperation>()));
    assert!(physical
        .expect("actual output boundary reached")
        .upgrade()
        .is_none());
    assert_eq!(provider.allocations.record_count().unwrap(), count);
    input.allocation_lease().validate_live().unwrap();
    input.release().unwrap();
}

async fn empty_integer_cast_uses_actual_minimum_without_compute_geometry() {
    let mut provider = owner().await;
    let input = provider
        .upload_integer_exec(&HostIntegerTensorView {
            data: HostIntegerDataView::I64(&[]),
            shape: &[0, 2, 3],
        })
        .unwrap();
    provider.workgroup_config.scalar = 0;
    provider.adapter_limits.max_compute_workgroups_per_dimension = 0;
    let output = provider
        .cast_to_integer_exec(&input, IntegerElementType::U64)
        .await
        .unwrap();
    let entry = provider.get_entry_raw(&output).unwrap();
    assert_eq!(entry.buffer.size(), 4);
    assert_eq!(entry.allocated_bytes, 4);
    assert_eq!(output.physical().requested_bytes(), 4);
    assert_eq!(output.shape(), &[0, 2, 3]);
    assert_eq!(
        provider.download_integer_exec(&output).await.unwrap().data,
        HostIntegerDataOwned::U64(vec![])
    );
    input.allocation_lease().validate_live().unwrap();
    output.release().unwrap();
    input.release().unwrap();
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires actual native WGPU adapter; run explicitly in cast qualification"]
async fn native_empty_cast_minimum_without_compute_geometry() {
    empty_integer_cast_uses_actual_minimum_without_compute_geometry().await;
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_empty_cast_minimum_without_compute_geometry() {
    empty_integer_cast_uses_actual_minimum_without_compute_geometry().await;
}

#[cfg(not(target_arch = "wasm32"))]
#[tokio::test]
#[ignore = "requires an actual native WGPU adapter; run explicitly in cast qualification"]
async fn native_all_classes_and_integer_cast_owner_failure() {
    all_classes_and_typed_postallocation_failure().await;
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test::wasm_bindgen_test(async)]
async fn browser_all_classes_and_integer_cast_owner_failure() {
    all_classes_and_typed_postallocation_failure().await;
}
