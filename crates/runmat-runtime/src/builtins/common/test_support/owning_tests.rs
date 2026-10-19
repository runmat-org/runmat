use super::*;

#[test]
fn f32_fixture_uploads_actual_logical_storage_without_relabeling() {
    with_f32_test_provider(|provider| {
        let handle = runmat_accelerate_api::upload_logical_protecting(
            provider,
            &runmat_accelerate_api::HostLogicalTensorView {
                data: &[0, 1, 1, 0],
                shape: &[2, 1, 2],
            },
            &[],
        )
        .unwrap();
        assert!(runmat_accelerate_api::handle_is_logical(&handle));
        assert_eq!(handle.shape(), &[2, 1, 2]);
        assert_eq!(handle.device_id(), provider.device_id());
        let clone = handle.clone();
        let value = block_on(crate::dispatcher::gather_if_needed_async(
            &Value::GpuTensor(handle),
        ))
        .unwrap();
        let Value::LogicalArray(array) = value else {
            panic!("logical producer class lost");
        };
        assert_eq!(array.data, vec![0, 1, 1, 0]);
        assert_eq!(array.shape, vec![2, 1, 2]);
        clone.allocation_lease().validate_live().unwrap();
    });
}

#[test]
fn rejection_fixture_counts_actual_final_lease_retirement() {
    with_rejecting_native_result_provider(|provider| {
        let output = provider.rejected_handle().unwrap();
        assert_eq!(
            runmat_accelerate_api::handle_storage(&output),
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
        );
        let independently_retained = output.clone();
        drop(output);
        assert_eq!(provider.free_count(), 0);
        independently_retained
            .allocation_lease()
            .validate_live()
            .unwrap();
        drop(independently_retained);
        assert_eq!(provider.free_count(), 1);
    });
}
