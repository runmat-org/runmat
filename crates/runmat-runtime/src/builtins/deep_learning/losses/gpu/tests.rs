use super::*;
use runmat_accelerate_api::{HostNumericDataView, HostNumericTensorView, ProviderPrecision};

#[test]
fn operational_error_with_unsupported_words_does_not_become_fallback() {
    let error = anyhow::anyhow!("unsupported operation encountered a device failure");
    assert!(!runmat_accelerate_api::is_unsupported_provider_operation(
        &error
    ));
    let error = provider_error(error);
    assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
    assert!(error.source.is_some());
}

#[test]
fn result_validation_rejects_alias_and_shape_before_publication() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let input = provider
            .upload(&HostTensorView {
                data: &[0.8, 0.2],
                shape: &[1, 2],
            })
            .unwrap();
        let result = provider
            .upload(&HostTensorView {
                data: &[0.1, 0.2],
                shape: &[1, 2],
            })
            .unwrap();
        let precision = runmat_accelerate_api::handle_precision(&input);
        assert!(validate_result_handle(&input, &[1, 2], precision, provider, &[&input]).is_err());
        assert!(validate_result_handle(&result, &[2, 1], precision, provider, &[&input]).is_err());
        let wrong_precision = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[0.1, 0.2]),
                shape: &[1, 2],
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        assert!(
            validate_result_handle(&wrong_precision, &[1, 2], precision, provider, &[&input],)
                .is_err()
        );
        validate_result_handle(&result, &[1, 2], precision, provider, &[&input]).unwrap();
    });
}

#[test]
fn host_factor_upload_uses_prediction_precision_and_keeps_borrowed_inputs_live() {
    crate::builtins::common::test_support::with_f32_test_provider(|provider| {
        let shape = [1usize, 2usize];
        let predictions = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[0.8, 0.2]),
                shape: &shape,
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        provider.reset_telemetry();
        let result = materialize_gpu_loss_factor(
            provider,
            Some(GpuLossFactor::Host(vec![2.0, 3.0])),
            &shape,
            ProviderPrecision::F32,
            "weights",
            &[&predictions],
        )
        .unwrap();
        let MaterializedFactor::Ready(Some(weights)) = result else {
            panic!("expected resident factor")
        };
        assert_eq!(
            runmat_accelerate_api::handle_precision(&weights),
            Some(ProviderPrecision::F32)
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let gathered =
            crate::builtins::common::test_support::gather(Value::GpuTensor(weights)).unwrap();
        assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
        assert_eq!(gathered.materialize_f64(), vec![2.0, 3.0]);
        assert!(predictions.allocation_lease().validate_live().is_ok());
    });
}

#[test]
fn invalid_f32_factor_is_rejected_before_upload_or_borrowed_input_download() {
    crate::builtins::common::test_support::with_f32_test_provider(|provider| {
        let shape = [1usize, 1usize];
        let predictions = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&[0.8]),
                shape: &shape,
                storage: GpuTensorStorage::Real,
            })
            .unwrap();
        provider.reset_telemetry();
        let error = materialize_gpu_loss_factor(
            provider,
            Some(GpuLossFactor::Host(vec![f64::MAX])),
            &shape,
            ProviderPrecision::F32,
            "weights",
            &[&predictions],
        )
        .err()
        .expect("f32 overflow must reject");
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert!(predictions.allocation_lease().validate_live().is_ok());
    });
}
