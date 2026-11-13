use super::*;
use futures::executor::block_on;

#[test]
fn explicit_resident_parameter_never_falls_back_through_host_download() {
    use runmat_accelerate_api::GpuHandleProvenance;
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let tensor = Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let handle = upload_tensor_protecting_typed(provider, &tensor, &[])
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        let Value::Object(layer) = object_with_identity(
            FULLY_CONNECTED_LAYER_CLASS,
            vec![
                ("Weights", Value::GpuTensor(handle.clone())),
                ("OutputSize", Value::Num(1.0)),
            ],
        )
        .expect("fixture object") else {
            panic!("layer object expected")
        };
        provider.reset_telemetry();
        let error = block_on(prepare_host_parameter(
            &layer,
            ParameterRole::Weights,
            "forward",
        ))
        .unwrap_err();
        assert!(error.message.contains("explicit resident Weights"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let retained =
            crate::builtins::common::test_support::gather(Value::GpuTensor(handle)).unwrap();
        assert_eq!(retained.numeric_dtype(), NumericDType::F32);
        assert_eq!(retained.materialize_f64(), vec![1.0, 2.0]);
    });
}

#[test]
fn matching_resident_parameters_do_not_cross_the_host_boundary() {
    use runmat_accelerate_api::{handle_precision, GpuHandleProvenance};
    crate::builtins::common::test_support::with_test_provider(|provider| {
        for dtype in [NumericDType::F32, NumericDType::F64] {
            let upload = |values, shape| {
                let tensor = Tensor::new_with_dtype(values, shape, dtype).unwrap();
                upload_tensor_protecting_typed(provider, &tensor, &[])
                    .unwrap()
                    .with_provenance(GpuHandleProvenance::Explicit)
                    .unwrap()
            };
            let input = upload(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]);
            let weights = upload(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]);
            let bias = upload(vec![10.0, 20.0], vec![2, 1]);
            let Value::Object(layer) = object_with_identity(
                FULLY_CONNECTED_LAYER_CLASS,
                vec![
                    ("Weights", Value::GpuTensor(weights.clone())),
                    ("Bias", Value::GpuTensor(bias.clone())),
                    ("OutputSize", Value::Num(2.0)),
                ],
            )
            .expect("fixture object") else {
                panic!("expected layer object");
            };
            provider.reset_telemetry();
            let output = block_on(super::super::fully_connected_forward_gpu(
                provider,
                input.clone(),
                &layer,
                "forward",
            ))
            .unwrap();
            let transfers = provider.telemetry_snapshot();
            assert_eq!(transfers.download_bytes, 0);
            assert_eq!(transfers.upload_bytes, 0);
            assert_eq!(output.shape(), &[2, 2]);
            assert_eq!(handle_precision(&output), handle_precision(&input));
            let actual =
                crate::builtins::common::test_support::gather(Value::GpuTensor(output)).unwrap();
            assert_eq!(actual.numeric_dtype(), dtype);
            assert_eq!(actual.materialize_f64(), vec![15.0, 21.0, 31.0, 45.0]);
            for (retained, expected) in [
                (input, vec![1.0, 3.0, 2.0, 4.0]),
                (weights, vec![1.0, 3.0, 2.0, 4.0]),
                (bias, vec![10.0, 20.0]),
            ] {
                let actual =
                    crate::builtins::common::test_support::gather(Value::GpuTensor(retained))
                        .unwrap();
                assert_eq!(actual.materialize_f64(), expected);
            }
        }
    });
}

#[test]
fn invalid_parameter_layout_retires_only_the_returned_reference() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let input = upload_tensor_protecting_typed(provider, &tensor, &[]).unwrap();
        let error = validate_parameter_result(
            input.clone(),
            &input,
            provider,
            &[1, 2],
            GpuOutputAliasPolicy::AllowInput,
            "forward",
        )
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:gpu:ProviderPayloadMismatch")
        );
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        let actual =
            crate::builtins::common::test_support::gather(Value::GpuTensor(input)).unwrap();
        assert_eq!(actual.materialize_f64(), vec![1.0, 2.0]);
    });
}

#[test]
fn square_transpose_cannot_return_the_unchanged_input_view() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let input = upload_tensor_protecting_typed(provider, &tensor, &[]).unwrap();
        let error = validate_parameter_result(
            input.clone(),
            &input,
            provider,
            &[2, 2],
            GpuOutputAliasPolicy::RequireDistinct,
            "forward",
        )
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:gpu:ProviderPayloadMismatch")
        );
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        let actual =
            crate::builtins::common::test_support::gather(Value::GpuTensor(input)).unwrap();
        assert_eq!(actual.materialize_f64(), vec![1.0, 3.0, 2.0, 4.0]);
    });
}
