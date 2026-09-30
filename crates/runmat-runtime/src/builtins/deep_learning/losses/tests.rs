use super::*;
use futures::executor::block_on;
use runmat_accelerate_api::GpuHandleProvenance;
use runmat_value::IntegerStorage;

#[test]
fn loss_formats_admit_lowercase_repeated_spatial_axes_and_reject_bad_labels() {
    let shape = [2, 3, 2, 2];
    let layout = parse_loss_format(Some("sscb"), &shape, "DataFormat")
        .unwrap()
        .unwrap();
    assert_eq!(
        layout.axes(),
        &[Axis::Spatial, Axis::Spatial, Axis::Channel, Axis::Batch]
    );
    assert_eq!(batch_size_for_shape(&shape, Some(&layout)), 2.0);
    assert_eq!(
        parse_loss_format(Some("cctu"), &shape, "DataFormat")
            .unwrap_err()
            .identifier(),
        Some("RunMat:deepLearning:InvalidInput")
    );
    assert!(parse_loss_format(Some("sscx"), &shape, "DataFormat").is_err());
    assert!(parse_loss_format(Some("cb"), &shape, "DataFormat").is_err());
    assert!(parse_loss_format(Some(""), &[], "DataFormat")
        .unwrap()
        .is_none());
}

#[test]
fn repeated_spatial_weight_axes_broadcast_by_occurrence() {
    let data_shape = [2, 3, 2, 2];
    let data_format = parse_loss_format(Some("sscb"), &data_shape, "DataFormat")
        .unwrap()
        .unwrap();
    let weight_shape = [2, 3];
    let weight_format = parse_loss_format(Some("ss"), &weight_shape, "WeightsFormat")
        .unwrap()
        .unwrap();
    let weights = LossWeights {
        data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        shape: weight_shape.to_vec(),
    };
    let output = weights
        .materialize_for_layout(&data_shape, 24, Some(&data_format), Some(&weight_format))
        .unwrap();
    assert_eq!(&output[..6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(&output[6..12], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(&output[18..24], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn plain_numeric_pair_is_an_explicit_runmat_extension() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let err = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        vec![],
    ))
    .expect_err("numeric pair is not a documented MATLAB dlarray form");
    assert_eq!(
        err.identifier(),
        Some("RunMat:compatibility:CrossentropyPlainNumericExtension")
    );
}

#[test]
fn integer_input_is_rejected_by_strict_mode_before_payload_conversion() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let err = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new_integer(IntegerStorage::U64(vec![1, 0]), vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        vec![],
    ))
    .expect_err("typed integer prediction extension");
    assert_eq!(
        err.identifier(),
        Some("RunMat:compatibility:CrossentropyIntegerInputExtension")
    );
}

#[test]
fn excess_outputs_reject_before_provider_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let input = provider
            .upload(&HostTensorView {
                data: &[0.8, 0.2],
                shape: &[1, 2],
            })
            .expect("resident input");
        provider.reset_telemetry();
        let _outputs = crate::output_count::push_output_count(Some(2));
        let err = block_on(crossentropy_builtin(
            Value::GpuTensor(input.clone()),
            Value::GpuTensor(input),
            vec![],
        ))
        .expect_err("crossentropy has one output");
        assert_eq!(err.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn malformed_host_control_rejects_before_resident_prediction_download() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let prediction = provider
            .upload(&HostTensorView {
                data: &[0.8, 0.2],
                shape: &[1, 2],
            })
            .expect("resident prediction");
        provider.reset_telemetry();
        let err = block_on(crossentropy_builtin(
            Value::GpuTensor(prediction),
            Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
            vec![
                Value::String("Reduction".into()),
                Value::String("bogus".into()),
            ],
        ))
        .expect_err("invalid host reduction");
        assert_eq!(err.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn invalid_resident_data_format_rejects_before_download_and_keeps_sources_live() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let prediction = provider
            .upload(&HostTensorView {
                data: &[0.8, 0.2, 0.7, 0.3, 0.6, 0.4],
                shape: &[2, 3],
            })
            .unwrap();
        let target = provider
            .upload(&HostTensorView {
                data: &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                shape: &[2, 3],
            })
            .unwrap();
        provider.reset_telemetry();
        let err = block_on(crossentropy_builtin(
            Value::GpuTensor(prediction.clone()),
            Value::GpuTensor(target.clone()),
            vec![
                Value::String("DataFormat".into()),
                Value::String("CC".into()),
            ],
        ))
        .expect_err("repeated channel axis");
        assert_eq!(err.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert!(prediction.allocation_lease().validate_live().is_ok());
        assert!(target.allocation_lease().validate_live().is_ok());
    });
}

#[test]
fn invalid_host_target_rejects_before_provider_execution() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let prediction = provider
            .upload(&HostTensorView {
                data: &[0.8, 0.2],
                shape: &[1, 2],
            })
            .expect("resident prediction");
        provider.reset_telemetry();
        let err = block_on(crossentropy_builtin(
            Value::GpuTensor(prediction),
            Value::Tensor(Tensor::new(vec![2.0, 0.0], vec![1, 2]).unwrap()),
            vec![],
        ))
        .expect_err("invalid target probability");
        assert_eq!(err.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn explicit_resident_weights_keep_unreduced_losses_resident_and_explicit() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let shape = [1usize, 2usize];
        let predictions = provider
            .upload(&HostTensorView {
                data: &[0.8, 0.2],
                shape: &shape,
            })
            .unwrap();
        let targets = provider
            .upload(&HostTensorView {
                data: &[1.0, 0.0],
                shape: &shape,
            })
            .unwrap();
        let weights = provider
            .upload(&HostTensorView {
                data: &[2.0, 3.0],
                shape: &shape,
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        provider.reset_telemetry();
        let output = block_on(crossentropy_builtin(
            Value::GpuTensor(predictions.clone()),
            Value::GpuTensor(targets.clone()),
            vec![
                Value::GpuTensor(weights.clone()),
                Value::String("ClassificationMode".into()),
                Value::String("multi-label".into()),
                Value::String("Reduction".into()),
                Value::String("none".into()),
            ],
        ))
        .unwrap();
        let Value::GpuTensor(handle) = &output else {
            panic!("expected resident losses")
        };
        assert!(runmat_accelerate_api::handle_is_explicit(handle));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let losses = crate::builtins::common::test_support::gather(output).unwrap();
        assert!((losses.materialize_f64()[0] + 2.0 * 0.8_f64.ln()).abs() < 1e-12);
        assert!((losses.materialize_f64()[1] + 3.0 * 0.8_f64.ln()).abs() < 1e-12);
        for input in [&predictions, &targets, &weights] {
            assert!(input.allocation_lease().validate_live().is_ok());
        }
    });
}

#[test]
fn explicit_unsupported_normalization_rejects_without_download_while_automatic_falls_back() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let shape = [1usize, 2usize];
        let predictions = provider
            .upload(&HostTensorView {
                data: &[0.8, 0.2],
                shape: &shape,
            })
            .unwrap();
        let targets = provider
            .upload(&HostTensorView {
                data: &[1.0, 0.0],
                shape: &shape,
            })
            .unwrap();
        let explicit = predictions
            .clone()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        let options = || {
            vec![
                Value::String("ClassificationMode".into()),
                Value::String("multi-label".into()),
                Value::String("Mask".into()),
                Value::LogicalArray(
                    runmat_value::LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap(),
                ),
                Value::String("NormalizationFactor".into()),
                Value::String("mask-included".into()),
            ]
        };
        provider.reset_telemetry();
        let error = block_on(crossentropy_builtin(
            Value::GpuTensor(explicit),
            Value::GpuTensor(targets.clone()),
            options(),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:Unsupported"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert!(predictions.allocation_lease().validate_live().is_ok());
        assert!(targets.allocation_lease().validate_live().is_ok());

        provider.reset_telemetry();
        let result = block_on(crossentropy_builtin(
            Value::GpuTensor(predictions),
            Value::GpuTensor(targets),
            options(),
        ))
        .unwrap();
        let Value::Num(loss) = result else {
            panic!("expected host fallback scalar")
        };
        assert!((loss + 0.8_f64.ln()).abs() < 1e-12);
        assert!(provider.telemetry_snapshot().download_bytes > 0);
    });
}
