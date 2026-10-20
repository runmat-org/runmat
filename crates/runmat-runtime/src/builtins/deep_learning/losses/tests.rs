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
fn recognized_name_value_options_pass_public_preflight() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let output = block_on(crossentropy_builtin(
        None,
        Value::Tensor(Tensor::new(vec![0.8, 0.2, 0.7, 0.3, 0.6, 0.4], vec![2, 3]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0], vec![2, 3]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("DataFormat".into()),
            Value::String("cB".into()),
            Value::String("Reduction".into()),
            Value::String("none".into()),
        ]),
    ))
    .expect("recognized options pass preflight");
    let Value::Tensor(losses) = output else {
        panic!("unreduced numeric pair returns an array");
    };
    assert_eq!(losses.shape(), &[2, 3]);
    let expected = [-0.8_f64.ln(), 0.0, -0.7_f64.ln(), 0.0, -0.6_f64.ln(), 0.0];
    for (actual, expected) in losses.materialize_f64().iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-12);
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn unreduced_numeric_loss_preserves_rank_zero_and_one_mask_geometry() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    for (shape, predictions, targets) in [
        (vec![], vec![0.8], vec![1.0]),
        (vec![2], vec![0.8, 0.2], vec![1.0, 0.0]),
    ] {
        let mask = Value::Tensor(Tensor::new(vec![1.0; predictions.len()], shape.clone()).unwrap());
        let output = block_on(crossentropy_builtin(
            None,
            Value::Tensor(Tensor::new(predictions, shape.clone()).unwrap()),
            Value::Tensor(Tensor::new(targets, shape.clone()).unwrap()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("Reduction".into()),
                Value::String("none".into()),
                Value::String("Mask".into()),
                mask,
            ]),
        ))
        .unwrap();
        let Value::Tensor(losses) = output else {
            panic!("unreduced numeric loss tensor expected")
        };
        assert_eq!(losses.shape(), shape);
        assert!((losses.materialize_f64()[0] + 0.8_f64.ln()).abs() < 1.0e-12);
    }

    let error = block_on(crossentropy_builtin(
        None,
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![2]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Mask".into()),
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
        ]),
    ))
    .unwrap_err();
    assert!(error
        .message()
        .contains("Mask must have the same size as predictions"));
}

#[test]
fn plain_numeric_pair_is_an_explicit_runmat_extension() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let err = block_on(crossentropy_builtin(
        None,
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect_err("numeric pair is not a documented MATLAB dlarray form");
    assert_eq!(
        err.identifier(),
        Some("RunMat:compatibility:CrossentropyPlainNumericExtension")
    );
}

#[test]
fn single_resident_crossentropy_reduction_publishes_double_without_download() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let predictions = [0.8_f32, 0.2];
        let targets = [1.0_f32, 0.0];
        let upload = |data: &[f32]| {
            provider
                .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
                    data: runmat_accelerate_api::HostNumericDataView::F32(data),
                    shape: &[2, 1],
                    storage: runmat_accelerate_api::GpuTensorStorage::Real,
                })
                .unwrap()
        };
        let predictions = upload(&predictions);
        let targets = upload(&targets);
        provider.reset_telemetry();
        let result = block_on(crossentropy_builtin(
            None,
            Value::GpuTensor(predictions.clone()),
            Value::GpuTensor(targets.clone()),
            crate::builtins::common::test_support::builtin_arguments(Vec::new()),
        ))
        .unwrap();
        let Value::Num(loss) = result else {
            panic!("reduced loss must be a double scalar")
        };
        assert!((loss + f64::from(0.8_f32).ln() / 2.0).abs() < 1e-6);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        predictions.allocation_lease().validate_live().unwrap();
        targets.allocation_lease().validate_live().unwrap();
    });
}

#[test]
fn integer_input_is_rejected_by_strict_mode_before_payload_conversion() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let err = block_on(crossentropy_builtin(
        None,
        Value::Tensor(Tensor::new_integer(IntegerStorage::U64(vec![1, 0]), vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
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
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let err = block_on(crossentropy_builtin(
            None,
            Value::GpuTensor(input.clone()),
            Value::GpuTensor(input),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
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
            None,
            Value::GpuTensor(prediction),
            Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("Reduction".into()),
                Value::String("bogus".into()),
            ]),
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
            None,
            Value::GpuTensor(prediction.clone()),
            Value::GpuTensor(target.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("DataFormat".into()),
                Value::String("CC".into()),
            ]),
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
            None,
            Value::GpuTensor(prediction),
            Value::Tensor(Tensor::new(vec![2.0, 0.0], vec![1, 2]).unwrap()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
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
            None,
            Value::GpuTensor(predictions.clone()),
            Value::GpuTensor(targets.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(weights.clone()),
                Value::String("ClassificationMode".into()),
                Value::String("multi-label".into()),
                Value::String("Reduction".into()),
                Value::String("none".into()),
            ]),
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
            None,
            Value::GpuTensor(explicit),
            Value::GpuTensor(targets.clone()),
            crate::builtins::common::test_support::builtin_arguments(options()),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:Unsupported"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert!(predictions.allocation_lease().validate_live().is_ok());
        assert!(targets.allocation_lease().validate_live().is_ok());

        provider.reset_telemetry();
        let result = block_on(crossentropy_builtin(
            None,
            Value::GpuTensor(predictions),
            Value::GpuTensor(targets),
            crate::builtins::common::test_support::builtin_arguments(options()),
        ))
        .unwrap();
        let Value::Num(loss) = result else {
            panic!("expected host fallback scalar")
        };
        assert!((loss + 0.8_f64.ln()).abs() < 1e-12);
        assert!(provider.telemetry_snapshot().download_bytes > 0);
    });
}
