use super::*;
use futures::executor::block_on;
use runmat_value::IntegerStorage;

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
