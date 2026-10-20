//! Raw resident inputs use the same admitted route as dlarray payloads.
use super::*;
use crate::builtins::deep_learning::model::predict::predict_deep_learning_object;

#[test]
fn raw_single_forward_and_predict_keep_explicit_residency_without_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let values = [1.0_f32, 3.0, 5.0, 2.0, 4.0, 6.0];
        let input = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&values),
                shape: &[3, 2],
                storage: runmat_accelerate_api::GpuTensorStorage::Real,
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        for caller in [
            execution::NetworkCaller::Forward,
            execution::NetworkCaller::Predict,
        ] {
            provider.reset_telemetry();
            let network_owner = dense_network();
            let result = match caller {
                execution::NetworkCaller::Forward => block_on(forward_builtin(
                    network_owner.value().clone(),
                    Value::GpuTensor(input.clone()),
                    crate::builtins::common::test_support::builtin_arguments(vec![]),
                ))
                .unwrap(),
                execution::NetworkCaller::Predict => {
                    let Value::Object(network) = network_owner.value() else {
                        panic!("network expected")
                    };
                    block_on(predict_deep_learning_object(
                        network.clone(),
                        Value::GpuTensor(input.clone()),
                        vec![],
                    ))
                    .unwrap()
                    .remove(0)
                }
            };
            let Value::GpuTensor(output) = result else {
                panic!("raw resident output expected for {caller:?}")
            };
            assert_eq!(output.shape(), &[3, 1]);
            assert_eq!(
                handle_provenance(&output),
                Some(GpuHandleProvenance::Explicit)
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            let gathered =
                crate::builtins::common::test_support::gather(Value::GpuTensor(output)).unwrap();
            assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
            assert_eq!(gathered.as_f32_slice().unwrap(), &[0.5, 2.5, 4.5]);
        }
        let retained =
            crate::builtins::common::test_support::gather(Value::GpuTensor(input)).unwrap();
        assert_eq!(retained.as_f32_slice().unwrap(), values);
    });
}

#[test]
fn raw_unsupported_route_respects_explicit_and_automatic_input_intent() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let sequence = block_on(sequence_input_layer_builtin(
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        let recurrent = block_on(lstm_layer_builtin(
            Value::Num(1.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        let network = block_on(dlnetwork_builtin(vec![Value::Cell(
            CellArray::new(vec![sequence, recurrent], 2, 1).unwrap(),
        )]))
        .unwrap();
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = [2, 1, 3];
        let Value::Tensor(expected) = block_on(forward_builtin(
            network.value().clone(),
            Value::Tensor(Tensor::new(values.to_vec(), shape.to_vec()).unwrap()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap() else {
            panic!("host recurrent output expected")
        };
        let handle = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &shape,
            })
            .unwrap();
        let explicit = handle
            .clone()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        provider.reset_telemetry();
        let error = block_on(forward_builtin(
            network.value().clone(),
            Value::GpuTensor(explicit),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:Unsupported"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert_eq!(provider.telemetry_snapshot().upload_bytes, 0);

        let automatic = handle
            .clone()
            .with_provenance(GpuHandleProvenance::Automatic)
            .unwrap();
        let Value::Tensor(actual) = block_on(forward_builtin(
            network.value().clone(),
            Value::GpuTensor(automatic),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap() else {
            panic!("automatic unsupported route may use host output")
        };
        assert!(provider.telemetry_snapshot().download_bytes > 0);
        assert_eq!(actual.numeric_dtype(), expected.numeric_dtype());
        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(actual.materialize_f64(), expected.materialize_f64());
        let original =
            crate::builtins::common::test_support::gather(Value::GpuTensor(handle)).unwrap();
        assert_eq!(original.materialize_f64(), values);
    });
}
