use futures::executor::block_on;
use runmat_accelerate_api::{
    handle_provenance, GpuHandleProvenance, HostNumericDataView, HostNumericTensorView,
    HostTensorView,
};
use runmat_value::{CellArray, NumericDType, Tensor, Value};

use crate::builtins::deep_learning::{layers::*, model::*, object};

fn dense_network() -> Value {
    let input = block_on(feature_input_layer_builtin(Value::Num(2.0), vec![])).unwrap();
    let dense = block_on(fully_connected_layer_builtin(
        Value::Num(1.0),
        vec![
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![2.0, -1.0], vec![1, 2]).unwrap()),
            Value::String("Bias".into()),
            Value::Tensor(Tensor::new(vec![0.5], vec![1, 1]).unwrap()),
        ],
    ))
    .unwrap();
    block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input, dense], 2, 1).unwrap(),
    )]))
    .unwrap()
}

#[test]
fn explicit_nonsquare_cb_is_transposed_on_provider_and_restored_without_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let source = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let handle = provider
            .upload(&HostTensorView {
                data: &source,
                shape: &[2, 3],
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        let input = object(
            "dlarray",
            vec![
                ("Data", Value::GpuTensor(handle.clone())),
                ("Format", Value::String("CB".into())),
                ("Labels", Value::String("CB".into())),
            ],
        );
        provider.reset_telemetry();
        let Value::Object(result) =
            block_on(forward_builtin(dense_network(), input, vec![])).unwrap()
        else {
            panic!("dlarray output expected")
        };
        let Some(Value::GpuTensor(output)) = result.properties.get("Data") else {
            panic!("resident output expected")
        };
        assert_eq!(output.shape(), &[1, 3]);
        assert_eq!(
            handle_provenance(output),
            Some(GpuHandleProvenance::Explicit)
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert_eq!(
            result.properties.get("Format"),
            Some(&Value::String("CB".into()))
        );
        let gathered =
            crate::builtins::common::test_support::gather(Value::GpuTensor(output.clone()))
                .unwrap();
        assert_eq!(gathered.materialize_f64(), vec![0.5, 2.5, 4.5]);
        let original =
            crate::builtins::common::test_support::gather(Value::GpuTensor(handle)).unwrap();
        assert_eq!(original.materialize_f64(), source);
    });
}

#[test]
fn single_cb_projects_double_parameters_once_and_keeps_native_resident_result() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let source = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let handle = provider
            .upload_numeric(&HostNumericTensorView {
                data: HostNumericDataView::F32(&source),
                shape: &[2, 3],
                storage: runmat_accelerate_api::GpuTensorStorage::Real,
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        let input = object(
            "dlarray",
            vec![
                ("Data", Value::GpuTensor(handle)),
                ("Format", Value::String("CB".into())),
                ("Labels", Value::String("CB".into())),
            ],
        );
        provider.reset_telemetry();
        let Value::Object(result) =
            block_on(forward_builtin(dense_network(), input, vec![])).unwrap()
        else {
            panic!("dlarray output expected")
        };
        let Some(Value::GpuTensor(output)) = result.properties.get("Data") else {
            panic!("resident output expected")
        };
        assert_eq!(output.shape(), &[1, 3]);
        assert_eq!(
            runmat_accelerate_api::handle_precision(output),
            Some(runmat_accelerate_api::ProviderPrecision::F32)
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let gathered =
            crate::builtins::common::test_support::gather(Value::GpuTensor(output.clone()))
                .unwrap();
        assert_eq!(gathered.numeric_dtype(), NumericDType::F32);
        assert_eq!(gathered.materialize_f64(), vec![0.5, 2.5, 4.5]);
    });
}

#[test]
fn cb_softmax_reduces_channels_not_batches() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let input_layer = block_on(feature_input_layer_builtin(Value::Num(2.0), vec![])).unwrap();
        let softmax = block_on(softmax_layer_builtin(vec![])).unwrap();
        let network = block_on(dlnetwork_builtin(vec![Value::Cell(
            CellArray::new(vec![input_layer, softmax], 2, 1).unwrap(),
        )]))
        .unwrap();
        let source = [1.0, 3.0, -2.0, 2.0, 0.0, 4.0];
        let host = object(
            "dlarray",
            vec![
                (
                    "Data",
                    Value::Tensor(Tensor::new(source.to_vec(), vec![2, 3]).unwrap()),
                ),
                ("Format", Value::String("CB".into())),
                ("Labels", Value::String("CB".into())),
            ],
        );
        let Value::Object(expected) =
            block_on(forward_builtin(network.clone(), host, vec![])).unwrap()
        else {
            panic!("host dlarray expected")
        };
        let Some(Value::Tensor(expected)) = expected.properties.get("Data") else {
            panic!("host tensor expected")
        };
        let handle = provider
            .upload(&HostTensorView {
                data: &source,
                shape: &[2, 3],
            })
            .unwrap();
        let resident = object(
            "dlarray",
            vec![
                ("Data", Value::GpuTensor(handle)),
                ("Format", Value::String("CB".into())),
                ("Labels", Value::String("CB".into())),
            ],
        );
        provider.reset_telemetry();
        let Value::Object(actual) = block_on(forward_builtin(network, resident, vec![])).unwrap()
        else {
            panic!("resident dlarray expected")
        };
        let Some(Value::GpuTensor(actual)) = actual.properties.get("Data") else {
            panic!("resident tensor expected")
        };
        assert_eq!(actual.shape(), &[2, 3]);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let actual =
            crate::builtins::common::test_support::gather(Value::GpuTensor(actual.clone()))
                .unwrap();
        for (left, right) in actual
            .materialize_f64()
            .iter()
            .zip(expected.materialize_f64())
        {
            assert!((left - right).abs() < 1.0e-12);
        }
    });
}

#[test]
fn malformed_transpose_result_rejects_only_returned_reference() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let source = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = provider
            .upload(&HostTensorView {
                data: &source,
                shape: &[2, 3],
            })
            .unwrap();
        provider.reset_telemetry();
        let error = super::validate_layout_transpose(input.clone(), &input, provider, "forward")
            .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:gpu:ProviderPayloadMismatch")
        );
        assert_eq!(error.gpu_gather_retry(), crate::GpuGatherRetry::Never);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let retained =
            crate::builtins::common::test_support::gather(Value::GpuTensor(input)).unwrap();
        assert_eq!(retained.materialize_f64(), source);
    });
}
