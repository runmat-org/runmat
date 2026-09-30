use futures::executor::block_on;
use runmat_accelerate_api::{GpuHandleProvenance, HostTensorView};
use runmat_value::{CellArray, IntegerStorage, Tensor, Value};

use super::*;
use crate::builtins::deep_learning::layers::{
    feature_input_layer_builtin, fully_connected_layer_builtin,
};
use crate::builtins::deep_learning::model::dlnetwork_builtin;

fn network() -> Value {
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
fn invalid_observations_option_is_rejected_before_resident_access() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let handle = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[1, 2],
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        provider.reset_telemetry();
        let error = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
            network(),
            Value::GpuTensor(handle),
            vec![
                Value::String("ObservationsIn".into()),
                Value::String("pages".into()),
            ],
        ))
        .unwrap_err();
        assert!(error.message().contains("unsupported ObservationsIn"));
        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.upload_bytes, 0);
        assert_eq!(telemetry.download_bytes, 0);
    });
}

#[test]
fn excess_predict_outputs_fail_before_resident_access() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let handle = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[1, 2],
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        provider.reset_telemetry();
        let _outputs = crate::output_count::push_output_count(Some(2));
        let error = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
            network(),
            Value::GpuTensor(handle),
            vec![],
        ))
        .unwrap_err();
        assert!(error.message().contains("too many output arguments"));
        let telemetry = provider.telemetry_snapshot();
        assert_eq!(telemetry.upload_bytes, 0);
        assert_eq!(telemetry.download_bytes, 0);
    });
}

#[test]
fn explicit_resident_prediction_keeps_rows_and_never_downloads() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let source = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
        let handle = provider
            .upload(&HostTensorView {
                data: &source,
                shape: &[3, 2],
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        provider.reset_telemetry();
        let output = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
            network(),
            Value::GpuTensor(handle),
            vec![],
        ))
        .unwrap();
        let Value::GpuTensor(output) = output else {
            panic!("resident score tensor expected")
        };
        assert_eq!(output.shape(), &[3, 1]);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let gathered =
            crate::builtins::common::test_support::gather(Value::GpuTensor(output)).unwrap();
        assert_eq!(gathered.materialize_f64(), vec![0.5, 2.5, 4.5]);
    });
}

#[test]
fn explicit_column_observations_return_batch_major_scores_without_download() {
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
        provider.reset_telemetry();
        let output = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
            network(),
            Value::GpuTensor(handle),
            vec![
                Value::String("ObservationsIn".into()),
                Value::String("columns".into()),
            ],
        ))
        .unwrap();
        let Value::GpuTensor(output) = output else {
            panic!("resident score tensor expected")
        };
        assert_eq!(output.shape(), &[3, 1]);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let gathered =
            crate::builtins::common::test_support::gather(Value::GpuTensor(output)).unwrap();
        assert_eq!(gathered.materialize_f64(), vec![0.5, 2.5, 4.5]);
    });
}

#[test]
fn documented_integer_predictors_do_not_inherit_forward_extension_gate() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let input = Value::Tensor(
        Tensor::new_integer(IntegerStorage::I32(vec![1, 3, 2, 4]), vec![2, 2]).unwrap(),
    );
    let output = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
        network(),
        input,
        vec![],
    ))
    .unwrap();
    let Value::Tensor(output) = output else {
        panic!("host score tensor expected")
    };
    assert_eq!(output.shape, vec![2, 1]);
    assert_eq!(output.materialize_f64(), vec![0.5, 2.5]);
}

#[test]
fn formatted_predictor_keeps_its_typed_axis_order() {
    let input = object(
        "dlarray",
        vec![
            (
                "Data",
                Value::Tensor(Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]).unwrap()),
            ),
            ("Format", Value::String("CB".into())),
            ("Labels", Value::String("CB".into())),
        ],
    );
    let output = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
        network(),
        input,
        vec![],
    ))
    .unwrap();
    let Value::Tensor(output) = output else {
        panic!("raw score tensor expected")
    };
    assert_eq!(output.shape, vec![3, 1]);
    assert_eq!(output.materialize_f64(), vec![0.5, 2.5, 4.5]);
}
