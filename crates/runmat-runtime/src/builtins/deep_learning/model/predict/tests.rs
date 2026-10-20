use crate::builtins::deep_learning::arguments::DeepLearningObjectEdit as _;
use futures::executor::block_on;
use runmat_accelerate_api::{GpuHandleProvenance, HostTensorView};
use runmat_value::{CellArray, IntegerStorage, Tensor, Value};

use super::*;
use crate::builtins::deep_learning::layers::feature_input_layer_builtin;
use crate::builtins::deep_learning::model::dlnetwork_builtin;
use crate::builtins::deep_learning::tests::fully_connected_layer_builtin;

#[path = "tests/complex.rs"]
mod complex;

fn network() -> crate::execution::RootedValue {
    let input = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let dense = block_on(fully_connected_layer_builtin(
        None,
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
        let network = network();
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
            network.value().clone(),
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
        let network = network();
        let handle = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[1, 2],
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        provider.reset_telemetry();
        let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let error = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
            network.value().clone(),
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
fn direct_network_predict_rejects_excess_outputs_before_provider_access() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let network_owner = network();
        let Value::Object(network) = network_owner.value() else {
            panic!("network object expected")
        };
        let handle = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[1, 2],
            })
            .unwrap()
            .with_provenance(GpuHandleProvenance::Explicit)
            .unwrap();
        provider.reset_telemetry();
        let _outputs = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let error = block_on(predict_deep_learning_object(
            network.clone(),
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
fn two_output_network_predict_returns_state() {
    let network = network();
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let value = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
        network.value().clone(),
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        vec![],
    ))
    .unwrap();
    let Value::OutputList(outputs) = value else {
        panic!("prediction and State outputs expected")
    };
    assert_eq!(outputs.len(), 2);
    assert!(matches!(&outputs[0], Value::Tensor(_)));
    assert!(
        matches!(&outputs[1], Value::Object(object) if crate::builtins::table::is_tabular_object(object))
    );
}

#[test]
fn explicit_resident_prediction_keeps_rows_and_never_downloads() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let network = network();
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
            network.value().clone(),
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
        let network = network();
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
            network.value().clone(),
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
    let network = network();
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let input = Value::Tensor(
        Tensor::new_integer(IntegerStorage::I32(vec![1, 3, 2, 4]), vec![2, 2]).unwrap(),
    );
    let output = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
        network.value().clone(),
        input,
        vec![],
    ))
    .unwrap();
    let Value::Tensor(output) = output else {
        panic!("host score tensor expected")
    };
    assert_eq!(output.shape(), &[2, 1]);
    assert_eq!(output.materialize_f64(), vec![0.5, 2.5]);
}

#[test]
fn formatted_predictor_keeps_its_typed_axis_order() {
    let network = network();
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
    )
    .expect("fixture object");
    let output = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
        network.value().clone(),
        input,
        vec![],
    ))
    .unwrap();
    let Value::Object(output) = output else {
        panic!("formatted dlarray score expected")
    };
    assert!(output.class_name().is(DLARRAY_CLASS));
    assert_eq!(
        output.properties().get("Format"),
        Some(&Value::String("CB".into()))
    );
    assert_eq!(
        output.properties().get("Labels"),
        Some(&Value::String("CB".into()))
    );
    let Some(Value::Tensor(data)) = output.properties().get("Data") else {
        panic!("host score data expected")
    };
    assert_eq!(data.shape(), &[1, 3]);
    assert_eq!(data.materialize_f64(), vec![0.5, 2.5, 4.5]);
}

#[test]
fn unformatted_dlarray_keeps_wrapper_and_same_rank_input_order() {
    let network = network();
    let input = object(
        "dlarray",
        vec![
            (
                "Data",
                Value::Tensor(Tensor::new(vec![1., 3., 5., 2., 4., 6.], vec![3, 2]).unwrap()),
            ),
            ("Format", Value::String(String::new().into())),
            ("Labels", Value::String(String::new().into())),
        ],
    )
    .expect("fixture object");
    let output = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
        network.value().clone(),
        input,
        vec![],
    ))
    .unwrap();
    let Value::Object(output) = output else {
        panic!("unformatted dlarray score expected")
    };
    assert!(output.class_name().is(DLARRAY_CLASS));
    assert_eq!(
        output.properties().get("Format"),
        Some(&Value::String(String::new().into()))
    );
    let Some(Value::Tensor(data)) = output.properties().get("Data") else {
        panic!("unformatted host score expected")
    };
    assert_eq!(data.shape(), &[3, 1]);
    assert_eq!(data.materialize_f64(), vec![0.5, 2.5, 4.5]);
}

#[test]
fn source_dlarray_metadata_is_preserved_and_disagreement_is_rejected() {
    let input = object(
        "dlarray",
        vec![
            (
                "Data",
                Value::Tensor(Tensor::new(vec![1., 2.], vec![2, 1]).unwrap()),
            ),
            ("Format", Value::String("CB".into())),
            ("Labels", Value::String("CB".into())),
            ("TraceMarker", Value::String("retained".into())),
        ],
    )
    .expect("fixture object");
    let prepared = prepare_input(input.clone(), ObservationsIn::Rows).unwrap();
    assert_eq!(prepared, input);
    let prepared = prepare_input(input, ObservationsIn::Columns).unwrap();
    let Value::Object(prepared) = prepared else {
        panic!("dlarray expected")
    };
    assert_eq!(
        prepared.properties().get("TraceMarker"),
        Some(&Value::String("retained".into()))
    );

    let mut disagreement = prepared.clone();
    disagreement
        .replace_dl_property("Labels", Value::String("BC".into()))
        .expect("fixture property");
    let error = prepare_input(Value::Object(disagreement), ObservationsIn::Rows).unwrap_err();
    assert!(error.message().contains("Format and Labels disagree"));

    let mut conflicting = prepared;
    conflicting
        .replace_dl_property("Format", Value::String("BC".into()))
        .expect("fixture property");
    conflicting
        .replace_dl_property("Labels", Value::String("BC".into()))
        .expect("fixture property");
    let error = prepare_input(Value::Object(conflicting), ObservationsIn::Columns).unwrap_err();
    assert!(error.message().contains("conflicts"));
}
