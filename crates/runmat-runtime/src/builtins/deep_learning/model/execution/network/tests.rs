use futures::executor::block_on;
use runmat_value::{CellArray, IntegerStorage, NumericDType, Tensor, Value};

use super::*;
use crate::builtins::deep_learning::tests::fully_connected_layer_builtin;
use crate::builtins::deep_learning::{layers::*, model::dlnetwork_builtin, object};

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

fn network_object(value: &Value) -> &runmat_value::ObjectInstance {
    let Value::Object(object) = value else {
        panic!("network object expected")
    };
    object
}

#[test]
fn predict_column_input_returns_batch_major_native_single_scores() {
    let network = network();
    let input = object(
        "dlarray",
        vec![
            (
                "Data",
                Value::Tensor(Tensor::from_f32(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]).unwrap()),
            ),
            ("Format", Value::String("CB".into())),
            ("Labels", Value::String("CB".into())),
        ],
    )
    .expect("fixture object");
    let result = block_on(execute_network(NetworkExecutionRequest {
        runtime: None,
        network: network_object(network.value()),
        input: &input,
        caller: NetworkCaller::Predict,
        requested_outputs: None,
        output_layout: NetworkOutputLayout::ObservationMajor,
    }))
    .unwrap();
    assert_eq!(result.output_axes.shape(), &[3, 1]);
    assert_eq!(
        result.output_axes.layout().axes(),
        &[
            runmat_builtins::DeepLearningAxisLabel::Batch,
            runmat_builtins::DeepLearningAxisLabel::Channel,
        ]
    );
    let Value::Tensor(scores) = result.data else {
        panic!("host score tensor expected")
    };
    assert_eq!(scores.numeric_dtype(), NumericDType::F32);
    assert_eq!(scores.shape(), result.output_axes.shape());
    assert_eq!(scores.as_f32_slice().unwrap(), &[0.5, 2.5, 4.5]);
}

#[test]
fn caller_identity_selects_integer_compatibility_without_duplicate_numerics() {
    let network = network();
    let input = object(
        "dlarray",
        vec![
            (
                "Data",
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::I32(vec![1, 3, 2, 4]), vec![2, 2]).unwrap(),
                ),
            ),
            ("Format", Value::String("BC".into())),
            ("Labels", Value::String("BC".into())),
        ],
    )
    .expect("fixture object");
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(execute_network(NetworkExecutionRequest {
        runtime: None,
        network: network_object(network.value()),
        input: &input,
        caller: NetworkCaller::Forward,
        requested_outputs: None,
        output_layout: NetworkOutputLayout::Preserve,
    }))
    .err()
    .expect("forward integer extension should reject in MATLAB mode");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:ForwardIntegerDataExtension")
    );
    let result = block_on(execute_network(NetworkExecutionRequest {
        runtime: None,
        network: network_object(network.value()),
        input: &input,
        caller: NetworkCaller::Predict,
        requested_outputs: None,
        output_layout: NetworkOutputLayout::ObservationMajor,
    }))
    .unwrap();
    let Value::Tensor(scores) = result.data else {
        panic!("score tensor expected")
    };
    assert_eq!(scores.materialize_f64(), vec![0.5, 2.5]);
}
