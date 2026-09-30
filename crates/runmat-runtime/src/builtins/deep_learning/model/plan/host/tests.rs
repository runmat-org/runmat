use futures::executor::block_on;
use runmat_value::{CellArray, NumericDType, NumericStorage, Tensor, Value};

use crate::builtins::deep_learning::{layers::*, model::*, object};

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
fn nonsquare_bc_and_cb_use_distinct_physical_channel_axes() {
    let net = network();
    let raw_bc =
        Value::Tensor(Tensor::new(vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0], vec![3, 2]).unwrap());
    let Value::Tensor(raw) = block_on(forward_builtin(net.clone(), raw_bc, vec![])).unwrap() else {
        panic!("raw tensor result expected")
    };
    assert_eq!(raw.shape, vec![3, 1]);
    assert_eq!(raw.materialize_f64(), vec![0.5, 2.5, 4.5]);

    let cb = object(
        "dlarray",
        vec![
            (
                "Data",
                Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap()),
            ),
            ("Format", Value::String("CB".into())),
            ("Labels", Value::String("CB".into())),
        ],
    );
    let Value::Object(wrapped) = block_on(forward_builtin(net, cb, vec![])).unwrap() else {
        panic!("dlarray result expected")
    };
    let Some(Value::Tensor(result)) = wrapped.properties.get("Data") else {
        panic!("tensor data expected")
    };
    assert_eq!(result.shape, vec![1, 3]);
    assert_eq!(result.materialize_f64(), vec![0.5, 2.5, 4.5]);
    assert_eq!(
        wrapped.properties.get("Format"),
        Some(&Value::String("CB".into()))
    );
}

#[test]
fn single_input_projects_double_parameters_only_at_execution_boundary() {
    let net = network();
    let input = Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap());
    let Value::Tensor(output) = block_on(forward_builtin(net.clone(), input, vec![])).unwrap()
    else {
        panic!("tensor result expected")
    };
    assert_eq!(output.numeric_dtype(), NumericDType::F32);
    assert_eq!(output.shape, vec![2, 1]);
    assert_eq!(
        output.into_numeric_storage().unwrap(),
        NumericStorage::F32(vec![-0.5, 0.5])
    );
    let Value::Object(net) = net else {
        panic!("network object")
    };
    let rows = parameters::view(&net, "forward").unwrap();
    assert!(rows
        .iter()
        .filter_map(|row| row.value)
        .all(|value| match value {
            Value::Tensor(tensor) => tensor.numeric_dtype() == NumericDType::F64,
            _ => false,
        }));
}

#[test]
fn stateless_two_output_forward_publishes_canonical_empty_state() {
    let _count = crate::output_count::push_output_count(Some(2));
    let input = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
    let Value::OutputList(outputs) = block_on(forward_builtin(network(), input, vec![])).unwrap()
    else {
        panic!("[Y, state] result expected")
    };
    assert_eq!(outputs.len(), 2);
    assert!(matches!(&outputs[0], Value::Tensor(_)));
    assert!(matches!(&outputs[1], Value::Object(_)));
}
