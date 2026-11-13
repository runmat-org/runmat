use super::*;
use crate::builtins::deep_learning::arguments::DeepLearningObjectEdit as _;
use crate::builtins::deep_learning::autodiff::{
    array_methods::{dlarray_plus_builtin, dlarray_sum_builtin, dlarray_times_builtin},
    gradient::dlgradient_direct,
    tape::{Tape, TapeActivation},
};
use crate::builtins::deep_learning::{model::forward_builtin, object};
use futures::executor::block_on;
use runmat_builtins::DeepLearningLayerKind as K;

mod parents;
mod presentation;

#[test]
fn output_annotation_errors_identify_the_actual_network_caller() {
    for caller in [NetworkCaller::Forward, NetworkCaller::Predict] {
        let network = ObjectInstance::new(model::DLNETWORK_CLASS.owned());
        let nodes = NetworkOutputNodes {
            prediction: None,
            state: vec![(
                StateKey {
                    layer_index: 0,
                    layer_name: "recurrent".into(),
                    role: StateRole::HiddenState,
                    channels: 1,
                },
                0,
            )],
        };
        let error = annotate_outputs(&network, Value::Num(1.0), None, nodes, caller).unwrap_err();
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert!(error.to_string().starts_with(caller.function()));
        let error = attach(Value::String("not numerical".into()), 0, caller).unwrap_err();
        assert!(error.to_string().starts_with(caller.function()));
    }
}

fn layer(kind: K, fields: &[(&str, Value)]) -> Value {
    let mut layer = ObjectInstance::new(kind.class().owned());
    layer
        .replace_dl_property("Name", Value::String(format!("{kind:?}").into()))
        .expect("fixture property");
    for (key, value) in fields {
        layer
            .replace_dl_property(key, value.clone())
            .expect("fixture property");
    }
    Value::Object(layer)
}

fn values(data: Vec<f64>, shape: Vec<usize>, dtype: NumericDType) -> Value {
    Value::Tensor(
        match dtype {
            NumericDType::F32 => {
                Tensor::from_f32(data.into_iter().map(|x| x as f32).collect(), shape)
            }
            NumericDType::F64 => Tensor::new(data, shape),
            _ => panic!("floating fixture"),
        }
        .unwrap(),
    )
}

fn filled(value: f64, shape: &[usize], dtype: NumericDType) -> Value {
    values(vec![value; shape.iter().product()], shape.to_vec(), dtype)
}

fn network(kind: K, dtype: NumericDType) -> crate::execution::RootedValue {
    let fields = match kind {
        K::FullyConnected => vec![
            ("OutputSize", Value::Num(2.0)),
            (
                "Weights",
                values(vec![0.4, -0.2, 0.3, 0.6], vec![2, 2], dtype),
            ),
            ("Bias", filled(0.1, &[2, 1], dtype)),
        ],
        K::Convolution1D => vec![
            ("FilterSize", Value::Num(2.0)),
            ("NumFilters", Value::Num(2.0)),
            ("Weights", filled(0.2, &[2, 2, 2], dtype)),
            ("Bias", filled(0.1, &[1, 2], dtype)),
        ],
        K::Lstm | K::Bilstm => {
            let gates = if kind == K::Lstm { 8 } else { 16 };
            vec![
                ("NumHiddenUnits", Value::Num(2.0)),
                ("InputWeights", filled(0.2, &[gates, 2], dtype)),
                ("RecurrentWeights", filled(0.1, &[gates, 2], dtype)),
                ("Bias", filled(0.1, &[gates, 1], dtype)),
            ]
        }
        K::LayerNormalization => vec![
            ("Scale", filled(1.2, &[2, 1], dtype)),
            ("Offset", filled(0.1, &[2, 1], dtype)),
        ],
        K::Elu => vec![("Alpha", Value::Num(1.2))],
        K::Relu | K::Softmax | K::GlobalAveragePooling1D => Vec::new(),
        _ => panic!("executing layer fixture"),
    };
    let layers = vec![
        layer(K::SequenceInput, &[("InputSize", Value::Num(2.0))]),
        layer(kind, &fields),
    ];
    block_on(model::dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(layers, 2, 1).unwrap(),
    )]))
    .unwrap()
}

fn input(data: Vec<f64>, dtype: NumericDType) -> Value {
    object(
        "dlarray",
        vec![
            ("Data", values(data, vec![2, 1, 3], dtype)),
            ("Format", Value::String("CBT".into())),
            ("Labels", Value::String("CBT".into())),
        ],
    )
    .expect("fixture object")
}

fn tensor(value: &Value) -> &Tensor {
    match value {
        Value::Tensor(tensor) => tensor,
        Value::Object(object) => match object.properties().get("Data") {
            Some(Value::Tensor(tensor)) => tensor,
            _ => panic!("tensor Data"),
        },
        _ => panic!("tensor result"),
    }
}

fn squared_output(net: &Value, data: Vec<f64>, dtype: NumericDType) -> f64 {
    let output = block_on(forward_builtin(
        net.clone(),
        input(data, dtype),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    tensor(&output)
        .materialize_f64()
        .into_iter()
        .map(|x| x * x)
        .sum()
}

#[test]
fn public_forward_tape_uses_every_numerical_layer_and_matches_finite_differences() {
    let data = vec![-0.7, 0.3, 0.5, -0.4, 1.2, 0.8];
    for dtype in [NumericDType::F32, NumericDType::F64] {
        for kind in [
            K::FullyConnected,
            K::Convolution1D,
            K::Lstm,
            K::Bilstm,
            K::LayerNormalization,
            K::GlobalAveragePooling1D,
            K::Relu,
            K::Elu,
            K::Softmax,
        ] {
            let net = network(kind, dtype);
            let gradient = {
                let _activation = TapeActivation::enter(Rc::new(RefCell::new(Tape::new())));
                let x = annotate_dlarray_value(input(data.clone(), dtype)).unwrap();
                let y = block_on(forward_builtin(
                    net.value().clone(),
                    x.clone(),
                    crate::builtins::common::test_support::builtin_arguments(vec![]),
                ))
                .unwrap();
                assert_eq!(tensor(&y).numeric_dtype(), dtype, "{kind:?}");
                let loss = dlarray_sum_builtin(
                    dlarray_times_builtin(y.clone(), y).unwrap(),
                    crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                        "all".into(),
                    )]),
                )
                .unwrap();
                let first = block_on(dlgradient_direct(
                    loss.clone(),
                    crate::builtins::common::test_support::builtin_arguments(vec![x.clone()]),
                ))
                .unwrap();
                let second = block_on(dlgradient_direct(
                    loss,
                    crate::builtins::common::test_support::builtin_arguments(vec![x]),
                ))
                .unwrap();
                assert_eq!(tensor(&first), tensor(&second), "repeat reverse {kind:?}");
                assert_eq!(tensor(&first).numeric_dtype(), dtype);
                assert_eq!(tensor(&first).shape(), &[2, 1, 3]);
                tensor(&first).materialize_f64()
            };
            let step = if dtype == NumericDType::F32 {
                0.001
            } else {
                0.00001
            };
            let tolerance = if dtype == NumericDType::F32 {
                0.003
            } else {
                0.00001
            };
            for index in 0..data.len() {
                let mut upper = data.clone();
                upper[index] += step;
                let mut lower = data.clone();
                lower[index] -= step;
                let expected = (squared_output(net.value(), upper, dtype)
                    - squared_output(net.value(), lower, dtype))
                    / (2.0 * step);
                assert!(
                    (gradient[index] - expected).abs() < tolerance,
                    "{kind:?} {dtype:?} input {index}: {} vs {expected}",
                    gradient[index]
                );
            }
        }
    }
}

#[test]
fn feature_input_and_both_output_layers_preserve_the_public_tape() {
    for output in [K::ClassificationOutput, K::RegressionOutput] {
        let layers = vec![
            layer(K::FeatureInput, &[("InputSize", Value::Num(2.0))]),
            layer(output, &[]),
        ];
        let net = block_on(model::dlnetwork_builtin(vec![Value::Cell(
            CellArray::new(layers, 2, 1).unwrap(),
        )]))
        .unwrap();
        let _activation = TapeActivation::enter(Rc::new(RefCell::new(Tape::new())));
        let x = annotate_dlarray_value(
            object(
                "dlarray",
                vec![
                    (
                        "Data",
                        values(vec![1.0, -2.0], vec![2, 1], NumericDType::F32),
                    ),
                    ("Format", Value::String("CB".into())),
                    ("Labels", Value::String("CB".into())),
                ],
            )
            .expect("fixture object"),
        )
        .unwrap();
        let y = block_on(forward_builtin(
            net.value().clone(),
            x.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        assert_eq!(tensor(&y).as_f32_slice().unwrap(), &[1.0, -2.0]);
        let loss = dlarray_sum_builtin(
            dlarray_times_builtin(y.clone(), y).unwrap(),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "all".into(),
            )]),
        )
        .unwrap();
        let gradient = block_on(dlgradient_direct(
            loss,
            crate::builtins::common::test_support::builtin_arguments(vec![x]),
        ))
        .unwrap();
        assert_eq!(tensor(&gradient).as_f32_slice().unwrap(), &[2.0, -4.0]);
    }
}

fn hidden_state(net: &Value, x: Value) -> Value {
    let outputs = {
        let _count = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        block_on(forward_builtin(
            net.clone(),
            x,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap()
    };
    let Value::OutputList(outputs) = outputs else {
        panic!("[Y, State]")
    };
    let mut outputs = outputs
        .try_into_owned()
        .expect("unique forward outputs")
        .into_iter();
    let Value::Object(network) = net else {
        panic!("network")
    };
    let mut hidden = None;
    let prediction = outputs.next().expect("prediction output");
    model::state::map_result_values(
        network,
        outputs.next().expect("State output"),
        "forward",
        |key, value| {
            if key.role == StateRole::HiddenState {
                hidden = Some(value.clone());
            }
            Ok(value)
        },
    )
    .unwrap();
    drop(prediction);
    hidden.expect("canonical HiddenState output")
}

#[test]
fn returned_recurrent_state_is_an_output_of_the_shared_forward_trace() {
    let data = vec![-0.7, 0.3, 0.5, -0.4, 1.2, 0.8];
    for kind in [K::Lstm, K::Bilstm] {
        let net = network(kind, NumericDType::F64);
        let gradient = {
            let tape = Rc::new(RefCell::new(Tape::new()));
            let _activation = TapeActivation::enter(tape.clone());
            let x = annotate_dlarray_value(input(data.clone(), NumericDType::F64)).unwrap();
            let hidden = hidden_state(net.value(), x.clone());
            let hidden_node = dlarray_node_id(&hidden).expect("State is not a disconnected leaf");
            {
                let nodes = &tape.borrow().nodes;
                let NodeKind::Network {
                    execution: state_trace,
                    output: NetworkOutput::State { .. },
                } = &nodes[hidden_node].kind
                else {
                    panic!("State output node")
                };
                let prediction_trace = nodes
                    .iter()
                    .find_map(|node| match &node.kind {
                        NodeKind::Network {
                            execution,
                            output: NetworkOutput::Prediction,
                        } => Some(execution),
                        _ => None,
                    })
                    .expect("prediction output node");
                assert!(Rc::ptr_eq(state_trace, prediction_trace));
            }
            let loss = dlarray_sum_builtin(
                hidden,
                crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                    "all".into(),
                )]),
            )
            .unwrap();
            let gradient = block_on(dlgradient_direct(
                loss,
                crate::builtins::common::test_support::builtin_arguments(vec![x]),
            ))
            .unwrap();
            tensor(&gradient).materialize_f64()
        };
        for index in 0..data.len() {
            let mut upper = data.clone();
            upper[index] += 0.00001;
            let mut lower = data.clone();
            lower[index] -= 0.00001;
            let sum = |data| {
                tensor(&hidden_state(net.value(), input(data, NumericDType::F64)))
                    .materialize_f64()
                    .into_iter()
                    .sum::<f64>()
            };
            let expected = (sum(upper) - sum(lower)) / 0.00002;
            assert!(
                (gradient[index] - expected).abs() < 0.00001,
                "{kind:?} State input {index}"
            );
        }
    }
}

fn prediction_and_states(net: &Value, x: Value) -> [Value; 3] {
    let outputs = {
        let _count = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        block_on(forward_builtin(
            net.clone(),
            x,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap()
    };
    let Value::OutputList(outputs) = outputs else {
        panic!("[Y, State]")
    };
    let mut outputs = outputs
        .try_into_owned()
        .expect("unique forward outputs")
        .into_iter();
    let Value::Object(network) = net else {
        panic!("network")
    };
    let mut hidden = None;
    let mut cell = None;
    let prediction = outputs.next().expect("prediction output");
    model::state::map_result_values(
        network,
        outputs.next().expect("State output"),
        "forward",
        |key, value| {
            match key.role {
                StateRole::HiddenState => hidden = Some(value.clone()),
                StateRole::CellState => cell = Some(value.clone()),
            }
            Ok(value)
        },
    )
    .unwrap();
    [prediction, hidden.unwrap(), cell.unwrap()]
}

#[test]
fn joint_prediction_hidden_and_cell_loss_runs_one_reverse_per_shared_trace() {
    let data = vec![-0.7, 0.3, 0.5, -0.4, 1.2, 0.8];
    for dtype in [NumericDType::F32, NumericDType::F64] {
        for kind in [K::Lstm, K::Bilstm] {
            let net = network(kind, dtype);
            let analytic = {
                let tape = Rc::new(RefCell::new(Tape::new()));
                let _activation = TapeActivation::enter(tape.clone());
                let x = annotate_dlarray_value(input(data.clone(), dtype)).unwrap();
                let outputs = prediction_and_states(net.value(), x.clone());
                let execution = {
                    let graph = tape.borrow();
                    let id = dlarray_node_id(&outputs[0]).unwrap();
                    let NodeKind::Network { execution, .. } = &graph.nodes[id].kind else {
                        panic!("network prediction")
                    };
                    execution.clone()
                };
                let mut losses = outputs.into_iter().map(|output| {
                    dlarray_sum_builtin(
                        dlarray_times_builtin(output.clone(), output).unwrap(),
                        crate::builtins::common::test_support::builtin_arguments(vec![
                            Value::String("all".into()),
                        ]),
                    )
                    .unwrap()
                });
                let loss = dlarray_plus_builtin(
                    dlarray_plus_builtin(losses.next().unwrap(), losses.next().unwrap()).unwrap(),
                    losses.next().unwrap(),
                )
                .unwrap();
                let first = block_on(dlgradient_direct(
                    loss.clone(),
                    crate::builtins::common::test_support::builtin_arguments(vec![x.clone()]),
                ))
                .unwrap();
                assert_eq!(execution.reverse_calls.get(), 1);
                let second = block_on(dlgradient_direct(
                    loss,
                    crate::builtins::common::test_support::builtin_arguments(vec![x]),
                ))
                .unwrap();
                assert_eq!(execution.reverse_calls.get(), 2);
                assert_eq!(tensor(&first), tensor(&second));
                assert_eq!(tensor(&first).numeric_dtype(), dtype);
                tensor(&first).materialize_f64()
            };
            let step = if dtype == NumericDType::F32 {
                0.001
            } else {
                0.00001
            };
            let tolerance = if dtype == NumericDType::F32 {
                0.003
            } else {
                0.00001
            };
            let objective = |values| {
                prediction_and_states(net.value(), input(values, dtype))
                    .iter()
                    .flat_map(|output| tensor(output).materialize_f64())
                    .map(|value| value * value)
                    .sum::<f64>()
            };
            for index in 0..data.len() {
                let mut upper = data.clone();
                upper[index] += step;
                let mut lower = data.clone();
                lower[index] -= step;
                let expected = (objective(upper) - objective(lower)) / (2.0 * step);
                assert!(
                    (analytic[index] - expected).abs() < tolerance,
                    "{kind:?} {dtype:?} joint loss input {index}: {} vs {expected}",
                    analytic[index]
                );
            }
        }
    }
}
