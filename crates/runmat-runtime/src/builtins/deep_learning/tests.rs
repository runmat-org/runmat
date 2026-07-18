use super::graph::*;
use super::layers::*;
use super::sequences::*;
use super::training::*;
use futures::executor::block_on;
use runmat_builtins::{CellArray, NumericDType, Tensor, Value};

fn layer_name(value: &Value) -> String {
    let Value::Object(object) = value else {
        panic!("expected object");
    };
    match object.properties.get("Name").unwrap() {
        Value::String(name) => name.clone(),
        other => panic!("expected string name, got {other:?}"),
    }
}

#[test]
fn layer_constructors_preserve_core_properties_and_name_values() {
    let fc = block_on(fully_connected_layer_builtin(
        Value::Num(10.0),
        vec![Value::String("Name".into()), Value::String("fc1".into())],
    ))
    .unwrap();
    let Value::Object(object) = &fc else {
        panic!("expected object");
    };
    assert_eq!(object.class_name, "nnet.cnn.layer.FullyConnectedLayer");
    assert_eq!(object.properties.get("OutputSize"), Some(&Value::Num(10.0)));
    assert_eq!(layer_name(&fc), "fc1");

    let conv = block_on(convolution_1d_layer_builtin(
        Value::Num(3.0),
        Value::Num(16.0),
        vec![
            Value::String("Padding".into()),
            Value::String("causal".into()),
            Value::String("Name".into()),
            Value::String("conv".into()),
        ],
    ))
    .unwrap();
    let Value::Object(object) = &conv else {
        panic!("expected object");
    };
    assert_eq!(object.properties.get("FilterSize"), Some(&Value::Num(3.0)));
    assert_eq!(object.properties.get("NumFilters"), Some(&Value::Num(16.0)));
    assert_eq!(
        object.properties.get("Padding"),
        Some(&Value::String("causal".into()))
    );
    assert_eq!(layer_name(&conv), "conv");
}

#[test]
fn recurrent_and_input_layers_accept_common_shapes() {
    let input = block_on(sequence_input_layer_builtin(
        Value::Tensor(Tensor::new(vec![4.0, 2.0], vec![1, 2]).unwrap()),
        vec![Value::String("Name".into()), Value::String("seq".into())],
    ))
    .unwrap();
    assert_eq!(layer_name(&input), "seq");
    let lstm = block_on(lstm_layer_builtin(
        Value::Num(8.0),
        vec![
            Value::String("OutputMode".into()),
            Value::String("sequence".into()),
        ],
    ))
    .unwrap();
    let Value::Object(object) = &lstm else {
        panic!("expected object");
    };
    assert_eq!(
        object.properties.get("NumHiddenUnits"),
        Some(&Value::Num(8.0))
    );
    assert_eq!(
        object.properties.get("OutputMode"),
        Some(&Value::String("sequence".into()))
    );
}

#[test]
fn training_options_and_layer_graph_materialize_metadata() {
    let opts = block_on(training_options_builtin(
        Value::String("adam".into()),
        vec![
            Value::String("MaxEpochs".into()),
            Value::Num(5.0),
            Value::String("Verbose".into()),
            Value::Bool(false),
        ],
    ))
    .unwrap();
    let Value::Object(options) = opts else {
        panic!("expected options object");
    };
    assert_eq!(
        options.properties.get("SolverName"),
        Some(&Value::String("adam".into()))
    );
    assert_eq!(options.properties.get("MaxEpochs"), Some(&Value::Num(5.0)));
    assert_eq!(options.properties.get("Verbose"), Some(&Value::Bool(false)));

    let relu = block_on(relu_layer_builtin(vec![
        Value::String("Name".into()),
        Value::String("relu".into()),
    ]))
    .unwrap();
    let graph = block_on(layer_graph_builtin(vec![Value::Cell(
        CellArray::new(vec![relu], 1, 1).unwrap(),
    )]))
    .unwrap();
    let Value::Object(graph) = graph else {
        panic!("expected graph object");
    };
    assert_eq!(graph.class_name, "nnet.cnn.LayerGraph");
    let Value::StringArray(names) = graph.properties.get("LayerNames").unwrap() else {
        panic!("expected names");
    };
    assert_eq!(names.data, vec!["relu"]);
}

#[test]
fn analyze_network_counts_layers() {
    let relu = block_on(relu_layer_builtin(vec![
        Value::String("Name".into()),
        Value::String("relu".into()),
    ]))
    .unwrap();
    let analysis = block_on(analyze_network_builtin(Value::Cell(
        CellArray::new(vec![relu], 1, 1).unwrap(),
    )))
    .unwrap();
    let Value::Object(object) = analysis else {
        panic!("expected analysis object");
    };
    assert_eq!(object.properties.get("NumLayers"), Some(&Value::Num(1.0)));
}

#[test]
fn combvec_matches_neural_network_column_order() {
    let out = block_on(combvec_builtin(vec![
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]).unwrap()),
    ]))
    .unwrap();
    let Value::Tensor(t) = out else {
        panic!("expected tensor");
    };
    assert_eq!(t.shape, vec![3, 6]);
    assert_eq!(
        t.data,
        vec![
            1.0, 2.0, 10.0, 3.0, 4.0, 10.0, 1.0, 2.0, 20.0, 3.0, 4.0, 20.0, 1.0, 2.0, 30.0, 3.0,
            4.0, 30.0,
        ]
    );
}

#[test]
fn padsequences_defaults_to_uniform_right_padding() {
    let seqs = Value::Cell(
        CellArray::new(
            vec![
                Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
                Value::Tensor(Tensor::new(vec![3.0], vec![1, 1]).unwrap()),
            ],
            1,
            2,
        )
        .unwrap(),
    );
    let out = block_on(padsequences_builtin(
        seqs,
        vec![
            Value::Num(2.0),
            Value::String("PaddingValue".into()),
            Value::Num(-1.0),
        ],
    ))
    .unwrap();
    let Value::Tensor(tensor) = out else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.shape, vec![1, 2, 1, 2]);
    assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, -1.0]);
}

#[test]
fn padsequences_left_truncation_keeps_tail_for_cell_output() {
    let seqs = Value::Cell(
        CellArray::new(
            vec![
                Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap()),
                Value::Tensor(Tensor::new(vec![4.0, 5.0], vec![1, 2]).unwrap()),
            ],
            1,
            2,
        )
        .unwrap(),
    );
    let out = block_on(padsequences_builtin(
        seqs,
        vec![
            Value::Num(2.0),
            Value::String("Length".into()),
            Value::Num(2.0),
            Value::String("PaddingDirection".into()),
            Value::String("left".into()),
            Value::String("UniformOutput".into()),
            Value::Bool(false),
        ],
    ))
    .unwrap();
    let Value::Cell(cell) = out else {
        panic!("expected cell");
    };
    assert_eq!(cell.shape, vec![1, 2]);
    let Value::Tensor(first) = &cell.data[0] else {
        panic!("expected tensor");
    };
    assert_eq!(first.data, vec![2.0, 3.0]);
}

#[test]
fn dlarray_preserves_data_and_format_labels() {
    let data = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
    let out = block_on(dlarray_builtin(
        data.clone(),
        vec![Value::String("CB".into())],
    ))
    .expect("dlarray");
    let Value::Object(object) = out else {
        panic!("expected object");
    };
    assert_eq!(object.class_name, "dlarray");
    assert_eq!(object.properties.get("Data"), Some(&data));
    assert_eq!(
        object.properties.get("Format"),
        Some(&Value::String("CB".into()))
    );
}

#[test]
fn crossentropy_computes_mean_loss() {
    let out = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        vec![],
    ))
    .expect("crossentropy");
    let Value::Num(loss) = out else {
        panic!("expected scalar");
    };
    assert!((loss - 0.11157177565710488).abs() < 1.0e-12);
}

#[test]
fn adamupdate_computes_bias_corrected_tensor_step() {
    let _guard = crate::output_count::push_output_count(Some(3));
    let out = block_on(adamupdate_builtin(vec![
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.1, -0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap()),
        Value::Num(1.0),
    ]))
    .expect("adamupdate");
    let Value::OutputList(outputs) = out else {
        panic!("expected output list");
    };
    assert_eq!(outputs.len(), 3);
    let Value::Tensor(parameters) = &outputs[0] else {
        panic!("expected tensor parameters");
    };
    assert!((parameters.data[0] - 0.9990000001).abs() < 1.0e-9);
    assert!((parameters.data[1] - 2.00099999995).abs() < 1.0e-9);
    let Value::Tensor(average_grad) = &outputs[1] else {
        panic!("expected tensor averageGrad");
    };
    assert!((average_grad.data[0] - 0.01).abs() < 1.0e-12);
    assert!((average_grad.data[1] + 0.02).abs() < 1.0e-12);
    let Value::Tensor(average_sq_grad) = &outputs[2] else {
        panic!("expected tensor averageSqGrad");
    };
    assert!((average_sq_grad.data[0] - 0.00001).abs() < 1.0e-12);
    assert!((average_sq_grad.data[1] - 0.00004).abs() < 1.0e-12);
}

#[test]
fn adamupdate_default_call_returns_updated_parameters_directly() {
    let out = block_on(adamupdate_builtin(vec![
        Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.1], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
        Value::Num(1.0),
    ]))
    .expect("adamupdate");
    let Value::Tensor(parameters) = out else {
        panic!("expected direct tensor output");
    };
    assert!((parameters.data[0] - 0.9990000001).abs() < 1.0e-9);
}

#[test]
fn adamupdate_preserves_dlarray_wrapper_for_state_outputs() {
    let _guard = crate::output_count::push_output_count(Some(3));
    let params = block_on(dlarray_builtin(
        Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        vec![Value::String("CB".into())],
    ))
    .unwrap();
    let out = block_on(adamupdate_builtin(vec![
        params,
        Value::Tensor(Tensor::new(vec![0.5], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
        Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
        Value::Num(1.0),
    ]))
    .expect("adamupdate");
    let Value::OutputList(outputs) = out else {
        panic!("expected output list");
    };
    let Value::Object(updated) = &outputs[0] else {
        panic!("expected dlarray object");
    };
    assert_eq!(updated.class_name, "dlarray");
    assert_eq!(
        updated.properties.get("Format"),
        Some(&Value::String("CB".into()))
    );
    for value in &outputs[1..] {
        let Value::Object(state) = value else {
            panic!("expected dlarray state object");
        };
        assert_eq!(state.class_name, "dlarray");
        assert_eq!(
            state.properties.get("Format"),
            Some(&Value::String("CB".into()))
        );
    }
}

#[test]
fn adamupdate_rejects_invalid_shapes_outputs_and_hyperparameters() {
    let err = block_on(adamupdate_builtin(vec![
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.1], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap()),
        Value::Num(1.0),
    ]))
    .unwrap_err();
    assert!(err.to_string().contains("grad must match"));

    let err = block_on(adamupdate_builtin(vec![
        Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.1], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
        Value::Num(1.0),
        Value::Num(-0.001),
    ]))
    .unwrap_err();
    assert!(err.to_string().contains("learnRate must be positive"));

    let _guard = crate::output_count::push_output_count(Some(4));
    let err = block_on(adamupdate_builtin(vec![
        Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.1], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
        Value::Num(1.0),
    ]))
    .unwrap_err();
    assert!(err.to_string().contains("at most 3"));
}

#[test]
fn adamupdate_accepts_zero_decay_and_rejects_integer_tensor_state() {
    let out = block_on(adamupdate_builtin(vec![
        Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.1], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
        Value::Num(10_000_000_000.0),
        Value::Num(0.001),
        Value::Num(0.0),
        Value::Num(0.0),
    ]))
    .expect("zero decay and large iteration");
    let Value::Tensor(parameters) = out else {
        panic!("expected tensor parameters");
    };
    assert!(parameters.data[0].is_finite());

    let integer_params = Tensor::new_with_dtype(vec![1.0], vec![1, 1], NumericDType::U8).unwrap();
    let err = block_on(adamupdate_builtin(vec![
        Value::Tensor(integer_params),
        Value::Tensor(Tensor::new(vec![0.1], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
        Value::Num(1.0),
    ]))
    .unwrap_err();
    assert!(err.to_string().contains("double or single"));
}

#[test]
fn dlfeval_dispatches_function_handles() {
    let out = block_on(dlfeval_builtin(vec![
        Value::FunctionHandle("plus".into()),
        Value::Num(2.0),
        Value::Num(3.0),
    ]))
    .expect("dlfeval plus");
    assert_eq!(out, Value::Num(5.0));
}

#[test]
fn dlfeval_preserves_requested_output_count() {
    let _guard = crate::output_count::push_output_count(Some(2));
    let out = block_on(dlfeval_builtin(vec![
        Value::FunctionHandle("deal".into()),
        Value::String("a".into()),
        Value::String("b".into()),
    ]))
    .expect("dlfeval deal");
    let Value::OutputList(outputs) = out else {
        panic!("expected output list");
    };
    assert_eq!(
        outputs,
        vec![Value::String("a".into()), Value::String("b".into())]
    );
}

#[test]
fn dlfeval_registered_dispatch_preserves_requested_outputs() {
    let out = block_on(crate::call_builtin_async_with_outputs(
        "dlfeval",
        &[
            Value::FunctionHandle("deal".into()),
            Value::String("left".into()),
            Value::String("right".into()),
        ],
        2,
    ))
    .expect("registered dlfeval deal");
    let Value::OutputList(outputs) = out else {
        panic!("expected output list");
    };
    assert_eq!(
        outputs,
        vec![Value::String("left".into()), Value::String("right".into())]
    );
}

#[test]
fn dlfeval_rejects_missing_or_non_callable_function() {
    let err = block_on(dlfeval_builtin(vec![])).unwrap_err();
    assert!(err.to_string().contains("expected a function handle"));

    let err = block_on(dlfeval_builtin(vec![Value::Num(1.0)])).unwrap_err();
    assert!(err.to_string().contains("expected a function handle"));

    let err = block_on(dlfeval_builtin(vec![Value::String("@plus".into())])).unwrap_err();
    assert!(err.to_string().contains("expected a function handle"));
}

#[test]
fn dlarray_and_crossentropy_reject_unsupported_forms() {
    let err = block_on(dlarray_builtin(
        Value::Num(1.0),
        vec![Value::String("CB".into()), Value::String("Extra".into())],
    ))
    .unwrap_err();
    assert!(err.to_string().contains("optional dimension-label"));

    let err = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(Vec::new(), vec![1, 0]).unwrap()),
        Value::Tensor(Tensor::new(Vec::new(), vec![1, 0]).unwrap()),
        vec![],
    ))
    .unwrap_err();
    assert!(err.to_string().contains("must not be empty"));
}

#[test]
fn unsupported_training_apis_return_clear_errors() {
    let err = block_on(train_network_builtin(vec![])).unwrap_err();
    assert!(err.to_string().contains("requires optimizer"));
    let err = block_on(export_onnx_network_builtin(vec![])).unwrap_err();
    assert!(err.to_string().contains("ONNX"));
}
