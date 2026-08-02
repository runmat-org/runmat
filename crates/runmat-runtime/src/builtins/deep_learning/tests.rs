use super::graph::*;
use super::layers::*;
use super::losses::*;
use super::model::*;
use super::object;
use super::onnx::*;
use super::sequences::*;
use super::supervised::*;
use super::training::*;
use futures::executor::block_on;
use runmat_accelerate_api::{handle_precision, handle_storage, GpuTensorStorage, HostTensorView};
use runmat_builtins::{
    CellArray, IntValue, IntegerStorage, LogicalArray, NumericDType, NumericStorage,
    ObjectInstance, StringArray, StructValue, Tensor, Value,
};

#[runmat_macros::runtime_builtin(
    name = "__deep_learning_square_grad",
    category = "deep_learning/tests",
    summary = "Test helper for dlarray autodiff.",
    keywords = "test,dlgradient",
    type_resolver(super::any_type),
    descriptor(crate::builtins::deep_learning::DLFEVAL_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::tests"
)]
async fn deep_learning_square_grad_helper(x: Value) -> crate::BuiltinResult<Value> {
    let y = crate::call_builtin_async("times", &[x.clone(), x.clone()]).await?;
    let loss = crate::call_builtin_async("sum", &[y, Value::String("all".into())]).await?;
    let _guard = crate::output_count::push_output_count(Some(1));
    let grad = super::autodiff::dlgradient_builtin(loss.clone(), vec![x]).await?;
    Ok(Value::OutputList(vec![loss, grad]))
}

#[runmat_macros::runtime_builtin(
    name = "__deep_learning_network_grad",
    category = "deep_learning/tests",
    summary = "Test helper for dlnetwork autodiff.",
    keywords = "test,dlnetwork,dlgradient",
    type_resolver(super::any_type),
    descriptor(crate::builtins::deep_learning::DLFEVAL_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::tests"
)]
async fn deep_learning_network_grad_helper(net: Value, x: Value) -> crate::BuiltinResult<Value> {
    let y = forward_builtin(net.clone(), x, vec![]).await?;
    let loss = super::autodiff::dlarray_sum_builtin(y, vec![Value::String("all".into())])?;
    super::autodiff::dlgradient_builtin(loss, vec![net]).await
}

fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>, poison: f64) -> Value {
    let mut tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
    tensor.data.fill(poison);
    Value::Tensor(tensor)
}

fn cleared_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let mut tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
    tensor.data.clear();
    Value::Tensor(tensor)
}

#[test]
fn dlarray_node_ids_preserve_typed_integer_bounds() {
    let node_id = |value| {
        let mut object = ObjectInstance::new("dlarray".to_string());
        object
            .properties
            .insert("__runmat_ad_node".to_string(), value);
        super::autodiff::dlarray_node_id(&Value::Object(object))
    };

    assert_eq!(node_id(Value::Int(IntValue::U16(7))), Some(7));
    assert_eq!(node_id(Value::Num(7.0)), Some(7));
    assert_eq!(node_id(Value::Int(IntValue::I8(-1))), None);
    assert_eq!(
        node_id(Value::Int(IntValue::U64(u64::MAX))),
        Some(usize::MAX)
    );
    assert_eq!(node_id(Value::Num(7.5)), None);
    assert_eq!(node_id(Value::Num(f64::INFINITY)), None);

    let boundary = node_id(Value::Num(usize::MAX as f64));
    if usize::BITS == 64 {
        assert_eq!(boundary, None);
    } else {
        assert_eq!(boundary, Some(usize::MAX));
    }
    assert_eq!(node_id(Value::Num((usize::MAX as f64) + 1.0)), None);
}

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
fn layer_constructors_read_typed_integer_size_storage_exactly() {
    let output_size = cleared_int_tensor(IntegerStorage::U16(vec![10]), vec![1, 1]);
    let fc = block_on(fully_connected_layer_builtin(output_size, vec![])).unwrap();
    let Value::Object(object) = &fc else {
        panic!("expected object");
    };
    assert_eq!(object.properties.get("OutputSize"), Some(&Value::Num(10.0)));

    let input_size = poisoned_int_tensor(IntegerStorage::U16(vec![2, 3]), vec![1, 2], f64::NAN);
    let input = block_on(feature_input_layer_builtin(input_size, vec![])).unwrap();
    let Value::Object(object) = &input else {
        panic!("expected object");
    };
    assert_eq!(
        object.properties.get("InputSize"),
        Some(&Value::Tensor(
            Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap()
        ))
    );

    let exact = (1_u64 << 53) + 1;
    let large_input = poisoned_int_tensor(IntegerStorage::U64(vec![exact]), vec![1, 1], f64::NAN);
    let large = block_on(feature_input_layer_builtin(large_input, vec![]));
    if usize::BITS == 64 {
        assert!(large.is_ok());
    } else {
        assert!(large.is_err());
    }
}

#[test]
fn deep_learning_integer_parsers_reject_unrepresentable_double_bounds() {
    let fc_boundary = block_on(fully_connected_layer_builtin(
        Value::Num(usize::MAX as f64),
        vec![],
    ));
    if usize::BITS == 64 {
        assert!(fc_boundary.is_err());
    } else {
        assert!(fc_boundary.is_ok());
    }
    assert!(block_on(fully_connected_layer_builtin(
        Value::Num((usize::MAX as f64) + 1.0),
        vec![],
    ))
    .is_err());
    assert!(block_on(fully_connected_layer_builtin(
        Value::Tensor(Tensor::new(vec![f64::NAN], vec![1, 1]).expect("nan scalar")),
        vec![],
    ))
    .is_err());

    let vector_boundary = Tensor::new(vec![usize::MAX as f64], vec![1, 1]).expect("input size");
    let feature_boundary = block_on(feature_input_layer_builtin(
        Value::Tensor(vector_boundary),
        vec![],
    ));
    if usize::BITS == 64 {
        assert!(feature_boundary.is_err());
    } else {
        assert!(feature_boundary.is_ok());
    }

    let mut layer = ObjectInstance::new("nnet.cnn.layer.FullyConnectedLayer".to_string());
    let exact = (1_u64 << 53) + 1;
    layer
        .properties
        .insert("OutputSize".to_string(), Value::Int(IntValue::U64(exact)));
    let parsed = positive_property_usize(&layer, "OutputSize", "forward");
    if usize::BITS == 64 {
        assert_eq!(parsed.unwrap(), exact as usize);
    } else {
        assert!(parsed.is_err());
    }
}

#[test]
fn structural_integer_parsers_ignore_poisoned_mirrors_for_every_storage_class() {
    let storages = [
        IntegerStorage::I8(vec![1]),
        IntegerStorage::I16(vec![1]),
        IntegerStorage::I32(vec![1]),
        IntegerStorage::I64(vec![1]),
        IntegerStorage::U8(vec![1]),
        IntegerStorage::U16(vec![1]),
        IntegerStorage::U32(vec![1]),
        IntegerStorage::U64(vec![1]),
    ];

    for storage in storages {
        let value = poisoned_int_tensor(storage, vec![1, 1], f64::NAN);
        assert_eq!(super::positive_usize(&value, "test", "size").unwrap(), 1);
        assert!(super::logical_scalar(&value, "test", "flag").unwrap());
        assert_eq!(integer_scalar(&value, "BatchSize").unwrap(), 1);
        assert!(matches!(
            SequenceLength::parse(&value).unwrap(),
            SequenceLength::Fixed(1)
        ));
    }
}

#[test]
fn export_onnx_integer_options_reject_typed_values_beyond_i64() {
    let value = poisoned_int_tensor(
        IntegerStorage::U64(vec![(i64::MAX as u64) + 1]),
        vec![1, 1],
        f64::NAN,
    );
    assert!(integer_scalar(&value, "BatchSize").is_err());
    assert!(integer_scalar(&Value::Num(i64::MAX as f64), "BatchSize").is_err());
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
fn dlnetwork_initialize_option_reads_typed_integer_storage_exactly() {
    let relu = block_on(relu_layer_builtin(vec![
        Value::String("Name".into()),
        Value::String("relu".into()),
    ]))
    .unwrap();
    let initialize = cleared_int_tensor(IntegerStorage::U8(vec![0]), vec![1, 1]);
    let net = block_on(dlnetwork_builtin(vec![
        Value::Cell(CellArray::new(vec![relu], 1, 1).unwrap()),
        Value::String("Initialize".into()),
        initialize,
    ]))
    .unwrap();
    let Value::Object(object) = net else {
        panic!("expected network object");
    };
    assert_eq!(object.class_name, "dlnetwork");
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

fn feature_network_layers() -> Value {
    let input = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        vec![
            Value::String("Name".into()),
            Value::String("features".into()),
        ],
    ))
    .unwrap();
    let fc = block_on(fully_connected_layer_builtin(
        Value::Num(2.0),
        vec![
            Value::String("Name".into()),
            Value::String("fc".into()),
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![1.0, -1.0, 0.0, 1.0], vec![2, 2]).unwrap()),
            Value::String("Bias".into()),
            Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![2, 1]).unwrap()),
        ],
    ))
    .unwrap();
    let softmax = block_on(softmax_layer_builtin(vec![
        Value::String("Name".into()),
        Value::String("prob".into()),
    ]))
    .unwrap();
    Value::Cell(CellArray::new(vec![input, fc, softmax], 3, 1).unwrap())
}

fn dense_feature_network_layers() -> Value {
    let input = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        vec![
            Value::String("Name".into()),
            Value::String("features".into()),
        ],
    ))
    .unwrap();
    let first = block_on(fully_connected_layer_builtin(
        Value::Num(2.0),
        vec![
            Value::String("Name".into()),
            Value::String("first".into()),
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![1.0, -1.0, 0.5, 2.0], vec![2, 2]).unwrap()),
            Value::String("Bias".into()),
            Value::Tensor(Tensor::new(vec![0.25, -0.5], vec![2, 1]).unwrap()),
        ],
    ))
    .unwrap();
    let elu = block_on(elu_layer_builtin(vec![
        Value::Num(1.25),
        Value::String("Name".into()),
        Value::String("elu".into()),
    ]))
    .unwrap();
    let relu = block_on(relu_layer_builtin(vec![
        Value::String("Name".into()),
        Value::String("relu".into()),
    ]))
    .unwrap();
    let second = block_on(fully_connected_layer_builtin(
        Value::Num(1.0),
        vec![
            Value::String("Name".into()),
            Value::String("second".into()),
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![2.0, -0.25], vec![1, 2]).unwrap()),
            Value::String("Bias".into()),
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        ],
    ))
    .unwrap();
    Value::Cell(CellArray::new(vec![input, first, elu, relu, second], 5, 1).unwrap())
}

fn elu_feature_network_layers() -> Value {
    let input = block_on(feature_input_layer_builtin(Value::Num(2.0), vec![])).unwrap();
    let elu = block_on(elu_layer_builtin(vec![Value::Num(1.25)])).unwrap();
    Value::Cell(CellArray::new(vec![input, elu], 2, 1).unwrap())
}

#[test]
fn dlnetwork_materializes_metadata_and_learnables() {
    let net = block_on(dlnetwork_builtin(vec![feature_network_layers()])).unwrap();
    let Value::Object(object) = net else {
        panic!("expected dlnetwork object");
    };
    assert_eq!(object.class_name, "dlnetwork");
    assert_eq!(
        object.properties.get("Initialized"),
        Some(&Value::Bool(true))
    );
    let Value::StringArray(names) = object.properties.get("LayerNames").unwrap() else {
        panic!("expected layer names");
    };
    assert_eq!(names.data, vec!["features", "fc", "prob"]);
    let Value::Struct(learnables) = object.properties.get("Learnables").unwrap() else {
        panic!("expected learnables struct");
    };
    let Value::Cell(values) = learnables.fields.get("Value").unwrap() else {
        panic!("expected learnable values");
    };
    assert_eq!(values.data.len(), 2);
}

#[test]
fn dlnetwork_initialize_false_preserves_uninitialized_learnables() {
    let input = block_on(feature_input_layer_builtin(Value::Num(2.0), vec![])).unwrap();
    let fc = block_on(fully_connected_layer_builtin(Value::Num(2.0), vec![])).unwrap();
    let layers = Value::Cell(CellArray::new(vec![input, fc], 2, 1).unwrap());
    let net = block_on(dlnetwork_builtin(vec![
        layers,
        Value::String("Initialize".into()),
        Value::Bool(false),
    ]))
    .unwrap();
    let Value::Object(object) = &net else {
        panic!("expected dlnetwork object");
    };
    assert_eq!(
        object.properties.get("Initialized"),
        Some(&Value::Bool(false))
    );
    let Value::Struct(learnables) = object.properties.get("Learnables").unwrap() else {
        panic!("expected learnables struct");
    };
    let Value::Cell(values) = learnables.fields.get("Value").unwrap() else {
        panic!("expected learnable values");
    };
    assert!(values.data.is_empty());

    let err = block_on(forward_builtin(
        net,
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        vec![],
    ))
    .unwrap_err();
    assert!(err.message.contains("layer is missing Weights"));
}

#[test]
fn forward_and_predict_execute_supported_feedforward_networks() {
    let net = block_on(dlnetwork_builtin(vec![feature_network_layers()])).unwrap();
    let input = Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap());
    let out = block_on(forward_builtin(net.clone(), input.clone(), vec![])).unwrap();
    let Value::Tensor(tensor) = out else {
        panic!("expected tensor output");
    };
    assert_eq!(tensor.shape, vec![2, 2]);
    assert!((tensor.data[0] - 0.5).abs() < 1.0e-12);
    assert!((tensor.data[1] - 0.8807970779778823).abs() < 1.0e-12);
    assert!((tensor.data[2] - 0.5).abs() < 1.0e-12);
    assert!((tensor.data[3] - 0.11920292202211755).abs() < 1.0e-12);

    let predicted = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
        net,
        input,
        vec![],
    ))
    .unwrap();
    let Value::Tensor(predicted) = predicted else {
        panic!("expected predict tensor");
    };
    assert_eq!(predicted.shape, vec![2, 2]);
    assert!((predicted.data[1] - 0.8807970779778823).abs() < 1.0e-12);
}

#[test]
fn forward_preserves_native_single_through_fully_connected_and_softmax() {
    let input_layer = block_on(feature_input_layer_builtin(Value::Num(2.0), vec![])).unwrap();
    let fully_connected = block_on(fully_connected_layer_builtin(
        Value::Num(2.0),
        vec![
            Value::String("Weights".into()),
            Value::Tensor(
                Tensor::from_f32(vec![1.0, -1.0, 0.5, 2.0], vec![2, 2]).expect("weights"),
            ),
            Value::String("Bias".into()),
            Value::Tensor(Tensor::from_f32(vec![0.0, 0.0], vec![2, 1]).expect("bias")),
        ],
    ))
    .unwrap();
    let softmax = block_on(softmax_layer_builtin(vec![])).unwrap();
    let network = block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input_layer, fully_connected, softmax], 3, 1).unwrap(),
    )]))
    .unwrap();
    let input =
        Value::Tensor(Tensor::from_f32(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).expect("input"));
    let output = block_on(forward_builtin(network, input, vec![])).expect("forward");
    let Value::Tensor(output) = output else {
        panic!("expected tensor output");
    };
    let NumericStorage::F32(values) = output.into_numeric_storage().expect("output storage") else {
        panic!("expected native single output");
    };
    assert_eq!(values.len(), 4);
    assert!((values[0] - 0.26894143).abs() < 1.0e-6, "{values:?}");
    assert!((values[1] - 0.5).abs() < 1.0e-6, "{values:?}");
    assert!((values[2] - 0.7310586).abs() < 1.0e-6, "{values:?}");
    assert!((values[3] - 0.5).abs() < 1.0e-6, "{values:?}");
}

#[test]
fn forward_and_predict_read_typed_integer_storage_exactly() {
    let input_layer = block_on(feature_input_layer_builtin(Value::Num(2.0), vec![])).unwrap();
    let fc = block_on(fully_connected_layer_builtin(
        Value::Num(1.0),
        vec![
            Value::String("Weights".into()),
            poisoned_int_tensor(IntegerStorage::I32(vec![10, 1]), vec![1, 2], 99.0),
            Value::String("Bias".into()),
            poisoned_int_tensor(IntegerStorage::I32(vec![5]), vec![1, 1], 99.0),
        ],
    ))
    .unwrap();
    let net = block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input_layer, fc], 2, 1).unwrap(),
    )]))
    .unwrap();
    let input = poisoned_int_tensor(IntegerStorage::I32(vec![1, 2, 3, 4]), vec![2, 2], 99.0);

    let out = block_on(forward_builtin(net.clone(), input.clone(), vec![])).unwrap();
    let Value::Tensor(tensor) = out else {
        panic!("expected forward tensor output");
    };
    assert_eq!(tensor.shape, vec![2, 1]);
    assert_eq!(tensor.data, vec![18.0, 29.0]);

    let predicted = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
        net,
        input,
        vec![],
    ))
    .unwrap();
    let Value::Tensor(predicted) = predicted else {
        panic!("expected predict tensor output");
    };
    assert_eq!(predicted.shape, vec![2, 1]);
    assert_eq!(predicted.data, vec![18.0, 29.0]);
}

#[test]
fn predict_observations_in_columns_reads_typed_integer_storage_exactly() {
    let input_layer = block_on(feature_input_layer_builtin(Value::Num(2.0), vec![])).unwrap();
    let fc = block_on(fully_connected_layer_builtin(
        Value::Num(1.0),
        vec![
            Value::String("Weights".into()),
            poisoned_int_tensor(IntegerStorage::I32(vec![10, 1]), vec![1, 2], 99.0),
            Value::String("Bias".into()),
            poisoned_int_tensor(IntegerStorage::I32(vec![5]), vec![1, 1], 99.0),
        ],
    ))
    .unwrap();
    let net = block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input_layer, fc], 2, 1).unwrap(),
    )]))
    .unwrap();
    let input = poisoned_int_tensor(IntegerStorage::I32(vec![1, 2, 3, 4]), vec![2, 2], 99.0);

    let predicted = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
        net,
        input,
        vec![
            Value::String("ObservationsIn".into()),
            Value::String("columns".into()),
        ],
    ))
    .unwrap();

    let Value::Tensor(predicted) = predicted else {
        panic!("expected predict tensor output");
    };
    assert_eq!(predicted.shape, vec![2, 1]);
    assert_eq!(predicted.data, vec![17.0, 39.0]);
    assert!(predicted.integer_storage().is_none());
}

#[test]
fn forward_preserves_dlarray_wrapper() {
    let net = block_on(dlnetwork_builtin(vec![feature_network_layers()])).unwrap();
    let data = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
    let dl = block_on(dlarray_builtin(data, vec![Value::String("CB".into())])).unwrap();
    let out = block_on(forward_builtin(net, dl, vec![])).unwrap();
    let Value::Object(object) = out else {
        panic!("expected dlarray output");
    };
    assert_eq!(object.class_name, "dlarray");
    assert_eq!(
        object.properties.get("Format"),
        Some(&Value::String("CB".into()))
    );
    assert!(matches!(
        object.properties.get("Data"),
        Some(Value::Tensor(_))
    ));
}

#[test]
fn forward_runs_fully_connected_gpu_dlarray_without_host_gather() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let net = block_on(dlnetwork_builtin(vec![dense_feature_network_layers()])).unwrap();
        let cpu_input = Value::Tensor(Tensor::new(vec![1.0, 2.0, -1.0, 0.5], vec![2, 2]).unwrap());
        let cpu = block_on(forward_builtin(net.clone(), cpu_input, vec![])).unwrap();
        let Value::Tensor(expected) = cpu else {
            panic!("expected CPU tensor output");
        };
        let shape = [2usize, 2usize];
        let handle = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0, -1.0, 0.5],
                shape: &shape,
            })
            .expect("upload");
        provider.reset_telemetry();
        let dl = block_on(dlarray_builtin(
            Value::GpuTensor(handle),
            vec![Value::String("CB".into())],
        ))
        .expect("gpu dlarray");

        let output = block_on(forward_builtin(net, dl, vec![])).expect("GPU forward");
        let Value::Object(output) = output else {
            panic!("expected GPU dlarray output");
        };
        let Some(Value::GpuTensor(handle)) = output.properties.get("Data") else {
            panic!("expected resident GPU output data");
        };
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let actual =
            crate::builtins::common::test_support::gather(Value::GpuTensor(handle.clone()))
                .expect("gather GPU output for parity assertion");
        assert_eq!(actual.shape, expected.shape);
        assert_eq!(actual.data, expected.data);
    });
}

#[test]
fn forward_runs_arbitrary_alpha_elu_gpu_dlarray_with_cpu_parity() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let net = block_on(dlnetwork_builtin(vec![elu_feature_network_layers()])).unwrap();
        let values = vec![-2.0, 0.0, 3.0, f64::NAN];
        let cpu = block_on(forward_builtin(
            net.clone(),
            Value::Tensor(Tensor::new(values.clone(), vec![2, 2]).unwrap()),
            vec![],
        ))
        .unwrap();
        let Value::Tensor(expected) = cpu else {
            panic!("expected CPU tensor output");
        };
        let shape = [2usize, 2usize];
        let handle = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &shape,
            })
            .expect("upload");
        provider.reset_telemetry();
        let dl = block_on(dlarray_builtin(Value::GpuTensor(handle), vec![])).unwrap();
        let output = block_on(forward_builtin(net, dl, vec![])).expect("GPU ELU forward");
        let Value::Object(output) = output else {
            panic!("expected GPU dlarray output");
        };
        let Some(Value::GpuTensor(handle)) = output.properties.get("Data") else {
            panic!("expected resident GPU output data");
        };
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let actual =
            crate::builtins::common::test_support::gather(Value::GpuTensor(handle.clone()))
                .expect("gather GPU output for parity assertion");
        assert_eq!(actual.shape, expected.shape);
        for (actual, expected) in actual.data.iter().zip(&expected.data) {
            assert!(actual == expected || (actual.is_nan() && expected.is_nan()));
        }
    });
}

#[test]
fn forward_runs_softmax_gpu_dlarray_without_host_gather() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let net = block_on(dlnetwork_builtin(vec![feature_network_layers()])).unwrap();
        let values = [1.0, 2.0, -1.0, 0.5];
        let cpu = block_on(forward_builtin(
            net.clone(),
            Value::Tensor(Tensor::new(values.to_vec(), vec![2, 2]).unwrap()),
            vec![],
        ))
        .expect("CPU forward");
        let Value::Tensor(expected) = cpu else {
            panic!("expected CPU tensor output");
        };
        let shape = [2usize, 2usize];
        let handle = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &shape,
            })
            .expect("upload");
        provider.reset_telemetry();
        let dl = block_on(dlarray_builtin(
            Value::GpuTensor(handle),
            vec![Value::String("CB".into())],
        ))
        .expect("gpu dlarray");

        let output = block_on(forward_builtin(net, dl, vec![])).expect("GPU softmax forward");
        let Value::Object(output) = output else {
            panic!("expected GPU dlarray output");
        };
        let Some(Value::GpuTensor(handle)) = output.properties.get("Data") else {
            panic!("expected resident GPU output data");
        };
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let actual =
            crate::builtins::common::test_support::gather(Value::GpuTensor(handle.clone()))
                .expect("gather GPU output for parity assertion");
        assert_eq!(actual.shape, expected.shape);
        for (actual, expected) in actual.data.iter().zip(&expected.data) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
    });
}

#[test]
fn forward_reports_invalid_gpu_softmax_without_host_gather() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let net = block_on(dlnetwork_builtin(vec![feature_network_layers()])).unwrap();
        let values = [f64::INFINITY, 0.0, 0.0, 1.0];
        let shape = [2usize, 2usize];
        let handle = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &shape,
            })
            .expect("upload");
        provider.reset_telemetry();
        let dl = block_on(dlarray_builtin(Value::GpuTensor(handle), vec![])).expect("gpu dlarray");

        let error = block_on(forward_builtin(net, dl, vec![]))
            .expect_err("infinite Softmax input must not be normalized");

        assert!(error
            .to_string()
            .contains("softmax produced invalid normalization"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn forward_rejects_complex_gpu_dlarray_before_dense_dispatch() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let net = block_on(dlnetwork_builtin(vec![dense_feature_network_layers()])).unwrap();
        let shape = [1usize, 2usize];
        let handle = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &shape,
            })
            .expect("upload");
        runmat_accelerate_api::set_handle_storage(&handle, GpuTensorStorage::ComplexInterleaved);
        provider.reset_telemetry();
        let dl = block_on(dlarray_builtin(
            Value::GpuTensor(handle),
            vec![Value::String("CB".into())],
        ))
        .expect("gpu dlarray");

        let err = block_on(forward_builtin(net, dl, vec![])).unwrap_err();

        assert!(err.to_string().contains("real dlarray input"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn export_onnx_network_writes_supported_feedforward_graph() {
    let input = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        vec![Value::String("Name".into()), Value::String("input".into())],
    ))
    .unwrap();
    let fc = block_on(fully_connected_layer_builtin(
        Value::Num(2.0),
        vec![
            Value::String("Name".into()),
            Value::String("dense".into()),
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![1.0, -1.0, 0.5, 2.0], vec![2, 2]).unwrap()),
            Value::String("Bias".into()),
            Value::Tensor(Tensor::new(vec![0.25, -0.25], vec![2, 1]).unwrap()),
        ],
    ))
    .unwrap();
    let relu = block_on(relu_layer_builtin(vec![
        Value::String("Name".into()),
        Value::String("relu".into()),
    ]))
    .unwrap();
    let elu = block_on(elu_layer_builtin(vec![
        Value::Num(1.25),
        Value::String("Name".into()),
        Value::String("elu".into()),
    ]))
    .unwrap();
    let softmax = block_on(softmax_layer_builtin(vec![
        Value::String("Name".into()),
        Value::String("prob".into()),
    ]))
    .unwrap();
    let output = block_on(classification_layer_builtin(vec![
        Value::String("Name".into()),
        Value::String("class".into()),
    ]))
    .unwrap();
    let net = block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input, fc, relu, elu, softmax, output], 6, 1).unwrap(),
    )]))
    .unwrap();

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("network.onnx");
    let result = block_on(export_onnx_network_builtin(vec![
        net,
        Value::String(path.to_string_lossy().into_owned()),
        Value::String("OpsetVersion".into()),
        Value::Num(14.0),
        Value::String("BatchSize".into()),
        Value::Num(4.0),
        Value::String("InputNames".into()),
        Value::String("features".into()),
        Value::String("OutputNames".into()),
        Value::String("scores".into()),
    ]))
    .unwrap();
    assert_eq!(result, Value::OutputList(Vec::new()));

    let bytes = std::fs::read(path).unwrap();
    assert!(bytes.len() > 200);
    for needle in [
        "runmat",
        "RunMatExportedNetwork",
        "Gemm",
        "Relu",
        "Elu",
        "Softmax",
        "Identity",
        "features",
        "scores",
        "dense.Weights",
        "dense.Bias",
    ] {
        assert!(
            contains_bytes(&bytes, needle.as_bytes()),
            "missing ONNX payload string {needle}"
        );
    }
}

#[test]
fn export_onnx_options_read_typed_integer_storage_exactly() {
    let net = block_on(dlnetwork_builtin(vec![feature_network_layers()])).unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("typed-options.onnx");
    let result = block_on(export_onnx_network_builtin(vec![
        net,
        Value::String(path.to_string_lossy().into_owned()),
        Value::String("OpsetVersion".into()),
        cleared_int_tensor(IntegerStorage::U16(vec![14]), vec![1, 1]),
        Value::String("BatchSize".into()),
        cleared_int_tensor(IntegerStorage::U16(vec![4]), vec![1, 1]),
        Value::String("Verbose".into()),
        cleared_int_tensor(IntegerStorage::U8(vec![0]), vec![1, 1]),
    ]))
    .unwrap();
    assert_eq!(result, Value::OutputList(Vec::new()));
    assert!(path.exists());
}

#[test]
fn export_onnx_network_rejects_unexportable_forms_deterministically() {
    let input = block_on(sequence_input_layer_builtin(Value::Num(2.0), vec![])).unwrap();
    let lstm = block_on(lstm_layer_builtin(Value::Num(4.0), vec![])).unwrap();
    let net = object(
        "dlnetwork",
        vec![(
            "Layers",
            Value::Cell(CellArray::new(vec![input, lstm], 2, 1).unwrap()),
        )],
    );
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("unsupported.onnx");
    let err = block_on(export_onnx_network_builtin(vec![
        net,
        Value::String(path.to_string_lossy().into_owned()),
    ]))
    .unwrap_err();
    assert!(err
        .message
        .contains("is not supported for RunMat forward execution"));

    let net = block_on(dlnetwork_builtin(vec![feature_network_layers()])).unwrap();
    let _guard = crate::output_count::push_output_count(Some(1));
    let err = block_on(export_onnx_network_builtin(vec![
        net,
        Value::String(tmp.path().join("out.onnx").to_string_lossy().into_owned()),
    ]))
    .unwrap_err();
    assert!(err.message.contains("does not return output arguments"));
}

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    haystack
        .windows(needle.len())
        .any(|window| window == needle)
}

#[test]
fn dlfeval_dlgradient_differentiates_dlarray_elementwise_loss() {
    let x = block_on(dlarray_builtin(
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap()),
        vec![Value::String("CB".into())],
    ))
    .expect("dlarray");
    let _guard = crate::output_count::push_output_count(Some(2));
    let out = block_on(dlfeval_builtin(vec![
        Value::FunctionHandle("__deep_learning_square_grad".into()),
        x,
    ]))
    .expect("dlfeval square grad");
    let Value::OutputList(values) = out else {
        panic!("expected output list");
    };
    let Value::Object(loss) = &values[0] else {
        panic!("expected traced loss dlarray");
    };
    let Value::Tensor(loss_data) = loss.properties.get("Data").unwrap() else {
        panic!("expected loss tensor");
    };
    assert_eq!(loss_data.data, vec![14.0]);
    let Value::Object(grad) = &values[1] else {
        panic!("expected gradient dlarray");
    };
    let Value::Tensor(grad_data) = grad.properties.get("Data").unwrap() else {
        panic!("expected gradient tensor");
    };
    assert_eq!(grad_data.data, vec![2.0, 4.0, 6.0]);
    assert_eq!(
        grad.properties.get("Format"),
        Some(&Value::String("CB".into()))
    );
}

#[test]
fn dlfeval_dlgradient_preserves_native_single_storage() {
    let x = block_on(dlarray_builtin(
        Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap()),
        vec![Value::String("CB".into())],
    ))
    .expect("dlarray");
    let _guard = crate::output_count::push_output_count(Some(2));
    let out = block_on(dlfeval_builtin(vec![
        Value::FunctionHandle("__deep_learning_square_grad".into()),
        x,
    ]))
    .expect("dlfeval square grad");
    let Value::OutputList(values) = out else {
        panic!("expected output list");
    };
    let Value::Object(loss) = &values[0] else {
        panic!("expected traced loss dlarray");
    };
    let Value::Tensor(loss) = loss.properties.get("Data").unwrap() else {
        panic!("expected loss tensor");
    };
    assert_eq!(
        loss.clone().into_numeric_storage().expect("loss storage"),
        NumericStorage::F32(vec![14.0])
    );
    let Value::Object(gradient) = &values[1] else {
        panic!("expected gradient dlarray");
    };
    let Value::Tensor(gradient) = gradient.properties.get("Data").unwrap() else {
        panic!("expected gradient tensor");
    };
    assert_eq!(
        gradient
            .clone()
            .into_numeric_storage()
            .expect("gradient storage"),
        NumericStorage::F32(vec![2.0, 4.0, 6.0])
    );
}

#[test]
fn dlgradient_returns_dlnetwork_learnable_gradients() {
    let input = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        vec![Value::String("Name".into()), Value::String("in".into())],
    ))
    .unwrap();
    let fc = block_on(fully_connected_layer_builtin(
        Value::Num(2.0),
        vec![
            Value::String("Name".into()),
            Value::String("fc".into()),
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![1.0, -1.0, 0.5, 2.0], vec![2, 2]).unwrap()),
            Value::String("Bias".into()),
            Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![2, 1]).unwrap()),
        ],
    ))
    .unwrap();
    let net = block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input, fc], 2, 1).unwrap(),
    )]))
    .unwrap();
    let x = block_on(dlarray_builtin(
        Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap()),
        vec![Value::String("CB".into())],
    ))
    .unwrap();

    let gradients = block_on(dlfeval_builtin(vec![
        Value::FunctionHandle("__deep_learning_network_grad".into()),
        net,
        x,
    ]))
    .expect("network gradients");
    let Value::Struct(st) = gradients else {
        panic!("expected learnables gradient struct");
    };
    let Value::Cell(values) = st.fields.get("Value").unwrap() else {
        panic!("expected Value cell");
    };
    let Value::Object(weight_grad) = &values.data[0] else {
        panic!("expected dlarray weight gradient");
    };
    let Value::Tensor(weight_data) = weight_grad.properties.get("Data").unwrap() else {
        panic!("expected tensor weight gradient");
    };
    assert_eq!(weight_data.data, vec![4.0, 4.0, 6.0, 6.0]);
    let Value::Object(bias_grad) = &values.data[1] else {
        panic!("expected dlarray bias gradient");
    };
    let Value::Tensor(bias_data) = bias_grad.properties.get("Data").unwrap() else {
        panic!("expected tensor bias gradient");
    };
    assert_eq!(bias_data.data, vec![2.0, 2.0]);
}

#[test]
fn dlupdate_traverses_dlnetwork_learnables_and_rebuilds_layers() {
    let net = block_on(dlnetwork_builtin(vec![feature_network_layers()])).unwrap();
    let out = block_on(dlupdate_builtin(vec![
        Value::FunctionHandle("plus".into()),
        net.clone(),
        net,
    ]))
    .expect("network dlupdate");
    let Value::Object(network) = out else {
        panic!("expected network object");
    };
    let Value::Cell(layers) = network.properties.get("Layers").unwrap() else {
        panic!("expected layer cell");
    };
    let Value::Object(fc) = &layers.data[1] else {
        panic!("expected fc layer");
    };
    let Value::Tensor(weights) = fc.properties.get("Weights").unwrap() else {
        panic!("expected tensor weights");
    };
    assert_eq!(weights.data, vec![2.0, -2.0, 0.0, 2.0]);
}

#[test]
fn dlgradient_rejects_gpu_backed_dlarray_autodiff_with_provider_gap() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let shape = [1usize, 2usize];
        let handle = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &shape,
            })
            .expect("upload");
        let x = block_on(dlarray_builtin(
            Value::GpuTensor(handle),
            vec![Value::String("CB".into())],
        ))
        .expect("gpu dlarray");
        let err = block_on(dlfeval_builtin(vec![
            Value::FunctionHandle("__deep_learning_square_grad".into()),
            x,
        ]))
        .unwrap_err();
        assert!(err.to_string().contains("provider-resident tape kernels"));
    });
}

#[test]
fn dlnetwork_rejects_layers_without_forward_support() {
    let input = block_on(sequence_input_layer_builtin(Value::Num(2.0), vec![])).unwrap();
    let lstm = block_on(lstm_layer_builtin(Value::Num(4.0), vec![])).unwrap();
    let err = block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input, lstm], 2, 1).unwrap(),
    )]))
    .unwrap_err();
    assert!(err
        .message
        .contains("is not supported for RunMat forward execution"));
}

fn regression_training_layers() -> Value {
    let input = block_on(feature_input_layer_builtin(
        Value::Num(1.0),
        vec![Value::String("Name".into()), Value::String("x".into())],
    ))
    .unwrap();
    let fc = block_on(fully_connected_layer_builtin(
        Value::Num(1.0),
        vec![
            Value::String("Name".into()),
            Value::String("fc".into()),
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
            Value::String("Bias".into()),
            Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
        ],
    ))
    .unwrap();
    let output = block_on(regression_layer_builtin(vec![
        Value::String("Name".into()),
        Value::String("out".into()),
    ]))
    .unwrap();
    Value::Cell(CellArray::new(vec![input, fc, output], 3, 1).unwrap())
}

fn classification_training_layers() -> Value {
    let input = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        vec![
            Value::String("Name".into()),
            Value::String("features".into()),
        ],
    ))
    .unwrap();
    let fc = block_on(fully_connected_layer_builtin(
        Value::Num(2.0),
        vec![
            Value::String("Name".into()),
            Value::String("fc".into()),
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap()),
            Value::String("Bias".into()),
            Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![2, 1]).unwrap()),
        ],
    ))
    .unwrap();
    let softmax = block_on(softmax_layer_builtin(vec![
        Value::String("Name".into()),
        Value::String("prob".into()),
    ]))
    .unwrap();
    let output = block_on(classification_layer_builtin(vec![
        Value::String("Name".into()),
        Value::String("class".into()),
    ]))
    .unwrap();
    Value::Cell(CellArray::new(vec![input, fc, softmax, output], 4, 1).unwrap())
}

fn adam_training_options() -> Value {
    block_on(training_options_builtin(
        Value::String("adam".into()),
        vec![
            Value::String("MaxEpochs".into()),
            Value::Num(120.0),
            Value::String("MiniBatchSize".into()),
            Value::Num(4.0),
            Value::String("InitialLearnRate".into()),
            Value::Num(0.05),
            Value::String("Shuffle".into()),
            Value::String("never".into()),
            Value::String("Verbose".into()),
            Value::Bool(false),
        ],
    ))
    .unwrap()
}

#[test]
fn train_network_regression_updates_weights_and_predicts() {
    let x = Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap());
    let y = Value::Tensor(Tensor::new(vec![1.0, 3.0, 5.0, 7.0], vec![4, 1]).unwrap());
    let net = block_on(train_network_builtin(vec![
        x.clone(),
        y,
        regression_training_layers(),
        adam_training_options(),
    ]))
    .unwrap();
    let Value::Object(object) = &net else {
        panic!("expected trained network object");
    };
    assert_eq!(object.class_name, "SeriesNetwork");
    assert!(matches!(
        object.properties.get("TrainingInfo"),
        Some(Value::Struct(_))
    ));
    let predicted = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
        net,
        x,
        vec![],
    ))
    .unwrap();
    let Value::Tensor(predicted) = predicted else {
        panic!("expected prediction tensor");
    };
    assert_eq!(predicted.shape, vec![4, 1]);
    assert!((predicted.data[0] - 1.0).abs() < 0.35);
    assert!((predicted.data[3] - 7.0).abs() < 0.35);
}

#[test]
fn train_network_regression_reads_typed_integer_storage_exactly() {
    let x = poisoned_int_tensor(IntegerStorage::I32(vec![0, 1, 2, 3]), vec![4, 1], 99.0);
    let y = poisoned_int_tensor(IntegerStorage::I32(vec![1, 3, 5, 7]), vec![4, 1], 99.0);
    let net = block_on(train_network_builtin(vec![
        x.clone(),
        y,
        regression_training_layers(),
        adam_training_options(),
    ]))
    .unwrap();

    let predicted = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
        net,
        x,
        vec![],
    ))
    .unwrap();
    let Value::Tensor(predicted) = predicted else {
        panic!("expected prediction tensor");
    };
    assert_eq!(predicted.shape, vec![4, 1]);
    assert!((predicted.data[0] - 1.0).abs() < 0.35);
    assert!((predicted.data[3] - 7.0).abs() < 0.35);
}

#[test]
fn train_network_classification_supports_string_labels() {
    let x = Value::Tensor(
        Tensor::new(vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0], vec![4, 2]).unwrap(),
    );
    let y = Value::StringArray(
        StringArray::new(
            vec!["left".into(), "right".into(), "left".into(), "right".into()],
            vec![4, 1],
        )
        .unwrap(),
    );
    let net = block_on(train_network_builtin(vec![
        x.clone(),
        y,
        classification_training_layers(),
        adam_training_options(),
    ]))
    .unwrap();
    let Value::Object(object) = &net else {
        panic!("expected trained network object");
    };
    let Value::StringArray(classes) = object.properties.get("Classes").unwrap() else {
        panic!("expected classes");
    };
    assert_eq!(classes.data, vec!["left", "right"]);
    let predicted = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
        net,
        x,
        vec![],
    ))
    .unwrap();
    let Value::Tensor(scores) = predicted else {
        panic!("expected scores");
    };
    assert!(scores.data[0] > scores.data[4]);
    assert!(scores.data[5] > scores.data[1]);
}

#[test]
fn trainnet_crossentropy_supports_categorical_labels() {
    let x = Value::Tensor(
        Tensor::new(vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0], vec![4, 2]).unwrap(),
    );
    let mut categorical = ObjectInstance::new("categorical".into());
    categorical.properties.insert(
        "Codes".into(),
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 1.0, 2.0], vec![4, 1]).unwrap()),
    );
    categorical.properties.insert(
        "Categories".into(),
        Value::StringArray(
            StringArray::new(vec!["left".into(), "right".into()], vec![1, 2]).unwrap(),
        ),
    );
    let net = block_on(dlnetwork_builtin(vec![classification_training_layers()])).unwrap();
    let trained = block_on(trainnet_builtin(vec![
        x.clone(),
        Value::Object(categorical),
        net,
        Value::String("crossentropy".into()),
        adam_training_options(),
    ]))
    .unwrap();
    let Value::Object(object) = &trained else {
        panic!("expected dlnetwork");
    };
    assert_eq!(object.class_name, "dlnetwork");
    assert!(matches!(
        object.properties.get("TrainingInfo"),
        Some(Value::Struct(_))
    ));
}

#[test]
fn trainnet_rejects_custom_loss_until_autodiff_exists() {
    let err = block_on(trainnet_builtin(vec![
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
        block_on(dlnetwork_builtin(vec![regression_training_layers()])).unwrap(),
        Value::FunctionHandle("loss".into()),
        adam_training_options(),
    ]))
    .unwrap_err();
    assert!(err
        .message
        .contains("custom loss function handles require autodiff"));
}

#[test]
fn train_network_rejects_gpu_execution_environment() {
    let opts = block_on(training_options_builtin(
        Value::String("adam".into()),
        vec![
            Value::String("ExecutionEnvironment".into()),
            Value::String("gpu".into()),
        ],
    ))
    .unwrap();
    let err = block_on(train_network_builtin(vec![
        Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap()),
        regression_training_layers(),
        opts,
    ]))
    .unwrap_err();
    assert!(err
        .message
        .contains("ExecutionEnvironment 'gpu' requires native GPU"));
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
fn combvec_reads_typed_integer_storage_exactly() {
    let out = block_on(combvec_builtin(vec![
        poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 3, 4]), vec![2, 2], 99.0),
        poisoned_int_tensor(IntegerStorage::U16(vec![10, 20]), vec![1, 2], 99.0),
    ]))
    .unwrap();
    let Value::Tensor(t) = out else {
        panic!("expected tensor");
    };
    assert_eq!(t.shape, vec![3, 4]);
    assert_eq!(
        t.data,
        vec![1.0, 2.0, 10.0, 3.0, 4.0, 10.0, 1.0, 2.0, 20.0, 3.0, 4.0, 20.0]
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
fn padsequences_reads_typed_integer_storage_exactly() {
    let seqs = Value::Cell(
        CellArray::new(
            vec![
                poisoned_int_tensor(IntegerStorage::I16(vec![1, 2]), vec![1, 2], 99.0),
                poisoned_int_tensor(IntegerStorage::U8(vec![3]), vec![1, 1], 99.0),
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
fn dlarray_rejects_every_exact_integer_storage_class() {
    let storages = [
        runmat_builtins::IntegerStorage::I8(vec![1]),
        runmat_builtins::IntegerStorage::I16(vec![1]),
        runmat_builtins::IntegerStorage::I32(vec![1]),
        runmat_builtins::IntegerStorage::I64(vec![1]),
        runmat_builtins::IntegerStorage::U8(vec![1]),
        runmat_builtins::IntegerStorage::U16(vec![1]),
        runmat_builtins::IntegerStorage::U32(vec![1]),
        runmat_builtins::IntegerStorage::U64(vec![1]),
    ];

    for storage in storages {
        let data = Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).unwrap());
        let error = block_on(dlarray_builtin(data, vec![])).expect_err("integer dlarray error");
        assert!(error.message().contains("integer data is not supported"));
    }
}

#[test]
fn dlarray_preserves_gpu_array_residency() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let out = block_on(dlarray_builtin(
            Value::GpuTensor(handle.clone()),
            vec![Value::String("CB".into())],
        ))
        .expect("dlarray gpu");
        let Value::Object(object) = out else {
            panic!("expected object");
        };
        assert_eq!(object.class_name, "dlarray");
        let Value::GpuTensor(wrapped) = object.properties.get("Data").unwrap() else {
            panic!("expected gpu tensor data");
        };
        assert_eq!(wrapped.buffer_id, handle.buffer_id);
        assert_eq!(wrapped.device_id, handle.device_id);
        assert_eq!(wrapped.shape, handle.shape);
        assert_eq!(handle_precision(wrapped), handle_precision(&handle));
        assert_eq!(handle_storage(wrapped), handle_storage(&handle));
        assert_eq!(
            object.properties.get("Format"),
            Some(&Value::String("CB".into()))
        );
    });
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
fn crossentropy_gpu_inputs_use_provider_for_scalar_reduction() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let shape = [1usize, 2usize];
        let predictions = provider
            .upload(&HostTensorView {
                data: &[0.8, 0.2],
                shape: &shape,
            })
            .expect("upload predictions");
        let targets = provider
            .upload(&HostTensorView {
                data: &[1.0, 0.0],
                shape: &shape,
            })
            .expect("upload targets");

        provider.reset_telemetry();
        let out = block_on(crossentropy_builtin(
            Value::GpuTensor(predictions),
            Value::GpuTensor(targets),
            vec![],
        ))
        .expect("crossentropy gpu");
        let Value::Num(loss) = out else {
            panic!("expected scalar");
        };
        assert!((loss - 0.11157177565710488).abs() < 1.0e-12);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn crossentropy_gpu_reduction_none_returns_resident_losses() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let shape = [1usize, 2usize];
        let predictions = provider
            .upload(&HostTensorView {
                data: &[0.8, 0.2],
                shape: &shape,
            })
            .expect("upload predictions");
        let targets = provider
            .upload(&HostTensorView {
                data: &[1.0, 0.0],
                shape: &shape,
            })
            .expect("upload targets");

        provider.reset_telemetry();
        let out = block_on(crossentropy_builtin(
            Value::GpuTensor(predictions),
            Value::GpuTensor(targets),
            vec![
                Value::String("ClassificationMode".into()),
                Value::String("multi-label".into()),
                Value::String("Reduction".into()),
                Value::String("none".into()),
            ],
        ))
        .expect("crossentropy gpu unreduced");
        assert!(matches!(out, Value::GpuTensor(_)));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let gathered = crate::builtins::common::test_support::gather(out).expect("gather losses");
        assert_eq!(gathered.shape, shape);
        assert!((gathered.data[0] + 0.8_f64.ln()).abs() < 1.0e-12);
        assert!((gathered.data[1] + 0.8_f64.ln()).abs() < 1.0e-12);
    });
}

#[test]
fn crossentropy_gpu_supports_host_weights_without_gathering_predictions() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let shape = [2usize, 2usize];
        let predictions = provider
            .upload(&HostTensorView {
                data: &[0.8, 0.2, 0.25, 0.75],
                shape: &shape,
            })
            .expect("upload predictions");
        let targets = provider
            .upload(&HostTensorView {
                data: &[1.0, 0.0, 0.0, 1.0],
                shape: &shape,
            })
            .expect("upload targets");

        provider.reset_telemetry();
        let out = block_on(crossentropy_builtin(
            Value::GpuTensor(predictions),
            Value::GpuTensor(targets),
            vec![
                Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![2, 1]).unwrap()),
                Value::String("DataFormat".into()),
                Value::String("CB".into()),
            ],
        ))
        .expect("weighted gpu crossentropy");
        let Value::Num(loss) = out else {
            panic!("expected scalar");
        };
        let expected = (-2.0_f64 * 0.8_f64.ln() - 0.75_f64.ln()) / 2.0;
        assert!((loss - expected).abs() < 1.0e-12);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn crossentropy_gpu_supports_host_mask_for_unreduced_losses() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let shape = [1usize, 2usize];
        let predictions = provider
            .upload(&HostTensorView {
                data: &[0.8, 0.2],
                shape: &shape,
            })
            .expect("upload predictions");
        let targets = provider
            .upload(&HostTensorView {
                data: &[1.0, 0.0],
                shape: &shape,
            })
            .expect("upload targets");

        provider.reset_telemetry();
        let out = block_on(crossentropy_builtin(
            Value::GpuTensor(predictions),
            Value::GpuTensor(targets),
            vec![
                Value::String("Mask".into()),
                Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap()),
                Value::String("ClassificationMode".into()),
                Value::String("multi-label".into()),
                Value::String("Reduction".into()),
                Value::String("none".into()),
            ],
        ))
        .expect("masked gpu crossentropy");
        assert!(matches!(out, Value::GpuTensor(_)));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let gathered = crate::builtins::common::test_support::gather(out).expect("gather losses");
        assert_eq!(gathered.shape, shape);
        assert!((gathered.data[0] + 0.8_f64.ln()).abs() < 1.0e-12);
        assert_eq!(gathered.data[1], 0.0);
    });
}

#[test]
fn crossentropy_gpu_supports_resident_weights_and_mask_without_downloads() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let shape = [1usize, 2usize];
        let predictions = provider
            .upload(&HostTensorView {
                data: &[0.8, 0.2],
                shape: &shape,
            })
            .expect("upload predictions");
        let targets = provider
            .upload(&HostTensorView {
                data: &[1.0, 0.0],
                shape: &shape,
            })
            .expect("upload targets");
        let weights = provider
            .upload(&HostTensorView {
                data: &[2.0, 3.0],
                shape: &shape,
            })
            .expect("upload weights");
        let mask = provider
            .upload(&HostTensorView {
                data: &[1.0, 0.0],
                shape: &shape,
            })
            .expect("upload mask");

        provider.reset_telemetry();
        let out = block_on(crossentropy_builtin(
            Value::GpuTensor(predictions),
            Value::GpuTensor(targets),
            vec![
                Value::GpuTensor(weights),
                Value::String("Mask".into()),
                Value::GpuTensor(mask),
                Value::String("ClassificationMode".into()),
                Value::String("multi-label".into()),
            ],
        ))
        .expect("resident weighted masked gpu crossentropy");
        let Value::Num(loss) = out else {
            panic!("expected scalar");
        };
        assert!((loss + 0.8_f64.ln()).abs() < 1.0e-12);
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn crossentropy_supports_dataformat_class_weights_and_batch_normalization() {
    let out = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2, 0.25, 0.75], vec![2, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap()),
        vec![
            Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![2, 1]).unwrap()),
            Value::String("DataFormat".into()),
            Value::String("CB".into()),
        ],
    ))
    .expect("weighted crossentropy");
    let Value::Num(loss) = out else {
        panic!("expected scalar");
    };
    let expected = (-2.0_f64 * 0.8_f64.ln() - 0.75_f64.ln()) / 2.0;
    assert!(
        (loss - expected).abs() < 1.0e-12,
        "loss {loss} expected {expected}"
    );
}

#[test]
fn crossentropy_supports_weights_name_value_and_weights_format_mapping() {
    let out = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2, 0.25, 0.75], vec![2, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap()),
        vec![
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap()),
            Value::String("DataFormat".into()),
            Value::String("CB".into()),
            Value::String("WeightsFormat".into()),
            Value::String("UC".into()),
        ],
    ))
    .expect("formatted class weights");
    let Value::Num(loss) = out else {
        panic!("expected scalar");
    };
    let expected = (-2.0_f64 * 0.8_f64.ln() - 0.75_f64.ln()) / 2.0;
    assert!((loss - expected).abs() < 1.0e-12);

    let out = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2, 0.25, 0.75], vec![2, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap()),
        vec![
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![0.5, 2.0], vec![1, 2]).unwrap()),
            Value::String("DataFormat".into()),
            Value::String("CB".into()),
            Value::String("WeightsFormat".into()),
            Value::String("UB".into()),
        ],
    ))
    .expect("formatted observation weights");
    let Value::Num(loss) = out else {
        panic!("expected scalar");
    };
    let expected = (-0.5_f64 * 0.8_f64.ln() - 2.0_f64 * 0.75_f64.ln()) / 2.0;
    assert!((loss - expected).abs() < 1.0e-12);
}

#[test]
fn crossentropy_remaps_full_size_formatted_weights() {
    let out = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2, 0.25, 0.75], vec![2, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap()),
        vec![
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![2.0, 3.0, 5.0, 7.0], vec![2, 2]).unwrap()),
            Value::String("DataFormat".into()),
            Value::String("CB".into()),
            Value::String("WeightsFormat".into()),
            Value::String("BC".into()),
        ],
    ))
    .expect("formatted full-size weights");
    let Value::Num(loss) = out else {
        panic!("expected scalar");
    };
    let expected = (-2.0_f64 * 0.8_f64.ln() - 7.0_f64 * 0.75_f64.ln()) / 2.0;
    assert!((loss - expected).abs() < 1.0e-12);

    let err = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2, 0.25, 0.75], vec![2, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap()),
        vec![
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]).unwrap()),
            Value::String("DataFormat".into()),
            Value::String("CB".into()),
            Value::String("WeightsFormat".into()),
            Value::String("TC".into()),
        ],
    ))
    .unwrap_err();
    assert!(err.to_string().contains("not present in DataFormat"));
}

#[test]
fn crossentropy_supports_multilabel_reduction_none_and_mask_normalization() {
    let unreduced = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        vec![
            Value::String("ClassificationMode".into()),
            Value::String("multi-label".into()),
            Value::String("Reduction".into()),
            Value::String("none".into()),
        ],
    ))
    .expect("multilabel unreduced crossentropy");
    let Value::Tensor(tensor) = unreduced else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.shape, vec![1, 2]);
    assert!((tensor.data[0] + 0.8_f64.ln()).abs() < 1.0e-12);
    assert!((tensor.data[1] + 0.8_f64.ln()).abs() < 1.0e-12);

    let masked = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        vec![
            Value::String("Mask".into()),
            Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap()),
            Value::String("NormalizationFactor".into()),
            Value::String("mask-included".into()),
        ],
    ))
    .expect("masked crossentropy");
    let Value::Num(loss) = masked else {
        panic!("expected scalar");
    };
    assert!((loss + 0.8_f64.ln()).abs() < 1.0e-12);
}

#[test]
fn crossentropy_unreduced_output_preserves_native_single_storage() {
    let out = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::from_f32(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::from_f32(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        vec![
            Value::String("ClassificationMode".into()),
            Value::String("multi-label".into()),
            Value::String("Reduction".into()),
            Value::String("none".into()),
        ],
    ))
    .expect("single unreduced crossentropy");
    let Value::Tensor(losses) = out else {
        panic!("expected tensor");
    };
    assert_eq!(losses.numeric_dtype(), NumericDType::F32);
    assert!(matches!(
        losses.into_numeric_storage().unwrap(),
        NumericStorage::F32(_)
    ));
}

#[test]
fn crossentropy_mask_reads_typed_integer_storage_exactly() {
    let masked = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        vec![
            Value::String("Mask".into()),
            poisoned_int_tensor(IntegerStorage::U8(vec![1, 0]), vec![1, 2], 2.0),
            Value::String("NormalizationFactor".into()),
            Value::String("mask-included".into()),
        ],
    ))
    .expect("masked crossentropy");
    let Value::Num(loss) = masked else {
        panic!("expected scalar");
    };
    assert!((loss + 0.8_f64.ln()).abs() < 1.0e-12);
}

#[test]
fn crossentropy_payloads_read_typed_integer_storage_as_double_loss_inputs() {
    let out = block_on(crossentropy_builtin(
        poisoned_int_tensor(IntegerStorage::U8(vec![1, 0]), vec![1, 2], 0.5),
        poisoned_int_tensor(IntegerStorage::U8(vec![1, 0]), vec![1, 2], 0.5),
        vec![
            Value::String("Reduction".into()),
            Value::String("none".into()),
        ],
    ))
    .expect("integer payload crossentropy");
    let Value::Tensor(losses) = out else {
        panic!("expected tensor");
    };
    assert_eq!(losses.dtype, NumericDType::F64);
    assert!(losses.integer_storage().is_none());
    assert!(losses.data[0].abs() < 1.0e-9);
    assert_eq!(losses.data[1], 0.0);
}

#[test]
fn crossentropy_rejects_incompatible_shapes_masks_and_formats() {
    let err = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap()),
        vec![],
    ))
    .unwrap_err();
    assert!(err.to_string().contains("prediction shape"));

    let err = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        vec![Value::String("Mask".into()), Value::Bool(true)],
    ))
    .unwrap_err();
    assert!(err.to_string().contains("same size"));

    let err = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        vec![
            Value::String("Mask".into()),
            Value::Tensor(Tensor::new(vec![0.5, 1.0], vec![1, 2]).unwrap()),
        ],
    ))
    .unwrap_err();
    assert!(err.to_string().contains("binary"));

    let err = block_on(crossentropy_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        vec![
            Value::String("DataFormat".into()),
            Value::String("CC".into()),
        ],
    ))
    .unwrap_err();
    assert!(err.to_string().contains("cannot repeat"));
}

#[test]
fn crossentropy_preserves_dlarray_output_for_dlarray_inputs() {
    let predictions = block_on(dlarray_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![2, 1]).unwrap()),
        vec![Value::String("CB".into())],
    ))
    .expect("prediction dlarray");
    let targets = block_on(dlarray_builtin(
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap()),
        vec![Value::String("CB".into())],
    ))
    .expect("target dlarray");
    let out =
        block_on(crossentropy_builtin(predictions, targets, vec![])).expect("dlarray crossentropy");
    let Value::Object(object) = out else {
        panic!("expected dlarray object");
    };
    assert_eq!(object.class_name, "dlarray");
    let Value::Num(loss) = object.properties.get("Data").unwrap() else {
        panic!("expected scalar data");
    };
    assert!((*loss + 0.8_f64.ln()).abs() < 1.0e-12);

    let predictions = block_on(dlarray_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![2, 1]).unwrap()),
        vec![Value::String("CB".into())],
    ))
    .expect("prediction dlarray");
    let targets = block_on(dlarray_builtin(
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap()),
        vec![Value::String("CB".into())],
    ))
    .expect("target dlarray");
    let out = block_on(crossentropy_builtin(
        predictions,
        targets,
        vec![
            Value::String("Reduction".into()),
            Value::String("none".into()),
        ],
    ))
    .expect("unreduced dlarray crossentropy");
    let Value::Object(object) = out else {
        panic!("expected dlarray object");
    };
    assert_eq!(
        object.properties.get("Format"),
        Some(&Value::String(String::new()))
    );
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
fn adamupdate_hyperparameters_read_typed_integer_storage_exactly() {
    let out = block_on(adamupdate_builtin(vec![
        Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.1], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
        cleared_int_tensor(IntegerStorage::U16(vec![1]), vec![1, 1]),
        cleared_int_tensor(IntegerStorage::U16(vec![1]), vec![1, 1]),
        cleared_int_tensor(IntegerStorage::U8(vec![0]), vec![1, 1]),
        cleared_int_tensor(IntegerStorage::U8(vec![0]), vec![1, 1]),
    ]))
    .expect("adamupdate");
    let Value::Tensor(parameters) = out else {
        panic!("expected tensor parameters");
    };
    assert!((parameters.data[0] - 0.0000001).abs() < 1.0e-9);
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
fn dlupdate_maps_matching_struct_and_cell_trees() {
    let mut left = StructValue::new();
    left.insert(
        "Weights",
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
    );
    left.insert(
        "Bias",
        Value::Cell(CellArray::new(vec![Value::Num(3.0)], 1, 1).unwrap()),
    );
    let mut right = StructValue::new();
    right.insert(
        "Weights",
        Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![1, 2]).unwrap()),
    );
    right.insert(
        "Bias",
        Value::Cell(CellArray::new(vec![Value::Num(4.0)], 1, 1).unwrap()),
    );

    let out = block_on(dlupdate_builtin(vec![
        Value::FunctionHandle("plus".into()),
        Value::Struct(left),
        Value::Struct(right),
    ]))
    .expect("dlupdate plus");
    let Value::Struct(out) = out else {
        panic!("expected struct output");
    };
    let Value::Tensor(weights) = out.fields.get("Weights").unwrap() else {
        panic!("expected tensor weights");
    };
    assert_eq!(weights.data, vec![11.0, 22.0]);
    let Value::Cell(bias) = out.fields.get("Bias").unwrap() else {
        panic!("expected cell bias");
    };
    assert_eq!(bias.data, vec![Value::Num(7.0)]);
}

#[test]
fn dlupdate_reconstructs_multiple_output_trees() {
    let _guard = crate::output_count::push_output_count(Some(2));
    let first = Value::Cell(CellArray::new(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap());
    let second =
        Value::Cell(CellArray::new(vec![Value::Num(10.0), Value::Num(20.0)], 1, 2).unwrap());

    let out = block_on(dlupdate_builtin(vec![
        Value::FunctionHandle("deal".into()),
        first.clone(),
        second.clone(),
    ]))
    .expect("dlupdate deal");
    let Value::OutputList(outputs) = out else {
        panic!("expected output list");
    };
    assert_eq!(outputs, vec![first, second]);
}

#[test]
fn dlupdate_maps_learnables_table_value_variable_only() {
    let left = crate::builtins::table::table_from_columns(
        vec!["Layer".into(), "Parameter".into(), "Value".into()],
        vec![
            Value::StringArray(
                StringArray::new(vec!["fc".into(), "fc".into()], vec![2, 1]).unwrap(),
            ),
            Value::StringArray(
                StringArray::new(vec!["Weights".into(), "Bias".into()], vec![2, 1]).unwrap(),
            ),
            Value::Cell(
                CellArray::new(
                    vec![
                        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
                        Value::Num(3.0),
                    ],
                    2,
                    1,
                )
                .unwrap(),
            ),
        ],
    )
    .expect("left table");
    let right = crate::builtins::table::table_from_columns(
        vec!["Layer".into(), "Parameter".into(), "Value".into()],
        vec![
            Value::StringArray(
                StringArray::new(vec!["fc".into(), "fc".into()], vec![2, 1]).unwrap(),
            ),
            Value::StringArray(
                StringArray::new(vec!["Weights".into(), "Bias".into()], vec![2, 1]).unwrap(),
            ),
            Value::Cell(
                CellArray::new(
                    vec![
                        Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![1, 2]).unwrap()),
                        Value::Num(4.0),
                    ],
                    2,
                    1,
                )
                .unwrap(),
            ),
        ],
    )
    .expect("right table");

    let out = block_on(dlupdate_builtin(vec![
        Value::FunctionHandle("plus".into()),
        left,
        right,
    ]))
    .expect("dlupdate table");
    let Value::Object(object) = out else {
        panic!("expected table object");
    };
    let vars = crate::builtins::table::table_variables(&object).expect("table variables");
    assert_eq!(
        vars.fields.get("Layer"),
        Some(&Value::StringArray(
            StringArray::new(vec!["fc".into(), "fc".into()], vec![2, 1]).unwrap()
        ))
    );
    let Value::Cell(values) = vars.fields.get("Value").unwrap() else {
        panic!("expected Value cell");
    };
    let Value::Tensor(weights) = &values.data[0] else {
        panic!("expected weights tensor");
    };
    assert_eq!(weights.data, vec![11.0, 22.0]);
    assert_eq!(values.data[1], Value::Num(7.0));
}

#[test]
fn dlupdate_treats_dlarray_objects_as_leaves() {
    let data = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
    let dlarray = block_on(dlarray_builtin(
        data.clone(),
        vec![Value::String("CB".into())],
    ))
    .expect("dlarray");

    let out = block_on(dlupdate_builtin(vec![
        Value::FunctionHandle("class".into()),
        dlarray.clone(),
    ]))
    .expect("dlupdate dlarray");
    assert_eq!(out, Value::String("dlarray".into()));
}

#[test]
fn dlupdate_rejects_non_callable_and_mismatched_trees() {
    let err = block_on(dlupdate_builtin(vec![
        Value::String("@plus".into()),
        Value::Num(1.0),
    ]))
    .unwrap_err();
    assert!(err.to_string().contains("expected a function handle"));

    let err = block_on(dlupdate_builtin(vec![
        Value::FunctionHandle("plus".into()),
        Value::Cell(CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap()),
        Value::Cell(CellArray::new(vec![Value::Num(2.0), Value::Num(3.0)], 1, 2).unwrap()),
    ]))
    .unwrap_err();
    assert!(err.to_string().contains("same shape"));

    let generic_table = crate::builtins::table::table_from_columns(
        vec!["A".into(), "B".into()],
        vec![Value::Num(1.0), Value::Num(2.0)],
    )
    .expect("generic table");
    let err = block_on(dlupdate_builtin(vec![
        Value::FunctionHandle("plus".into()),
        generic_table,
    ]))
    .unwrap_err();
    assert!(err.to_string().contains("learnables tables"));

    let malformed_learnables = crate::builtins::table::table_from_columns(
        vec!["Layer".into(), "Parameter".into(), "Value".into()],
        vec![
            Value::StringArray(StringArray::new(vec!["fc".into()], vec![1, 1]).unwrap()),
            Value::StringArray(StringArray::new(vec!["Weights".into()], vec![1, 1]).unwrap()),
            Value::Num(1.0),
        ],
    )
    .expect("malformed learnables table");
    let err = block_on(dlupdate_builtin(vec![
        Value::FunctionHandle("plus".into()),
        malformed_learnables,
    ]))
    .unwrap_err();
    assert!(err.to_string().contains("Value variable must be a cell"));
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
    assert!(err.to_string().contains("expected X, Y, layers"));
    let err = block_on(export_onnx_network_builtin(vec![])).unwrap_err();
    assert!(err.to_string().contains("ONNX"));
}
