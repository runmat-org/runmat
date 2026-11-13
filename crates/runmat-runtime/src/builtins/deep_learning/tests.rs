use super::graph::*;
use super::layers::*;
use super::losses::*;
use super::model::*;
use super::object;
use super::onnx::*;
use super::sequences::*;
use super::supervised::*;
use super::training::*;
use crate::builtins::deep_learning::arguments::DeepLearningObjectEdit as _;
use futures::executor::block_on;
use runmat_accelerate_api::{handle_precision, handle_storage, GpuTensorStorage, HostTensorView};
use runmat_value::{
    CellArray, IntValue, IntegerStorage, LogicalArray, NumericDType, NumericStorage,
    ObjectInstance, StringArray, StructArray, Tensor, Value,
};

pub(crate) trait FixtureBuiltinArguments {
    fn into_fixture_arguments(self) -> crate::call::BuiltinArguments;
}

impl FixtureBuiltinArguments for Vec<Value> {
    fn into_fixture_arguments(self) -> crate::call::BuiltinArguments {
        crate::builtins::common::test_support::builtin_arguments(self)
    }
}

impl FixtureBuiltinArguments for crate::call::BuiltinArguments {
    fn into_fixture_arguments(self) -> crate::call::BuiltinArguments {
        self
    }
}

pub(crate) async fn fully_connected_layer_builtin<A: FixtureBuiltinArguments>(
    runtime: Option<crate::context::RuntimeContext>,
    output_size: Value,
    options: A,
) -> crate::BuiltinResult<Value> {
    super::layers::fully_connected_layer_builtin(
        runtime,
        output_size,
        options.into_fixture_arguments(),
    )
    .await
}

pub(crate) async fn fully_connected_layer_fixture<A: FixtureBuiltinArguments>(
    output_size: Value,
    options: A,
) -> crate::BuiltinResult<Value> {
    fully_connected_layer_builtin(None, output_size, options).await
}

pub(crate) trait FixtureArrayMut {
    type Item;

    fn make_fixture_data_unique(&mut self);
    fn fixture_data_mut_unique(&mut self) -> Option<&mut [Self::Item]>;
}

impl FixtureArrayMut for CellArray {
    type Item = Value;

    fn make_fixture_data_unique(&mut self) {
        self.make_data_unique_unbounded()
            .expect("ordinary Cell fixture backing");
    }

    fn fixture_data_mut_unique(&mut self) -> Option<&mut [Self::Item]> {
        self.try_data_mut_unique()
    }
}

impl FixtureArrayMut for StringArray {
    type Item = runmat_value::StringElement;

    fn make_fixture_data_unique(&mut self) {
        self.make_data_unique_unbounded()
            .expect("ordinary String fixture backing");
    }

    fn fixture_data_mut_unique(&mut self) -> Option<&mut [Self::Item]> {
        self.try_data_mut_unique()
    }
}

impl<T: FixtureArrayMut + ?Sized> FixtureArrayMut for &mut T {
    type Item = T::Item;

    fn make_fixture_data_unique(&mut self) {
        (**self).make_fixture_data_unique();
    }

    fn fixture_data_mut_unique(&mut self) -> Option<&mut [Self::Item]> {
        (**self).fixture_data_mut_unique()
    }
}

pub(crate) fn fixture_array_mut<T: FixtureArrayMut>(array: &mut T) -> &mut [T::Item] {
    array.make_fixture_data_unique();
    array
        .fixture_data_mut_unique()
        .expect("detached fixture array is unique")
}

// Test-only structural edits explicitly construct an ordinary destination;
// production mutation must admit its own map/key destination instead.
pub(crate) fn fixture_field_mut<'a>(
    structure: &'a mut runmat_value::StructValue,
    name: &str,
) -> Option<&'a mut Value> {
    if !structure.fields.is_unique() {
        let mut builder =
            runmat_value::UnboundedStructValueBuilder::with_capacity(structure.fields.len());
        for entry in structure.fields.entries() {
            builder.insert_text(entry.key().clone(), entry.value().clone());
        }
        *structure = builder.finish();
    }
    structure.fields.try_get_mut_unique(name)
}

fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>, _poison: f64) -> Value {
    let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
    Value::Tensor(tensor)
}

fn cleared_int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
    let tensor = Tensor::new_integer(storage, shape).expect("integer tensor");
    Value::Tensor(tensor)
}

fn layer_name(value: &Value) -> String {
    let Value::Object(object) = value else {
        panic!("expected object");
    };
    match object.properties().get("Name").unwrap() {
        Value::String(runmat_value::StringElement::Text(name)) => name.to_string(),
        other => panic!("expected string name, got {other:?}"),
    }
}

#[test]
fn layer_constructors_preserve_core_properties_and_name_values() {
    let fc = block_on(fully_connected_layer_builtin(
        None,
        Value::Num(10.0),
        vec![Value::String("Name".into()), Value::String("fc1".into())],
    ))
    .unwrap();
    let Value::Object(object) = &fc else {
        panic!("expected object");
    };
    assert_eq!(
        object.class_name().display_name(),
        "nnet.cnn.layer.FullyConnectedLayer"
    );
    assert_eq!(
        object.properties().get("OutputSize"),
        Some(&Value::Num(10.0))
    );
    assert_eq!(layer_name(&fc), "fc1");

    let conv = block_on(convolution_1d_layer_builtin(
        Value::Num(3.0),
        Value::Num(16.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Padding".into()),
            Value::String("causal".into()),
            Value::String("Name".into()),
            Value::String("conv".into()),
        ]),
    ))
    .unwrap();
    let Value::Object(object) = &conv else {
        panic!("expected object");
    };
    assert_eq!(
        object.properties().get("FilterSize"),
        Some(&Value::Num(3.0))
    );
    assert_eq!(
        object.properties().get("NumFilters"),
        Some(&Value::Num(16.0))
    );
    assert_eq!(
        object.properties().get("Padding"),
        Some(&Value::String("causal".into()))
    );
    assert_eq!(layer_name(&conv), "conv");
}

#[test]
fn layer_option_syntax_precedes_resident_parameter_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let values = [1.0, 2.0];
        let shape = [1usize, 2usize];
        let resident = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &shape,
            })
            .expect("resident option");
        provider.reset_telemetry();
        let error = block_on(fully_connected_layer_builtin(
            None,
            Value::Num(2.0),
            vec![
                Value::String("Weights".into()),
                Value::GpuTensor(resident.clone()),
                Value::String("Name".into()),
                Value::Num(1.0),
            ],
        ))
        .expect_err("invalid host Name");
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let alpha = [1.0];
        let scalar_shape = [1usize, 1usize];
        let resident_alpha = provider
            .upload(&HostTensorView {
                data: &alpha,
                shape: &scalar_shape,
            })
            .expect("resident alpha");
        provider.reset_telemetry();
        let error = block_on(elu_layer_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(resident_alpha),
                Value::String("Name".into()),
                Value::Num(2.0),
            ]),
        ))
        .expect_err("invalid host Name before resident alpha");
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn layer_constructors_read_typed_integer_size_storage_exactly() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let output_size = cleared_int_tensor(IntegerStorage::U16(vec![10]), vec![1, 1]);
    let fc = block_on(
        crate::builtins::deep_learning::tests::fully_connected_layer_fixture(output_size, vec![]),
    )
    .unwrap();
    let Value::Object(object) = &fc else {
        panic!("expected object");
    };
    assert_eq!(
        object.properties().get("OutputSize"),
        Some(&Value::Num(10.0))
    );

    let input_size = poisoned_int_tensor(IntegerStorage::U16(vec![2, 3]), vec![1, 2], f64::NAN);
    let input = block_on(feature_input_layer_builtin(
        input_size,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let Value::Object(object) = &input else {
        panic!("expected object");
    };
    assert_eq!(
        object.properties().get("InputSize"),
        Some(&Value::Tensor(
            Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap()
        ))
    );

    let exact = (1_u64 << 53) + 1;
    let large_input = poisoned_int_tensor(IntegerStorage::U64(vec![exact]), vec![1, 1], f64::NAN);
    let error = block_on(feature_input_layer_builtin(
        large_input,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect_err("structural metadata must not round a wide input dimension");
    assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
    let large_output = poisoned_int_tensor(IntegerStorage::U64(vec![exact]), vec![1, 1], f64::NAN);
    let error = block_on(fully_connected_layer_builtin(None, large_output, vec![]))
        .expect_err("structural metadata must not round a wide output dimension");
    assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));

    if usize::BITS == 64 {
        let aligned = 1_u64 << 54;
        let input_size =
            poisoned_int_tensor(IntegerStorage::U64(vec![aligned]), vec![1, 1], f64::NAN);
        let layer = block_on(feature_input_layer_builtin(
            input_size,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect("aligned wide dimension remains exact in double metadata");
        let Value::Object(layer) = layer else {
            panic!("expected feature layer")
        };
        assert_eq!(
            layer.properties().get("InputSize"),
            Some(&Value::Tensor(
                Tensor::new(vec![aligned as f64], vec![1, 1]).unwrap()
            ))
        );
    }
}

#[test]
fn feature_input_vector_size_is_a_runmat_metadata_extension() {
    let input_size = Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap());
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(feature_input_layer_builtin(
        input_size.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect_err("MATLAB mode accepts a scalar feature count only");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:FeatureInputVectorExtension")
    );
    drop(_strict);
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let layer = block_on(feature_input_layer_builtin(
        input_size,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let Value::Object(object) = layer else {
        panic!("expected layer")
    };
    assert_eq!(
        object.properties().get("InputSize"),
        Some(&Value::Tensor(
            Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap()
        ))
    );
    let error = feature_input_width(&object, "forward")
        .expect_err("current model path must not ignore trailing feature dimensions");
    assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
}

#[test]
fn feature_input_vector_gate_precedes_option_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let data = [1.0];
        let shape = [1usize, 1usize];
        let resident = provider
            .upload(&HostTensorView {
                data: &data,
                shape: &shape,
            })
            .unwrap();
        provider.reset_telemetry();
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let input_size = Value::Tensor(Tensor::new(vec![2.0, 3.0], vec![1, 2]).unwrap());
        let error = block_on(feature_input_layer_builtin(
            input_size,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("Weights".into()),
                Value::GpuTensor(resident),
            ]),
        ))
        .expect_err("vector extension gate must precede resident option handling");
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:FeatureInputVectorExtension")
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn deep_learning_integer_parsers_reject_unrepresentable_double_bounds() {
    let fc_boundary = block_on(fully_connected_layer_builtin(
        None,
        Value::Num(usize::MAX as f64),
        vec![],
    ));
    if usize::BITS == 64 {
        assert!(fc_boundary.is_err());
    } else {
        assert!(fc_boundary.is_ok());
    }
    assert!(block_on(fully_connected_layer_builtin(
        None,
        Value::Num((usize::MAX as f64) + 1.0),
        vec![],
    ))
    .is_err());
    assert!(block_on(fully_connected_layer_builtin(
        None,
        Value::Tensor(Tensor::new(vec![f64::NAN], vec![1, 1]).expect("nan scalar")),
        vec![],
    ))
    .is_err());

    let vector_boundary = Tensor::new(vec![usize::MAX as f64], vec![1, 1]).expect("input size");
    let feature_boundary = block_on(feature_input_layer_builtin(
        Value::Tensor(vector_boundary),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ));
    if usize::BITS == 64 {
        assert!(feature_boundary.is_err());
    } else {
        assert!(feature_boundary.is_ok());
    }

    let mut layer = ObjectInstance::new("nnet.cnn.layer.FullyConnectedLayer".to_string());
    let exact = (1_u64 << 53) + 1;
    layer
        .replace_dl_property("OutputSize", Value::Int(IntValue::U64(exact)))
        .expect("fixture property");
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
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Name".into()),
            Value::String("seq".into()),
        ]),
    ))
    .unwrap();
    assert_eq!(layer_name(&input), "seq");
    let lstm = block_on(lstm_layer_builtin(
        Value::Num(8.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("OutputMode".into()),
            Value::String("sequence".into()),
        ]),
    ))
    .unwrap();
    let Value::Object(object) = &lstm else {
        panic!("expected object");
    };
    assert_eq!(
        object.properties().get("NumHiddenUnits"),
        Some(&Value::Num(8.0))
    );
    assert_eq!(
        object.properties().get("OutputMode"),
        Some(&Value::String("sequence".into()))
    );
    let default_lstm = block_on(lstm_layer_builtin(
        Value::Num(4.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let Value::Object(default_lstm) = default_lstm else {
        panic!("expected LSTM object")
    };
    assert_eq!(
        default_lstm.properties().get("OutputMode"),
        Some(&Value::String("sequence".into()))
    );
    let last_bilstm = block_on(bilstm_layer_builtin(
        Value::Num(4.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("OutputMode".into()),
            Value::String("last".into()),
        ]),
    ))
    .unwrap();
    let Value::Object(last_bilstm) = last_bilstm else {
        panic!("expected BiLSTM object")
    };
    assert_eq!(
        last_bilstm.properties().get("OutputMode"),
        Some(&Value::String("last".into()))
    );
}

#[test]
fn training_options_and_layer_graph_materialize_metadata() {
    let opts = block_on(training_options_builtin(
        Value::String("adam".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("MaxEpochs".into()),
            Value::Num(5.0),
            Value::String("Verbose".into()),
            Value::Bool(false),
        ]),
    ))
    .unwrap();
    let Value::Object(options) = opts else {
        panic!("expected options object");
    };
    assert_eq!(
        options.properties().get("SolverName"),
        Some(&Value::String("adam".into()))
    );
    assert_eq!(
        options.properties().get("MaxEpochs"),
        Some(&Value::Num(5.0))
    );
    assert_eq!(
        options.properties().get("Verbose"),
        Some(&Value::Bool(false))
    );

    let relu = block_on(relu_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Name".into()),
            Value::String("relu".into()),
        ]),
    ))
    .unwrap();
    let graph = block_on(layer_graph_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Cell(
            CellArray::new(vec![relu], 1, 1).unwrap(),
        )]),
    ))
    .unwrap();
    let Value::Object(graph) = graph else {
        panic!("expected graph object");
    };
    assert_eq!(graph.class_name().display_name(), "nnet.cnn.LayerGraph");
    let Value::StringArray(names) = graph.properties().get("LayerNames").unwrap() else {
        panic!("expected names");
    };
    assert_eq!(names.data(), vec!["relu"]);
}

#[test]
fn dlnetwork_initialize_option_reads_typed_integer_storage_exactly() {
    let relu = block_on(relu_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Name".into()),
            Value::String("relu".into()),
        ]),
    ))
    .unwrap();
    let initialize = cleared_int_tensor(IntegerStorage::U8(vec![0]), vec![1, 1]);
    let net = block_on(dlnetwork_builtin(vec![
        Value::Cell(CellArray::new(vec![relu], 1, 1).unwrap()),
        Value::String("Initialize".into()),
        initialize,
    ]))
    .unwrap();
    let Value::Object(object) = net.value() else {
        panic!("expected network object");
    };
    assert_eq!(object.class_name().display_name(), "dlnetwork");
}

#[test]
fn dlnetwork_constructor_options_are_case_insensitive_and_last_value_wins() {
    let relu = block_on(relu_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Name".into()),
            Value::String("relu".into()),
        ]),
    ))
    .unwrap();
    let net = block_on(dlnetwork_builtin(vec![
        Value::Cell(CellArray::new(vec![relu], 1, 1).unwrap()),
        Value::String("Initialize".into()),
        Value::Bool(false),
        Value::String("INITIALIZE".into()),
        Value::Bool(true),
        Value::String("outputnames".into()),
        Value::String("relu".into()),
    ]))
    .unwrap();
    let Value::Object(object) = net.value() else {
        panic!("expected network object")
    };
    assert_eq!(
        object.properties().get("Initialize"),
        Some(&Value::Bool(true))
    );
    assert_eq!(
        object.properties().get("Initialized"),
        Some(&Value::Bool(true))
    );
}

#[test]
fn analyze_network_counts_layers() {
    let relu = block_on(relu_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Name".into()),
            Value::String("relu".into()),
        ]),
    ))
    .unwrap();
    let analysis = block_on(analyze_network_builtin(Value::Cell(
        CellArray::new(vec![relu], 1, 1).unwrap(),
    )))
    .unwrap();
    let Value::Object(object) = analysis else {
        panic!("expected analysis object");
    };
    assert_eq!(object.properties().get("NumLayers"), Some(&Value::Num(1.0)));
}

#[test]
fn graph_invalid_inputs_do_not_download_provider_values() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let values = [1.0, 2.0];
        let shape = [1usize, 2usize];
        let handle = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &shape,
            })
            .expect("upload fixture");
        provider.reset_telemetry();
        let error = block_on(layer_graph_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                handle.clone(),
            )]),
        ))
        .expect_err("numeric resident input is not a layer collection");
        assert!(error.to_string().contains("layers must be a layer object"));
        let error = block_on(analyze_network_builtin(Value::GpuTensor(handle.clone())))
            .expect_err("numeric resident input is not a layer collection");
        assert!(error.to_string().contains("expected a layer graph"));
        let error = block_on(layer_graph_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Object(ObjectInstance::new(
                    runmat_builtins::RELU_LAYER_CLASS.owned(),
                )),
                Value::GpuTensor(handle),
            ]),
        ))
        .expect_err("invalid arity precedes provider access");
        assert!(error.to_string().contains("expected zero inputs or one"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

fn feature_network_layers() -> Value {
    let input = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Name".into()),
            Value::String("features".into()),
        ]),
    ))
    .unwrap();
    let fc = block_on(fully_connected_layer_builtin(
        None,
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
    let softmax = block_on(softmax_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Name".into()),
            Value::String("prob".into()),
        ]),
    ))
    .unwrap();
    Value::Cell(CellArray::new(vec![input, fc, softmax], 3, 1).unwrap())
}

fn dense_feature_network_layers() -> Value {
    let input = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Name".into()),
            Value::String("features".into()),
        ]),
    ))
    .unwrap();
    let first = block_on(fully_connected_layer_builtin(
        None,
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
    let elu = block_on(elu_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1.25),
            Value::String("Name".into()),
            Value::String("elu".into()),
        ]),
    ))
    .unwrap();
    let relu = block_on(relu_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Name".into()),
            Value::String("relu".into()),
        ]),
    ))
    .unwrap();
    let second = block_on(fully_connected_layer_builtin(
        None,
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
    let input = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let elu = block_on(elu_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.25)]),
    ))
    .unwrap();
    Value::Cell(CellArray::new(vec![input, elu], 2, 1).unwrap())
}

#[test]
fn forward_rejects_malformed_elu_alpha_before_resident_preparation() {
    let invalid = [
        Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        Value::Num(f64::NAN),
        Value::Num(f64::INFINITY),
    ];
    for alpha in invalid {
        let net_owner = block_on(dlnetwork_builtin(vec![elu_feature_network_layers()])).unwrap();
        let mut net = net_owner.value().clone();
        let Value::Object(network) = &mut net else {
            panic!("network expected")
        };
        let Some(Value::Cell(layers)) = network.fixture_dl_property_mut("Layers") else {
            panic!("layers expected")
        };
        let Value::Object(elu) =
            &mut crate::builtins::deep_learning::tests::fixture_array_mut(layers)[1]
        else {
            panic!("ELU layer expected")
        };
        elu.replace_dl_property("Alpha", alpha)
            .expect("fixture property");
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 2],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        let resident = block_on(dlarray_builtin(
            Value::GpuTensor(allocation.handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        for input in [
            Value::Tensor(Tensor::new(vec![1.0, -1.0], vec![1, 2]).unwrap()),
            resident,
        ] {
            let error = block_on(forward_builtin(
                net.clone(),
                input,
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            ))
            .unwrap_err();
            assert!(error
                .to_string()
                .contains("Alpha must be a finite numeric scalar"));
        }
        assert_eq!(allocation.probe.retirements(), 0);
        drop(allocation.handle);
        assert_eq!(allocation.probe.retirements(), 1);
    }
}

#[test]
fn dlnetwork_materializes_metadata_and_learnables() {
    let net = block_on(dlnetwork_builtin(vec![feature_network_layers()])).unwrap();
    let Value::Object(object) = net.value() else {
        panic!("expected dlnetwork object");
    };
    assert_eq!(object.class_name().display_name(), "dlnetwork");
    assert_eq!(
        object.properties().get("Initialized"),
        Some(&Value::Bool(true))
    );
    let Value::StringArray(names) = object.properties().get("LayerNames").unwrap() else {
        panic!("expected layer names");
    };
    assert_eq!(names.data(), vec!["features", "fc", "prob"]);
    assert!(!object.properties().contains_key("Learnables"));
    let Value::Object(learnables) = super::model::project_learnables(object, "dlnetwork").unwrap()
    else {
        panic!("expected learnables table");
    };
    let variables = crate::builtins::table::table_variables(&learnables).unwrap();
    assert!(matches!(
        variables.fields.get("Layer"),
        Some(Value::StringArray(_))
    ));
    assert!(matches!(
        variables.fields.get("Parameter"),
        Some(Value::StringArray(_))
    ));
    let Value::Cell(values) = variables.fields.get("Value").unwrap() else {
        panic!("expected learnable values");
    };
    assert_eq!(values.data().len(), 2);
    assert!(values.data().iter().all(
        |value| matches!(value, Value::Object(object) if object.class_name().is(super::DLARRAY_CLASS))
    ));
    let property = crate::class_registry::lookup_property(
        object.class_name(),
        &runmat_types::MemberName::from("Learnables"),
    )
    .expect("network Learnables property is registered")
    .0;
    assert!(property.is_dependent);
    assert!(crate::class_registry::lookup_method(
        object.class_name(),
        &runmat_types::MethodName::from("get.Learnables"),
    )
    .is_some());
    assert!(crate::class_registry::lookup_method(
        object.class_name(),
        &runmat_types::MethodName::from("set.Learnables"),
    )
    .is_some());
}

#[test]
fn network_learnables_setter_updates_only_canonical_layers() {
    let net = block_on(dlnetwork_builtin(vec![feature_network_layers()])).unwrap();
    let Value::Object(network) = net.value() else {
        panic!("expected network")
    };
    let Value::Object(table) = super::model::project_learnables(network, "set.Learnables").unwrap()
    else {
        panic!("expected table")
    };
    let mut variables = crate::builtins::table::table_variables(&table).unwrap();
    let Value::Cell(values) =
        crate::builtins::deep_learning::tests::fixture_field_mut(&mut variables, "Value").unwrap()
    else {
        panic!("expected Value cell")
    };
    crate::builtins::deep_learning::tests::fixture_array_mut(values)[1] =
        Value::Tensor(Tensor::new(vec![5.0, 7.0], vec![2, 1]).unwrap());
    let replacement =
        crate::builtins::table::table_replace_variables_like(&table, variables).unwrap();
    let Value::Object(updated) =
        super::model::apply_learnables(network.clone(), replacement, "set.Learnables").unwrap()
    else {
        panic!("expected updated network")
    };
    assert!(!updated.properties().contains_key("Learnables"));
    let Value::Cell(layers) = updated.properties().get("Layers").unwrap() else {
        panic!("expected Layers cell")
    };
    let Value::Object(fc) = &layers.data()[1] else {
        panic!("expected FC layer")
    };
    let Value::Tensor(bias) = fc.properties().get("Bias").unwrap() else {
        panic!("expected bias")
    };
    assert_eq!(bias.materialize_f64(), vec![5.0, 7.0]);
    let Value::Object(projected) =
        super::model::project_learnables(&updated, "get.Learnables").unwrap()
    else {
        panic!("expected projection")
    };
    let variables = crate::builtins::table::table_variables(&projected).unwrap();
    let Value::Cell(values) = variables.fields.get("Value").unwrap() else {
        panic!("expected Value cell")
    };
    let Value::Object(public_bias) = &values.data()[1] else {
        panic!("expected dlarray learnable")
    };
    assert_eq!(
        public_bias.properties().get("Data"),
        Some(&Value::Tensor(bias.clone()))
    );
}

#[test]
fn network_learnables_reject_mismatched_identity_and_shape_before_publish() {
    let net = block_on(dlnetwork_builtin(vec![feature_network_layers()])).unwrap();
    let Value::Object(network) = net.value() else {
        panic!("expected network")
    };
    let Value::Object(table) = super::model::project_learnables(network, "set.Learnables").unwrap()
    else {
        panic!("expected table")
    };
    let mut variables = crate::builtins::table::table_variables(&table).unwrap();
    let Value::StringArray(names) =
        crate::builtins::deep_learning::tests::fixture_field_mut(&mut variables, "Layer").unwrap()
    else {
        panic!("expected Layer string array")
    };
    crate::builtins::deep_learning::tests::fixture_array_mut(names)[0] = "missing".into();
    let bad = crate::builtins::table::table_replace_variables_like(&table, variables).unwrap();
    assert!(
        super::model::apply_learnables(network.clone(), bad, "set.Learnables")
            .unwrap_err()
            .message
            .contains("identity differs")
    );

    let mut variables = crate::builtins::table::table_variables(&table).unwrap();
    let Value::Cell(values) =
        crate::builtins::deep_learning::tests::fixture_field_mut(&mut variables, "Value").unwrap()
    else {
        panic!("expected Value cell")
    };
    crate::builtins::deep_learning::tests::fixture_array_mut(values)[0] =
        Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap());
    let bad = crate::builtins::table::table_replace_variables_like(&table, variables).unwrap();
    assert!(
        super::model::apply_learnables(network.clone(), bad, "set.Learnables")
            .unwrap_err()
            .message
            .contains("shape")
    );
    assert!(!network.properties().contains_key("Learnables"));
}

#[test]
fn network_learnables_accepts_reordered_rows_but_rejects_duplicate_keys() {
    let net = block_on(dlnetwork_builtin(vec![feature_network_layers()])).unwrap();
    let Value::Object(network) = net.value() else {
        panic!("expected network")
    };
    let Value::Object(table) = super::model::project_learnables(network, "get.Learnables").unwrap()
    else {
        panic!("expected table")
    };
    let mut variables = crate::builtins::table::table_variables(&table).unwrap();
    for name in ["Layer", "Parameter"] {
        let Value::StringArray(column) =
            crate::builtins::deep_learning::tests::fixture_field_mut(&mut variables, name).unwrap()
        else {
            panic!("expected string column")
        };
        crate::builtins::deep_learning::tests::fixture_array_mut(column).swap(0, 1);
    }
    let Value::Cell(values) =
        crate::builtins::deep_learning::tests::fixture_field_mut(&mut variables, "Value").unwrap()
    else {
        panic!("expected Value cell")
    };
    crate::builtins::deep_learning::tests::fixture_array_mut(values).swap(0, 1);
    let reordered =
        crate::builtins::table::table_replace_variables_like(&table, variables.clone()).unwrap();
    assert!(super::model::apply_learnables(network.clone(), reordered, "set.Learnables").is_ok());
    let Value::StringArray(parameters) =
        crate::builtins::deep_learning::tests::fixture_field_mut(&mut variables, "Parameter")
            .unwrap()
    else {
        panic!("expected string column")
    };
    let duplicate_parameter = parameters.data()[0].clone();
    crate::builtins::deep_learning::tests::fixture_array_mut(parameters)[1] = duplicate_parameter;
    let Value::StringArray(layers) =
        crate::builtins::deep_learning::tests::fixture_field_mut(&mut variables, "Layer").unwrap()
    else {
        panic!("expected string column")
    };
    let duplicate_layer = layers.data()[0].clone();
    crate::builtins::deep_learning::tests::fixture_array_mut(layers)[1] = duplicate_layer;
    let duplicate =
        crate::builtins::table::table_replace_variables_like(&table, variables).unwrap();
    assert!(
        super::model::apply_learnables(network.clone(), duplicate, "set.Learnables")
            .unwrap_err()
            .message
            .contains("duplicate")
    );
}

#[test]
fn network_learnables_logical_storage_cannot_bypass_admission() {
    let net = block_on(dlnetwork_builtin(vec![feature_network_layers()])).unwrap();
    let Value::Object(network) = net.value() else {
        panic!("expected network")
    };
    let Value::Object(table) = super::model::project_learnables(network, "get.Learnables").unwrap()
    else {
        panic!("expected table")
    };
    let mut variables = crate::builtins::table::table_variables(&table).unwrap();
    let Value::Cell(values) =
        crate::builtins::deep_learning::tests::fixture_field_mut(&mut variables, "Value").unwrap()
    else {
        panic!("expected Value cell")
    };
    crate::builtins::deep_learning::tests::fixture_array_mut(values)[0] = Value::Bool(false);
    let invalid = crate::builtins::table::table_replace_variables_like(&table, variables).unwrap();
    assert!(
        super::model::apply_learnables(network.clone(), invalid, "set.Learnables")
            .unwrap_err()
            .message
            .contains("real numeric storage")
    );
}

#[test]
fn network_learnables_empty_then_replacement_updates_initialization_lifecycle() {
    let net = block_on(dlnetwork_builtin(vec![feature_network_layers()])).unwrap();
    let Value::Object(network) = net.value() else {
        panic!("expected network")
    };
    let Value::Object(table) = super::model::project_learnables(network, "set.Learnables").unwrap()
    else {
        panic!("expected table")
    };
    let mut variables = crate::builtins::table::table_variables(&table).unwrap();
    let Value::Cell(values) =
        crate::builtins::deep_learning::tests::fixture_field_mut(&mut variables, "Value").unwrap()
    else {
        panic!("expected Value cell")
    };
    crate::builtins::deep_learning::tests::fixture_array_mut(values)[1] =
        Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap());
    let empty = crate::builtins::table::table_replace_variables_like(&table, variables).unwrap();
    let Value::Object(uninitialized) =
        super::model::apply_learnables(network.clone(), empty, "set.Learnables").unwrap()
    else {
        panic!("expected network")
    };
    assert_eq!(
        uninitialized.properties().get("Initialized"),
        Some(&Value::Bool(false))
    );
    let Value::Cell(layers) = uninitialized.properties().get("Layers").unwrap() else {
        panic!("expected layers")
    };
    let Value::Object(fc) = &layers.data()[1] else {
        panic!("expected FC")
    };
    assert!(!fc.properties().contains_key("Bias"));

    let Value::Object(table) =
        super::model::project_learnables(&uninitialized, "get.Learnables").unwrap()
    else {
        panic!("expected table")
    };
    let mut variables = crate::builtins::table::table_variables(&table).unwrap();
    let Value::Cell(values) =
        crate::builtins::deep_learning::tests::fixture_field_mut(&mut variables, "Value").unwrap()
    else {
        panic!("expected Value cell")
    };
    crate::builtins::deep_learning::tests::fixture_array_mut(values)[1] =
        Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![2, 1]).unwrap());
    let restored = crate::builtins::table::table_replace_variables_like(&table, variables).unwrap();
    let Value::Object(initialized) =
        super::model::apply_learnables(uninitialized, restored, "set.Learnables").unwrap()
    else {
        panic!("expected network")
    };
    assert_eq!(
        initialized.properties().get("Initialized"),
        Some(&Value::Bool(true))
    );
}

#[test]
fn dlnetwork_rejects_unknown_options_and_preserves_resident_learnables() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let input = block_on(feature_input_layer_builtin(
            Value::Num(2.0),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        let mut fc = block_on(
            crate::builtins::deep_learning::tests::fully_connected_layer_fixture(
                Value::Num(1.0),
                vec![],
            ),
        )
        .unwrap();
        let weights = provider
            .upload(&HostTensorView {
                data: &[2.0, 3.0],
                shape: &[1, 2],
            })
            .unwrap();
        let Value::Object(fc_object) = &mut fc else {
            panic!("layer object expected")
        };
        fc_object
            .replace_dl_property("Weights", Value::GpuTensor(weights))
            .expect("fixture property");
        let layers = Value::Cell(CellArray::new(vec![input, fc], 2, 1).unwrap());
        provider.reset_telemetry();
        let err = block_on(dlnetwork_builtin(vec![
            layers.clone(),
            Value::String("InputNames".into()),
            Value::String("x".into()),
        ]))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let net = block_on(dlnetwork_builtin(vec![layers])).unwrap();
        let Value::Object(object) = net.value() else {
            panic!("network object expected")
        };
        let Value::Cell(layers) = object.properties().get("Layers").unwrap() else {
            panic!("layers cell expected")
        };
        let Value::Object(fc) = &layers.data()[1] else {
            panic!("layer expected")
        };
        assert!(matches!(
            fc.properties().get("Weights"),
            Some(Value::GpuTensor(_))
        ));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn forward_rejects_host_known_late_matrix_error_before_resident_input_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let net_owner = block_on(dlnetwork_builtin(vec![dense_feature_network_layers()])).unwrap();
        let mut net = net_owner.value().clone();
        let Value::Object(net_object) = &mut net else {
            panic!("network expected")
        };
        let Value::Cell(layers) = net_object.fixture_dl_property_mut("Layers").unwrap() else {
            panic!("layers expected")
        };
        let Value::Object(fc) =
            &mut crate::builtins::deep_learning::tests::fixture_array_mut(layers)[1]
        else {
            panic!("fc expected")
        };
        fc.replace_dl_property(
            "Weights",
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap()),
        )
        .expect("fixture property");
        let input = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[1, 2],
            })
            .unwrap();
        provider.reset_telemetry();
        let err = block_on(forward_builtin(
            net,
            Value::GpuTensor(input),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert!(err.message.contains("Weights has shape [1, 3]"));
        assert!(err.message.contains("expected Matrix"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn forward_integer_predictors_use_declared_runmat_extension_before_execution() {
    let net = block_on(dlnetwork_builtin(vec![dense_feature_network_layers()])).unwrap();
    let input =
        Value::Tensor(Tensor::new_integer(IntegerStorage::U16(vec![1, 2]), vec![1, 2]).unwrap());
    {
        let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = block_on(forward_builtin(
            net.value().clone(),
            input.clone(),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:compatibility:ForwardIntegerDataExtension")
        );
    }
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let Value::Tensor(result) = block_on(forward_builtin(
        net.value().clone(),
        input,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap() else {
        panic!("host numeric output expected")
    };
    assert_eq!(result.numeric_dtype(), NumericDType::F64);
    assert_eq!(result.shape(), &[1, 1]);
}

#[test]
fn forward_host_input_does_not_gather_unused_dlarray_metadata() {
    let net = block_on(dlnetwork_builtin(vec![dense_feature_network_layers()])).unwrap();
    let mut input = block_on(dlarray_builtin(
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .unwrap();
    let Value::Object(object) = &mut input else {
        panic!("dlarray expected")
    };
    object
        .replace_dl_property(
            "UnusedMetadata",
            Value::GpuTensor(allocation.handle.clone()),
        )
        .expect("fixture property");
    let output = block_on(forward_builtin(
        net.value().clone(),
        input,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect("unused resident metadata() must not require a provider");
    let Value::Object(output) = output else {
        panic!("dlarray result expected")
    };
    assert!(
        matches!(output.properties().get("Data"), Some(Value::Tensor(data)) if data.shape() == [1, 1])
    );
    assert_eq!(allocation.probe.retirements(), 0);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn dlnetwork_initialize_false_preserves_uninitialized_learnables() {
    let input = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let fc = block_on(
        crate::builtins::deep_learning::tests::fully_connected_layer_fixture(
            Value::Num(2.0),
            vec![],
        ),
    )
    .unwrap();
    let layers = Value::Cell(CellArray::new(vec![input, fc], 2, 1).unwrap());
    let net = block_on(dlnetwork_builtin(vec![
        layers,
        Value::String("Initialize".into()),
        Value::Bool(false),
    ]))
    .unwrap();
    let Value::Object(object) = net.value() else {
        panic!("expected dlnetwork object");
    };
    assert_eq!(
        object.properties().get("Initialized"),
        Some(&Value::Bool(false))
    );
    let Value::Object(learnables) = super::model::project_learnables(object, "dlnetwork").unwrap()
    else {
        panic!("expected learnables table");
    };
    let variables = crate::builtins::table::table_variables(&learnables).unwrap();
    let Value::Cell(values) = variables.fields.get("Value").unwrap() else {
        panic!("expected learnable values");
    };
    assert_eq!(values.data().len(), 2);
    assert!(values
        .data()
        .iter()
        .all(|value| matches!(value, Value::Tensor(tensor) if tensor.shape() == [0, 0])));

    let err = block_on(forward_builtin(
        net.value().clone(),
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:deepLearning:InvalidInput"));
    assert!(
        err.message
            .contains("parameters and state must be initialized"),
        "{err}"
    );
}

#[test]
fn forward_and_predict_execute_supported_feedforward_networks() {
    let net = block_on(dlnetwork_builtin(vec![feature_network_layers()])).unwrap();
    let input = Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap());
    let out = block_on(forward_builtin(
        net.value().clone(),
        input.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let Value::Tensor(tensor) = out else {
        panic!("expected tensor output");
    };
    assert_eq!(tensor.shape(), &[2, 2]);
    assert!((tensor.materialize_f64()[0] - 0.5).abs() < 1.0e-12);
    assert!((tensor.materialize_f64()[1] - 0.8807970779778823).abs() < 1.0e-12);
    assert!((tensor.materialize_f64()[2] - 0.5).abs() < 1.0e-12);
    assert!((tensor.materialize_f64()[3] - 0.11920292202211755).abs() < 1.0e-12);

    let predicted = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
        net.value().clone(),
        input,
        vec![],
    ))
    .unwrap();
    let Value::Tensor(predicted) = predicted else {
        panic!("expected predict tensor");
    };
    assert_eq!(predicted.shape(), &[2, 2]);
    assert!((predicted.materialize_f64()[1] - 0.8807970779778823).abs() < 1.0e-12);
}

#[test]
fn canonical_forward_uses_invocation_outputs_and_cancellation() {
    use std::rc::Rc;

    use runmat_execution::{cancellation::CancellationSource, CancellationReason};

    let network_owner = block_on(dlnetwork_builtin(vec![feature_network_layers()])).unwrap();
    let network = network_owner.value().clone();
    let input = Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap());
    let args = [network, input];
    let bindings = crate::builtin::runtime_builtin_bindings_by_name("forward");
    assert_eq!(bindings.len(), 1);
    let runtime = crate::context::RuntimeContext::new(Rc::new(
        crate::execution::RuntimeExecutionService::new(),
    ));
    let _ambient_outputs =
        crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let output = block_on((bindings[0].implementation)(
        crate::builtin::RuntimeBuiltinInvocation::new(&args, Some(runtime.clone()), Some(2)),
    ))
    .unwrap();
    assert_eq!(
        output.kind(),
        runmat_value::ValueSequenceKind::CommaSeparated
    );
    assert_eq!(output.len(), 2);
    let output = block_on((bindings[0].implementation)(
        crate::builtin::RuntimeBuiltinInvocation::new(&args, Some(runtime.clone()), Some(1)),
    ))
    .unwrap();
    assert_eq!(output.kind(), runmat_value::ValueSequenceKind::Single);
    assert_eq!(output.len(), 1);

    runtime.set_error_namespace("Invocation");
    let source = CancellationSource::new();
    source.cancel(CancellationReason::User);
    let cancelled = runtime.clone().with_cancellation_token(source.token());
    let error = block_on((bindings[0].implementation)(
        crate::builtin::RuntimeBuiltinInvocation::new(&args, Some(cancelled), Some(1)),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("Invocation:Cancelled"));

    let error = block_on((bindings[0].implementation)(
        crate::builtin::RuntimeBuiltinInvocation::new(&args, Some(runtime), Some(3)),
    ))
    .unwrap_err();
    assert!(error.message().contains("at most two outputs"));
}

#[test]
fn forward_preserves_native_single_through_fully_connected_and_softmax() {
    let input_layer = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let fully_connected = block_on(fully_connected_layer_builtin(
        None,
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
    let softmax = block_on(softmax_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let network_owner = block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input_layer, fully_connected, softmax], 3, 1).unwrap(),
    )]))
    .unwrap();
    let network = network_owner.value().clone();
    let input =
        Value::Tensor(Tensor::from_f32(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).expect("input"));
    let output = block_on(forward_builtin(
        network,
        input,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect("forward");
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
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let input_layer = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let fc = block_on(fully_connected_layer_builtin(
        None,
        Value::Num(1.0),
        vec![
            Value::String("Weights".into()),
            poisoned_int_tensor(IntegerStorage::I32(vec![10, 1]), vec![1, 2], 99.0),
            Value::String("Bias".into()),
            poisoned_int_tensor(IntegerStorage::I32(vec![5]), vec![1, 1], 99.0),
        ],
    ))
    .unwrap();
    let net_owner = block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input_layer, fc], 2, 1).unwrap(),
    )]))
    .unwrap();
    let net = net_owner.value().clone();
    let input = poisoned_int_tensor(IntegerStorage::I32(vec![1, 2, 3, 4]), vec![2, 2], 99.0);

    let out = block_on(forward_builtin(
        net.clone(),
        input.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let Value::Tensor(tensor) = out else {
        panic!("expected forward tensor output");
    };
    assert_eq!(tensor.shape(), &[2, 1]);
    assert_eq!(tensor.materialize_f64(), vec![18.0, 29.0]);

    let predicted = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
        net,
        input,
        vec![],
    ))
    .unwrap();
    let Value::Tensor(predicted) = predicted else {
        panic!("expected predict tensor output");
    };
    assert_eq!(predicted.shape(), &[2, 1]);
    assert_eq!(predicted.materialize_f64(), vec![18.0, 29.0]);
}

#[test]
fn predict_observations_in_columns_reads_typed_integer_storage_exactly() {
    let input_layer = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let fc = block_on(fully_connected_layer_builtin(
        None,
        Value::Num(1.0),
        vec![
            Value::String("Weights".into()),
            poisoned_int_tensor(IntegerStorage::I32(vec![10, 1]), vec![1, 2], 99.0),
            Value::String("Bias".into()),
            poisoned_int_tensor(IntegerStorage::I32(vec![5]), vec![1, 1], 99.0),
        ],
    ))
    .unwrap();
    let net_owner = block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input_layer, fc], 2, 1).unwrap(),
    )]))
    .unwrap();
    let net = net_owner.value().clone();
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
    assert_eq!(predicted.shape(), &[2, 1]);
    assert_eq!(predicted.materialize_f64(), vec![17.0, 39.0]);
    assert!(predicted.integer_storage().is_none());
}

#[test]
fn forward_preserves_dlarray_wrapper() {
    let net_owner = block_on(dlnetwork_builtin(vec![feature_network_layers()])).unwrap();
    let net = net_owner.value().clone();
    let data = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap());
    let dl = block_on(dlarray_builtin(
        data,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String("CB".into())]),
    ))
    .unwrap();
    let out = block_on(forward_builtin(
        net,
        dl,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let Value::Object(object) = out else {
        panic!("expected dlarray output");
    };
    assert_eq!(object.class_name().display_name(), "dlarray");
    assert_eq!(
        object.properties().get("Format"),
        Some(&Value::String("CB".into()))
    );
    assert!(matches!(
        object.properties().get("Data"),
        Some(Value::Tensor(_))
    ));
}

#[test]
fn forward_runs_fully_connected_gpu_dlarray_without_host_gather() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let net_owner = block_on(dlnetwork_builtin(vec![dense_feature_network_layers()])).unwrap();
        let net = net_owner.value().clone();
        let values = [1.0, 2.0, -1.0, 0.5, 3.0, -2.0];
        let shape = [2usize, 3usize]; // CB: two channels, three observations.
        let cpu_input = block_on(dlarray_builtin(
            Value::Tensor(Tensor::new(values.to_vec(), shape.to_vec()).unwrap()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "CB".into(),
            )]),
        ))
        .unwrap();
        let Value::Object(cpu) = block_on(forward_builtin(
            net.clone(),
            cpu_input,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap() else {
            panic!("expected CPU dlarray output");
        };
        let Some(Value::Tensor(expected)) = cpu.properties().get("Data") else {
            panic!("expected CPU tensor data");
        };
        let handle = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &shape,
            })
            .expect("upload");
        provider.reset_telemetry();
        let dl = block_on(dlarray_builtin(
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "CB".into(),
            )]),
        ))
        .expect("gpu dlarray");

        let output = block_on(forward_builtin(
            net,
            dl,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect("GPU forward");
        let Value::Object(output) = output else {
            panic!("expected GPU dlarray output");
        };
        let Some(Value::GpuTensor(handle)) = output.properties().get("Data") else {
            panic!("expected resident GPU output data");
        };
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let actual =
            crate::builtins::common::test_support::gather(Value::GpuTensor(handle.clone()))
                .expect("gather GPU output for parity assertion");
        assert_eq!(actual.shape(), expected.shape());
        assert_eq!(actual.materialize_f64(), expected.materialize_f64());
    });
}

#[test]
fn forward_resident_outputs_preserve_explicit_and_automatic_input_intent() {
    use runmat_accelerate_api::{handle_provenance, GpuHandleProvenance};
    crate::builtins::common::test_support::with_test_provider(|provider| {
        for layers in [dense_feature_network_layers(), elu_feature_network_layers()] {
            let net_owner = block_on(dlnetwork_builtin(vec![layers])).unwrap();
            let net = net_owner.value().clone();
            for intent in [
                GpuHandleProvenance::Automatic,
                GpuHandleProvenance::Explicit,
            ] {
                let uploaded = provider
                    .upload(&HostTensorView {
                        data: &[1.0, -2.0],
                        shape: &[1, 2],
                    })
                    .unwrap();
                let handle = uploaded.with_provenance(intent).unwrap();
                let dlarray = block_on(dlarray_builtin(
                    Value::GpuTensor(handle.clone()),
                    crate::builtins::common::test_support::builtin_arguments(vec![]),
                ))
                .unwrap();
                provider.reset_telemetry();
                let output = block_on(forward_builtin(
                    net.clone(),
                    dlarray,
                    crate::builtins::common::test_support::builtin_arguments(vec![]),
                ))
                .unwrap();
                let Value::Object(output) = output else {
                    panic!("dlarray expected")
                };
                let Some(Value::GpuTensor(result)) = output.properties().get("Data") else {
                    panic!("resident output expected")
                };
                assert_eq!(handle_provenance(result), Some(intent));
                assert_eq!(handle_provenance(&handle), Some(intent));
                assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
                let retained =
                    crate::builtins::common::test_support::gather(Value::GpuTensor(handle.clone()))
                        .unwrap();
                assert_eq!(retained.materialize_f64(), vec![1.0, -2.0]);
            }
        }
    });
}

#[test]
fn forward_runs_arbitrary_alpha_elu_gpu_dlarray_with_cpu_parity() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let net_owner = block_on(dlnetwork_builtin(vec![elu_feature_network_layers()])).unwrap();
        let net = net_owner.value().clone();
        let values = vec![-2.0, 0.0, 3.0, f64::NAN];
        let cpu = block_on(forward_builtin(
            net.clone(),
            Value::Tensor(Tensor::new(values.clone(), vec![2, 2]).unwrap()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
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
        let dl = block_on(dlarray_builtin(
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap();
        let output = block_on(forward_builtin(
            net,
            dl,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect("GPU ELU forward");
        let Value::Object(output) = output else {
            panic!("expected GPU dlarray output");
        };
        let Some(Value::GpuTensor(handle)) = output.properties().get("Data") else {
            panic!("expected resident GPU output data");
        };
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let actual =
            crate::builtins::common::test_support::gather(Value::GpuTensor(handle.clone()))
                .expect("gather GPU output for parity assertion");
        assert_eq!(actual.shape(), expected.shape());
        for (actual, expected) in actual
            .materialize_f64()
            .iter()
            .zip(&expected.materialize_f64())
        {
            assert!(actual == expected || (actual.is_nan() && expected.is_nan()));
        }
    });
}

#[test]
fn forward_runs_softmax_gpu_dlarray_without_host_gather() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let net_owner = block_on(dlnetwork_builtin(vec![feature_network_layers()])).unwrap();
        let net = net_owner.value().clone();
        let values = [1.0, 2.0, -1.0, 0.5, 3.0, -2.0];
        let shape = [2usize, 3usize]; // CB: two channels, three observations.
        let cpu_input = block_on(dlarray_builtin(
            Value::Tensor(Tensor::new(values.to_vec(), shape.to_vec()).unwrap()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "CB".into(),
            )]),
        ))
        .unwrap();
        let Value::Object(cpu) = block_on(forward_builtin(
            net.clone(),
            cpu_input,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect("CPU forward") else {
            panic!("expected CPU dlarray output");
        };
        let Some(Value::Tensor(expected)) = cpu.properties().get("Data") else {
            panic!("expected CPU tensor data");
        };
        let handle = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &shape,
            })
            .expect("upload");
        provider.reset_telemetry();
        let dl = block_on(dlarray_builtin(
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "CB".into(),
            )]),
        ))
        .expect("gpu dlarray");

        let output = block_on(forward_builtin(
            net,
            dl,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect("GPU softmax forward");
        let Value::Object(output) = output else {
            panic!("expected GPU dlarray output");
        };
        let Some(Value::GpuTensor(handle)) = output.properties().get("Data") else {
            panic!("expected resident GPU output data");
        };
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let actual =
            crate::builtins::common::test_support::gather(Value::GpuTensor(handle.clone()))
                .expect("gather GPU output for parity assertion");
        assert_eq!(actual.shape(), expected.shape());
        for (actual, expected) in actual
            .materialize_f64()
            .iter()
            .zip(&expected.materialize_f64())
        {
            assert!((actual - expected).abs() < 1.0e-12);
        }
    });
}

#[test]
fn forward_propagates_nonfinite_gpu_softmax_without_host_gather() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let net_owner = block_on(dlnetwork_builtin(vec![feature_network_layers()])).unwrap();
        let net = net_owner.value().clone();
        let values = [f64::INFINITY, 0.0, 0.0, 1.0];
        let shape = [2usize, 2usize];
        let handle = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &shape,
            })
            .expect("upload");
        provider.reset_telemetry();
        let dl = block_on(dlarray_builtin(
            Value::GpuTensor(handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect("gpu dlarray");

        let output = block_on(forward_builtin(
            net,
            dl,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect("nonfinite Softmax output remains resident");
        let Value::Object(output) = output else {
            panic!("expected GPU dlarray output");
        };
        let Some(Value::GpuTensor(result)) = output.properties().get("Data") else {
            panic!("expected resident GPU output data");
        };
        assert_eq!(result.shape(), &shape);
        assert_eq!(
            handle_precision(result),
            Some(runmat_accelerate_api::ProviderPrecision::F64)
        );
        assert_eq!(handle_storage(result), GpuTensorStorage::Real);
        assert_eq!(result.device_id(), handle.device_id());
        assert_ne!(result.view_identity(), handle.view_identity());
        assert!(result.allocation_lease().validate_live().is_ok());
        assert!(handle.allocation_lease().validate_live().is_ok());
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        let gathered =
            crate::builtins::common::test_support::gather(Value::GpuTensor(result.clone()))
                .expect("gather output for nonfinite oracle");
        assert_eq!(gathered.shape(), &shape);
        let values = gathered.materialize_f64();
        assert!(values[0].is_nan());
        assert!(values[2].is_nan());
        let expected = 1.0 / (1.0 + std::f64::consts::E);
        assert!((values[1] - expected).abs() < 1.0e-12);
        assert!((values[3] - (1.0 - expected)).abs() < 1.0e-12);
    });
}

#[test]
fn forward_rejects_complex_gpu_dlarray_before_dense_dispatch() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let net_owner = block_on(dlnetwork_builtin(vec![dense_feature_network_layers()])).unwrap();
        let net = net_owner.value().clone();
        let shape = [2usize, 1usize]; // CB: admitted channel extent is two.
        let handle = provider
            .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
                data: runmat_accelerate_api::HostNumericDataView::F64(&[1.0, 0.0, 2.0, 0.0]),
                shape: &shape,
                storage: GpuTensorStorage::ComplexInterleaved,
            })
            .expect("upload");
        provider.reset_telemetry();
        let dl = block_on(dlarray_builtin(
            Value::GpuTensor(handle),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "CB".into(),
            )]),
        ))
        .expect("gpu dlarray");

        let err = block_on(forward_builtin(
            net,
            dl,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .unwrap_err();

        assert_eq!(err.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert!(err
            .message()
            .contains("input must contain supported numeric data"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn export_onnx_network_writes_supported_feedforward_graph() {
    let input = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Name".into()),
            Value::String("input".into()),
        ]),
    ))
    .unwrap();
    let fc = block_on(fully_connected_layer_builtin(
        None,
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
    let relu = block_on(relu_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Name".into()),
            Value::String("relu".into()),
        ]),
    ))
    .unwrap();
    let elu = block_on(elu_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(1.25),
            Value::String("Name".into()),
            Value::String("elu".into()),
        ]),
    ))
    .unwrap();
    let softmax = block_on(softmax_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Name".into()),
            Value::String("prob".into()),
        ]),
    ))
    .unwrap();
    let output = block_on(classification_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Name".into()),
            Value::String("class".into()),
        ]),
    ))
    .unwrap();
    let net_owner = block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input, fc, relu, elu, softmax, output], 6, 1).unwrap(),
    )]))
    .unwrap();
    let net = net_owner.value().clone();

    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("network.onnx");
    let result = block_on(export_onnx_network_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            net,
            Value::String(path.to_string_lossy().into_owned().into()),
            Value::String("OpsetVersion".into()),
            Value::Num(14.0),
            Value::String("BatchSize".into()),
            Value::Num(4.0),
            Value::String("InputNames".into()),
            Value::String("features".into()),
            Value::String("OutputNames".into()),
            Value::String("scores".into()),
        ]),
    ))
    .unwrap();
    assert_eq!(result, Value::output_list(Vec::new()));

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
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let net_owner = block_on(dlnetwork_builtin(vec![feature_network_layers()])).unwrap();
    let net = net_owner.value().clone();
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("typed-options.onnx");
    let result = block_on(export_onnx_network_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            net,
            Value::String(path.to_string_lossy().into_owned().into()),
            Value::String("OpsetVersion".into()),
            cleared_int_tensor(IntegerStorage::U16(vec![14]), vec![1, 1]),
            Value::String("BatchSize".into()),
            cleared_int_tensor(IntegerStorage::U16(vec![4]), vec![1, 1]),
            Value::String("Verbose".into()),
            cleared_int_tensor(IntegerStorage::U8(vec![0]), vec![1, 1]),
        ]),
    ))
    .unwrap();
    assert_eq!(result, Value::output_list(Vec::new()));
    assert!(path.exists());
}

#[test]
fn export_onnx_network_rejects_unexportable_forms_deterministically() {
    let input = block_on(sequence_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let lstm = block_on(lstm_layer_builtin(
        Value::Num(4.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let net = object(
        "dlnetwork",
        vec![(
            "Layers",
            Value::Cell(CellArray::new(vec![input, lstm], 2, 1).unwrap()),
        )],
    )
    .expect("fixture object");
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("unsupported.onnx");
    let err = block_on(export_onnx_network_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            net,
            Value::String(path.to_string_lossy().into_owned().into()),
        ]),
    ))
    .unwrap_err();
    assert!(err.message.contains("has no supported ONNX lowering"));
    assert_eq!(err.identifier(), Some("RunMat:deepLearning:Unsupported"));
    assert!(
        !path.exists(),
        "unsupported lowering must not create a file"
    );

    let net_owner = block_on(dlnetwork_builtin(vec![feature_network_layers()])).unwrap();
    let net = net_owner.value().clone();
    let _guard = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
    let err = block_on(export_onnx_network_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            net,
            Value::String(
                tmp.path()
                    .join("out.onnx")
                    .to_string_lossy()
                    .into_owned()
                    .into(),
            ),
        ]),
    ))
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
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String("CB".into())]),
    ))
    .expect("dlarray");
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let out = block_on(dlfeval_builtin(vec![
        Value::FunctionHandle("__deep_learning_square_grad".into()),
        x,
    ]))
    .expect("dlfeval square grad");
    let Value::OutputList(values) = out.value() else {
        panic!("expected output list");
    };
    let Value::Object(loss) = &values[0] else {
        panic!("expected traced loss dlarray");
    };
    let Value::Tensor(loss_data) = loss.properties().get("Data").unwrap() else {
        panic!("expected loss tensor");
    };
    assert_eq!(loss_data.materialize_f64(), vec![14.0]);
    let Value::Object(grad) = &values[1] else {
        panic!("expected gradient dlarray");
    };
    let Value::Tensor(grad_data) = grad.properties().get("Data").unwrap() else {
        panic!("expected gradient tensor");
    };
    assert_eq!(grad_data.materialize_f64(), vec![2.0, 4.0, 6.0]);
    assert_eq!(
        grad.properties().get("Format"),
        Some(&Value::String("CB".into()))
    );
}

#[test]
fn dlfeval_dlgradient_preserves_native_single_storage() {
    let x = block_on(dlarray_builtin(
        Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String("CB".into())]),
    ))
    .expect("dlarray");
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let out = block_on(dlfeval_builtin(vec![
        Value::FunctionHandle("__deep_learning_square_grad".into()),
        x,
    ]))
    .expect("dlfeval square grad");
    let Value::OutputList(values) = out.value() else {
        panic!("expected output list");
    };
    let Value::Object(loss) = &values[0] else {
        panic!("expected traced loss dlarray");
    };
    let Value::Tensor(loss) = loss.properties().get("Data").unwrap() else {
        panic!("expected loss tensor");
    };
    assert_eq!(
        loss.clone().into_numeric_storage().expect("loss storage"),
        NumericStorage::F32(vec![14.0])
    );
    let Value::Object(gradient) = &values[1] else {
        panic!("expected gradient dlarray");
    };
    let Value::Tensor(gradient) = gradient.properties().get("Data").unwrap() else {
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
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Name".into()),
            Value::String("in".into()),
        ]),
    ))
    .unwrap();
    let fc = block_on(fully_connected_layer_builtin(
        None,
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
    let net_owner = block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input, fc], 2, 1).unwrap(),
    )]))
    .unwrap();
    let net = net_owner.value().clone();
    let x = block_on(dlarray_builtin(
        Value::Tensor(Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String("CB".into())]),
    ))
    .unwrap();

    let gradients = block_on(dlfeval_builtin(vec![
        Value::FunctionHandle("__deep_learning_network_grad".into()),
        net,
        x,
    ]))
    .expect("network gradients");
    let Value::Object(table) = gradients.value() else {
        panic!("expected learnables gradient table");
    };
    let variables = crate::builtins::table::table_variables(table).unwrap();
    let Value::Cell(values) = variables.fields.get("Value").unwrap() else {
        panic!("expected Value cell");
    };
    let Value::Object(weight_grad) = &values.data()[0] else {
        panic!("expected dlarray weight gradient");
    };
    let Value::Tensor(weight_data) = weight_grad.properties().get("Data").unwrap() else {
        panic!("expected tensor weight gradient");
    };
    // CB columns are samples (1,3) and (2,4). A summed output has unit
    // upstream adjoints, so each output row receives channel sums (3,7).
    assert_eq!(weight_data.materialize_f64(), vec![3.0, 3.0, 7.0, 7.0]);
    let Value::Object(bias_grad) = &values.data()[1] else {
        panic!("expected dlarray bias gradient");
    };
    let Value::Tensor(bias_data) = bias_grad.properties().get("Data").unwrap() else {
        panic!("expected tensor bias gradient");
    };
    assert_eq!(bias_data.materialize_f64(), vec![2.0, 2.0]);
}

#[test]
fn dlupdate_traverses_dlnetwork_learnables_and_rebuilds_layers() {
    let net_owner = block_on(dlnetwork_builtin(vec![feature_network_layers()])).unwrap();
    let net = net_owner.value().clone();
    let out = block_on(dlupdate_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::FunctionHandle("plus".into()),
            net.clone(),
            net,
        ]),
    ))
    .expect("network dlupdate");
    let Value::Object(network) = out.value() else {
        panic!("expected network object");
    };
    let Value::Cell(layers) = network.properties().get("Layers").unwrap() else {
        panic!("expected layer cell");
    };
    let Value::Object(fc) = &layers.data()[1] else {
        panic!("expected fc layer");
    };
    let weights = match fc.properties().get("Weights").unwrap() {
        Value::Tensor(weights) => weights,
        Value::Object(dlarray) => match dlarray.properties().get("Data").unwrap() {
            Value::Tensor(weights) => weights,
            other => panic!("expected tensor data, got {other:?}"),
        },
        other => panic!("expected tensor or dlarray weights, got {other:?}"),
    };
    assert_eq!(weights.materialize_f64(), vec![2.0, -2.0, 0.0, 2.0]);
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
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "CB".into(),
            )]),
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
fn default_recurrent_network_initializes_canonical_parameters_and_state() {
    let input = block_on(sequence_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let lstm = block_on(lstm_layer_builtin(
        Value::Num(4.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let network_owner = block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input, lstm], 2, 1).unwrap(),
    )]))
    .expect("recognized recurrent layers initialize from known input geometry");
    let Value::Object(network) = network_owner.value() else {
        panic!("network expected");
    };
    assert_eq!(
        network.properties().get("Initialized"),
        Some(&Value::Bool(true))
    );
    let learnables = super::model::project_learnables(network, "get.Learnables").unwrap();
    assert!(
        matches!(learnables, Value::Object(table) if table.class_name().is(runmat_types::standard::TABLE))
    );
    let state = block_on(network_owner.with_owned_async(|network| async move {
        let state = super::model::get_state_builtin(network).await?;
        crate::execution::RootedValue::new(state).map_err(|error| error.into_runtime_error())
    }))
    .unwrap();
    assert!(
        matches!(state.value(), Value::Object(table) if table.class_name().is(runmat_types::standard::TABLE))
    );
}

fn regression_training_layers() -> Value {
    let input = block_on(feature_input_layer_builtin(
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Name".into()),
            Value::String("x".into()),
        ]),
    ))
    .unwrap();
    let fc = block_on(fully_connected_layer_builtin(
        None,
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
    let output = block_on(regression_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Name".into()),
            Value::String("out".into()),
        ]),
    ))
    .unwrap();
    Value::Cell(CellArray::new(vec![input, fc, output], 3, 1).unwrap())
}

fn classification_training_layers() -> Value {
    let input = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Name".into()),
            Value::String("features".into()),
        ]),
    ))
    .unwrap();
    let fc = block_on(fully_connected_layer_builtin(
        None,
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
    let softmax = block_on(softmax_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Name".into()),
            Value::String("prob".into()),
        ]),
    ))
    .unwrap();
    let output = block_on(classification_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Name".into()),
            Value::String("class".into()),
        ]),
    ))
    .unwrap();
    Value::Cell(CellArray::new(vec![input, fc, softmax, output], 4, 1).unwrap())
}

fn adam_training_options() -> Value {
    block_on(training_options_builtin(
        Value::String("adam".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![
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
        ]),
    ))
    .unwrap()
}

#[test]
fn train_network_regression_updates_weights_and_predicts() {
    let x = Value::Tensor(Tensor::new(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]).unwrap());
    let y = Value::Tensor(Tensor::new(vec![1.0, 3.0, 5.0, 7.0], vec![4, 1]).unwrap());
    let net = block_on(train_network_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            x.clone(),
            y,
            regression_training_layers(),
            adam_training_options(),
        ]),
    ))
    .unwrap();
    let Value::Object(object) = net.value() else {
        panic!("expected trained network object");
    };
    assert_eq!(object.class_name().display_name(), "SeriesNetwork");
    assert!(matches!(
        object.properties().get("TrainingInfo"),
        Some(Value::Struct(_))
    ));
    let predicted = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
        net.value().clone(),
        x,
        vec![],
    ))
    .unwrap();
    let Value::Tensor(predicted) = predicted else {
        panic!("expected prediction tensor");
    };
    assert_eq!(predicted.shape(), &[4, 1]);
    assert!((predicted.materialize_f64()[0] - 1.0).abs() < 0.35);
    assert!((predicted.materialize_f64()[3] - 7.0).abs() < 0.35);
}

#[test]
fn train_network_regression_reads_typed_integer_storage_exactly() {
    let x = poisoned_int_tensor(IntegerStorage::I32(vec![0, 1, 2, 3]), vec![4, 1], 99.0);
    let y = poisoned_int_tensor(IntegerStorage::I32(vec![1, 3, 5, 7]), vec![4, 1], 99.0);
    let net = block_on(train_network_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            x.clone(),
            y,
            regression_training_layers(),
            adam_training_options(),
        ]),
    ))
    .unwrap();

    let predicted = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
        net.value().clone(),
        x,
        vec![],
    ))
    .unwrap();
    let Value::Tensor(predicted) = predicted else {
        panic!("expected prediction tensor");
    };
    assert_eq!(predicted.shape(), &[4, 1]);
    assert!((predicted.materialize_f64()[0] - 1.0).abs() < 0.35);
    assert!((predicted.materialize_f64()[3] - 7.0).abs() < 0.35);
}

#[test]
fn supervised_classification_keeps_wide_integer_label_identities_distinct() {
    let first = (1_u64 << 53) + 1;
    let labels = poisoned_int_tensor(IntegerStorage::U64(vec![first, first + 1]), vec![2, 1], 0.0);
    let options = block_on(training_options_builtin(
        Value::String("adam".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("MaxEpochs".into()),
            Value::Num(1.0),
        ]),
    ))
    .unwrap();
    let trained = block_on(train_network_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap()),
            labels,
            classification_training_layers(),
            options,
        ]),
    ))
    .unwrap();
    let Value::Object(network) = trained.value() else {
        panic!("expected trained network object")
    };
    let Some(Value::StringArray(classes)) = network.properties().get("Classes") else {
        panic!("expected classification classes")
    };
    assert_eq!(
        classes.data(),
        vec![first.to_string(), (first + 1).to_string()]
    );
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
    let net = block_on(train_network_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            x.clone(),
            y,
            classification_training_layers(),
            adam_training_options(),
        ]),
    ))
    .unwrap();
    let Value::Object(object) = net.value() else {
        panic!("expected trained network object");
    };
    let Value::StringArray(classes) = object.properties().get("Classes").unwrap() else {
        panic!("expected classes");
    };
    assert_eq!(classes.data(), vec!["left", "right"]);
    let predicted = block_on(crate::builtins::stats::ml::linear_model::predict_builtin(
        net.value().clone(),
        x,
        vec![],
    ))
    .unwrap();
    let Value::Tensor(scores) = predicted else {
        panic!("expected scores");
    };
    assert!(scores.materialize_f64()[0] > scores.materialize_f64()[4]);
    assert!(scores.materialize_f64()[5] > scores.materialize_f64()[1]);
}

#[test]
fn trainnet_crossentropy_supports_categorical_labels() {
    let x = Value::Tensor(
        Tensor::new(vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0], vec![4, 2]).unwrap(),
    );
    let mut categorical = ObjectInstance::new(runmat_types::standard::CATEGORICAL);
    categorical
        .replace_dl_property(
            "Codes",
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 1.0, 2.0], vec![4, 1]).unwrap()),
        )
        .expect("fixture property");
    categorical
        .replace_dl_property(
            "Categories",
            Value::StringArray(
                StringArray::new(vec!["left".into(), "right".into()], vec![1, 2]).unwrap(),
            ),
        )
        .expect("fixture property");
    let net = block_on(dlnetwork_builtin(vec![classification_training_layers()])).unwrap();
    let trained = block_on(trainnet_builtin(vec![
        x.clone(),
        Value::Object(categorical),
        net.value().clone(),
        Value::String("crossentropy".into()),
        adam_training_options(),
    ]))
    .unwrap();
    let Value::Object(object) = trained.value() else {
        panic!("expected dlnetwork");
    };
    assert_eq!(object.class_name().display_name(), "dlnetwork");
    assert!(matches!(
        object.properties().get("TrainingInfo"),
        Some(Value::Struct(_))
    ));
}

#[test]
fn trainnet_rejects_custom_loss_until_autodiff_exists() {
    let network = block_on(dlnetwork_builtin(vec![regression_training_layers()])).unwrap();
    let err = block_on(trainnet_builtin(vec![
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
        network.value().clone(),
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
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("ExecutionEnvironment".into()),
            Value::String("gpu".into()),
        ]),
    ))
    .unwrap();
    let err = block_on(train_network_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap()),
            regression_training_layers(),
            opts,
        ]),
    ))
    .unwrap_err();
    assert!(err
        .message
        .contains("ExecutionEnvironment 'gpu' requires native GPU"));
}

#[test]
fn supervised_host_admission_precedes_resident_predictor_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let predictor = provider
            .upload(&HostTensorView {
                data: &[0.0, 1.0],
                shape: &[2, 1],
            })
            .unwrap();
        let wrong_rows = Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap());
        for trainnet in [false, true] {
            provider.reset_telemetry();
            let network = if trainnet {
                Some(block_on(dlnetwork_builtin(vec![regression_training_layers()])).unwrap())
            } else {
                None
            };
            let args = if trainnet {
                vec![
                    Value::GpuTensor(predictor.clone()),
                    wrong_rows.clone(),
                    network.as_ref().expect("trainnet network").value().clone(),
                    Value::String("mse".into()),
                    adam_training_options(),
                ]
            } else {
                vec![
                    Value::GpuTensor(predictor.clone()),
                    wrong_rows.clone(),
                    regression_training_layers(),
                    adam_training_options(),
                ]
            };
            let error = if trainnet {
                block_on(trainnet_builtin(args)).unwrap_err()
            } else {
                block_on(train_network_builtin(
                    crate::builtins::common::test_support::builtin_arguments(args),
                ))
                .unwrap_err()
            };
            assert!(error.message.contains("one row per observation"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        }

        provider.reset_telemetry();
        let args = vec![
            Value::GpuTensor(predictor),
            Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap()),
            regression_training_layers(),
            adam_training_options(),
        ];
        let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
        let error = block_on(train_network_builtin(
            crate::builtins::common::test_support::builtin_arguments(args),
        ))
        .unwrap_err();
        assert!(error.message.contains("at most 2"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn supervised_rejects_mutated_elu_alpha_before_resident_predictor_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let predictor = provider
            .upload(&HostTensorView {
                data: &[0.0, 1.0],
                shape: &[2, 1],
            })
            .unwrap();
        let invalid_alpha = [
            Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            Value::Num(f64::NAN),
        ];
        for alpha in invalid_alpha {
            let Value::Cell(layers) = regression_training_layers() else {
                panic!("expected regression layer collection")
            };
            let mut values = layers
                .into_parts()
                .0
                .try_into_owned()
                .expect("fresh fixture cell")
                .try_into_unbounded_vec()
                .expect("ordinary fixture cell");
            let mut elu_value = block_on(elu_layer_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            ))
            .unwrap();
            let Value::Object(ref mut elu_object) = elu_value else {
                panic!("expected ELU layer")
            };
            elu_object
                .replace_dl_property("Alpha", alpha)
                .expect("fixture property");
            values.insert(2, elu_value);
            let layers = Value::Cell(CellArray::new(values, 4, 1).unwrap());

            provider.reset_telemetry();
            let error = block_on(train_network_builtin(
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::GpuTensor(predictor.clone()),
                    Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap()),
                    layers,
                    adam_training_options(),
                ]),
            ))
            .unwrap_err();
            assert!(error.message.contains("Alpha"), "{error:?}");
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        }
    });
}

#[test]
fn supervised_integer_learnable_rejects_before_initializer_callback_and_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let predictor = provider
            .upload(&HostTensorView {
                data: &[0.0, 1.0],
                shape: &[2, 1],
            })
            .unwrap();
        let Value::Cell(mut layers) = regression_training_layers() else {
            panic!("expected regression layers")
        };
        let Value::Object(fc) =
            &mut crate::builtins::deep_learning::tests::fixture_array_mut(&mut layers)[1]
        else {
            panic!("expected fully connected layer")
        };
        fc.replace_dl_property(
            "Weights",
            Value::Tensor(Tensor::new_integer(IntegerStorage::I32(vec![1]), vec![1, 1]).unwrap()),
        )
        .expect("fixture property");
        fc.replace_dl_property(
            "Bias",
            Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
        )
        .expect("fixture property");
        fc.replace_dl_property(
            "BiasInitializer",
            Value::FunctionHandle("__missing_training_initializer_callback".into()),
        )
        .expect("fixture property");
        provider.reset_telemetry();
        let error = block_on(train_network_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(predictor),
                Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap()),
                Value::Cell(layers),
                adam_training_options(),
            ]),
        ))
        .unwrap_err();
        assert!(
            error
                .message
                .contains("filled Weights must be single or double"),
            "{error:?}"
        );
        assert!(!error.message.contains("initializer"), "{error:?}");
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn supervised_rejects_empty_predictor_observations() {
    let empty = Value::Tensor(Tensor::new(Vec::new(), vec![0, 1]).unwrap());
    let error = block_on(train_network_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            empty,
            Value::Tensor(Tensor::new(Vec::new(), vec![0, 1]).unwrap()),
            regression_training_layers(),
            adam_training_options(),
        ]),
    ))
    .unwrap_err();
    assert!(error.message.contains("at least one observation"));
}

#[test]
fn supervised_target_mismatch_precedes_initializer_admission_and_effects() {
    let input = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let mut dense = block_on(
        crate::builtins::deep_learning::tests::fully_connected_layer_fixture(
            Value::Num(3.0),
            vec![],
        ),
    )
    .unwrap();
    let Value::Object(ref mut dense_object) = dense else {
        panic!("fully connected layer expected")
    };
    // If target admission were delayed until after initialization, this
    // deliberately invalid initializer would win (and a callable one could
    // run). The target mismatch must be rejected while the plan is pure.
    dense_object
        .replace_dl_property(
            "WeightsInitializer",
            Value::String("not-an-initializer".into()),
        )
        .expect("fixture property");
    let layers = Value::Cell(CellArray::new(vec![input, dense], 2, 1).unwrap());
    let error = block_on(trainnet_builtin(vec![
        Value::Tensor(Tensor::new(vec![1.; 4], vec![2, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.; 2], vec![2, 1]).unwrap()),
        layers,
        Value::String("mse".into()),
        adam_training_options(),
    ]))
    .unwrap_err();
    assert!(error.message.contains("Y has 1 channels"), "{error:?}");
    assert!(!error.message.contains("WeightsInitializer"), "{error:?}");
}

#[test]
fn supervised_training_observes_pre_cancelled_execution() {
    let cancellation = runmat_execution::cancellation::CancellationSource::new();
    cancellation.cancel(runmat_execution::CancellationReason::User);
    let _guard = crate::interrupt::replace_interrupt(Some(cancellation.token()));
    let options = block_on(training_options_builtin(
        Value::String("adam".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("MaxEpochs".into()),
            Value::Num(1.0),
        ]),
    ))
    .unwrap();
    let error = block_on(train_network_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap()),
            regression_training_layers(),
            options,
        ]),
    ))
    .unwrap_err();
    assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
}

#[test]
fn supervised_two_outputs_include_typed_loss_history() {
    let options = block_on(training_options_builtin(
        Value::String("adam".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("MaxEpochs".into()),
            Value::Num(2.0),
        ]),
    ))
    .unwrap();
    let args = vec![
        Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap()),
        regression_training_layers(),
        options,
    ];
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let result = block_on(train_network_builtin(
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .unwrap();
    let Value::OutputList(outputs) = result.value() else {
        panic!("expected two requested outputs")
    };
    assert_eq!(outputs.len(), 2);
    assert!(matches!(outputs[0], Value::Object(_)));
    let Value::Struct(info) = &outputs[1] else {
        panic!("expected training information")
    };
    let Value::Tensor(loss_history) = &info.fields["TrainingLoss"] else {
        panic!("expected typed loss history")
    };
    assert_eq!(loss_history.shape(), &[2, 1]);
    assert_eq!(loss_history.numeric_dtype(), NumericDType::F64);
    assert!(loss_history
        .materialize_f64()
        .iter()
        .all(|loss| loss.is_finite()));
}

#[test]
fn supervised_training_gathers_only_consumed_nested_learnables() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let weights = provider
            .upload(&HostTensorView {
                data: &[0.0],
                shape: &[1, 1],
            })
            .unwrap();
        let unrelated = provider
            .upload(&HostTensorView {
                data: &[7.0, 8.0],
                shape: &[1, 2],
            })
            .unwrap();
        let Value::Cell(mut layers) = regression_training_layers() else {
            panic!("expected layer collection")
        };
        let Value::Object(fc) =
            &mut crate::builtins::deep_learning::tests::fixture_array_mut(&mut layers)[1]
        else {
            panic!("expected fully connected layer")
        };
        fc.replace_dl_property("Weights", Value::GpuTensor(weights))
            .expect("fixture property");
        // Unknown layer options are rejected by the typed layer schema. Put
        // this unrelated resident payload in opaque network metadata instead
        // and verify training still downloads only the consumed learnable.
        let network_owner = block_on(dlnetwork_builtin(vec![Value::Cell(layers)])).unwrap();
        let mut network = network_owner.value().clone();
        let Value::Object(network_object) = &mut network else {
            panic!("network expected")
        };
        network_object
            .replace_dl_property("Unrelated", Value::GpuTensor(unrelated))
            .expect("fixture property");
        let options = block_on(training_options_builtin(
            Value::String("adam".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("MaxEpochs".into()),
                Value::Num(1.0),
            ]),
        ))
        .unwrap();
        provider.reset_telemetry();
        let trained = block_on(train_network_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap()),
                network,
                options,
            ]),
        ))
        .unwrap();
        assert!(matches!(trained.value(), Value::Object(_)));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 8);
    });
}

#[test]
fn trainnet_single_precision_conv_pool_dense_step_uses_shared_kernels() {
    let input = block_on(sequence_input_layer_builtin(
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let conv = block_on(convolution_1d_layer_builtin(
        Value::Num(1.0),
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Weights".into()),
            Value::Tensor(Tensor::from_f32(vec![1.0], vec![1, 1, 1]).unwrap()),
            Value::String("Bias".into()),
            Value::Tensor(Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap()),
        ]),
    ))
    .unwrap();
    let pool = block_on(global_average_pooling_1d_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let dense = block_on(fully_connected_layer_builtin(
        None,
        Value::Num(1.0),
        vec![
            Value::String("Weights".into()),
            Value::Tensor(Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap()),
            Value::String("Bias".into()),
            Value::Tensor(Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap()),
        ],
    ))
    .unwrap();
    let network_owner = block_on(dlnetwork_builtin(vec![Value::Cell(
        CellArray::new(vec![input, conv, pool, dense], 4, 1).unwrap(),
    )]))
    .unwrap();
    let network = network_owner.value().clone();
    let predictor = block_on(dlarray_builtin(
        Value::Tensor(Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String("CBT".into())]),
    ))
    .unwrap();
    let target = Value::Tensor(Tensor::from_f32(vec![0.0, 0.0], vec![2, 1]).unwrap());
    let options = block_on(training_options_builtin(
        Value::String("sgdm".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("MaxEpochs".into()),
            Value::Num(1.0),
            Value::String("MiniBatchSize".into()),
            Value::Num(2.0),
            Value::String("InitialLearnRate".into()),
            Value::Num(0.01),
            Value::String("Momentum".into()),
            Value::Num(0.0),
            Value::String("Shuffle".into()),
            Value::String("never".into()),
        ]),
    ))
    .unwrap();
    let trained = block_on(trainnet_builtin(vec![
        predictor.clone(),
        target,
        network,
        Value::String("mse".into()),
        options,
    ]))
    .unwrap();
    let Value::Object(trained_object) = trained.value() else {
        panic!("expected trained dlnetwork")
    };
    let Some(Value::Struct(info)) = trained_object.properties().get("TrainingInfo") else {
        panic!("expected training information")
    };
    let Some(Value::Tensor(history)) = info.fields.get("TrainingLoss") else {
        panic!("expected training loss")
    };
    assert!((history.materialize_f64()[0] - 12.5).abs() < 1.0e-4);
    let prediction = block_on(forward_builtin(
        trained.value().clone(),
        predictor,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let Value::Object(prediction) = prediction else {
        panic!("formatted predictor should produce dlarray")
    };
    let Some(Value::Tensor(scores)) = prediction.properties().get("Data") else {
        panic!("expected native-class prediction data")
    };
    assert_eq!(scores.shape(), &[1, 2]);
    assert_eq!(scores.numeric_dtype(), NumericDType::F32);
    let scores = scores.as_f32_slice().unwrap();
    assert!(scores[0] < 3.0 && scores[0] > 0.0, "{scores:?}");
    assert!(scores[1] < 4.0 && scores[1] > 0.0, "{scores:?}");
}

#[test]
fn trainnet_recurrent_modes_preserve_canonical_state_across_minibatches() {
    let predictor = block_on(dlarray_builtin(
        Value::Tensor(Tensor::from_f32(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], vec![1, 2, 3]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String("CBT".into())]),
    ))
    .unwrap();
    for bidirectional in [false, true] {
        for sequence in [false, true] {
            let input = block_on(sequence_input_layer_builtin(
                Value::Num(1.0),
                crate::builtins::common::test_support::builtin_arguments(vec![]),
            ))
            .unwrap();
            let mode = if sequence { "sequence" } else { "last" };
            let recurrent_options = vec![
                Value::String("OutputMode".into()),
                Value::String(mode.into()),
            ];
            let recurrent = if bidirectional {
                block_on(bilstm_layer_builtin(
                    Value::Num(1.0),
                    crate::builtins::common::test_support::builtin_arguments(recurrent_options),
                ))
                .unwrap()
            } else {
                block_on(lstm_layer_builtin(
                    Value::Num(1.0),
                    crate::builtins::common::test_support::builtin_arguments(recurrent_options),
                ))
                .unwrap()
            };
            let network_owner = block_on(dlnetwork_builtin(vec![Value::Cell(
                CellArray::new(vec![input, recurrent], 2, 1).unwrap(),
            )]))
            .unwrap();
            let network = network_owner.value().clone();
            let Value::Object(initial) = &network else {
                panic!("expected dlnetwork")
            };
            let initial_state = initial
                .properties()
                .get("_State")
                .expect("canonical recurrent State")
                .clone();
            let channels = if bidirectional { 2 } else { 1 };
            let target_shape = if sequence {
                vec![2, channels, 3]
            } else {
                vec![2, channels]
            };
            let target = Value::Tensor(
                Tensor::from_f32(vec![0.0; target_shape.iter().product()], target_shape).unwrap(),
            );
            let options = block_on(training_options_builtin(
                Value::String("sgdm".into()),
                crate::builtins::common::test_support::builtin_arguments(vec![
                    Value::String("MaxEpochs".into()),
                    Value::Num(1.0),
                    Value::String("MiniBatchSize".into()),
                    Value::Num(2.0),
                    Value::String("InitialLearnRate".into()),
                    Value::Num(0.005),
                    Value::String("Momentum".into()),
                    Value::Num(0.0),
                    Value::String("Shuffle".into()),
                    Value::String("never".into()),
                ]),
            ))
            .unwrap();
            let trained = block_on(trainnet_builtin(vec![
                predictor.clone(),
                target,
                network,
                Value::String("mse".into()),
                options,
            ]))
            .unwrap();
            let Value::Object(trained) = trained.value() else {
                panic!("expected trained dlnetwork")
            };
            assert_eq!(trained.properties().get("_State"), Some(&initial_state));
            let Some(Value::Struct(info)) = trained.properties().get("TrainingInfo") else {
                panic!("expected training information")
            };
            let Some(Value::Tensor(history)) = info.fields.get("TrainingLoss") else {
                panic!("expected training loss history")
            };
            assert!(history.materialize_f64()[0].is_finite());
        }
    }
}

#[test]
fn train_network_normalization_activations_softmax_matches_classification_oracle() {
    let input = block_on(feature_input_layer_builtin(
        Value::Num(2.0),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let normalization = block_on(layer_normalization_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Scale".into()),
            Value::Tensor(Tensor::from_f32(vec![1.0, 1.0], vec![2, 1]).unwrap()),
            Value::String("Offset".into()),
            Value::Tensor(Tensor::from_f32(vec![0.0, 0.0], vec![2, 1]).unwrap()),
        ]),
    ))
    .unwrap();
    let relu = block_on(relu_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let elu = block_on(elu_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let dense = block_on(fully_connected_layer_builtin(
        None,
        Value::Num(2.0),
        vec![
            Value::String("Weights".into()),
            Value::Tensor(Tensor::from_f32(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap()),
            Value::String("Bias".into()),
            Value::Tensor(Tensor::from_f32(vec![0.0, 0.0], vec![2, 1]).unwrap()),
        ],
    ))
    .unwrap();
    let softmax = block_on(softmax_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let classification = block_on(classification_layer_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap();
    let layers = Value::Cell(
        CellArray::new(
            vec![
                input,
                normalization,
                relu,
                elu,
                dense,
                softmax,
                classification,
            ],
            7,
            1,
        )
        .unwrap(),
    );
    let predictor = Value::Tensor(Tensor::from_f32(vec![3.0, 1.0, 1.0, 3.0], vec![2, 2]).unwrap());
    let labels = Value::StringArray(
        StringArray::new(vec!["left".into(), "right".into()], vec![2, 1]).unwrap(),
    );
    let options = block_on(training_options_builtin(
        Value::String("sgdm".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("MaxEpochs".into()),
            Value::Num(1.0),
            Value::String("MiniBatchSize".into()),
            Value::Num(2.0),
            Value::String("InitialLearnRate".into()),
            Value::Num(0.01),
            Value::String("Momentum".into()),
            Value::Num(0.0),
            Value::String("Shuffle".into()),
            Value::String("never".into()),
            Value::String("Verbose".into()),
            Value::Bool(false),
        ]),
    ))
    .unwrap();
    let trained = block_on(train_network_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            predictor, labels, layers, options,
        ]),
    ))
    .unwrap();
    let Value::Object(trained) = trained.value() else {
        panic!("expected trained SeriesNetwork")
    };
    let Some(Value::Struct(info)) = trained.properties().get("TrainingInfo") else {
        panic!("expected training information")
    };
    let Some(Value::Tensor(history)) = info.fields.get("TrainingLoss") else {
        panic!("expected training loss history")
    };
    // With per-observation channel normalization, the two logits are close
    // to [1, 0] and [0, 1]. The first cross-entropy is -ln(sigmoid(1)).
    let expected = (1.0_f64 + (-1.0_f64).exp()).ln();
    assert!((history.materialize_f64()[0] - expected).abs() < 1.0e-3);
}

#[test]
fn supervised_verbose_controls_epoch_console_reporting() {
    for verbose in [false, true] {
        let options = block_on(training_options_builtin(
            Value::String("sgdm".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("MaxEpochs".into()),
                Value::Num(1.0),
                Value::String("MiniBatchSize".into()),
                Value::Num(2.0),
                Value::String("Verbose".into()),
                Value::Bool(verbose),
            ]),
        ))
        .unwrap();
        let capture = crate::console::begin_capture().expect("console capture");
        block_on(train_network_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
                Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap()),
                regression_training_layers(),
                options,
            ]),
        ))
        .unwrap();
        let console = capture.finish().expect("captured text");
        if verbose {
            assert!(console.contains("Epoch 1/1: training loss "), "{console}");
        } else {
            assert!(console.is_empty(), "{console}");
        }
    }
}

#[test]
fn supervised_l2_regularizes_weights_but_not_default_bias() {
    let Value::Cell(mut layers) = regression_training_layers() else {
        panic!("expected regression layers")
    };
    let Value::Object(dense) =
        &mut crate::builtins::deep_learning::tests::fixture_array_mut(&mut layers)[1]
    else {
        panic!("expected fully connected layer")
    };
    dense
        .replace_dl_property(
            "Weights",
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        )
        .expect("fixture property");
    let options = block_on(training_options_builtin(
        Value::String("sgdm".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("MaxEpochs".into()),
            Value::Num(1.0),
            Value::String("MiniBatchSize".into()),
            Value::Num(1.0),
            Value::String("InitialLearnRate".into()),
            Value::Num(0.1),
            Value::String("Momentum".into()),
            Value::Num(0.0),
            Value::String("L2Regularization".into()),
            Value::Num(0.5),
            Value::String("Verbose".into()),
            Value::Bool(false),
        ]),
    ))
    .unwrap();
    let trained = block_on(train_network_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
            Value::Cell(layers),
            options,
        ]),
    ))
    .unwrap();
    let Value::Object(network) = trained.value() else {
        panic!("expected trained network")
    };
    let Some(Value::Cell(layers)) = network.properties().get("Layers") else {
        panic!("expected canonical Layers")
    };
    let Value::Object(dense) = &layers.data()[1] else {
        panic!("expected trained fully connected layer")
    };
    let Some(Value::Tensor(weights)) = dense.properties().get("Weights") else {
        panic!("expected trained Weights")
    };
    let Some(Value::Tensor(bias)) = dense.properties().get("Bias") else {
        panic!("expected trained Bias")
    };
    assert!((weights.as_f64_slice().unwrap()[0] - 0.95).abs() < 1.0e-12);
    assert_eq!(bias.as_f64_slice().unwrap()[0], 0.0);
    let Some(Value::Struct(info)) = network.properties().get("TrainingInfo") else {
        panic!("expected training information")
    };
    let Some(Value::Tensor(history)) = info.fields.get("TrainingLoss") else {
        panic!("expected data-only loss history")
    };
    assert_eq!(history.as_f64_slice().unwrap(), &[0.0]);
}

#[test]
fn train_network_rmsprop_executes_native_optimizer_step() {
    let options = block_on(training_options_builtin(
        Value::String("rmsprop".into()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("MaxEpochs".into()),
            Value::Num(1.0),
            Value::String("MiniBatchSize".into()),
            Value::Num(2.0),
            Value::String("InitialLearnRate".into()),
            Value::Num(0.01),
            Value::String("Verbose".into()),
            Value::Bool(false),
        ]),
    ))
    .unwrap();
    let trained = block_on(train_network_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(Tensor::new(vec![0.0, 1.0], vec![2, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![1.0, 3.0], vec![2, 1]).unwrap()),
            regression_training_layers(),
            options,
        ]),
    ))
    .unwrap();
    let Value::Object(network) = trained.value() else {
        panic!("expected trained network")
    };
    assert_eq!(
        network.properties().get("TrainedWith"),
        Some(&Value::String("rmsprop".into()))
    );
    let Some(Value::Cell(layers)) = network.properties().get("Layers") else {
        panic!("expected canonical Layers")
    };
    let Value::Object(dense) = &layers.data()[1] else {
        panic!("expected fully connected layer")
    };
    let Some(Value::Tensor(weights)) = dense.properties().get("Weights") else {
        panic!("expected trained Weights")
    };
    assert!(weights.as_f64_slice().unwrap()[0] > 0.0);
}

#[test]
fn combvec_matches_neural_network_column_order() {
    let out = block_on(combvec_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]).unwrap()),
        ]),
    ))
    .unwrap();
    let Value::Tensor(t) = out else {
        panic!("expected tensor");
    };
    assert_eq!(t.shape(), &[3, 6]);
    assert_eq!(
        t.materialize_f64(),
        vec![
            1.0, 2.0, 10.0, 3.0, 4.0, 10.0, 1.0, 2.0, 20.0, 3.0, 4.0, 20.0, 1.0, 2.0, 30.0, 3.0,
            4.0, 30.0,
        ]
    );
}

#[test]
fn combvec_resident_rank_one_vector_has_host_row_geometry() {
    let row = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2]).unwrap());
    let other = Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![1, 2]).unwrap());
    let host = block_on(combvec_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![row, other.clone()]),
    ))
    .unwrap();
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let resident = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[2],
            })
            .unwrap();
        let output = block_on(combvec_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(resident.clone()),
                other,
            ]),
        ))
        .unwrap();
        assert_eq!(output, host);
        let Value::Tensor(matrix) = output else {
            panic!("combvec must produce a host matrix")
        };
        assert_eq!(matrix.shape(), &[2, 4]);
        assert_eq!(
            matrix.materialize_f64(),
            vec![1.0, 10.0, 2.0, 10.0, 1.0, 20.0, 2.0, 20.0]
        );
        assert!(resident.allocation_lease().validate_live().is_ok());
    });
}

#[test]
fn combvec_rank_one_capacity_rejects_before_resident_download() {
    let first = vec![1.0; 1001];
    let second = Value::Tensor(Tensor::new(vec![2.0; 1000], vec![1, 1000]).unwrap());
    let host_error = block_on(combvec_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(Tensor::new(first.clone(), vec![1001]).unwrap()),
            second.clone(),
        ]),
    ))
    .expect_err("1,001,000 Cartesian columns exceed capacity");
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let resident = provider
            .upload(&HostTensorView {
                data: &first,
                shape: &[1001],
            })
            .unwrap();
        provider.reset_telemetry();
        let error = block_on(combvec_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(resident.clone()),
                second,
            ]),
        ))
        .expect_err("resident rank-one input must have the same column count");
        assert_eq!(error.identifier(), host_error.identifier());
        assert!(error.to_string().contains("supported capacity"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert!(resident.allocation_lease().validate_live().is_ok());
    });
}

#[test]
fn combvec_reads_typed_integer_storage_exactly() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = block_on(combvec_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 3, 4]), vec![2, 2], 99.0),
            poisoned_int_tensor(IntegerStorage::U16(vec![10, 20]), vec![1, 2], 99.0),
        ]),
    ))
    .unwrap();
    let Value::Tensor(t) = out else {
        panic!("expected tensor");
    };
    assert_eq!(t.shape(), &[3, 4]);
    assert_eq!(
        t.materialize_f64(),
        vec![1.0, 2.0, 10.0, 3.0, 4.0, 10.0, 1.0, 2.0, 20.0, 3.0, 4.0, 20.0]
    );
}

#[test]
fn combvec_typed_integer_form_obeys_compatibility_profile() {
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(combvec_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(IntValue::U8(2))]),
    ))
    .expect_err("typed integer input is a RunMat extension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:CombvecIntegerInputExtension")
    );
}

#[test]
fn sequence_host_errors_precede_provider_downloads() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let values = [1.0, 2.0];
        let shape = [1usize, 2usize];
        let resident = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &shape,
            })
            .expect("resident input");
        provider.reset_telemetry();
        let invalid = block_on(combvec_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(resident.clone()),
                Value::String("not numeric".into()),
            ]),
        ))
        .expect_err("invalid later host input");
        assert_eq!(
            invalid.identifier(),
            Some("RunMat:deepLearning:InvalidInput")
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let bad_sequences =
            Value::Cell(CellArray::new(vec![Value::String("bad".into())], 1, 1).unwrap());
        let invalid = block_on(padsequences_builtin(
            bad_sequences,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                resident.clone(),
            )]),
        ))
        .expect_err("invalid host sequence before resident dimension");
        assert_eq!(
            invalid.identifier(),
            Some("RunMat:deepLearning:InvalidInput")
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let sequences = Value::Cell(CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap());
        let invalid = block_on(padsequences_builtin(
            sequences,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(resident),
                Value::String("NoSuchOption".into()),
                Value::Num(1.0),
            ]),
        ))
        .expect_err("invalid option before resident dimension");
        assert_eq!(
            invalid.identifier(),
            Some("RunMat:deepLearning:InvalidInput")
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn padsequences_selectively_gathers_nested_resident_numeric_sequences() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let shape = [1usize, 2usize];
        let resident = provider
            .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
                data: runmat_accelerate_api::HostNumericDataView::F32(&[1.25, 2.5]),
                shape: &shape,
                storage: GpuTensorStorage::Real,
            })
            .expect("single sequence upload");
        provider.reset_telemetry();
        let sequences =
            Value::Cell(CellArray::new(vec![Value::GpuTensor(resident.clone())], 1, 1).unwrap());
        let output = block_on(padsequences_builtin(
            sequences,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(2.0)]),
        ))
        .expect("resident single sequence should pad on host");
        let Value::Tensor(output) = output else {
            panic!("expected stacked tensor")
        };
        assert_eq!(output.numeric_dtype(), NumericDType::F32);
        assert_eq!(output.shape(), &[1, 2, 1]);
        assert_eq!(output.materialize_f64(), vec![1.25, 2.5]);
        assert!(provider.telemetry_snapshot().download_bytes > 0);

        let wide = (1_u64 << 53) + 1;
        let resident_wide = provider
            .upload_numeric(&runmat_accelerate_api::HostNumericTensorView {
                data: runmat_accelerate_api::HostNumericDataView::U64(&[wide]),
                shape: &[1, 1],
                storage: GpuTensorStorage::Real,
            })
            .expect("wide integer sequence upload");
        let sequences =
            Value::Cell(CellArray::new(vec![Value::GpuTensor(resident_wide)], 1, 1).unwrap());
        let output = block_on(padsequences_builtin(
            sequences,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(2.0)]),
        ))
        .expect("resident integer sequence should retain native class");
        let Value::Tensor(output) = output else {
            panic!("expected stacked tensor")
        };
        assert_eq!(output.numeric_dtype(), NumericDType::U64);
        assert_eq!(
            output
                .integer_storage()
                .and_then(|storage| storage.value_at(0)),
            Some(IntValue::U64(wide))
        );
    });
}

#[test]
fn padsequences_nested_resident_preflight_rejects_later_host_errors_without_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let shape = [1usize, 2usize];
        let resident = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &shape,
            })
            .unwrap();
        let bad_cell = Value::Cell(
            CellArray::new(
                vec![
                    Value::GpuTensor(resident.clone()),
                    Value::String("invalid".into()),
                ],
                1,
                2,
            )
            .unwrap(),
        );
        provider.reset_telemetry();
        let error = block_on(padsequences_builtin(
            bad_cell,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(2.0)]),
        ))
        .expect_err("later host cell is invalid");
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let valid_cell =
            Value::Cell(CellArray::new(vec![Value::GpuTensor(resident.clone())], 1, 1).unwrap());
        let error = block_on(padsequences_builtin(
            valid_cell,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "bad dim".into(),
            )]),
        ))
        .expect_err("host padding dimension is invalid");
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let mismatched = Value::Cell(
            CellArray::new(
                vec![
                    Value::GpuTensor(resident),
                    Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap()),
                ],
                1,
                2,
            )
            .unwrap(),
        );
        let error = block_on(padsequences_builtin(
            mismatched,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(2.0)]),
        ))
        .expect_err("known non-padding shape mismatch");
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[path = "tests/pad_options.rs"]
mod pad_options;

#[test]
fn padsequences_rank_admission_bounds_metadata_without_allocating_distant_axes() {
    let sequences = Value::Cell(CellArray::new(vec![Value::Num(7.0)], 1, 1).unwrap());
    let error = block_on(padsequences_builtin(
        sequences,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(IntValue::U64(
            super::MAX_PAD_ELEMENTS as u64,
        ))]),
    ))
    .expect_err("rank beyond the output-work metadata budget");
    assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));

    let sequences = Value::Cell(CellArray::new(vec![Value::Num(7.0)], 1, 1).unwrap());
    let output = block_on(padsequences_builtin(
        sequences,
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(4.0)]),
    ))
    .expect("ordinary higher-dimensional padding remains supported");
    let Value::Tensor(output) = output else {
        panic!("expected tensor")
    };
    assert_eq!(output.shape(), &[1, 1, 1, 1, 1]);
    assert_eq!(output.materialize_f64(), vec![7.0]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn padsequences_preserves_rank_zero_and_rank_one_sequence_geometry() {
    for (sequence, expected_shape, expected_values) in [
        (
            Tensor::new(vec![7.0], Vec::new()).unwrap(),
            vec![1, 1],
            vec![7.0],
        ),
        (
            Tensor::new(vec![7.0, 8.0], vec![2]).unwrap(),
            vec![2, 1],
            vec![7.0, 8.0],
        ),
    ] {
        let input = Value::Cell(CellArray::new(vec![Value::Tensor(sequence)], 1, 1).unwrap());
        let result = block_on(padsequences_builtin(
            input,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(1.0)]),
        ))
        .expect("pad exact-rank sequence");
        let Value::Tensor(result) = result else {
            panic!("expected stacked tensor")
        };
        assert_eq!(result.shape(), expected_shape.as_slice());
        assert_eq!(result.materialize_f64(), expected_values);
    }
}

#[test]
fn resident_padding_dimension_resolves_before_nested_sequence_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let resident_sequence = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[1, 2],
            })
            .expect("resident sequence");
        let resident_dimension = provider
            .upload(&HostTensorView {
                data: &[2.0],
                shape: &[1, 1],
            })
            .expect("resident dimension");
        let sequences = Value::Cell(
            CellArray::new(
                vec![
                    Value::GpuTensor(resident_sequence),
                    Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap()),
                ],
                1,
                2,
            )
            .unwrap(),
        );
        provider.reset_telemetry();
        let error = block_on(padsequences_builtin(
            sequences,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                resident_dimension,
            )]),
        ))
        .expect_err("known nonpadding shape mismatch");
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert_eq!(
            provider.telemetry_snapshot().download_bytes,
            u64::try_from(std::mem::size_of::<f64>()).expect("scalar byte size")
        );
    });
}

#[test]
fn sequence_resident_operands_gather_once_and_return_host_results() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let values = [1.0, 2.0];
        let shape = [1usize, 2usize];
        let columns = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &shape,
            })
            .expect("resident matrix");
        provider.reset_telemetry();
        let result = block_on(combvec_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(columns),
                Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![1, 2]).unwrap()),
            ]),
        ))
        .expect("host Cartesian result");
        let Value::Tensor(result) = result else {
            panic!("host tensor expected")
        };
        assert_eq!(result.shape(), &[2, 4]);
        assert_eq!(
            result.materialize_f64(),
            vec![1.0, 10.0, 2.0, 10.0, 1.0, 20.0, 2.0, 20.0]
        );
        assert!(provider.telemetry_snapshot().download_bytes > 0);

        let dimension = [2.0];
        let scalar_shape = [1usize, 1usize];
        let resident_dim = provider
            .upload(&HostTensorView {
                data: &dimension,
                shape: &scalar_shape,
            })
            .expect("resident dimension");
        let sequences = Value::Cell(
            CellArray::new(
                vec![
                    Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
                    Value::Num(3.0),
                ],
                1,
                2,
            )
            .unwrap(),
        );
        provider.reset_telemetry();
        let result = block_on(padsequences_builtin(
            sequences,
            crate::builtins::common::test_support::builtin_arguments(vec![Value::GpuTensor(
                resident_dim,
            )]),
        ))
        .expect("host padding result");
        let Value::Tensor(result) = result else {
            panic!("host tensor expected")
        };
        assert_eq!(result.shape(), &[1, 2, 2]);
        assert_eq!(result.materialize_f64(), vec![1.0, 2.0, 3.0, 0.0]);
        assert!(provider.telemetry_snapshot().download_bytes > 0);
    });
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
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::String("PaddingValue".into()),
            Value::Num(-1.0),
        ]),
    ))
    .unwrap();
    let Value::Tensor(tensor) = out else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.shape(), &[1, 2, 2]);
    assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0, 3.0, -1.0]);
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
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::String("PaddingValue".into()),
            Value::Num(-1.0),
        ]),
    ))
    .unwrap();
    let Value::Tensor(tensor) = out else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.shape(), &[1, 2, 2]);
    assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0, 3.0, -1.0]);
    assert_eq!(tensor.numeric_dtype(), NumericDType::F64);
}

#[test]
fn padsequences_uniform_mixed_storage_applies_padding_in_result_class() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let sequences = Value::Cell(
        CellArray::new(
            vec![
                Value::Tensor(Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).unwrap()),
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U8(vec![3]), vec![1, 1]).unwrap(),
                ),
            ],
            1,
            2,
        )
        .unwrap(),
    );
    let result = block_on(padsequences_builtin(
        sequences,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::String("PaddingValue".into()),
            Value::Num(-0.5),
        ]),
    ))
    .expect("mixed storage padding");
    let Value::OutputList(outputs) = result else {
        panic!("data and mask outputs")
    };
    let Value::Tensor(data) = &outputs[0] else {
        panic!("numeric padding output")
    };
    assert_eq!(data.numeric_dtype(), NumericDType::F64);
    assert_eq!(data.shape(), &[1, 2, 2]);
    assert_eq!(data.materialize_f64(), vec![1.0, 2.0, 3.0, -0.5]);
    let Value::LogicalArray(mask) = &outputs[1] else {
        panic!("logical validity mask")
    };
    assert_eq!(mask.shape.as_slice(), data.shape());
    assert_eq!(mask.data, vec![1, 1, 1, 0]);
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
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::String("Length".into()),
            Value::Num(2.0),
            Value::String("PaddingDirection".into()),
            Value::String("left".into()),
            Value::String("UniformOutput".into()),
            Value::Bool(false),
        ]),
    ))
    .unwrap();
    let Value::Cell(cell) = out else {
        panic!("expected cell");
    };
    assert_eq!(cell.shape, vec![1, 2]);
    let Value::Tensor(first) = &cell.data()[0] else {
        panic!("expected tensor");
    };
    assert_eq!(first.materialize_f64(), vec![2.0, 3.0]);
}

#[test]
fn padsequences_preserves_wide_integer_class_and_returns_logical_mask() {
    let wide = (1_u64 << 53) + 1;
    let sequences = Value::Cell(
        CellArray::new(
            vec![
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U64(vec![wide, u64::MAX]), vec![1, 2])
                        .unwrap(),
                ),
                Value::Tensor(
                    Tensor::new_integer(IntegerStorage::U64(vec![7]), vec![1, 1]).unwrap(),
                ),
            ],
            1,
            2,
        )
        .unwrap(),
    );
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let output = block_on(padsequences_builtin(
        sequences.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::Int(IntValue::U8(2))]),
    ))
    .unwrap();
    let Value::OutputList(outputs) = output else {
        panic!("two outputs expected")
    };
    let [Value::Tensor(data), Value::LogicalArray(mask)] = outputs.as_slice() else {
        panic!("typed data and logical mask expected")
    };
    assert_eq!(data.shape(), &[1, 2, 2]);
    assert_eq!(data.host_buffer().numeric_dtype(), NumericDType::U64);
    assert_eq!(
        data.numeric_value_at(0),
        Some(runmat_value::NumericScalar::U64(wide))
    );
    assert_eq!(
        data.numeric_value_at(1),
        Some(runmat_value::NumericScalar::U64(u64::MAX))
    );
    assert_eq!(
        data.numeric_value_at(2),
        Some(runmat_value::NumericScalar::U64(7))
    );
    assert_eq!(
        data.numeric_value_at(3),
        Some(runmat_value::NumericScalar::U64(0))
    );
    assert_eq!(mask.shape.as_slice(), data.shape());
    assert_eq!(mask.data.as_slice(), &[1, 1, 1, 0]);
    drop(_guard);

    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let output = block_on(padsequences_builtin(
        sequences,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Num(2.0),
            Value::String("UniformOutput".into()),
            Value::Bool(false),
        ]),
    ))
    .unwrap();
    let Value::OutputList(outputs) = output else {
        panic!("two cell outputs expected")
    };
    let [Value::Cell(data), Value::Cell(mask)] = outputs.as_slice() else {
        panic!("cell data and masks expected")
    };
    assert_eq!(data.shape, vec![1, 2]);
    assert_eq!(mask.shape, data.shape);
    let Value::Tensor(second) = &data.data()[1] else {
        panic!("typed sequence expected")
    };
    assert_eq!(
        second.numeric_value_at(0),
        Some(runmat_value::NumericScalar::U64(7))
    );
    let Value::LogicalArray(second_mask) = &mask.data()[1] else {
        panic!("logical sequence mask expected")
    };
    assert_eq!(second_mask.data.as_slice(), &[1, 0]);
}

#[test]
fn dlarray_preserves_data_and_format_labels() {
    let data = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
    let out = block_on(dlarray_builtin(
        data.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String("CB".into())]),
    ))
    .expect("dlarray");
    let Value::Object(object) = out else {
        panic!("expected object");
    };
    assert_eq!(object.class_name().display_name(), "dlarray");
    assert_eq!(object.properties().get("Data"), Some(&data));
    assert_eq!(
        object.properties().get("Format"),
        Some(&Value::String("CB".into()))
    );
}

#[test]
fn dlarray_rejects_every_exact_integer_storage_class() {
    let storages = [
        runmat_value::IntegerStorage::I8(vec![1]),
        runmat_value::IntegerStorage::I16(vec![1]),
        runmat_value::IntegerStorage::I32(vec![1]),
        runmat_value::IntegerStorage::I64(vec![1]),
        runmat_value::IntegerStorage::U8(vec![1]),
        runmat_value::IntegerStorage::U16(vec![1]),
        runmat_value::IntegerStorage::U32(vec![1]),
        runmat_value::IntegerStorage::U64(vec![1]),
    ];

    for storage in storages {
        let data = Value::Tensor(Tensor::new_integer(storage, vec![1, 1]).unwrap());
        let error = block_on(dlarray_builtin(
            data,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("integer dlarray error");
        assert!(error.message().contains("integer data is not supported"));
    }
}

#[test]
fn dlarray_preserves_gpu_array_residency() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload");
        let out = block_on(dlarray_builtin(
            Value::GpuTensor(handle.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "CB".into(),
            )]),
        ))
        .expect("dlarray gpu");
        let Value::Object(object) = out else {
            panic!("expected object");
        };
        assert_eq!(object.class_name().display_name(), "dlarray");
        let Value::GpuTensor(wrapped) = object.properties().get("Data").unwrap() else {
            panic!("expected gpu tensor data");
        };
        assert_eq!(wrapped.allocation_identity(), handle.allocation_identity());
        assert_eq!(wrapped.view_identity(), handle.view_identity());
        assert_eq!(wrapped.shape(), handle.shape());
        assert_eq!(handle_precision(wrapped), handle_precision(&handle));
        assert_eq!(handle_storage(wrapped), handle_storage(&handle));
        assert_eq!(
            object.properties().get("Format"),
            Some(&Value::String("CB".into()))
        );
    });
}

#[test]
fn dlarray_rejects_non_numeric_payloads_and_format_without_download() {
    for data in [
        Value::String("not numeric".into()),
        Value::Cell(CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap()),
        object("Example", vec![("Data", Value::Num(1.0))]).expect("fixture object"),
    ] {
        let error = block_on(dlarray_builtin(
            data,
            crate::builtins::common::test_support::builtin_arguments(vec![]),
        ))
        .expect_err("invalid Data storage");
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
    }
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let values = [1.0, 2.0];
        let shape = [1usize, 2usize];
        let resident = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &shape,
            })
            .unwrap();
        provider.reset_telemetry();
        let error = block_on(dlarray_builtin(
            Value::GpuTensor(resident.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::Num(12.0)]),
        ))
        .expect_err("numeric Format rejects before any Data access");
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let wrapped = block_on(dlarray_builtin(
            Value::GpuTensor(resident.clone()),
            crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
                "CB".into(),
            )]),
        ))
        .unwrap();
        let Value::Object(wrapped) = wrapped else {
            panic!("expected dlarray")
        };
        let Value::GpuTensor(data) = &wrapped.properties()[runmat_builtins::DLARRAY_DATA_PROPERTY]
        else {
            panic!("expected resident Data")
        };
        assert_eq!(data.allocation_identity(), resident.allocation_identity());
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn fixed_output_deep_learning_bindings_admit_discard_and_reject_extra_outputs() {
    for (name, args) in [
        ("fullyConnectedLayer", vec![Value::Num(2.0)]),
        ("layerGraph", vec![]),
        (
            "analyzeNetwork",
            vec![Value::Cell(CellArray::new(vec![], 0, 0).unwrap())],
        ),
        ("combvec", vec![Value::Num(1.0)]),
        ("dlarray", vec![Value::Num(1.0)]),
    ] {
        block_on(crate::call_builtin_async_with_outputs(name, &args, 0))
            .unwrap_or_else(|error| panic!("{name} discard invocation failed: {error}"));
        let error = block_on(crate::call_builtin_async_with_outputs(name, &args, 2))
            .expect_err("fixed-output binding must reject a second output");
        assert_eq!(
            error.identifier(),
            Some("RunMat:deepLearning:InvalidInput"),
            "{name}"
        );
    }
}

#[test]
fn fixed_output_rejection_precedes_resident_input_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let resident = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[1, 2],
            })
            .expect("resident array");
        for (name, args) in [
            ("combvec", vec![Value::GpuTensor(resident.clone())]),
            ("dlarray", vec![Value::GpuTensor(resident.clone())]),
        ] {
            provider.reset_telemetry();
            let error = block_on(crate::call_builtin_async_with_outputs(name, &args, 2))
                .expect_err("second output must reject before provider work");
            assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0, "{name}");
        }
    });
}

#[test]
fn training_options_store_exact_controls_and_resident_metadata_without_gather() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let resident = provider
            .upload(&HostTensorView {
                data: &[1.0, 2.0],
                shape: &[1, 2],
            })
            .expect("resident option value");
        let wide = (1_u64 << 53) + 1;
        provider.reset_telemetry();
        let value = block_on(training_options_builtin(
            Value::String("adam".into()),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("MaxEpochs".into()),
                Value::Int(IntValue::U64(wide)),
                Value::String("ValidationData".into()),
                Value::GpuTensor(resident.clone()),
            ]),
        ))
        .expect("training metadata");
        let Value::Object(options) = value else {
            panic!("expected options object")
        };
        assert_eq!(
            options.properties()["MaxEpochs"],
            Value::Int(IntValue::U64(wide))
        );
        let Value::GpuTensor(retained) = &options.properties()["ValidationData"] else {
            panic!("expected resident metadata")
        };
        assert_eq!(
            retained.allocation_identity(),
            resident.allocation_identity()
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let error = block_on(crate::call_builtin_async_with_outputs(
            "trainingOptions",
            &[
                Value::String("adam".into()),
                Value::String("ValidationData".into()),
                Value::GpuTensor(resident),
            ],
            2,
        ))
        .expect_err("fixed result count");
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn discarded_training_options_still_validate_solver_and_pairs() {
    let error = block_on(crate::call_builtin_async_with_outputs(
        "trainingOptions",
        &[Value::String("invalid".into())],
        0,
    ))
    .expect_err("discarded result still validates solver");
    assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
    let error = block_on(crate::call_builtin_async_with_outputs(
        "trainingOptions",
        &[
            Value::String("adam".into()),
            Value::String("MaxEpochs".into()),
        ],
        0,
    ))
    .expect_err("discarded result still validates complete pairs");
    assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
}

#[test]
fn crossentropy_computes_mean_loss() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = block_on(crossentropy_builtin(
        None,
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect("crossentropy");
    let Value::Num(loss) = out else {
        panic!("expected scalar");
    };
    assert!((loss - 0.11157177565710488).abs() < 1.0e-12);
}

#[test]
fn crossentropy_gpu_inputs_use_provider_for_scalar_reduction() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
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
            None,
            Value::GpuTensor(predictions),
            Value::GpuTensor(targets),
            crate::builtins::common::test_support::builtin_arguments(vec![]),
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
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
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
            None,
            Value::GpuTensor(predictions),
            Value::GpuTensor(targets),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("ClassificationMode".into()),
                Value::String("multi-label".into()),
                Value::String("Reduction".into()),
                Value::String("none".into()),
            ]),
        ))
        .expect("crossentropy gpu unreduced");
        assert!(matches!(out, Value::GpuTensor(_)));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let gathered = crate::builtins::common::test_support::gather(out).expect("gather losses");
        assert_eq!(gathered.shape(), shape);
        assert!((gathered.materialize_f64()[0] + 0.8_f64.ln()).abs() < 1.0e-12);
        assert!((gathered.materialize_f64()[1] + 0.8_f64.ln()).abs() < 1.0e-12);
    });
}

#[test]
fn crossentropy_gpu_supports_host_weights_without_gathering_predictions() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
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
            None,
            Value::GpuTensor(predictions),
            Value::GpuTensor(targets),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![2, 1]).unwrap()),
                Value::String("DataFormat".into()),
                Value::String("CB".into()),
            ]),
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
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
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
            None,
            Value::GpuTensor(predictions),
            Value::GpuTensor(targets),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::String("Mask".into()),
                Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap()),
                Value::String("ClassificationMode".into()),
                Value::String("multi-label".into()),
                Value::String("Reduction".into()),
                Value::String("none".into()),
            ]),
        ))
        .expect("masked gpu crossentropy");
        assert!(matches!(out, Value::GpuTensor(_)));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let gathered = crate::builtins::common::test_support::gather(out).expect("gather losses");
        assert_eq!(gathered.shape(), shape);
        assert!((gathered.materialize_f64()[0] + 0.8_f64.ln()).abs() < 1.0e-12);
        assert_eq!(gathered.materialize_f64()[1], 0.0);
    });
}

#[test]
fn crossentropy_gpu_supports_resident_weights_and_mask_without_downloads() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
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
            None,
            Value::GpuTensor(predictions),
            Value::GpuTensor(targets),
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(weights),
                Value::String("Mask".into()),
                Value::GpuTensor(mask),
                Value::String("ClassificationMode".into()),
                Value::String("multi-label".into()),
            ]),
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
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = block_on(crossentropy_builtin(
        None,
        Value::Tensor(Tensor::new(vec![0.8, 0.2, 0.25, 0.75], vec![2, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![2, 1]).unwrap()),
            Value::String("DataFormat".into()),
            Value::String("CB".into()),
        ]),
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
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = block_on(crossentropy_builtin(
        None,
        Value::Tensor(Tensor::new(vec![0.8, 0.2, 0.25, 0.75], vec![2, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap()),
            Value::String("DataFormat".into()),
            Value::String("CB".into()),
            Value::String("WeightsFormat".into()),
            Value::String("UC".into()),
        ]),
    ))
    .expect("formatted class weights");
    let Value::Num(loss) = out else {
        panic!("expected scalar");
    };
    let expected = (-2.0_f64 * 0.8_f64.ln() - 0.75_f64.ln()) / 2.0;
    assert!((loss - expected).abs() < 1.0e-12);

    let out = block_on(crossentropy_builtin(
        None,
        Value::Tensor(Tensor::new(vec![0.8, 0.2, 0.25, 0.75], vec![2, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![0.5, 2.0], vec![1, 2]).unwrap()),
            Value::String("DataFormat".into()),
            Value::String("CB".into()),
            Value::String("WeightsFormat".into()),
            Value::String("UB".into()),
        ]),
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
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = block_on(crossentropy_builtin(
        None,
        Value::Tensor(Tensor::new(vec![0.8, 0.2, 0.25, 0.75], vec![2, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![2.0, 3.0, 5.0, 7.0], vec![2, 2]).unwrap()),
            Value::String("DataFormat".into()),
            Value::String("CB".into()),
            Value::String("WeightsFormat".into()),
            Value::String("BC".into()),
        ]),
    ))
    .expect("formatted full-size weights");
    let Value::Num(loss) = out else {
        panic!("expected scalar");
    };
    let expected = (-2.0_f64 * 0.8_f64.ln() - 7.0_f64 * 0.75_f64.ln()) / 2.0;
    assert!((loss - expected).abs() < 1.0e-12);

    let err = block_on(crossentropy_builtin(
        None,
        Value::Tensor(Tensor::new(vec![0.8, 0.2, 0.25, 0.75], vec![2, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Weights".into()),
            Value::Tensor(Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]).unwrap()),
            Value::String("DataFormat".into()),
            Value::String("CB".into()),
            Value::String("WeightsFormat".into()),
            Value::String("TC".into()),
        ]),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("not present in DataFormat"));
}

#[test]
fn crossentropy_supports_multilabel_reduction_none_and_mask_normalization() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let unreduced = block_on(crossentropy_builtin(
        None,
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("ClassificationMode".into()),
            Value::String("multi-label".into()),
            Value::String("Reduction".into()),
            Value::String("none".into()),
        ]),
    ))
    .expect("multilabel unreduced crossentropy");
    let Value::Tensor(tensor) = unreduced else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.shape(), &[1, 2]);
    assert!((tensor.materialize_f64()[0] + 0.8_f64.ln()).abs() < 1.0e-12);
    assert!((tensor.materialize_f64()[1] + 0.8_f64.ln()).abs() < 1.0e-12);

    let masked = block_on(crossentropy_builtin(
        None,
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Mask".into()),
            Value::LogicalArray(LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap()),
            Value::String("NormalizationFactor".into()),
            Value::String("mask-included".into()),
        ]),
    ))
    .expect("masked crossentropy");
    let Value::Num(loss) = masked else {
        panic!("expected scalar");
    };
    assert!((loss + 0.8_f64.ln()).abs() < 1.0e-12);
}

#[test]
fn crossentropy_unreduced_output_preserves_native_single_storage() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = block_on(crossentropy_builtin(
        None,
        Value::Tensor(Tensor::from_f32(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::from_f32(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("ClassificationMode".into()),
            Value::String("multi-label".into()),
            Value::String("Reduction".into()),
            Value::String("none".into()),
        ]),
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
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let masked = block_on(crossentropy_builtin(
        None,
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Mask".into()),
            poisoned_int_tensor(IntegerStorage::U8(vec![1, 0]), vec![1, 2], 2.0),
            Value::String("NormalizationFactor".into()),
            Value::String("mask-included".into()),
        ]),
    ))
    .expect("masked crossentropy");
    let Value::Num(loss) = masked else {
        panic!("expected scalar");
    };
    assert!((loss + 0.8_f64.ln()).abs() < 1.0e-12);
}

#[test]
fn crossentropy_payloads_read_typed_integer_storage_as_double_loss_inputs() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = block_on(crossentropy_builtin(
        None,
        poisoned_int_tensor(IntegerStorage::U8(vec![1, 0]), vec![1, 2], 0.5),
        poisoned_int_tensor(IntegerStorage::U8(vec![1, 0]), vec![1, 2], 0.5),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Reduction".into()),
            Value::String("none".into()),
        ]),
    ))
    .expect("integer payload crossentropy");
    let Value::Tensor(losses) = out else {
        panic!("expected tensor");
    };
    assert_eq!(losses.numeric_dtype(), NumericDType::F64);
    assert!(losses.integer_storage().is_none());
    assert!(losses.materialize_f64()[0].abs() < 1.0e-9);
    assert_eq!(losses.materialize_f64()[1], 0.0);
}

#[test]
fn crossentropy_rejects_incompatible_shapes_masks_and_formats() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let err = block_on(crossentropy_builtin(
        None,
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("prediction shape"));

    let err = block_on(crossentropy_builtin(
        None,
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Mask".into()),
            Value::Bool(true),
        ]),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("same size"));

    let err = block_on(crossentropy_builtin(
        None,
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Mask".into()),
            Value::Tensor(Tensor::new(vec![0.5, 1.0], vec![1, 2]).unwrap()),
        ]),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("binary"));

    let err = block_on(crossentropy_builtin(
        None,
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![1, 2]).unwrap()),
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![1, 2]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("DataFormat".into()),
            Value::String("CC".into()),
        ]),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("cannot repeat"));
}

#[test]
fn crossentropy_preserves_dlarray_output_for_dlarray_inputs() {
    let predictions = block_on(dlarray_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![2, 1]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String("CB".into())]),
    ))
    .expect("prediction dlarray");
    let targets = block_on(dlarray_builtin(
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String("CB".into())]),
    ))
    .expect("target dlarray");
    let out = block_on(crossentropy_builtin(
        None,
        predictions,
        targets,
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .expect("dlarray crossentropy");
    let Value::Object(object) = out else {
        panic!("expected dlarray object");
    };
    assert_eq!(object.class_name().display_name(), "dlarray");
    let Value::Num(loss) = object.properties().get("Data").unwrap() else {
        panic!("expected scalar data");
    };
    assert!((*loss + 0.8_f64.ln()).abs() < 1.0e-12);

    let predictions = block_on(dlarray_builtin(
        Value::Tensor(Tensor::new(vec![0.8, 0.2], vec![2, 1]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String("CB".into())]),
    ))
    .expect("prediction dlarray");
    let targets = block_on(dlarray_builtin(
        Value::Tensor(Tensor::new(vec![1.0, 0.0], vec![2, 1]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String("CB".into())]),
    ))
    .expect("target dlarray");
    let out = block_on(crossentropy_builtin(
        None,
        predictions,
        targets,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Reduction".into()),
            Value::String("none".into()),
        ]),
    ))
    .expect("unreduced dlarray crossentropy");
    let Value::Object(object) = out else {
        panic!("expected dlarray object");
    };
    assert_eq!(
        object.properties().get("Format"),
        Some(&Value::String(String::new().into()))
    );
}

#[test]
fn adamupdate_computes_bias_corrected_tensor_step() {
    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let out = block_on(adamupdate_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.1, -0.2], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap()),
            Value::Num(1.0),
        ]),
    ))
    .expect("adamupdate");
    let Value::OutputList(outputs) = out else {
        panic!("expected output list");
    };
    assert_eq!(outputs.len(), 3);
    let Value::Tensor(parameters) = &outputs[0] else {
        panic!("expected tensor parameters");
    };
    assert!((parameters.materialize_f64()[0] - 0.9990000001).abs() < 1.0e-9);
    assert!((parameters.materialize_f64()[1] - 2.00099999995).abs() < 1.0e-9);
    let Value::Tensor(average_grad) = &outputs[1] else {
        panic!("expected tensor averageGrad");
    };
    assert!((average_grad.materialize_f64()[0] - 0.01).abs() < 1.0e-12);
    assert!((average_grad.materialize_f64()[1] + 0.02).abs() < 1.0e-12);
    let Value::Tensor(average_sq_grad) = &outputs[2] else {
        panic!("expected tensor averageSqGrad");
    };
    assert!((average_sq_grad.materialize_f64()[0] - 0.00001).abs() < 1.0e-12);
    assert!((average_sq_grad.materialize_f64()[1] - 0.00004).abs() < 1.0e-12);
}

#[test]
fn adamupdate_default_call_returns_updated_parameters_directly() {
    let out = block_on(adamupdate_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.1], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
            Value::Num(1.0),
        ]),
    ))
    .expect("adamupdate");
    let Value::Tensor(parameters) = out else {
        panic!("expected direct tensor output");
    };
    assert!((parameters.materialize_f64()[0] - 0.9990000001).abs() < 1.0e-9);
}

#[test]
fn adamupdate_preserves_dlarray_wrapper_for_state_outputs() {
    let params = block_on(dlarray_builtin(
        Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String("CB".into())]),
    ))
    .unwrap();
    let _guard = crate::output_count::push_output_count(Some(3)).expect("output-count scope");
    let out = block_on(adamupdate_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            params,
            Value::Tensor(Tensor::new(vec![0.5], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
            Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
            Value::Num(1.0),
        ]),
    ))
    .expect("adamupdate");
    let Value::OutputList(outputs) = out else {
        panic!("expected output list");
    };
    let Value::Object(updated) = &outputs[0] else {
        panic!("expected dlarray object");
    };
    assert_eq!(updated.class_name().display_name(), "dlarray");
    assert_eq!(
        updated.properties().get("Format"),
        Some(&Value::String("CB".into()))
    );
    for value in &outputs[1..] {
        let Value::Object(state) = value else {
            panic!("expected dlarray state object");
        };
        assert_eq!(state.class_name().display_name(), "dlarray");
        assert_eq!(
            state.properties().get("Format"),
            Some(&Value::String("CB".into()))
        );
    }
}

#[test]
fn adamupdate_rejects_invalid_shapes_outputs_and_hyperparameters() {
    let err = block_on(adamupdate_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.1], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap()),
            Value::Num(1.0),
        ]),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("grad must match"));

    let err = block_on(adamupdate_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.1], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
            Value::Num(1.0),
            Value::Num(-0.001),
        ]),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("learnRate must be positive"));

    let _guard = crate::output_count::push_output_count(Some(4)).expect("output-count scope");
    let err = block_on(adamupdate_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.1], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
            Value::Num(1.0),
        ]),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("at most 3"));
}

#[test]
fn adamupdate_accepts_zero_decay_and_rejects_integer_tensor_state() {
    let out = block_on(adamupdate_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.1], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
            Value::Num(10_000_000_000.0),
            Value::Num(0.001),
            Value::Num(0.0),
            Value::Num(0.0),
        ]),
    ))
    .expect("zero decay and large iteration");
    let Value::Tensor(parameters) = out else {
        panic!("expected tensor parameters");
    };
    assert!(parameters.materialize_f64()[0].is_finite());

    let integer_params = Tensor::new_with_dtype(vec![1.0], vec![1, 1], NumericDType::U8).unwrap();
    let err = block_on(adamupdate_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(integer_params),
            Value::Tensor(Tensor::new(vec![0.1], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
            Value::Num(1.0),
        ]),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("double or single"));
}

#[test]
fn adamupdate_hyperparameters_read_typed_integer_storage_exactly() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = block_on(adamupdate_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.1], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
            Value::Tensor(Tensor::new(vec![0.0], vec![1, 1]).unwrap()),
            cleared_int_tensor(IntegerStorage::U16(vec![1]), vec![1, 1]),
            cleared_int_tensor(IntegerStorage::U16(vec![1]), vec![1, 1]),
            cleared_int_tensor(IntegerStorage::U8(vec![0]), vec![1, 1]),
            cleared_int_tensor(IntegerStorage::U8(vec![0]), vec![1, 1]),
        ]),
    ))
    .expect("adamupdate");
    let Value::Tensor(parameters) = out else {
        panic!("expected tensor parameters");
    };
    assert!((parameters.materialize_f64()[0] - 0.0000001).abs() < 1.0e-9);
}

#[test]
fn adamupdate_known_host_errors_precede_mixed_residency_download() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let values = [1.0, 2.0];
        let shape = [1usize, 2usize];
        let resident = provider
            .upload(&HostTensorView {
                data: &values,
                shape: &shape,
            })
            .expect("resident parameters");
        provider.reset_telemetry();
        let err = block_on(adamupdate_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(resident.clone()),
                Value::Tensor(Tensor::new(vec![0.1], vec![1, 1]).unwrap()),
                Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                Value::Num(1.0),
            ]),
        ))
        .expect_err("known shape mismatch");
        assert_eq!(err.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let err = block_on(adamupdate_builtin(
            None,
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::GpuTensor(resident),
                Value::Tensor(Tensor::new(vec![0.1, 0.2], vec![1, 2]).unwrap()),
                Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).unwrap()),
                Value::Num(1.0),
                Value::Num(-1.0),
            ]),
        ))
        .expect_err("invalid later host control");
        assert_eq!(err.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn adamupdate_typed_iteration_is_an_explicit_runmat_extension() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let args = vec![
        Value::Num(2.0),
        Value::Num(1.0),
        Value::Num(0.0),
        Value::Num(0.0),
        Value::Int(IntValue::U64(9_007_199_254_740_993)),
    ];
    let out = block_on(adamupdate_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(args.clone()),
    ))
    .expect("exact wide iteration");
    assert!(matches!(out, Value::Num(value) if value.is_finite()));
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = block_on(adamupdate_builtin(
        None,
        crate::builtins::common::test_support::builtin_arguments(args),
    ))
    .expect_err("typed integer extension");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:AdamupdateIntegerScalarExtension")
    );
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
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let out = block_on(dlfeval_builtin(vec![
        Value::FunctionHandle("deal".into()),
        Value::String("a".into()),
        Value::String("b".into()),
    ]))
    .expect("dlfeval deal");
    let Value::OutputList(outputs) = out.value() else {
        panic!("expected output list");
    };
    assert_eq!(
        outputs.as_slice(),
        &[Value::String("a".into()), Value::String("b".into())]
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
    let Value::OutputList(outputs) = out.value() else {
        panic!("expected output list");
    };
    assert_eq!(
        outputs.as_slice(),
        &[Value::String("left".into()), Value::String("right".into())]
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
fn dlfeval_invalid_handle_preflights_before_resident_argument_access() {
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let resident = provider
            .upload(&HostTensorView {
                data: &[1.0],
                shape: &[1, 1],
            })
            .expect("resident callback argument");
        provider.reset_telemetry();
        let error = block_on(dlfeval_builtin(vec![
            Value::String("plus".into()),
            Value::GpuTensor(resident),
        ]))
        .expect_err("text is not a function handle");
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn dlfeval_owned_resident_argument_survives_callback_and_unpolled_drop() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let resident = provider
            .upload(&HostTensorView {
                data: &[3.0, 4.0],
                shape: &[1, 2],
            })
            .unwrap();
        provider.reset_telemetry();
        let output = block_on(dlfeval_builtin(vec![
            Value::FunctionHandle("deal".into()),
            Value::GpuTensor(resident.clone()),
        ]))
        .unwrap();
        let Value::GpuTensor(returned) = output.value() else {
            panic!("callback must retain the resident value")
        };
        assert_eq!(
            returned.allocation_identity(),
            resident.allocation_identity()
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let unpolled = dlfeval_builtin(vec![
            Value::FunctionHandle("deal".into()),
            Value::GpuTensor(resident.clone()),
        ]);
        drop(unpolled);
        assert!(resident.allocation_lease().validate_live().is_ok());
        assert!(returned.allocation_lease().validate_live().is_ok());
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
    });
}

#[test]
fn dlupdate_maps_matching_struct_and_cell_trees() {
    let mut left = runmat_value::UnboundedStructValueBuilder::new();
    left.insert(
        "Weights",
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
    );
    left.insert(
        "Bias",
        Value::Cell(CellArray::new(vec![Value::Num(3.0)], 1, 1).unwrap()),
    );
    let mut right = runmat_value::UnboundedStructValueBuilder::new();
    right.insert(
        "Weights",
        Value::Tensor(Tensor::new(vec![10.0, 20.0], vec![1, 2]).unwrap()),
    );
    right.insert(
        "Bias",
        Value::Cell(CellArray::new(vec![Value::Num(4.0)], 1, 1).unwrap()),
    );

    let out = block_on(dlupdate_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::FunctionHandle("plus".into()),
            Value::Struct(left.finish()),
            Value::Struct(right.finish()),
        ]),
    ))
    .expect("dlupdate plus");
    let Value::Struct(out) = out.value() else {
        panic!("expected struct output");
    };
    let Value::Tensor(weights) = out.fields.get("Weights").unwrap() else {
        panic!("expected tensor weights");
    };
    assert_eq!(weights.materialize_f64(), vec![11.0, 22.0]);
    let Value::Cell(bias) = out.fields.get("Bias").unwrap() else {
        panic!("expected cell bias");
    };
    assert_eq!(bias.data(), vec![Value::Num(7.0)]);
}

#[test]
fn dlupdate_borrowed_resident_tree_survives_callback_and_preflight_rejection() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    crate::builtins::common::test_support::with_test_provider(|provider| {
        let resident = provider
            .upload(&HostTensorView {
                data: &[5.0, 6.0],
                shape: &[1, 2],
            })
            .unwrap();
        let tree =
            || Value::Cell(CellArray::new(vec![Value::GpuTensor(resident.clone())], 1, 1).unwrap());
        provider.reset_telemetry();
        let output = block_on(dlupdate_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::FunctionHandle("deal".into()),
                tree(),
            ]),
        ))
        .unwrap();
        let Value::Cell(output) = output.value() else {
            panic!("callback must reconstruct the cell tree")
        };
        let Value::GpuTensor(returned) = &output.data()[0] else {
            panic!("resident leaf must remain resident")
        };
        assert_eq!(
            returned.allocation_identity(),
            resident.allocation_identity()
        );
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);

        let mismatch =
            Value::Cell(CellArray::new(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap());
        let error = block_on(dlupdate_builtin(
            crate::builtins::common::test_support::builtin_arguments(vec![
                Value::FunctionHandle("deal".into()),
                tree(),
                mismatch,
            ]),
        ))
        .expect_err("preflight must reject unlike trees before callbacks");
        assert_eq!(error.identifier(), Some("RunMat:deepLearning:InvalidInput"));
        assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
        assert!(resident.allocation_lease().validate_live().is_ok());
    });
}

#[test]
fn dlupdate_preserves_typed_struct_array_shape_and_schema() {
    let element = |value| {
        let mut structure = runmat_value::UnboundedStructValueBuilder::new();
        structure.insert("Weight", Value::Num(value));
        structure.finish()
    };
    let left = StructArray::new(vec![element(1.0), element(2.0)], vec![2, 1]).unwrap();
    let right = StructArray::new(vec![element(10.0), element(20.0)], vec![2, 1]).unwrap();
    let output = block_on(dlupdate_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::FunctionHandle("plus".into()),
            Value::StructArray(left),
            Value::StructArray(right),
        ]),
    ))
    .expect("dlupdate structure array");
    let Value::StructArray(output) = output.value() else {
        panic!("expected typed structure array");
    };
    assert_eq!(output.shape(), &[2, 1]);
    assert_eq!(output.field_names().collect::<Vec<_>>(), ["Weight"]);
    assert_eq!(
        output.field_values("Weight").unwrap(),
        [Value::Num(11.0), Value::Num(22.0)]
    );
}

#[test]
fn dlupdate_reconstructs_multiple_output_trees() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let first = Value::Cell(CellArray::new(vec![Value::Num(1.0), Value::Num(2.0)], 1, 2).unwrap());
    let second =
        Value::Cell(CellArray::new(vec![Value::Num(10.0), Value::Num(20.0)], 1, 2).unwrap());

    let out = block_on(dlupdate_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::FunctionHandle("deal".into()),
            first.clone(),
            second.clone(),
        ]),
    ))
    .expect("dlupdate deal");
    let Value::OutputList(outputs) = out.value() else {
        panic!("expected output list");
    };
    assert_eq!(outputs.as_slice(), &[first, second]);
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

    let out = block_on(dlupdate_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::FunctionHandle("plus".into()),
            left,
            right,
        ]),
    ))
    .expect("dlupdate table");
    let Value::Object(object) = out.value() else {
        panic!("expected table object");
    };
    let vars = crate::builtins::table::table_variables(object).expect("table variables");
    assert_eq!(
        vars.fields.get("Layer"),
        Some(&Value::StringArray(
            StringArray::new(vec!["fc".into(), "fc".into()], vec![2, 1]).unwrap()
        ))
    );
    let Value::Cell(values) = vars.fields.get("Value").unwrap() else {
        panic!("expected Value cell");
    };
    let Value::Tensor(weights) = &values.data()[0] else {
        panic!("expected weights tensor");
    };
    assert_eq!(weights.materialize_f64(), vec![11.0, 22.0]);
    assert_eq!(values.data()[1], Value::Num(7.0));
}

#[test]
fn dlupdate_treats_dlarray_objects_as_leaves() {
    let data = Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
    let dlarray = block_on(dlarray_builtin(
        data.clone(),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String("CB".into())]),
    ))
    .expect("dlarray");

    let out = block_on(dlupdate_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::FunctionHandle("class".into()),
            dlarray.clone(),
        ]),
    ))
    .expect("dlupdate dlarray");
    assert_eq!(out.value(), &Value::String("dlarray".into()));
}

#[test]
fn dlupdate_rejects_non_callable_and_mismatched_trees() {
    let err = block_on(dlupdate_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("@plus".into()),
            Value::Num(1.0),
        ]),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("expected a function handle"));

    let err = block_on(dlupdate_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::FunctionHandle("plus".into()),
            Value::Cell(CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap()),
            Value::Cell(CellArray::new(vec![Value::Num(2.0), Value::Num(3.0)], 1, 2).unwrap()),
        ]),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("same shape"));

    let generic_table = crate::builtins::table::table_from_columns(
        vec!["A".into(), "B".into()],
        vec![Value::Num(1.0), Value::Num(2.0)],
    )
    .expect("generic table");
    let err = block_on(dlupdate_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::FunctionHandle("plus".into()),
            generic_table,
        ]),
    ))
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
    let err = block_on(dlupdate_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::FunctionHandle("plus".into()),
            malformed_learnables,
        ]),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("Value variable must be a cell"));
}

#[test]
fn dlupdate_discarded_result_still_invokes_callback_and_retains_failure_source() {
    use std::error::Error;

    let _guard = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let err = block_on(dlupdate_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::FunctionHandle("__missing_dlupdate_callback".into()),
            Value::Num(1.0),
        ]),
    ))
    .expect_err("discarded callback call must still execute");
    assert_eq!(err.identifier(), Some("RunMat:deepLearning:InvalidInput"));
    assert!(err.source().is_some());
}

#[test]
fn dlarray_and_crossentropy_reject_unsupported_forms() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let err = block_on(dlarray_builtin(
        Value::Num(1.0),
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("CB".into()),
            Value::String("Extra".into()),
        ]),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("optional dimension-label"));

    let err = block_on(crossentropy_builtin(
        None,
        Value::Tensor(Tensor::new(Vec::new(), vec![1, 0]).unwrap()),
        Value::Tensor(Tensor::new(Vec::new(), vec![1, 0]).unwrap()),
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("must not be empty"));
}

#[test]
fn unsupported_training_apis_return_clear_errors() {
    let err = block_on(train_network_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("expected X, Y, layers"));
    let err = block_on(export_onnx_network_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap_err();
    assert!(err.to_string().contains("ONNX"));
}
