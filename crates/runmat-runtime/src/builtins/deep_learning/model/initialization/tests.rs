use super::*;
use futures::executor::block_on;
use runmat_builtins::DeepLearningLayerKind as K;
use std::sync::atomic::{AtomicUsize, Ordering};

static CALLBACK_INVOCATIONS: AtomicUsize = AtomicUsize::new(0);

#[runmat_macros::runtime_builtin(
    name = "__deep_learning_initializer_once",
    category = "deep_learning/tests",
    summary = "Test helper for one-shot network initialization.",
    keywords = "test,dlnetwork,initializer",
    type_resolver(crate::builtins::deep_learning::any_type),
    descriptor(runmat_builtins::DLFEVAL_DESCRIPTOR),
    builtin_path = "crate::builtins::deep_learning::model::initialization::tests"
)]
async fn initializer_once(shape: Value) -> BuiltinResult<Value> {
    CALLBACK_INVOCATIONS.fetch_add(1, Ordering::SeqCst);
    let Value::Tensor(shape) = shape else {
        panic!("shape vector")
    };
    let dimensions: Vec<usize> = shape
        .host_buffer()
        .materialize_f64()
        .into_iter()
        .map(|x| x as usize)
        .collect();
    let count = dimensions.iter().product();
    Tensor::from_f32(vec![0.25; count], dimensions)
        .map(Value::Tensor)
        .map_err(|error| deep_learning_error("test", error))
}

fn layer(kind: K, name: &str, properties: &[(&str, Value)]) -> Value {
    let mut object = ObjectInstance::new(kind.class().owned());
    object
        .properties
        .insert("Name".into(), Value::String(name.into()));
    for (name, value) in properties {
        object.properties.insert((*name).into(), value.clone());
    }
    Value::Object(object)
}

fn property<'a>(layer: &'a Value, name: &str) -> &'a Value {
    let Value::Object(layer) = layer else {
        panic!("layer object")
    };
    layer.properties.get(name).expect("initialized property")
}

fn shape(layer: &Value, name: &str) -> Vec<usize> {
    let Value::Tensor(value) = property(layer, name) else {
        panic!("tensor property")
    };
    value.shape().to_vec()
}

#[test]
fn initializes_all_admitted_parameter_roles_from_canonical_geometry() {
    let mut feature = vec![
        layer(K::FeatureInput, "x", &[("InputSize", Value::Num(3.0))]),
        layer(
            K::FullyConnected,
            "dense",
            &[("OutputSize", Value::Num(2.0))],
        ),
        layer(K::LayerNormalization, "norm", &[]),
    ];
    block_on(initialise_layers(&mut feature, "test")).unwrap();
    assert_eq!(shape(&feature[1], "Weights"), [2, 3]);
    assert_eq!(shape(&feature[1], "Bias"), [2, 1]);
    assert_eq!(shape(&feature[2], "Scale"), [2, 1]);
    assert_eq!(shape(&feature[2], "Offset"), [2, 1]);
    let Value::Tensor(scale) = property(&feature[2], "Scale") else {
        unreachable!()
    };
    assert_eq!(scale.host_buffer().materialize_f64(), [1.0, 1.0]);

    let mut conv = vec![
        layer(
            K::SequenceInput,
            "sequence",
            &[("InputSize", Value::Num(3.0))],
        ),
        layer(
            K::Convolution1D,
            "conv",
            &[
                ("FilterSize", Value::Num(4.0)),
                ("NumFilters", Value::Num(2.0)),
            ],
        ),
    ];
    block_on(initialise_layers(&mut conv, "test")).unwrap();
    assert_eq!(shape(&conv[1], "Weights"), [4, 3, 2]);
    assert_eq!(shape(&conv[1], "Bias"), [1, 2]);

    for (kind, rows, channels) in [(K::Lstm, 8, 2), (K::Bilstm, 16, 4)] {
        let mut recurrent = vec![
            layer(
                K::SequenceInput,
                "sequence",
                &[("InputSize", Value::Num(3.0))],
            ),
            layer(kind, "memory", &[("NumHiddenUnits", Value::Num(2.0))]),
        ];
        block_on(initialise_layers(&mut recurrent, "test")).unwrap();
        assert_eq!(shape(&recurrent[1], "InputWeights"), [rows, 3]);
        assert_eq!(shape(&recurrent[1], "RecurrentWeights"), [rows, 2]);
        assert_eq!(shape(&recurrent[1], "Bias"), [rows, 1]);
        let Value::Tensor(bias) = property(&recurrent[1], "Bias") else {
            unreachable!()
        };
        let bias = bias.host_buffer().materialize_f64();
        assert_eq!(&bias[2..4], &[1.0, 1.0]);
        if kind == K::Bilstm {
            assert_eq!(&bias[10..12], &[1.0, 1.0]);
        }
        let state = super::super::state::initial_with_defaults(&recurrent, true, "test").unwrap();
        let Value::Object(table) = state else {
            panic!("State table")
        };
        let values = super::super::table_rows::values(&table, "State", "test").unwrap();
        assert_eq!(values.data.len(), 2);
        for value in &values.data {
            let Value::Tensor(state) = value else {
                panic!("State tensor")
            };
            assert_eq!(state.shape(), &[channels, 1]);
            assert!(state
                .host_buffer()
                .materialize_f64()
                .iter()
                .all(|x| *x == 0.0));
        }
    }
}

#[test]
fn invalid_later_initializer_rejects_before_any_layer_mutation() {
    let mut layers = vec![
        layer(K::FeatureInput, "x", &[("InputSize", Value::Num(2.0))]),
        layer(
            K::FullyConnected,
            "first",
            &[
                ("OutputSize", Value::Num(3.0)),
                (
                    "WeightsInitializer",
                    Value::FunctionHandle("__deep_learning_initializer_once".into()),
                ),
            ],
        ),
        layer(
            K::FullyConnected,
            "last",
            &[
                ("OutputSize", Value::Num(2.0)),
                ("WeightsInitializer", Value::String("bogus".into())),
            ],
        ),
    ];
    let before = layers.clone();
    CALLBACK_INVOCATIONS.store(0, Ordering::SeqCst);
    let error = block_on(initialise_layers(&mut layers, "test")).unwrap_err();
    assert!(error.to_string().contains("unsupported WeightsInitializer"));
    assert_eq!(layers, before);
    assert_eq!(CALLBACK_INVOCATIONS.load(Ordering::SeqCst), 0);
}

#[test]
fn callback_initializer_is_invoked_once_and_published_atomically() {
    let mut layers = vec![
        layer(K::FeatureInput, "x", &[("InputSize", Value::Num(2.0))]),
        layer(
            K::FullyConnected,
            "dense",
            &[
                ("OutputSize", Value::Num(3.0)),
                (
                    "WeightsInitializer",
                    Value::FunctionHandle("__deep_learning_initializer_once".into()),
                ),
            ],
        ),
    ];
    CALLBACK_INVOCATIONS.store(0, Ordering::SeqCst);
    block_on(initialise_layers(&mut layers, "test")).unwrap();
    assert_eq!(CALLBACK_INVOCATIONS.load(Ordering::SeqCst), 1);
    let Value::Tensor(weights) = property(&layers[1], "Weights") else {
        unreachable!()
    };
    assert_eq!(weights.shape(), &[3, 2]);
    assert_eq!(weights.host_buffer().materialize_f64(), vec![0.25; 6]);
    assert_eq!(weights.numeric_dtype(), NumericDType::F32);
    assert_eq!(shape(&layers[1], "Bias"), [3, 1]);
    let Value::Tensor(bias) = property(&layers[1], "Bias") else {
        unreachable!()
    };
    assert_eq!(bias.numeric_dtype(), NumericDType::F32);
}

#[test]
fn invalid_network_output_name_precedes_initializer_callback() {
    let layers = vec![
        layer(K::FeatureInput, "x", &[("InputSize", Value::Num(2.0))]),
        layer(
            K::FullyConnected,
            "dense",
            &[
                ("OutputSize", Value::Num(3.0)),
                (
                    "WeightsInitializer",
                    Value::FunctionHandle("__deep_learning_initializer_once".into()),
                ),
            ],
        ),
    ];
    CALLBACK_INVOCATIONS.store(0, Ordering::SeqCst);
    let error = block_on(super::super::dlnetwork_builtin(vec![
        Value::Cell(CellArray::new(layers, 2, 1).unwrap()),
        Value::String("OutputNames".into()),
        Value::String("missing".into()),
    ]))
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("OutputNames must name the final layer"));
    assert_eq!(CALLBACK_INVOCATIONS.load(Ordering::SeqCst), 0);
}

#[test]
fn explicit_single_weights_select_single_for_missing_bias_and_default_state() {
    let weights = Value::Tensor(Tensor::from_f32(vec![0.1; 8 * 3], vec![8, 3]).unwrap());
    let mut layers = vec![
        layer(
            K::SequenceInput,
            "sequence",
            &[("InputSize", Value::Num(3.0))],
        ),
        layer(
            K::Lstm,
            "memory",
            &[
                ("NumHiddenUnits", Value::Num(2.0)),
                ("InputWeights", weights),
            ],
        ),
    ];
    block_on(initialise_layers(&mut layers, "test")).unwrap();
    for role in ["RecurrentWeights", "Bias"] {
        let Value::Tensor(value) = property(&layers[1], role) else {
            unreachable!()
        };
        assert_eq!(value.numeric_dtype(), NumericDType::F32);
    }
    let Value::Object(state) =
        super::super::state::initial_with_defaults(&layers, true, "test").unwrap()
    else {
        unreachable!()
    };
    let values = super::super::table_rows::values(&state, "State", "test").unwrap();
    assert!(values.data.iter().all(|value| matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)));
}

#[test]
fn orthogonal_initializer_uses_unit_columns_for_tall_recurrent_matrix() {
    let Value::Tensor(matrix) = generate::builtin(
        plan::BuiltinInitializer::Orthogonal,
        K::Lstm,
        &[8, 2],
        NumericDType::F64,
        "test",
    )
    .unwrap() else {
        unreachable!()
    };
    let values = matrix.host_buffer().materialize_f64();
    let left: f64 = values[..8].iter().map(|x| x * x).sum();
    let right: f64 = values[8..].iter().map(|x| x * x).sum();
    let dot: f64 = values[..8]
        .iter()
        .zip(&values[8..])
        .map(|(a, b)| a * b)
        .sum();
    assert!((left - 1.0).abs() < 1e-12);
    assert!((right - 1.0).abs() < 1e-12);
    assert!(dot.abs() < 1e-12);
}
