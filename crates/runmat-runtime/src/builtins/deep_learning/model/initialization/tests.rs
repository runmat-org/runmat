use super::*;
use futures::executor::block_on;
use runmat_builtins::DeepLearningLayerKind as K;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

static CALLBACK_INVOCATIONS: AtomicUsize = AtomicUsize::new(0);
// The shared callback fixture is observed by multiple parallel lib tests.
static CALLBACK_TEST_LOCK: Mutex<()> = Mutex::new(());

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
        .replace_dl_property("Name", Value::String(name.into()))
        .expect("fixture property");
    for (name, value) in properties {
        object
            .replace_dl_property(name, value.clone())
            .expect("fixture property");
    }
    Value::Object(object)
}

fn property<'a>(layer: &'a Value, name: &str) -> &'a Value {
    let Value::Object(layer) = layer else {
        panic!("layer object")
    };
    layer.properties().get(name).expect("initialized property")
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
        assert_eq!(values.data().len(), 2);
        for value in values.data() {
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
    let _callback_guard = CALLBACK_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
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
    let _callback_guard = CALLBACK_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
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
    let _callback_guard = CALLBACK_TEST_LOCK
        .lock()
        .unwrap_or_else(|error| error.into_inner());
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
    assert!(values.data().iter().all(|value| matches!(value, Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32)));
}

#[test]
fn orthogonal_initializer_uses_unit_columns_for_tall_recurrent_matrix() {
    let mut turn = crate::builtins::common::random::try_acquire_turn(None, "test")
        .expect("ordered random turn");
    let Value::Tensor(matrix) = generate::builtin(
        &mut turn,
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

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn staged_dlarray_extra_handle_survives_collecting_publication_retirement() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    runmat_gc::gc_test_context(|| {
        let initial = runmat_gc::gc_allocate_rooted(Value::String("extra".into()))
            .expect("allocate managed dlarray property");
        let handle = initial.handle();
        let mut dlarray = ObjectInstance::new(super::super::DLARRAY_CLASS);
        dlarray
            .replace_dl_property(
                "Data",
                Value::Tensor(Tensor::new(vec![1.0], vec![1, 1]).expect("numeric data")),
            )
            .expect("fixture property");
        dlarray
            .replace_dl_property(
                "Extra",
                Value::HandleObject(runmat_value::HandleRef {
                    class_name: "ExtraPayload".into(),
                    target: handle,
                    valid: true,
                }),
            )
            .expect("fixture property");
        let staged = RootedValue::new(Value::Object(dlarray)).expect("root staged value");
        initial.unroot().expect("transfer to staged owner");
        parameters::validate_initialized_parameter(
            staged.value(),
            &ParameterShape::Elements(1),
            ParameterRole::Weights,
            "test",
        )
        .expect("numeric Data admits extra property");
        let mut layers = vec![Value::Object(ObjectInstance::new("TestLayer"))];
        thread_local! {
            static OBSERVED_HANDLE: std::cell::RefCell<Option<runmat_gc::GcHandle>> =
                const { std::cell::RefCell::new(None) };
        }
        OBSERVED_HANDLE.with(|slot| *slot.borrow_mut() = Some(handle));
        let observed = Arc::new(AtomicBool::new(false));
        let observed_in_callback = Arc::clone(&observed);
        let _observer = runmat_gc::RootRetirementObserver::register(Arc::new(move |event| {
            if !OBSERVED_HANDLE.with(|slot| {
                slot.borrow()
                    .is_some_and(|handle| event == runmat_gc::RootRetirement::Explicit(handle))
            }) || observed_in_callback.swap(true, Ordering::SeqCst)
            {
                return;
            }
            runmat_gc::gc_collect_major().expect("collect as staged root retires");
            OBSERVED_HANDLE.with(|slot| {
                let handle = slot.borrow().expect("observed handle remains selected");
                assert_eq!(
                    runmat_gc::gc_clone_value(&handle).expect("publication lease is live"),
                    Value::String("extra".into())
                );
            });
        }))
        .expect("register collecting retirement observer");
        let published = publish_staged_parameters(
            None,
            &mut layers,
            vec![StagedParameter {
                layer_index: 0,
                role: ParameterRole::Weights,
                value: staged,
                generated: false,
                existing_precision: None,
            }],
        )
        .expect("publish one initialized parameter");
        assert!(observed.load(Ordering::SeqCst));
        drop(published);
        runmat_gc::gc_collect_major().expect("collect after publication lease retires");
        assert!(runmat_gc::gc_clone_value(&handle).is_err());
        OBSERVED_HANDLE.with(|slot| *slot.borrow_mut() = None);
    });
}
