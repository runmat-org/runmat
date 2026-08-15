use futures::executor::block_on;
use runmat_accelerate_api::{GpuHandleProvenance, GpuTensorHandle, IntegerElementType};
use runmat_builtins::{IntValue, IntegerStorage, NumericDType, Tensor, Value};

const PACKET: [(&str, usize, &[usize]); 8] = [
    ("pulstran", 3, &[8, 8, 8]),
    ("rectpuls", 2, &[8, 8]),
    ("rand", 3, &[8, 8, 8]),
    ("randi", 4, &[8, 8, 6, 2]),
    ("randn", 2, &[8, 8]),
    ("random", 2, &[8, 8]),
    ("randperm", 1, &[8]),
    ("randsample", 3, &[8, 8, 8]),
];

fn integer_tensor(storage: IntegerStorage, shape: &[usize]) -> Value {
    Value::Tensor(Tensor::new_integer(storage, shape.to_vec()).expect("integer tensor"))
}

fn floating_tensor(values: Vec<f64>, shape: &[usize]) -> Value {
    Value::Tensor(Tensor::new(values, shape.to_vec()).expect("floating tensor"))
}

fn call(name: &str, args: Vec<Value>) -> runmat_runtime::BuiltinResult<Value> {
    block_on(runmat_runtime::call_builtin_async(name, &args))
}

fn all_integer_scalars(value: i8) -> Vec<IntValue> {
    vec![
        IntValue::I8(value),
        IntValue::I16(value as i16),
        IntValue::I32(value as i32),
        IntValue::I64(value as i64),
        IntValue::U8(value as u8),
        IntValue::U16(value as u16),
        IntValue::U32(value as u32),
        IntValue::U64(value as u64),
    ]
}

fn synthetic_resident_integer(buffer_id: u64, shape: Vec<usize>) -> GpuTensorHandle {
    let handle = GpuTensorHandle {
        shape,
        device_id: u32::MAX - 17,
        buffer_id,
    };
    runmat_accelerate_api::set_handle_integer_type(&handle, IntegerElementType::U64);
    runmat_accelerate_api::set_handle_provenance(&handle, GpuHandleProvenance::Explicit);
    handle
}

#[test]
fn random_and_waveform_packet_has_precise_integer_metadata() {
    for (name, expected_forms, expected_class_counts) in PACKET {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert_eq!(builtin.integer_capabilities.len(), expected_forms, "{name}");
        assert!(builtin.integer_audit.is_none(), "{name}");
        let class_counts = builtin
            .integer_capabilities
            .iter()
            .flat_map(|capability| capability.inputs.iter().map(|input| input.classes.len()))
            .collect::<Vec<_>>();
        assert_eq!(class_counts, expected_class_counts, "{name}");
    }

    let randi = runmat_builtins::builtin_function_by_name("randi").expect("randi");
    assert_eq!(
        randi.integer_capabilities[0].overflow,
        runmat_builtins::BuiltinIntegerOverflowRule::Saturate
    );
    assert_eq!(
        randi.integer_capabilities[2].inputs[0].availability,
        runmat_builtins::BuiltinIntegerInputAvailability::Documented
    );
    assert_eq!(
        randi.integer_capabilities[3].inputs[0].availability,
        runmat_builtins::BuiltinIntegerInputAvailability::RunMatOnly
    );
}

#[test]
fn random_and_waveform_extensions_are_independent_declarative_records() {
    let expected = [
        ("pulstran", 5),
        ("rectpuls", 3),
        ("rand", 2),
        ("randi", 2),
        ("randn", 2),
        ("random", 2),
        ("randperm", 2),
        ("randsample", 5),
    ];
    let mut ids = std::collections::HashSet::new();
    for (name, count) in expected {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert_eq!(builtin.extensions.len(), count, "{name}");
        for extension in builtin.extensions {
            assert_eq!(
                extension.mode,
                runmat_builtins::BuiltinExtensionMode::RunMatOnly,
                "{name}: {}",
                extension.id
            );
            assert!(
                extension.error_identifier.is_some(),
                "{name}: {}",
                extension.id
            );
            assert!(
                ids.insert(extension.id),
                "duplicate extension {}",
                extension.id
            );
        }
    }
    assert_eq!(ids.len(), 23);
}

#[test]
fn strict_mode_rejects_every_new_typed_integer_extension_by_stable_identifier() {
    let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let typed_scalar = || integer_tensor(IntegerStorage::U8(vec![1]), &[1, 1]);
    let typed_vector = || integer_tensor(IntegerStorage::U8(vec![0, 1]), &[1, 2]);
    let floating_vector = || floating_tensor(vec![0.0, 1.0], &[1, 2]);
    let cases = vec![
        (
            "pulstran",
            vec![typed_vector(), floating_vector(), Value::from("rectpuls")],
            "RunMat:compatibility:PulstranIntegerTimeExtension",
        ),
        (
            "pulstran",
            vec![floating_vector(), typed_vector(), Value::from("rectpuls")],
            "RunMat:compatibility:PulstranIntegerDelayExtension",
        ),
        (
            "pulstran",
            vec![floating_vector(), floating_vector(), typed_vector()],
            "RunMat:compatibility:PulstranIntegerPrototypeExtension",
        ),
        (
            "pulstran",
            vec![
                floating_vector(),
                floating_vector(),
                Value::from("rectpuls"),
                typed_scalar(),
            ],
            "RunMat:compatibility:PulstranIntegerParameterExtension",
        ),
        (
            "rectpuls",
            vec![typed_vector()],
            "RunMat:compatibility:RectpulsIntegerTimeExtension",
        ),
        (
            "rectpuls",
            vec![floating_vector(), typed_scalar()],
            "RunMat:compatibility:RectpulsIntegerWidthExtension",
        ),
        (
            "randn",
            vec![integer_tensor(IntegerStorage::U8(vec![2, 3]), &[2, 1])],
            "RunMat:compatibility:RandnColumnSizeVectorExtension",
        ),
        (
            "random",
            vec![Value::from("Normal"), typed_scalar(), Value::Num(1.0)],
            "RunMat:compatibility:RandomIntegerParametersExtension",
        ),
        (
            "random",
            vec![
                Value::from("Normal"),
                Value::Num(0.0),
                Value::Num(1.0),
                typed_scalar(),
            ],
            "RunMat:compatibility:RandomIntegerSizeExtension",
        ),
        (
            "randsample",
            vec![typed_scalar(), Value::Num(1.0)],
            "RunMat:compatibility:RandsampleIntegerRangeExtension",
        ),
        (
            "randsample",
            vec![typed_vector(), Value::Num(1.0)],
            "RunMat:compatibility:RandsampleIntegerPopulationExtension",
        ),
        (
            "randsample",
            vec![Value::Num(2.0), typed_scalar()],
            "RunMat:compatibility:RandsampleIntegerCountExtension",
        ),
        (
            "randsample",
            vec![Value::Num(2.0), Value::Num(1.0), typed_scalar()],
            "RunMat:compatibility:RandsampleIntegerReplacementExtension",
        ),
        (
            "randsample",
            vec![
                Value::Num(2.0),
                Value::Num(1.0),
                Value::Bool(true),
                typed_vector(),
            ],
            "RunMat:compatibility:RandsampleIntegerWeightsExtension",
        ),
    ];
    for (name, args, identifier) in cases {
        let error = call(name, args).expect_err("strict extension must reject");
        assert_eq!(error.identifier(), Some(identifier), "{name}: {error:?}");
    }
}

#[test]
fn strict_resident_integer_gates_run_before_provider_lookup() {
    let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let handles = [
        synthetic_resident_integer(1, vec![1, 2]),
        synthetic_resident_integer(2, vec![1, 1]),
    ];
    let cases = [
        (
            "rectpuls",
            vec![Value::GpuTensor(handles[0].clone())],
            "RunMat:compatibility:RectpulsIntegerTimeExtension",
        ),
        (
            "randn",
            vec![Value::GpuTensor(handles[1].clone())],
            "RunMat:compatibility:RandnResidentSizeControlExtension",
        ),
    ];
    for (name, args, identifier) in cases {
        let error = call(name, args).expect_err("resident extension gate must reject first");
        assert_eq!(error.identifier(), Some(identifier), "{name}: {error:?}");
    }
    for handle in handles {
        runmat_accelerate_api::clear_handle_metadata(&handle);
    }
}

#[test]
fn floating_extensions_reject_lossy_wide_integer_boundaries() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let wide = Value::Int(IntValue::U64(9_007_199_254_740_993));
    let cases = [
        (
            "pulstran",
            vec![wide.clone(), Value::Num(0.0), Value::from("rectpuls")],
        ),
        ("rectpuls", vec![wide.clone()]),
        (
            "random",
            vec![Value::from("Normal"), wide.clone(), Value::Num(1.0)],
        ),
        (
            "randsample",
            vec![Value::Num(1.0), Value::Num(1.0), Value::Bool(true), wide],
        ),
    ];
    for (name, args) in cases {
        let error = call(name, args).expect_err("lossy integer boundary must reject");
        assert!(
            error.message().contains("exactly representable as double"),
            "{name}: {}",
            error.message()
        );
    }
}

#[test]
fn documented_random_size_controls_accept_all_eight_integer_classes() {
    let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for size in all_integer_scalars(2) {
        for (name, args) in [
            ("rand", vec![Value::Int(size.clone())]),
            ("randi", vec![Value::Num(3.0), Value::Int(size.clone())]),
            ("randn", vec![Value::Int(size.clone())]),
            ("randperm", vec![Value::Int(size.clone())]),
        ] {
            let value =
                call(name, args).unwrap_or_else(|error| panic!("{name}/{size:?}: {error:?}"));
            let Value::Tensor(tensor) = value else {
                panic!("{name}/{size:?}: expected tensor");
            };
            if name == "randperm" {
                assert_eq!(tensor.shape, vec![1, 2], "{name}/{size:?}");
            } else {
                assert_eq!(tensor.shape, vec![2, 2], "{name}/{size:?}");
            }
            assert_eq!(tensor.numeric_dtype(), NumericDType::F64, "{name}/{size:?}");
        }
    }
}

#[test]
fn rectpuls_uses_left_closed_right_open_edges() {
    let values = floating_tensor(vec![-0.5, 0.0, 0.5], &[1, 3]);
    let Value::Tensor(output) = call("rectpuls", vec![values]).expect("rectpuls") else {
        panic!("expected tensor");
    };
    assert_eq!(output.materialize_f64(), vec![1.0, 1.0, 0.0]);
}

#[test]
fn randsample_preserves_exact_wide_integer_population_values() {
    let _runmat = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let base = 9_007_199_254_740_992_u64;
    let population = integer_tensor(IntegerStorage::U64(vec![base, base + 1]), &[1, 2]);
    let Value::Tensor(output) =
        call("randsample", vec![population, Value::Num(2.0)]).expect("wide integer population")
    else {
        panic!("expected integer tensor");
    };
    assert_eq!(output.shape, vec![1, 2]);
    let IntegerStorage::U64(values) = output.integer_storage().expect("uint64 storage") else {
        panic!("expected uint64 storage");
    };
    let mut values = values.clone();
    values.sort_unstable();
    assert_eq!(values, vec![base, base + 1]);
}
