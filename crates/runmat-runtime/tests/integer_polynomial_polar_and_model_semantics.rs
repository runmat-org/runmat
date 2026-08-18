use futures::executor::block_on;
use runmat_accelerate_api::{GpuHandleProvenance, GpuTensorHandle, IntegerElementType};
use runmat_builtins::{
    ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, ObjectInstance, Tensor, Value,
};

const PACKET: [(&str, usize); 15] = [
    ("perfcurve", 4),
    ("pie", 2),
    ("pivot", 4),
    ("pol2cart", 3),
    ("polarhistogram", 4),
    ("polarplot", 3),
    ("polarscatter", 3),
    ("pole", 1),
    ("polyder", 1),
    ("polyfit", 4),
    ("polyval", 3),
    ("pow2", 3),
    ("ppval", 1),
    ("predict", 3),
    ("print", 1),
];

fn tensor(storage: IntegerStorage, shape: &[usize]) -> Value {
    Value::Tensor(Tensor::new_integer(storage, shape.to_vec()).expect("integer tensor"))
}

fn floating(values: Vec<f64>, shape: &[usize]) -> Value {
    Value::Tensor(Tensor::new(values, shape.to_vec()).expect("floating tensor"))
}

fn resident_integer(buffer_id: u64, shape: Vec<usize>) -> Value {
    let handle = GpuTensorHandle {
        shape,
        device_id: u32::MAX - 442,
        buffer_id,
        descriptor: Default::default(),
    };
    runmat_accelerate_api::set_handle_integer_type(&handle, IntegerElementType::U64);
    runmat_accelerate_api::set_handle_provenance(&handle, GpuHandleProvenance::Explicit);
    Value::GpuTensor(handle)
}

fn integer_cases(values: &[i64]) -> Vec<(&'static str, Value, Value)> {
    let unsigned = values.iter().map(|value| *value as u64).collect::<Vec<_>>();
    vec![
        (
            "int8",
            tensor(
                IntegerStorage::I8(values.iter().map(|value| *value as i8).collect()),
                &[1, values.len()],
            ),
            Value::Int(IntValue::I8(*values.last().expect("value") as i8)),
        ),
        (
            "int16",
            tensor(
                IntegerStorage::I16(values.iter().map(|value| *value as i16).collect()),
                &[1, values.len()],
            ),
            Value::Int(IntValue::I16(*values.last().expect("value") as i16)),
        ),
        (
            "int32",
            tensor(
                IntegerStorage::I32(values.iter().map(|value| *value as i32).collect()),
                &[1, values.len()],
            ),
            Value::Int(IntValue::I32(*values.last().expect("value") as i32)),
        ),
        (
            "int64",
            tensor(IntegerStorage::I64(values.to_vec()), &[1, values.len()]),
            Value::Int(IntValue::I64(*values.last().expect("value"))),
        ),
        (
            "uint8",
            tensor(
                IntegerStorage::U8(unsigned.iter().map(|value| *value as u8).collect()),
                &[1, values.len()],
            ),
            Value::Int(IntValue::U8(*unsigned.last().expect("value") as u8)),
        ),
        (
            "uint16",
            tensor(
                IntegerStorage::U16(unsigned.iter().map(|value| *value as u16).collect()),
                &[1, values.len()],
            ),
            Value::Int(IntValue::U16(*unsigned.last().expect("value") as u16)),
        ),
        (
            "uint32",
            tensor(
                IntegerStorage::U32(unsigned.iter().map(|value| *value as u32).collect()),
                &[1, values.len()],
            ),
            Value::Int(IntValue::U32(*unsigned.last().expect("value") as u32)),
        ),
        (
            "uint64",
            tensor(IntegerStorage::U64(unsigned.clone()), &[1, values.len()]),
            Value::Int(IntValue::U64(*unsigned.last().expect("value"))),
        ),
    ]
}

#[test]
fn polynomial_polar_and_model_packet_has_class_complete_metadata() {
    for (name, expected_forms) in PACKET {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert_eq!(builtin.integer_capabilities.len(), expected_forms, "{name}");
        assert!(builtin.integer_audit.is_none(), "{name}");
        for capability in builtin.integer_capabilities {
            assert!(!capability.inputs.is_empty(), "{name}: {}", capability.form);
            for input in capability.inputs {
                assert_eq!(input.classes.len(), 8, "{name}: {}", input.name);
            }
        }
    }
}

#[test]
fn polynomial_polar_and_model_extensions_are_declarative_and_independent() {
    let expected = [
        ("perfcurve", 4),
        ("pie", 1),
        ("pivot", 0),
        ("pol2cart", 3),
        ("polarhistogram", 0),
        ("polarplot", 1),
        ("polarscatter", 0),
        ("pole", 0),
        ("polyder", 1),
        ("polyfit", 3),
        ("polyval", 3),
        ("pow2", 3),
        ("ppval", 1),
        ("predict", 2),
        ("print", 1),
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
fn strict_mode_rejects_each_runmat_only_integer_surface_before_evaluation() {
    let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let complex = Value::ComplexTensor(
        ComplexTensor::new_integer(
            IntegerComplexStorage::new(IntegerStorage::U8(vec![1]), IntegerStorage::U8(vec![1]))
                .expect("complex storage"),
            vec![1, 1],
        )
        .expect("complex integer"),
    );
    let statistical_model = Value::Object(ObjectInstance::new("LinearModel".to_string()));
    let calls = vec![
        (
            "perfcurve",
            vec![
                tensor(IntegerStorage::U8(vec![0, 1]), &[2, 1]),
                floating(vec![0.0, 1.0], &[2, 1]),
                Value::Int(IntValue::U8(1)),
            ],
        ),
        ("pie", vec![tensor(IntegerStorage::U8(vec![1, 2]), &[1, 2])]),
        (
            "pol2cart",
            vec![Value::Int(IntValue::U8(0)), Value::Num(1.0)],
        ),
        ("polarplot", vec![complex, Value::Num(1.0)]),
        (
            "polyder",
            vec![tensor(IntegerStorage::U8(vec![1, 2]), &[1, 2])],
        ),
        (
            "polyfit",
            vec![
                tensor(IntegerStorage::U8(vec![0, 1]), &[1, 2]),
                floating(vec![0.0, 1.0], &[1, 2]),
                Value::Num(1.0),
            ],
        ),
        (
            "polyval",
            vec![
                tensor(IntegerStorage::U8(vec![1, 2]), &[1, 2]),
                Value::Num(1.0),
            ],
        ),
        ("pow2", vec![Value::Int(IntValue::U8(1))]),
        (
            "ppval",
            vec![Value::Bool(false), Value::Int(IntValue::U8(1))],
        ),
        (
            "predict",
            vec![statistical_model, Value::Int(IntValue::U8(1))],
        ),
        ("print", vec![Value::Int(IntValue::U8(1))]),
    ];
    for (name, args) in calls {
        let error = block_on(runmat_runtime::call_builtin_async(name, &args))
            .expect_err("RunMat-only integer form must reject");
        assert!(
            error
                .identifier()
                .is_some_and(|identifier| identifier.starts_with("RunMat:compatibility:")),
            "{name}: {error:?}"
        );
    }
}

#[test]
fn strict_resident_integer_gates_run_before_provider_lookup() {
    let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let calls = [
        ("pow2", vec![resident_integer(1, vec![1, 2])]),
        (
            "perfcurve",
            vec![
                resident_integer(2, vec![2, 1]),
                floating(vec![0.0, 1.0], &[2, 1]),
                Value::Num(1.0),
            ],
        ),
        (
            "predict",
            vec![
                Value::Object(ObjectInstance::new("LinearModel".to_string())),
                resident_integer(3, vec![1, 2]),
            ],
        ),
    ];
    for (name, args) in calls {
        let error = block_on(runmat_runtime::call_builtin_async(name, &args))
            .expect_err("strict resident extension must reject without a provider");
        assert!(
            error
                .identifier()
                .is_some_and(|identifier| identifier.starts_with("RunMat:compatibility:")),
            "{name}: {error:?}"
        );
        for value in args {
            if let Value::GpuTensor(handle) = value {
                runmat_accelerate_api::clear_handle_metadata(&handle);
            }
        }
    }
}

#[test]
fn floating_extensions_reject_lossy_wide_uint64_boundaries() {
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let wide = 9_007_199_254_740_993_u64;
    let calls = [
        (
            "pol2cart",
            vec![Value::Int(IntValue::U64(wide)), Value::Num(1.0)],
        ),
        ("polyder", vec![Value::Int(IntValue::U64(wide))]),
        (
            "polyfit",
            vec![
                tensor(IntegerStorage::U64(vec![0, wide]), &[1, 2]),
                floating(vec![0.0, 1.0], &[1, 2]),
                Value::Num(1.0),
            ],
        ),
        (
            "polyval",
            vec![Value::Int(IntValue::U64(wide)), Value::Num(1.0)],
        ),
        ("pow2", vec![Value::Int(IntValue::U64(wide))]),
        (
            "ppval",
            vec![Value::Bool(false), Value::Int(IntValue::U64(wide))],
        ),
        (
            "perfcurve",
            vec![
                floating(vec![0.0, 1.0], &[2, 1]),
                tensor(IntegerStorage::U64(vec![0, wide]), &[2, 1]),
                Value::Num(1.0),
            ],
        ),
    ];
    for (name, args) in calls {
        let error = block_on(runmat_runtime::call_builtin_async(name, &args))
            .expect_err("lossy integer boundary must reject");
        assert!(
            error.message().contains("exactly representable as double"),
            "{name}: {}",
            error.message()
        );
    }
}

#[test]
fn exact_integer_compute_extensions_accept_every_native_class() {
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for (class, values, positive) in integer_cases(&[0, 1]) {
        block_on(runmat_runtime::call_builtin_async(
            "pol2cart",
            &[values.clone(), floating(vec![1.0, 1.0], &[1, 2])],
        ))
        .unwrap_or_else(|error| panic!("{class}: pol2cart: {error:?}"));
        block_on(runmat_runtime::call_builtin_async(
            "polyder",
            &[values.clone()],
        ))
        .unwrap_or_else(|error| panic!("{class}: polyder: {error:?}"));
        block_on(runmat_runtime::call_builtin_async(
            "polyfit",
            &[
                values.clone(),
                floating(vec![1.0, 3.0], &[1, 2]),
                Value::Num(1.0),
            ],
        ))
        .unwrap_or_else(|error| panic!("{class}: polyfit: {error:?}"));
        block_on(runmat_runtime::call_builtin_async(
            "polyval",
            &[values.clone(), floating(vec![0.0, 1.0], &[1, 2])],
        ))
        .unwrap_or_else(|error| panic!("{class}: polyval: {error:?}"));
        block_on(runmat_runtime::call_builtin_async(
            "pow2",
            &[values.clone()],
        ))
        .unwrap_or_else(|error| panic!("{class}: pow2: {error:?}"));
        block_on(runmat_runtime::call_builtin_async(
            "perfcurve",
            &[values, floating(vec![0.0, 1.0], &[2, 1]), positive],
        ))
        .unwrap_or_else(|error| panic!("{class}: perfcurve: {error:?}"));
    }
}

#[test]
fn perfcurve_keeps_wide_integer_labels_distinct_without_a_f64_mirror() {
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let base = 9_007_199_254_740_992_u64;
    let output = block_on(runmat_runtime::call_builtin_async(
        "perfcurve",
        &[
            tensor(IntegerStorage::U64(vec![base, base + 1]), &[2, 1]),
            floating(vec![0.0, 1.0], &[2, 1]),
            Value::Int(IntValue::U64(base + 1)),
        ],
    ))
    .expect("wide labels remain distinct");
    assert!(matches!(output, Value::Tensor(_)));
}
