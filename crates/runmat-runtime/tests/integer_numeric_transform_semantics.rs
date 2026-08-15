use futures::executor::block_on;
use runmat_builtins::{IntValue, IntegerStorage, NumericDType, Tensor, Value};

const PACKET: [(&str, usize); 8] = [
    ("pchip", 3),
    ("pdf", 2),
    ("pdist", 2),
    ("pdist2", 3),
    ("peaks", 2),
    ("periodogram", 2),
    ("perms", 1),
    ("permute", 2),
];

fn tensor(storage: IntegerStorage, shape: &[usize]) -> Value {
    Value::Tensor(Tensor::new_integer(storage, shape.to_vec()).expect("integer tensor"))
}

fn floating(values: Vec<f64>, shape: &[usize]) -> Value {
    Value::Tensor(Tensor::new(values, shape.to_vec()).expect("floating tensor"))
}

fn all_integer_vectors() -> Vec<(&'static str, Value)> {
    vec![
        ("int8", tensor(IntegerStorage::I8(vec![1, 2]), &[1, 2])),
        ("int16", tensor(IntegerStorage::I16(vec![1, 2]), &[1, 2])),
        ("int32", tensor(IntegerStorage::I32(vec![1, 2]), &[1, 2])),
        ("int64", tensor(IntegerStorage::I64(vec![1, 2]), &[1, 2])),
        ("uint8", tensor(IntegerStorage::U8(vec![1, 2]), &[1, 2])),
        ("uint16", tensor(IntegerStorage::U16(vec![1, 2]), &[1, 2])),
        ("uint32", tensor(IntegerStorage::U32(vec![1, 2]), &[1, 2])),
        ("uint64", tensor(IntegerStorage::U64(vec![1, 2]), &[1, 2])),
    ]
}

#[test]
fn numeric_transform_packet_has_class_complete_capability_metadata() {
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
fn numeric_transform_extensions_are_declarative_and_independent() {
    let expected = [
        ("pchip", 3),
        ("pdf", 2),
        ("pdist", 2),
        ("pdist2", 3),
        ("peaks", 2),
        ("periodogram", 2),
        ("perms", 0),
        ("permute", 0),
    ];
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
        }
    }
}

#[test]
fn strict_mode_rejects_runmat_only_integer_forms_before_evaluation() {
    let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    let calls = [
        (
            "pchip",
            vec![
                tensor(IntegerStorage::U8(vec![0, 1]), &[1, 2]),
                floating(vec![0.0, 1.0], &[1, 2]),
                Value::Num(0.5),
            ],
        ),
        (
            "pdf",
            vec![
                Value::from("Normal"),
                Value::Int(IntValue::U8(0)),
                Value::Num(0.0),
                Value::Num(1.0),
            ],
        ),
        (
            "pdist",
            vec![tensor(IntegerStorage::U8(vec![0, 1]), &[2, 1])],
        ),
        (
            "pdist2",
            vec![
                tensor(IntegerStorage::U8(vec![0, 1]), &[2, 1]),
                floating(vec![0.0, 1.0], &[2, 1]),
            ],
        ),
        ("peaks", vec![Value::Int(IntValue::U8(3))]),
        (
            "periodogram",
            vec![tensor(IntegerStorage::U8(vec![0, 1, 0, 1]), &[4, 1])],
        ),
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
fn floating_extensions_reject_lossy_wide_uint64_boundaries() {
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let wide = 9_007_199_254_740_993_u64;
    let calls = [
        (
            "pchip",
            vec![
                tensor(IntegerStorage::U64(vec![0, wide]), &[1, 2]),
                floating(vec![0.0, 1.0], &[1, 2]),
                Value::Num(0.5),
            ],
        ),
        (
            "pchip",
            vec![
                floating(vec![0.0, 1.0], &[1, 2]),
                tensor(IntegerStorage::U64(vec![0, wide]), &[1, 2]),
                Value::Num(0.5),
            ],
        ),
        (
            "pchip",
            vec![
                floating(vec![0.0, 1.0], &[1, 2]),
                floating(vec![0.0, 1.0], &[1, 2]),
                Value::Int(IntValue::U64(wide)),
            ],
        ),
        (
            "pdf",
            vec![
                Value::from("Normal"),
                Value::Int(IntValue::U64(wide)),
                Value::Num(0.0),
                Value::Num(1.0),
            ],
        ),
        (
            "pdf",
            vec![
                Value::from("Normal"),
                Value::Num(0.0),
                Value::Int(IntValue::U64(wide)),
                Value::Num(1.0),
            ],
        ),
        (
            "pdist",
            vec![tensor(IntegerStorage::U64(vec![0, wide]), &[2, 1])],
        ),
        (
            "pdist",
            vec![
                floating(vec![0.0, 1.0], &[2, 1]),
                Value::from("minkowski"),
                Value::Int(IntValue::U64(wide)),
            ],
        ),
        (
            "pdist",
            vec![
                floating(vec![0.0, 1.0], &[2, 1]),
                Value::from("CacheSize"),
                Value::Int(IntValue::U64(wide)),
            ],
        ),
        (
            "pdist2",
            vec![
                tensor(IntegerStorage::U64(vec![0, wide]), &[2, 1]),
                floating(vec![0.0, 1.0], &[2, 1]),
            ],
        ),
        (
            "pdist2",
            vec![
                floating(vec![0.0, 1.0], &[2, 1]),
                floating(vec![0.0, 1.0], &[2, 1]),
                Value::from("minkowski"),
                Value::Int(IntValue::U64(wide)),
            ],
        ),
        (
            "periodogram",
            vec![tensor(IntegerStorage::U64(vec![0, wide]), &[2, 1])],
        ),
        (
            "periodogram",
            vec![
                floating(vec![0.0, 1.0], &[2, 1]),
                tensor(IntegerStorage::U64(vec![1, wide]), &[2, 1]),
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
fn documented_reordering_forms_preserve_all_integer_classes() {
    for (class, input) in all_integer_vectors() {
        let output =
            block_on(runmat_runtime::call_builtin_async("perms", &[input])).expect("integer perms");
        let Value::Tensor(output) = output else {
            panic!("{class}: expected tensor");
        };
        assert_ne!(output.numeric_dtype(), NumericDType::F64, "{class}");
        assert_eq!(
            output
                .integer_storage()
                .expect("integer output")
                .class_name(),
            class
        );

        let data = tensor(
            IntegerStorage::U64(vec![9_007_199_254_740_993, 9_007_199_254_740_994]),
            &[1, 2],
        );
        let order = match class {
            "int8" => tensor(IntegerStorage::I8(vec![2, 1]), &[1, 2]),
            "int16" => tensor(IntegerStorage::I16(vec![2, 1]), &[1, 2]),
            "int32" => tensor(IntegerStorage::I32(vec![2, 1]), &[1, 2]),
            "int64" => tensor(IntegerStorage::I64(vec![2, 1]), &[1, 2]),
            "uint8" => tensor(IntegerStorage::U8(vec![2, 1]), &[1, 2]),
            "uint16" => tensor(IntegerStorage::U16(vec![2, 1]), &[1, 2]),
            "uint32" => tensor(IntegerStorage::U32(vec![2, 1]), &[1, 2]),
            "uint64" => tensor(IntegerStorage::U64(vec![2, 1]), &[1, 2]),
            _ => unreachable!(),
        };
        let output = block_on(runmat_runtime::call_builtin_async(
            "permute",
            &[data, order],
        ))
        .expect("integer permute");
        let Value::Tensor(output) = output else {
            panic!("{class}: expected permuted tensor");
        };
        assert_eq!(
            output.integer_storage(),
            Some(&IntegerStorage::U64(vec![
                9_007_199_254_740_993,
                9_007_199_254_740_994
            ]))
        );
        assert_eq!(output.shape, vec![2, 1]);
    }
}

#[test]
fn pdist2_typed_selection_count_remains_structural() {
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let output = block_on(runmat_runtime::call_builtin_async(
        "pdist2",
        &[
            floating(vec![0.0, 2.0], &[2, 1]),
            floating(vec![1.0], &[1, 1]),
            Value::from("Smallest"),
            Value::Int(IntValue::U64(2)),
        ],
    ))
    .expect("structural selection count");
    let Value::Tensor(output) = output else {
        panic!("expected selected distances");
    };
    assert_eq!(output.shape, vec![2, 1]);

    let wide = 9_007_199_254_740_993_u64;
    let output = block_on(runmat_runtime::call_builtin_async(
        "pdist2",
        &[
            floating(vec![0.0, 2.0], &[2, 1]),
            floating(vec![1.0], &[1, 1]),
            Value::from("Smallest"),
            Value::Int(IntValue::U64(wide)),
        ],
    ))
    .expect("wide structural selection count clamps to observation count");
    let Value::Tensor(output) = output else {
        panic!("expected selected distances");
    };
    assert_eq!(output.shape, vec![2, 1]);
}

#[test]
fn distance_cache_size_controls_accept_every_integer_class() {
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for cache_size in [
        IntValue::I8(8),
        IntValue::I16(8),
        IntValue::I32(8),
        IntValue::I64(8),
        IntValue::U8(8),
        IntValue::U16(8),
        IntValue::U32(8),
        IntValue::U64(8),
    ] {
        let output = block_on(runmat_runtime::call_builtin_async(
            "pdist",
            &[
                floating(vec![0.0, 2.0], &[2, 1]),
                Value::from("CacheSize"),
                Value::Int(cache_size.clone()),
            ],
        ))
        .expect("typed pdist CacheSize");
        assert!(matches!(output, Value::Num(_) | Value::Tensor(_)));

        let output = block_on(runmat_runtime::call_builtin_async(
            "pdist2",
            &[
                floating(vec![0.0, 2.0], &[2, 1]),
                floating(vec![1.0], &[1, 1]),
                Value::from("CacheSize"),
                Value::Int(cache_size),
            ],
        ))
        .expect("typed pdist2 CacheSize");
        assert!(matches!(output, Value::Tensor(_)));
    }
}
