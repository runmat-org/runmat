use futures::executor::block_on;
use runmat_builtins::{IntValue, IntegerStorage, NumericDType, StringArray, Tensor, Value};

fn call(name: &str, args: Vec<Value>) -> runmat_runtime::BuiltinResult<Value> {
    block_on(runmat_runtime::call_builtin_async(name, &args))
}

fn integer_size_vectors() -> Vec<(NumericDType, Value)> {
    vec![
        (
            NumericDType::I8,
            integer(IntegerStorage::I8(vec![2, 1]), &[1, 2]),
        ),
        (
            NumericDType::I16,
            integer(IntegerStorage::I16(vec![2, 1]), &[1, 2]),
        ),
        (
            NumericDType::I32,
            integer(IntegerStorage::I32(vec![2, 1]), &[1, 2]),
        ),
        (
            NumericDType::I64,
            integer(IntegerStorage::I64(vec![2, 1]), &[1, 2]),
        ),
        (
            NumericDType::U8,
            integer(IntegerStorage::U8(vec![2, 1]), &[1, 2]),
        ),
        (
            NumericDType::U16,
            integer(IntegerStorage::U16(vec![2, 1]), &[1, 2]),
        ),
        (
            NumericDType::U32,
            integer(IntegerStorage::U32(vec![2, 1]), &[1, 2]),
        ),
        (
            NumericDType::U64,
            integer(IntegerStorage::U64(vec![2, 1]), &[1, 2]),
        ),
    ]
}

fn integer(storage: IntegerStorage, shape: &[usize]) -> Value {
    Value::Tensor(Tensor::new_integer(storage, shape.to_vec()).expect("integer tensor"))
}

fn type_names(names: &[&str]) -> Value {
    Value::StringArray(
        StringArray::new(
            names.iter().map(|name| (*name).to_string()).collect(),
            vec![1, names.len()],
        )
        .expect("type names"),
    )
}

#[test]
fn packet_metadata_is_complete_and_extensions_are_independent() {
    let expected = [
        ("table", 2, 1),
        ("table2array", 1, 0),
        ("table2cell", 1, 0),
        ("table2struct", 2, 1),
        ("table2timetable", 4, 3),
        ("timetable", 4, 3),
        ("timetable2table", 2, 2),
        ("timerange", 1, 2),
        ("tan", 1, 4),
        ("tand", 1, 3),
        ("tanh", 1, 3),
    ];
    for (name, capabilities, extensions) in expected {
        let builtin = runmat_builtins::builtin_function_by_name(name).expect("registered builtin");
        assert_eq!(builtin.integer_capabilities.len(), capabilities, "{name}");
        assert_eq!(builtin.extensions.len(), extensions, "{name}");
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
fn table_preallocation_decodes_every_integer_size_class_and_allocates_native_columns() {
    let variable_types = [
        ("int8", NumericDType::I8),
        ("int16", NumericDType::I16),
        ("int32", NumericDType::I32),
        ("int64", NumericDType::I64),
        ("uint8", NumericDType::U8),
        ("uint16", NumericDType::U16),
        ("uint32", NumericDType::U32),
        ("uint64", NumericDType::U64),
    ];
    for ((size_dtype, size), (type_name, output_dtype)) in
        integer_size_vectors().into_iter().zip(variable_types)
    {
        let table = call(
            "table",
            vec![
                Value::from("Size"),
                size,
                Value::from("VariableTypes"),
                type_names(&[type_name]),
            ],
        )
        .expect("preallocated table");
        let Value::Tensor(array) = call("table2array", vec![table]).expect("table2array") else {
            panic!("expected numeric array")
        };
        assert_eq!(array.shape, vec![2, 1], "size class {size_dtype:?}");
        assert_eq!(array.numeric_dtype(), output_dtype, "{type_name}");
        assert!(array
            .integer_storage()
            .expect("integer output")
            .exact_values()
            .iter()
            .all(IntValue::is_zero));
    }
}

#[test]
fn table_conversions_preserve_wide_integer_payloads() {
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    let storage = IntegerStorage::U64(vec![9_007_199_254_740_993, u64::MAX]);
    let table = call("table", vec![integer(storage.clone(), &[2, 1])]).expect("table");

    let Value::Tensor(array) = call("table2array", vec![table.clone()]).expect("array") else {
        panic!("expected tensor")
    };
    assert_eq!(array.integer_storage(), Some(&storage));

    let Value::Cell(cells) = call("table2cell", vec![table.clone()]).expect("cell") else {
        panic!("expected cell")
    };
    assert_eq!(
        cells.get(0, 0).unwrap(),
        Value::Int(IntValue::U64(9_007_199_254_740_993))
    );
    assert_eq!(
        cells.get(1, 0).unwrap(),
        Value::Int(IntValue::U64(u64::MAX))
    );

    let Value::Struct(structure) = call(
        "table2struct",
        vec![table.clone(), Value::from("ToScalar"), Value::Bool(true)],
    )
    .expect("scalar struct") else {
        panic!("expected scalar struct")
    };
    let Value::Tensor(field) = structure.fields.get("Var1").expect("Var1") else {
        panic!("expected integer field")
    };
    assert_eq!(field.integer_storage(), Some(&storage));

    let numeric_times = IntegerStorage::U64(vec![9_007_199_254_740_995, u64::MAX - 1]);
    let timetable = call(
        "table2timetable",
        vec![
            table.clone(),
            Value::from("RowTimes"),
            integer(numeric_times.clone(), &[2, 1]),
        ],
    )
    .expect("numeric row-time extension");
    let table_with_times = call(
        "timetable2table",
        vec![timetable, Value::from("ConvertRowTimes"), Value::Bool(true)],
    )
    .expect("row times to table");
    let Value::Tensor(with_times) = call("table2array", vec![table_with_times]).expect("array")
    else {
        panic!("expected integer array")
    };
    assert_eq!(
        with_times.integer_storage(),
        Some(&IntegerStorage::U64(vec![
            9_007_199_254_740_995,
            u64::MAX - 1,
            9_007_199_254_740_993,
            u64::MAX,
        ]))
    );

    let timetable = call(
        "table2timetable",
        vec![
            table,
            Value::from("SampleRate"),
            Value::Int(IntValue::U16(4)),
        ],
    )
    .expect("timetable");
    let Value::Tensor(round_trip) = call("table2array", vec![timetable]).expect("round trip")
    else {
        panic!("expected integer timetable data")
    };
    assert_eq!(round_trip.integer_storage(), Some(&storage));
}

#[test]
fn strict_mode_rejects_tangent_extensions_and_integer_toscalar() {
    let _strict = runmat_runtime::compatibility::push_runmat_extensions_enabled(false);
    for name in ["tan", "tand", "tanh"] {
        let error = call(name, vec![Value::Int(IntValue::I16(1))])
            .expect_err("integer tangent input must be gated");
        assert!(error
            .identifier()
            .is_some_and(|identifier| identifier.starts_with("RunMat:compatibility:")));
    }
    let table = call("table", vec![Value::Num(1.0)]).expect("table");
    let error = call(
        "table2struct",
        vec![table, Value::from("ToScalar"), Value::Int(IntValue::U8(1))],
    )
    .expect_err("integer ToScalar must be gated");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:Table2StructIntegerToScalarExtension")
    );
    let table = call(
        "table",
        vec![Value::Tensor(
            Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap(),
        )],
    )
    .expect("table");
    let error = call(
        "table2timetable",
        vec![
            table,
            Value::from("RowTimes"),
            integer(IntegerStorage::I64(vec![1, 2]), &[2, 1]),
        ],
    )
    .expect_err("numeric RowTimes must be gated");
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:Table2TimetableNumericRowTimesExtension")
    );
}

#[test]
fn tangent_extensions_preserve_single_and_handle_wide_integer_boundaries_deliberately() {
    let _extensions = runmat_runtime::compatibility::push_runmat_extensions_enabled(true);
    for name in ["tan", "tand", "tanh"] {
        let single = Value::Tensor(Tensor::from_f32(vec![0.0, 0.5], vec![1, 2]).unwrap());
        let Value::Tensor(output) = call(name, vec![single]).expect("single tangent") else {
            panic!("expected single tensor")
        };
        assert_eq!(output.numeric_dtype(), NumericDType::F32, "{name}");
    }

    let wide = 9_007_199_254_740_993_u64;
    for name in ["tan", "tanh"] {
        call(name, vec![Value::Int(IntValue::U64(wide))])
            .expect_err("lossy binary64 boundary must reject");
    }
    let expected = ((wide % 360) as f64).to_radians().tan();
    let Value::Num(actual) =
        call("tand", vec![Value::Int(IntValue::U64(wide))]).expect("wide exact degree reduction")
    else {
        panic!("expected double scalar")
    };
    assert!((actual - expected).abs() < 1e-12);
}
