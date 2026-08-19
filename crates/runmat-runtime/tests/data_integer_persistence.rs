use runmat_runtime::data::{DataArrayPayload, DataArrayValues};
use runmat_value::{CellArray, IntValue, IntegerStorage, NumericDType, StructValue, Tensor, Value};

fn create_array(path: String, dtype: &str, shape: Vec<usize>, chunk: Vec<usize>) -> Value {
    let mut array = StructValue::new();
    array
        .fields
        .insert("dtype".to_string(), Value::String(dtype.to_string()));
    array.fields.insert(
        "shape".to_string(),
        Value::Tensor(
            Tensor::new(
                shape.iter().map(|&value| value as f64).collect(),
                vec![1, shape.len()],
            )
            .expect("shape tensor"),
        ),
    );
    array.fields.insert(
        "chunk".to_string(),
        Value::Tensor(
            Tensor::new(
                chunk.iter().map(|&value| value as f64).collect(),
                vec![1, chunk.len()],
            )
            .expect("chunk tensor"),
        ),
    );
    let mut arrays = StructValue::new();
    arrays.insert("samples", Value::Struct(array));
    let mut schema = StructValue::new();
    schema.insert("arrays", Value::Struct(arrays));
    let dataset =
        runmat_runtime::call_builtin("data.create", &[Value::String(path), Value::Struct(schema)])
            .expect("create dataset");
    runmat_runtime::call_builtin(
        "Dataset.array",
        &[dataset, Value::String("samples".to_string())],
    )
    .expect("open data array")
}

#[test]
fn data_payloads_roundtrip_all_integer_classes_and_legacy_json() {
    let cases = vec![
        ("int8", DataArrayValues::I8(vec![i8::MIN, i8::MAX])),
        ("int16", DataArrayValues::I16(vec![i16::MIN, i16::MAX])),
        ("int32", DataArrayValues::I32(vec![i32::MIN, i32::MAX])),
        ("int64", DataArrayValues::I64(vec![i64::MIN, i64::MAX])),
        ("uint8", DataArrayValues::U8(vec![0, u8::MAX])),
        ("uint16", DataArrayValues::U16(vec![0, u16::MAX])),
        ("uint32", DataArrayValues::U32(vec![0, u32::MAX])),
        ("uint64", DataArrayValues::U64(vec![0, u64::MAX])),
    ];
    for (dtype, values) in cases {
        let payload = DataArrayPayload {
            dtype: dtype.to_string(),
            shape: vec![1, 2],
            values: values.clone(),
            imaginary_values: None,
        };
        let bytes = serde_json::to_vec(&payload).expect("encode payload");
        let decoded: DataArrayPayload = serde_json::from_slice(&bytes).expect("decode payload");
        assert_eq!(decoded.values, values, "{dtype}");
    }

    let legacy = br#"{"dtype":"f64","shape":[1,2],"values":[1,2]}"#;
    let decoded: DataArrayPayload = serde_json::from_slice(legacy).expect("decode legacy payload");
    assert_eq!(decoded.values, DataArrayValues::F64(vec![1.0, 2.0]));
}

#[test]
fn data_arrays_keep_exact_integer_storage_through_chunked_api_paths() {
    let cases = vec![
        ("int8", IntegerStorage::I8(vec![i8::MIN, i8::MAX])),
        ("int16", IntegerStorage::I16(vec![i16::MIN, i16::MAX])),
        ("int32", IntegerStorage::I32(vec![i32::MIN, i32::MAX])),
        ("int64", IntegerStorage::I64(vec![i64::MIN, i64::MAX])),
        ("uint8", IntegerStorage::U8(vec![0, u8::MAX])),
        ("uint16", IntegerStorage::U16(vec![0, u16::MAX])),
        ("uint32", IntegerStorage::U32(vec![0, u32::MAX])),
        ("uint64", IntegerStorage::U64(vec![0, u64::MAX])),
    ];
    for (dtype, storage) in cases {
        let dir = tempfile::tempdir().expect("tempdir");
        let array = create_array(
            dir.path()
                .join(format!("{dtype}.data"))
                .display()
                .to_string(),
            dtype,
            vec![2, 1],
            vec![1, 1],
        );
        let input = Tensor::new_integer(storage.clone(), vec![2, 1]).expect("integer tensor");
        runmat_runtime::call_builtin("DataArray.write", &[array.clone(), Value::Tensor(input)])
            .expect("write array");
        let Value::Tensor(read_back) =
            runmat_runtime::call_builtin("DataArray.read", &[array]).expect("read array")
        else {
            panic!("expected tensor");
        };
        assert_eq!(read_back.integer_storage(), Some(&storage), "{dtype}");
    }
}

#[test]
fn data_arrays_keep_native_single_storage_through_chunked_api_paths() {
    let dir = tempfile::tempdir().expect("tempdir");
    let array = create_array(
        dir.path().join("f32.data").display().to_string(),
        "f32",
        vec![2, 1],
        vec![1, 1],
    );
    let input = Tensor::from_f32(vec![0.1, -2.5], vec![2, 1]).expect("single tensor");
    runmat_runtime::call_builtin("DataArray.write", &[array.clone(), Value::Tensor(input)])
        .expect("write single array");
    let Value::Tensor(read_back) =
        runmat_runtime::call_builtin("DataArray.read", &[array]).expect("read single array")
    else {
        panic!("expected tensor");
    };
    assert_eq!(read_back.numeric_dtype(), NumericDType::F32);
    assert_eq!(read_back.materialize_f64(), vec![f64::from(0.1_f32), -2.5]);
}

#[test]
fn uint64_data_array_slice_fill_and_transaction_paths_remain_exact() {
    let dir = tempfile::tempdir().expect("tempdir");
    let dataset_path = dir.path().join("uint64.data").display().to_string();
    let array = create_array(dataset_path.clone(), "uint64", vec![2, 2], vec![1, 1]);
    runmat_runtime::call_builtin(
        "DataArray.fill",
        &[array.clone(), Value::Int(IntValue::U64(u64::MAX))],
    )
    .expect("fill array");

    let slice = Value::Cell(
        CellArray::new(
            vec![Value::Int(IntValue::I32(1)), Value::String(":".to_string())],
            1,
            2,
        )
        .expect("slice"),
    );
    let replacement = Tensor::new_integer(
        IntegerStorage::U64(vec![1_u64 << 63, u64::MAX - 1]),
        vec![1, 2],
    )
    .expect("replacement");
    runmat_runtime::call_builtin(
        "DataArray.write",
        &[array.clone(), slice, Value::Tensor(replacement)],
    )
    .expect("slice write");

    let dataset = runmat_runtime::call_builtin("data.open", &[Value::String(dataset_path)])
        .expect("open dataset");
    let tx = runmat_runtime::call_builtin("Dataset.begin", &[dataset]).expect("begin transaction");
    runmat_runtime::call_builtin(
        "DataTransaction.fill",
        &[
            tx.clone(),
            Value::String("samples".to_string()),
            Value::Int(IntValue::U64(1_u64 << 63)),
        ],
    )
    .expect("queue transaction fill");
    runmat_runtime::call_builtin("DataTransaction.commit", &[tx]).expect("commit transaction");

    let Value::Tensor(read_back) =
        runmat_runtime::call_builtin("DataArray.read", &[array]).expect("read array")
    else {
        panic!("expected tensor");
    };
    assert_eq!(
        read_back.integer_storage(),
        Some(&IntegerStorage::U64(vec![1_u64 << 63; 4]))
    );
}

#[test]
fn dataset_lifecycle_preserves_wide_uint64_payloads_exactly() {
    let dir = tempfile::tempdir().expect("tempdir");
    let source_path = dir.path().join("source.data").display().to_string();
    let copy_path = dir.path().join("copy.data").display().to_string();
    let export_path = dir.path().join("export.data").display().to_string();
    let import_path = dir.path().join("import.data").display().to_string();
    let moved_path = dir.path().join("moved.data").display().to_string();
    let array = create_array(source_path.clone(), "uint64", vec![1, 2], vec![1, 2]);
    let storage = IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]);
    let input = Tensor::new_integer(storage.clone(), vec![1, 2]).expect("integer tensor");
    runmat_runtime::call_builtin("DataArray.write", &[array, Value::Tensor(input)])
        .expect("write source");

    runmat_runtime::call_builtin(
        "data.copy",
        &[Value::String(source_path), Value::String(copy_path.clone())],
    )
    .expect("copy dataset");
    runmat_runtime::call_builtin(
        "data.export",
        &[
            Value::String(copy_path),
            Value::String("data".to_string()),
            Value::String(export_path.clone()),
        ],
    )
    .expect("export dataset");
    runmat_runtime::call_builtin(
        "data.import",
        &[
            Value::String(import_path.clone()),
            Value::String("data".to_string()),
            Value::String(export_path),
        ],
    )
    .expect("import dataset");
    runmat_runtime::call_builtin(
        "data.move",
        &[
            Value::String(import_path),
            Value::String(moved_path.clone()),
        ],
    )
    .expect("move dataset");

    assert_eq!(
        runmat_runtime::call_builtin("data.exists", &[Value::String(moved_path.clone())])
            .expect("exists"),
        Value::Bool(true)
    );
    let inspected =
        runmat_runtime::call_builtin("data.inspect", &[Value::String(moved_path.clone())])
            .expect("inspect");
    assert!(matches!(inspected, Value::Struct(_)));
    let listed = runmat_runtime::call_builtin(
        "data.list",
        &[Value::String(dir.path().display().to_string())],
    )
    .expect("list");
    assert!(matches!(listed, Value::Cell(_)));

    let dataset = runmat_runtime::call_builtin("data.open", &[Value::String(moved_path.clone())])
        .expect("open moved dataset");
    let array = runmat_runtime::call_builtin(
        "Dataset.array",
        &[dataset, Value::String("samples".to_string())],
    )
    .expect("open moved array");
    let Value::Tensor(read_back) =
        runmat_runtime::call_builtin("DataArray.read", &[array]).expect("read moved array")
    else {
        panic!("expected tensor");
    };
    assert_eq!(read_back.integer_storage(), Some(&storage));

    runmat_runtime::call_builtin("data.delete", &[Value::String(moved_path.clone())])
        .expect("delete dataset");
    assert_eq!(
        runmat_runtime::call_builtin("data.exists", &[Value::String(moved_path)])
            .expect("exists after delete"),
        Value::Bool(false)
    );
}
