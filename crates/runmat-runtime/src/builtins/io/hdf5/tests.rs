use super::test_arguments::{h5read_builtin, h5write_builtin};
use futures::executor::block_on;
use runmat_value::{ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage};
use tempfile::tempdir;

use super::imp::type_name;
use super::*;

#[test]
fn h5write_and_h5read_round_trip_column_major_numeric() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("roundtrip.h5");
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

    block_on(h5write_builtin(
        None,
        Value::String(path.display().to_string().into()),
        Value::String("/group/data".to_string().into()),
        Value::Tensor(tensor),
        Vec::new(),
    ))
    .expect("write");

    let out = block_on(h5read_builtin(
        None,
        Value::String(path.display().to_string().into()),
        Value::String("/group/data".to_string().into()),
        Vec::new(),
    ))
    .expect("read");

    match out {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 3]);
            assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn hdf5_round_trips_native_single_storage() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("single_roundtrip.h5");
    let tensor = Tensor::from_f32(vec![1.25, 2.5, 3.75, 5.0], vec![2, 2]).unwrap();

    block_on(h5write_builtin(
        None,
        Value::String(path.display().to_string().into()),
        Value::String("/single".to_string().into()),
        Value::Tensor(tensor),
        Vec::new(),
    ))
    .expect("write single");

    let out = block_on(h5read_builtin(
        None,
        Value::String(path.display().to_string().into()),
        Value::String("/single".to_string().into()),
        Vec::new(),
    ))
    .expect("read single");
    let Value::Tensor(tensor) = out else {
        panic!("expected tensor");
    };
    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(
        tensor.into_numeric_storage().expect("single storage"),
        NumericStorage::F32(vec![1.25, 2.5, 3.75, 5.0])
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn hdf5_round_trips_low_rank_exact_uint64_geometry() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("low_rank_uint64.h5");

    for (dataset, shape, values) in [
        ("/rank_zero", Vec::new(), vec![u64::MAX]),
        ("/rank_one", vec![2], vec![9_007_199_254_740_993, u64::MAX]),
    ] {
        let tensor = Tensor::new_integer(IntegerStorage::U64(values.clone()), shape.clone())
            .expect("exact uint64 input");
        block_on(h5write_builtin(
            None,
            Value::String(path.display().to_string().into()),
            Value::String(dataset.to_string().into()),
            Value::Tensor(tensor),
            Vec::new(),
        ))
        .expect("write low-rank integer");
        let output = block_on(h5read_builtin(
            None,
            Value::String(path.display().to_string().into()),
            Value::String(dataset.to_string().into()),
            Vec::new(),
        ))
        .expect("read low-rank integer");
        if shape.is_empty() {
            assert_eq!(output, Value::Int(IntValue::U64(u64::MAX)));
        } else {
            let Value::Tensor(output) = output else {
                panic!("rank-one HDF5 integer must remain a tensor");
            };
            assert_eq!(output.shape(), shape.as_slice());
            assert_eq!(output.integer_storage(), Some(&IntegerStorage::U64(values)));
        }
    }
}

#[test]
fn h5write_rejects_typed_complex_integer_without_float_coercion() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("typed_complex.h5");
    let storage = IntegerComplexStorage::new(
        IntegerStorage::U64(vec![u64::MAX]),
        IntegerStorage::U64(vec![1_u64 << 63]),
    )
    .expect("matching components");
    let tensor = ComplexTensor::new_integer(storage, vec![1, 1]).expect("typed complex");

    let err = block_on(h5write_builtin(
        None,
        Value::String(path.display().to_string().into()),
        Value::String("/data".to_string().into()),
        Value::ComplexTensor(tensor),
        Vec::new(),
    ))
    .expect_err("typed complex integer HDF5 writes are unsupported");

    assert_eq!(err.identifier(), Some("RunMat:hdf5:UnsupportedType"));
}

#[test]
#[cfg(target_pointer_width = "64")]
fn hdf5_selection_vectors_read_integer_storage_exactly() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let exact = (1_u64 << 53) + 1;

    assert_eq!(
        numeric_vector_usize(&Value::Int(IntValue::U64(exact)), H5READ_NAME, "start")
            .expect("scalar vector"),
        vec![exact as usize]
    );

    let vector =
        Tensor::new_integer(IntegerStorage::U64(vec![exact, 3]), vec![1, 2]).expect("vector");
    assert_eq!(
        numeric_vector_usize(&Value::Tensor(vector), H5WRITE_NAME, "count").expect("vector"),
        vec![exact as usize, 3]
    );

    assert!(numeric_vector_usize(&Value::Int(IntValue::U8(0)), H5READ_NAME, "start").is_err());
    assert!(numeric_vector_usize(&Value::Int(IntValue::I8(-1)), H5READ_NAME, "start").is_err());
}

#[test]
fn hdf5_selection_vectors_reject_unrepresentable_double_bounds() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let boundary = numeric_vector_usize(&Value::Num(usize::MAX as f64), H5READ_NAME, "start");
    if usize::BITS == 64 {
        assert!(boundary.is_err());
    } else {
        assert_eq!(boundary.unwrap(), vec![usize::MAX]);
    }
    assert!(
        numeric_vector_usize(&Value::Num((usize::MAX as f64) + 1.0), H5READ_NAME, "start").is_err()
    );
}

#[test]
fn hdf5_round_trips_all_integer_classes_without_f64_conversion() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("integer_classes.h5");
    let cases = [
        ("i8", IntegerStorage::I8(vec![i8::MIN, -1, 0, i8::MAX])),
        ("i16", IntegerStorage::I16(vec![i16::MIN, -1, 0, i16::MAX])),
        ("i32", IntegerStorage::I32(vec![i32::MIN, -1, 0, i32::MAX])),
        ("i64", IntegerStorage::I64(vec![i64::MIN, -1, 0, i64::MAX])),
        ("u8", IntegerStorage::U8(vec![0, 1, 127, u8::MAX])),
        ("u16", IntegerStorage::U16(vec![0, 1, 32_768, u16::MAX])),
        ("u32", IntegerStorage::U32(vec![0, 1, 1 << 31, u32::MAX])),
        (
            "u64",
            IntegerStorage::U64(vec![0, 1, 1_u64 << 63, u64::MAX]),
        ),
    ];

    for (name, storage) in cases {
        let tensor = Tensor::new_integer(storage.clone(), vec![2, 2]).expect("integer tensor");
        block_on(h5write_builtin(
            None,
            Value::String(path.display().to_string().into()),
            Value::String(format!("/{name}").into()),
            Value::Tensor(tensor),
            Vec::new(),
        ))
        .expect("write integer dataset");

        let out = block_on(h5read_builtin(
            None,
            Value::String(path.display().to_string().into()),
            Value::String(format!("/{name}").into()),
            Vec::new(),
        ))
        .expect("read integer dataset");
        assert!(matches!(out, Value::Tensor(tensor) if tensor.integer_storage() == Some(&storage)));

        let file = hdf5::File::open(&path).expect("open hdf5 file");
        let dtype = file
            .dataset(format!("/{name}").as_str())
            .expect("dataset")
            .dtype()
            .and_then(|dtype| dtype.to_descriptor())
            .expect("type descriptor");
        assert_eq!(type_name(&dtype), storage.class_name());
    }
}

#[test]
fn hdf5_integer_scalar_selection_and_attribute_preserve_native_types() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("integer_paths.h5");

    block_on(h5write_builtin(
        None,
        Value::String(path.display().to_string().into()),
        Value::String("/scalar".to_string().into()),
        Value::Int(IntValue::U64(u64::MAX)),
        Vec::new(),
    ))
    .expect("write scalar");
    let scalar = block_on(h5read_builtin(
        None,
        Value::String(path.display().to_string().into()),
        Value::String("/scalar".to_string().into()),
        Vec::new(),
    ))
    .expect("read scalar");
    assert_eq!(scalar, Value::Int(IntValue::U64(u64::MAX)));

    let initial = Tensor::new_integer(IntegerStorage::I64(vec![1, 2, 3, 4]), vec![2, 2])
        .expect("initial tensor");
    block_on(h5write_builtin(
        None,
        Value::String(path.display().to_string().into()),
        Value::String("/selected".to_string().into()),
        Value::Tensor(initial),
        Vec::new(),
    ))
    .expect("write selected dataset");
    let patch = Tensor::new_integer(IntegerStorage::I64(vec![i64::MIN, i64::MAX]), vec![2, 1])
        .expect("patch tensor");
    block_on(h5write_builtin(
        None,
        Value::String(path.display().to_string().into()),
        Value::String("/selected".to_string().into()),
        Value::Tensor(patch),
        vec![
            Value::Tensor(Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap()),
        ],
    ))
    .expect("write integer selection");
    let selected = block_on(h5read_builtin(
        None,
        Value::String(path.display().to_string().into()),
        Value::String("/selected".to_string().into()),
        Vec::new(),
    ))
    .expect("read selected dataset");
    assert!(matches!(selected, Value::Tensor(tensor)
            if tensor.integer_storage() == Some(&IntegerStorage::I64(vec![i64::MIN, i64::MAX, 3, 4]))));

    block_on(h5writeatt_builtin(
        None,
        Value::String(path.display().to_string().into()),
        Value::String("/selected".to_string().into()),
        Value::String("limits".to_string().into()),
        Value::Int(IntValue::U32(u32::MAX)),
    ))
    .expect("write integer attribute");
    let file = hdf5::File::open(&path).expect("open hdf5 file");
    let dtype = file
        .dataset("/selected")
        .expect("dataset")
        .attr("limits")
        .expect("attribute")
        .dtype()
        .and_then(|dtype| dtype.to_descriptor())
        .expect("type descriptor");
    assert_eq!(type_name(&dtype), "uint32");
}

#[test]
fn h5read_supports_start_count_stride_selection() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("slice.h5");
    let tensor = Tensor::new((1..=12).map(|v| v as f64).collect(), vec![3, 4]).unwrap();
    block_on(h5write_builtin(
        None,
        Value::String(path.display().to_string().into()),
        Value::String("/data".to_string().into()),
        Value::Tensor(tensor),
        Vec::new(),
    ))
    .expect("write");

    let out = block_on(h5read_builtin(
        None,
        Value::String(path.display().to_string().into()),
        Value::String("/data".to_string().into()),
        vec![
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap()),
        ],
    ))
    .expect("slice");

    match out {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[2, 2]);
            assert_eq!(t.materialize_f64(), vec![4.0, 6.0, 7.0, 9.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn h5write_supports_start_count_stride_selection() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("partial_write.h5");
    let base = Tensor::zeros(vec![3, 4]);
    block_on(h5write_builtin(
        None,
        Value::String(path.display().to_string().into()),
        Value::String("/data".to_string().into()),
        Value::Tensor(base),
        Vec::new(),
    ))
    .expect("write base");

    let patch = Tensor::new(vec![40.0, 60.0, 70.0, 90.0], vec![2, 2]).unwrap();
    block_on(h5write_builtin(
        None,
        Value::String(path.display().to_string().into()),
        Value::String("/data".to_string().into()),
        Value::Tensor(patch),
        vec![
            Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![2.0, 2.0], vec![1, 2]).unwrap()),
            Value::Tensor(Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap()),
        ],
    ))
    .expect("partial write");

    let out = block_on(h5read_builtin(
        None,
        Value::String(path.display().to_string().into()),
        Value::String("/data".to_string().into()),
        Vec::new(),
    ))
    .expect("read");

    match out {
        Value::Tensor(t) => {
            assert_eq!(t.shape(), &[3, 4]);
            assert_eq!(
                t.materialize_f64(),
                vec![0.0, 0.0, 0.0, 40.0, 0.0, 60.0, 70.0, 0.0, 90.0, 0.0, 0.0, 0.0]
            );
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn h5write_full_existing_dataset_preserves_attributes() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("rewrite_preserves_attrs.h5");
    block_on(h5write_builtin(
        None,
        Value::String(path.display().to_string().into()),
        Value::String("/data".to_string().into()),
        Value::Tensor(Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap()),
        Vec::new(),
    ))
    .expect("initial write");
    block_on(h5writeatt_builtin(
        None,
        Value::String(path.display().to_string().into()),
        Value::String("/data".to_string().into()),
        Value::String("units".to_string().into()),
        Value::String("arb".to_string().into()),
    ))
    .expect("write attr");

    block_on(h5write_builtin(
        None,
        Value::String(path.display().to_string().into()),
        Value::String("/data".to_string().into()),
        Value::Tensor(Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap()),
        Vec::new(),
    ))
    .expect("rewrite existing");

    let out = block_on(h5read_builtin(
        None,
        Value::String(path.display().to_string().into()),
        Value::String("/data".to_string().into()),
        Vec::new(),
    ))
    .expect("read rewritten");
    assert!(matches!(out, Value::Tensor(t) if t.materialize_f64() == vec![3.0, 4.0]));

    let info = block_on(h5info_builtin(
        Value::String(path.display().to_string().into()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
            "/data".to_string().into(),
        )]),
    ))
    .expect("dataset info");
    let Value::Struct(info) = info else {
        panic!("expected dataset info struct");
    };
    let Value::Cell(attributes) = info.fields.get("Attributes").unwrap() else {
        panic!("expected attributes cell");
    };
    assert!(attributes.data().iter().any(|value| {
            matches!(value, Value::Struct(attr) if attr.fields.get("Name") == Some(&Value::String("units".to_string().into())))
        }));
}

#[test]
fn h5writeatt_h5info_and_h5disp_report_metadata() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("info.h5");
    block_on(h5write_builtin(
        None,
        Value::String(path.display().to_string().into()),
        Value::String("/data".to_string().into()),
        Value::Bool(true),
        Vec::new(),
    ))
    .expect("write");
    block_on(h5writeatt_builtin(
        None,
        Value::String(path.display().to_string().into()),
        Value::String("/data".to_string().into()),
        Value::String("units".to_string().into()),
        Value::String("arb".to_string().into()),
    ))
    .expect("write attr");

    let info = block_on(h5info_builtin(
        Value::String(path.display().to_string().into()),
        crate::builtins::common::test_support::builtin_arguments(Vec::new()),
    ))
    .expect("info");
    match info {
        Value::Struct(s) => {
            assert!(s.fields.contains_key("Groups"));
            assert!(s.fields.contains_key("Datasets"));
        }
        other => panic!("expected struct, got {other:?}"),
    }

    let text = block_on(h5disp_builtin(
        Value::String(path.display().to_string().into()),
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
            "/".to_string().into(),
        )]),
    ))
    .expect("disp");
    match text {
        Value::String(runmat_value::StringElement::Text(s)) => {
            assert!(s.contains("Dataset: /data"));
            assert!(s.contains("Attribute: units"));
        }
        other => panic!("expected string, got {other:?}"),
    }
}

#[test]
fn h5read_rejects_unsupported_compound_dataset() {
    let _fs_lock = runmat_filesystem::provider_override_lock();
    #[derive(hdf5::H5Type, Clone, Copy)]
    #[repr(C)]
    #[allow(non_local_definitions)]
    struct Pair {
        a: f64,
        b: f64,
    }

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("compound.h5");
    let file = hdf5::File::create(&path).expect("file");
    file.new_dataset_builder()
        .with_data(&[Pair { a: 1.0, b: 2.0 }])
        .create("/compound")
        .expect("dataset");

    let err = block_on(h5read_builtin(
        None,
        Value::String(path.display().to_string().into()),
        Value::String("/compound".to_string().into()),
        Vec::new(),
    ))
    .expect_err("compound unsupported");
    assert_eq!(err.identifier(), Some("RunMat:hdf5:UnsupportedType"));
}
