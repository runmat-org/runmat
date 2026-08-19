use futures::executor::block_on;
use runmat_value::{CharArray, IntegerStorage, NumericDType, SparseTensor, Tensor, Value};

#[test]
fn zeros_single_uses_f32_dtype() {
    let result = runmat_runtime::call_builtin(
        "zeros",
        &[
            Value::Num(2.0),
            Value::Num(3.0),
            Value::String("single".into()),
        ],
    )
    .expect("zeros single");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![2, 3]);
            assert_eq!(t.numeric_dtype(), NumericDType::F32);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn ones_single_uses_f32_dtype() {
    let result = runmat_runtime::call_builtin(
        "ones",
        &[
            Value::Num(3.0),
            Value::Num(4.0),
            Value::String("single".into()),
        ],
    )
    .expect("ones single");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![3, 4]);
            assert_eq!(t.numeric_dtype(), NumericDType::F32);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn zeros_like_proto_preserves_numeric_dtype() {
    let proto = Tensor::new_with_dtype(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], NumericDType::F32)
        .expect("proto tensor");
    let result = runmat_runtime::call_builtin(
        "zeros",
        &[Value::String("like".into()), Value::Tensor(proto.clone())],
    )
    .expect("zeros like proto");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, proto.shape);
            assert_eq!(t.numeric_dtype(), NumericDType::F32);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn zeros_like_sparse_proto_preserves_sparse_storage() {
    let proto =
        SparseTensor::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 2.0]).expect("sparse proto");
    let result = runmat_runtime::call_builtin(
        "zeros",
        &[
            Value::Num(3.0),
            Value::Num(4.0),
            Value::String("like".into()),
            Value::SparseTensor(proto),
        ],
    )
    .expect("zeros like sparse proto");
    match result {
        Value::SparseTensor(sparse) => {
            assert_eq!(sparse.shape(), vec![3, 4]);
            assert_eq!(sparse.nnz(), 0);
            assert_eq!(sparse.col_ptrs, vec![0, 0, 0, 0, 0]);
            assert!(sparse.row_indices.is_empty());
            assert!(sparse.as_f64_slice().is_some_and(<[f64]>::is_empty));
        }
        other => panic!("expected sparse result, got {other:?}"),
    }
}

#[test]
fn zeros_like_typed_sparse_proto_preserves_integer_class() {
    let cases = vec![
        IntegerStorage::I8(vec![i8::MIN, i8::MAX]),
        IntegerStorage::I16(vec![i16::MIN, i16::MAX]),
        IntegerStorage::I32(vec![i32::MIN, i32::MAX]),
        IntegerStorage::I64(vec![i64::MIN, i64::MAX]),
        IntegerStorage::U8(vec![1, u8::MAX]),
        IntegerStorage::U16(vec![1, u16::MAX]),
        IntegerStorage::U32(vec![1, u32::MAX]),
        IntegerStorage::U64(vec![1, u64::MAX]),
    ];

    for storage in cases {
        let proto = SparseTensor::new_integer(2, 2, vec![0, 1, 2], vec![0, 1], storage.clone())
            .expect("typed sparse prototype");
        let result = runmat_runtime::call_builtin(
            "zeros",
            &[
                Value::Num(3.0),
                Value::Num(4.0),
                Value::String("like".into()),
                Value::SparseTensor(proto),
            ],
        )
        .expect("zeros like typed sparse prototype");

        match result {
            Value::SparseTensor(sparse) => {
                assert_eq!(sparse.shape(), vec![3, 4]);
                assert_eq!(sparse.nnz(), 0);
                assert_eq!(sparse.col_ptrs, vec![0, 0, 0, 0, 0]);
                assert!(sparse.row_indices.is_empty());
                assert_eq!(sparse.nnz(), 0);
                assert_eq!(
                    sparse.integer_storage(),
                    Some(&storage.zeros_like(0)),
                    "class {}",
                    storage.class_name()
                );
            }
            other => panic!("expected sparse result, got {other:?}"),
        }
    }

    let proto = SparseTensor::new_integer(
        2,
        5,
        vec![0, 1, 2, 2, 2, 2],
        vec![0, 1],
        IntegerStorage::U64(vec![1, u64::MAX]),
    )
    .expect("uint64 sparse prototype");
    let result = runmat_runtime::call_builtin(
        "zeros",
        &[Value::String("like".into()), Value::SparseTensor(proto)],
    )
    .expect("zeros like typed sparse prototype shape");
    match result {
        Value::SparseTensor(sparse) => {
            assert_eq!(sparse.shape(), vec![2, 5]);
            assert_eq!(sparse.col_ptrs, vec![0, 0, 0, 0, 0, 0]);
            assert_eq!(sparse.integer_storage(), Some(&IntegerStorage::U64(vec![])));
        }
        other => panic!("expected sparse result, got {other:?}"),
    }
}

#[test]
fn typed_sparse_scalar_indexing_preserves_all_integer_classes() {
    let cases = vec![
        IntegerStorage::I8(vec![i8::MIN]),
        IntegerStorage::I16(vec![i16::MIN]),
        IntegerStorage::I32(vec![i32::MIN]),
        IntegerStorage::I64(vec![i64::MIN]),
        IntegerStorage::U8(vec![u8::MAX]),
        IntegerStorage::U16(vec![u16::MAX]),
        IntegerStorage::U32(vec![u32::MAX]),
        IntegerStorage::U64(vec![u64::MAX]),
    ];

    for storage in cases {
        let sparse = SparseTensor::new_integer(2, 2, vec![0, 1, 1], vec![0], storage.clone())
            .expect("typed sparse");
        for indices in [&[1.0][..], &[1.0, 1.0][..], &[2.0, 2.0][..]] {
            let result = block_on(
                runmat_runtime::builtins::common::indexing::perform_indexing(
                    &Value::SparseTensor(sparse.clone()),
                    indices,
                ),
            )
            .expect("sparse scalar index");
            let Value::SparseTensor(result) = result else {
                panic!("expected sparse scalar result");
            };
            let expected = if indices == [1.0] || indices == [1.0, 1.0] {
                storage.clone()
            } else {
                storage.zeros_like(0)
            };
            assert_eq!(result.integer_storage(), Some(&expected));
        }
    }
}

#[test]
fn sparse_full_and_nonzeros_preserve_exact_integer_storage() {
    let cases = vec![
        IntegerStorage::I8(vec![0, i8::MIN, i8::MAX, 0]),
        IntegerStorage::I16(vec![0, i16::MIN, i16::MAX, 0]),
        IntegerStorage::I32(vec![0, i32::MIN, i32::MAX, 0]),
        IntegerStorage::I64(vec![0, i64::MIN, i64::MAX, 0]),
        IntegerStorage::U8(vec![0, 1, u8::MAX, 0]),
        IntegerStorage::U16(vec![0, 1, u16::MAX, 0]),
        IntegerStorage::U32(vec![0, 1, u32::MAX, 0]),
        IntegerStorage::U64(vec![0, 1, u64::MAX, 0]),
    ];

    for storage in cases {
        let source = Tensor::new_integer(storage.clone(), vec![2, 2]).expect("integer tensor");
        let sparse = match runmat_runtime::call_builtin("sparse", &[Value::Tensor(source)])
            .expect("sparse integer tensor")
        {
            Value::SparseTensor(sparse) => sparse,
            other => panic!("expected sparse tensor, got {other:?}"),
        };
        assert_eq!(
            sparse.integer_storage().map(IntegerStorage::class_name),
            Some(storage.class_name())
        );
        assert_eq!(sparse.integer_at(1, 0), storage.value_at(1));
        assert_eq!(sparse.integer_at(0, 1), storage.value_at(2));
        assert_eq!(sparse.integer_at(0, 0), None);
        assert_eq!(sparse.integer_at(1, 1), None);

        let full =
            match runmat_runtime::call_builtin("full", &[Value::SparseTensor(sparse.clone())])
                .expect("full integer sparse")
            {
                Value::Tensor(tensor) => tensor,
                other => panic!("expected tensor, got {other:?}"),
            };
        assert_eq!(full.integer_storage(), Some(&storage));

        let nonzeros =
            match runmat_runtime::call_builtin("nonzeros", &[Value::SparseTensor(sparse)])
                .expect("nonzeros integer sparse")
            {
                Value::Tensor(tensor) => tensor,
                other => panic!("expected tensor, got {other:?}"),
            };
        assert_eq!(
            nonzeros.integer_storage().map(IntegerStorage::class_name),
            Some(storage.class_name())
        );
    }
}

#[test]
fn sparse_triplets_preserve_integer_values_and_saturate_duplicates() {
    let rows = Tensor::new(vec![1.0, 1.0, 2.0], vec![3, 1]).expect("row subscripts");
    let cols = Tensor::new(vec![1.0, 1.0, 2.0], vec![3, 1]).expect("column subscripts");
    let values = Tensor::new_integer(IntegerStorage::I8(vec![100, 100, 0]), vec![3, 1])
        .expect("int8 values");

    let sparse = match runmat_runtime::call_builtin(
        "sparse",
        &[
            Value::Tensor(rows),
            Value::Tensor(cols),
            Value::Tensor(values),
        ],
    )
    .expect("integer sparse triplets")
    {
        Value::SparseTensor(sparse) => sparse,
        other => panic!("expected sparse tensor, got {other:?}"),
    };
    assert_eq!(sparse.shape(), vec![2, 2]);
    assert_eq!(
        sparse.integer_storage(),
        Some(&IntegerStorage::I8(vec![i8::MAX]))
    );

    let dense = match runmat_runtime::call_builtin("full", &[Value::SparseTensor(sparse)])
        .expect("full integer triplets")
    {
        Value::Tensor(tensor) => tensor,
        other => panic!("expected tensor, got {other:?}"),
    };
    assert_eq!(
        dense.integer_storage(),
        Some(&IntegerStorage::I8(vec![i8::MAX, 0, 0, 0]))
    );
}

#[test]
fn sparse_triplets_preserve_uint64_and_expand_integer_scalars() {
    let rows = Tensor::new(vec![1.0, 1.0], vec![2, 1]).expect("row subscripts");
    let cols = Tensor::new(vec![1.0, 1.0], vec![2, 1]).expect("column subscripts");
    let values = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX, 1]), vec![2, 1])
        .expect("uint64 values");
    let sparse = match runmat_runtime::call_builtin(
        "sparse",
        &[
            Value::Tensor(rows),
            Value::Tensor(cols),
            Value::Tensor(values),
        ],
    )
    .expect("uint64 sparse triplets")
    {
        Value::SparseTensor(sparse) => sparse,
        other => panic!("expected sparse tensor, got {other:?}"),
    };
    assert_eq!(
        sparse.integer_storage(),
        Some(&IntegerStorage::U64(vec![u64::MAX]))
    );

    let rows = Tensor::new_integer(IntegerStorage::I16(vec![1, 2]), vec![2, 1])
        .expect("integer row subscripts");
    let cols = Value::Int(runmat_value::IntValue::I16(1));
    let sparse = match runmat_runtime::call_builtin(
        "sparse",
        &[
            Value::Tensor(rows),
            cols,
            Value::Int(runmat_value::IntValue::I16(3)),
        ],
    )
    .expect("expanded integer scalar sparse triplets")
    {
        Value::SparseTensor(sparse) => sparse,
        other => panic!("expected sparse tensor, got {other:?}"),
    };
    assert_eq!(
        sparse.integer_storage(),
        Some(&IntegerStorage::I16(vec![3, 3]))
    );
}

#[test]
fn sparse_triplets_accept_matrix_subscripts_and_integer_values() {
    let rows = Tensor::new(vec![1.0, 2.0, 1.0, 2.0], vec![2, 2]).expect("row subscripts");
    let cols = Tensor::new(vec![1.0, 1.0, 2.0, 2.0], vec![2, 2]).expect("column subscripts");
    let values = Tensor::new_integer(IntegerStorage::U32(vec![1, 2, 3, u32::MAX]), vec![2, 2])
        .expect("uint32 values");

    let sparse = match runmat_runtime::call_builtin(
        "sparse",
        &[
            Value::Tensor(rows),
            Value::Tensor(cols),
            Value::Tensor(values),
        ],
    )
    .expect("matrix integer sparse triplets")
    {
        Value::SparseTensor(sparse) => sparse,
        other => panic!("expected sparse tensor, got {other:?}"),
    };

    assert_eq!(
        sparse.integer_storage(),
        Some(&IntegerStorage::U32(vec![1, 2, 3, u32::MAX]))
    );
    assert_eq!(
        sparse
            .to_dense()
            .expect("dense matrix triplets")
            .integer_storage(),
        Some(&IntegerStorage::U32(vec![1, 2, 3, u32::MAX]))
    );
}

#[test]
fn transpose_preserves_exact_sparse_integer_storage() {
    let sparse = SparseTensor::new_integer(
        3,
        2,
        vec![0, 2, 3],
        vec![0, 2, 1],
        IntegerStorage::U64(vec![u64::MAX, 7, 9]),
    )
    .expect("uint64 sparse");

    let transposed = match runmat_runtime::call_builtin("transpose", &[Value::SparseTensor(sparse)])
        .expect("transpose sparse")
    {
        Value::SparseTensor(sparse) => sparse,
        other => panic!("expected sparse tensor, got {other:?}"),
    };
    assert_eq!(transposed.shape(), vec![2, 3]);
    assert_eq!(
        transposed.integer_storage(),
        Some(&IntegerStorage::U64(vec![u64::MAX, 9, 7]))
    );
}

#[test]
fn randn_single_sets_f32_dtype() {
    let result = runmat_runtime::call_builtin(
        "randn",
        &[
            Value::Num(4.0),
            Value::Num(5.0),
            Value::String("single".into()),
        ],
    )
    .expect("randn single");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, vec![4, 5]);
            assert_eq!(t.numeric_dtype(), NumericDType::F32);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn randn_like_proto_preserves_dtype() {
    let proto = Tensor::new_with_dtype(vec![0.0, 0.0, 0.0], vec![3, 1], NumericDType::F32)
        .expect("proto tensor");
    let result = runmat_runtime::call_builtin(
        "randn",
        &[Value::String("like".into()), Value::Tensor(proto.clone())],
    )
    .expect("randn like");
    match result {
        Value::Tensor(t) => {
            assert_eq!(t.shape, proto.shape);
            assert_eq!(t.numeric_dtype(), NumericDType::F32);
        }
        other => panic!("expected tensor result, got {other:?}"),
    }
}

#[test]
fn gpu_array_single_roundtrip_preserves_dtype() {
    runmat_accelerate::simple_provider::register_inprocess_provider();
    let host = Tensor::new_with_dtype(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], NumericDType::F32)
        .expect("host tensor");
    let gpu = runmat_runtime::call_builtin(
        "gpuArray",
        &[
            Value::Tensor(host.clone()),
            Value::CharArray(CharArray::new_row("single")),
        ],
    )
    .expect("gpuArray single upload");
    if let Value::GpuTensor(ref handle) = gpu {
        let expected_handle_precision = runmat_accelerate_api::ProviderPrecision::F32;
        let precision =
            runmat_accelerate_api::handle_precision(handle).unwrap_or(expected_handle_precision);
        assert_eq!(precision, expected_handle_precision);
        let expected_dtype = match precision {
            runmat_accelerate_api::ProviderPrecision::F32 => NumericDType::F32,
            runmat_accelerate_api::ProviderPrecision::F64 => NumericDType::F64,
        };
        let gathered = runmat_runtime::dispatcher::gather_if_needed(&gpu).expect("gather single");
        match gathered {
            Value::Tensor(t) => {
                assert_eq!(t.shape, host.shape);
                assert_eq!(t.numeric_dtype(), expected_dtype);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }

        let direct_eval = block_on(
            runmat_runtime::builtins::acceleration::gpu::gather::evaluate(std::slice::from_ref(
                &gpu,
            )),
        )
        .expect("gather eval")
        .into_first();
        match direct_eval {
            Value::Tensor(t) => {
                assert_eq!(t.shape, host.shape);
                assert_eq!(t.numeric_dtype(), expected_dtype);
            }
            other => panic!("expected tensor from gather::evaluate, got {other:?}"),
        }

        let builtin_gathered = runmat_runtime::call_builtin("gather", std::slice::from_ref(&gpu))
            .expect("gather builtin");
        match builtin_gathered {
            Value::Tensor(t) => {
                assert_eq!(t.shape, host.shape);
                assert_eq!(t.numeric_dtype(), expected_dtype);
            }
            other => panic!("expected tensor result from builtin gather, got {other:?}"),
        }
    } else {
        panic!("expected gpu tensor");
    }
}
