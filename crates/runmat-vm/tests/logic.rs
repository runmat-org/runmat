#[path = "support/mod.rs"]
mod test_helpers;

use test_helpers::execute_source;

use runmat_builtins::{IntValue, IntegerStorage, Value};

fn logical_truth(value: &Value) -> bool {
    match value {
        Value::Bool(value) => *value,
        Value::Num(value) => *value != 0.0,
        other => panic!("expected logical value, got {other:?}"),
    }
}

fn sparse_scalar(value: &Value) -> f64 {
    match value {
        Value::SparseTensor(sparse) if sparse.shape() == vec![1, 1] => {
            sparse.get(0, 0).unwrap_or(0.0)
        }
        other => panic!("expected sparse scalar value, got {other:?}"),
    }
}

#[test]
fn logical_operators_and_short_circuit() {
    let vars =
        execute_source("a = 0 && (1/0); b = 1 || (1/0); c = 0 & 5; d = 0 | 5; e = ~0; f = ~5;")
            .unwrap();
    assert!(!logical_truth(&vars[0]));
    assert!(logical_truth(&vars[1]));
    assert!(!logical_truth(&vars[2]));
    assert!(logical_truth(&vars[3]));
    assert!(logical_truth(&vars[4]));
    assert!(!logical_truth(&vars[5]));
}

#[test]
fn short_circuit_or_accepts_boolean_lhs_without_numeric_coercion() {
    let vars = execute_source(
        "tau = []; flight_duration = 10; guard = isempty(tau) || tau(end) < flight_duration;",
    )
    .unwrap();
    assert!(logical_truth(&vars[2]));
}

#[test]
fn integer_scalar_arithmetic_keeps_int64_and_uint64_exact_through_vm_dispatch() {
    let vars = execute_source(
        "u = uint64(9223372036854775808); up = u + 1; down = uint64(18446744073709551615) - 1; lo = int64(-9223372036854775808) + 1; reverse = 1 - int64(-9223372036854775808); same = up .* 1; row = uint64([9223372036854775808 18446744073709551615]); rowdown = row - 1;",
    )
    .expect("integer arithmetic should execute");

    assert_eq!(vars[0], Value::Int(IntValue::U64(1_u64 << 63)));
    assert_eq!(vars[1], Value::Int(IntValue::U64((1_u64 << 63) + 1)));
    assert_eq!(vars[2], Value::Int(IntValue::U64(u64::MAX - 1)));
    assert_eq!(vars[3], Value::Int(IntValue::I64(i64::MIN + 1)));
    assert_eq!(vars[4], Value::Int(IntValue::I64(i64::MAX)));
    assert_eq!(vars[5], Value::Int(IntValue::U64((1_u64 << 63) + 1)));
    assert!(matches!(
        &vars[7],
        Value::Tensor(tensor)
            if tensor.shape == vec![1, 2]
                && tensor.integer_storage()
                    == Some(&IntegerStorage::U64(vec![(1_u64 << 63) - 1, u64::MAX - 1]))
    ));
}

#[test]
fn complex_integer_values_preserve_exact_components_through_vm_dispatch() {
    let vars = execute_source(
        "r = uint64([9223372036854775808 18446744073709551615]); z = complex(r, 1); zr = real(z); zi = imag(z); scalar = complex(int64(-9223372036854775808), int64(7)); sr = real(scalar); si = imag(scalar); tf = isreal(z); high = uint64(9223372036854775808) + 1; highz = complex(high, 1); highr = real(highz); picked = z([2 1]); pickedreal = real(picked); reshaped = reshape(z, 2, 1); reshapedreal = real(reshaped); scalarreshape = reshape(high, 1, 1, 1);",
    )
    .expect("integer complex construction should execute");

    assert!(matches!(
        &vars[1],
        Value::ComplexTensor(tensor)
            if tensor.shape == vec![1, 2]
                && tensor.integer_data.as_ref().map(|storage| (&storage.real, &storage.imag))
                    == Some((
                        &IntegerStorage::U64(vec![9_223_372_036_854_775_808, u64::MAX]),
                        &IntegerStorage::U64(vec![1, 1]),
                    ))
    ));
    assert!(matches!(
        &vars[2],
        Value::Tensor(tensor)
            if tensor.integer_storage()
                == Some(&IntegerStorage::U64(vec![9_223_372_036_854_775_808, u64::MAX]))
    ));
    assert!(matches!(
        &vars[3],
        Value::Tensor(tensor)
            if tensor.integer_storage() == Some(&IntegerStorage::U64(vec![1, 1]))
    ));
    assert!(matches!(
        &vars[4],
        Value::ComplexTensor(tensor)
            if tensor.integer_data.as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::I64(vec![i64::MIN]),
                    &IntegerStorage::I64(vec![7]),
                ))
    ));
    assert_eq!(vars[5], Value::Int(IntValue::I64(i64::MIN)));
    assert_eq!(vars[6], Value::Int(IntValue::I64(7)));
    assert!(!logical_truth(&vars[7]));
    assert_eq!(
        vars[10],
        Value::Int(IntValue::U64(9_223_372_036_854_775_809))
    );
    assert!(matches!(
        &vars[11],
        Value::ComplexTensor(tensor)
            if tensor.integer_data.as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::U64(vec![u64::MAX, 9_223_372_036_854_775_808]),
                    &IntegerStorage::U64(vec![1, 1]),
                ))
    ));
    assert!(matches!(
        &vars[12],
        Value::Tensor(tensor)
            if tensor.integer_storage()
                == Some(&IntegerStorage::U64(vec![u64::MAX, 9_223_372_036_854_775_808]))
    ));
    assert!(matches!(
        &vars[13],
        Value::ComplexTensor(tensor)
            if tensor.shape == vec![2, 1]
                && tensor.integer_data.as_ref().map(|storage| (&storage.real, &storage.imag))
                    == Some((
                        &IntegerStorage::U64(vec![9_223_372_036_854_775_808, u64::MAX]),
                        &IntegerStorage::U64(vec![1, 1]),
                    ))
    ));
    assert!(matches!(
        &vars[14],
        Value::Tensor(tensor)
            if tensor.shape == vec![2, 1]
                && tensor.integer_storage()
                    == Some(&IntegerStorage::U64(vec![9_223_372_036_854_775_808, u64::MAX]))
    ));
    assert!(matches!(
        &vars[15],
        Value::Tensor(tensor)
            if tensor.shape == vec![1, 1, 1]
                && tensor.integer_storage()
                    == Some(&IntegerStorage::U64(vec![9_223_372_036_854_775_809]))
    ));
}

#[test]
fn complex_integer_slice_assignment_preserves_exact_components_through_vm_dispatch() {
    let vars = execute_source(
        "a = complex(uint64([1 2; 3 4]), uint64([10 20; 30 40])); rhs = complex(uint64([18446744073709551615 9223372036854775808]), uint64([7 8])); a(:, :) = rhs; ar = real(a); ai = imag(a); b = complex(uint64([1 2; 3 4]), uint64([10 20; 30 40])); b(1:end, :) = rhs;",
    )
    .expect("typed complex slice assignment should execute");

    assert!(matches!(
        &vars[0],
        Value::ComplexTensor(tensor)
            if tensor.integer_data.as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::U64(vec![u64::MAX, u64::MAX, 9_223_372_036_854_775_808, 9_223_372_036_854_775_808]),
                    &IntegerStorage::U64(vec![7, 7, 8, 8]),
                ))
    ));
    assert!(matches!(
        &vars[2],
        Value::Tensor(tensor)
            if tensor.integer_storage()
                == Some(&IntegerStorage::U64(vec![u64::MAX, u64::MAX, 9_223_372_036_854_775_808, 9_223_372_036_854_775_808]))
    ));
    assert!(matches!(
        &vars[3],
        Value::Tensor(tensor)
            if tensor.integer_storage() == Some(&IntegerStorage::U64(vec![7, 7, 8, 8]))
    ));
    assert!(matches!(
        &vars[4],
        Value::ComplexTensor(tensor)
            if tensor.integer_data.as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::U64(vec![u64::MAX, u64::MAX, 9_223_372_036_854_775_808, 9_223_372_036_854_775_808]),
                    &IntegerStorage::U64(vec![7, 7, 8, 8]),
                ))
    ));
}

#[test]
fn complex_integer_shape_transforms_preserve_exact_components_through_vm_dispatch() {
    let vars = execute_source(
        "a = complex(uint64(reshape([9223372036854775808 18446744073709551615 3 4], 2, 2)), uint64(reshape([7 8 9 10], 2, 2))); p = permute(a, [2 1]); q = ipermute(p, [2 1]); r = repmat(a, 2, 2); s = squeeze(reshape(a, 1, 2, 2, 1)); qr = real(q); rr = real(r); sr = real(s);",
    )
    .expect("typed complex integer shape transforms should execute");

    assert!(matches!(
        &vars[2],
        Value::ComplexTensor(tensor)
            if tensor.integer_data.as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::U64(vec![9_223_372_036_854_775_808, u64::MAX, 3, 4]),
                    &IntegerStorage::U64(vec![7, 8, 9, 10]),
                ))
    ));
    assert!(matches!(
        &vars[3],
        Value::ComplexTensor(tensor)
            if tensor.shape == vec![4, 4]
                && tensor.integer_data.as_ref().map(|storage| (&storage.real, &storage.imag))
                    == Some((
                        &IntegerStorage::U64(vec![
                            9_223_372_036_854_775_808,
                            u64::MAX,
                            9_223_372_036_854_775_808,
                            u64::MAX,
                            3,
                            4,
                            3,
                            4,
                            9_223_372_036_854_775_808,
                            u64::MAX,
                            9_223_372_036_854_775_808,
                            u64::MAX,
                            3,
                            4,
                            3,
                            4,
                        ]),
                        &IntegerStorage::U64(vec![7, 8, 7, 8, 9, 10, 9, 10, 7, 8, 7, 8, 9, 10, 9, 10]),
                    ))
    ));
    assert!(matches!(
        &vars[4],
        Value::ComplexTensor(tensor)
            if tensor.shape == vec![2, 2]
                && tensor.integer_data.as_ref().map(|storage| (&storage.real, &storage.imag))
                    == Some((
                        &IntegerStorage::U64(vec![9_223_372_036_854_775_808, u64::MAX, 3, 4]),
                        &IntegerStorage::U64(vec![7, 8, 9, 10]),
                    ))
    ));
}

#[test]
fn integer_casts_preserve_complex_storage_for_every_integer_class_through_vm_dispatch() {
    let vars = execute_source(
        "z = complex([1.5 -2.5], [0.49 -1.5]); a = int8(z); b = int16(z); c = int32(z); d = int64(z); e = uint8(z); f = uint16(z); g = uint32(z); h = uint64(z); flags = [isreal(a) isreal(b) isreal(c) isreal(d) isreal(e) isreal(f) isreal(g) isreal(h)]; q = complex(uint64([9223372036854775808 18446744073709551615]), uint64([1 2])); q64 = int64(q);",
    )
    .expect("complex integer casts should execute");

    let expected = vec![
        (
            IntegerStorage::I8(vec![2, -3]),
            IntegerStorage::I8(vec![0, -2]),
        ),
        (
            IntegerStorage::I16(vec![2, -3]),
            IntegerStorage::I16(vec![0, -2]),
        ),
        (
            IntegerStorage::I32(vec![2, -3]),
            IntegerStorage::I32(vec![0, -2]),
        ),
        (
            IntegerStorage::I64(vec![2, -3]),
            IntegerStorage::I64(vec![0, -2]),
        ),
        (
            IntegerStorage::U8(vec![2, 0]),
            IntegerStorage::U8(vec![0, 0]),
        ),
        (
            IntegerStorage::U16(vec![2, 0]),
            IntegerStorage::U16(vec![0, 0]),
        ),
        (
            IntegerStorage::U32(vec![2, 0]),
            IntegerStorage::U32(vec![0, 0]),
        ),
        (
            IntegerStorage::U64(vec![2, 0]),
            IntegerStorage::U64(vec![0, 0]),
        ),
    ];
    for (value, (real, imag)) in vars[1..9].iter().zip(expected) {
        let Value::ComplexTensor(tensor) = value else {
            panic!("integer cast must preserve complex tensor storage: {value:?}");
        };
        assert_eq!(tensor.shape, vec![1, 2]);
        assert_eq!(
            tensor
                .integer_data
                .as_ref()
                .map(|storage| (&storage.real, &storage.imag)),
            Some((&real, &imag))
        );
    }
    assert!(matches!(
        &vars[9],
        Value::LogicalArray(flags) if flags.data == vec![0; 8]
    ));
    assert!(matches!(
        &vars[11],
        Value::ComplexTensor(tensor)
            if tensor.integer_data.as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::I64(vec![i64::MAX, i64::MAX]),
                    &IntegerStorage::I64(vec![1, 2]),
                ))
    ));
}

#[test]
fn conj_preserves_typed_complex_storage_through_vm_dispatch() {
    let vars = execute_source(
        "a = complex(int8([1 -2]), int8([3 -128])); b = conj(a); u = complex(uint64([9223372036854775808 18446744073709551615]), uint64([1 2])); v = conj(u); z = conj(complex(uint16(7), uint16(0))); tf = [isreal(b) isreal(v) isreal(z)]; w = conj(complex(7)); tw = isreal(w);",
    )
    .expect("typed complex conjugates should execute");

    assert!(matches!(
        &vars[1],
        Value::ComplexTensor(tensor)
            if tensor.integer_data.as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::I8(vec![1, -2]),
                    &IntegerStorage::I8(vec![-3, i8::MAX]),
                ))
    ));
    assert!(matches!(
        &vars[3],
        Value::ComplexTensor(tensor)
            if tensor.integer_data.as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]),
                    &IntegerStorage::U64(vec![0, 0]),
                ))
    ));
    assert!(matches!(
        &vars[4],
        Value::ComplexTensor(tensor)
            if tensor.integer_data.as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::U16(vec![7]),
                    &IntegerStorage::U16(vec![0]),
                ))
    ));
    assert!(matches!(
        &vars[5],
        Value::LogicalArray(flags) if flags.data == vec![0; 3]
    ));
    assert!(matches!(&vars[6], Value::Complex(re, im) if *re == 7.0 && *im == 0.0));
    assert!(!logical_truth(&vars[7]));
}

#[test]
fn typed_complex_integer_deletion_preserves_paired_exact_storage_through_vm_dispatch() {
    let vars = execute_source(
        "a = complex(uint64([9223372036854775808 2 18446744073709551615]), uint64([7 8 9])); a(2) = []; b = complex(int16([1 2 3 4]), int16([-1 -2 -3 -4])); b([4 2]) = [];",
    )
    .expect("typed complex integer deletion should execute");

    assert!(matches!(
        &vars[0],
        Value::ComplexTensor(tensor)
            if tensor.shape == vec![1, 2]
                && tensor.integer_data.as_ref().map(|storage| (&storage.real, &storage.imag))
                    == Some((
                        &IntegerStorage::U64(vec![1_u64 << 63, u64::MAX]),
                        &IntegerStorage::U64(vec![7, 9]),
                    ))
    ));
    assert!(matches!(
        &vars[1],
        Value::ComplexTensor(tensor)
            if tensor.shape == vec![1, 2]
                && tensor.integer_data.as_ref().map(|storage| (&storage.real, &storage.imag))
                    == Some((
                        &IntegerStorage::I16(vec![1, 3]),
                        &IntegerStorage::I16(vec![-1, -3]),
                    ))
    ));
}

#[test]
fn typed_complex_integer_scalar_assignment_preserves_paired_exact_storage_through_vm_dispatch() {
    let vars = execute_source(
        "a = complex(uint64([9223372036854775808 2]), uint64([7 8])); a(2) = complex(uint64(18446744073709551615), uint64(3)); a(3) = complex(4, 5); b = complex(int8([1 2; 3 4]), int8([-1 -2; -3 -4])); b(2, 1) = complex(int8(-128), int8(127));",
    )
    .expect("typed complex integer scalar assignment should execute");

    assert!(matches!(
        &vars[0],
        Value::ComplexTensor(tensor)
            if tensor.shape == vec![1, 3]
                && tensor.integer_data.as_ref().map(|storage| (&storage.real, &storage.imag))
                    == Some((
                        &IntegerStorage::U64(vec![1_u64 << 63, u64::MAX, 4]),
                        &IntegerStorage::U64(vec![7, 3, 5]),
                    ))
    ));
    assert!(matches!(
        &vars[1],
        Value::ComplexTensor(tensor)
            if tensor.integer_data.as_ref().map(|storage| (&storage.real, &storage.imag))
                == Some((
                    &IntegerStorage::I8(vec![1, i8::MIN, 2, 4]),
                    &IntegerStorage::I8(vec![-1, i8::MAX, -2, -4]),
                ))
    ));
}

#[test]
fn issparse_reports_sparse_storage_through_vm_dispatch() {
    let vars = execute_source(
        "s = sparse([1 2], [1 2], [10 20], 2, 2); a = issparse(s); b = issparse([10 0; 0 20]); c = issparse(42);",
    )
    .unwrap();
    assert!(logical_truth(&vars[1]));
    assert!(!logical_truth(&vars[2]));
    assert!(!logical_truth(&vars[3]));
}

#[test]
fn full_densifies_sparse_storage_through_vm_dispatch() {
    let vars = execute_source(
        "s = sparse([1 3 2], [1 1 2], [10 30 20], 3, 2); a = full(s); b = full([1 0; 0 2]); c = issparse(a);",
    )
    .unwrap();
    assert!(matches!(
        &vars[1],
        Value::Tensor(tensor)
            if tensor.shape == vec![3, 2]
                && tensor.data == vec![10.0, 0.0, 30.0, 0.0, 20.0, 0.0]
    ));
    assert!(matches!(
        &vars[2],
        Value::Tensor(tensor)
            if tensor.shape == vec![2, 2]
                && tensor.data == vec![1.0, 0.0, 0.0, 2.0]
    ));
    assert!(!logical_truth(&vars[3]));
}

#[test]
fn sparse_indexing_reads_stored_unstored_and_column_major_values() {
    let vars = execute_source(
        "s = sparse([1 3 2], [1 1 3], [10 30 23], 3, 3); a = s(1,1); b = s(2,1); c = s(8); d = s(end,end); tf = [issparse(a), issparse(b), issparse(c), issparse(d)]; e = s([1],[1]);",
    )
    .unwrap();
    assert_eq!(sparse_scalar(&vars[1]), 10.0);
    assert_eq!(sparse_scalar(&vars[2]), 0.0);
    assert_eq!(sparse_scalar(&vars[3]), 23.0);
    assert_eq!(sparse_scalar(&vars[4]), 0.0);
    assert!(matches!(
        &vars[5],
        Value::LogicalArray(logical)
            if logical.shape == vec![1, 4] && logical.data == vec![1, 1, 1, 1]
    ));
    assert_eq!(sparse_scalar(&vars[6]), 10.0);
}

#[test]
fn sparse_slice_indexing_preserves_sparse_outputs() {
    let vars = execute_source(
        "s = sparse([1 3 2], [1 1 3], [10 30 23], 3, 3); c = full(s(:,1)); r = full(s(2,:)); sub = s([1 2], [1 3]); d = full(sub); tf = issparse(sub); lin = full(s(:)); lin_tf = issparse(s(:)); pick = full(s([1 8])); pick_tf = issparse(s([1 8])); rev = full(s(3:-1:1,1)); full_range = full(s(1:end)); full_range_tf = issparse(s(1:end));",
    )
    .unwrap();
    assert!(matches!(
        &vars[1],
        Value::Tensor(tensor) if tensor.shape == vec![3, 1] && tensor.data == vec![10.0, 0.0, 30.0]
    ));
    assert!(matches!(
        &vars[2],
        Value::Tensor(tensor) if tensor.shape == vec![1, 3] && tensor.data == vec![0.0, 0.0, 23.0]
    ));
    assert!(matches!(
        &vars[3],
        Value::SparseTensor(sparse)
            if sparse.shape() == vec![2, 2]
                && sparse.get(0, 0) == Some(10.0)
                && sparse.get(1, 0).unwrap_or(0.0) == 0.0
                && sparse.get(0, 1).unwrap_or(0.0) == 0.0
                && sparse.get(1, 1) == Some(23.0)
    ));
    assert!(matches!(
        &vars[4],
        Value::Tensor(tensor) if tensor.shape == vec![2, 2] && tensor.data == vec![10.0, 0.0, 0.0, 23.0]
    ));
    assert!(logical_truth(&vars[5]));
    assert!(matches!(
        &vars[6],
        Value::Tensor(tensor)
            if tensor.shape == vec![9, 1]
                && tensor.data == vec![10.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0, 23.0, 0.0]
    ));
    assert!(logical_truth(&vars[7]));
    assert!(matches!(
        &vars[8],
        Value::Tensor(tensor) if tensor.shape == vec![1, 2] && tensor.data == vec![10.0, 23.0]
    ));
    assert!(logical_truth(&vars[9]));
    assert!(matches!(
        &vars[10],
        Value::Tensor(tensor) if tensor.shape == vec![3, 1] && tensor.data == vec![30.0, 0.0, 10.0]
    ));
    assert!(matches!(
        &vars[11],
        Value::Tensor(tensor)
            if tensor.shape == vec![9, 1]
                && tensor.data == vec![10.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0, 23.0, 0.0]
    ));
    assert!(logical_truth(&vars[12]));
}

#[test]
fn typed_sparse_slice_indexing_preserves_uint64_storage() {
    let vars = execute_source(
        "s = sparse([1 3 2], [1 1 3], uint64([1 9223372036854775808 4]), 3, 3); a = s(3,1); z = s(2,1); lin = s(:); pick = s([1 3 8]); sub = s([3 1], [1 3]); empty = s([],1);",
    )
    .unwrap();

    let expected_scalar = IntegerStorage::U64(vec![9_223_372_036_854_775_808]);
    assert!(matches!(
        &vars[1],
        Value::SparseTensor(sparse)
            if sparse.shape() == vec![1, 1]
                && sparse.integer_storage() == Some(&expected_scalar)
    ));
    assert!(matches!(
        &vars[2],
        Value::SparseTensor(sparse)
            if sparse.shape() == vec![1, 1]
                && sparse.integer_storage() == Some(&IntegerStorage::U64(vec![]))
    ));
    assert!(matches!(
        &vars[3],
        Value::SparseTensor(sparse)
            if sparse.shape() == vec![9, 1]
                && sparse.row_indices == vec![0, 2, 7]
                && sparse.integer_storage() == Some(&IntegerStorage::U64(vec![1, 9_223_372_036_854_775_808, 4]))
    ));
    let Value::SparseTensor(pick) = &vars[4] else {
        panic!("expected typed sparse linear selection, got {:?}", vars[4]);
    };
    assert_eq!(pick.shape(), vec![1, 3]);
    assert_eq!(pick.col_ptrs, vec![0, 1, 2, 3]);
    assert_eq!(pick.row_indices, vec![0, 0, 0]);
    assert_eq!(
        pick.integer_storage(),
        Some(&IntegerStorage::U64(vec![1, 9_223_372_036_854_775_808, 4]))
    );
    let Value::SparseTensor(sub) = &vars[5] else {
        panic!("expected typed sparse matrix selection, got {:?}", vars[5]);
    };
    assert_eq!(sub.shape(), vec![2, 2]);
    assert_eq!(sub.col_ptrs, vec![0, 2, 2]);
    assert_eq!(sub.row_indices, vec![0, 1]);
    assert_eq!(
        sub.integer_storage(),
        Some(&IntegerStorage::U64(vec![9_223_372_036_854_775_808, 1]))
    );
    assert!(matches!(
        &vars[6],
        Value::SparseTensor(sparse)
            if sparse.shape() == vec![0, 1]
                && sparse.integer_storage() == Some(&IntegerStorage::U64(vec![]))
    ));
}

#[test]
fn typed_sparse_find_preserves_exact_values_and_directional_order() {
    let vars = execute_source(
        "s = sparse(uint64([0 9223372036854775808;18446744073709551615 0])); [i,j,v] = find(s); [il,jl,vl] = find(s,1,'last');",
    )
    .expect("execute typed sparse find");
    assert!(matches!(
        &vars[1],
        Value::Tensor(tensor) if tensor.shape == vec![2, 1] && tensor.data == vec![2.0, 1.0]
    ));
    assert!(matches!(
        &vars[2],
        Value::Tensor(tensor) if tensor.shape == vec![2, 1] && tensor.data == vec![1.0, 2.0]
    ));
    assert!(matches!(
        &vars[3],
        Value::Tensor(tensor)
            if tensor.integer_storage() == Some(&IntegerStorage::U64(vec![u64::MAX, 9_223_372_036_854_775_808]))
    ));
    assert!(matches!(&vars[4], Value::Num(value) if *value == 1.0));
    assert!(matches!(&vars[5], Value::Num(value) if *value == 2.0));
    assert_eq!(
        vars[6],
        Value::Int(IntValue::U64(9_223_372_036_854_775_808))
    );
}

#[test]
fn sparse_assignment_updates_scalar_and_selector_entries() {
    let vars = execute_source(
        "s = sparse([1], [1], [5], 2, 2); s(2,2) = 7; s(1) = 0; s(:,1) = [1;2]; s(1:2,2) = [3;4]; s([3 3]) = [5 6]; f = full(s); n = nnz(s); t = sparse(uint64([0 0;0 0])); t(:,1) = uint64([1;9223372036854775808]);",
    )
    .expect("execute sparse assignment");
    assert!(matches!(
        &vars[1],
        Value::Tensor(tensor) if tensor.data == vec![1.0, 2.0, 6.0, 4.0]
    ));
    assert!(matches!(&vars[2], Value::Num(value) if *value == 4.0));
    assert!(matches!(
        &vars[3],
        Value::SparseTensor(sparse)
            if sparse.col_ptrs == vec![0, 2, 2]
                && sparse.row_indices == vec![0, 1]
                && sparse.integer_storage()
                    == Some(&IntegerStorage::U64(vec![1, 9_223_372_036_854_775_808]))
    ));

    let deleted = execute_source(
        "s = sparse([1 3 2], [1 1 2], [1 3 2], 3, 2); s(:,1) = []; a = full(s); s([1 3],:) = []; b = full(s); t = sparse(uint64([1 0 9223372036854775808])); t(1,2) = []; c = full(t); u = sparse([1; 0; 3]); u(2) = []; d = full(u);",
    )
    .expect("execute sparse structural deletion");
    assert!(matches!(
        &deleted[1],
        Value::Tensor(tensor) if tensor.shape == vec![3, 1] && tensor.data == vec![0.0, 2.0, 0.0]
    ));
    assert!(matches!(
        &deleted[2],
        Value::Tensor(tensor) if tensor.shape == vec![1, 1] && tensor.data == vec![2.0]
    ));
    assert!(matches!(
        &deleted[4],
        Value::Tensor(tensor)
            if tensor.shape == vec![1, 2]
                && tensor.integer_storage() == Some(&IntegerStorage::U64(vec![1, 9_223_372_036_854_775_808]))
    ));
    assert!(matches!(
        &deleted[6],
        Value::Tensor(tensor) if tensor.shape == vec![2, 1] && tensor.data == vec![1.0, 3.0]
    ));

    let deletion_err = execute_source("s = sparse([1], [1], [5], 2, 2); s(1,1) = [];").unwrap_err();
    assert_eq!(
        deletion_err.identifier(),
        Some("RunMat:UnsupportedDeletion")
    );

    let expression_deleted = execute_source(
        "s = sparse([1 3 2], [1 1 2], [1 3 2], 3, 2); s(:,1) = []; rows = [1 3]; s(rows,:) = []; f = full(s);",
    )
    .expect("execute expression-backed sparse row deletion");
    assert!(matches!(
        &expression_deleted[2],
        Value::Tensor(tensor) if tensor.shape == vec![1, 1] && tensor.data == vec![2.0]
    ));

    let all_deleted =
        execute_source("s = sparse(uint64([1 0; 0 9223372036854775808])); s(:,:) = [];")
            .expect("delete all sparse entries structurally");
    assert!(matches!(
        &all_deleted[0],
        Value::SparseTensor(sparse)
            if sparse.shape() == vec![0, 0]
                && sparse.integer_storage() == Some(&IntegerStorage::U64(vec![]))
    ));

    let grown = execute_source(
        "s = sparse(uint64([1 0])); s(1,4) = uint64(9223372036854775808); a = full(s); s(3,6) = uint64(7); b = full(s); z = sparse(uint64([])); z(5) = uint64(9); c = full(z); q = sparse([1]); q(1,4) = 0; d = full(q);",
    )
    .expect("execute sparse scalar growth");
    assert!(matches!(
        &grown[1],
        Value::Tensor(tensor)
            if tensor.shape == vec![1, 4]
                && tensor.integer_storage() == Some(&IntegerStorage::U64(vec![1, 0, 0, 9_223_372_036_854_775_808]))
    ));
    assert!(matches!(
        &grown[2],
        Value::Tensor(tensor)
            if tensor.shape == vec![3, 6]
                && tensor.integer_storage().is_some()
                && tensor.integer_storage().and_then(|storage| storage.value_at(17)) == Some(IntValue::U64(7))
    ));
    assert!(matches!(
        &grown[4],
        Value::Tensor(tensor)
            if tensor.shape == vec![1, 5]
                && tensor.integer_storage() == Some(&IntegerStorage::U64(vec![0, 0, 0, 0, 9]))
    ));
    assert!(matches!(
        &grown[6],
        Value::Tensor(tensor) if tensor.shape == vec![1, 4] && tensor.data == vec![1.0, 0.0, 0.0, 0.0]
    ));

    let selector_grown = execute_source(
        "s = sparse(uint64([1 2;3 4])); s([3 4],[4 5]) = uint64([5 9223372036854775808;7 8]); a = full(s); r = 5:6; c = [6 8]; s(r,c) = uint64([9 10;11 12]); b = full(s); t = sparse(uint64([1;2])); t(:,4) = uint64([9223372036854775808;6]); q = full(t); u = sparse(uint64([1 0;0 2])); u([4],[5]) = uint64(0); ue = full(u); un = nnz(u);",
    )
    .expect("execute sparse selector growth");
    assert!(matches!(
        &selector_grown[1],
        Value::Tensor(tensor)
            if tensor.shape == vec![4, 5]
                && tensor.integer_storage().and_then(|storage| storage.value_at(18))
                    == Some(IntValue::U64(9_223_372_036_854_775_808))
                && tensor.integer_storage().and_then(|storage| storage.value_at(0))
                    == Some(IntValue::U64(1))
                && tensor.integer_storage().and_then(|storage| storage.value_at(4))
                    == Some(IntValue::U64(2))
    ));
    assert!(matches!(
        &selector_grown[4],
        Value::Tensor(tensor)
            if tensor.shape == vec![6, 8]
                && tensor.integer_storage().and_then(|storage| storage.value_at(34))
                    == Some(IntValue::U64(9))
                && tensor.integer_storage().and_then(|storage| storage.value_at(47))
                    == Some(IntValue::U64(12))
                && tensor.integer_storage().and_then(|storage| storage.value_at(1))
                    == Some(IntValue::U64(3))
    ));
    assert!(matches!(
        &selector_grown[6],
        Value::Tensor(tensor)
            if tensor.shape == vec![2, 4]
                && tensor.integer_storage().and_then(|storage| storage.value_at(6))
                    == Some(IntValue::U64(9_223_372_036_854_775_808))
                && tensor.integer_storage().and_then(|storage| storage.value_at(7))
                    == Some(IntValue::U64(6))
    ));
    assert!(matches!(
        &selector_grown[7],
        Value::SparseTensor(sparse)
            if sparse.shape() == vec![4, 5]
                && sparse.integer_storage() == Some(&IntegerStorage::U64(vec![1, 2]))
    ));
    assert!(matches!(
        &selector_grown[8],
        Value::Tensor(tensor)
            if tensor.shape == vec![4, 5]
                && tensor.integer_storage().and_then(|storage| storage.value_at(0))
                    == Some(IntValue::U64(1))
                && tensor.integer_storage().and_then(|storage| storage.value_at(5))
                    == Some(IntValue::U64(2))
    ));
    assert!(matches!(&selector_grown[9], Value::Num(value) if *value == 2.0));

    let invalid_slice_err =
        execute_source("s = sparse([1], [1], [5], 2, 2); s([0]) = 0;").unwrap_err();
    assert_eq!(
        invalid_slice_err.identifier(),
        Some("RunMat:IndexOutOfBounds")
    );
}

#[test]
fn sparse_arithmetic_interops_with_dense_and_scalar_values() {
    let vars = execute_source(
        "s = sparse([1 3 2], [1 1 2], [10 30 20], 3, 2); t = sparse([3 1 2], [1 2 2], [5 7 -20], 3, 2); a = s + t; af = full(a); atf = issparse(a); b = s + 2; c = [1 2; 3 4; 5 6] - s; d = s .* [2 2; 3 3; 4 4]; df = full(d); dtf = issparse(d); e = 3 .* s; ef = full(e); etf = issparse(e);",
    )
    .unwrap();
    assert!(matches!(
        &vars[3],
        Value::Tensor(tensor)
            if tensor.shape == vec![3, 2]
                && tensor.data == vec![10.0, 0.0, 35.0, 7.0, 0.0, 0.0]
    ));
    assert!(logical_truth(&vars[4]));
    assert!(matches!(
        &vars[5],
        Value::Tensor(tensor)
            if tensor.shape == vec![3, 2]
                && tensor.data == vec![12.0, 2.0, 32.0, 2.0, 22.0, 2.0]
    ));
    assert!(matches!(
        &vars[6],
        Value::Tensor(tensor)
            if tensor.shape == vec![3, 2]
                && tensor.data == vec![-9.0, 3.0, -25.0, 2.0, -16.0, 6.0]
    ));
    assert!(matches!(
        &vars[8],
        Value::Tensor(tensor)
            if tensor.shape == vec![3, 2]
                && tensor.data == vec![20.0, 0.0, 120.0, 0.0, 60.0, 0.0]
    ));
    assert!(logical_truth(&vars[9]));
    assert!(matches!(
        &vars[11],
        Value::Tensor(tensor)
            if tensor.shape == vec![3, 2]
                && tensor.data == vec![30.0, 0.0, 90.0, 0.0, 60.0, 0.0]
    ));
    assert!(logical_truth(&vars[12]));
}

#[test]
fn sparse_arithmetic_handles_sparse_scalar_and_complex_interop() {
    let vars = execute_source(
        "s = sparse([1 3 2], [1 1 2], [10 30 20], 3, 2); cs = s + complex(1, 2); ct = complex(1, -1) .* s; sf = sparse(1, 1, 2, 1, 1) + s; sff = full(sf); sft = issparse(sf);",
    )
    .unwrap();
    assert!(matches!(
        &vars[1],
        Value::ComplexTensor(tensor)
            if tensor.shape == vec![3, 2]
                && tensor.data[0] == (11.0, 2.0)
                && tensor.data[1] == (1.0, 2.0)
                && tensor.data[2] == (31.0, 2.0)
    ));
    assert!(matches!(
        &vars[2],
        Value::ComplexTensor(tensor)
            if tensor.shape == vec![3, 2]
                && tensor.data[0] == (10.0, -10.0)
                && tensor.data[1] == (0.0, -0.0)
                && tensor.data[2] == (30.0, -30.0)
    ));
    assert!(matches!(
        &vars[4],
        Value::Tensor(tensor)
            if tensor.shape == vec![3, 2]
                && tensor.data == vec![12.0, 2.0, 32.0, 2.0, 22.0, 2.0]
    ));
    assert!(logical_truth(&vars[5]));
}
