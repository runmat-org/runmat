use super::*;

#[test]
fn decomposition_ldivide_is_not_a_public_method() {
    let _ = call_constructor(vec![tensor(&[1.0], 1, 1)]).expect("register class");
    let class =
        crate::class_registry::get_class(&CLASS_IDENTITY.owned()).expect("decomposition class");
    assert!(!class.methods.contains_key("ldivide"));

    assert!(runmat_builtins::builtin_function_by_name("decomposition.ldivide").is_none());
}

#[test]
fn decomposition_left_solve_reuses_mldivide_semantics() {
    let a = tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2);
    let b = tensor(&[8.0, 12.0], 2, 1);
    let d = call_constructor(vec![a]).expect("decomposition");

    let result = call_mldivide(d, b).expect("solve");

    let values = approx_vec(result);
    assert!((values[0] - 4.0).abs() < 1e-12);
    assert!((values[1] - 3.0).abs() < 1e-12);
}

#[test]
fn decomposition_right_solve_uses_rhs_object() {
    let a = tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2);
    let b = tensor(&[8.0, 12.0], 1, 2);
    let d = call_constructor(vec![a]).expect("decomposition");

    let result = call_mrdivide(b, d).expect("right solve");

    let values = approx_vec(result);
    assert!((values[0] - 4.0).abs() < 1e-12);
    assert!((values[1] - 3.0).abs() < 1e-12);
}

#[test]
fn scalar_multiply_and_divide_adjust_solve_matrix() {
    let a = tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2);
    let b = tensor(&[8.0, 12.0], 2, 1);
    let d = call_constructor(vec![a]).expect("decomposition");
    let scaled = call_mtimes(Value::Num(2.0), d).expect("scale");
    let divided = call_rdivide(scaled, Value::Num(4.0)).expect("divide");

    let result = call_mldivide(divided, b).expect("solve");

    let values = approx_vec(result);
    assert!((values[0] - 8.0).abs() < 1e-12);
    assert!((values[1] - 6.0).abs() < 1e-12);
}

#[test]
fn ctranspose_toggles_property_and_solves_transposed_system() {
    let a = tensor(&[1.0, 0.0, 2.0, 1.0], 2, 2);
    let b = tensor(&[3.0, 1.0], 2, 1);
    let d = call_constructor(vec![a]).expect("decomposition");
    let dt = call_ctranspose(d).expect("ctranspose");

    let prop = block_on(decomposition_subsref(
        dt.clone(),
        OBJECT_INDEX_MEMBER.to_string(),
        Value::String(IS_CONJUGATE_TRANSPOSED_FIELD.to_string()),
    ))
    .expect("property");
    assert_eq!(prop, Value::Bool(true));

    let result = call_mldivide(dt, b).expect("solve");

    let values = approx_vec(result);
    assert!((values[0] - 3.0).abs() < 1e-12);
    assert!((values[1] + 5.0).abs() < 1e-12);
}

#[test]
fn stored_matrix_helpers_read_typed_integer_storage_exactly() {
    let matrix = typed_int_tensor(IntegerStorage::I16(vec![1, 2, 3, 4, 5, 6]), vec![2, 3]);

    match conjugate_transpose(matrix).expect("ctranspose") {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.numeric_dtype(), NumericDType::I16);
            assert_eq!(tensor.shape, vec![3, 2]);
            assert_eq!(
                tensor::tensor_values_f64(&tensor),
                vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
            );
        }
        other => panic!("expected tensor, got {other:?}"),
    }

    let wide = (1_u64 << 53) + 1;
    let matrix = Value::Tensor(
        Tensor::new_integer(IntegerStorage::U64(vec![wide, 2, 3, 4]), vec![2, 2])
            .expect("wide integer matrix"),
    );
    match conjugate_transpose(matrix).expect("wide integer ctranspose") {
        Value::Tensor(tensor) => {
            assert_eq!(tensor.numeric_dtype(), NumericDType::U64);
            assert_eq!(tensor.numeric_value_at(0), Some(NumericScalar::U64(wide)));
            assert_eq!(tensor.numeric_value_at(1), Some(NumericScalar::U64(3)));
            assert_eq!(tensor.numeric_value_at(2), Some(NumericScalar::U64(2)));
            assert_eq!(tensor.numeric_value_at(3), Some(NumericScalar::U64(4)));
        }
        other => panic!("expected tensor, got {other:?}"),
    }

    let scaled = scale_matrix(
        typed_int_tensor(IntegerStorage::I16(vec![1, 2, 3, 4]), vec![2, 2]),
        (2.0, 0.0),
    )
    .expect("scale");
    match scaled {
        Value::Tensor(tensor) => {
            assert_eq!(tensor::tensor_values_f64(&tensor), vec![2.0, 4.0, 6.0, 8.0]);
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

#[test]
fn stored_matrix_helpers_read_typed_complex_integer_storage_exactly() {
    let matrix = mirrorless_complex_int_tensor(
        IntegerStorage::I16(vec![1, 2, 3, 4]),
        IntegerStorage::I16(vec![5, 6, 7, 8]),
        vec![2, 2],
    );

    match conjugate_transpose(matrix).expect("ctranspose") {
        Value::ComplexTensor(tensor) => {
            assert_eq!(tensor.shape, vec![2, 2]);
            assert_eq!(
                tensor.materialize_f64(),
                vec![(1.0, -5.0), (3.0, -7.0), (2.0, -6.0), (4.0, -8.0)]
            );
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }

    let scaled = scale_matrix(
        mirrorless_complex_int_tensor(
            IntegerStorage::I16(vec![1, 2]),
            IntegerStorage::I16(vec![3, 4]),
            vec![2, 1],
        ),
        (2.0, 0.0),
    )
    .expect("scale");
    match scaled {
        Value::ComplexTensor(tensor) => {
            assert_eq!(tensor.materialize_f64(), vec![(2.0, 6.0), (4.0, 8.0)]);
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }

    let wide = (1_i64 << 53) + 1;
    let matrix = mirrorless_complex_int_tensor(
        IntegerStorage::I64(vec![wide, 2, 3, 4]),
        IntegerStorage::I64(vec![5, -6, 7, -8]),
        vec![2, 2],
    );
    let Value::ComplexTensor(transposed) =
        conjugate_transpose(matrix).expect("wide complex ctranspose")
    else {
        panic!("expected exact complex integer tensor")
    };
    let storage = transposed.integer_storage().expect("exact storage");
    assert_eq!(storage.real, IntegerStorage::I64(vec![wide, 3, 2, 4]));
    assert_eq!(storage.imag, IntegerStorage::I64(vec![-5, -7, 6, 8]));
}

#[test]
fn constructor_structure_helpers_read_typed_integer_storage_exactly() {
    let upper_kept = keep_triangle(
        mirrorless_complex_int_tensor(
            IntegerStorage::I16(vec![1, 99, 0, 4]),
            IntegerStorage::I16(vec![2, 99, 0, 5]),
            vec![2, 2],
        ),
        TriangularFlag::Upper,
    )
    .expect("keep upper");
    match upper_kept {
        Value::ComplexTensor(tensor) => {
            assert_eq!(
                tensor.materialize_f64(),
                vec![(1.0, 2.0), (0.0, 0.0), (0.0, 0.0), (4.0, 5.0)]
            );
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }

    let symmetrized = symmetrize_from_triangle(
        mirrorless_complex_int_tensor(
            IntegerStorage::I16(vec![1, 2, 0, 4]),
            IntegerStorage::I16(vec![0, 3, 0, 0]),
            vec![2, 2],
        ),
        TriangularFlag::Lower,
    )
    .expect("symmetrize lower");
    match symmetrized {
        Value::ComplexTensor(tensor) => {
            assert_eq!(
                tensor.materialize_f64(),
                vec![(1.0, 0.0), (2.0, 3.0), (2.0, -3.0), (4.0, 0.0)]
            );
        }
        other => panic!("expected complex tensor, got {other:?}"),
    }
}

#[test]
fn structure_predicates_read_typed_integer_storage_exactly() {
    let diagonal = typed_int_tensor(IntegerStorage::I16(vec![1, 0, 0, 4]), vec![2, 2]);
    assert!(is_diagonal_matrix(&diagonal));
    assert!(is_triangular_matrix(&diagonal));
    assert!(is_hermitian_matrix(&diagonal));

    let not_diagonal = typed_int_tensor(IntegerStorage::I16(vec![1, 2, 0, 4]), vec![2, 2]);
    assert!(!is_diagonal_matrix(&not_diagonal));
    assert!(is_lower_triangular(&not_diagonal));
    assert!(!is_upper_triangular(&not_diagonal));
    assert!(!is_hermitian_matrix(&not_diagonal));

    let hermitian = mirrorless_complex_int_tensor(
        IntegerStorage::I16(vec![1, 2, 2, 4]),
        IntegerStorage::I16(vec![0, 3, -3, 0]),
        vec![2, 2],
    );
    assert!(is_hermitian_matrix(&hermitian));
    assert!(!matrix_is_real(&hermitian));

    let real_complex = mirrorless_complex_int_tensor(
        IntegerStorage::I16(vec![1, 0, 0, 4]),
        IntegerStorage::I16(vec![0, 0, 0, 0]),
        vec![2, 2],
    );
    assert!(is_diagonal_matrix(&real_complex));
    assert!(matrix_is_real(&real_complex));
}

#[test]
fn properties_expose_type_size_and_flags() {
    let a = tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2);
    let d =
        call_constructor(vec![a, Value::String("diagonal".to_string())]).expect("decomposition");

    let ty = block_on(decomposition_subsref(
        d.clone(),
        OBJECT_INDEX_MEMBER.to_string(),
        Value::String(TYPE_FIELD.to_string()),
    ))
    .expect("Type");
    assert_eq!(ty, Value::String("diagonal".to_string()));

    let size = block_on(decomposition_subsref(
        d,
        OBJECT_INDEX_MEMBER.to_string(),
        Value::String(MATRIX_SIZE_FIELD.to_string()),
    ))
    .expect("MatrixSize");
    assert_eq!(approx_vec(size), vec![2.0, 2.0]);
}
