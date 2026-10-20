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
fn properties_expose_type_size_and_flags() {
    let a = tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2);
    let d = call_constructor(vec![a, Value::String("diagonal".to_string().into())])
        .expect("decomposition");

    let ty = block_on(decomposition_subsref(
        d.clone(),
        OBJECT_INDEX_MEMBER.to_string(),
        Value::String(TYPE_FIELD.to_string().into()),
    ))
    .expect("Type");
    assert_eq!(ty, Value::String("diagonal".to_string().into()));

    let size = block_on(decomposition_subsref(
        d,
        OBJECT_INDEX_MEMBER.to_string(),
        Value::String(MATRIX_SIZE_FIELD.to_string().into()),
    ))
    .expect("MatrixSize");
    assert_eq!(approx_vec_rooted(size), vec![2.0, 2.0]);
}
#[test]
fn registered_constructor_and_ctranspose_dispatch_work_together() {
    let a = tensor(&[1.0, 0.0, 2.0, 1.0], 2, 2);
    let d = block_on(crate::call_builtin_async("decomposition", &[a])).expect("constructor");
    let dt = d
        .with_owned(|d| block_on(crate::call_builtin_async("ctranspose", &[d])))
        .expect("dispatch");
    let b = tensor(&[3.0, 1.0], 2, 1);

    let result = dt.with_owned(|dt| call_mldivide(dt, b)).expect("solve");

    let values = approx_vec(result);
    assert!((values[0] - 3.0).abs() < 1e-12);
    assert!((values[1] + 5.0).abs() < 1e-12);
}

#[test]
fn private_matrix_storage_is_not_publicly_readable() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
    let d = call_constructor(vec![tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2)]).expect("decomposition");

    let subsref_err = block_on(decomposition_subsref(
        d.clone(),
        OBJECT_INDEX_MEMBER.to_string(),
        Value::String(MATRIX_FIELD.to_string().into()),
    ))
    .expect_err("private subsref should fail");
    assert!(subsref_err
        .message()
        .contains("unknown decomposition property"));

    let getfield_err = block_on(crate::call_builtin_async(
        "getfield",
        &[d, Value::String(MATRIX_FIELD.to_string().into())],
    ))
    .expect_err("private getfield should fail");
    assert!(matches!(
        getfield_err.identifier(),
        Some("RunMat:PropertyPrivateAccess")
            | Some("RunMat:compatibility:GetfieldObjectFamilyExtension")
    ));
}

#[test]
fn ctranspose_updates_rectangular_matrix_size_property() {
    let a = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let d = call_constructor(vec![a]).expect("decomposition");
    let dt = call_ctranspose(d).expect("ctranspose");

    let size = block_on(decomposition_subsref(
        dt,
        OBJECT_INDEX_MEMBER.to_string(),
        Value::String(MATRIX_SIZE_FIELD.to_string().into()),
    ))
    .expect("MatrixSize");

    assert_eq!(approx_vec_rooted(size), vec![3.0, 2.0]);
}

#[test]
fn complex_scale_marks_decomposition_nonreal() {
    let a = tensor(&[2.0, 0.0, 0.0, 4.0], 2, 2);
    let d = call_constructor(vec![a]).expect("decomposition");
    let scaled = call_mtimes(Value::Complex(0.0, 1.0), d).expect("complex scale");

    let is_real = block_on(decomposition_subsref(
        scaled,
        OBJECT_INDEX_MEMBER.to_string(),
        Value::String(IS_REAL_FIELD.to_string().into()),
    ))
    .expect("IsReal");

    assert_eq!(is_real, Value::Bool(false));
}

#[test]
fn complex_single_decomposition_reports_single_datatype() {
    let matrix = Value::ComplexTensor(
        ComplexTensor::from_f32(
            vec![(2.0, 1.0), (0.0, 0.0), (0.0, 0.0), (4.0, -1.0)],
            vec![2, 2],
        )
        .expect("complex single matrix"),
    );
    let d = call_constructor(vec![matrix]).expect("complex single decomposition");
    let datatype = block_on(decomposition_subsref(
        d,
        OBJECT_INDEX_MEMBER.to_string(),
        Value::String(DATATYPE_FIELD.to_string().into()),
    ))
    .expect("Datatype");
    assert_eq!(datatype, Value::String("single".to_string().into()));
}
