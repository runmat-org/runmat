use super::*;
use futures::executor::block_on;
use runmat_value::{IntValue, IntegerStorage, StringArray};

fn admit_options(rest: Vec<Value>) -> crate::BuiltinResult<Vec<super::OptionInput>> {
    super::admit_options(crate::builtins::common::test_support::builtin_arguments(
        rest,
    ))
}

async fn lassoglm_builtin(
    x: Value,
    y: Value,
    distribution: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    super::lassoglm_builtin(
        x,
        y,
        distribution,
        crate::builtins::common::test_support::builtin_arguments(rest),
    )
    .await
}

fn tensor(data: Vec<f64>, shape: Vec<usize>) -> Value {
    Value::Tensor(Tensor::new(data, shape).unwrap())
}

fn poisoned_int_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
    let tensor = Tensor::new_integer(storage, vec![rows, cols]).unwrap();
    Value::Tensor(tensor)
}

fn cleared_int_tensor(storage: IntegerStorage, rows: usize, cols: usize) -> Value {
    let tensor = Tensor::new_integer(storage, vec![rows, cols]).unwrap();
    Value::Tensor(tensor)
}

fn output_pair(value: Value) -> (Value, Value) {
    let Value::OutputList(values) = value else {
        panic!("expected output list");
    };
    (values[0].clone(), values[1].clone())
}

fn row_field<'a>(info: &'a runmat_value::StructValue, name: &str) -> &'a Tensor {
    let Some(Value::Tensor(tensor)) = info.fields.get(name) else {
        panic!("expected tensor field {name}");
    };
    tensor
}

#[test]
fn normal_explicit_lambda_matches_linear_trend() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let out = block_on(lassoglm_builtin(
        tensor(vec![0.0, 1.0, 2.0, 3.0, 4.0], vec![5, 1]),
        tensor(vec![1.0, 3.0, 5.0, 7.0, 9.0], vec![5, 1]),
        Value::CharArray(CharArray::new_row("normal")),
        vec![
            Value::CharArray(CharArray::new_row("Lambda")),
            tensor(vec![0.0], vec![1, 1]),
            Value::CharArray(CharArray::new_row("Standardize")),
            Value::Bool(false),
        ],
    ))
    .unwrap();
    let (b, fit_info) = output_pair(out);
    let Value::Tensor(coeffs) = b else {
        panic!("expected B tensor");
    };
    assert_eq!(coeffs.shape(), &[1, 1]);
    assert!((coeffs.materialize_f64()[0] - 2.0).abs() < 1.0e-3);
    let Value::Struct(info) = fit_info else {
        panic!("expected FitInfo");
    };
    let intercept = row_field(&info, "Intercept");
    assert!((intercept.materialize_f64()[0] - 1.0).abs() < 1.0e-3);
    assert!(info.fields.contains_key("Deviance"));
    assert!(info.fields.contains_key("Distribution"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn lassoglm_rank_zero_response_reaches_row_count_admission() {
    let error = block_on(lassoglm_builtin(
        tensor(vec![0.0, 1.0], vec![2, 1]),
        tensor(vec![1.0], Vec::new()),
        Value::String("normal".into()),
        vec![Value::String("Lambda".into()), Value::Num(0.0)],
    ))
    .expect_err("one rank-zero response cannot match two observations");
    assert!(error
        .message()
        .contains("Y length 1 must match the number of rows in X (2)"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn lassoglm_rank_one_response_retains_vector_admission() {
    let output = block_on(lassoglm_builtin(
        tensor(vec![0.0, 1.0], vec![2, 1]),
        tensor(vec![1.0, 3.0], vec![2]),
        Value::String("normal".into()),
        vec![
            Value::String("Lambda".into()),
            Value::Num(0.0),
            Value::String("Standardize".into()),
            Value::Bool(false),
        ],
    ))
    .expect("rank-one response");
    let Value::Tensor(coefficients) = output else {
        panic!("lassoglm coefficients");
    };
    assert_eq!(coefficients.shape(), &[1, 1]);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn lassoglm_numeric_options_retain_vector_shape_admission() {
    assert_eq!(
        numeric_vector(&tensor(vec![0.25], Vec::new()), "Lambda").unwrap(),
        vec![0.25]
    );
    assert_eq!(
        numeric_vector(&tensor(vec![0.25, 0.5], vec![2]), "Lambda").unwrap(),
        vec![0.25, 0.5]
    );
    let error = numeric_vector(&tensor(vec![0.25; 4], vec![2, 2]), "Lambda")
        .expect_err("matrix option is not a vector");
    assert!(error.message().contains("Lambda must be a numeric vector"));
}

#[test]
fn binomial_and_poisson_paths_return_fitinfo() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let out = block_on(lassoglm_builtin(
        tensor(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![6, 1]),
        tensor(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![6, 1]),
        Value::String("binomial".to_string().into()),
        vec![
            Value::CharArray(CharArray::new_row("Lambda")),
            tensor(vec![0.0, 0.1], vec![1, 2]),
            Value::CharArray(CharArray::new_row("CV")),
            Value::Num(3.0),
            Value::CharArray(CharArray::new_row("MaxIter")),
            Value::Num(500.0),
        ],
    ))
    .unwrap();
    let (b, fit_info) = output_pair(out);
    let Value::Tensor(coeffs) = b else {
        panic!("expected B tensor");
    };
    assert_eq!(coeffs.shape(), &[1, 2]);
    let Value::Struct(info) = fit_info else {
        panic!("expected FitInfo");
    };
    assert!(info.fields.contains_key("SE"));
    assert!(info.fields.contains_key("LambdaMinDeviance"));

    let out = block_on(lassoglm_builtin(
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![4, 1]),
        tensor(vec![1.0, 2.0, 4.0, 7.0], vec![4, 1]),
        Value::String("poisson".to_string().into()),
        vec![
            Value::CharArray(CharArray::new_row("Lambda")),
            tensor(vec![0.0], vec![1, 1]),
            Value::CharArray(CharArray::new_row("Options")),
            {
                let mut st = runmat_value::UnboundedStructValueBuilder::new();
                st.insert("MaxIter", Value::Num(400.0));
                st.insert("TolX", Value::Num(1.0e-6));
                Value::Struct(st.finish())
            },
        ],
    ))
    .unwrap();
    let (b, _) = output_pair(out);
    let Value::Tensor(coeffs) = b else {
        panic!("expected B tensor");
    };
    assert_eq!(coeffs.shape(), &[1, 1]);
    assert!(coeffs.materialize_f64()[0].is_finite());
}

#[test]
fn binomial_size_and_single_row_count_response_are_supported() {
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let out = block_on(lassoglm_builtin(
        tensor(vec![0.0, 1.0, 2.0], vec![3, 1]),
        tensor(vec![0.2, 0.5, 0.8], vec![3, 1]),
        Value::String("binomial".to_string().into()),
        vec![
            Value::CharArray(CharArray::new_row("Lambda")),
            tensor(vec![0.0, 0.2], vec![1, 2]),
            Value::CharArray(CharArray::new_row("BinomialSize")),
            tensor(vec![10.0, 12.0, 14.0], vec![3, 1]),
            Value::CharArray(CharArray::new_row("Weights")),
            tensor(vec![1.0, 2.0, 1.0], vec![3, 1]),
            Value::CharArray(CharArray::new_row("CV")),
            Value::Num(3.0),
        ],
    ))
    .unwrap();
    let (b, fit_info) = output_pair(out);
    let Value::Tensor(coeffs) = b else {
        panic!("expected B tensor");
    };
    assert_eq!(coeffs.shape(), &[1, 2]);
    let Value::Struct(info) = fit_info else {
        panic!("expected FitInfo");
    };
    let deviance = row_field(&info, "Deviance");
    assert_eq!(deviance.materialize_f64().len(), 2);
    assert!(deviance
        .materialize_f64()
        .iter()
        .all(|value| value.is_finite()));

    let out = block_on(lassoglm_builtin(
        tensor(vec![1.0], vec![1, 1]),
        tensor(vec![8.0, 10.0], vec![1, 2]),
        Value::String("binomial".to_string().into()),
        vec![
            Value::CharArray(CharArray::new_row("Lambda")),
            tensor(vec![0.1], vec![1, 1]),
        ],
    ))
    .unwrap();
    let (b, _) = output_pair(out);
    let Value::Tensor(coeffs) = b else {
        panic!("expected B tensor");
    };
    assert_eq!(coeffs.shape(), &[1, 1]);
}

#[test]
fn rejects_unsupported_link_and_excessive_resource_options() {
    let err = block_on(lassoglm_builtin(
        tensor(vec![0.0, 1.0, 2.0], vec![3, 1]),
        tensor(vec![0.0, 1.0, 1.0], vec![3, 1]),
        Value::String("binomial".to_string().into()),
        vec![
            Value::CharArray(CharArray::new_row("Link")),
            Value::CharArray(CharArray::new_row("probit")),
        ],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:lassoglm:InvalidArgument"));

    let err = block_on(lassoglm_builtin(
        tensor(vec![0.0, 1.0, 2.0], vec![3, 1]),
        tensor(vec![0.0, 1.0, 1.0], vec![3, 1]),
        Value::String("binomial".to_string().into()),
        vec![
            Value::CharArray(CharArray::new_row("NumLambda")),
            Value::Num((MAX_NUM_LAMBDA + 1) as f64),
        ],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:lassoglm:InvalidArgument"));
}

#[test]
fn predictor_geometry_rejects_nonsingleton_tail_and_keeps_singleton_tail() {
    let rejected = block_on(lassoglm_builtin(
        tensor(vec![1.0; 8], vec![2, 2, 2]),
        tensor(vec![1.0, 2.0], vec![2, 1]),
        Value::from("normal"),
        Vec::new(),
    ));
    assert_eq!(
        rejected
            .expect_err("third predictor axis must not be ignored")
            .identifier(),
        Some("RunMat:lassoglm:InvalidArgument")
    );
    let accepted = block_on(lassoglm_builtin(
        tensor(vec![0.0, 1.0, 2.0, 3.0, 1.0, 0.0, 1.0, 0.0], vec![4, 2, 1]),
        tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]),
        Value::from("normal"),
        vec![Value::from("Lambda"), Value::Num(0.1)],
    ));
    assert!(
        accepted.is_ok(),
        "singleton tail retains matrix geometry: {accepted:?}"
    );
}

#[test]
fn invalid_host_option_precedes_unregistered_predictor_download() {
    let resident = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![2, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("unregistered resident fixture")
        .handle;
    let error = block_on(lassoglm_builtin(
        Value::GpuTensor(resident),
        tensor(vec![1.0, 2.0], vec![2, 1]),
        Value::from("normal"),
        vec![Value::from("Alpha"), Value::Num(2.0)],
    ))
    .expect_err("host option must reject before provider access");
    assert_eq!(error.identifier(), Some("RunMat:lassoglm:InvalidArgument"));
}

#[test]
fn validates_binomial_counts_and_predictor_names() {
    let err = block_on(lassoglm_builtin(
        tensor(vec![0.0, 1.0], vec![2, 1]),
        tensor(vec![2.0, 1.0, 1.0, 1.0], vec![2, 2]),
        Value::String("binomial".to_string().into()),
        Vec::new(),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:lassoglm:InvalidArgument"));

    let err = block_on(lassoglm_builtin(
        tensor(vec![0.0, 1.0, 2.0, 3.0], vec![2, 2]),
        tensor(vec![1.0, 2.0], vec![2, 1]),
        Value::String("normal".to_string().into()),
        vec![
            Value::CharArray(CharArray::new_row("PredictorNames")),
            Value::StringArray(StringArray::new(vec!["x".into()], vec![1, 1]).unwrap()),
        ],
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:lassoglm:InvalidArgument"));
}

#[test]
fn lassoglm_reads_typed_integer_storage_exactly() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let _guard = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let out = block_on(lassoglm_builtin(
        poisoned_int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3, 4]), 5, 1),
        poisoned_int_tensor(IntegerStorage::I16(vec![1, 3, 5, 7, 9]), 5, 1),
        Value::String("normal".to_string().into()),
        vec![
            Value::CharArray(CharArray::new_row("Lambda")),
            cleared_int_tensor(IntegerStorage::U8(vec![0]), 1, 1),
            Value::CharArray(CharArray::new_row("Weights")),
            poisoned_int_tensor(IntegerStorage::U16(vec![1, 1, 2, 2, 1]), 5, 1),
            Value::CharArray(CharArray::new_row("MaxIter")),
            cleared_int_tensor(IntegerStorage::U16(vec![300]), 1, 1),
            Value::CharArray(CharArray::new_row("Standardize")),
            Value::Bool(false),
        ],
    ))
    .unwrap();
    let (b, fit_info) = output_pair(out);
    let Value::Tensor(coeffs) = b else {
        panic!("expected B tensor");
    };
    assert_eq!(coeffs.shape(), &[1, 1]);
    assert!((coeffs.materialize_f64()[0] - 2.0).abs() < 1.0e-3);
    let Value::Struct(info) = fit_info else {
        panic!("expected FitInfo");
    };
    assert!(info.fields.contains_key("Lambda"));
}

#[test]
fn lassoglm_gates_integer_roles_before_gather_and_rejects_lossy_data() {
    let integer_data = cleared_int_tensor(IntegerStorage::U8(vec![1, 2]), 2, 1);
    let floating = tensor(vec![1.0, 2.0], vec![2, 1]);
    let distr = Value::from("normal");
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let err = ensure_extensions(&integer_data, &floating, &distr, &[])
        .expect_err("strict mode must reject integer data before gather");
    assert_eq!(
        err.identifier(),
        Some("RunMat:compatibility:LassoglmIntegerDataExtension")
    );
    let err = ensure_extensions(
        &floating,
        &tensor(vec![1.0, 2.0], vec![2, 1]),
        &distr,
        &admit_options(vec![
            Value::from("NumLambda"),
            Value::Int(IntValue::U16(20)),
        ])
        .unwrap(),
    )
    .expect_err("strict mode must reject integer controls before gather");
    assert_eq!(
        err.identifier(),
        Some("RunMat:compatibility:LassoglmIntegerControlExtension")
    );
    drop(_strict);

    let wide = cleared_int_tensor(IntegerStorage::U64(vec![(1_u64 << 53) + 1]), 1, 1);
    let err = value_to_real_tensor("X", wide).expect_err("lossy integer must reject");
    assert!(err.message.contains("exactly representable"));
}

#[test]
fn lassoglm_positive_usize_parses_integer_bounds_exactly() {
    assert_eq!(
        positive_usize(&Value::Int(IntValue::U16(3)), "NumLambda").unwrap(),
        3
    );
    assert_eq!(
        positive_usize(
            &cleared_int_tensor(IntegerStorage::U64(vec![3]), 1, 1),
            "NumLambda"
        )
        .unwrap(),
        3
    );
    assert_eq!(
        positive_usize(
            &cleared_int_tensor(IntegerStorage::U64(vec![usize::MAX as u64]), 1, 1),
            "NumLambda"
        )
        .unwrap(),
        usize::MAX
    );

    for value in [
        Value::Int(IntValue::I8(-1)),
        Value::Num(1.5),
        Value::Num(usize::MAX as f64),
        Value::Num(usize::MAX as f64 + 1.0),
    ] {
        assert!(positive_usize(&value, "NumLambda").is_err());
    }
}

#[test]
fn lassoglm_reads_typed_integer_binomial_counts_exactly() {
    let _runmat = crate::compatibility::push_runmat_extensions_enabled(true);
    let out = block_on(lassoglm_builtin(
        poisoned_int_tensor(IntegerStorage::I16(vec![0, 1, 2, 3, 4, 5]), 6, 1),
        poisoned_int_tensor(
            IntegerStorage::U16(vec![1, 1, 2, 2, 3, 3, 5, 5, 5, 5, 5, 5]),
            6,
            2,
        ),
        Value::String("binomial".to_string().into()),
        vec![
            Value::CharArray(CharArray::new_row("Lambda")),
            poisoned_int_tensor(IntegerStorage::U8(vec![0]), 1, 1),
        ],
    ))
    .unwrap();
    let Value::Tensor(coeffs) = out else {
        panic!("expected B tensor");
    };
    assert_eq!(coeffs.shape(), &[1, 1]);
    assert!(coeffs.materialize_f64()[0].is_finite());
}

#[test]
fn lassoglm_empty_numeric_helper_uses_typed_integer_storage_len() {
    let empty = Tensor::new_integer(IntegerStorage::U16(Vec::new()), vec![0, 1]).unwrap();

    assert!(is_empty_numeric(&Value::Tensor(empty)));
}

#[test]
fn residency_policy_distinguishes_automatic_and_explicit_handles() {
    let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("genuine unregistered resident fixture")
        .handle;
    let host = tensor(vec![1.0], vec![1, 1]);
    let distr = Value::from("normal");
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);

    let automatic = Value::GpuTensor(
        handle
            .clone()
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
            .expect("validated provenance"),
    );
    ensure_extensions(&automatic, &host, &distr, &[])
        .expect("automatic residency must remain transparent to compatibility policy");

    let explicit = Value::GpuTensor(
        handle
            .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            .expect("validated provenance"),
    );
    let err = ensure_extensions(&explicit, &host, &distr, &[])
        .expect_err("an explicit gpuArray must require the resident-input extension");
    assert_eq!(
        err.identifier(),
        Some("RunMat:compatibility:LassoglmResidentInputExtension")
    );
}

#[test]
fn call_surface_validation_precedes_residency_policy() {
    let handle = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("genuine unregistered resident fixture")
        .handle
        .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Explicit)
        .expect("validated provenance");
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);
    let err = block_on(lassoglm_builtin(
        Value::GpuTensor(handle.clone()),
        tensor(vec![1.0], vec![1, 1]),
        Value::Num(1.0),
        vec![Value::from("Lambda")],
    ))
    .expect_err("invalid distribution text must fail without provider access");
    assert_eq!(err.identifier(), Some("RunMat:lassoglm:InvalidArgument"));
}

#[test]
fn nested_options_integer_roles_are_gated_by_semantic_role() {
    let host = tensor(vec![1.0], vec![1, 1]);
    let distr = Value::from("normal");
    let _strict = crate::compatibility::push_runmat_extensions_enabled(false);

    for (field, expected) in [
        (
            "MaxIter",
            "RunMat:compatibility:LassoglmIntegerControlExtension",
        ),
        (
            "TolX",
            "RunMat:compatibility:LassoglmIntegerParameterExtension",
        ),
        (
            "UseParallel",
            "RunMat:compatibility:LassoglmIntegerBooleanExtension",
        ),
    ] {
        let mut options = runmat_value::UnboundedStructValueBuilder::new();
        options.insert(field, Value::Int(IntValue::U8(1)));
        let rest = [Value::from("Options"), Value::Struct(options.finish())];
        let rest = admit_options(rest.to_vec()).unwrap();
        let err = ensure_extensions(&host, &host, &distr, &rest)
            .expect_err("nested native integer option must be gated");
        assert_eq!(err.identifier(), Some(expected));
    }
}

#[test]
fn integer_boolean_tensor_matches_declared_role() {
    assert!(scalar_bool(
        &cleared_int_tensor(IntegerStorage::U16(vec![1]), 1, 1),
        "Standardize"
    )
    .unwrap());
    assert!(scalar_bool(
        &cleared_int_tensor(IntegerStorage::I8(vec![2]), 1, 1),
        "Standardize"
    )
    .is_err());
}
