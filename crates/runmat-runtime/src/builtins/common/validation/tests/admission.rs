use super::*;
type Validator = fn(crate::call::BuiltinArguments) -> BuiltinResult<Value>;
fn validators() -> Vec<(&'static str, Validator, Vec<Value>)> {
    vec![
        (
            "mustBeA",
            must_be_a::evaluate as Validator,
            vec![Value::Num(1.0), Value::String("double".into())],
        ),
        (
            "mustBeColumn",
            must_be_column::evaluate as Validator,
            vec![Value::Num(1.0)],
        ),
        (
            "mustBeFloat",
            must_be_float::evaluate as Validator,
            vec![Value::Num(1.0)],
        ),
        (
            "mustBeNonempty",
            must_be_nonempty::evaluate as Validator,
            vec![Value::Num(1.0)],
        ),
        (
            "mustBeNonsparse",
            must_be_nonsparse::evaluate as Validator,
            vec![Value::Num(1.0)],
        ),
        (
            "mustBeNumeric",
            must_be_numeric::evaluate as Validator,
            vec![Value::Num(1.0)],
        ),
        (
            "mustBeNumericOrLogical",
            must_be_numeric_or_logical::evaluate as Validator,
            vec![Value::Num(1.0)],
        ),
        (
            "mustBeScalarOrEmpty",
            must_be_scalar_or_empty::evaluate as Validator,
            vec![Value::Num(1.0)],
        ),
        (
            "mustBeSparse",
            must_be_sparse::evaluate as Validator,
            vec![Value::Tensor(Tensor::new(vec![], vec![0, 0]).unwrap())],
        ),
        (
            "mustBeText",
            must_be_text::evaluate as Validator,
            vec![Value::String("valid_name".into())],
        ),
        (
            "mustBeTextScalar",
            must_be_text_scalar::evaluate as Validator,
            vec![Value::String("valid_name".into())],
        ),
        (
            "mustBeNonzeroLengthText",
            must_be_nonzero_length_text::evaluate as Validator,
            vec![Value::String("valid_name".into())],
        ),
        (
            "mustBeUnderlyingType",
            must_be_underlying_type::evaluate as Validator,
            vec![Value::Num(1.0), Value::String("double".into())],
        ),
        (
            "mustBeValidVariableName",
            must_be_valid_variable_name::evaluate as Validator,
            vec![Value::String("valid_name".into())],
        ),
        (
            "mustBeVector",
            must_be_vector::evaluate as Validator,
            vec![Value::Num(1.0)],
        ),
        (
            "validateFunctionSignaturesJSON",
            validate_function_signatures_json::evaluate as Validator,
            vec![Value::String("{}".into())],
        ),
    ]
}

#[test]
fn zero_output_and_implicit_calls_execute_and_assigned_outputs_reject() {
    let _mode = crate::compatibility::push_runmat_extensions_enabled(true);
    for (name, evaluate, args) in validators() {
        for count in [None, Some(0)] {
            let _outputs =
                crate::output_count::push_output_count(count).expect("output-count scope");
            assert!(
                evaluate(
                    crate::builtins::common::validation::tests::retained_arguments(args.clone())
                )
                .is_ok(),
                "{name} count {count:?}"
            );
            assert!(
                invoke_validator(name, args.clone()).is_ok(),
                "{name} retained API"
            );
        }
        for count in [1, 2, 4] {
            let _outputs =
                crate::output_count::push_output_count(Some(count)).expect("output-count scope");
            let error = evaluate(
                crate::builtins::common::validation::tests::retained_arguments(args.clone()),
            )
            .unwrap_err();
            assert_eq!(
                error.identifier(),
                Some(format!("RunMat:{name}:TooManyOutputs").as_str())
            );
        }
    }
}

#[test]
fn canonical_dispatch_preserves_zero_output_statements() {
    let _mode = crate::compatibility::push_runmat_extensions_enabled(true);
    for (name, _, args) in validators() {
        futures::executor::block_on(crate::call_builtin_async_with_outputs(name, &args, 0))
            .unwrap_or_else(|error| panic!("{name}: {error}"));
    }
}

#[test]
fn excess_outputs_and_invalid_controls_precede_provider_lookup() {
    let allocation = resident_fixture(vec![1, 1], runmat_accelerate_api::NumericElementType::F64);
    let handle = allocation.handle;
    for (name, evaluate, mut args) in validators() {
        args[0] = Value::GpuTensor(handle.clone());
        let _outputs = crate::output_count::push_output_count(Some(1)).expect("output-count scope");
        let error = evaluate(crate::builtins::common::validation::tests::retained_arguments(args))
            .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some(format!("RunMat:{name}:TooManyOutputs").as_str())
        );
    }
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    for evaluate in [
        must_be_a::evaluate as Validator,
        must_be_underlying_type::evaluate,
    ] {
        let error = evaluate(
            crate::builtins::common::validation::tests::retained_arguments(vec![
                Value::GpuTensor(handle.clone()),
                Value::Num(1.0),
            ]),
        )
        .unwrap_err();
        assert_eq!(
            error.identifier(),
            Some("RunMat:argumentValidation:InvalidArgument")
        );
    }
    let error = must_be_vector::evaluate(
        crate::builtins::common::validation::tests::retained_arguments(vec![
            Value::GpuTensor(handle),
            Value::String("bad".into()),
        ]),
    )
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:mustBeVector:InvalidArgument")
    );
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn vector_option_accepts_existing_single_text_cell_form() {
    let option =
        Value::Cell(CellArray::new(vec![Value::String("ALLOW-ALL-EMPTIES".into())], 1, 1).unwrap());
    let empty = Value::Tensor(Tensor::new(vec![], vec![0, 2, 3]).unwrap());
    assert!(must_be_vector::evaluate(
        crate::builtins::common::validation::tests::retained_arguments(vec![empty, option])
    )
    .is_ok());
}

#[test]
fn json_payload_extension_is_gated_without_filesystem_access() {
    let _outputs = crate::output_count::push_output_count(Some(0)).expect("output-count scope");
    let _mode = crate::compatibility::push_runmat_extensions_enabled(false);
    let error = validate_function_signatures_json::evaluate(
        crate::builtins::common::validation::tests::retained_arguments(vec![Value::String(
            "{}".into(),
        )]),
    )
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        Some("RunMat:compatibility:ValidateFunctionSignaturesJSONPayloadExtension")
    );
}
