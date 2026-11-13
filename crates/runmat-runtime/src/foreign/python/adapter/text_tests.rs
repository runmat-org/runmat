use super::*;
use runmat_value::{CellArray, CharArray, IntValue, StringArray};

#[test]
fn positional_integer_indices_keep_exact_wide_bits() {
    for one_based in [1, (1_u64 << 53) + 1, u64::MAX] {
        assert!(matches!(
            python_index(Some(Value::Int(IntValue::U64(one_based)))).unwrap(),
            PythonValue::Unsigned(index) if index == one_based - 1
        ));
    }
    assert!(matches!(
        python_index(Some(Value::Int(IntValue::I64(i64::MAX)))).unwrap(),
        PythonValue::Unsigned(index) if index == i64::MAX as u64 - 1
    ));
    for invalid in [
        Value::Int(IntValue::I64(-1)),
        Value::Int(IntValue::U64(0)),
        Value::Num(u64::MAX as f64),
        Value::Num(f64::INFINITY),
        Value::Num(1.5),
    ] {
        assert_eq!(
            python_index(Some(invalid)).unwrap_err().identifier(),
            Some("RunMat:Foreign:InvalidCall")
        );
    }
}

fn row(text: &str) -> Value {
    Value::CharArray(
        CharArray::from_unicode_scalars_with_shape(
            text.chars().collect(),
            vec![1, text.chars().count(), 1],
        )
        .expect("effective row"),
    )
}

#[test]
fn strict_scalar_text_accepts_rows_and_scalar_strings_without_numeric_coercion() {
    assert_eq!(string_argument(Some(row("code")), "code").unwrap(), "code");
    assert_eq!(
        string_argument(
            Some(Value::StringArray(
                StringArray::new(vec!["name".into()], vec![1, 1]).unwrap()
            )),
            "name"
        )
        .unwrap(),
        "name"
    );
    for invalid in [
        Value::Num(3.0),
        Value::Int(IntValue::I64(3)),
        Value::Bool(true),
        Value::String(String::new().into()),
        Value::CharArray(CharArray::from_unicode_scalars(vec!['x', 'y'], 2, 1).unwrap()),
        Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![1, 2]).unwrap()),
    ] {
        assert_eq!(
            string_argument(Some(invalid), "Python code")
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:InvalidCall")
        );
    }
}

#[test]
fn output_name_cells_and_script_arguments_require_actual_text() {
    let names = Value::Cell(
        CellArray::new(vec![row("answer"), Value::String("other".into())], 1, 2).unwrap(),
    );
    assert_eq!(
        output_names(Some(names)).unwrap(),
        vec!["answer".to_string(), "other".to_string()]
    );
    for invalid in [
        Value::Num(42.0),
        Value::Int(IntValue::I64(1)),
        Value::Bool(false),
    ] {
        let cell = Value::Cell(CellArray::new(vec![invalid.clone()], 1, 1).unwrap());
        assert_eq!(
            output_names(Some(cell)).unwrap_err().identifier(),
            Some("RunMat:Foreign:InvalidCall")
        );
        let pair = vec![
            row("Arguments"),
            Value::Cell(CellArray::new(vec![invalid], 1, 1).unwrap()),
        ];
        assert_eq!(
            split_file_inputs(pair).unwrap_err().identifier(),
            Some("RunMat:Foreign:InvalidCall")
        );
    }
    let pair = vec![
        row("Arguments"),
        Value::Cell(CellArray::new(vec![row("--flag")], 1, 1).unwrap()),
    ];
    let (arguments, remaining) = split_file_inputs(pair).unwrap();
    assert_eq!(arguments, vec!["--flag".to_string()]);
    assert!(remaining.is_empty());
}

#[test]
fn empty_output_names_keep_the_existing_container_contract() {
    // Unlike source and attribute text, the output-name decoder historically
    // accepts empty text here; changing name validity belongs to a separate
    // public-contract review, not the no-coercion repair.
    assert_eq!(
        output_names(Some(Value::String(String::new().into()))).unwrap(),
        vec![String::new()]
    );
    let cell =
        Value::Cell(CellArray::new(vec![Value::String(String::new().into())], 1, 1).unwrap());
    assert_eq!(output_names(Some(cell)).unwrap(), vec![String::new()]);
}

#[test]
fn invalid_python_controls_do_not_discover_or_start_an_interpreter() {
    use crate::execution::RuntimeExecutionService;

    let configuration = PythonRuntimeConfiguration {
        executable: Some("/runmat/nonexistent-python-for-preflight".into()),
        ..PythonRuntimeConfiguration::default()
    };
    let adapter =
        PythonAdapter::with_configuration(ForeignHandleRegistry::default(), configuration)
            .expect("register host without starting Python");
    let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
    let invalid_calls = [
        ("invoke_qualified", vec![Value::Bool(true)]),
        ("get_member", vec![Value::Num(1.0)]),
        ("set_member", vec![Value::Num(1.0)]),
        ("invoke_member", vec![Value::Num(1.0)]),
        ("get_item", vec![Value::Num(1.0)]),
        ("set_item", vec![Value::Num(1.0)]),
        ("iterate", vec![Value::Num(1.0)]),
        (
            "pyrun",
            vec![Value::Num(1.0), Value::String("answer".into())],
        ),
        (
            "pyrun",
            vec![
                Value::String("answer = 1".into()),
                Value::Cell(CellArray::new(vec![Value::Bool(true)], 1, 1).unwrap()),
            ],
        ),
        (
            "pyrun",
            vec![
                Value::String("answer = 1".into()),
                Value::String("answer".into()),
                Value::Int(IntValue::I64(1)),
                Value::Num(2.0),
            ],
        ),
        (
            "pyrunfile",
            vec![Value::Bool(true), Value::String("answer".into())],
        ),
        (
            "pyrunfile",
            vec![
                Value::String("script.py".into()),
                Value::String("answer".into()),
                Value::String("Arguments".into()),
                Value::Cell(CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap()),
            ],
        ),
    ];
    for (symbol, arguments) in invalid_calls {
        let error = futures::executor::block_on(adapter.invoke(
            context.clone(),
            ForeignCall {
                adapter: PYTHON_ADAPTER_ID.into(),
                symbol: symbol.into(),
                arguments: crate::call::RootedFunctionArguments::ordinary(
                    crate::execution::RootedValues::new(arguments).unwrap(),
                ),
                requested_outputs: 1,
            },
        ))
        .into_invoked_result()
        .expect_err("invalid control must precede interpreter discovery");
        assert_eq!(
            error.identifier(),
            Some("RunMat:Foreign:InvalidCall"),
            "{symbol}"
        );
        assert!(!adapter.is_running(), "{symbol} started an interpreter");
    }
}
