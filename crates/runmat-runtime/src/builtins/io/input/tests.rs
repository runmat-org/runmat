use super::*;
use crate::interaction::{push_queued_response, InteractionResponse};

#[path = "tests/retry.rs"]
mod retry;

fn input_with_empty_prompt() -> BuiltinResult<crate::execution::RootedValue> {
    futures::executor::block_on(input_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
            String::new().into(),
        )]),
    ))
}

#[test]
fn input_descriptor_signatures_cover_core_forms() {
    let labels: Vec<&str> = INPUT_DESCRIPTOR
        .signatures
        .iter()
        .map(|sig| sig.label)
        .collect();
    assert!(labels.contains(&"value = input()"));
    assert!(labels.contains(&"value = input(prompt)"));
    assert!(labels.contains(&"value = input(prompt, stringFlag)"));
    assert!(labels.contains(&"value = input(stringFlag, prompt)"));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn numeric_input_parses_scalar() {
    push_queued_response(Ok(InteractionResponse::Line("41".into())));
    let value = input_with_empty_prompt().expect("input");
    assert_eq!(value, Value::Num(41.0));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn string_mode_returns_char_row() {
    push_queued_response(Ok(InteractionResponse::Line("RunMat".into())));
    let prompt = Value::CharArray(CharArray::new_row("Name: "));
    let mode = Value::String("s".to_string().into());
    let value = futures::executor::block_on(input_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![prompt, mode]),
    ))
    .expect("input");
    assert_eq!(value, Value::CharArray(CharArray::new_row("RunMat")));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn empty_response_returns_empty_tensor() {
    push_queued_response(Ok(InteractionResponse::Line("   ".into())));
    let value = input_with_empty_prompt().expect("input");
    match value.value() {
        Value::Tensor(t) => assert!(t.materialize_f64().is_empty()),
        other => panic!("expected empty tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn matrix_literal_parses_without_eval_hook() {
    // The fast-path parser handles `[1 2 3]` directly, so no eval hook (and
    // therefore no recursive interpret() call) is needed.
    push_queued_response(Ok(InteractionResponse::Line("[1 2 3]".into())));
    let value = input_with_empty_prompt().expect("input");
    match value.value() {
        Value::Tensor(t) => {
            assert_eq!(t.rows(), 1);
            assert_eq!(t.cols(), 3);
            assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 3.0]);
        }
        other => panic!("expected 1×3 tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn named_constants_parse_without_eval_hook() {
    push_queued_response(Ok(InteractionResponse::Line("pi".into())));
    let value = input_with_empty_prompt().expect("input");
    assert_eq!(value, Value::Num(std::f64::consts::PI));
}

/// `e` is not a MATLAB built-in constant. The fast-path parser must not map
/// it to Euler's number; it should fall through so the eval hook or
/// `str2double` can handle it (which will NaN or error on an unknown identifier).
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn bare_e_is_not_eulers_number() {
    assert_eq!(parse_scalar_value("e"), None);
    assert_eq!(parse_scalar_value("E"), None);
}

/// `[1 e 3]` must not silently produce `[1.0, 2.718…, 3.0]`.
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn matrix_with_bare_e_does_not_parse() {
    assert_eq!(parse_matrix_literal("[1 e 3]"), None);
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn true_input_returns_logical_not_double() {
    push_queued_response(Ok(InteractionResponse::Line("true".into())));
    let value = input_with_empty_prompt().expect("input");
    assert_eq!(value, Value::Bool(true));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn false_input_returns_logical_not_double() {
    push_queued_response(Ok(InteractionResponse::Line("false".into())));
    let value = input_with_empty_prompt().expect("input");
    assert_eq!(value, Value::Bool(false));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn bool_input_is_case_insensitive() {
    push_queued_response(Ok(InteractionResponse::Line("TRUE".into())));
    let value = input_with_empty_prompt().expect("input");
    assert_eq!(value, Value::Bool(true));
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn column_vector_parses_without_eval_hook() {
    push_queued_response(Ok(InteractionResponse::Line("[1;2;3]".into())));
    let value = input_with_empty_prompt().expect("input");
    match value.value() {
        Value::Tensor(t) => {
            assert_eq!(t.rows(), 3);
            assert_eq!(t.cols(), 1);
            assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 3.0]);
        }
        other => panic!("expected 3×1 tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_row_vector_parses_as_logical_array() {
    push_queued_response(Ok(InteractionResponse::Line("[true false]".into())));
    let value = input_with_empty_prompt().expect("input");
    match value.value() {
        Value::LogicalArray(la) => {
            assert_eq!(la.shape, vec![1, 2]);
            assert_eq!(la.data, vec![1, 0]);
        }
        other => panic!("expected LogicalArray, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_column_vector_parses_as_logical_array() {
    push_queued_response(Ok(InteractionResponse::Line("[true; false]".into())));
    let value = input_with_empty_prompt().expect("input");
    match value.value() {
        Value::LogicalArray(la) => {
            assert_eq!(la.shape, vec![2, 1]);
            assert_eq!(la.data, vec![1, 0]);
        }
        other => panic!("expected LogicalArray, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn mixed_logical_and_numeric_coerces_to_double_tensor() {
    push_queued_response(Ok(InteractionResponse::Line("[true 2.0]".into())));
    let value = input_with_empty_prompt().expect("input");
    match value.value() {
        Value::Tensor(t) => {
            assert_eq!(t.rows(), 1);
            assert_eq!(t.cols(), 2);
            assert_eq!(t.materialize_f64(), vec![1.0, 2.0]);
        }
        other => panic!("expected Tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn matrix_2x2_column_major_layout() {
    // [1 2; 3 4] → get2(r,c) must return element at row r, col c, not the transpose.
    // Column-major storage: data = [1, 3, 2, 4] (not the row-major [1, 2, 3, 4]).
    push_queued_response(Ok(InteractionResponse::Line("[1 2; 3 4]".into())));
    let value = input_with_empty_prompt().expect("input");
    match value.value() {
        Value::Tensor(t) => {
            assert_eq!(t.rows(), 2);
            assert_eq!(t.cols(), 2);
            assert_eq!(t.get2(0, 0).unwrap(), 1.0, "(0,0) should be 1");
            assert_eq!(t.get2(0, 1).unwrap(), 2.0, "(0,1) should be 2");
            assert_eq!(t.get2(1, 0).unwrap(), 3.0, "(1,0) should be 3");
            assert_eq!(t.get2(1, 1).unwrap(), 4.0, "(1,1) should be 4");
        }
        other => panic!("expected 2×2 tensor, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn logical_matrix_2x2_column_major_layout() {
    // [true false; false true] → column-major data = [1, 0, 0, 1].
    push_queued_response(Ok(InteractionResponse::Line(
        "[true false; false true]".into(),
    )));
    let value = input_with_empty_prompt().expect("input");
    match value.value() {
        Value::LogicalArray(la) => {
            assert_eq!(la.shape, vec![2, 2]);
            // column-major: col 0 first ([true, false]), then col 1 ([false, true])
            assert_eq!(la.data, vec![1, 0, 0, 1]);
        }
        other => panic!("expected 2×2 LogicalArray, got {other:?}"),
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
#[test]
fn invalid_string_flag_errors_before_prompt() {
    push_queued_response(Ok(InteractionResponse::Line("ignored".into())));
    let prompt = Value::String("Ready?".to_string().into());
    let bad_flag = Value::String("not-string-mode".to_string().into());
    let err = futures::executor::block_on(input_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![prompt, bad_flag]),
    ))
    .unwrap_err();
    assert_eq!(err.identifier(), Some("RunMat:input:InvalidStringFlag"));
}

#[test]
fn strict_mode_gates_runmat_only_call_forms_before_interaction() {
    let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
    let no_prompt = futures::executor::block_on(input_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![]),
    ))
    .unwrap_err();
    assert_eq!(
        no_prompt.identifier(),
        INPUT_NO_PROMPT_EXTENSION.error_identifier
    );
    let swapped = futures::executor::block_on(input_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("s".into()),
            Value::String("Prompt: ".into()),
        ]),
    ))
    .unwrap_err();
    assert_eq!(
        swapped.identifier(),
        INPUT_SWAPPED_ARGUMENTS_EXTENSION.error_identifier
    );
}

#[test]
fn resident_prompt_rejects_without_provider_access() {
    let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
        .allocate_zeroed(
            vec![1, 1],
            runmat_accelerate_api::NumericElementType::F64,
            runmat_accelerate_api::GpuTensorStorage::Real,
        )
        .expect("genuine unregistered allocation");
    let original = allocation.handle.wire_descriptor();
    assert!(
        crate::builtins::common::gpu_helpers::exact_provider_for_handle(&allocation.handle)
            .is_none()
    );
    let resident = Value::GpuTensor(allocation.handle.clone());
    let error = futures::executor::block_on(input_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![resident]),
    ))
    .unwrap_err();
    assert_eq!(
        error.identifier(),
        INPUT_ERROR_INVALID_PROMPT_TYPE.identifier
    );
    assert_eq!(allocation.handle.wire_descriptor(), original);
    assert_eq!(allocation.probe.retirements(), 0);
    drop(allocation.handle);
    assert_eq!(allocation.probe.retirements(), 1);
}

#[test]
fn input_integer_audit_is_explicitly_not_applicable() {
    assert_eq!(
        INPUT_INTEGER_AUDIT.kind,
        BuiltinIntegerAuditKind::NotApplicable
    );
    assert_eq!(INPUT_EXTENSIONS.len(), 2);
}

#[test]
fn paged_prompt_and_flag_reject_before_interaction_but_trailing_singleton_works() {
    let paged = CharArray::from_unicode_scalars_with_shape("ab".chars().collect(), vec![1, 1, 2])
        .expect("paged char");
    let err = futures::executor::block_on(input_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::CharArray(paged)]),
    ))
    .expect_err("paged prompt must fail");
    assert_eq!(err.identifier(), INPUT_ERROR_PROMPT_ROW_VECTOR.identifier);
    let paged_flag =
        CharArray::from_unicode_scalars_with_shape("ss".chars().collect(), vec![1, 1, 2])
            .expect("paged flag");
    let err = futures::executor::block_on(input_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![
            Value::String("Prompt: ".into()),
            Value::CharArray(paged_flag),
        ]),
    ))
    .expect_err("paged flag must fail");
    assert_eq!(err.identifier(), INPUT_ERROR_INVALID_STRING_FLAG.identifier);
    let trailing = CharArray::from_unicode_scalars_with_shape("s".chars().collect(), vec![1, 1, 1])
        .expect("trailing singleton");
    assert_eq!(
        futures::executor::block_on(parse_string_flag(&Value::CharArray(trailing)))
            .expect("valid flag"),
        InputStringMode::RawText
    );
}

#[test]
fn excessive_outputs_reject_before_interaction() {
    let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
    let err = futures::executor::block_on(input_builtin(
        crate::builtins::common::test_support::builtin_arguments(vec![Value::String(
            "Prompt: ".into(),
        )]),
    ))
    .expect_err("too many outputs");
    assert_eq!(err.identifier(), INPUT_ERROR_TOO_MANY_OUTPUTS.identifier);
}
