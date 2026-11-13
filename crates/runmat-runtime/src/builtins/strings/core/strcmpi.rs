//! MATLAB-compatible `strcmpi` builtin for RunMat (case-insensitive string comparison).

use runmat_builtins::{
    BuiltinErrorDescriptor, STRCMPI_DESCRIPTOR, STRCMPI_ERROR_INTERNAL,
    STRCMPI_ERROR_INVALID_INPUT, STRCMPI_ERROR_SHAPE_MISMATCH, STRCMPI_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};
use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::tensor;
use crate::builtins::strings::common::contains_numeric_or_resident_text_input;
use crate::builtins::strings::core::comparison_observation::{
    host_string_lengths, ComparisonMode, ScalarComparisonWorkload,
};
use crate::builtins::strings::search::text_utils::{logical_result, TextCollection, TextElement};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "strcmpi";

fn strcmpi_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    strcmpi_error_with_message(error.message, error)
}

fn strcmpi_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn remap_strcmpi_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

#[runtime_builtin(
    name = "strcmpi",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::core::strcmpi",
    requested_outputs = true,
    call_observation = true
)]
async fn strcmpi_builtin(
    requested_outputs: Option<usize>,
    observation: Option<crate::dispatcher::direct_call_observation::CallAdmissionSink>,
    a: Value,
    b: Value,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &STRCMPI_DESCRIPTOR,
        &STRCMPI_ERROR_TOO_MANY_OUTPUTS,
        BUILTIN_NAME,
    )?;
    if contains_numeric_or_resident_text_input(&a) || contains_numeric_or_resident_text_input(&b) {
        return Ok(Value::Bool(false));
    }
    let host_lengths = host_string_lengths(&a, &b);
    let a = gather_if_needed_async(&a)
        .await
        .map_err(remap_strcmpi_flow)?;
    let b = gather_if_needed_async(&b)
        .await
        .map_err(remap_strcmpi_flow)?;
    let left = TextCollection::from_argument(BUILTIN_NAME, a, "first argument")
        .map_err(|_| strcmpi_error(&STRCMPI_ERROR_INVALID_INPUT))?;
    let right = TextCollection::from_argument(BUILTIN_NAME, b, "second argument")
        .map_err(|_| strcmpi_error(&STRCMPI_ERROR_INVALID_INPUT))?;
    evaluate_strcmpi(&left, &right, host_lengths, requested_outputs, observation)
}

fn evaluate_strcmpi(
    left: &TextCollection,
    right: &TextCollection,
    host_lengths: Option<(usize, usize)>,
    requested_outputs: Option<usize>,
    observation: Option<crate::dispatcher::direct_call_observation::CallAdmissionSink>,
) -> BuiltinResult<Value> {
    let shape = broadcast_shapes(BUILTIN_NAME, &left.shape, &right.shape)
        .map_err(|_| strcmpi_error(&STRCMPI_ERROR_SHAPE_MISMATCH))?;
    let total = tensor::element_count(&shape);
    if total == 1 && left.shape == [1, 1] && right.shape == [1, 1] {
        if let (Some((left_utf8_bytes, right_utf8_bytes)), Some(observation)) =
            (host_lengths, observation)
        {
            observation.admit(ScalarComparisonWorkload {
                left_utf8_bytes,
                right_utf8_bytes,
                mode: ComparisonMode::CaseInsensitive,
                prefix: None,
                requested_outputs,
            });
        }
    }
    if total == 0 {
        return logical_result(BUILTIN_NAME, Vec::new(), shape)
            .map_err(|_| strcmpi_error(&STRCMPI_ERROR_INTERNAL));
    }
    let left_strides = compute_strides(&left.shape);
    let right_strides = compute_strides(&right.shape);
    let left_lower = left.lowercased();
    let right_lower = right.lowercased();
    let mut data = Vec::with_capacity(total);
    for linear in 0..total {
        let li = broadcast_index(linear, &shape, &left.shape, &left_strides);
        let ri = broadcast_index(linear, &shape, &right.shape, &right_strides);
        let equal = match (&left.elements[li], &right.elements[ri]) {
            (TextElement::Missing, _) => false,
            (_, TextElement::Missing) => false,
            (TextElement::Text(_), TextElement::Text(_)) => {
                match (&left_lower[li], &right_lower[ri]) {
                    (Some(lhs), Some(rhs)) => lhs == rhs,
                    _ => false,
                }
            }
        };
        data.push(if equal { 1 } else { 0 });
    }
    logical_result(BUILTIN_NAME, data, shape).map_err(|_| strcmpi_error(&STRCMPI_ERROR_INTERNAL))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::RuntimeError;
    use runmat_value::{CellArray, CharArray, LogicalArray, StringArray};

    fn strcmpi_builtin(a: Value, b: Value) -> BuiltinResult<Value> {
        futures::executor::block_on(super::strcmpi_builtin(None, None, a, b))
    }

    fn error_message(err: RuntimeError) -> String {
        err.to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_rejects_excess_outputs_before_comparing() {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = strcmpi_builtin(Value::String("a".into()), Value::String("A".into()))
            .expect_err("fixed output bound");
        assert_eq!(
            error.identifier(),
            STRCMPI_ERROR_TOO_MANY_OUTPUTS.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_string_scalar_true_ignores_case() {
        let result = strcmpi_builtin(
            Value::String("RunMat".into()),
            Value::String("runmat".into()),
        )
        .expect("strcmpi");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_string_scalar_false_when_text_differs() {
        let result = strcmpi_builtin(
            Value::String("RunMat".into()),
            Value::String("runtime".into()),
        )
        .expect("strcmpi");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_string_array_broadcast_scalar_case_insensitive() {
        let array = StringArray::new(
            vec!["red".into(), "green".into(), "blue".into()],
            vec![1, 3],
        )
        .unwrap();
        let result = strcmpi_builtin(Value::StringArray(array), Value::String("GREEN".into()))
            .expect("strcmpi");
        let expected = LogicalArray::new(vec![0, 1, 0], vec![1, 3]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_char_array_row_compare_casefold() {
        let chars =
            CharArray::from_unicode_scalars(vec!['c', 'a', 't', 'D', 'O', 'G'], 2, 3).unwrap();
        let result =
            strcmpi_builtin(Value::CharArray(chars), Value::String("CaT".into())).expect("cmp");
        let expected = LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_char_array_to_char_array_casefold() {
        let left = CharArray::from_unicode_scalars(vec!['A', 'b', 'C', 'd'], 2, 2).unwrap();
        let right = CharArray::from_unicode_scalars(vec!['a', 'B', 'x', 'Y'], 2, 2).unwrap();
        let result =
            strcmpi_builtin(Value::CharArray(left), Value::CharArray(right)).expect("strcmpi");
        let expected = LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_cell_array_scalar_casefold() {
        let cell = CellArray::new(
            vec![
                Value::from("North"),
                Value::from("east"),
                Value::from("South"),
            ],
            1,
            3,
        )
        .unwrap();
        let result =
            strcmpi_builtin(Value::Cell(cell), Value::String("EAST".into())).expect("strcmpi");
        let expected = LogicalArray::new(vec![0, 1, 0], vec![1, 3]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_cell_array_vs_cell_array_broadcast() {
        let left = CellArray::new(vec![Value::from("North"), Value::from("East")], 1, 2).unwrap();
        let right = CellArray::new(vec![Value::from("north")], 1, 1).unwrap();
        let result = strcmpi_builtin(Value::Cell(left), Value::Cell(right)).expect("strcmpi");
        let expected = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_string_array_multi_dimensional_broadcast() {
        let left = StringArray::new(vec!["north".into(), "south".into()], vec![2, 1]).unwrap();
        let right = StringArray::new(
            vec!["NORTH".into(), "EAST".into(), "SOUTH".into()],
            vec![1, 3],
        )
        .unwrap();
        let result =
            strcmpi_builtin(Value::StringArray(left), Value::StringArray(right)).expect("strcmpi");
        let expected = LogicalArray::new(vec![1, 0, 0, 0, 0, 1], vec![2, 3]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_missing_strings_compare_false() {
        let strings = StringArray::from_elements_unbounded(
            vec![runmat_value::StringElement::Missing],
            vec![1, 1],
        )
        .unwrap();
        let result = strcmpi_builtin(
            Value::StringArray(strings.clone()),
            Value::StringArray(strings),
        )
        .expect("strcmpi");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_char_array_trailing_space_not_equal() {
        let chars = CharArray::from_unicode_scalars(vec!['c', 'a', 't', ' '], 1, 4).unwrap();
        let result =
            strcmpi_builtin(Value::CharArray(chars), Value::String("cat".into())).expect("strcmpi");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_size_mismatch_error() {
        let left = StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap();
        let right = StringArray::new(vec!["a".into(), "b".into(), "c".into()], vec![3, 1]).unwrap();
        let err = error_message(
            strcmpi_builtin(Value::StringArray(left), Value::StringArray(right))
                .expect_err("size mismatch"),
        );
        assert!(err.contains(STRCMPI_ERROR_SHAPE_MISMATCH.message));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_unsupported_numeric_argument_returns_false() {
        let result =
            strcmpi_builtin(Value::Num(1.0), Value::String("a".into())).expect("comparison");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_direct_numeric_false_precedes_invalid_peer_container() {
        let malformed = Value::Struct(runmat_value::StructValue::new());
        for (a, b) in [
            (Value::Num(1.0), malformed.clone()),
            (malformed, Value::Num(1.0)),
        ] {
            assert_eq!(strcmpi_builtin(a, b).unwrap(), Value::Bool(false));
        }
        let malformed_cell = CellArray::new(vec![Value::Num(42.0)], 1, 1).unwrap();
        let error = strcmpi_builtin(Value::Cell(malformed_cell), Value::String("a".into()))
            .expect_err("invalid host cell without direct numeric input");
        assert_eq!(error.identifier(), STRCMPI_ERROR_INVALID_INPUT.identifier);
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        assert_eq!(
            strcmpi_builtin(
                Value::GpuTensor(allocation.handle.clone()),
                Value::Struct(runmat_value::StructValue::new()),
            )
            .unwrap(),
            Value::Bool(false)
        );
        assert!(allocation.handle.allocation_lease().validate_live().is_ok());
        assert_eq!(allocation.probe.retirements(), 0);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_cell_array_invalid_element_errors() {
        let cell = CellArray::new(vec![Value::Num(42.0)], 1, 1).unwrap();
        let err = error_message(
            strcmpi_builtin(Value::Cell(cell), Value::String("test".into()))
                .expect_err("cell element type"),
        );
        assert!(err.contains(STRCMPI_ERROR_INVALID_INPUT.message));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_empty_char_array_returns_empty() {
        let chars = CharArray::from_code_units(Vec::new(), 0, 3).unwrap();
        let result = strcmpi_builtin(Value::CharArray(chars), Value::String("anything".into()))
            .expect("cmp");
        let expected = LogicalArray::new(Vec::<u8>::new(), vec![0, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn strcmpi_with_wgpu_provider_matches_expected() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let names = StringArray::new(vec!["North".into(), "south".into()], vec![2, 1]).unwrap();
        let comparison = StringArray::new(vec!["north".into()], vec![1, 1]).unwrap();
        let result = strcmpi_builtin(Value::StringArray(names), Value::StringArray(comparison))
            .expect("strcmpi");
        let expected = LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }
}
