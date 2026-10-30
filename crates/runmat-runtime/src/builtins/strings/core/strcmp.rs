//! MATLAB-compatible `strcmp` builtin for RunMat.

use runmat_builtins::{
    BuiltinErrorDescriptor, STRCMP_DESCRIPTOR, STRCMP_ERROR_INTERNAL, STRCMP_ERROR_INVALID_INPUT,
    STRCMP_ERROR_SHAPE_MISMATCH, STRCMP_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};
use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::tensor;
use crate::builtins::strings::common::contains_numeric_or_resident_text_input;
use crate::builtins::strings::search::text_utils::{logical_result, TextCollection, TextElement};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "strcmp";

#[derive(serde::Serialize)]
pub(crate) struct ScalarStringCompareWorkload {
    pub left_utf8_bytes: usize,
    pub right_utf8_bytes: usize,
    pub requested_outputs: Option<usize>,
}

impl crate::dispatcher::direct_call_observation::HostCallDomainFacts
    for ScalarStringCompareWorkload
{
    const ROUTE_ID: &'static [u8] = b"runmat.strcmp.host-string-scalar.v1";

    fn requested_outputs(&self) -> Option<usize> {
        self.requested_outputs
    }
}

fn strcmp_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    strcmp_error_with_message(error.message, error)
}

fn strcmp_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn remap_strcmp_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

#[runtime_builtin(
    name = "strcmp",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::core::strcmp",
    requested_outputs = true,
    call_observation = true
)]
async fn strcmp_builtin(
    requested_outputs: Option<usize>,
    observation: Option<crate::dispatcher::direct_call_observation::CallAdmissionSink>,
    a: Value,
    b: Value,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &STRCMP_DESCRIPTOR,
        &STRCMP_ERROR_TOO_MANY_OUTPUTS,
        BUILTIN_NAME,
    )?;
    if contains_numeric_or_resident_text_input(&a) || contains_numeric_or_resident_text_input(&b) {
        return Ok(Value::Bool(false));
    }
    let host_scalar_lengths = match (&a, &b) {
        (Value::String(left), Value::String(right)) => left
            .as_str()
            .zip(right.as_str())
            .map(|(left, right)| (left.len(), right.len())),
        _ => None,
    };
    let a = gather_if_needed_async(&a)
        .await
        .map_err(remap_strcmp_flow)?;
    let b = gather_if_needed_async(&b)
        .await
        .map_err(remap_strcmp_flow)?;
    let left = TextCollection::from_argument(BUILTIN_NAME, a, "first argument")
        .map_err(|_| strcmp_error(&STRCMP_ERROR_INVALID_INPUT))?;
    let right = TextCollection::from_argument(BUILTIN_NAME, b, "second argument")
        .map_err(|_| strcmp_error(&STRCMP_ERROR_INVALID_INPUT))?;
    evaluate_strcmp(
        &left,
        &right,
        host_scalar_lengths,
        requested_outputs,
        observation,
    )
}

fn evaluate_strcmp(
    left: &TextCollection,
    right: &TextCollection,
    host_scalar_lengths: Option<(usize, usize)>,
    requested_outputs: Option<usize>,
    observation: Option<crate::dispatcher::direct_call_observation::CallAdmissionSink>,
) -> BuiltinResult<Value> {
    let shape = broadcast_shapes(BUILTIN_NAME, &left.shape, &right.shape)
        .map_err(|_| strcmp_error(&STRCMP_ERROR_SHAPE_MISMATCH))?;
    let total = tensor::element_count(&shape);
    if total == 1 && left.shape == [1, 1] && right.shape == [1, 1] {
        if let (Some((left_utf8_bytes, right_utf8_bytes)), Some(observation)) =
            (host_scalar_lengths, observation)
        {
            observation.admit(ScalarStringCompareWorkload {
                left_utf8_bytes,
                right_utf8_bytes,
                requested_outputs,
            });
        }
    }
    if total == 0 {
        return logical_result(BUILTIN_NAME, Vec::new(), shape)
            .map_err(|_| strcmp_error(&STRCMP_ERROR_INTERNAL));
    }
    let left_strides = compute_strides(&left.shape);
    let right_strides = compute_strides(&right.shape);
    let mut data = Vec::with_capacity(total);
    for linear in 0..total {
        let li = broadcast_index(linear, &shape, &left.shape, &left_strides);
        let ri = broadcast_index(linear, &shape, &right.shape, &right_strides);
        let equal = match (&left.elements[li], &right.elements[ri]) {
            (TextElement::Missing, _) => false,
            (_, TextElement::Missing) => false,
            (TextElement::Text(lhs), TextElement::Text(rhs)) => lhs == rhs,
        };
        data.push(if equal { 1 } else { 0 });
    }
    logical_result(BUILTIN_NAME, data, shape).map_err(|_| strcmp_error(&STRCMP_ERROR_INTERNAL))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::RuntimeError;
    use runmat_value::{CellArray, CharArray, LogicalArray, StringArray};

    fn strcmp_builtin(a: Value, b: Value) -> BuiltinResult<Value> {
        futures::executor::block_on(super::strcmp_builtin(None, None, a, b))
    }

    fn error_message(err: RuntimeError) -> String {
        err.to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_rejects_excess_outputs_before_comparing() {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = strcmp_builtin(Value::String("a".into()), Value::String("a".into()))
            .expect_err("fixed output bound");
        assert_eq!(error.identifier(), STRCMP_ERROR_TOO_MANY_OUTPUTS.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_string_scalar_true() {
        let result = strcmp_builtin(
            Value::String("RunMat".into()),
            Value::String("RunMat".into()),
        )
        .expect("strcmp");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_string_scalar_false() {
        let result = strcmp_builtin(
            Value::String("RunMat".into()),
            Value::String("runmat".into()),
        )
        .expect("strcmp");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_string_array_broadcast_scalar() {
        let array = StringArray::new(
            vec!["red".into(), "green".into(), "blue".into()],
            vec![1, 3],
        )
        .unwrap();
        let result =
            strcmp_builtin(Value::StringArray(array), Value::String("green".into())).expect("cmp");
        let expected = LogicalArray::new(vec![0, 1, 0], vec![1, 3]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_char_array_row_compare() {
        let chars =
            CharArray::from_unicode_scalars(vec!['c', 'a', 't', 'd', 'o', 'g'], 2, 3).unwrap();
        let result =
            strcmp_builtin(Value::CharArray(chars), Value::String("cat".into())).expect("cmp");
        let expected = LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_char_array_to_char_array() {
        let left = CharArray::from_unicode_scalars(vec!['a', 'b', 'c', 'd'], 2, 2).unwrap();
        let right = CharArray::from_unicode_scalars(vec!['a', 'b', 'x', 'y'], 2, 2).unwrap();
        let result =
            strcmp_builtin(Value::CharArray(left), Value::CharArray(right)).expect("strcmp");
        let expected = LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_cell_array_scalar() {
        let cell = CellArray::new(
            vec![
                Value::from("apple"),
                Value::from("pear"),
                Value::from("grape"),
            ],
            1,
            3,
        )
        .unwrap();
        let result =
            strcmp_builtin(Value::Cell(cell), Value::String("grape".into())).expect("strcmp");
        let expected = LogicalArray::new(vec![0, 0, 1], vec![1, 3]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_cell_array_to_cell_array_broadcasts() {
        let left = CellArray::new(vec![Value::from("red"), Value::from("blue")], 2, 1).unwrap();
        let right = CellArray::new(vec![Value::from("red")], 1, 1).unwrap();
        let result = strcmp_builtin(Value::Cell(left), Value::Cell(right)).expect("strcmp");
        let expected = LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_string_array_multi_dimensional_broadcast() {
        let left = StringArray::new(vec!["north".into(), "south".into()], vec![2, 1]).unwrap();
        let right = StringArray::new(
            vec!["north".into(), "east".into(), "south".into()],
            vec![1, 3],
        )
        .unwrap();
        let result =
            strcmp_builtin(Value::StringArray(left), Value::StringArray(right)).expect("strcmp");
        let expected = LogicalArray::new(vec![1, 0, 0, 0, 0, 1], vec![2, 3]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_char_array_trailing_space_is_not_equal() {
        let chars = CharArray::from_unicode_scalars(vec!['c', 'a', 't', ' '], 1, 4).unwrap();
        let result =
            strcmp_builtin(Value::CharArray(chars), Value::String("cat".into())).expect("strcmp");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_char_array_empty_rows_returns_empty() {
        let chars = CharArray::from_code_units(Vec::new(), 0, 0).unwrap();
        let result = strcmp_builtin(Value::CharArray(chars), Value::String("anything".into()))
            .expect("strcmp");
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![0, 1]);
                assert!(array.data.is_empty());
            }
            other => panic!("expected empty logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_missing_strings_compare_false() {
        let strings = StringArray::from_elements_unbounded(
            vec![runmat_value::StringElement::Missing],
            vec![1, 1],
        )
        .unwrap();
        let result = strcmp_builtin(
            Value::StringArray(strings.clone()),
            Value::StringArray(strings),
        )
        .expect("strcmp");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_missing_string_false() {
        let array = StringArray::from_elements_unbounded(
            vec!["alpha".into(), runmat_value::StringElement::Missing],
            vec![1, 2],
        )
        .unwrap();
        let result =
            strcmp_builtin(Value::StringArray(array), Value::String("alpha".into())).expect("cmp");
        let expected = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_size_mismatch_error() {
        let left = StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap();
        let right = StringArray::new(vec!["a".into(), "b".into(), "c".into()], vec![3, 1]).unwrap();
        let err = error_message(
            strcmp_builtin(Value::StringArray(left), Value::StringArray(right))
                .expect_err("size mismatch"),
        );
        assert!(err.contains(STRCMP_ERROR_SHAPE_MISMATCH.message));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_unsupported_numeric_argument_returns_false() {
        let result =
            strcmp_builtin(Value::Num(1.0), Value::String("a".into())).expect("comparison");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_direct_numeric_false_precedes_invalid_peer_container() {
        let malformed = Value::Struct(runmat_value::StructValue::new());
        for (a, b) in [
            (Value::Num(1.0), malformed.clone()),
            (malformed, Value::Num(1.0)),
        ] {
            assert_eq!(strcmp_builtin(a, b).unwrap(), Value::Bool(false));
        }
        let malformed_cell = CellArray::new(vec![Value::Num(42.0)], 1, 1).unwrap();
        let error = strcmp_builtin(Value::Cell(malformed_cell), Value::String("a".into()))
            .expect_err("invalid host cell without direct numeric input");
        assert_eq!(error.identifier(), STRCMP_ERROR_INVALID_INPUT.identifier);
        let allocation = runmat_accelerate_api::test_support::TestAllocationOwner::new()
            .allocate_zeroed(
                vec![1, 1],
                runmat_accelerate_api::NumericElementType::F64,
                runmat_accelerate_api::GpuTensorStorage::Real,
            )
            .unwrap();
        assert_eq!(
            strcmp_builtin(
                Value::GpuTensor(allocation.handle.clone()),
                Value::Struct(runmat_value::StructValue::new()),
            )
            .unwrap(),
            Value::Bool(false)
        );
        assert!(allocation.handle.allocation_lease().validate_live().is_ok());
        assert_eq!(allocation.probe.retirements(), 0);
    }
}
