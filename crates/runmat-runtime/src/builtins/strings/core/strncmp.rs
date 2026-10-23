//! MATLAB-compatible `strncmp` builtin for RunMat.

use runmat_builtins::{
    BuiltinErrorDescriptor, STRNCMP_DESCRIPTOR, STRNCMP_ERROR_INTERNAL,
    STRNCMP_ERROR_INVALID_INPUT, STRNCMP_ERROR_INVALID_PREFIX_LENGTH, STRNCMP_ERROR_SHAPE_MISMATCH,
    STRNCMP_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::{IntValue, Value};

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};
use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::tensor;
use crate::builtins::strings::common::contains_numeric_or_resident_text_input;
use crate::builtins::strings::core::comparison_observation::{
    host_string_lengths, ComparisonMode, PrefixCarrier, PrefixFacts, ScalarComparisonWorkload,
};
use crate::builtins::strings::search::text_utils::{logical_result, TextCollection, TextElement};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const FN_NAME: &str = "strncmp";

fn strncmp_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    strncmp_error_with_message(error.message, error)
}

fn strncmp_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(FN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn remap_strncmp_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, FN_NAME)
}

#[runtime_builtin(
    name = "strncmp",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::core::strncmp",
    requested_outputs = true,
    call_observation = true
)]
async fn strncmp_builtin(
    requested_outputs: Option<usize>,
    observation: Option<crate::dispatcher::direct_call_observation::CallAdmissionSink>,
    a: Value,
    b: Value,
    n: Value,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &STRNCMP_DESCRIPTOR,
        &STRNCMP_ERROR_TOO_MANY_OUTPUTS,
        FN_NAME,
    )?;
    if contains_numeric_or_resident_text_input(&a) || contains_numeric_or_resident_text_input(&b) {
        return Ok(Value::Bool(false));
    }
    let host_lengths = host_string_lengths(&a, &b);
    let prefix_carrier = PrefixCarrier::immediate(&n);
    let a = gather_if_needed_async(&a)
        .await
        .map_err(remap_strncmp_flow)?;
    let b = gather_if_needed_async(&b)
        .await
        .map_err(remap_strncmp_flow)?;
    let n = gather_if_needed_async(&n)
        .await
        .map_err(remap_strncmp_flow)?;

    let limit = parse_prefix_length(n)?;
    let left = TextCollection::from_argument(FN_NAME, a, "first argument")
        .map_err(|_| strncmp_error(&STRNCMP_ERROR_INVALID_INPUT))?;
    let right = TextCollection::from_argument(FN_NAME, b, "second argument")
        .map_err(|_| strncmp_error(&STRNCMP_ERROR_INVALID_INPUT))?;
    evaluate_strncmp(
        &left,
        &right,
        limit,
        host_lengths,
        prefix_carrier,
        requested_outputs,
        observation,
    )
}

fn evaluate_strncmp(
    left: &TextCollection,
    right: &TextCollection,
    limit: usize,
    host_lengths: Option<(usize, usize)>,
    prefix_carrier: Option<PrefixCarrier>,
    requested_outputs: Option<usize>,
    observation: Option<crate::dispatcher::direct_call_observation::CallAdmissionSink>,
) -> BuiltinResult<Value> {
    let shape = broadcast_shapes(FN_NAME, &left.shape, &right.shape)
        .map_err(|_| strncmp_error(&STRNCMP_ERROR_SHAPE_MISMATCH))?;
    let total = tensor::element_count(&shape);
    if total == 1 && left.shape == [1, 1] && right.shape == [1, 1] {
        if let (Some((left_utf8_bytes, right_utf8_bytes)), Some(carrier), Some(observation)) =
            (host_lengths, prefix_carrier, observation)
        {
            observation.admit(ScalarComparisonWorkload {
                left_utf8_bytes,
                right_utf8_bytes,
                mode: ComparisonMode::PrefixSensitive,
                prefix: Some(PrefixFacts {
                    limit_chars: limit,
                    carrier,
                }),
                requested_outputs,
            });
        }
    }
    if total == 0 {
        return logical_result(FN_NAME, Vec::new(), shape)
            .map_err(|_| strncmp_error(&STRNCMP_ERROR_INTERNAL));
    }

    let left_strides = compute_strides(&left.shape);
    let right_strides = compute_strides(&right.shape);
    let mut data = Vec::with_capacity(total);

    for linear in 0..total {
        let li = broadcast_index(linear, &shape, &left.shape, &left_strides);
        let ri = broadcast_index(linear, &shape, &right.shape, &right_strides);
        let equal = if limit == 0 {
            true
        } else {
            match (&left.elements[li], &right.elements[ri]) {
                (TextElement::Missing, _) | (_, TextElement::Missing) => false,
                (TextElement::Text(lhs), TextElement::Text(rhs)) => prefix_equal(lhs, rhs, limit),
            }
        };
        data.push(if equal { 1 } else { 0 });
    }

    logical_result(FN_NAME, data, shape).map_err(|_| strncmp_error(&STRNCMP_ERROR_INTERNAL))
}

fn prefix_equal(lhs: &str, rhs: &str, limit: usize) -> bool {
    if limit == 0 {
        return true;
    }
    let mut lhs_iter = lhs.chars();
    let mut rhs_iter = rhs.chars();
    let mut compared = 0usize;

    while compared < limit {
        let left_char = lhs_iter.next();
        let right_char = rhs_iter.next();
        match (left_char, right_char) {
            (Some(lc), Some(rc)) => {
                if lc != rc {
                    return false;
                }
            }
            (None, Some(_)) | (Some(_), None) => {
                return false;
            }
            (None, None) => {
                return true;
            }
        }
        compared += 1;
    }

    true
}

fn parse_prefix_length(value: Value) -> BuiltinResult<usize> {
    match value {
        Value::Int(i) => parse_prefix_length_from_integer(&i),
        Value::Num(n) => parse_prefix_length_from_float(n),
        Value::Bool(b) => Ok(if b { 1 } else { 0 }),
        Value::Tensor(tensor) => {
            if tensor::tensor_element_len(&tensor) != 1 {
                return Err(strncmp_error(&STRNCMP_ERROR_INVALID_PREFIX_LENGTH));
            }
            if let Some(value) = tensor
                .integer_storage()
                .and_then(|storage| storage.value_at(0))
            {
                return parse_prefix_length_from_integer(&value);
            }
            parse_prefix_length_from_float(tensor::tensor_value_f64(&tensor, 0))
        }
        Value::LogicalArray(array) => {
            if array.data.len() != 1 {
                return Err(strncmp_error(&STRNCMP_ERROR_INVALID_PREFIX_LENGTH));
            }
            Ok(if array.data[0] != 0 { 1 } else { 0 })
        }
        _ => Err(strncmp_error(&STRNCMP_ERROR_INVALID_PREFIX_LENGTH)),
    }
}

fn parse_prefix_length_from_integer(value: &IntValue) -> BuiltinResult<usize> {
    if value.try_to_i64().is_some_and(|value| value < 0) {
        return Ok(0);
    }
    value
        .try_to_usize()
        .ok_or_else(|| strncmp_error(&STRNCMP_ERROR_INVALID_PREFIX_LENGTH))
}

fn parse_prefix_length_from_float(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() || value.fract() != 0.0 {
        return Err(strncmp_error(&STRNCMP_ERROR_INVALID_PREFIX_LENGTH));
    }
    if value < 0.0 {
        return Ok(0);
    }
    if value > usize::MAX as f64 || (usize::BITS == 64 && value == usize::MAX as f64) {
        return Err(strncmp_error(&STRNCMP_ERROR_INVALID_PREFIX_LENGTH));
    }
    Ok(value as usize)
}

#[cfg(test)]
#[path = "strncmp/integrality_tests.rs"]
mod integrality_tests;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};
    use runmat_value::{
        CellArray, CharArray, IntValue, IntegerStorage, LogicalArray, StringArray, Tensor,
    };

    fn strncmp_builtin(a: Value, b: Value, n: Value) -> BuiltinResult<Value> {
        futures::executor::block_on(super::strncmp_builtin(None, None, a, b, n))
    }

    fn error_message(err: crate::RuntimeError) -> String {
        err.to_string()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strncmp_rejects_excess_outputs_before_gathering() {
        let _outputs = crate::output_count::push_output_count(Some(2)).expect("output-count scope");
        let error = strncmp_builtin(
            Value::String("text".into()),
            Value::String("text".into()),
            Value::Int(IntValue::U8(4)),
        )
        .expect_err("fixed output bound");
        assert_eq!(
            error.identifier(),
            STRNCMP_ERROR_TOO_MANY_OUTPUTS.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strncmp_exact_prefix_true() {
        let result = strncmp_builtin(
            Value::String("RunMat".into()),
            Value::String("Runway".into()),
            Value::Int(IntValue::I32(3)),
        )
        .expect("strncmp");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strncmp_mismatch_within_prefix_false() {
        let result = strncmp_builtin(
            Value::String("RunMat".into()),
            Value::String("Runway".into()),
            Value::Int(IntValue::I32(4)),
        )
        .expect("strncmp");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strncmp_longer_string_after_prefix_false() {
        let result = strncmp_builtin(
            Value::String("cat".into()),
            Value::String("cater".into()),
            Value::Int(IntValue::I32(4)),
        )
        .expect("strncmp");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strncmp_zero_length_always_true() {
        let result = strncmp_builtin(
            Value::String("alpha".into()),
            Value::String("omega".into()),
            Value::Num(0.0),
        )
        .expect("strncmp");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn strncmp_gathers_automatic_resident_integer_prefix_length() {
        test_support::with_test_provider(|provider| {
            let count = [3_u64];
            let handle = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&count),
                    shape: &[1, 1],
                })
                .expect("upload exact integer prefix length")
                .with_provenance(runmat_accelerate_api::GpuHandleProvenance::Automatic)
                .expect("automatic residency");
            let result = strncmp_builtin(
                Value::String("delta".into()),
                Value::String("deluge".into()),
                Value::GpuTensor(handle.clone()),
            )
            .expect("gather automatic N before comparison");
            assert_eq!(result, Value::Bool(true));
            handle
                .allocation_lease()
                .validate_live()
                .expect("input owner remains live after borrowed gather");
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strncmp_prefix_length_bool_true_compares_first_character() {
        let result = strncmp_builtin(
            Value::String("alpha".into()),
            Value::String("array".into()),
            Value::Bool(true),
        )
        .expect("strncmp");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strncmp_prefix_length_bool_false_treated_as_zero() {
        let result = strncmp_builtin(
            Value::String("alpha".into()),
            Value::String("omega".into()),
            Value::Bool(false),
        )
        .expect("strncmp");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strncmp_prefix_length_logical_array_scalar() {
        let logical = LogicalArray::new(vec![1], vec![1]).unwrap();
        let result = strncmp_builtin(
            Value::String("beta".into()),
            Value::String("theta".into()),
            Value::LogicalArray(logical),
        )
        .expect("strncmp");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strncmp_prefix_length_tensor_scalar_double() {
        let limit = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let result = strncmp_builtin(
            Value::String("gamma".into()),
            Value::String("gamut".into()),
            Value::Tensor(limit),
        )
        .expect("strncmp");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strncmp_prefix_length_typed_integer_tensor_reads_exact_storage() {
        let limit =
            Tensor::new_integer(IntegerStorage::U64(vec![2]), vec![1, 1]).expect("integer tensor");
        let result = strncmp_builtin(
            Value::String("gamma".into()),
            Value::String("gamut".into()),
            Value::Tensor(limit),
        )
        .expect("strncmp");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strncmp_char_array_rows() {
        let chars = CharArray::from_unicode_scalars(
            vec![
                'c', 'a', 't', ' ', ' ', 'c', 'a', 'm', 'e', 'l', 'c', 'o', 'w', ' ', ' ',
            ],
            3,
            5,
        )
        .unwrap();
        let result = strncmp_builtin(
            Value::CharArray(chars),
            Value::String("ca".into()),
            Value::Int(IntValue::I32(2)),
        )
        .expect("strncmp");
        let expected = LogicalArray::new(vec![1, 1, 0], vec![3, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strncmp_cell_arrays_broadcast() {
        let left = CellArray::new(
            vec![
                Value::from("red"),
                Value::from("green"),
                Value::from("blue"),
            ],
            1,
            3,
        )
        .unwrap();
        let right = CellArray::new(
            vec![
                Value::from("rose"),
                Value::from("gray"),
                Value::from("black"),
            ],
            1,
            3,
        )
        .unwrap();
        let result = strncmp_builtin(
            Value::Cell(left),
            Value::Cell(right),
            Value::Int(IntValue::I32(2)),
        )
        .expect("strncmp");
        let expected = LogicalArray::new(vec![0, 1, 1], vec![1, 3]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strncmp_string_array_broadcast_scalar() {
        let strings = StringArray::new(
            vec!["north".into(), "south".into(), "east".into()],
            vec![1, 3],
        )
        .unwrap();
        let result = strncmp_builtin(
            Value::StringArray(strings),
            Value::String("no".into()),
            Value::Int(IntValue::I32(2)),
        )
        .expect("strncmp");
        let expected = LogicalArray::new(vec![1, 0, 0], vec![1, 3]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strncmp_missing_string_false_when_prefix_positive() {
        let strings =
            StringArray::new(vec!["<missing>".into(), "value".into()], vec![1, 2]).unwrap();
        let result = strncmp_builtin(
            Value::StringArray(strings),
            Value::String("val".into()),
            Value::Int(IntValue::I32(3)),
        )
        .expect("strncmp");
        let expected = LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strncmp_missing_zero_length_true() {
        let strings = StringArray::new(vec!["<missing>".into()], vec![1, 1]).unwrap();
        let result = strncmp_builtin(
            Value::StringArray(strings),
            Value::String("anything".into()),
            Value::Int(IntValue::I32(0)),
        )
        .expect("strncmp");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strncmp_size_mismatch_error() {
        let left = StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap();
        let right = StringArray::new(vec!["a".into(), "b".into(), "c".into()], vec![3, 1]).unwrap();
        let err = error_message(
            strncmp_builtin(
                Value::StringArray(left),
                Value::StringArray(right),
                Value::Int(IntValue::I32(1)),
            )
            .expect_err("size mismatch"),
        );
        assert!(err.contains(STRNCMP_ERROR_SHAPE_MISMATCH.message));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strncmp_invalid_length_type_errors() {
        let err = error_message(
            strncmp_builtin(
                Value::String("abc".into()),
                Value::String("abc".into()),
                Value::String("3".into()),
            )
            .expect_err("invalid prefix length"),
        );
        assert!(err.contains("prefix length"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strncmp_negative_length_is_treated_as_zero() {
        let result = strncmp_builtin(
            Value::String("abc".into()),
            Value::String("xyz".into()),
            Value::Num(-1.0),
        )
        .expect("negative length becomes zero");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn strncmp_prefix_length_treats_negative_typed_integer_tensor_as_zero() {
        let limit =
            Tensor::new_integer(IntegerStorage::I64(vec![-1]), vec![1, 1]).expect("integer tensor");
        let result = strncmp_builtin(
            Value::String("abc".into()),
            Value::String("xyz".into()),
            Value::Tensor(limit),
        )
        .expect("negative length becomes zero");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn strncmp_prefix_length_rejects_unrepresentable_double_boundary() {
        let boundary = if usize::BITS == 64 {
            usize::MAX as f64
        } else {
            (usize::MAX as f64) + 1.0
        };
        let err = error_message(
            strncmp_builtin(
                Value::String("abc".into()),
                Value::String("abc".into()),
                Value::Num(boundary),
            )
            .expect_err("unrepresentable prefix length"),
        );
        assert!(err.to_ascii_lowercase().contains("prefix length"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn strncmp_prefix_length_from_gpu_tensor() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };
        use runmat_accelerate_api::HostTensorView;

        let provider = match register_wgpu_provider(WgpuProviderOptions::default()) {
            Ok(provider) => provider,
            Err(_) => return,
        };
        let tensor = Tensor::new(vec![3.0], vec![1, 1]).unwrap();
        let view = HostTensorView {
            data: &tensor.materialize_f64(),
            shape: tensor.shape(),
        };
        let handle = provider.upload(&view).expect("upload prefix length to GPU");
        let result = strncmp_builtin(
            Value::String("delta".into()),
            Value::String("deluge".into()),
            Value::GpuTensor(handle.clone()),
        )
        .expect("strncmp");
        assert_eq!(result, Value::Bool(true));
        handle
            .allocation_lease()
            .validate_live()
            .expect("borrowed input remains live");
        let physical = handle.allocation_lease().downgrade();
        handle.release().expect("release final fixture reference");
        assert!(
            physical.acquire().is_none(),
            "physical lease must expire after final release"
        );
    }
}
