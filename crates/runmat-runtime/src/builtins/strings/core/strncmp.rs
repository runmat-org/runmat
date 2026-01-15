//! MATLAB-compatible `strncmp` builtin for RunMat.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::strings::search::text_utils::{logical_result, TextCollection, TextElement};
use crate::gather_if_needed;

const FN_NAME: &str = "strncmp";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::core::strncmp")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "strncmp",
    op_kind: GpuOpKind::Custom("string-prefix-compare"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Performs host-side prefix comparisons; GPU inputs are gathered before evaluation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::core::strncmp")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "strncmp",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Produces logical host results and is not eligible for GPU fusion.",
};

#[runtime_builtin(
    name = "strncmp",
    category = "strings/core",
    summary = "Compare text inputs for equality up to N leading characters (case-sensitive).",
    keywords = "strncmp,string compare,prefix,text equality",
    accel = "sink",
    builtin_path = "crate::builtins::strings::core::strncmp"
)]
fn strncmp_builtin(a: Value, b: Value, n: Value) -> Result<Value, String> {
    let a = gather_if_needed(&a).map_err(|e| format!("{FN_NAME}: {e}"))?;
    let b = gather_if_needed(&b).map_err(|e| format!("{FN_NAME}: {e}"))?;
    let n = gather_if_needed(&n).map_err(|e| format!("{FN_NAME}: {e}"))?;

    let limit = parse_prefix_length(n)?;
    let left = TextCollection::from_argument(FN_NAME, a, "first argument")?;
    let right = TextCollection::from_argument(FN_NAME, b, "second argument")?;
    evaluate_strncmp(&left, &right, limit)
}

fn evaluate_strncmp(
    left: &TextCollection,
    right: &TextCollection,
    limit: usize,
) -> Result<Value, String> {
    let shape = broadcast_shapes(FN_NAME, &left.shape, &right.shape)?;
    let total = tensor::element_count(&shape);
    if total == 0 {
        return logical_result(FN_NAME, Vec::new(), shape);
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

    logical_result(FN_NAME, data, shape)
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

fn parse_prefix_length(value: Value) -> Result<usize, String> {
    match value {
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < 0 {
                return Err(format!(
                    "{FN_NAME}: prefix length must be a nonnegative integer"
                ));
            }
            Ok(raw as usize)
        }
        Value::Num(n) => parse_prefix_length_from_float(n),
        Value::Bool(b) => Ok(if b { 1 } else { 0 }),
        Value::Tensor(tensor) => {
            if tensor.data.len() != 1 {
                return Err(format!(
                    "{FN_NAME}: prefix length must be a nonnegative integer scalar"
                ));
            }
            parse_prefix_length_from_float(tensor.data[0])
        }
        Value::LogicalArray(array) => {
            if array.data.len() != 1 {
                return Err(format!(
                    "{FN_NAME}: prefix length must be a nonnegative integer scalar"
                ));
            }
            Ok(if array.data[0] != 0 { 1 } else { 0 })
        }
        other => Err(format!(
            "{FN_NAME}: prefix length must be a nonnegative integer scalar, received {other:?}"
        )),
    }
}

fn parse_prefix_length_from_float(value: f64) -> Result<usize, String> {
    if !value.is_finite() {
        return Err(format!(
            "{FN_NAME}: prefix length must be a finite nonnegative integer"
        ));
    }
    if value < 0.0 {
        return Err(format!(
            "{FN_NAME}: prefix length must be a nonnegative integer"
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(format!(
            "{FN_NAME}: prefix length must be a nonnegative integer"
        ));
    }
    if rounded > (usize::MAX as f64) {
        return Err(format!(
            "{FN_NAME}: prefix length exceeds the maximum supported size"
        ));
    }
    Ok(rounded as usize)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use runmat_builtins::{CellArray, CharArray, IntValue, LogicalArray, StringArray, Tensor};

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
    fn strncmp_char_array_rows() {
        let chars = CharArray::new(
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
        let err = strncmp_builtin(
            Value::StringArray(left),
            Value::StringArray(right),
            Value::Int(IntValue::I32(1)),
        )
        .expect_err("size mismatch");
        assert!(err.contains("size mismatch"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strncmp_invalid_length_type_errors() {
        let err = strncmp_builtin(
            Value::String("abc".into()),
            Value::String("abc".into()),
            Value::String("3".into()),
        )
        .expect_err("invalid prefix length");
        assert!(err.contains("prefix length"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strncmp_negative_length_errors() {
        let err = strncmp_builtin(
            Value::String("abc".into()),
            Value::String("abc".into()),
            Value::Num(-1.0),
        )
        .expect_err("negative length");
        assert!(err.to_ascii_lowercase().contains("nonnegative"));
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
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload prefix length to GPU");
        let result = strncmp_builtin(
            Value::String("delta".into()),
            Value::String("deluge".into()),
            Value::GpuTensor(handle.clone()),
        )
        .expect("strncmp");
        assert_eq!(result, Value::Bool(true));
        let _ = provider.free(&handle);
    }
}
