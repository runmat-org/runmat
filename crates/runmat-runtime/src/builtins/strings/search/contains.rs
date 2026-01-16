//! MATLAB-compatible `contains` builtin for RunMat.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::gather_if_needed;

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};

use super::text_utils::{logical_result, parse_ignore_case, TextCollection, TextElement};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::search::contains")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "contains",
    op_kind: GpuOpKind::Custom("string-search"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Executes entirely on the host; inputs are gathered from the GPU before performing substring checks.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::search::contains")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "contains",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Text operation; not eligible for fusion and materialises host logical results.",
};

#[runtime_builtin(
    name = "contains",
    category = "strings/search",
    summary = "Return logical values indicating whether text inputs contain specific patterns.",
    keywords = "contains,substring,text,ignorecase,search",
    accel = "sink",
    builtin_path = "crate::builtins::strings::search::contains"
)]
fn contains_builtin(text: Value, pattern: Value, rest: Vec<Value>) -> Result<Value, String> {
    let text = gather_if_needed(&text).map_err(|e| format!("contains: {e}"))?;
    let pattern = gather_if_needed(&pattern).map_err(|e| format!("contains: {e}"))?;
    let ignore_case = parse_ignore_case("contains", &rest)?;
    let subject = TextCollection::from_subject("contains", text)?;
    let patterns = TextCollection::from_pattern("contains", pattern)?;
    evaluate_contains(&subject, &patterns, ignore_case)
}

fn evaluate_contains(
    subject: &TextCollection,
    patterns: &TextCollection,
    ignore_case: bool,
) -> Result<Value, String> {
    let output_shape = broadcast_shapes("contains", &subject.shape, &patterns.shape)?;
    let total = tensor::element_count(&output_shape);
    if total == 0 {
        return logical_result("contains", Vec::new(), output_shape);
    }

    let subject_strides = compute_strides(&subject.shape);
    let pattern_strides = compute_strides(&patterns.shape);
    let subject_lower = if ignore_case {
        Some(subject.lowercased())
    } else {
        None
    };
    let pattern_lower = if ignore_case {
        Some(patterns.lowercased())
    } else {
        None
    };

    let mut data = Vec::with_capacity(total);
    for linear in 0..total {
        let subject_idx = broadcast_index(linear, &output_shape, &subject.shape, &subject_strides);
        let pattern_idx = broadcast_index(linear, &output_shape, &patterns.shape, &pattern_strides);
        let value = match (
            &subject.elements[subject_idx],
            &patterns.elements[pattern_idx],
        ) {
            (TextElement::Missing, _) => false,
            (_, TextElement::Missing) => false,
            (TextElement::Text(text), TextElement::Text(pattern)) => {
                if pattern.is_empty() {
                    true
                } else if ignore_case {
                    let lowered_subject = subject_lower
                        .as_ref()
                        .and_then(|vec| vec[subject_idx].as_deref())
                        .expect("lowercase subject available");
                    let lowered_pattern = pattern_lower
                        .as_ref()
                        .and_then(|vec| vec[pattern_idx].as_deref())
                        .expect("lowercase pattern available");
                    lowered_subject.contains(lowered_pattern)
                } else {
                    text.contains(pattern.as_str())
                }
            }
        };
        data.push(if value { 1 } else { 0 });
    }
    logical_result("contains", data, output_shape)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_builtins::{CellArray, CharArray, IntValue, LogicalArray, StringArray};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_string_scalar_true() {
        let result = contains_builtin(
            Value::String("RunMat".into()),
            Value::String("Run".into()),
            Vec::new(),
        )
        .expect("contains");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_string_scalar_false() {
        let result = contains_builtin(
            Value::String("RunMat".into()),
            Value::String("forge".into()),
            Vec::new(),
        )
        .expect("contains");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_ignore_case_option() {
        let result = contains_builtin(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![Value::String("IgnoreCase".into()), Value::Bool(true)],
        )
        .expect("contains");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_string_array_scalar_pattern() {
        let array = StringArray::new(
            vec!["alpha".into(), "beta".into(), "gamma".into()],
            vec![3, 1],
        )
        .unwrap();
        let result = contains_builtin(
            Value::StringArray(array),
            Value::String("a".into()),
            Vec::new(),
        )
        .expect("contains");
        let expected = LogicalArray::new(vec![1, 1, 1], vec![3, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_elementwise_patterns() {
        let subjects = StringArray::new(
            vec!["hydrogen".into(), "helium".into(), "lithium".into()],
            vec![3, 1],
        )
        .unwrap();
        let patterns =
            StringArray::new(vec!["gen".into(), "ium".into(), "iron".into()], vec![3, 1]).unwrap();
        let result = contains_builtin(
            Value::StringArray(subjects),
            Value::StringArray(patterns),
            Vec::new(),
        )
        .expect("contains");
        let expected = LogicalArray::new(vec![1, 1, 0], vec![3, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_broadcast_pattern_column_vector() {
        let patterns = CharArray::new(vec!['s', 'n', 'x'], 3, 1).unwrap();
        let result = contains_builtin(
            Value::String("saturn".into()),
            Value::CharArray(patterns),
            Vec::new(),
        )
        .expect("contains char");
        let expected = LogicalArray::new(vec![1, 1, 0], vec![3, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_cell_array_patterns() {
        let cell = CellArray::new(
            vec![
                Value::from("Mercury"),
                Value::from("Venus"),
                Value::from("Mars"),
            ],
            1,
            3,
        )
        .unwrap();
        let result = contains_builtin(Value::Cell(cell), Value::String("us".into()), Vec::new())
            .expect("contains");
        let expected = LogicalArray::new(vec![0, 1, 0], vec![1, 3]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_missing_strings_false() {
        let array = StringArray::new(vec!["<missing>".into()], vec![1, 1]).unwrap();
        let result = contains_builtin(
            Value::StringArray(array),
            Value::String("a".into()),
            Vec::new(),
        )
        .expect("contains");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_empty_pattern_true() {
        let result = contains_builtin(
            Value::String("foo".into()),
            Value::String("".into()),
            Vec::new(),
        )
        .expect("contains");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_invalid_option_name() {
        let err = contains_builtin(
            Value::String("foo".into()),
            Value::String("f".into()),
            vec![Value::String("IgnoreCases".into()), Value::Bool(true)],
        )
        .unwrap_err();
        assert!(err.contains("unknown option"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_ignore_case_string_flag() {
        let result = contains_builtin(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::String("on".into()),
            ],
        )
        .expect("contains");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_ignore_case_numeric_flag() {
        let result = contains_builtin(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::Int(IntValue::I32(0)),
            ],
        )
        .expect("contains");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_ignore_case_invalid_value() {
        let err = contains_builtin(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::String("maybe".into()),
            ],
        )
        .unwrap_err();
        assert!(err.contains("invalid value"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_ignore_case_missing_value() {
        let err = contains_builtin(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![Value::String("IgnoreCase".into())],
        )
        .unwrap_err();
        assert!(err.contains("expected a value after 'IgnoreCase'"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_mismatched_shapes_error() {
        let text = StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap();
        let pattern =
            StringArray::new(vec!["a".into(), "b".into(), "c".into()], vec![3, 1]).unwrap();
        let err = contains_builtin(
            Value::StringArray(text),
            Value::StringArray(pattern),
            Vec::new(),
        )
        .unwrap_err();
        assert!(err.contains("size mismatch"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_invalid_subject_type() {
        let err =
            contains_builtin(Value::Num(1.0), Value::String("a".into()), Vec::new()).unwrap_err();
        assert!(err.contains("first argument must be text"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_invalid_pattern_type() {
        let err =
            contains_builtin(Value::String("foo".into()), Value::Num(1.0), Vec::new()).unwrap_err();
        assert!(err.contains("pattern must be text"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_cell_invalid_element_error() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let err =
            contains_builtin(Value::Cell(cell), Value::String("a".into()), Vec::new()).unwrap_err();
        assert!(err.contains("cell array elements"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_zero_sized_inputs() {
        let subjects = StringArray::new(Vec::<String>::new(), vec![0, 1]).unwrap();
        let result = contains_builtin(
            Value::StringArray(subjects),
            Value::String("a".into()),
            Vec::new(),
        )
        .expect("contains");
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![0, 1]);
                assert!(array.data.is_empty());
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_missing_pattern_false() {
        let result = contains_builtin(
            Value::String("alpha".into()),
            Value::String("<missing>".into()),
            Vec::new(),
        )
        .expect("contains");
        assert_eq!(result, Value::Bool(false));
    }
}
