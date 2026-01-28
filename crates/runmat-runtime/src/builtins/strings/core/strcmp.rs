//! MATLAB-compatible `strcmp` builtin for RunMat.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};
use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::strings::search::text_utils::{logical_result, TextCollection, TextElement};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::core::strcmp")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "strcmp",
    op_kind: GpuOpKind::Custom("string-compare"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Performs host-side text comparisons; GPU operands are gathered automatically before evaluation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::core::strcmp")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "strcmp",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Produces logical results on the host; not eligible for GPU fusion.",
};

#[allow(dead_code)]
fn strcmp_flow(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("strcmp").build()
}

fn remap_strcmp_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, "strcmp")
}

#[runtime_builtin(
    name = "strcmp",
    category = "strings/core",
    summary = "Compare text inputs for exact matches (case-sensitive).",
    keywords = "strcmp,string compare,text equality",
    accel = "sink",
    builtin_path = "crate::builtins::strings::core::strcmp"
)]
async fn strcmp_builtin(a: Value, b: Value) -> crate::BuiltinResult<Value> {
    let a = gather_if_needed_async(&a)
        .await
        .map_err(remap_strcmp_flow)?;
    let b = gather_if_needed_async(&b)
        .await
        .map_err(remap_strcmp_flow)?;
    let left = TextCollection::from_argument("strcmp", a, "first argument")?;
    let right = TextCollection::from_argument("strcmp", b, "second argument")?;
    evaluate_strcmp(&left, &right)
}

fn evaluate_strcmp(left: &TextCollection, right: &TextCollection) -> BuiltinResult<Value> {
    let shape = broadcast_shapes("strcmp", &left.shape, &right.shape)?;
    let total = tensor::element_count(&shape);
    if total == 0 {
        return logical_result("strcmp", Vec::new(), shape);
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
    logical_result("strcmp", data, shape)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::RuntimeError;
    use runmat_builtins::{CellArray, CharArray, LogicalArray, StringArray};

    fn strcmp_builtin(a: Value, b: Value) -> BuiltinResult<Value> {
        futures::executor::block_on(super::strcmp_builtin(a, b))
    }

    fn error_message(err: RuntimeError) -> String {
        err.message().to_string()
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
        let chars = CharArray::new(vec!['c', 'a', 't', 'd', 'o', 'g'], 2, 3).unwrap();
        let result =
            strcmp_builtin(Value::CharArray(chars), Value::String("cat".into())).expect("cmp");
        let expected = LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_char_array_to_char_array() {
        let left = CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).unwrap();
        let right = CharArray::new(vec!['a', 'b', 'x', 'y'], 2, 2).unwrap();
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
        let chars = CharArray::new(vec!['c', 'a', 't', ' '], 1, 4).unwrap();
        let result =
            strcmp_builtin(Value::CharArray(chars), Value::String("cat".into())).expect("strcmp");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_char_array_empty_rows_returns_empty() {
        let chars = CharArray::new(Vec::new(), 0, 0).unwrap();
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
        let strings = StringArray::new(vec!["<missing>".into()], vec![1, 1]).unwrap();
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
        let array = StringArray::new(vec!["alpha".into(), "<missing>".into()], vec![1, 2]).unwrap();
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
        assert!(err.contains("size mismatch"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_invalid_argument_type() {
        let err = error_message(
            strcmp_builtin(Value::Num(1.0), Value::String("a".into())).expect_err("invalid type"),
        );
        assert!(err.contains("first argument must be text"));
    }
}
