//! MATLAB-compatible `strcmpi` builtin for RunMat (case-insensitive string comparison).

use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};
use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::strings::search::text_utils::{logical_result, TextCollection, TextElement};
use crate::builtins::strings::type_resolvers::logical_text_match_type;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::core::strcmpi")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "strcmpi",
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
    notes: "Runs entirely on the CPU; GPU operands are gathered before comparison.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::core::strcmpi")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "strcmpi",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Produces logical host results; not eligible for GPU fusion.",
};

const BUILTIN_NAME: &str = "strcmpi";

const STRCMPI_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "tf",
    ty: BuiltinParamType::LogicalArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Logical comparison result.",
}];

const STRCMPI_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "A",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "First text input (string/char/cell/string array).",
    },
    BuiltinParamDescriptor {
        name: "B",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second text input (string/char/cell/string array).",
    },
];

const STRCMPI_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "tf = strcmpi(A, B)",
    inputs: &STRCMPI_INPUTS,
    outputs: &STRCMPI_OUTPUT,
}];

const STRCMPI_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRCMPI.INVALID_INPUT",
    identifier: Some("RunMat:strcmpi:InvalidInput"),
    when: "At least one input is not a supported text container.",
    message: "strcmpi: text inputs must be string/char/cell/string-array values",
};

const STRCMPI_ERROR_SHAPE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRCMPI.SHAPE_MISMATCH",
    identifier: Some("RunMat:strcmpi:ShapeMismatch"),
    when: "Inputs are not broadcast-compatible for elementwise comparison.",
    message: "strcmpi: input sizes are not broadcast-compatible",
};

const STRCMPI_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.STRCMPI.INTERNAL",
    identifier: Some("RunMat:strcmpi:InternalError"),
    when: "Internal logical result assembly failed.",
    message: "strcmpi: internal error",
};

const STRCMPI_ERRORS: [BuiltinErrorDescriptor; 3] = [
    STRCMPI_ERROR_INVALID_INPUT,
    STRCMPI_ERROR_SHAPE_MISMATCH,
    STRCMPI_ERROR_INTERNAL,
];

pub const STRCMPI_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &STRCMPI_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &STRCMPI_ERRORS,
};

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
    category = "strings/core",
    summary = "Compare text inputs for equality without considering case.",
    keywords = "strcmpi,string compare,text equality",
    accel = "sink",
    type_resolver(logical_text_match_type),
    descriptor(crate::builtins::strings::core::strcmpi::STRCMPI_DESCRIPTOR),
    builtin_path = "crate::builtins::strings::core::strcmpi"
)]
async fn strcmpi_builtin(a: Value, b: Value) -> crate::BuiltinResult<Value> {
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
    evaluate_strcmpi(&left, &right)
}

fn evaluate_strcmpi(left: &TextCollection, right: &TextCollection) -> BuiltinResult<Value> {
    let shape = broadcast_shapes(BUILTIN_NAME, &left.shape, &right.shape)
        .map_err(|_| strcmpi_error(&STRCMPI_ERROR_SHAPE_MISMATCH))?;
    let total = tensor::element_count(&shape);
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
    use runmat_builtins::{CellArray, CharArray, LogicalArray, ResolveContext, StringArray, Type};

    fn strcmpi_builtin(a: Value, b: Value) -> BuiltinResult<Value> {
        futures::executor::block_on(super::strcmpi_builtin(a, b))
    }

    fn error_message(err: RuntimeError) -> String {
        err.to_string()
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
        let chars = CharArray::new(vec!['c', 'a', 't', 'D', 'O', 'G'], 2, 3).unwrap();
        let result =
            strcmpi_builtin(Value::CharArray(chars), Value::String("CaT".into())).expect("cmp");
        let expected = LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_char_array_to_char_array_casefold() {
        let left = CharArray::new(vec!['A', 'b', 'C', 'd'], 2, 2).unwrap();
        let right = CharArray::new(vec!['a', 'B', 'x', 'Y'], 2, 2).unwrap();
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
        let strings = StringArray::new(vec!["<missing>".into()], vec![1, 1]).unwrap();
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
        let chars = CharArray::new(vec!['c', 'a', 't', ' '], 1, 4).unwrap();
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
    fn strcmpi_invalid_argument_type() {
        let err = error_message(
            strcmpi_builtin(Value::Num(1.0), Value::String("a".into())).expect_err("invalid type"),
        );
        assert!(err.contains(STRCMPI_ERROR_INVALID_INPUT.message));
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
        let chars = CharArray::new(Vec::<char>::new(), 0, 3).unwrap();
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

    #[test]
    fn strcmpi_type_is_logical_match() {
        assert_eq!(
            logical_text_match_type(
                &[Type::String, Type::String],
                &ResolveContext::new(Vec::new()),
            ),
            Type::Bool
        );
    }
}
