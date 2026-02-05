//! MATLAB-compatible `sprintf` builtin that mirrors printf-style formatting semantics.

use runmat_builtins::{CharArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::format::{
    decode_escape_sequences, extract_format_string, flatten_arguments, format_variadic_with_cursor,
    ArgCursor,
};
use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::type_resolvers::string_scalar_type;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::core::sprintf")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "sprintf",
    op_kind: GpuOpKind::Custom("format"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Formatting runs on the CPU; GPU tensors are gathered before substitution.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::core::sprintf")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "sprintf",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Formatting is a residency sink and is not fused; callers should treat sprintf as a CPU-only builtin.",
};

fn sprintf_flow(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin("sprintf").build()
}

fn remap_sprintf_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, "sprintf")
}

#[runtime_builtin(
    name = "sprintf",
    category = "strings/core",
    summary = "Format data into a character vector using printf-style placeholders.",
    keywords = "sprintf,format,printf,text",
    accel = "format",
    sink = true,
    type_resolver(string_scalar_type),
    builtin_path = "crate::builtins::strings::core::sprintf"
)]
async fn sprintf_builtin(format_spec: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let gathered_spec = gather_if_needed_async(&format_spec)
        .await
        .map_err(remap_sprintf_flow)?;
    let raw_format =
        extract_format_string(&gathered_spec, "sprintf").map_err(remap_sprintf_flow)?;
    let format_string =
        decode_escape_sequences("sprintf", &raw_format).map_err(remap_sprintf_flow)?;
    let flattened_args = flatten_arguments(&rest, "sprintf")
        .await
        .map_err(remap_sprintf_flow)?;
    let mut cursor = ArgCursor::new(&flattened_args);
    let mut output = String::new();

    loop {
        let step =
            format_variadic_with_cursor(&format_string, &mut cursor).map_err(remap_sprintf_flow)?;
        output.push_str(&step.output);

        if step.consumed == 0 {
            if cursor.remaining() > 0 {
                return Err(sprintf_flow(
                    "sprintf: formatSpec contains no conversion specifiers but additional arguments were supplied",
                ));
            }
            break;
        }

        if cursor.remaining() == 0 {
            break;
        }
    }

    char_row_value(&output)
}

fn char_row_value(text: &str) -> BuiltinResult<Value> {
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let array = CharArray::new(chars, 1, len).map_err(|e| sprintf_flow(format!("sprintf: {e}")))?;
    Ok(Value::CharArray(array))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{builtins::common::test_support, make_cell};
    use runmat_builtins::{
        CharArray, IntValue, ResolveContext, StringArray, Tensor, Type,
    };

    fn sprintf_builtin(format_spec: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::sprintf_builtin(format_spec, rest))
    }

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    fn char_value_to_string(value: Value) -> String {
        match value {
            Value::CharArray(ca) => ca.data.into_iter().collect(),
            other => panic!("expected char output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_basic_integer() {
        let result = sprintf_builtin(
            Value::String("Value: %d".to_string()),
            vec![Value::Int(IntValue::I32(42))],
        )
        .expect("sprintf");
        assert_eq!(char_value_to_string(result), "Value: 42");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_float_precision() {
        let result = sprintf_builtin(
            Value::String("pi ~= %.3f".to_string()),
            vec![Value::Num(std::f64::consts::PI)],
        )
        .expect("sprintf");
        assert_eq!(char_value_to_string(result), "pi ~= 3.142");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_array_repeat() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result = sprintf_builtin(
            Value::String("%d ".to_string()),
            vec![Value::Tensor(tensor)],
        )
        .expect("sprintf");
        assert_eq!(char_value_to_string(result), "1 2 3 ");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_star_width() {
        let args = vec![
            Value::Int(IntValue::I32(6)),
            Value::Int(IntValue::I32(2)),
            Value::Num(12.345),
        ];
        let result = sprintf_builtin(Value::String("%*.*f".to_string()), args).expect("sprintf");
        assert_eq!(char_value_to_string(result), " 12.35");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_literal_percent() {
        let result =
            sprintf_builtin(Value::String("%% complete".to_string()), Vec::new()).expect("sprintf");
        assert_eq!(char_value_to_string(result), "% complete");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_gpu_numeric() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let value = Value::GpuTensor(handle);
            let result =
                sprintf_builtin(Value::String("%0.1f,".to_string()), vec![value]).expect("sprintf");
            assert_eq!(char_value_to_string(result), "1.0,2.0,");
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_matrix_column_major() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result = sprintf_builtin(
            Value::String("%0.0f ".to_string()),
            vec![Value::Tensor(tensor)],
        )
        .expect("sprintf");
        assert_eq!(char_value_to_string(result), "1 3 2 4 ");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_not_enough_arguments_error() {
        let err = error_message(
            sprintf_builtin(
                Value::String("%d %d".to_string()),
                vec![Value::Int(IntValue::I32(1))],
            )
            .expect_err("sprintf should error"),
        );
        assert!(
            err.contains("not enough input arguments"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_extra_arguments_error() {
        let err = error_message(
            sprintf_builtin(
                Value::String("literal text".to_string()),
                vec![Value::Int(IntValue::I32(1))],
            )
            .expect_err("sprintf should error"),
        );
        assert!(
            err.contains("contains no conversion specifiers"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_format_spec_multirow_error() {
        let chars = CharArray::new("hi!".chars().collect(), 3, 1).unwrap();
        let err = error_message(
            sprintf_builtin(Value::CharArray(chars), Vec::new()).expect_err("sprintf"),
        );
        assert!(
            err.contains("formatSpec must be a character row vector"),
            "unexpected error: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_percent_c_from_numeric() {
        let result = sprintf_builtin(
            Value::String("%c".to_string()),
            vec![Value::Int(IntValue::I32(65))],
        )
        .expect("sprintf");
        assert_eq!(char_value_to_string(result), "A");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_cell_arguments() {
        let cell = make_cell(
            vec![
                Value::Num(1.0),
                Value::String("two".to_string()),
                Value::Num(3.0),
            ],
            3,
            1,
        )
        .expect("cell");
        let result = sprintf_builtin(Value::String("%0.0f %s %0.0f".to_string()), vec![cell])
            .expect("sprintf");
        assert_eq!(char_value_to_string(result), "1 two 3");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_string_array_column_major() {
        let data = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];
        let array =
            StringArray::new(data, vec![3, 1]).expect("string array construction must succeed");
        let result = sprintf_builtin(
            Value::String("%s ".to_string()),
            vec![Value::StringArray(array)],
        )
        .expect("sprintf");
        assert_eq!(char_value_to_string(result), "alpha beta gamma ");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_complex_s_conversion() {
        let result = sprintf_builtin(
            Value::String("%s".to_string()),
            vec![Value::Complex(1.5, -2.0)],
        )
        .expect("sprintf");
        assert_eq!(char_value_to_string(result), "1.5-2i");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_escape_sequences() {
        let result = sprintf_builtin(
            Value::String("Line 1\\nLine 2\\t(tab)".to_string()),
            Vec::new(),
        )
        .expect("sprintf");
        assert_eq!(char_value_to_string(result), "Line 1\nLine 2\t(tab)");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_hex_and_octal_escapes() {
        let result =
            sprintf_builtin(Value::String("\\x41\\101".to_string()), Vec::new()).expect("sprintf");
        assert_eq!(char_value_to_string(result), "AA");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_unknown_escape_preserved() {
        let result =
            sprintf_builtin(Value::String("Value\\q".to_string()), Vec::new()).expect("sprintf");
        assert_eq!(char_value_to_string(result), "Value\\q");
    }

    #[test]
    fn sprintf_type_is_string_scalar() {
        assert_eq!(
            string_scalar_type(&[Type::String], &ResolveContext::new(Vec::new())),
            Type::String
        );
    }
}
