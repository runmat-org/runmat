//! MATLAB-compatible `sprintf` builtin that mirrors printf-style formatting semantics.

use runmat_builtins::{
    BuiltinErrorDescriptor, SPRINTF_DESCRIPTOR, SPRINTF_ERROR_ARGUMENT_MISMATCH,
    SPRINTF_ERROR_INTERNAL, SPRINTF_ERROR_INVALID_FORMAT_SPEC, SPRINTF_ERROR_TOO_MANY_OUTPUTS,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, Value};

use crate::builtins::common::format::{
    decode_escape_sequences, extract_format_string, flatten_argument_elements,
    format_variadic_with_cursor, ArgCursor,
};
use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::common::contains_numeric_or_resident_text_input;
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

const BUILTIN_NAME: &str = "sprintf";

fn sprintf_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    sprintf_error_with_message(error.message, error)
}

fn sprintf_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn remap_sprintf_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

#[runtime_builtin(
    name = "sprintf",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::core::sprintf"
)]
async fn sprintf_builtin(
    format_spec: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &SPRINTF_DESCRIPTOR,
        &SPRINTF_ERROR_TOO_MANY_OUTPUTS,
        "sprintf",
    )?;
    if contains_numeric_or_resident_text_input(&format_spec) {
        return Err(sprintf_error(&SPRINTF_ERROR_INVALID_FORMAT_SPEC));
    }
    let gathered_spec = gather_if_needed_async(&format_spec)
        .await
        .map_err(remap_sprintf_flow)?;
    let raw_format =
        extract_format_string(&gathered_spec, "sprintf").map_err(remap_sprintf_flow)?;
    let format_string =
        decode_escape_sequences("sprintf", &raw_format).map_err(remap_sprintf_flow)?;
    let flattened_args = flatten_argument_elements(rest.view(), "sprintf")
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
                return Err(sprintf_error_with_message(
                    "sprintf: formatSpec contains no conversion specifiers but additional arguments were supplied",
                    &SPRINTF_ERROR_ARGUMENT_MISMATCH,
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
    let len = text.encode_utf16().count();
    let array = CharArray::from_unicode_scalars(chars, 1, len)
        .map_err(|_| sprintf_error(&SPRINTF_ERROR_INTERNAL))?;
    Ok(Value::CharArray(array))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{builtins::common::test_support, make_cell};
    use runmat_builtins::{infer_catalog_call, SPRINTF_CATALOG_ENTRY};
    use runmat_types::{
        CallRequest, LiteralContext, OutputSelection, RequestedOutputCount, ShapeFact, StorageFact,
        ValueFact, ValueKindFact,
    };
    use runmat_value::{
        CharArray, IntValue, IntegerComplexStorage, IntegerStorage, StringArray, Tensor,
    };

    fn sprintf_builtin(format_spec: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(super::sprintf_builtin(
            format_spec,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }

    fn error_message(err: crate::RuntimeError) -> String {
        err.message().to_string()
    }

    fn char_value_to_string(value: Value) -> String {
        match value {
            Value::CharArray(ca) => runmat_value::decode_utf16_code_units(ca.data()).unwrap(),
            other => panic!("expected char output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_basic_integer() {
        let result = sprintf_builtin(
            Value::String("Value: %d".to_string().into()),
            vec![Value::Int(IntValue::I32(42))],
        )
        .expect("sprintf");
        assert_eq!(char_value_to_string(result), "Value: 42");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_float_precision() {
        let result = sprintf_builtin(
            Value::String("pi ~= %.3f".to_string().into()),
            vec![Value::Num(std::f64::consts::PI)],
        )
        .expect("sprintf");
        assert_eq!(char_value_to_string(result), "pi ~= 3.142");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_shared_scientific_and_general_conversions() {
        let result = sprintf_builtin(
            Value::String("%.1e|%.3g|%d".to_string().into()),
            vec![Value::Num(8.8e8), Value::Num(0.000123456), Value::Num(2.5)],
        )
        .expect("sprintf");
        assert_eq!(
            char_value_to_string(result),
            "8.8e+08|0.000123|2.500000e+00"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_array_repeat() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let result = sprintf_builtin(
            Value::String("%d ".to_string().into()),
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
        let result =
            sprintf_builtin(Value::String("%*.*f".to_string().into()), args).expect("sprintf");
        assert_eq!(char_value_to_string(result), " 12.35");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_literal_percent() {
        let result = sprintf_builtin(Value::String("%% complete".to_string().into()), Vec::new())
            .expect("sprintf");
        assert_eq!(char_value_to_string(result), "% complete");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_gpu_numeric() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: tensor.shape(),
            };
            let handle = provider.upload(&view).expect("upload");
            let value = Value::GpuTensor(handle);
            let result = sprintf_builtin(Value::String("%0.1f,".to_string().into()), vec![value])
                .expect("sprintf");
            assert_eq!(char_value_to_string(result), "1.0,2.0,");
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_matrix_column_major() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result = sprintf_builtin(
            Value::String("%0.0f ".to_string().into()),
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
                Value::String("%d %d".to_string().into()),
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
                Value::String("literal text".to_string().into()),
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
        let chars = CharArray::from_unicode_scalars("hi!".chars().collect(), 3, 1).unwrap();
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
    fn sprintf_unsupported_specifier_reports_stable_identifier() {
        let err = sprintf_builtin(
            Value::String("%q".to_string().into()),
            vec![Value::Num(1.0)],
        )
        .expect_err("sprintf should error");
        assert_eq!(
            err.identifier(),
            Some("RunMat:format:UnsupportedSpecifier"),
            "unsupported formatter specifiers should expose a stable identifier"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_percent_c_from_numeric() {
        let result = sprintf_builtin(
            Value::String("%c".to_string().into()),
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
                Value::String("two".to_string().into()),
                Value::Num(3.0),
            ],
            3,
            1,
        )
        .expect("cell");
        let result = sprintf_builtin(
            Value::String("%0.0f %s %0.0f".to_string().into()),
            vec![cell],
        )
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
            Value::String("%s ".to_string().into()),
            vec![Value::StringArray(array)],
        )
        .expect("sprintf");
        assert_eq!(char_value_to_string(result), "alpha beta gamma ");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_complex_s_conversion() {
        let result = sprintf_builtin(
            Value::String("%s".to_string().into()),
            vec![Value::Complex(1.5, -2.0)],
        )
        .expect("sprintf");
        assert_eq!(char_value_to_string(result), "1.5-2i");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_complex_integer_tensor_s_conversion_preserves_exact_storage() {
        let storage = IntegerComplexStorage::new(
            IntegerStorage::U64(vec![u64::MAX, 1_u64 << 63]),
            IntegerStorage::U64(vec![7, 0]),
        )
        .expect("matching complex integer storage");
        let tensor = runmat_value::ComplexTensor::new_integer(storage, vec![1, 2])
            .expect("complex integer tensor");
        let result = sprintf_builtin(
            Value::String("%s ".to_string().into()),
            vec![Value::ComplexTensor(tensor)],
        )
        .expect("sprintf");
        assert_eq!(
            char_value_to_string(result),
            format!("{}+7i {} ", u64::MAX, 1_u64 << 63)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_escape_sequences() {
        let result = sprintf_builtin(
            Value::String("Line 1\\nLine 2\\t(tab)".to_string().into()),
            Vec::new(),
        )
        .expect("sprintf");
        assert_eq!(char_value_to_string(result), "Line 1\nLine 2\t(tab)");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_hex_and_octal_escapes() {
        let result = sprintf_builtin(Value::String("\\x41\\101".to_string().into()), Vec::new())
            .expect("sprintf");
        assert_eq!(char_value_to_string(result), "AA");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn sprintf_unknown_escape_preserved() {
        let result = sprintf_builtin(Value::String("Value\\q".to_string().into()), Vec::new())
            .expect("sprintf");
        assert_eq!(char_value_to_string(result), "Value\\q");
    }

    #[test]
    fn sprintf_catalog_infers_character_row_for_text() {
        let request = CallRequest {
            arguments: vec![ValueFact::scalar(ValueKindFact::String)],
            literals: LiteralContext::new(Vec::new()),
            outputs: OutputSelection::new(RequestedOutputCount::Exactly(1)),
        };
        let inferred = infer_catalog_call(&SPRINTF_CATALOG_ENTRY, &request);
        assert!(
            inferred.diagnostics.is_empty(),
            "{:?}",
            inferred.diagnostics
        );
        assert_eq!(inferred.outputs.len(), 1);
        assert_eq!(inferred.outputs[0].kind, ValueKindFact::Character);
        assert_eq!(
            inferred.outputs[0].shape,
            ShapeFact::from(vec![Some(1), None])
        );
        assert_eq!(inferred.outputs[0].storage, StorageFact::Dense);
    }
}
