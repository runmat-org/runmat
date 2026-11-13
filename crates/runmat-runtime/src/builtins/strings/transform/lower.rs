//! MATLAB-compatible `lower` builtin with GPU-aware semantics for RunMat.

use crate::context::host_allocation::HostAllocationError;
use crate::context::RuntimeContext;
use runmat_builtins::{
    BuiltinErrorDescriptor, LOWER_ERROR_CELL_ELEMENT, LOWER_ERROR_INTERNAL,
    LOWER_ERROR_INVALID_INPUT, LOWER_STRING_CELL_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, StringArray, Value};

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::common::{
    cell_has_valid_string_scalar, transform_character_case, transform_scalar_case,
    transform_string_array_case, TextCase,
};
use crate::{build_runtime_error, gather_if_needed_async, make_cell, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::transform::lower")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "lower",
    op_kind: GpuOpKind::Custom("string-transform"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Executes on host text; resident numeric inputs reject before provider access or gathering.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::transform::lower")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "lower",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Host text transformation; not eligible for fusion. Resident numeric inputs reject.",
};

const BUILTIN_NAME: &str = "lower";

fn map_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

fn lower_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn lower_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    lower_error_with_message(error.message, error)
}

#[runtime_builtin(
    name = "lower",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::transform::lower"
)]
async fn lower_builtin(runtime: Option<RuntimeContext>, value: Value) -> BuiltinResult<Value> {
    if crate::dispatcher::value_contains_gpu(&value) {
        return Err(lower_error(&LOWER_ERROR_INVALID_INPUT));
    }
    if cell_has_valid_string_scalar(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &LOWER_STRING_CELL_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let gathered = gather_if_needed_async(&value).await.map_err(map_flow)?;
    match gathered {
        Value::String(text) => {
            transform_scalar_case(runtime.as_ref(), text, TextCase::Lower, lower_admission)
                .map(Value::String)
        }
        Value::StringArray(array) => lower_string_array(runtime.as_ref(), array),
        Value::CharArray(array) => lower_char_array(runtime.as_ref(), array),
        Value::Cell(cell) => lower_cell_array(runtime.as_ref(), cell),
        _ => Err(lower_error(&LOWER_ERROR_INVALID_INPUT)),
    }
}

fn lower_string_array(
    runtime: Option<&RuntimeContext>,
    array: StringArray,
) -> BuiltinResult<Value> {
    transform_string_array_case(runtime, array, TextCase::Lower, lower_admission, |detail| {
        lower_error_with_message(format!("{BUILTIN_NAME}: {detail}"), &LOWER_ERROR_INTERNAL)
    })
    .map(Value::StringArray)
}

fn lower_admission(runtime: &RuntimeContext, cause: HostAllocationError) -> RuntimeError {
    build_runtime_error(format!("lower: host allocation: {cause}"))
        .with_builtin(BUILTIN_NAME)
        .with_identifier(format!(
            "{}:{}",
            runtime.error_namespace(),
            cause.runtime_identifier_suffix()
        ))
        .with_source(cause)
        .build()
}

fn lower_char_array(runtime: Option<&RuntimeContext>, array: CharArray) -> BuiltinResult<Value> {
    transform_character_case(runtime, array, TextCase::Lower, lower_admission, |detail| {
        lower_error_with_message(
            format!("{BUILTIN_NAME}: {detail}"),
            &LOWER_ERROR_INVALID_INPUT,
        )
    })
    .map(Value::CharArray)
}

fn lower_cell_array(runtime: Option<&RuntimeContext>, cell: CellArray) -> BuiltinResult<Value> {
    let (data, _, rows, cols) = cell.into_parts();
    let mut lowered_values = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            let lowered = lower_cell_element(runtime, &data[idx])?;
            lowered_values.push(lowered);
        }
    }
    make_cell(lowered_values, rows, cols).map_err(|e| {
        lower_error_with_message(format!("{BUILTIN_NAME}: {e}"), &LOWER_ERROR_INTERNAL)
    })
}

fn lower_cell_element(runtime: Option<&RuntimeContext>, value: &Value) -> BuiltinResult<Value> {
    match value {
        Value::String(text) => {
            transform_scalar_case(runtime, text.clone(), TextCase::Lower, lower_admission)
                .map(Value::String)
        }
        Value::StringArray(sa) if sa.data().len() == 1 => transform_scalar_case(
            runtime,
            sa.data()[0].clone(),
            TextCase::Lower,
            lower_admission,
        )
        .map(Value::String),
        Value::CharArray(ca) if ca.rows <= 1 => lower_char_array(runtime, ca.clone()),
        Value::CharArray(_) => Err(lower_error(&LOWER_ERROR_CELL_ELEMENT)),
        _ => Err(lower_error(&LOWER_ERROR_CELL_ELEMENT)),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn run_lower(value: Value) -> BuiltinResult<Value> {
        futures::executor::block_on(lower_builtin(None, value))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lower_string_scalar_value() {
        let result = run_lower(Value::String("RunMat".into())).expect("lower");
        assert_eq!(result, Value::String("runmat".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lower_string_array_preserves_shape() {
        let array = StringArray::new(
            vec![
                "GPU".into(),
                "ACCEL".into(),
                "<missing>".into(),
                "MiXeD".into(),
            ],
            vec![2, 2],
        )
        .unwrap();
        let result = run_lower(Value::StringArray(array)).expect("lower");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![2, 2]);
                assert_eq!(
                    sa.data().to_vec(),
                    vec!["gpu", "accel", "<missing>", "mixed"]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lower_char_array_multiple_rows() {
        let data: Vec<char> = vec!['C', 'A', 'T', 'D', 'O', 'G'];
        let array = CharArray::from_unicode_scalars(data, 2, 3).unwrap();
        let result = run_lower(Value::CharArray(array)).expect("lower");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 2);
                assert_eq!(ca.cols, 3);
                assert_eq!(ca.data(), runmat_value::encode_utf16_code_units("catdog"));
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lower_char_vector_handles_padding() {
        let array = CharArray::new_row("HELLO ");
        let result = run_lower(Value::CharArray(array)).expect("lower");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 6);
                let expected = runmat_value::encode_utf16_code_units("hello ");
                assert_eq!(ca.data(), expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lower_char_array_unicode_expansion_extends_width() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let data: Vec<char> = vec!['İ', 'A'];
        let array = CharArray::from_unicode_scalars(data, 1, 2).unwrap();
        let result = run_lower(Value::CharArray(array)).expect("lower");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 3);
                let expected = runmat_value::encode_utf16_code_units("i\u{307}a");
                assert_eq!(ca.data(), expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[test]
    fn lower_preserves_supplementary_code_unit_geometry_and_refuses_unpaired_units() {
        let result = run_lower(Value::CharArray(CharArray::new_row("🙂A"))).unwrap();
        let Value::CharArray(result) = result else {
            panic!("expected character output");
        };
        assert_eq!(result.shape(), &[1, 3]);
        assert_eq!(result.try_row_string().unwrap().as_deref(), Some("🙂a"));

        let malformed =
            CharArray::from_code_units(vec![runmat_value::Utf16CodeUnit::from_raw(0xD800)], 1, 1)
                .unwrap();
        let error = run_lower(Value::CharArray(malformed)).unwrap_err();
        assert_eq!(error.identifier(), LOWER_ERROR_INVALID_INPUT.identifier);
        assert!(error.message().contains("unpaired UTF-16 surrogate"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lower_strict_latin1_character_mapping() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let result = run_lower(Value::CharArray(CharArray::new_row("ÉCOLE"))).unwrap();
        assert_eq!(result, Value::CharArray(CharArray::new_row("école")));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lower_cell_array_mixed_content() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("RUN")),
                Value::String("Mat".into()),
            ],
            1,
            2,
        )
        .unwrap();
        let result = run_lower(Value::Cell(cell)).expect("lower");
        match result {
            Value::Cell(out) => {
                let first = out.get(0, 0).unwrap();
                let second = out.get(0, 1).unwrap();
                assert_eq!(first, Value::CharArray(CharArray::new_row("run")));
                assert_eq!(second, Value::String("mat".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lower_string_cell_is_mode_gated_before_gather() {
        let cell = CellArray::new(vec![Value::String("DATA".into())], 1, 1).unwrap();
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = run_lower(Value::Cell(cell.clone())).expect_err("strict cellstr contract");
            assert_eq!(
                error.identifier(),
                LOWER_STRING_CELL_EXTENSION.error_identifier
            );
            let scalar = StringArray::new(vec!["DATA".into()], vec![1, 1]).unwrap();
            let wrapped = CellArray::new(vec![Value::StringArray(scalar)], 1, 1).unwrap();
            let error = run_lower(Value::Cell(wrapped)).expect_err("scalar string-array cell");
            assert_eq!(
                error.identifier(),
                LOWER_STRING_CELL_EXTENSION.error_identifier
            );
        }
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = run_lower(Value::Cell(cell)).expect("RunMat text cell extension");
        let Value::Cell(result) = result else {
            panic!("cell output")
        };
        assert_eq!(result.get(0, 0).unwrap(), Value::String("data".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lower_invalid_mixed_cell_keeps_element_error() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let cell = CellArray::new(vec![Value::String("A".into()), Value::Num(1.0)], 1, 2).unwrap();
        let error = run_lower(Value::Cell(cell)).expect_err("invalid cell element");
        assert_eq!(error.identifier(), LOWER_ERROR_CELL_ELEMENT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lower_errors_on_invalid_input() {
        let err = run_lower(Value::Num(1.0)).unwrap_err();
        assert_eq!(err.to_string(), LOWER_ERROR_INVALID_INPUT.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lower_cell_errors_on_invalid_element() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let err = run_lower(Value::Cell(cell)).unwrap_err();
        assert_eq!(err.to_string(), LOWER_ERROR_CELL_ELEMENT.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lower_preserves_missing_string() {
        let result = run_lower(Value::String(runmat_value::StringElement::Missing)).expect("lower");
        assert_eq!(result, Value::String(runmat_value::StringElement::Missing));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn lower_cell_allows_empty_char_vector() {
        let empty_char = CharArray::from_unicode_scalars(Vec::new(), 1, 0).unwrap();
        let cell = CellArray::new(vec![Value::CharArray(empty_char.clone())], 1, 1).unwrap();
        let result = run_lower(Value::Cell(cell)).expect("lower");
        match result {
            Value::Cell(out) => {
                let element = out.get(0, 0).unwrap();
                assert_eq!(element, Value::CharArray(empty_char));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn lower_gpu_tensor_input_rejects_without_gathering() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let data = [1.0f64, 2.0];
        let shape = [2usize, 1usize];
        let handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &data,
                shape: &shape,
            })
            .expect("upload");
        let err = run_lower(Value::GpuTensor(handle.clone())).unwrap_err();
        assert_eq!(err.to_string(), LOWER_ERROR_INVALID_INPUT.message);
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
