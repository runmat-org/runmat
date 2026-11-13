//! MATLAB-compatible `strtrim` builtin with GPU-aware semantics for RunMat.

use crate::context::host_allocation::HostAllocationError;
use crate::context::RuntimeContext;
use runmat_builtins::{
    BuiltinErrorDescriptor, STRTRIM_ERROR_CELL_ELEMENT, STRTRIM_ERROR_INTERNAL,
    STRTRIM_ERROR_INVALID_INPUT, STRTRIM_STRING_CELL_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, StringArray, Value};

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::common::{
    cell_has_valid_string_scalar, char_row_to_string_slice,
    contains_numeric_or_resident_text_input, transform_scalar_borrowed,
    transform_string_array_borrowed,
};
use crate::{build_runtime_error, gather_if_needed_async, make_cell, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::transform::strtrim")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "strtrim",
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

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::strings::transform::strtrim"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "strtrim",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Host text transformation; not eligible for fusion. Resident numeric inputs reject.",
};

const BUILTIN_NAME: &str = "strtrim";

fn map_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

fn strtrim_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn strtrim_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    strtrim_error_with_message(error.message, error)
}

#[runtime_builtin(
    name = "strtrim",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::transform::strtrim"
)]
async fn strtrim_builtin(runtime: Option<RuntimeContext>, value: Value) -> BuiltinResult<Value> {
    if contains_numeric_or_resident_text_input(&value) {
        return Err(strtrim_error(&STRTRIM_ERROR_INVALID_INPUT));
    }
    if cell_has_valid_string_scalar(&value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &STRTRIM_STRING_CELL_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let gathered = gather_if_needed_async(&value).await.map_err(map_flow)?;
    match gathered {
        Value::String(text) => {
            let strict = !crate::compatibility::runmat_extensions_enabled();
            transform_scalar_borrowed(
                runtime.as_ref(),
                text,
                |value| trim_whitespace_with_mode(value, strict),
                strtrim_admission,
            )
            .map(Value::String)
        }
        Value::StringArray(array) => strtrim_string_array(runtime.as_ref(), array),
        Value::CharArray(array) => strtrim_char_array(array),
        Value::Cell(cell) => strtrim_cell_array(runtime.as_ref(), cell).await,
        _ => Err(strtrim_error(&STRTRIM_ERROR_INVALID_INPUT)),
    }
}

fn strtrim_string_array(
    runtime: Option<&RuntimeContext>,
    array: StringArray,
) -> BuiltinResult<Value> {
    let strict = !crate::compatibility::runmat_extensions_enabled();
    let out = transform_string_array_borrowed(
        runtime,
        array,
        |text| trim_whitespace_with_mode(text, strict),
        strtrim_admission,
        |e| strtrim_error_with_message(format!("{BUILTIN_NAME}: {e}"), &STRTRIM_ERROR_INTERNAL),
    )?;
    Ok(Value::StringArray(out))
}

fn strtrim_admission(runtime: &RuntimeContext, cause: HostAllocationError) -> RuntimeError {
    build_runtime_error(format!("strtrim: host allocation: {cause}"))
        .with_builtin(BUILTIN_NAME)
        .with_identifier(format!(
            "{}:{}",
            runtime.error_namespace(),
            cause.runtime_identifier_suffix()
        ))
        .with_source(cause)
        .build()
}

fn strtrim_char_array(array: CharArray) -> BuiltinResult<Value> {
    if array.rows == 0 {
        return Ok(Value::CharArray(array));
    }
    let (data, _, rows, cols) = array.into_parts();

    let mut trimmed_rows: Vec<Vec<runmat_value::Utf16CodeUnit>> = Vec::with_capacity(rows);
    let mut target_cols: usize = 0;
    for row in 0..rows {
        let text = char_row_to_string_slice(&data, cols, row).map_err(|e| {
            strtrim_error_with_message(format!("{BUILTIN_NAME}: {e}"), &STRTRIM_ERROR_INTERNAL)
        })?;
        let trimmed = trim_whitespace(&text);
        let units = trimmed
            .encode_utf16()
            .map(runmat_value::Utf16CodeUnit::from_raw)
            .collect();
        target_cols = target_cols.max(trimmed.encode_utf16().count());
        trimmed_rows.push(units);
    }

    let mut new_data = Vec::with_capacity(rows * target_cols);
    for mut units in trimmed_rows {
        let padding = target_cols - units.len();
        units.extend(std::iter::repeat_n(
            runmat_value::Utf16CodeUnit::from_raw(b' ' as u16),
            padding,
        ));
        new_data.extend(units);
    }

    CharArray::from_code_units(new_data, rows, target_cols)
        .map(Value::CharArray)
        .map_err(|e| {
            strtrim_error_with_message(format!("{BUILTIN_NAME}: {e}"), &STRTRIM_ERROR_INTERNAL)
        })
}

async fn strtrim_cell_array(
    runtime: Option<&RuntimeContext>,
    cell: CellArray,
) -> BuiltinResult<Value> {
    let (data, _, rows, cols) = cell.into_parts();
    let mut trimmed_values = Vec::with_capacity(rows * cols);
    for value in &data {
        let trimmed = strtrim_cell_element(runtime, value).await?;
        trimmed_values.push(trimmed);
    }
    make_cell(trimmed_values, rows, cols).map_err(|e| {
        strtrim_error_with_message(format!("{BUILTIN_NAME}: {e}"), &STRTRIM_ERROR_INTERNAL)
    })
}

async fn strtrim_cell_element(
    runtime: Option<&RuntimeContext>,
    value: &Value,
) -> BuiltinResult<Value> {
    match gather_if_needed_async(value).await.map_err(map_flow)? {
        Value::String(text) => {
            let strict = !crate::compatibility::runmat_extensions_enabled();
            transform_scalar_borrowed(
                runtime,
                text,
                |value| trim_whitespace_with_mode(value, strict),
                strtrim_admission,
            )
            .map(Value::String)
        }
        Value::StringArray(sa) if sa.data().len() == 1 => {
            let strict = !crate::compatibility::runmat_extensions_enabled();
            transform_scalar_borrowed(
                runtime,
                sa.data()[0].clone(),
                |value| trim_whitespace_with_mode(value, strict),
                strtrim_admission,
            )
            .map(Value::String)
        }
        Value::CharArray(ca) if ca.rows <= 1 => {
            if ca.rows == 0 {
                return Ok(Value::CharArray(ca));
            }
            let source = char_row_to_string_slice(ca.data(), ca.cols, 0).map_err(|e| {
                strtrim_error_with_message(format!("{BUILTIN_NAME}: {e}"), &STRTRIM_ERROR_INTERNAL)
            })?;
            let trimmed = trim_whitespace(&source);
            let chars: Vec<char> = trimmed.chars().collect();
            let cols = trimmed.encode_utf16().count();
            CharArray::from_unicode_scalars(chars, ca.rows, cols)
                .map(Value::CharArray)
                .map_err(|e| {
                    strtrim_error_with_message(
                        format!("{BUILTIN_NAME}: {e}"),
                        &STRTRIM_ERROR_INTERNAL,
                    )
                })
        }
        Value::CharArray(_) => Err(strtrim_error(&STRTRIM_ERROR_CELL_ELEMENT)),
        _ => Err(strtrim_error(&STRTRIM_ERROR_CELL_ELEMENT)),
    }
}

fn trim_whitespace(text: &str) -> String {
    trim_whitespace_with_mode(text, !crate::compatibility::runmat_extensions_enabled()).to_owned()
}

fn trim_whitespace_with_mode(text: &str, strict: bool) -> &str {
    text.trim_matches(|c: char| crate::builtins::strings::common::trimmable_whitespace(c, strict))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn run_strtrim(value: Value) -> BuiltinResult<Value> {
        futures::executor::block_on(strtrim_builtin(None, value))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_string_scalar_trims_whitespace() {
        let result =
            run_strtrim(Value::String("  RunMat  ".into())).expect("strtrim string scalar");
        assert_eq!(result, Value::String("RunMat".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_string_array_preserves_shape() {
        let array = StringArray::new(
            vec![
                " one ".into(),
                "<missing>".into(),
                "two".into(),
                " three ".into(),
            ],
            vec![2, 2],
        )
        .unwrap();
        let result = run_strtrim(Value::StringArray(array)).expect("strtrim string array");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![2, 2]);
                assert_eq!(sa.data().to_vec(), vec!["one", "<missing>", "two", "three"]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_char_array_multiple_rows() {
        let data: Vec<char> = "  cat  ".chars().chain(" dog   ".chars()).collect();
        let array = CharArray::from_unicode_scalars(data, 2, 7).unwrap();
        let result = run_strtrim(Value::CharArray(array)).expect("strtrim char array");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 2);
                assert_eq!(ca.cols, 3);
                assert_eq!(
                    runmat_value::decode_utf16_code_units(ca.data()).unwrap(),
                    "catdog"
                );
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_char_array_all_whitespace_yields_zero_width() {
        let array = CharArray::from_unicode_scalars("   ".chars().collect(), 1, 3).unwrap();
        let result = run_strtrim(Value::CharArray(array)).expect("strtrim char whitespace");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 0);
                assert!(ca.data().is_empty());
            }
            other => panic!("expected empty char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_cell_array_mixed_content() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("  GPU  ")),
                Value::String(" Accelerate ".into()),
            ],
            1,
            2,
        )
        .unwrap();
        let result = run_strtrim(Value::Cell(cell)).expect("strtrim cell array");
        match result {
            Value::Cell(out) => {
                let first = out.get(0, 0).unwrap();
                let second = out.get(0, 1).unwrap();
                assert_eq!(first, Value::CharArray(CharArray::new_row("GPU")));
                assert_eq!(second, Value::String("Accelerate".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_string_cell_is_mode_gated_before_gather() {
        let cell = CellArray::new(vec![Value::String(" data ".into())], 1, 1).unwrap();
        {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error =
                run_strtrim(Value::Cell(cell.clone())).expect_err("strict cellstr contract");
            assert_eq!(
                error.identifier(),
                STRTRIM_STRING_CELL_EXTENSION.error_identifier
            );
            let scalar = StringArray::new(vec![" data ".into()], vec![1, 1]).unwrap();
            let wrapped = CellArray::new(vec![Value::StringArray(scalar)], 1, 1).unwrap();
            let error = run_strtrim(Value::Cell(wrapped)).expect_err("scalar string-array cell");
            assert_eq!(
                error.identifier(),
                STRTRIM_STRING_CELL_EXTENSION.error_identifier
            );
        }
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = run_strtrim(Value::Cell(cell)).expect("RunMat text cell extension");
        let Value::Cell(result) = result else {
            panic!("cell output")
        };
        assert_eq!(result.get(0, 0).unwrap(), Value::String("data".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_invalid_mixed_cell_keeps_element_error() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let cell =
            CellArray::new(vec![Value::String(" a ".into()), Value::Num(1.0)], 1, 2).unwrap();
        let error = run_strtrim(Value::Cell(cell)).expect_err("invalid cell element");
        assert_eq!(error.identifier(), STRTRIM_ERROR_CELL_ELEMENT.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_preserves_missing_strings() {
        let result = run_strtrim(Value::String(runmat_value::StringElement::Missing))
            .expect("strtrim missing string");
        assert_eq!(result, Value::String(runmat_value::StringElement::Missing));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_handles_tabs_and_newlines() {
        let input = Value::String("\tMetrics \n".into());
        let result = run_strtrim(input).expect("strtrim tab/newline");
        assert_eq!(result, Value::String("Metrics".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_trims_unicode_whitespace() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let input = Value::String("\u{00A0}RunMat\u{2003}".into());
        let result = run_strtrim(input).expect("strtrim unicode whitespace");
        assert_eq!(result, Value::String("RunMat".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_strict_preserves_significant_whitespace() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        for significant in ['\u{0085}', '\u{00A0}', '\u{2007}', '\u{202F}'] {
            let input = format!("  {significant}RunMat{significant}  ");
            let expected = format!("{significant}RunMat{significant}");
            assert_eq!(
                run_strtrim(Value::String(input.into())).expect("strict string trim"),
                Value::String(expected.clone().into())
            );
            assert_eq!(
                run_strtrim(Value::CharArray(CharArray::new_row(&format!(
                    "  {significant}RunMat{significant}  "
                ))))
                .expect("strict character trim"),
                Value::CharArray(CharArray::new_row(&expected))
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_char_array_zero_rows_stable() {
        let array = CharArray::from_code_units(Vec::new(), 0, 0).unwrap();
        let result = run_strtrim(Value::CharArray(array.clone())).expect("strtrim 0x0 char");
        assert_eq!(result, Value::CharArray(array));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_cell_array_accepts_string_scalar() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let scalar = StringArray::new(vec![" padded ".into()], vec![1, 1]).unwrap();
        let cell = CellArray::new(vec![Value::StringArray(scalar)], 1, 1).unwrap();
        let trimmed = run_strtrim(Value::Cell(cell)).expect("strtrim cell string scalar");
        match trimmed {
            Value::Cell(out) => {
                let value = out.get(0, 0).expect("cell element");
                assert_eq!(value, Value::String("padded".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_cell_array_rejects_non_text() {
        let cell = CellArray::new(vec![Value::Num(5.0)], 1, 1).unwrap();
        let err = run_strtrim(Value::Cell(cell)).expect_err("strtrim cell non-text");
        assert!(err.to_string().contains("cell array elements"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strtrim_errors_on_invalid_input() {
        let err = run_strtrim(Value::Num(1.0)).unwrap_err();
        assert!(err.to_string().contains("strtrim"));
    }
}
