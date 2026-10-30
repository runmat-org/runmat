//! MATLAB-compatible `replace` builtin with GPU-aware semantics for RunMat.

use regex::Regex;
use runmat_builtins::{
    BuiltinErrorDescriptor, REPLACE_ERROR_CELL_ELEMENT, REPLACE_ERROR_EMPTY_PATTERN,
    REPLACE_ERROR_EMPTY_REPLACEMENT, REPLACE_ERROR_INTERNAL, REPLACE_ERROR_INVALID_INPUT,
    REPLACE_ERROR_PATTERN_TYPE, REPLACE_ERROR_REPLACEMENT_TYPE, REPLACE_ERROR_SIZE_MISMATCH,
};

use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, StringArray, Value};

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::common::{
    char_row_to_string_slice, contains_resident_text_input, is_missing_string,
};
use crate::builtins::strings::core::compat::pattern_regex;
use crate::{
    build_runtime_error, gather_if_needed_async, make_cell_with_shape, BuiltinResult, RuntimeError,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::transform::replace")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "replace",
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
        "Executes on the CPU; GPU-resident inputs are gathered to host memory prior to replacement.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::strings::transform::replace"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "replace",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "String manipulation builtin; not eligible for fusion plans and always gathers GPU inputs.",
};

const BUILTIN_NAME: &str = "replace";

fn map_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

fn replace_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn replace_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    replace_error_with_message(error.message, error)
}

#[runtime_builtin(
    name = "replace",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::transform::replace"
)]
async fn replace_builtin(text: Value, old: Value, new: Value) -> BuiltinResult<Value> {
    crate::output_count::validate_fixed_outputs(
        &runmat_builtins::REPLACE_DESCRIPTOR,
        &runmat_builtins::REPLACE_ERROR_TOO_MANY_OUTPUTS,
        "replace",
    )?;
    if [&text, &old, &new].into_iter().any(|value| {
        contains_resident_text_input(value)
            || crate::builtins::strings::common::has_unsupported_character_pages(value)
    }) {
        return Err(replace_error(&REPLACE_ERROR_INVALID_INPUT));
    }
    let text = gather_if_needed_async(&text).await.map_err(map_flow)?;
    let old = gather_if_needed_async(&old).await.map_err(map_flow)?;
    let new = gather_if_needed_async(&new).await.map_err(map_flow)?;

    let spec = ReplacementSpec::from_values(&old, &new)?;

    match text {
        Value::String(s) if s.is_missing() => Ok(Value::String(s)),
        Value::String(s) => Ok(Value::string(replace_string_scalar(
            s.as_str().expect("nonmissing subject"),
            &spec,
        ))),
        Value::StringArray(sa) => replace_string_array(sa, &spec),
        Value::CharArray(ca) => replace_char_array(ca, &spec),
        Value::Cell(cell) => replace_cell_array(cell, &spec),
        _ => Err(replace_error(&REPLACE_ERROR_INVALID_INPUT)),
    }
}

fn replace_string_scalar(text: &str, spec: &ReplacementSpec<'_>) -> String {
    spec.apply(text)
}

fn replace_string_array(array: StringArray, spec: &ReplacementSpec<'_>) -> BuiltinResult<Value> {
    let (data, shape, _, _) = array.into_parts();
    let mut replaced = Vec::with_capacity(data.len());
    let input: Vec<runmat_value::StringElement> = match data.try_into_owned() {
        Ok(values) => values.into_iter().collect(),
        Err(shared) => shared.iter().cloned().collect(),
    };
    for entry in input {
        if is_missing_string(&entry) {
            replaced.push(entry);
        } else {
            replaced.push(runmat_value::StringElement::text(
                spec.apply(entry.as_str().expect("nonmissing subject")),
            ));
        }
    }
    let result = StringArray::from_shared_storage(
        runmat_value::ExactArrayStorage::shared_unbounded(replaced),
        shape,
    )
    .map_err(|e| {
        replace_error_with_message(format!("{BUILTIN_NAME}: {e}"), &REPLACE_ERROR_INTERNAL)
    })?;
    Ok(Value::StringArray(result))
}

fn replace_char_array(array: CharArray, spec: &ReplacementSpec<'_>) -> BuiltinResult<Value> {
    if array.rows == 0 {
        return Ok(Value::CharArray(array));
    }
    let (data, _, rows, cols) = array.into_parts();

    let mut replaced_rows = Vec::with_capacity(rows);
    let mut target_cols = 0usize;
    for row in 0..rows {
        let slice = char_row_to_string_slice(&data, cols, row).map_err(|e| {
            replace_error_with_message(format!("{BUILTIN_NAME}: {e}"), &REPLACE_ERROR_INTERNAL)
        })?;
        let replaced = spec.apply(&slice);
        let len = replaced.encode_utf16().count();
        target_cols = target_cols.max(len);
        replaced_rows.push(replaced);
    }

    let mut flattened = Vec::with_capacity(rows * target_cols);
    for row_text in replaced_rows {
        let mut units = row_text
            .encode_utf16()
            .map(runmat_value::Utf16CodeUnit::from_raw)
            .collect::<Vec<_>>();
        units.extend(std::iter::repeat_n(
            runmat_value::Utf16CodeUnit::from_raw(b' ' as u16),
            target_cols - row_text.encode_utf16().count(),
        ));
        flattened.extend(units);
    }

    CharArray::from_code_units(flattened, rows, target_cols)
        .map(Value::CharArray)
        .map_err(|e| {
            replace_error_with_message(format!("{BUILTIN_NAME}: {e}"), &REPLACE_ERROR_INTERNAL)
        })
}

fn replace_cell_array(cell: CellArray, spec: &ReplacementSpec<'_>) -> BuiltinResult<Value> {
    let (data, shape, _, _) = cell.into_parts();
    let mut replaced = Vec::with_capacity(data.len());
    for value in &data {
        replaced.push(replace_cell_element(value, spec)?);
    }
    make_cell_with_shape(replaced, shape.to_vec()).map_err(|e| {
        replace_error_with_message(format!("{BUILTIN_NAME}: {e}"), &REPLACE_ERROR_INTERNAL)
    })
}

fn replace_cell_element(value: &Value, spec: &ReplacementSpec<'_>) -> BuiltinResult<Value> {
    match value {
        Value::String(text) => Ok(Value::String(match text.as_str() {
            Some(text) => runmat_value::StringElement::text(replace_string_scalar(text, spec)),
            None => runmat_value::StringElement::Missing,
        })),
        Value::StringArray(sa) if sa.data().len() == 1 => {
            replace_cell_element(&Value::String(sa.data()[0].clone()), spec)
        }
        Value::CharArray(ca) if ca.rows <= 1 => replace_char_array(ca.clone(), spec),
        Value::CharArray(_) => Err(replace_error(&REPLACE_ERROR_CELL_ELEMENT)),
        _ => Err(replace_error(&REPLACE_ERROR_CELL_ELEMENT)),
    }
}

fn extract_pattern_list(value: &Value) -> BuiltinResult<Vec<SearchPattern<'_>>> {
    if matches!(value, Value::Object(_)) {
        let regex = pattern_regex(value, BUILTIN_NAME).map_err(|err| {
            replace_error_with_message(err.message().to_string(), &REPLACE_ERROR_PATTERN_TYPE)
        })?;
        return Ok(vec![SearchPattern::Regex(Regex::new(&regex).map_err(
            |err| replace_error_with_message(err.to_string(), &REPLACE_ERROR_PATTERN_TYPE),
        )?)]);
    }
    extract_text_list(value, &REPLACE_ERROR_PATTERN_TYPE)
        .map(|items| items.into_iter().map(SearchPattern::Literal).collect())
}

fn extract_replacement_list(value: &Value) -> BuiltinResult<Vec<std::borrow::Cow<'_, str>>> {
    extract_text_list(value, &REPLACE_ERROR_REPLACEMENT_TYPE)
}

fn extract_text_list<'a>(
    value: &'a Value,
    type_error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<Vec<std::borrow::Cow<'a, str>>> {
    match value {
        Value::String(text) => Ok(vec![std::borrow::Cow::Borrowed(
            text.as_str().ok_or_else(|| replace_error(type_error))?,
        )]),
        Value::StringArray(array) => Ok(array
            .data()
            .iter()
            .map(|text| {
                text.as_str()
                    .map(std::borrow::Cow::Borrowed)
                    .ok_or_else(|| replace_error(type_error))
            })
            .collect::<BuiltinResult<Vec<_>>>()?),
        Value::CharArray(array) => {
            if array.rows == 0 {
                Ok(Vec::new())
            } else {
                let mut entries = Vec::with_capacity(array.rows);
                for row in 0..array.rows {
                    entries.push(std::borrow::Cow::Owned(
                        char_row_to_string_slice(array.data(), array.cols, row).map_err(|e| {
                            replace_error_with_message(
                                format!("{BUILTIN_NAME}: {e}"),
                                &REPLACE_ERROR_INTERNAL,
                            )
                        })?,
                    ));
                }
                Ok(entries)
            }
        }
        Value::Cell(cell) => {
            let mut entries = Vec::with_capacity(cell.data().len());
            for element in cell.data() {
                match element {
                    Value::String(text) => entries.push(std::borrow::Cow::Borrowed(
                        text.as_str().ok_or_else(|| replace_error(type_error))?,
                    )),
                    Value::StringArray(sa) if sa.data().len() == 1 => {
                        entries.push(std::borrow::Cow::Borrowed(
                            sa.data()[0]
                                .as_str()
                                .ok_or_else(|| replace_error(type_error))?,
                        ));
                    }
                    Value::CharArray(ca) if ca.rows <= 1 => {
                        if ca.rows == 0 {
                            entries.push(std::borrow::Cow::Borrowed(""));
                        } else {
                            entries.push(std::borrow::Cow::Owned(
                                char_row_to_string_slice(ca.data(), ca.cols, 0).map_err(|e| {
                                    replace_error_with_message(
                                        format!("{BUILTIN_NAME}: {e}"),
                                        &REPLACE_ERROR_INTERNAL,
                                    )
                                })?,
                            ));
                        }
                    }
                    Value::CharArray(_) => {
                        return Err(replace_error(&REPLACE_ERROR_CELL_ELEMENT));
                    }
                    _ => {
                        return Err(replace_error(&REPLACE_ERROR_CELL_ELEMENT));
                    }
                }
            }
            Ok(entries)
        }
        _ => Err(replace_error(type_error)),
    }
}

enum SearchPattern<'a> {
    Literal(std::borrow::Cow<'a, str>),
    Regex(Regex),
}

struct ReplacementSpec<'a> {
    pairs: Vec<(SearchPattern<'a>, std::borrow::Cow<'a, str>)>,
}

impl<'a> ReplacementSpec<'a> {
    fn from_values(old: &'a Value, new: &'a Value) -> BuiltinResult<Self> {
        let patterns = extract_pattern_list(old)?;
        if patterns.is_empty() {
            return Err(replace_error(&REPLACE_ERROR_EMPTY_PATTERN));
        }

        let replacements = extract_replacement_list(new)?;
        if replacements.is_empty() {
            return Err(replace_error(&REPLACE_ERROR_EMPTY_REPLACEMENT));
        }

        let pairs = if replacements.len() == patterns.len() {
            patterns.into_iter().zip(replacements).collect::<Vec<_>>()
        } else if replacements.len() == 1 {
            let replacement = replacements[0].clone();
            patterns
                .into_iter()
                .map(|pattern| (pattern, replacement.clone()))
                .collect::<Vec<_>>()
        } else {
            return Err(replace_error(&REPLACE_ERROR_SIZE_MISMATCH));
        };

        Ok(Self { pairs })
    }

    fn apply(&self, input: &str) -> String {
        let mut current = input.to_string();
        for (pattern, replacement) in &self.pairs {
            match pattern {
                SearchPattern::Literal(pattern) => {
                    if pattern.is_empty() && replacement.is_empty() {
                        continue;
                    }
                    if pattern == replacement {
                        continue;
                    }
                    current = current.replace(pattern.as_ref(), replacement.as_ref());
                }
                SearchPattern::Regex(pattern) => {
                    current = pattern
                        .replace_all(&current, replacement.as_ref())
                        .to_string();
                }
            }
        }
        current
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn replace_builtin(text: Value, old: Value, new: Value) -> BuiltinResult<Value> {
        futures::executor::block_on(super::replace_builtin(text, old, new))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn replace_string_scalar_single_term() {
        let result = replace_builtin(
            Value::String("RunMat runtime".into()),
            Value::String("runtime".into()),
            Value::String("engine".into()),
        )
        .expect("replace");
        assert_eq!(result, Value::String("RunMat engine".into()));
    }

    #[test]
    fn replace_preserves_every_nd_cell_element_and_shape() {
        let input = make_cell_with_shape(
            vec![
                Value::CharArray(CharArray::new_row("x")),
                Value::string("xx"),
            ],
            vec![1, 1, 2],
        )
        .expect("nd text cells");
        let Value::Cell(output) = replace_builtin(input, Value::string("x"), Value::string("Y"))
            .expect("replace nd cells")
        else {
            panic!("cell output");
        };
        assert_eq!(output.shape(), &[1, 1, 2]);
        assert_eq!(output.data().len(), 2);
        assert_eq!(output.data()[0], Value::CharArray(CharArray::new_row("Y")));
        assert_eq!(output.data()[1], Value::string("YY"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn replace_string_array_multiple_terms() {
        let strings = StringArray::new(
            vec!["gpu".into(), "cpu".into(), "<missing>".into()],
            vec![3, 1],
        )
        .unwrap();
        let result = replace_builtin(
            Value::StringArray(strings),
            Value::StringArray(
                StringArray::new(vec!["gpu".into(), "cpu".into()], vec![2, 1]).unwrap(),
            ),
            Value::String("device".into()),
        )
        .expect("replace");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![3, 1]);
                assert_eq!(
                    sa.data(),
                    vec![
                        String::from("device"),
                        String::from("device"),
                        String::from("<missing>")
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn replace_char_array_adjusts_width() {
        let chars = CharArray::from_unicode_scalars("matrix".chars().collect(), 1, 6).unwrap();
        let result = replace_builtin(
            Value::CharArray(chars),
            Value::String("matrix".into()),
            Value::String("tensor".into()),
        )
        .expect("replace");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 1);
                assert_eq!(out.cols, 6);
                assert_eq!(
                    runmat_value::decode_utf16_code_units(out.data()).unwrap(),
                    "tensor"
                );
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn replace_char_array_handles_padding() {
        let chars = CharArray::from_unicode_scalars(vec!['a', 'b', 'c', 'd'], 2, 2).unwrap();
        let result = replace_builtin(
            Value::CharArray(chars),
            Value::String("b".into()),
            Value::String("beta".into()),
        )
        .expect("replace");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 5);
                assert_eq!(
                    runmat_value::decode_utf16_code_units(out.data()).unwrap(),
                    "abetacd   "
                );
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn replace_cell_array_mixed_content() {
        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("Kernel Planner")),
                Value::String("GPU Fusion".into()),
            ],
            1,
            2,
        )
        .unwrap();
        let result = replace_builtin(
            Value::Cell(cell),
            Value::Cell(
                CellArray::new(
                    vec![Value::String("Kernel".into()), Value::String("GPU".into())],
                    1,
                    2,
                )
                .unwrap(),
            ),
            Value::StringArray(
                StringArray::new(vec!["Shader".into(), "Device".into()], vec![1, 2]).unwrap(),
            ),
        )
        .expect("replace");
        match result {
            Value::Cell(out) => {
                let first = out.get(0, 0).unwrap();
                let second = out.get(0, 1).unwrap();
                assert_eq!(
                    first,
                    Value::CharArray(CharArray::new_row("Shader Planner"))
                );
                assert_eq!(second, Value::String("Device Fusion".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn replace_errors_on_invalid_first_argument() {
        let err = replace_builtin(
            Value::Num(1.0),
            Value::String("a".into()),
            Value::String("b".into()),
        )
        .unwrap_err();
        assert_eq!(err.to_string(), REPLACE_ERROR_INVALID_INPUT.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn replace_errors_on_invalid_pattern_type() {
        let err = replace_builtin(
            Value::String("abc".into()),
            Value::Num(1.0),
            Value::String("x".into()),
        )
        .unwrap_err();
        assert_eq!(err.to_string(), REPLACE_ERROR_PATTERN_TYPE.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn replace_errors_on_size_mismatch() {
        let err = replace_builtin(
            Value::String("abc".into()),
            Value::StringArray(StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap()),
            Value::StringArray(
                StringArray::new(vec!["x".into(), "y".into(), "z".into()], vec![3, 1]).unwrap(),
            ),
        )
        .unwrap_err();
        assert_eq!(err.to_string(), REPLACE_ERROR_SIZE_MISMATCH.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn replace_preserves_missing_string() {
        let result = replace_builtin(
            Value::String(runmat_value::StringElement::Missing),
            Value::String("missing".into()),
            Value::String("value".into()),
        )
        .expect("replace");
        assert_eq!(result, Value::String(runmat_value::StringElement::Missing));
    }

    #[test]
    fn replace_accepts_pattern_object() {
        let pattern = crate::builtins::strings::core::compat::pattern_object(r"\d+");
        let result = replace_builtin(
            Value::String("run42mat".into()),
            pattern,
            Value::String("-".into()),
        )
        .expect("replace");
        assert_eq!(result, Value::String("run-mat".into()));
    }
}
