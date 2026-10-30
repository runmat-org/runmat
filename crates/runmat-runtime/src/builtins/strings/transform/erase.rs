//! MATLAB-compatible `erase` builtin with GPU-aware semantics for RunMat.
use regex::Regex;
use runmat_builtins::{
    BuiltinErrorDescriptor, ERASE_CHAR_MATRIX_EXTENSION, ERASE_ERROR_CELL_ELEMENT,
    ERASE_ERROR_INTERNAL, ERASE_ERROR_INVALID_INPUT, ERASE_ERROR_PATTERN_TYPE,
    ERASE_STRING_CELL_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CellArray, CharArray, StringArray, Value};

use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::common::{
    cell_contains_string, char_row_to_string_slice, is_missing_string, is_multirow_char_array,
};
use crate::builtins::strings::core::compat::pattern_regex;
use crate::{
    build_runtime_error, gather_if_needed_async, make_cell_with_shape, BuiltinResult, RuntimeError,
};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::transform::erase")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "erase",
    op_kind: GpuOpKind::Custom("string-transform"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "The builtin owns resident arguments so it can reject numeric values before provider access; accepted text executes on the host and produces host text.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::transform::erase")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "erase",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "String manipulation builtin; not eligible for fusion plans, and resident numeric values are rejected as non-text.",
};

const BUILTIN_NAME: &str = "erase";

fn map_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

fn erase_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn erase_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    erase_error_with_message(error.message, error)
}

#[runtime_builtin(
    name = "erase",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::transform::erase"
)]
async fn erase_builtin(text: Value, pattern: Value) -> BuiltinResult<Value> {
    if contains_numeric_or_resident(&text) {
        return Err(erase_error(&ERASE_ERROR_INVALID_INPUT));
    }
    if contains_numeric_or_resident(&pattern) {
        return Err(erase_error(&ERASE_ERROR_PATTERN_TYPE));
    }
    if is_multirow_char_array(&text) || is_multirow_char_array(&pattern) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ERASE_CHAR_MATRIX_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if cell_contains_string(&text) || cell_contains_string(&pattern) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ERASE_STRING_CELL_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let text = gather_if_needed_async(&text).await.map_err(map_flow)?;
    let pattern = gather_if_needed_async(&pattern).await.map_err(map_flow)?;

    let patterns = PatternList::from_value(&pattern)?;

    match text {
        Value::String(s) if s.is_missing() => Ok(Value::String(s)),
        Value::String(s) => Ok(Value::string(erase_string_scalar(
            s.as_str().expect("nonmissing subject"),
            &patterns,
        ))),
        Value::StringArray(sa) => erase_string_array(sa, &patterns),
        Value::CharArray(ca) => erase_char_array(ca, &patterns),
        Value::Cell(cell) => erase_cell_array(cell, &patterns),
        _ => Err(erase_error(&ERASE_ERROR_INVALID_INPUT)),
    }
}

fn contains_numeric_or_resident(value: &Value) -> bool {
    match value {
        Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::Tensor(_)
        | Value::LogicalArray(_)
        | Value::Complex(_, _)
        | Value::ComplexTensor(_)
        | Value::GpuTensor(_) => true,
        Value::Cell(cell) => cell.data().iter().any(contains_numeric_or_resident),
        Value::Struct(value) => value.fields.values().any(contains_numeric_or_resident),
        Value::StructArray(array) => array.any_value(contains_numeric_or_resident),
        _ => false,
    }
}

struct PatternList<'a> {
    entries: Vec<PatternEntry<'a>>,
}

enum PatternEntry<'a> {
    Literal(std::borrow::Cow<'a, str>),
    Regex(Regex),
}

impl<'a> PatternList<'a> {
    fn from_value(value: &'a Value) -> BuiltinResult<Self> {
        let entries = match value {
            Value::Object(_) => vec![PatternEntry::Regex(
                Regex::new(&pattern_regex(value, BUILTIN_NAME).map_err(|err| {
                    erase_error_with_message(err.message().to_string(), &ERASE_ERROR_PATTERN_TYPE)
                })?)
                .map_err(|err| {
                    erase_error_with_message(err.to_string(), &ERASE_ERROR_PATTERN_TYPE)
                })?,
            )],
            Value::String(text) => vec![PatternEntry::Literal(std::borrow::Cow::Borrowed(
                text.as_str()
                    .ok_or_else(|| erase_error(&ERASE_ERROR_PATTERN_TYPE))?,
            ))],
            Value::StringArray(array) => array
                .data()
                .iter()
                .map(|text| {
                    text.as_str()
                        .map(|text| PatternEntry::Literal(std::borrow::Cow::Borrowed(text)))
                        .ok_or_else(|| erase_error(&ERASE_ERROR_PATTERN_TYPE))
                })
                .collect::<BuiltinResult<Vec<_>>>()?,
            Value::CharArray(array) => {
                if array.rows == 0 {
                    Vec::new()
                } else {
                    let mut list = Vec::with_capacity(array.rows);
                    for row in 0..array.rows {
                        list.push(PatternEntry::Literal(std::borrow::Cow::Owned(
                            char_row_to_string_slice(array.data(), array.cols, row).map_err(
                                |e| {
                                    erase_error_with_message(
                                        format!("{BUILTIN_NAME}: {e}"),
                                        &ERASE_ERROR_INTERNAL,
                                    )
                                },
                            )?,
                        )));
                    }
                    list
                }
            }
            Value::Cell(cell) => {
                let mut list = Vec::with_capacity(cell.data().len());
                for handle in cell.data() {
                    match &handle {
                        Value::String(text) => {
                            list.push(PatternEntry::Literal(std::borrow::Cow::Borrowed(
                                text.as_str()
                                    .ok_or_else(|| erase_error(&ERASE_ERROR_PATTERN_TYPE))?,
                            )))
                        }
                        Value::StringArray(sa) if sa.data().len() == 1 => {
                            list.push(PatternEntry::Literal(std::borrow::Cow::Borrowed(
                                sa.data()[0]
                                    .as_str()
                                    .ok_or_else(|| erase_error(&ERASE_ERROR_PATTERN_TYPE))?,
                            )));
                        }
                        Value::CharArray(ca) if ca.rows == 0 => {
                            list.push(PatternEntry::Literal(std::borrow::Cow::Borrowed("")));
                        }
                        Value::CharArray(ca) if ca.rows == 1 => {
                            list.push(PatternEntry::Literal(std::borrow::Cow::Owned(
                                char_row_to_string_slice(ca.data(), ca.cols, 0).map_err(|e| {
                                    erase_error_with_message(
                                        format!("{BUILTIN_NAME}: {e}"),
                                        &ERASE_ERROR_INTERNAL,
                                    )
                                })?,
                            )));
                        }
                        Value::CharArray(_) => return Err(erase_error(&ERASE_ERROR_CELL_ELEMENT)),
                        _ => return Err(erase_error(&ERASE_ERROR_CELL_ELEMENT)),
                    }
                }
                list
            }
            _ => return Err(erase_error(&ERASE_ERROR_PATTERN_TYPE)),
        };
        Ok(Self { entries })
    }

    fn apply(&self, input: &str) -> String {
        if self.entries.is_empty() {
            return input.to_string();
        }
        let mut current = input.to_string();
        for pattern in &self.entries {
            match pattern {
                PatternEntry::Literal(pattern) => {
                    if pattern.is_empty() {
                        continue;
                    }
                    current = current.replace(pattern.as_ref(), "");
                }
                PatternEntry::Regex(pattern) => {
                    current = pattern.replace_all(&current, "").to_string();
                }
            }
            if current.is_empty() {
                break;
            }
        }
        current
    }
}

fn erase_string_scalar(text: &str, patterns: &PatternList<'_>) -> String {
    patterns.apply(text)
}

fn erase_string_array(array: StringArray, patterns: &PatternList<'_>) -> BuiltinResult<Value> {
    let (data, shape, _, _) = array.into_parts();
    let mut erased = Vec::with_capacity(data.len());
    let input: Vec<runmat_value::StringElement> = match data.try_into_owned() {
        Ok(values) => values.into_iter().collect(),
        Err(shared) => shared.iter().cloned().collect(),
    };
    for entry in input {
        if is_missing_string(&entry) {
            erased.push(entry);
        } else {
            erased.push(runmat_value::StringElement::text(
                patterns.apply(entry.as_str().expect("nonmissing subject")),
            ));
        }
    }
    StringArray::from_shared_storage(
        runmat_value::ExactArrayStorage::shared_unbounded(erased),
        shape,
    )
    .map(Value::StringArray)
    .map_err(|e| erase_error_with_message(format!("{BUILTIN_NAME}: {e}"), &ERASE_ERROR_INTERNAL))
}

fn erase_char_array(array: CharArray, patterns: &PatternList<'_>) -> BuiltinResult<Value> {
    if array.rows == 0 {
        return Ok(Value::CharArray(array));
    }
    let (data, _, rows, cols) = array.into_parts();

    let mut processed: Vec<String> = Vec::with_capacity(rows);
    let mut target_cols = 0usize;
    for row in 0..rows {
        let slice = char_row_to_string_slice(&data, cols, row).map_err(|e| {
            erase_error_with_message(format!("{BUILTIN_NAME}: {e}"), &ERASE_ERROR_INTERNAL)
        })?;
        let erased = patterns.apply(&slice);
        let len = erased.encode_utf16().count();
        if len > target_cols {
            target_cols = len;
        }
        processed.push(erased);
    }

    let mut flattened = Vec::with_capacity(rows * target_cols);
    for row_text in processed {
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
            erase_error_with_message(format!("{BUILTIN_NAME}: {e}"), &ERASE_ERROR_INTERNAL)
        })
}

fn erase_cell_array(cell: CellArray, patterns: &PatternList<'_>) -> BuiltinResult<Value> {
    let shape = cell.shape().to_vec();
    let mut values = Vec::with_capacity(cell.data().len());
    for handle in cell.data() {
        values.push(erase_cell_element(handle, patterns)?);
    }
    make_cell_with_shape(values, shape).map_err(|e| {
        erase_error_with_message(format!("{BUILTIN_NAME}: {e}"), &ERASE_ERROR_INTERNAL)
    })
}

fn erase_cell_element(value: &Value, patterns: &PatternList<'_>) -> BuiltinResult<Value> {
    match value {
        Value::String(text) => Ok(Value::String(match text.as_str() {
            Some(text) => runmat_value::StringElement::text(erase_string_scalar(text, patterns)),
            None => runmat_value::StringElement::Missing,
        })),
        Value::StringArray(sa) if sa.data().len() == 1 => {
            erase_cell_element(&Value::String(sa.data()[0].clone()), patterns)
        }
        Value::CharArray(ca) if ca.rows == 0 => Ok(Value::CharArray(ca.clone())),
        Value::CharArray(ca) if ca.rows == 1 => {
            let slice = char_row_to_string_slice(ca.data(), ca.cols, 0).map_err(|e| {
                erase_error_with_message(format!("{BUILTIN_NAME}: {e}"), &ERASE_ERROR_INTERNAL)
            })?;
            let erased = patterns.apply(&slice);
            Ok(Value::CharArray(CharArray::new_row(&erased)))
        }
        Value::CharArray(_) => Err(erase_error(&ERASE_ERROR_CELL_ELEMENT)),
        _ => Err(erase_error(&ERASE_ERROR_CELL_ELEMENT)),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn erase_builtin(text: Value, pattern: Value) -> BuiltinResult<Value> {
        futures::executor::block_on(super::erase_builtin(text, pattern))
    }

    #[test]
    fn erase_char_matrix_requires_runmat_mode() {
        let matrix = Value::CharArray(
            CharArray::from_unicode_scalars("abcdwxyz".chars().collect(), 2, 4)
                .expect("character matrix"),
        );
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        for (text, pattern) in [
            (matrix.clone(), Value::String("a".into())),
            (Value::String("abcd".into()), matrix),
        ] {
            let error = erase_builtin(text, pattern).expect_err("char matrix is an extension");
            assert_eq!(
                error.identifier(),
                ERASE_CHAR_MATRIX_EXTENSION.error_identifier
            );
        }
    }

    #[test]
    fn erase_string_cell_requires_runmat_mode() {
        let cell = Value::Cell(
            CellArray::new(vec![Value::String("abcd".into())], 1, 1).expect("string cell"),
        );
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        for (text, pattern) in [
            (cell.clone(), Value::String("a".into())),
            (Value::String("abcd".into()), cell),
        ] {
            let error = erase_builtin(text, pattern).expect_err("string cell is an extension");
            assert_eq!(
                error.identifier(),
                ERASE_STRING_CELL_EXTENSION.error_identifier
            );
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erase_string_scalar_single_pattern() {
        let result = erase_builtin(
            Value::String("RunMat runtime".into()),
            Value::String(" runtime".into()),
        )
        .expect("erase");
        assert_eq!(result, Value::String("RunMat".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erase_string_array_multiple_patterns() {
        let strings = StringArray::from_elements_unbounded(
            vec![
                "gpu".into(),
                "cpu".into(),
                "<missing>".into(),
                runmat_value::StringElement::Missing,
            ],
            vec![4, 1],
        )
        .unwrap();
        let result = erase_builtin(
            Value::StringArray(strings),
            Value::StringArray(StringArray::new(vec!["g".into(), "c".into()], vec![2, 1]).unwrap()),
        )
        .expect("erase");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![4, 1]);
                assert_eq!(
                    sa.data(),
                    vec![
                        "pu".into(),
                        "pu".into(),
                        "<missin>".into(),
                        runmat_value::StringElement::Missing,
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erase_string_array_shape_mismatch_applies_all_patterns() {
        let strings =
            StringArray::new(vec!["GPU kernel".into(), "CPU kernel".into()], vec![2, 1]).unwrap();
        let patterns = StringArray::new(vec!["GPU ".into(), "CPU ".into()], vec![1, 2]).unwrap();
        let result = erase_builtin(Value::StringArray(strings), Value::StringArray(patterns))
            .expect("erase");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![2, 1]);
                assert_eq!(
                    sa.data(),
                    vec![String::from("kernel"), String::from("kernel")]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erase_char_array_adjusts_width() {
        let chars = CharArray::from_unicode_scalars("matrix".chars().collect(), 1, 6).unwrap();
        let result =
            erase_builtin(Value::CharArray(chars), Value::String("tr".into())).expect("erase");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 1);
                assert_eq!(out.cols, 4);
                assert_eq!(
                    runmat_value::decode_utf16_code_units(out.data()).unwrap(),
                    "maix"
                );
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erase_char_array_handles_full_removal() {
        let chars = CharArray::new_row("abc");
        let result = erase_builtin(Value::CharArray(chars.clone()), Value::String("abc".into()))
            .expect("erase");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 1);
                assert_eq!(out.cols, 0);
                assert!(out.data().is_empty());
            }
            other => panic!("expected empty char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erase_char_array_multiple_rows_sequential_patterns() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let chars = CharArray::from_unicode_scalars(
            vec![
                'G', 'P', 'U', ' ', 'p', 'i', 'p', 'e', 'l', 'i', 'n', 'e', 'C', 'P', 'U', ' ',
                'p', 'i', 'p', 'e', 'l', 'i', 'n', 'e',
            ],
            2,
            12,
        )
        .unwrap();
        let patterns = CharArray::new_row("GPU ");
        let result =
            erase_builtin(Value::CharArray(chars), Value::CharArray(patterns)).expect("erase");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 12);
                let first = char_row_to_string_slice(out.data(), out.cols, 0).unwrap();
                let second = char_row_to_string_slice(out.data(), out.cols, 1).unwrap();
                assert_eq!(first.trim_end(), "pipeline");
                assert_eq!(second.trim_end(), "CPU pipeline");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erase_cell_array_mixed_content() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("Kernel Planner")),
                Value::String("GPU Fusion".into()),
            ],
            1,
            2,
        )
        .unwrap();
        let result = erase_builtin(
            Value::Cell(cell),
            Value::Cell(
                CellArray::new(
                    vec![
                        Value::String("Kernel ".into()),
                        Value::String("GPU ".into()),
                    ],
                    1,
                    2,
                )
                .unwrap(),
            ),
        )
        .expect("erase");
        match result {
            Value::Cell(out) => {
                let first = out.get(0, 0).unwrap();
                let second = out.get(0, 1).unwrap();
                assert_eq!(first, Value::CharArray(CharArray::new_row("Planner")));
                assert_eq!(second, Value::String("Fusion".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erase_cell_array_preserves_shape() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let cell = CellArray::new(
            vec![
                Value::String("alpha".into()),
                Value::String("beta".into()),
                Value::String("gamma".into()),
                Value::String("delta".into()),
            ],
            2,
            2,
        )
        .unwrap();
        let patterns = StringArray::new(vec!["a".into()], vec![1, 1]).unwrap();
        let result = erase_builtin(Value::Cell(cell), Value::StringArray(patterns)).expect("erase");
        match result {
            Value::Cell(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 2);
                assert_eq!(out.get(0, 0).unwrap(), Value::String("lph".into()));
                assert_eq!(out.get(1, 1).unwrap(), Value::String("delt".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erase_preserves_missing_string() {
        let result = erase_builtin(
            Value::String(runmat_value::StringElement::Missing),
            Value::String("missing".into()),
        )
        .expect("erase");
        assert_eq!(result, Value::String(runmat_value::StringElement::Missing));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erase_allows_empty_pattern_list() {
        let strings = StringArray::new(vec!["alpha".into(), "beta".into()], vec![2, 1]).unwrap();
        let pattern = StringArray::new(Vec::<String>::new(), vec![0, 0]).unwrap();
        let result = erase_builtin(
            Value::StringArray(strings.clone()),
            Value::StringArray(pattern),
        )
        .expect("erase");
        assert_eq!(result, Value::StringArray(strings));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erase_errors_on_invalid_first_argument() {
        let err = erase_builtin(Value::Num(1.0), Value::String("a".into())).unwrap_err();
        assert_eq!(err.to_string(), ERASE_ERROR_INVALID_INPUT.message);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn erase_errors_on_invalid_pattern_type() {
        let err = erase_builtin(Value::String("abc".into()), Value::Num(1.0)).unwrap_err();
        assert_eq!(err.to_string(), ERASE_ERROR_PATTERN_TYPE.message);
    }

    #[test]
    fn erase_rejects_nested_integer_text_before_conversion() {
        let cell = CellArray::new(
            vec![Value::Tensor(
                runmat_value::Tensor::new_integer(
                    runmat_value::IntegerStorage::U64(vec![u64::MAX]),
                    vec![1, 1],
                )
                .expect("integer tensor"),
            )],
            1,
            1,
        )
        .expect("cell");
        let error = erase_builtin(Value::Cell(cell), Value::String("x".into()))
            .expect_err("numeric cell is invalid text");
        assert_eq!(error.identifier(), ERASE_ERROR_INVALID_INPUT.identifier);
    }

    #[test]
    fn erase_rejects_resident_numeric_pattern_before_gather() {
        use crate::builtins::common::test_support;
        use runmat_accelerate_api::HostTensorView;

        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload(&HostTensorView {
                    data: &[1.0],
                    shape: &[1, 1],
                })
                .expect("upload");
            provider.reset_telemetry();
            let error = erase_builtin(
                Value::String("abc".into()),
                Value::GpuTensor(handle.clone()),
            )
            .expect_err("resident numeric pattern is invalid");
            assert_eq!(error.identifier(), ERASE_ERROR_PATTERN_TYPE.identifier);
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            handle.release().expect("release fixture handle");
        });
    }

    #[test]
    fn erase_dispatch_preserves_residency_until_builtin_validation() {
        assert_eq!(GPU_SPEC.residency, ResidencyPolicy::NewHandle);
    }

    #[test]
    fn erase_accepts_pattern_object() {
        let pattern = crate::builtins::strings::core::compat::pattern_object(r"\d+");
        let result = erase_builtin(Value::String("run42mat".into()), pattern).expect("erase");
        assert_eq!(result, Value::String("runmat".into()));
    }
}
