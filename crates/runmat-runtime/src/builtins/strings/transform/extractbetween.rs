//! MATLAB-compatible `extractBetween` builtin with GPU-aware semantics for RunMat.

use runmat_value::StringElement;
use std::cmp::min;

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};
use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::strings::common::{
    cell_contains_string, char_row_to_string_slice, is_multirow_char_array,
};
use crate::{
    build_runtime_error, gather_if_needed_async, make_cell_with_shape, BuiltinResult, RuntimeError,
};
#[cfg(test)]
use runmat_builtins::EXTRACT_BETWEEN_INTEGER_CAPABILITIES;
use runmat_builtins::{
    BuiltinErrorDescriptor, EXTRACT_BETWEEN_CHAR_MATRIX_EXTENSION as CHAR_MATRIX_EXTENSION,
    EXTRACT_BETWEEN_ERROR_BOUNDARY_TYPE, EXTRACT_BETWEEN_ERROR_CELL_ELEMENT,
    EXTRACT_BETWEEN_ERROR_INTERNAL, EXTRACT_BETWEEN_ERROR_INVALID_INPUT,
    EXTRACT_BETWEEN_ERROR_NAME_VALUE_PAIR, EXTRACT_BETWEEN_ERROR_OPTION_NAME,
    EXTRACT_BETWEEN_ERROR_OPTION_VALUE, EXTRACT_BETWEEN_ERROR_POSITION_TYPE,
    EXTRACT_BETWEEN_ERROR_SIZE_MISMATCH,
    EXTRACT_BETWEEN_FULL_BROADCAST_EXTENSION as FULL_BROADCAST_EXTENSION,
    EXTRACT_BETWEEN_RESIDENT_POSITION_EXTENSION as RESIDENT_POSITION_EXTENSION,
    EXTRACT_BETWEEN_STRING_CELL_EXTENSION as STRING_CELL_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, IntValue, NumericScalar, StringArray, Value};

mod cell_text;
use self::cell_text::{CellElementKind, TextCellScalar};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::strings::transform::extractbetween"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "extractBetween",
    op_kind: GpuOpKind::Custom("string-transform"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Runs on host text; an admitted resident numeric position control may gather before extraction. Resident text rejects before provider access.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::strings::transform::extractbetween"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "extractBetween",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Pure host text manipulation excluded from fusion; only admitted resident position controls may gather.",
};

const BUILTIN_NAME: &str = "extractBetween";

fn extract_between_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    extract_between_error_with_message(error.message, error)
}

fn extract_between_utf16_error(source: runmat_value::Utf16DecodeError) -> RuntimeError {
    let mut builder = build_runtime_error("extractBetween: malformed UTF-16 character input")
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = EXTRACT_BETWEEN_ERROR_INVALID_INPUT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn extract_between_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn map_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoundariesMode {
    Exclusive,
    Inclusive,
}

#[runtime_builtin(
    name = "extractBetween",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::transform::extractbetween"
)]
async fn extract_between_builtin(
    text: Value,
    start: Value,
    stop: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if !valid_extract_between_text_input(&text) {
        return Err(extract_between_error(&EXTRACT_BETWEEN_ERROR_INVALID_INPUT));
    }
    if [&start, &stop].into_iter().any(|boundary| {
        matches!(boundary, Value::Cell(_)) && !valid_extract_between_text_input(boundary)
    }) {
        return Err(extract_between_error(&EXTRACT_BETWEEN_ERROR_CELL_ELEMENT));
    }
    if is_resident_or_contains_resident(&start) || is_resident_or_contains_resident(&stop) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &RESIDENT_POSITION_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if is_multirow_char_array(&text)
        || is_multirow_char_array(&start)
        || is_multirow_char_array(&stop)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CHAR_MATRIX_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if cell_contains_string(&text) || cell_contains_string(&start) || cell_contains_string(&stop) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &STRING_CELL_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if raw_boundary_uses_full_broadcast(&text, &start)
        || raw_boundary_uses_full_broadcast(&text, &stop)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FULL_BROADCAST_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if rest
        .iter()
        .any(|value| is_numeric_or_resident(value) || contains_nested_numeric_or_resident(value))
    {
        return Err(extract_between_error(&EXTRACT_BETWEEN_ERROR_OPTION_VALUE));
    }
    let mode_override = parse_boundaries_option(rest.view()).await?;
    let text = gather_if_needed_async(&text).await.map_err(map_flow)?;
    let start = gather_if_needed_async(&start).await.map_err(map_flow)?;
    let stop = gather_if_needed_async(&stop).await.map_err(map_flow)?;

    let normalized_text = NormalizedText::from_value(text)?;
    let start_boundary = BoundaryArg::from_value(start)?;
    let stop_boundary = BoundaryArg::from_value(stop)?;

    if start_boundary.kind() != stop_boundary.kind() {
        return Err(extract_between_error(&EXTRACT_BETWEEN_ERROR_BOUNDARY_TYPE));
    }
    let boundary_kind = start_boundary.kind();
    let effective_mode = mode_override.unwrap_or(match boundary_kind {
        BoundaryKind::Text => BoundariesMode::Exclusive,
        BoundaryKind::Position => BoundariesMode::Inclusive,
    });

    let start_shape = start_boundary.shape();
    let stop_shape = stop_boundary.shape();
    let text_shape = normalized_text.shape();

    if !shape_is_scalar_or_same(start_shape, text_shape)
        || !shape_is_scalar_or_same(stop_shape, text_shape)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &FULL_BROADCAST_EXTENSION,
            BUILTIN_NAME,
        )?;
    }

    let shape_ts = broadcast_shapes(BUILTIN_NAME, text_shape, start_shape).map_err(|err| {
        extract_between_error_with_message(
            format!("{}: {err}", EXTRACT_BETWEEN_ERROR_SIZE_MISMATCH.message),
            &EXTRACT_BETWEEN_ERROR_SIZE_MISMATCH,
        )
    })?;
    let output_shape = broadcast_shapes(BUILTIN_NAME, &shape_ts, stop_shape).map_err(|err| {
        extract_between_error_with_message(
            format!("{}: {err}", EXTRACT_BETWEEN_ERROR_SIZE_MISMATCH.message),
            &EXTRACT_BETWEEN_ERROR_SIZE_MISMATCH,
        )
    })?;
    if !normalized_text.supports_shape(&output_shape) {
        return Err(extract_between_error(&EXTRACT_BETWEEN_ERROR_SIZE_MISMATCH));
    }

    let total: usize = output_shape.iter().copied().product();
    if total == 0 {
        return normalized_text.into_value(Vec::new(), output_shape);
    }

    let text_strides = compute_strides(text_shape);
    let start_strides = compute_strides(start_shape);
    let stop_strides = compute_strides(stop_shape);

    let mut results = Vec::with_capacity(total);

    for idx in 0..total {
        let text_idx = broadcast_index(idx, &output_shape, text_shape, &text_strides);
        let start_idx = broadcast_index(idx, &output_shape, start_shape, &start_strides);
        let stop_idx = broadcast_index(idx, &output_shape, stop_shape, &stop_strides);

        let result = match boundary_kind {
            BoundaryKind::Text => {
                let text_value = normalized_text.data(text_idx);
                let start_value = start_boundary.text(start_idx);
                let stop_value = stop_boundary.text(stop_idx);
                extract_with_text_boundaries(text_value, start_value, stop_value, effective_mode)
            }
            BoundaryKind::Position => {
                let text_value = normalized_text.data(text_idx);
                let start_value = start_boundary.position(start_idx);
                let stop_value = stop_boundary.position(stop_idx);
                extract_with_positions(text_value, start_value, stop_value, effective_mode)
            }
        };
        results.push(result);
    }

    normalized_text.into_value(results, output_shape)
}

fn valid_extract_between_text_input(value: &Value) -> bool {
    match value {
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => true,
        Value::Cell(cell) => {
            for element in cell.data() {
                if TextCellScalar::from_value(element).is_none() {
                    return false;
                }
            }
            true
        }
        _ => false,
    }
}

fn is_numeric_or_resident(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Bool(_)
            | Value::Tensor(_)
            | Value::LogicalArray(_)
            | Value::Complex(_, _)
            | Value::ComplexTensor(_)
            | Value::GpuTensor(_)
    )
}

fn contains_nested_numeric_or_resident(value: &Value) -> bool {
    match value {
        Value::Cell(cell) => cell.data().iter().any(|value| {
            is_numeric_or_resident(value) || contains_nested_numeric_or_resident(value)
        }),
        Value::Struct(value) => value.fields.values().any(|value| {
            is_numeric_or_resident(value) || contains_nested_numeric_or_resident(value)
        }),
        Value::StructArray(array) => array.any_value(|value| {
            is_numeric_or_resident(value) || contains_nested_numeric_or_resident(value)
        }),
        _ => false,
    }
}

fn is_resident_or_contains_resident(value: &Value) -> bool {
    match value {
        Value::GpuTensor(_) => true,
        Value::Cell(cell) => cell.data().iter().any(is_resident_or_contains_resident),
        Value::Struct(value) => value.fields.values().any(is_resident_or_contains_resident),
        Value::StructArray(array) => array.any_value(is_resident_or_contains_resident),
        _ => false,
    }
}

fn raw_text_shape(value: &Value) -> Option<Vec<usize>> {
    match value {
        Value::String(_) => Some(vec![1, 1]),
        Value::StringArray(array) => Some(array.shape.to_vec()),
        Value::CharArray(array) if array.rows <= 1 => Some(vec![1, 1]),
        Value::CharArray(array) => Some(vec![array.rows, 1]),
        Value::Cell(cell) => Some(cell.shape.to_vec()),
        _ => None,
    }
}

fn raw_boundary_shape(value: &Value) -> Option<Vec<usize>> {
    match value {
        Value::String(_) | Value::Num(_) | Value::Int(_) | Value::Object(_) => Some(vec![1, 1]),
        Value::StringArray(array) => Some(array.shape.to_vec()),
        Value::CharArray(array) if array.rows <= 1 => Some(vec![1, 1]),
        Value::CharArray(array) => Some(vec![array.rows, 1]),
        Value::Cell(cell) => Some(cell.shape.to_vec()),
        Value::Tensor(tensor) => Some(tensor.shape().to_vec()),
        Value::GpuTensor(handle) => Some(handle.shape().to_vec()),
        _ => None,
    }
}

fn raw_boundary_uses_full_broadcast(text: &Value, boundary: &Value) -> bool {
    let (Some(text_shape), Some(boundary_shape)) =
        (raw_text_shape(text), raw_boundary_shape(boundary))
    else {
        return false;
    };
    !shape_is_scalar_or_same(&boundary_shape, &text_shape)
}

fn shape_is_scalar_or_same(shape: &[usize], text_shape: &[usize]) -> bool {
    shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        == Some(1)
        || shape == text_shape
}

async fn parse_boundaries_option(
    args: crate::call::FunctionArgumentView<'_>,
) -> BuiltinResult<Option<BoundariesMode>> {
    if args.is_empty() {
        return Ok(None);
    }
    if !args.len().is_multiple_of(2) {
        return Err(extract_between_error(
            &EXTRACT_BETWEEN_ERROR_NAME_VALUE_PAIR,
        ));
    }

    let mut mode: Option<BoundariesMode> = None;
    let mut idx = 0;
    while idx < args.len() {
        let name_value = gather_if_needed_async(&args[idx]).await.map_err(map_flow)?;
        let name = value_to_string(&name_value)?
            .ok_or_else(|| extract_between_error(&EXTRACT_BETWEEN_ERROR_OPTION_NAME))?;
        if !name.eq_ignore_ascii_case("boundaries") {
            return Err(extract_between_error(&EXTRACT_BETWEEN_ERROR_OPTION_NAME));
        }
        let value = gather_if_needed_async(&args[idx + 1])
            .await
            .map_err(map_flow)?;
        let value_str = value_to_string(&value)?
            .ok_or_else(|| extract_between_error(&EXTRACT_BETWEEN_ERROR_OPTION_VALUE))?;
        let parsed_mode = if value_str.eq_ignore_ascii_case("inclusive") {
            BoundariesMode::Inclusive
        } else if value_str.eq_ignore_ascii_case("exclusive") {
            BoundariesMode::Exclusive
        } else {
            return Err(extract_between_error(&EXTRACT_BETWEEN_ERROR_OPTION_VALUE));
        };
        mode = Some(parsed_mode);
        idx += 2;
    }
    Ok(mode)
}

fn value_to_string(value: &Value) -> BuiltinResult<Option<String>> {
    Ok(match value {
        Value::String(s) => s.as_str().map(str::to_owned),
        Value::StringArray(sa) if sa.data().len() == 1 => sa.data()[0].as_str().map(str::to_owned),
        Value::CharArray(ca) if ca.rows <= 1 => {
            if ca.rows == 0 {
                Some(String::new())
            } else {
                Some(
                    char_row_to_string_slice(ca.data(), ca.cols, 0).map_err(|e| {
                        extract_between_error_with_message(
                            format!("{BUILTIN_NAME}: {e}"),
                            &EXTRACT_BETWEEN_ERROR_INTERNAL,
                        )
                    })?,
                )
            }
        }
        Value::CharArray(_) => None,
        Value::Cell(cell) if cell.data().len() == 1 => {
            let element = &cell.data()[0];
            value_to_string(element)?
        }
        _ => None,
    })
}

#[derive(Clone)]
struct ExtractResult {
    text: StringElement,
}

impl ExtractResult {
    fn missing() -> Self {
        Self {
            text: StringElement::Missing,
        }
    }

    fn text(text: String) -> Self {
        Self {
            text: StringElement::text(text),
        }
    }
}

fn extract_with_text_boundaries(
    text: &StringElement,
    start: &StringElement,
    stop: &StringElement,
    mode: BoundariesMode,
) -> ExtractResult {
    let (Some(text), Some(start), Some(stop)) = (text.as_str(), start.as_str(), stop.as_str())
    else {
        return ExtractResult::missing();
    };

    if let Some(start_idx) = text.find(start) {
        let search_start = start_idx + start.len();
        if search_start > text.len() {
            return ExtractResult::text(String::new());
        }
        if let Some(relative_end) = text[search_start..].find(stop) {
            let end_idx = search_start + relative_end;
            match mode {
                BoundariesMode::Inclusive => {
                    let end_capture = min(text.len(), end_idx + stop.len());
                    let slice = &text[start_idx..end_capture];
                    ExtractResult::text(slice.to_string())
                }
                BoundariesMode::Exclusive => {
                    if end_idx < search_start {
                        ExtractResult::text(String::new())
                    } else {
                        let slice = &text[search_start..end_idx];
                        ExtractResult::text(slice.to_string())
                    }
                }
            }
        } else {
            ExtractResult::text(String::new())
        }
    } else {
        ExtractResult::text(String::new())
    }
}

fn extract_with_positions(
    text: &StringElement,
    start: usize,
    stop: usize,
    mode: BoundariesMode,
) -> ExtractResult {
    let Some(text) = text.as_str() else {
        return ExtractResult::missing();
    };
    if text.is_empty() {
        return ExtractResult::text(String::new());
    }
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    if len == 0 {
        return ExtractResult::text(String::new());
    }

    if start == 0 || stop == 0 {
        return ExtractResult::text(String::new());
    }

    if start > len {
        return ExtractResult::text(String::new());
    }
    let stop_clamped = stop.min(len);
    if stop_clamped == 0 {
        return ExtractResult::text(String::new());
    }

    match mode {
        BoundariesMode::Inclusive => {
            if start > stop_clamped {
                return ExtractResult::text(String::new());
            }
            let start_idx = start - 1;
            let end_idx = stop_clamped - 1;
            if start_idx >= len || end_idx >= len || start_idx > end_idx {
                ExtractResult::text(String::new())
            } else {
                let slice: String = chars[start_idx..=end_idx].iter().collect();
                ExtractResult::text(slice)
            }
        }
        BoundariesMode::Exclusive => {
            if start + 1 >= stop_clamped {
                return ExtractResult::text(String::new());
            }
            let start_idx = start;
            let end_idx = stop_clamped - 2;
            if start_idx >= len || end_idx >= len || start_idx > end_idx {
                ExtractResult::text(String::new())
            } else {
                let slice: String = chars[start_idx..=end_idx].iter().collect();
                ExtractResult::text(slice)
            }
        }
    }
}

#[derive(Clone, Debug)]
struct CellInfo {
    shape: Vec<usize>,
    element_kinds: Vec<CellElementKind>,
}

#[derive(Clone, Debug)]
enum TextKind {
    StringScalar,
    StringArray,
    CharArray { rows: usize },
    CellArray(CellInfo),
}

#[derive(Clone, Debug)]
struct NormalizedText {
    data: Vec<StringElement>,
    shape: Vec<usize>,
    kind: TextKind,
}

impl NormalizedText {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::String(s) => Ok(Self {
                data: vec![s],
                shape: vec![1, 1],
                kind: TextKind::StringScalar,
            }),
            Value::StringArray(sa) => Ok(Self {
                data: sa.data().to_vec(),
                shape: sa.shape.to_vec(),
                kind: TextKind::StringArray,
            }),
            Value::CharArray(ca) => {
                let rows = ca.rows.max(1);
                let mut data = Vec::with_capacity(rows);
                if ca.rows == 0 {
                    data.push(StringElement::text(""));
                } else {
                    for row in 0..ca.rows {
                        data.push(
                            char_row_to_string_slice(ca.data(), ca.cols, row)
                                .map(StringElement::text)
                                .map_err(|e| {
                                    extract_between_error_with_message(
                                        format!("{BUILTIN_NAME}: {e}"),
                                        &EXTRACT_BETWEEN_ERROR_INTERNAL,
                                    )
                                })?,
                        );
                    }
                }
                Ok(Self {
                    data,
                    shape: if ca.rows <= 1 {
                        vec![1, 1]
                    } else {
                        vec![rows, 1]
                    },
                    kind: TextKind::CharArray { rows },
                })
            }
            Value::Cell(cell) => {
                let shape = cell.shape.to_vec();
                let mut data = Vec::with_capacity(cell.data().len());
                let mut kinds = Vec::with_capacity(cell.data().len());
                for element in cell.data() {
                    let scalar = TextCellScalar::from_value(element).ok_or_else(|| {
                        extract_between_error(&EXTRACT_BETWEEN_ERROR_CELL_ELEMENT)
                    })?;
                    data.push(scalar.text().map_err(extract_between_utf16_error)?);
                    kinds.push(scalar.kind());
                }
                Ok(Self {
                    data,
                    shape: shape.clone(),
                    kind: TextKind::CellArray(CellInfo {
                        shape,
                        element_kinds: kinds,
                    }),
                })
            }
            _ => Err(extract_between_error(&EXTRACT_BETWEEN_ERROR_INVALID_INPUT)),
        }
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self, idx: usize) -> &StringElement {
        &self.data[idx]
    }

    fn supports_shape(&self, output_shape: &[usize]) -> bool {
        match &self.kind {
            TextKind::StringScalar => true,
            TextKind::StringArray => true,
            TextKind::CharArray { .. } => output_shape == self.shape,
            TextKind::CellArray(info) => output_shape == info.shape,
        }
    }

    fn into_value(
        self,
        results: Vec<ExtractResult>,
        output_shape: Vec<usize>,
    ) -> BuiltinResult<Value> {
        match self.kind {
            TextKind::StringScalar => {
                if results.len() <= 1 {
                    let value = results
                        .into_iter()
                        .next()
                        .unwrap_or_else(|| ExtractResult::text(String::new()));
                    Ok(Value::String(value.text))
                } else {
                    let data = results.into_iter().map(|r| r.text).collect::<Vec<_>>();
                    let array =
                        StringArray::from_elements_unbounded(data, output_shape).map_err(|e| {
                            extract_between_error_with_message(
                                format!("{BUILTIN_NAME}: {e}"),
                                &EXTRACT_BETWEEN_ERROR_INTERNAL,
                            )
                        })?;
                    Ok(Value::StringArray(array))
                }
            }
            TextKind::StringArray => {
                let data = results.into_iter().map(|r| r.text).collect::<Vec<_>>();
                let array =
                    StringArray::from_elements_unbounded(data, output_shape).map_err(|e| {
                        extract_between_error_with_message(
                            format!("{BUILTIN_NAME}: {e}"),
                            &EXTRACT_BETWEEN_ERROR_INTERNAL,
                        )
                    })?;
                Ok(Value::StringArray(array))
            }
            TextKind::CharArray { rows } => {
                if results.len() != rows {
                    return Err(extract_between_error(&EXTRACT_BETWEEN_ERROR_SIZE_MISMATCH));
                }
                if rows == 1 {
                    let text = results
                        .into_iter()
                        .next()
                        .unwrap_or_else(|| ExtractResult::text(String::new()))
                        .text;
                    return make_cell_with_shape(
                        vec![match text.as_str() {
                            Some(text) => Value::CharArray(CharArray::new_row(text)),
                            None => Value::String(StringElement::Missing),
                        }],
                        vec![1, 1],
                    )
                    .map_err(|e| {
                        extract_between_error_with_message(
                            format!("{BUILTIN_NAME}: {e}"),
                            &EXTRACT_BETWEEN_ERROR_INTERNAL,
                        )
                    });
                }
                let mut max_width = 0usize;
                let mut row_strings = Vec::with_capacity(rows);
                for result in &results {
                    let width = result
                        .text
                        .as_str()
                        .ok_or_else(|| extract_between_error(&EXTRACT_BETWEEN_ERROR_BOUNDARY_TYPE))?
                        .encode_utf16()
                        .count();
                    max_width = max_width.max(width);
                    row_strings.push(result.text.clone());
                }
                let mut flattened = Vec::with_capacity(rows * max_width);
                for row in row_strings {
                    let mut units = row
                        .as_str()
                        .ok_or_else(|| extract_between_error(&EXTRACT_BETWEEN_ERROR_BOUNDARY_TYPE))?
                        .encode_utf16()
                        .map(runmat_value::Utf16CodeUnit::from_raw)
                        .collect::<Vec<_>>();
                    if units.len() < max_width {
                        units.resize(
                            max_width,
                            runmat_value::Utf16CodeUnit::from_raw(b' ' as u16),
                        );
                    }
                    flattened.extend(units);
                }
                CharArray::from_code_units(flattened, rows, max_width)
                    .map(Value::CharArray)
                    .map_err(|e| {
                        extract_between_error_with_message(
                            format!("{BUILTIN_NAME}: {e}"),
                            &EXTRACT_BETWEEN_ERROR_INTERNAL,
                        )
                    })
            }
            TextKind::CellArray(info) => {
                if results.len() != info.element_kinds.len() {
                    return Err(extract_between_error(&EXTRACT_BETWEEN_ERROR_SIZE_MISMATCH));
                }
                let mut values = Vec::with_capacity(results.len());
                for (idx, result) in results.into_iter().enumerate() {
                    match info.element_kinds[idx] {
                        CellElementKind::String => values.push(Value::String(result.text)),
                        CellElementKind::Char => {
                            values.push(match result.text.as_str() {
                                Some(text) => Value::CharArray(CharArray::new_row(text)),
                                None => Value::String(StringElement::Missing),
                            });
                        }
                    }
                }
                make_cell_with_shape(values, info.shape).map_err(|e| {
                    extract_between_error_with_message(
                        format!("{BUILTIN_NAME}: {e}"),
                        &EXTRACT_BETWEEN_ERROR_INTERNAL,
                    )
                })
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum BoundaryKind {
    Text,
    Position,
}

#[derive(Clone, Debug)]
enum BoundaryArg {
    Text(BoundaryText),
    Position(BoundaryPositions),
}

impl BoundaryArg {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::String(_) | Value::StringArray(_) | Value::CharArray(_) | Value::Cell(_) => {
                BoundaryText::from_value(value).map(BoundaryArg::Text)
            }
            Value::Num(_) | Value::Int(_) | Value::Tensor(_) => {
                BoundaryPositions::from_value(value).map(BoundaryArg::Position)
            }
            other => Err(extract_between_error_with_message(
                format!(
                    "{}: unsupported argument {other:?}",
                    EXTRACT_BETWEEN_ERROR_BOUNDARY_TYPE.message
                ),
                &EXTRACT_BETWEEN_ERROR_BOUNDARY_TYPE,
            )),
        }
    }

    fn kind(&self) -> BoundaryKind {
        match self {
            BoundaryArg::Text(_) => BoundaryKind::Text,
            BoundaryArg::Position(_) => BoundaryKind::Position,
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            BoundaryArg::Text(text) => &text.shape,
            BoundaryArg::Position(pos) => &pos.shape,
        }
    }

    fn text(&self, idx: usize) -> &StringElement {
        match self {
            BoundaryArg::Text(text) => &text.data[idx],
            BoundaryArg::Position(_) => unreachable!(),
        }
    }

    fn position(&self, idx: usize) -> usize {
        match self {
            BoundaryArg::Position(pos) => pos.data[idx],
            BoundaryArg::Text(_) => unreachable!(),
        }
    }
}

#[derive(Clone, Debug)]
struct BoundaryText {
    data: Vec<StringElement>,
    shape: Vec<usize>,
}

impl BoundaryText {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::String(s) => Ok(Self {
                data: vec![s],
                shape: vec![1, 1],
            }),
            Value::StringArray(sa) => Ok(Self {
                data: sa.data().to_vec(),
                shape: sa.shape.to_vec(),
            }),
            Value::CharArray(ca) => {
                let mut data = Vec::with_capacity(ca.rows);
                for row in 0..ca.rows {
                    data.push(
                        char_row_to_string_slice(ca.data(), ca.cols, row)
                            .map(StringElement::text)
                            .map_err(|e| {
                                extract_between_error_with_message(
                                    format!("{BUILTIN_NAME}: {e}"),
                                    &EXTRACT_BETWEEN_ERROR_INTERNAL,
                                )
                            })?,
                    );
                }
                Ok(Self {
                    data,
                    shape: vec![ca.rows, 1],
                })
            }
            Value::Cell(cell) => {
                let shape = cell.shape.to_vec();
                let mut data = Vec::with_capacity(cell.data().len());
                for element in cell.data() {
                    let scalar = TextCellScalar::from_value(element).ok_or_else(|| {
                        extract_between_error(&EXTRACT_BETWEEN_ERROR_CELL_ELEMENT)
                    })?;
                    data.push(scalar.text().map_err(extract_between_utf16_error)?);
                }
                Ok(Self { data, shape })
            }
            _ => Err(extract_between_error(&EXTRACT_BETWEEN_ERROR_BOUNDARY_TYPE)),
        }
    }
}

#[derive(Clone, Debug)]
struct BoundaryPositions {
    data: Vec<usize>,
    shape: Vec<usize>,
}

impl BoundaryPositions {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::Num(n) => Ok(Self {
                data: vec![parse_position(n)?],
                shape: vec![1, 1],
            }),
            Value::Int(i) => Ok(Self {
                data: vec![parse_position_int(i)?],
                shape: vec![1, 1],
            }),
            Value::Tensor(t) => {
                // Position vectors retain their full tensor shape. The empty
                // rank uses the historical boundary-domain projection [0, 1]
                // rather than the general scalar matrix projection.
                let shape = if t.shape().is_empty() {
                    vec![0, 1]
                } else {
                    t.shape().to_vec()
                };
                let mut data = Vec::with_capacity(t.len());
                for idx in 0..t.len() {
                    let entry = t.numeric_value_at(idx).ok_or_else(|| {
                        extract_between_error(&EXTRACT_BETWEEN_ERROR_POSITION_TYPE)
                    })?;
                    data.push(match entry {
                        NumericScalar::F64(value) => parse_position(value)?,
                        NumericScalar::F32(value) => parse_position(f64::from(value))?,
                        value => parse_position_int(
                            value
                                .into_int_value()
                                .expect("non-floating numeric scalar is integer"),
                        )?,
                    });
                }
                Ok(Self { data, shape })
            }
            _ => Err(extract_between_error(&EXTRACT_BETWEEN_ERROR_BOUNDARY_TYPE)),
        }
    }
}

fn parse_position(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 1.0 {
        return Err(extract_between_error(&EXTRACT_BETWEEN_ERROR_POSITION_TYPE));
    }
    if (value.fract()).abs() > f64::EPSILON {
        return Err(extract_between_error(&EXTRACT_BETWEEN_ERROR_POSITION_TYPE));
    }
    if value > usize::MAX.saturating_sub(1) as f64 {
        return Err(extract_between_error(&EXTRACT_BETWEEN_ERROR_POSITION_TYPE));
    }
    let parsed = value as usize;
    if parsed as f64 != value || parsed == usize::MAX {
        return Err(extract_between_error(&EXTRACT_BETWEEN_ERROR_POSITION_TYPE));
    }
    Ok(parsed)
}

fn parse_position_int(value: IntValue) -> BuiltinResult<usize> {
    value
        .try_to_usize()
        .filter(|position| *position > 0)
        .ok_or_else(|| extract_between_error(&EXTRACT_BETWEEN_ERROR_POSITION_TYPE))
}

#[cfg(test)]
pub(crate) mod tests;
