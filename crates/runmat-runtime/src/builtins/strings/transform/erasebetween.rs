//! MATLAB-compatible `eraseBetween` builtin with GPU-aware semantics for RunMat.

use runmat_value::StringElement;
use std::cmp::min;

use regex::Regex;

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};
use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::strings::common::{
    cell_contains_string, char_row_to_string_slice, is_multirow_char_array,
};
use crate::builtins::strings::core::compat::pattern_regex;
use crate::{
    build_runtime_error, gather_if_needed_async, make_cell_with_shape, BuiltinResult, RuntimeError,
};
use runmat_builtins::{
    BuiltinErrorDescriptor, ERASE_BETWEEN_CHAR_MATRIX_EXTENSION, ERASE_BETWEEN_ERROR_BOUNDARY_TYPE,
    ERASE_BETWEEN_ERROR_CELL_ELEMENT, ERASE_BETWEEN_ERROR_INTERNAL,
    ERASE_BETWEEN_ERROR_INVALID_INPUT, ERASE_BETWEEN_ERROR_NAME_VALUE_PAIR,
    ERASE_BETWEEN_ERROR_OPTION_NAME, ERASE_BETWEEN_ERROR_OPTION_VALUE,
    ERASE_BETWEEN_ERROR_POSITION_TYPE, ERASE_BETWEEN_ERROR_SIZE_MISMATCH,
    ERASE_BETWEEN_FULL_BROADCAST_EXTENSION, ERASE_BETWEEN_RESIDENT_POSITION_EXTENSION,
    ERASE_BETWEEN_STRING_CELL_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, IntValue, NumericScalar, StringArray, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

#[runmat_macros::register_gpu_spec(
    builtin_path = "crate::builtins::strings::transform::erasebetween"
)]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "eraseBetween",
    op_kind: GpuOpKind::Custom("string-transform"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "The builtin owns resident arguments so compatibility gates run before provider access; RunMat mode may then gather resident numeric position controls, while text and outputs remain on host.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::strings::transform::erasebetween"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "eraseBetween",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Pure string manipulation builtin; excluded from fusion plans, with resident numeric position controls handled only by an explicit extension.",
};

const BUILTIN_NAME: &str = "eraseBetween";

fn erase_between_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    erase_between_error_with_message(error.message, error)
}

fn erase_between_error_with_message(
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
    name = "eraseBetween",
    binding_variant = "default",
    builtin_path = "crate::builtins::strings::transform::erasebetween"
)]
async fn erase_between_builtin(
    text: Value,
    start: Value,
    stop: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if is_numeric_or_resident(&text) || contains_nested_numeric_or_resident(&text) {
        return Err(erase_between_error(&ERASE_BETWEEN_ERROR_INVALID_INPUT));
    }
    if is_resident_or_contains_resident(&start) || is_resident_or_contains_resident(&stop) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ERASE_BETWEEN_RESIDENT_POSITION_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if is_multirow_char_array(&text)
        || is_multirow_char_array(&start)
        || is_multirow_char_array(&stop)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ERASE_BETWEEN_CHAR_MATRIX_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if cell_contains_string(&text) || cell_contains_string(&start) || cell_contains_string(&stop) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ERASE_BETWEEN_STRING_CELL_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if raw_boundary_uses_full_broadcast(&text, &start)
        || raw_boundary_uses_full_broadcast(&text, &stop)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &ERASE_BETWEEN_FULL_BROADCAST_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let text = gather_if_needed_async(&text).await.map_err(map_flow)?;
    let start = gather_if_needed_async(&start).await.map_err(map_flow)?;
    let stop = gather_if_needed_async(&stop).await.map_err(map_flow)?;

    let mode_override = parse_boundaries_option(rest.view()).await?;

    let normalized_text = NormalizedText::from_value(text)?;
    let start_boundary = BoundaryArg::from_value(start)?;
    let stop_boundary = BoundaryArg::from_value(stop)?;

    if start_boundary.kind() != stop_boundary.kind() {
        return Err(erase_between_error(&ERASE_BETWEEN_ERROR_BOUNDARY_TYPE));
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
            &ERASE_BETWEEN_FULL_BROADCAST_EXTENSION,
            BUILTIN_NAME,
        )?;
    }

    let shape_ts = broadcast_shapes(BUILTIN_NAME, text_shape, start_shape).map_err(|err| {
        erase_between_error_with_message(
            format!("{}: {err}", ERASE_BETWEEN_ERROR_SIZE_MISMATCH.message),
            &ERASE_BETWEEN_ERROR_SIZE_MISMATCH,
        )
    })?;
    let output_shape = broadcast_shapes(BUILTIN_NAME, &shape_ts, stop_shape).map_err(|err| {
        erase_between_error_with_message(
            format!("{}: {err}", ERASE_BETWEEN_ERROR_SIZE_MISMATCH.message),
            &ERASE_BETWEEN_ERROR_SIZE_MISMATCH,
        )
    })?;
    if !normalized_text.supports_shape(&output_shape) {
        return Err(erase_between_error(&ERASE_BETWEEN_ERROR_SIZE_MISMATCH));
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
                let start_value = start_boundary.marker(start_idx);
                let stop_value = stop_boundary.marker(stop_idx);
                erase_with_text_boundaries(text_value, start_value, stop_value, effective_mode)
            }
            BoundaryKind::Position => {
                let text_value = normalized_text.data(text_idx);
                let start_value = start_boundary.position(start_idx);
                let stop_value = stop_boundary.position(stop_idx);
                erase_with_positions(text_value, start_value, stop_value, effective_mode)
            }
        };
        results.push(result);
    }

    normalized_text.into_value(results, output_shape)
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
        Value::CharArray(array) => Some(vec![array.rows, 1]),
        Value::Cell(cell) => Some(cell.shape.to_vec()),
        _ => None,
    }
}

fn raw_boundary_shape(value: &Value) -> Option<Vec<usize>> {
    match value {
        Value::String(_) | Value::Num(_) | Value::Int(_) | Value::Object(_) => Some(vec![1, 1]),
        Value::StringArray(array) => Some(array.shape.to_vec()),
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
        return Err(erase_between_error(&ERASE_BETWEEN_ERROR_NAME_VALUE_PAIR));
    }

    let mut mode: Option<BoundariesMode> = None;
    let mut idx = 0;
    while idx < args.len() {
        let name_value = gather_if_needed_async(&args[idx]).await.map_err(map_flow)?;
        let name = value_to_string(&name_value)?
            .ok_or_else(|| erase_between_error(&ERASE_BETWEEN_ERROR_OPTION_NAME))?;
        if !name.eq_ignore_ascii_case("boundaries") {
            return Err(erase_between_error(&ERASE_BETWEEN_ERROR_OPTION_NAME));
        }
        let value = gather_if_needed_async(&args[idx + 1])
            .await
            .map_err(map_flow)?;
        let value_str = value_to_string(&value)?
            .ok_or_else(|| erase_between_error(&ERASE_BETWEEN_ERROR_OPTION_VALUE))?;
        let parsed_mode = if value_str.eq_ignore_ascii_case("inclusive") {
            BoundariesMode::Inclusive
        } else if value_str.eq_ignore_ascii_case("exclusive") {
            BoundariesMode::Exclusive
        } else {
            return Err(erase_between_error(&ERASE_BETWEEN_ERROR_OPTION_VALUE));
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
                        erase_between_error_with_message(
                            format!("{BUILTIN_NAME}: {e}"),
                            &ERASE_BETWEEN_ERROR_INTERNAL,
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
struct EraseResult {
    text: StringElement,
}

impl EraseResult {
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

fn erase_with_text_boundaries(
    text: &StringElement,
    start: &BoundaryMarker,
    stop: &BoundaryMarker,
    mode: BoundariesMode,
) -> EraseResult {
    if text.is_missing() || start.is_missing() || stop.is_missing() {
        return EraseResult::missing();
    }

    let text = text.as_str().expect("nonmissing subject");
    if let Some((start_idx, start_end)) = start.find(text) {
        let search_start = start_end;
        if search_start > text.len() {
            return EraseResult::text(text.to_string());
        }
        if let Some((relative_end, relative_capture_end)) = stop.find(&text[search_start..]) {
            let end_idx = search_start + relative_end;
            match mode {
                BoundariesMode::Inclusive => {
                    let end_capture = min(text.len(), search_start + relative_capture_end);
                    let mut result = String::with_capacity(text.len());
                    result.push_str(&text[..start_idx]);
                    result.push_str(&text[end_capture..]);
                    EraseResult::text(result)
                }
                BoundariesMode::Exclusive => {
                    let mut result = String::with_capacity(text.len());
                    result.push_str(&text[..start_end]);
                    result.push_str(&text[end_idx..]);
                    EraseResult::text(result)
                }
            }
        } else {
            EraseResult::text(text.to_string())
        }
    } else {
        EraseResult::text(text.to_string())
    }
}

fn erase_with_positions(
    text: &StringElement,
    start: usize,
    stop: usize,
    mode: BoundariesMode,
) -> EraseResult {
    let Some(text) = text.as_str() else {
        return EraseResult::missing();
    };
    if text.is_empty() {
        return EraseResult::text(String::new());
    }
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    if len == 0 {
        return EraseResult::text(String::new());
    }

    if start == 0 || stop == 0 {
        return EraseResult::text(text.to_string());
    }

    if start > len {
        return EraseResult::text(text.to_string());
    }
    let stop_clamped = stop.min(len);

    match mode {
        BoundariesMode::Inclusive => {
            if stop_clamped < start {
                return EraseResult::text(text.to_string());
            }
            let start_idx = start - 1;
            let end_idx = stop_clamped - 1;
            if start_idx >= len || end_idx >= len || start_idx > end_idx {
                EraseResult::text(text.to_string())
            } else {
                let mut result = String::with_capacity(len);
                for (idx, ch) in chars.iter().enumerate() {
                    if idx < start_idx || idx > end_idx {
                        result.push(*ch);
                    }
                }
                EraseResult::text(result)
            }
        }
        BoundariesMode::Exclusive => {
            if start + 1 >= stop_clamped {
                return EraseResult::text(text.to_string());
            }
            let start_idx = start;
            let end_idx = stop_clamped - 2;
            if start_idx >= len || end_idx >= len || start_idx > end_idx {
                EraseResult::text(text.to_string())
            } else {
                let mut result = String::with_capacity(len);
                for (idx, ch) in chars.iter().enumerate() {
                    if idx >= start_idx && idx <= end_idx {
                        continue;
                    }
                    result.push(*ch);
                }
                EraseResult::text(result)
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
enum CellElementKind {
    String,
    Char,
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
                let rows = ca.rows;
                let mut data = Vec::with_capacity(rows);
                for row in 0..rows {
                    data.push(
                        char_row_to_string_slice(ca.data(), ca.cols, row)
                            .map(StringElement::text)
                            .map_err(|e| {
                                erase_between_error_with_message(
                                    format!("{BUILTIN_NAME}: {e}"),
                                    &ERASE_BETWEEN_ERROR_INTERNAL,
                                )
                            })?,
                    );
                }
                Ok(Self {
                    data,
                    shape: vec![rows, 1],
                    kind: TextKind::CharArray { rows },
                })
            }
            Value::Cell(cell) => {
                let shape = cell.shape.to_vec();
                let mut data = Vec::with_capacity(cell.data().len());
                let mut kinds = Vec::with_capacity(cell.data().len());
                for element in cell.data() {
                    match &element {
                        Value::String(s) => {
                            data.push(s.clone());
                            kinds.push(CellElementKind::String);
                        }
                        Value::StringArray(sa) if sa.data().len() == 1 => {
                            data.push(sa.data()[0].clone());
                            kinds.push(CellElementKind::String);
                        }
                        Value::CharArray(ca) if ca.rows <= 1 => {
                            if ca.rows == 0 {
                                data.push(StringElement::text(""));
                            } else {
                                data.push(
                                    char_row_to_string_slice(ca.data(), ca.cols, 0)
                                        .map(StringElement::text)
                                        .map_err(|e| {
                                            erase_between_error_with_message(
                                                format!("{BUILTIN_NAME}: {e}"),
                                                &ERASE_BETWEEN_ERROR_INTERNAL,
                                            )
                                        })?,
                                );
                            }
                            kinds.push(CellElementKind::Char);
                        }
                        Value::CharArray(_) => {
                            return Err(erase_between_error(&ERASE_BETWEEN_ERROR_CELL_ELEMENT))
                        }
                        _ => return Err(erase_between_error(&ERASE_BETWEEN_ERROR_CELL_ELEMENT)),
                    }
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
            _ => Err(erase_between_error(&ERASE_BETWEEN_ERROR_INVALID_INPUT)),
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
        results: Vec<EraseResult>,
        output_shape: Vec<usize>,
    ) -> BuiltinResult<Value> {
        match self.kind {
            TextKind::StringScalar => {
                let total: usize = output_shape.iter().product();
                if total == 0 {
                    let data = results.into_iter().map(|r| r.text).collect::<Vec<_>>();
                    let array =
                        StringArray::from_elements_unbounded(data, output_shape).map_err(|e| {
                            erase_between_error_with_message(
                                format!("{BUILTIN_NAME}: {e}"),
                                &ERASE_BETWEEN_ERROR_INTERNAL,
                            )
                        })?;
                    return Ok(Value::StringArray(array));
                }

                if results.len() <= 1 {
                    let value = results
                        .into_iter()
                        .next()
                        .unwrap_or_else(|| EraseResult::text(String::new()));
                    Ok(Value::String(value.text))
                } else {
                    let data = results.into_iter().map(|r| r.text).collect::<Vec<_>>();
                    let array =
                        StringArray::from_elements_unbounded(data, output_shape).map_err(|e| {
                            erase_between_error_with_message(
                                format!("{BUILTIN_NAME}: {e}"),
                                &ERASE_BETWEEN_ERROR_INTERNAL,
                            )
                        })?;
                    Ok(Value::StringArray(array))
                }
            }
            TextKind::StringArray => {
                let data = results.into_iter().map(|r| r.text).collect::<Vec<_>>();
                let array =
                    StringArray::from_elements_unbounded(data, output_shape).map_err(|e| {
                        erase_between_error_with_message(
                            format!("{BUILTIN_NAME}: {e}"),
                            &ERASE_BETWEEN_ERROR_INTERNAL,
                        )
                    })?;
                Ok(Value::StringArray(array))
            }
            TextKind::CharArray { rows } => {
                if rows == 0 {
                    return CharArray::from_code_units(Vec::new(), 0, 0)
                        .map(Value::CharArray)
                        .map_err(|e| {
                            erase_between_error_with_message(
                                format!("{BUILTIN_NAME}: {e}"),
                                &ERASE_BETWEEN_ERROR_INTERNAL,
                            )
                        });
                }
                if results.len() != rows {
                    return Err(erase_between_error(&ERASE_BETWEEN_ERROR_SIZE_MISMATCH));
                }
                let mut max_width = 0usize;
                let mut row_strings = Vec::with_capacity(rows);
                for result in &results {
                    let width = result
                        .text
                        .as_str()
                        .ok_or_else(|| erase_between_error(&ERASE_BETWEEN_ERROR_BOUNDARY_TYPE))?
                        .encode_utf16()
                        .count();
                    max_width = max_width.max(width);
                    row_strings.push(result.text.clone());
                }
                let mut flattened = Vec::with_capacity(rows * max_width);
                for row in row_strings {
                    let mut units = row
                        .as_str()
                        .ok_or_else(|| erase_between_error(&ERASE_BETWEEN_ERROR_BOUNDARY_TYPE))?
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
                        erase_between_error_with_message(
                            format!("{BUILTIN_NAME}: {e}"),
                            &ERASE_BETWEEN_ERROR_INTERNAL,
                        )
                    })
            }
            TextKind::CellArray(info) => {
                if results.len() != info.element_kinds.len() {
                    return Err(erase_between_error(&ERASE_BETWEEN_ERROR_SIZE_MISMATCH));
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
                    erase_between_error_with_message(
                        format!("{BUILTIN_NAME}: {e}"),
                        &ERASE_BETWEEN_ERROR_INTERNAL,
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
            Value::String(_)
            | Value::StringArray(_)
            | Value::CharArray(_)
            | Value::Cell(_)
            | Value::Object(_) => BoundaryText::from_value(value).map(BoundaryArg::Text),
            Value::Num(_) | Value::Int(_) | Value::Tensor(_) => {
                BoundaryPositions::from_value(value).map(BoundaryArg::Position)
            }
            other => Err(erase_between_error_with_message(
                format!(
                    "{}: unsupported argument {other:?}",
                    ERASE_BETWEEN_ERROR_BOUNDARY_TYPE.message
                ),
                &ERASE_BETWEEN_ERROR_BOUNDARY_TYPE,
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

    fn marker(&self, idx: usize) -> &BoundaryMarker {
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
    data: Vec<BoundaryMarker>,
    shape: Vec<usize>,
}

#[derive(Clone, Debug)]
enum BoundaryMarker {
    Literal(StringElement),
    Pattern(Regex),
}

impl BoundaryMarker {
    fn is_missing(&self) -> bool {
        matches!(self, Self::Literal(text) if text.is_missing())
    }

    fn find(&self, text: &str) -> Option<(usize, usize)> {
        match self {
            Self::Literal(marker) => marker.as_str().and_then(|marker| {
                text.find(marker)
                    .map(|start| (start, start.saturating_add(marker.len())))
            }),
            Self::Pattern(pattern) => pattern
                .find(text)
                .map(|matched| (matched.start(), matched.end())),
        }
    }
}

impl BoundaryText {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::String(s) => Ok(Self {
                data: vec![BoundaryMarker::Literal(s.clone())],
                shape: vec![1, 1],
            }),
            Value::StringArray(sa) => Ok(Self {
                data: sa
                    .data()
                    .iter()
                    .cloned()
                    .map(|text| BoundaryMarker::Literal(text))
                    .collect(),
                shape: sa.shape.to_vec(),
            }),
            Value::CharArray(ca) => {
                let mut data = Vec::with_capacity(ca.rows);
                for row in 0..ca.rows {
                    data.push(BoundaryMarker::Literal(
                        char_row_to_string_slice(ca.data(), ca.cols, row)
                            .map(StringElement::text)
                            .map_err(|e| {
                                erase_between_error_with_message(
                                    format!("{BUILTIN_NAME}: {e}"),
                                    &ERASE_BETWEEN_ERROR_INTERNAL,
                                )
                            })?,
                    ));
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
                    match &element {
                        Value::String(s) => data.push(BoundaryMarker::Literal(s.clone())),
                        Value::StringArray(sa) if sa.data().len() == 1 => {
                            data.push(BoundaryMarker::Literal(sa.data()[0].clone()));
                        }
                        Value::CharArray(ca) if ca.rows <= 1 => {
                            if ca.rows == 0 {
                                data.push(BoundaryMarker::Literal(StringElement::text("")));
                            } else {
                                data.push(BoundaryMarker::Literal(
                                    char_row_to_string_slice(ca.data(), ca.cols, 0)
                                        .map(StringElement::text)
                                        .map_err(|e| {
                                            erase_between_error_with_message(
                                                format!("{BUILTIN_NAME}: {e}"),
                                                &ERASE_BETWEEN_ERROR_INTERNAL,
                                            )
                                        })?,
                                ));
                            }
                        }
                        Value::CharArray(_) => {
                            return Err(erase_between_error(&ERASE_BETWEEN_ERROR_CELL_ELEMENT))
                        }
                        _ => return Err(erase_between_error(&ERASE_BETWEEN_ERROR_CELL_ELEMENT)),
                    }
                }
                Ok(Self { data, shape })
            }
            object @ Value::Object(_) => {
                let regex = pattern_regex(&object, BUILTIN_NAME).map_err(|error| {
                    erase_between_error_with_message(
                        error.message().to_string(),
                        &ERASE_BETWEEN_ERROR_BOUNDARY_TYPE,
                    )
                })?;
                let pattern = Regex::new(&regex).map_err(|error| {
                    erase_between_error_with_message(
                        format!("{}: {error}", ERASE_BETWEEN_ERROR_BOUNDARY_TYPE.message),
                        &ERASE_BETWEEN_ERROR_BOUNDARY_TYPE,
                    )
                })?;
                Ok(Self {
                    data: vec![BoundaryMarker::Pattern(pattern)],
                    shape: vec![1, 1],
                })
            }
            _ => Err(erase_between_error(&ERASE_BETWEEN_ERROR_BOUNDARY_TYPE)),
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
                let mut data = Vec::with_capacity(t.len());
                for idx in 0..t.len() {
                    let entry = t
                        .numeric_value_at(idx)
                        .ok_or_else(|| erase_between_error(&ERASE_BETWEEN_ERROR_POSITION_TYPE))?;
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
                let shape = if t.shape().is_empty() {
                    vec![0, 1]
                } else {
                    t.shape().to_vec()
                };
                Ok(Self { data, shape })
            }
            _ => Err(erase_between_error(&ERASE_BETWEEN_ERROR_BOUNDARY_TYPE)),
        }
    }
}

fn parse_position(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 1.0 {
        return Err(erase_between_error(&ERASE_BETWEEN_ERROR_POSITION_TYPE));
    }
    if (value.fract()).abs() > f64::EPSILON {
        return Err(erase_between_error(&ERASE_BETWEEN_ERROR_POSITION_TYPE));
    }
    if value > usize::MAX.saturating_sub(1) as f64 {
        return Err(erase_between_error(&ERASE_BETWEEN_ERROR_POSITION_TYPE));
    }
    let parsed = value as usize;
    if parsed as f64 != value || parsed == usize::MAX {
        return Err(erase_between_error(&ERASE_BETWEEN_ERROR_POSITION_TYPE));
    }
    Ok(parsed)
}

fn parse_position_int(value: IntValue) -> BuiltinResult<usize> {
    value
        .try_to_usize()
        .filter(|position| *position > 0)
        .ok_or_else(|| erase_between_error(&ERASE_BETWEEN_ERROR_POSITION_TYPE))
}

#[cfg(test)]
pub(crate) mod tests {
    #![allow(non_snake_case)]

    use super::*;
    use runmat_value::{CellArray, CharArray, IntegerStorage, NumericStorage, StringArray, Tensor};

    fn erase_between_builtin(
        text: Value,
        start: Value,
        stop: Value,
        rest: Vec<Value>,
    ) -> BuiltinResult<Value> {
        futures::executor::block_on(super::erase_between_builtin(
            text,
            start,
            stop,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn eraseBetween_rank_zero_position_retains_boundary_domain_shape() {
        let positions = || Tensor::new(vec![2.0], Vec::new()).unwrap();
        let parsed = BoundaryPositions::from_value(Value::Tensor(positions())).unwrap();
        assert_eq!(parsed.data, vec![2]);
        assert_eq!(parsed.shape, vec![0, 1]);
        let output = erase_between_builtin(
            Value::StringArray(StringArray::new(Vec::new(), vec![0, 1]).unwrap()),
            Value::Tensor(positions()),
            Value::Tensor(positions()),
            Vec::new(),
        )
        .unwrap();
        let Value::StringArray(output) = output else {
            panic!("expected string array");
        };
        assert_eq!(output.shape, vec![0, 1]);
    }

    #[test]
    fn eraseBetween_position_vectors_read_typed_integer_storage_exactly() {
        let positions =
            Tensor::new_integer(IntegerStorage::U16(vec![2, 4]), vec![1, 2]).expect("positions");

        let parsed = BoundaryPositions::from_value(Value::Tensor(positions)).unwrap();
        assert_eq!(parsed.data, vec![2, 4]);
        assert_eq!(parsed.shape, vec![1, 2]);
    }

    #[test]
    fn eraseBetween_position_vectors_read_native_single_storage() {
        let positions =
            Tensor::from_numeric_storage(NumericStorage::F32(vec![2.0, 4.0]), vec![1, 2])
                .expect("positions");
        let parsed = BoundaryPositions::from_value(Value::Tensor(positions)).unwrap();
        assert_eq!(parsed.data, vec![2, 4]);
    }

    #[test]
    fn eraseBetween_position_vectors_reject_nonpositive_integer_storage() {
        let positions =
            Tensor::new_integer(IntegerStorage::I16(vec![1, 0]), vec![1, 2]).expect("positions");

        assert!(BoundaryPositions::from_value(Value::Tensor(positions)).is_err());
    }

    #[test]
    fn eraseBetween_position_vectors_reject_oversized_double_positions() {
        assert!(BoundaryPositions::from_value(Value::Num(1.0e300)).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eraseBetween_text_default_exclusive() {
        let result = erase_between_builtin(
            Value::String("The quick brown fox".into()),
            Value::String("quick".into()),
            Value::String(" fox".into()),
            Vec::new(),
        )
        .expect("eraseBetween");
        assert_eq!(result, Value::String("The quick fox".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eraseBetween_text_inclusive_option() {
        let result = erase_between_builtin(
            Value::String("The quick brown fox jumps over the lazy dog".into()),
            Value::String(" brown".into()),
            Value::String("lazy".into()),
            vec![
                Value::String("Boundaries".into()),
                Value::String("inclusive".into()),
            ],
        )
        .expect("eraseBetween");
        assert_eq!(result, Value::String("The quick dog".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eraseBetween_numeric_positions_default_inclusive() {
        let result = erase_between_builtin(
            Value::String("Edgar Allen Poe".into()),
            Value::Num(6.0),
            Value::Num(11.0),
            Vec::new(),
        )
        .expect("eraseBetween");
        assert_eq!(result, Value::String("Edgar Poe".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eraseBetween_numeric_positions_int_inputs() {
        let result = erase_between_builtin(
            Value::String("abcdef".into()),
            Value::Int(IntValue::I32(2)),
            Value::Int(IntValue::I32(5)),
            Vec::new(),
        )
        .expect("eraseBetween");
        assert_eq!(result, Value::String("af".into()));
    }

    #[test]
    fn eraseBetween_full_broadcast_is_mode_gated() {
        let text = StringArray::new(vec!["abcd".into(), "efgh".into()], vec![2, 1]).unwrap();
        let start = Tensor::new_integer(IntegerStorage::U8(vec![1, 2]), vec![1, 2]).unwrap();
        let stop = Tensor::new_integer(IntegerStorage::U8(vec![2, 3]), vec![1, 2]).unwrap();
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = erase_between_builtin(
            Value::StringArray(text),
            Value::Tensor(start),
            Value::Tensor(stop),
            Vec::new(),
        )
        .expect_err("general broadcasting is an extension");
        assert_eq!(
            error.identifier(),
            ERASE_BETWEEN_FULL_BROADCAST_EXTENSION.error_identifier
        );
    }

    #[test]
    fn eraseBetween_resident_positions_are_gated_before_gather() {
        use crate::builtins::common::{gpu_helpers, test_support};

        test_support::with_test_provider(|provider| {
            let positions = Tensor::new_integer(IntegerStorage::U64(vec![2]), vec![1, 1]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &positions).expect("upload");
            provider.reset_telemetry();
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = erase_between_builtin(
                Value::String("abcd".into()),
                Value::GpuTensor(handle.clone()),
                Value::Num(3.0),
                Vec::new(),
            )
            .expect_err("resident positions are an extension");
            assert_eq!(
                error.identifier(),
                ERASE_BETWEEN_RESIDENT_POSITION_EXTENSION.error_identifier
            );
            assert_eq!(provider.telemetry_snapshot().download_bytes, 0);
            handle.release().expect("release fixture handle");
        });
    }

    #[test]
    fn eraseBetween_dispatch_preserves_residency_until_builtin_validation() {
        assert_eq!(GPU_SPEC.residency, ResidencyPolicy::NewHandle);
    }

    #[test]
    fn eraseBetween_accepts_scalar_pattern_boundaries() {
        let start = crate::builtins::strings::core::compat::pattern_object(r"<[^>]*>");
        let stop = crate::builtins::strings::core::compat::pattern_object(r"</[^>]*>");
        let result = erase_between_builtin(
            Value::String("<course>algebra</course>".into()),
            start,
            stop,
            Vec::new(),
        )
        .expect("pattern boundaries");
        assert_eq!(result, Value::String("<course></course>".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eraseBetween_numeric_positions_exclusive_option() {
        let result = erase_between_builtin(
            Value::String("small|medium|large".into()),
            Value::Num(6.0),
            Value::Num(13.0),
            vec![
                Value::String("Boundaries".into()),
                Value::String("exclusive".into()),
            ],
        )
        .expect("eraseBetween");
        assert_eq!(result, Value::String("small||large".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eraseBetween_start_not_found_returns_original() {
        let result = erase_between_builtin(
            Value::String("RunMat Accelerate".into()),
            Value::String("<".into()),
            Value::String(">".into()),
            Vec::new(),
        )
        .expect("eraseBetween");
        assert_eq!(result, Value::String("RunMat Accelerate".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eraseBetween_stop_not_found_returns_original() {
        let result = erase_between_builtin(
            Value::String("Device<GPU>".into()),
            Value::String("<".into()),
            Value::String(")".into()),
            Vec::new(),
        )
        .expect("eraseBetween");
        assert_eq!(result, Value::String("Device<GPU>".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eraseBetween_missing_string_propagates() {
        let strings = StringArray::from_elements_unbounded(
            vec![runmat_value::StringElement::Missing],
            vec![1, 1],
        )
        .unwrap();
        let result = erase_between_builtin(
            Value::StringArray(strings),
            Value::String("<".into()),
            Value::String(">".into()),
            Vec::new(),
        )
        .expect("eraseBetween");
        assert_eq!(
            result,
            Value::StringArray(
                StringArray::from_elements_unbounded(
                    vec![runmat_value::StringElement::Missing],
                    vec![1, 1]
                )
                .unwrap()
            )
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eraseBetween_zero_sized_broadcast_produces_empty_array() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let start = StringArray::new(Vec::new(), vec![0, 1]).unwrap();
        let stop = StringArray::new(Vec::new(), vec![0, 1]).unwrap();
        let result = erase_between_builtin(
            Value::String("abc".into()),
            Value::StringArray(start),
            Value::StringArray(stop),
            Vec::new(),
        )
        .expect("eraseBetween");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data().len(), 0);
                assert_eq!(sa.shape, vec![0, 1]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eraseBetween_numeric_positions_array() {
        let text = StringArray::new(vec!["abcd".into(), "wxyz".into()], vec![2, 1]).unwrap();
        let start = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let stop = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
        let result = erase_between_builtin(
            Value::StringArray(text),
            Value::Tensor(start),
            Value::Tensor(stop),
            Vec::new(),
        )
        .expect("eraseBetween");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data(), vec!["d".to_string(), "w".to_string()]);
                assert_eq!(sa.shape, vec![2, 1]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eraseBetween_cell_array_preserves_types() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("A[B]C")),
                Value::String("Planner<GPU>".into()),
            ],
            1,
            2,
        )
        .unwrap();
        let start = CellArray::new(
            vec![Value::String("[".into()), Value::String("<".into())],
            1,
            2,
        )
        .unwrap();
        let stop = CellArray::new(
            vec![Value::String("]".into()), Value::String(">".into())],
            1,
            2,
        )
        .unwrap();
        let result = erase_between_builtin(
            Value::Cell(cell),
            Value::Cell(start),
            Value::Cell(stop),
            vec![
                Value::String("Boundaries".into()),
                Value::String("inclusive".into()),
            ],
        )
        .expect("eraseBetween");
        match result {
            Value::Cell(out) => {
                let first = out.get(0, 0).unwrap();
                let second = out.get(0, 1).unwrap();
                assert_eq!(first, Value::CharArray(CharArray::new_row("AC")));
                assert_eq!(second, Value::String("Planner".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eraseBetween_char_array_default_and_inclusive() {
        let chars = CharArray::from_unicode_scalars(
            "Device<GPU>".chars().collect(),
            1,
            "Device<GPU>".len(),
        )
        .unwrap();
        let default = erase_between_builtin(
            Value::CharArray(chars.clone()),
            Value::String("<".into()),
            Value::String(">".into()),
            Vec::new(),
        )
        .expect("eraseBetween");
        match default {
            Value::CharArray(out) => {
                let text = runmat_value::decode_utf16_code_units(out.data()).unwrap();
                assert_eq!(text.trim_end(), "Device<>");
            }
            other => panic!("expected char array, got {other:?}"),
        }

        let inclusive = erase_between_builtin(
            Value::CharArray(chars),
            Value::String("<".into()),
            Value::String(">".into()),
            vec![
                Value::String("Boundaries".into()),
                Value::String("inclusive".into()),
            ],
        )
        .expect("eraseBetween");
        match inclusive {
            Value::CharArray(out) => {
                let text = runmat_value::decode_utf16_code_units(out.data()).unwrap();
                assert_eq!(text.trim_end(), "Device");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eraseBetween_option_with_char_arrays_case_insensitive() {
        let result = erase_between_builtin(
            Value::String("A<mid>B".into()),
            Value::String("<".into()),
            Value::String(">".into()),
            vec![
                Value::CharArray(CharArray::new_row("Boundaries")),
                Value::CharArray(CharArray::new_row("INCLUSIVE")),
            ],
        )
        .expect("eraseBetween");
        assert_eq!(result, Value::String("AB".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eraseBetween_text_scalar_broadcast() {
        let text =
            StringArray::new(vec!["alpha[GPU]".into(), "beta[GPU]".into()], vec![2, 1]).unwrap();
        let result = erase_between_builtin(
            Value::StringArray(text),
            Value::String("[".into()),
            Value::String("]".into()),
            Vec::new(),
        )
        .expect("eraseBetween");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data(), vec!["alpha[]".to_string(), "beta[]".to_string()]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eraseBetween_option_invalid_value() {
        let err = erase_between_builtin(
            Value::String("abc".into()),
            Value::String("a".into()),
            Value::String("c".into()),
            vec![
                Value::String("Boundaries".into()),
                Value::String("middle".into()),
            ],
        )
        .unwrap_err();
        assert_eq!(err.to_string(), ERASE_BETWEEN_ERROR_OPTION_VALUE.message);
        assert_eq!(
            err.identifier(),
            ERASE_BETWEEN_ERROR_OPTION_VALUE.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eraseBetween_option_name_error() {
        let err = erase_between_builtin(
            Value::String("abc".into()),
            Value::String("a".into()),
            Value::String("c".into()),
            vec![
                Value::String("Padding".into()),
                Value::String("inclusive".into()),
            ],
        )
        .unwrap_err();
        assert_eq!(err.to_string(), ERASE_BETWEEN_ERROR_OPTION_NAME.message);
        assert_eq!(err.identifier(), ERASE_BETWEEN_ERROR_OPTION_NAME.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eraseBetween_option_pair_error() {
        let err = erase_between_builtin(
            Value::String("abc".into()),
            Value::String("a".into()),
            Value::String("b".into()),
            vec![Value::String("Boundaries".into())],
        )
        .unwrap_err();
        assert_eq!(err.to_string(), ERASE_BETWEEN_ERROR_NAME_VALUE_PAIR.message);
        assert_eq!(
            err.identifier(),
            ERASE_BETWEEN_ERROR_NAME_VALUE_PAIR.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eraseBetween_position_type_error() {
        let err = erase_between_builtin(
            Value::String("abc".into()),
            Value::Num(0.5),
            Value::Num(2.0),
            Vec::new(),
        )
        .unwrap_err();
        assert_eq!(err.to_string(), ERASE_BETWEEN_ERROR_POSITION_TYPE.message);
        assert_eq!(
            err.identifier(),
            ERASE_BETWEEN_ERROR_POSITION_TYPE.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eraseBetween_mixed_boundary_error() {
        let err = erase_between_builtin(
            Value::String("abc".into()),
            Value::String("a".into()),
            Value::Num(3.0),
            Vec::new(),
        )
        .unwrap_err();
        assert_eq!(err.to_string(), ERASE_BETWEEN_ERROR_BOUNDARY_TYPE.message);
        assert_eq!(
            err.identifier(),
            ERASE_BETWEEN_ERROR_BOUNDARY_TYPE.identifier
        );
    }
}
