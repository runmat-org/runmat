//! MATLAB-compatible `eraseBetween` builtin with GPU-aware semantics for RunMat.

use std::cmp::min;

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};
use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::strings::common::{char_row_to_string_slice, is_missing_string};
use crate::{
    build_runtime_error, gather_if_needed_async, make_cell_with_shape, BuiltinResult, RuntimeError,
};
use runmat_builtins::{CharArray, IntValue, StringArray, Value};
use runmat_macros::runtime_builtin;

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
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Runs on the CPU; GPU-resident inputs are gathered before deletion and outputs remain on the host.",
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
    notes: "Pure string manipulation builtin; excluded from fusion plans and gathers GPU inputs immediately.",
};

const FN_NAME: &str = "eraseBetween";
const ARG_TYPE_ERROR: &str = "eraseBetween: first argument must be a string array, character array, or cell array of character vectors";
const BOUNDARY_TYPE_ERROR: &str =
    "eraseBetween: start and end arguments must both be text or both be numeric positions";
const POSITION_TYPE_ERROR: &str = "eraseBetween: position arguments must be positive integers";
const OPTION_PAIR_ERROR: &str = "eraseBetween: name-value arguments must appear in pairs";
const OPTION_NAME_ERROR: &str = "eraseBetween: unrecognized parameter name";
const OPTION_VALUE_ERROR: &str =
    "eraseBetween: 'Boundaries' must be either 'inclusive' or 'exclusive'";
const CELL_ELEMENT_ERROR: &str =
    "eraseBetween: cell array elements must be string scalars or character vectors";
const SIZE_MISMATCH_ERROR: &str =
    "eraseBetween: boundary sizes must be compatible with the text input";

fn runtime_error_for(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(FN_NAME).build()
}

fn map_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, FN_NAME)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BoundariesMode {
    Exclusive,
    Inclusive,
}

#[runtime_builtin(
    name = "eraseBetween",
    category = "strings/transform",
    summary = "Delete text between boundary markers with MATLAB-compatible semantics.",
    keywords = "eraseBetween,delete,boundaries,strings",
    accel = "sink",
    builtin_path = "crate::builtins::strings::transform::erasebetween"
)]
async fn erase_between_builtin(
    text: Value,
    start: Value,
    stop: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let text = gather_if_needed_async(&text).await.map_err(map_flow)?;
    let start = gather_if_needed_async(&start).await.map_err(map_flow)?;
    let stop = gather_if_needed_async(&stop).await.map_err(map_flow)?;

    let mode_override = parse_boundaries_option(&rest).await?;

    let normalized_text = NormalizedText::from_value(text)?;
    let start_boundary = BoundaryArg::from_value(start)?;
    let stop_boundary = BoundaryArg::from_value(stop)?;

    if start_boundary.kind() != stop_boundary.kind() {
        return Err(runtime_error_for(BOUNDARY_TYPE_ERROR));
    }
    let boundary_kind = start_boundary.kind();
    let effective_mode = mode_override.unwrap_or(match boundary_kind {
        BoundaryKind::Text => BoundariesMode::Exclusive,
        BoundaryKind::Position => BoundariesMode::Inclusive,
    });

    let start_shape = start_boundary.shape();
    let stop_shape = stop_boundary.shape();
    let text_shape = normalized_text.shape();

    let shape_ts = broadcast_shapes(FN_NAME, text_shape, start_shape).map_err(runtime_error_for)?;
    let output_shape =
        broadcast_shapes(FN_NAME, &shape_ts, stop_shape).map_err(runtime_error_for)?;
    if !normalized_text.supports_shape(&output_shape) {
        return Err(runtime_error_for(SIZE_MISMATCH_ERROR));
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

async fn parse_boundaries_option(args: &[Value]) -> BuiltinResult<Option<BoundariesMode>> {
    if args.is_empty() {
        return Ok(None);
    }
    if !args.len().is_multiple_of(2) {
        return Err(runtime_error_for(OPTION_PAIR_ERROR));
    }

    let mut mode: Option<BoundariesMode> = None;
    let mut idx = 0;
    while idx < args.len() {
        let name_value = gather_if_needed_async(&args[idx]).await.map_err(map_flow)?;
        let name =
            value_to_string(&name_value).ok_or_else(|| runtime_error_for(OPTION_NAME_ERROR))?;
        if !name.eq_ignore_ascii_case("boundaries") {
            return Err(runtime_error_for(OPTION_NAME_ERROR));
        }
        let value = gather_if_needed_async(&args[idx + 1])
            .await
            .map_err(map_flow)?;
        let value_str =
            value_to_string(&value).ok_or_else(|| runtime_error_for(OPTION_VALUE_ERROR))?;
        let parsed_mode = if value_str.eq_ignore_ascii_case("inclusive") {
            BoundariesMode::Inclusive
        } else if value_str.eq_ignore_ascii_case("exclusive") {
            BoundariesMode::Exclusive
        } else {
            return Err(runtime_error_for(OPTION_VALUE_ERROR));
        };
        mode = Some(parsed_mode);
        idx += 2;
    }
    Ok(mode)
}

fn value_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].clone()),
        Value::CharArray(ca) if ca.rows <= 1 => {
            if ca.rows == 0 {
                Some(String::new())
            } else {
                Some(char_row_to_string_slice(&ca.data, ca.cols, 0))
            }
        }
        Value::CharArray(_) => None,
        Value::Cell(cell) if cell.data.len() == 1 => {
            let element = &cell.data[0];
            value_to_string(element)
        }
        _ => None,
    }
}

#[derive(Clone)]
struct EraseResult {
    text: String,
}

impl EraseResult {
    fn missing() -> Self {
        Self {
            text: "<missing>".to_string(),
        }
    }

    fn text(text: String) -> Self {
        Self { text }
    }
}

fn erase_with_text_boundaries(
    text: &str,
    start: &str,
    stop: &str,
    mode: BoundariesMode,
) -> EraseResult {
    if is_missing_string(text) || is_missing_string(start) || is_missing_string(stop) {
        return EraseResult::missing();
    }

    if let Some(start_idx) = text.find(start) {
        let search_start = start_idx + start.len();
        if search_start > text.len() {
            return EraseResult::text(text.to_string());
        }
        if let Some(relative_end) = text[search_start..].find(stop) {
            let end_idx = search_start + relative_end;
            match mode {
                BoundariesMode::Inclusive => {
                    let end_capture = min(text.len(), end_idx + stop.len());
                    let mut result = String::with_capacity(text.len());
                    result.push_str(&text[..start_idx]);
                    result.push_str(&text[end_capture..]);
                    EraseResult::text(result)
                }
                BoundariesMode::Exclusive => {
                    let mut result = String::with_capacity(text.len());
                    result.push_str(&text[..search_start]);
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
    text: &str,
    start: usize,
    stop: usize,
    mode: BoundariesMode,
) -> EraseResult {
    if is_missing_string(text) {
        return EraseResult::missing();
    }
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
    data: Vec<String>,
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
                data: sa.data.clone(),
                shape: sa.shape.clone(),
                kind: TextKind::StringArray,
            }),
            Value::CharArray(ca) => {
                let rows = ca.rows;
                let mut data = Vec::with_capacity(rows);
                for row in 0..rows {
                    data.push(char_row_to_string_slice(&ca.data, ca.cols, row));
                }
                Ok(Self {
                    data,
                    shape: vec![rows, 1],
                    kind: TextKind::CharArray { rows },
                })
            }
            Value::Cell(cell) => {
                let shape = cell.shape.clone();
                let mut data = Vec::with_capacity(cell.data.len());
                let mut kinds = Vec::with_capacity(cell.data.len());
                for element in &cell.data {
                    match &**element {
                        Value::String(s) => {
                            data.push(s.clone());
                            kinds.push(CellElementKind::String);
                        }
                        Value::StringArray(sa) if sa.data.len() == 1 => {
                            data.push(sa.data[0].clone());
                            kinds.push(CellElementKind::String);
                        }
                        Value::CharArray(ca) if ca.rows <= 1 => {
                            if ca.rows == 0 {
                                data.push(String::new());
                            } else {
                                data.push(char_row_to_string_slice(&ca.data, ca.cols, 0));
                            }
                            kinds.push(CellElementKind::Char);
                        }
                        Value::CharArray(_) => return Err(runtime_error_for(CELL_ELEMENT_ERROR)),
                        _ => return Err(runtime_error_for(CELL_ELEMENT_ERROR)),
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
            _ => Err(runtime_error_for(ARG_TYPE_ERROR)),
        }
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self, idx: usize) -> &str {
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
                    let array = StringArray::new(data, output_shape)
                        .map_err(|e| runtime_error_for(format!("{FN_NAME}: {e}")))?;
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
                    let array = StringArray::new(data, output_shape)
                        .map_err(|e| runtime_error_for(format!("{FN_NAME}: {e}")))?;
                    Ok(Value::StringArray(array))
                }
            }
            TextKind::StringArray => {
                let data = results.into_iter().map(|r| r.text).collect::<Vec<_>>();
                let array = StringArray::new(data, output_shape)
                    .map_err(|e| runtime_error_for(format!("{FN_NAME}: {e}")))?;
                Ok(Value::StringArray(array))
            }
            TextKind::CharArray { rows } => {
                if rows == 0 {
                    return CharArray::new(Vec::new(), 0, 0)
                        .map(Value::CharArray)
                        .map_err(|e| runtime_error_for(format!("{FN_NAME}: {e}")));
                }
                if results.len() != rows {
                    return Err(runtime_error_for(SIZE_MISMATCH_ERROR));
                }
                let mut max_width = 0usize;
                let mut row_strings = Vec::with_capacity(rows);
                for result in &results {
                    let width = result.text.chars().count();
                    max_width = max_width.max(width);
                    row_strings.push(result.text.clone());
                }
                let mut flattened = Vec::with_capacity(rows * max_width);
                for row in row_strings {
                    let mut chars: Vec<char> = row.chars().collect();
                    if chars.len() < max_width {
                        chars.resize(max_width, ' ');
                    }
                    flattened.extend(chars);
                }
                CharArray::new(flattened, rows, max_width)
                    .map(Value::CharArray)
                    .map_err(|e| runtime_error_for(format!("{FN_NAME}: {e}")))
            }
            TextKind::CellArray(info) => {
                if results.len() != info.element_kinds.len() {
                    return Err(runtime_error_for(SIZE_MISMATCH_ERROR));
                }
                let mut values = Vec::with_capacity(results.len());
                for (idx, result) in results.into_iter().enumerate() {
                    match info.element_kinds[idx] {
                        CellElementKind::String => values.push(Value::String(result.text)),
                        CellElementKind::Char => {
                            let ca = CharArray::new_row(&result.text);
                            values.push(Value::CharArray(ca));
                        }
                    }
                }
                make_cell_with_shape(values, info.shape)
                    .map_err(|e| runtime_error_for(format!("{FN_NAME}: {e}")))
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
            other => Err(runtime_error_for(format!(
                "{BOUNDARY_TYPE_ERROR}: unsupported argument {other:?}"
            ))),
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

    fn text(&self, idx: usize) -> &str {
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
    data: Vec<String>,
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
                data: sa.data.clone(),
                shape: sa.shape.clone(),
            }),
            Value::CharArray(ca) => {
                let mut data = Vec::with_capacity(ca.rows);
                for row in 0..ca.rows {
                    data.push(char_row_to_string_slice(&ca.data, ca.cols, row));
                }
                Ok(Self {
                    data,
                    shape: vec![ca.rows, 1],
                })
            }
            Value::Cell(cell) => {
                let shape = cell.shape.clone();
                let mut data = Vec::with_capacity(cell.data.len());
                for element in &cell.data {
                    match &**element {
                        Value::String(s) => data.push(s.clone()),
                        Value::StringArray(sa) if sa.data.len() == 1 => {
                            data.push(sa.data[0].clone());
                        }
                        Value::CharArray(ca) if ca.rows <= 1 => {
                            if ca.rows == 0 {
                                data.push(String::new());
                            } else {
                                data.push(char_row_to_string_slice(&ca.data, ca.cols, 0));
                            }
                        }
                        Value::CharArray(_) => return Err(runtime_error_for(CELL_ELEMENT_ERROR)),
                        _ => return Err(runtime_error_for(CELL_ELEMENT_ERROR)),
                    }
                }
                Ok(Self { data, shape })
            }
            _ => Err(runtime_error_for(BOUNDARY_TYPE_ERROR)),
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
                let mut data = Vec::with_capacity(t.data.len());
                for &entry in &t.data {
                    data.push(parse_position(entry)?);
                }
                Ok(Self {
                    data,
                    shape: if t.shape.is_empty() {
                        vec![t.rows, t.cols.max(1)]
                    } else {
                        t.shape
                    },
                })
            }
            _ => Err(runtime_error_for(BOUNDARY_TYPE_ERROR)),
        }
    }
}

fn parse_position(value: f64) -> BuiltinResult<usize> {
    if !value.is_finite() || value < 1.0 {
        return Err(runtime_error_for(POSITION_TYPE_ERROR));
    }
    if (value.fract()).abs() > f64::EPSILON {
        return Err(runtime_error_for(POSITION_TYPE_ERROR));
    }
    if value > (usize::MAX as f64) {
        return Err(runtime_error_for(POSITION_TYPE_ERROR));
    }
    Ok(value as usize)
}

fn parse_position_int(value: IntValue) -> BuiltinResult<usize> {
    let val = value.to_i64();
    if val <= 0 {
        return Err(runtime_error_for(POSITION_TYPE_ERROR));
    }
    Ok(val as usize)
}

#[cfg(test)]
pub(crate) mod tests {
    #![allow(non_snake_case)]

    use super::*;
    use runmat_builtins::{CellArray, CharArray, StringArray, Tensor};

    fn erase_between_builtin(
        text: Value,
        start: Value,
        stop: Value,
        rest: Vec<Value>,
    ) -> BuiltinResult<Value> {
        futures::executor::block_on(super::erase_between_builtin(text, start, stop, rest))
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
        let strings = StringArray::new(vec!["<missing>".into()], vec![1, 1]).unwrap();
        let result = erase_between_builtin(
            Value::StringArray(strings),
            Value::String("<".into()),
            Value::String(">".into()),
            Vec::new(),
        )
        .expect("eraseBetween");
        assert_eq!(
            result,
            Value::StringArray(StringArray::new(vec!["<missing>".into()], vec![1, 1]).unwrap())
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eraseBetween_zero_sized_broadcast_produces_empty_array() {
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
                assert_eq!(sa.data.len(), 0);
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
                assert_eq!(sa.data, vec!["d".to_string(), "w".to_string()]);
                assert_eq!(sa.shape, vec![2, 1]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn eraseBetween_cell_array_preserves_types() {
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
        let chars =
            CharArray::new("Device<GPU>".chars().collect(), 1, "Device<GPU>".len()).unwrap();
        let default = erase_between_builtin(
            Value::CharArray(chars.clone()),
            Value::String("<".into()),
            Value::String(">".into()),
            Vec::new(),
        )
        .expect("eraseBetween");
        match default {
            Value::CharArray(out) => {
                let text: String = out.data.iter().collect();
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
                let text: String = out.data.iter().collect();
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
                assert_eq!(sa.data, vec!["alpha[]".to_string(), "beta[]".to_string()]);
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
        assert_eq!(err.to_string(), OPTION_VALUE_ERROR);
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
        assert_eq!(err.to_string(), OPTION_NAME_ERROR);
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
        assert_eq!(err.to_string(), OPTION_PAIR_ERROR);
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
        assert_eq!(err.to_string(), POSITION_TYPE_ERROR);
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
        assert_eq!(err.to_string(), BOUNDARY_TYPE_ERROR);
    }
}
