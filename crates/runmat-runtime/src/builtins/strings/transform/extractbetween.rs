//! MATLAB-compatible `extractBetween` builtin with GPU-aware semantics for RunMat.

use std::cmp::min;

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};
use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::strings::common::{char_row_to_string_slice, is_missing_string};
use crate::{
    build_runtime_error, gather_if_needed, make_cell_with_shape, BuiltinResult, RuntimeError,
};
use runmat_builtins::{CharArray, IntValue, StringArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "extractBetween",
        builtin_path = "crate::builtins::strings::transform::extractbetween"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "extractBetween"
category: "strings/transform"
keywords: ["extractBetween", "substring", "boundaries", "inclusive", "exclusive", "string array"]
summary: "Extract text that lies between two boundary markers using string or position inputs."
references:
  - https://www.mathworks.com/help/matlab/ref/extractbetween.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "matlab"
  notes: "Runs on the CPU. GPU-resident inputs are gathered before processing, results stay on the host, and the builtin is registered as an Accelerate sink."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 3
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::strings::transform::extractbetween::tests"
  integration: "builtins::strings::transform::extractbetween::tests::extractBetween_cell_array_preserves_types"
---

# What does the `extractBetween` function do in MATLAB / RunMat?
`extractBetween(text, start, stop)` locates the substring that appears between two boundary markers.
Markers can be text (string scalars, character vectors, or cells that contain them) or numeric
positions. The builtin mirrors MATLAB semantics for broadcasting, missing values, and the optional
`'Boundaries'` name-value argument.

## How does the `extractBetween` function behave in MATLAB / RunMat?
- Accepts **string scalars**, **string arrays**, **character arrays** (interpreted row-by-row), and
  **cell arrays** that contain string scalars or character vectors. Cell outputs preserve the
  element type (string vs. char) of each cell.
- Boundary inputs can be text or numeric positions. Both boundaries in a call must use the same
  kind of input; mixing text and numeric markers raises a size/type error.
- Scalar text markers follow MATLAB implicit expansion, applying to every element of the text
  input. Character-array and cell inputs must exactly match the text shape.
- The `'Boundaries'` name-value pair controls inclusivity. Text markers default to **exclusive**
  extraction, while numeric positions default to **inclusive** behaviour. Values are
  case-insensitive and must be `'exclusive'` or `'inclusive'`.
- Missing string scalars propagate: if the text, start marker, or end marker is `<missing>`,
  the result is also `<missing>`.
- When the start or end boundary cannot be located, `extractBetween` returns an empty string (or an
  appropriately padded empty row for character arrays).
- Numeric positions use 1-based indexing. Inputs are validated as positive integers, clamped to
  string length, and honour inclusivity rules exactly as MATLAB does.

## `extractBetween` Function GPU Execution Behaviour
Text manipulation executes on the CPU. When any argument resides on the GPU, RunMat gathers the
values to host memory, performs extraction, and leaves the results on the host. No Accelerate
provider hooks are required, and the builtin is registered as an Accelerate sink so fusion plans
never attempt to keep data on the device for this operation.

## Examples of using the `extractBetween` function in MATLAB / RunMat

### Extract text between words in a string
```matlab
txt = "RunMat accelerates MATLAB workloads";
segment = extractBetween(txt, "RunMat ", " workloads");
```
Expected output:
```matlab
segment = "accelerates MATLAB"
```

### Include boundary markers with the `'Boundaries'` option
```matlab
path = "snapshots/run/fusion.mat";
withMarkers = extractBetween(path, "snapshots/", ".mat", "Boundaries", "inclusive");
```
Expected output:
```matlab
withMarkers = "snapshots/run/fusion.mat"
```

### Use numeric positions for 1-based indexing
```matlab
name = "Accelerator";
middle = extractBetween(name, 3, 7);
```
Expected output:
```matlab
middle = "celer"
```

### Apply scalar text markers to each element of a string array
```matlab
files = ["runmat_accel.rs", "runmat_gc.rs"; "runmat_plot.rs", "runmat_cli.rs"];
stems = extractBetween(files, "runmat_", ".rs");
```
Expected output:
```matlab
stems = 2×2 string
    "accel"    "gc"
    "plot"     "cli"
```

### Work with character arrays while preserving row padding
```matlab
chars = char("Device<GPU>", "Planner<Fusion>");
tokens = extractBetween(chars, "<", ">");
```
Expected output:
```matlab
tokens =

  2×6 char array

    "GPU   "
    "Fusion"
```

### Preserve element types in cell arrays
```matlab
C = {'<missing>', 'A[B]C'; "Planner <Fusion>", "Device<GPU>"};
out = extractBetween(C, "<", ">");
```
Expected output:
```matlab
out =
  2×2 cell array
    {'<missing>'}    {'B'}
    {"Fusion"}       {"GPU"}
```

### Handle missing strings without throwing errors
```matlab
txt = ["<missing>", "Planner<GPU>"];
tokens = extractBetween(txt, "<", ">");
```
Expected output:
```matlab
tokens = 1×2 string
    "<missing>"    "GPU"
```

## FAQ

### Which argument types does `extractBetween` accept?
The first argument can be a string scalar, string array, character array, or cell array of character
vectors / string scalars. Boundary arguments can be text (string, character array, or cell) or numeric
positions supplied as scalars, vectors, or arrays.

### Can the start and end arguments mix text and numeric positions?
No. Both boundaries must be text markers or both must be numeric positions. Mixing types raises a
size/type error, mirroring MATLAB.

### What happens when a boundary is not found?
`extractBetween` returns the empty string (`""`). Character-array outputs contain space padded rows
of the appropriate length.

### How does `'Boundaries','inclusive'` behave with numeric positions?
Inclusive mode returns the substring that includes both indices. Exclusive mode removes the characters
at the specified start and end positions, yielding the text strictly between the two indices.

### Does `extractBetween` support implicit expansion?
Yes. Scalar boundaries expand against array inputs following MATLAB implicit expansion rules. Cell and
character array inputs must retain their original shape; attempting to expand them produces a size
mismatch error.

### Are GPU inputs supported?
Yes. Inputs stored on a GPU are gathered automatically. The function executes on the CPU, returns
host-side results, and fusion planning treats the builtin as a residency sink.

## See Also
[replace](./replace), [split](./split), [join](./join), [contains](./contains), [strfind](./strfind)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/strings/transform/extractbetween.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/extractbetween.rs)
- Found an issue? Please [open a GitHub issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

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
    notes: "Runs on the CPU; GPU-resident inputs are gathered before extraction and outputs are returned on the host.",
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
    notes: "Pure string manipulation builtin; excluded from fusion plans and gathers GPU inputs immediately.",
};

const FN_NAME: &str = "extractBetween";
const ARG_TYPE_ERROR: &str = "extractBetween: first argument must be a string array, character array, or cell array of character vectors";
const BOUNDARY_TYPE_ERROR: &str =
    "extractBetween: start and end arguments must both be text or both be numeric positions";
const POSITION_TYPE_ERROR: &str = "extractBetween: position arguments must be positive integers";
const OPTION_PAIR_ERROR: &str = "extractBetween: name-value arguments must appear in pairs";
const OPTION_NAME_ERROR: &str = "extractBetween: unrecognized parameter name";
const OPTION_VALUE_ERROR: &str =
    "extractBetween: 'Boundaries' must be either 'inclusive' or 'exclusive'";
const CELL_ELEMENT_ERROR: &str =
    "extractBetween: cell array elements must be string scalars or character vectors";
const SIZE_MISMATCH_ERROR: &str =
    "extractBetween: boundary sizes must be compatible with the text input";

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
    name = "extractBetween",
    category = "strings/transform",
    summary = "Extract substrings between boundary markers using MATLAB-compatible semantics.",
    keywords = "extractBetween,substring,boundaries,strings",
    accel = "sink",
    builtin_path = "crate::builtins::strings::transform::extractbetween"
)]
fn extract_between_builtin(
    text: Value,
    start: Value,
    stop: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    let text = gather_if_needed(&text).map_err(map_flow)?;
    let start = gather_if_needed(&start).map_err(map_flow)?;
    let stop = gather_if_needed(&stop).map_err(map_flow)?;

    let mode_override = parse_boundaries_option(&rest)?;

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

fn parse_boundaries_option(args: &[Value]) -> BuiltinResult<Option<BoundariesMode>> {
    if args.is_empty() {
        return Ok(None);
    }
    if !args.len().is_multiple_of(2) {
        return Err(runtime_error_for(OPTION_PAIR_ERROR));
    }

    let mut mode: Option<BoundariesMode> = None;
    let mut idx = 0;
    while idx < args.len() {
        let name_value = gather_if_needed(&args[idx]).map_err(map_flow)?;
        let name =
            value_to_string(&name_value).ok_or_else(|| runtime_error_for(OPTION_NAME_ERROR))?;
        if !name.eq_ignore_ascii_case("boundaries") {
            return Err(runtime_error_for(OPTION_NAME_ERROR));
        }
        let value = gather_if_needed(&args[idx + 1]).map_err(map_flow)?;
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
struct ExtractResult {
    text: String,
}

impl ExtractResult {
    fn missing() -> Self {
        Self {
            text: "<missing>".to_string(),
        }
    }

    fn text(text: String) -> Self {
        Self { text }
    }
}

fn extract_with_text_boundaries(
    text: &str,
    start: &str,
    stop: &str,
    mode: BoundariesMode,
) -> ExtractResult {
    if is_missing_string(text) || is_missing_string(start) || is_missing_string(stop) {
        return ExtractResult::missing();
    }

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
    text: &str,
    start: usize,
    stop: usize,
    mode: BoundariesMode,
) -> ExtractResult {
    if is_missing_string(text) {
        return ExtractResult::missing();
    }
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
    use crate::builtins::common::test_support;
    use runmat_builtins::{CellArray, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn extractBetween_basic_string() {
        let result = extract_between_builtin(
            Value::String("RunMat accelerates MATLAB".into()),
            Value::String("RunMat ".into()),
            Value::String(" MATLAB".into()),
            Vec::new(),
        )
        .expect("extractBetween");
        assert_eq!(result, Value::String("accelerates".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn extractBetween_inclusive_option() {
        let result = extract_between_builtin(
            Value::String("a[b]c".into()),
            Value::String("[".into()),
            Value::String("]".into()),
            vec![
                Value::String("Boundaries".into()),
                Value::String("inclusive".into()),
            ],
        )
        .expect("extractBetween");
        assert_eq!(result, Value::String("[b]".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn extractBetween_numeric_positions() {
        let result = extract_between_builtin(
            Value::String("Accelerator".into()),
            Value::Num(3.0),
            Value::Num(7.0),
            Vec::new(),
        )
        .expect("extractBetween");
        assert_eq!(result, Value::String("celer".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn extractBetween_numeric_positions_exclusive_option() {
        let result = extract_between_builtin(
            Value::String("Accelerator".into()),
            Value::Num(3.0),
            Value::Num(7.0),
            vec![
                Value::String("Boundaries".into()),
                Value::String("exclusive".into()),
            ],
        )
        .expect("extractBetween");
        assert_eq!(result, Value::String("ele".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn extractBetween_numeric_positions_clamps_stop() {
        let result = extract_between_builtin(
            Value::String("Accelerator".into()),
            Value::Num(3.0),
            Value::Num(100.0),
            Vec::new(),
        )
        .expect("extractBetween");
        assert_eq!(result, Value::String("celerator".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn extractBetween_numeric_positions_start_past_length() {
        let result = extract_between_builtin(
            Value::String("abc".into()),
            Value::Num(10.0),
            Value::Num(12.0),
            Vec::new(),
        )
        .expect("extractBetween");
        assert_eq!(result, Value::String(String::new()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn extractBetween_string_array_broadcast() {
        let array = StringArray::new(
            vec!["runmat_accel.rs".into(), "runmat_gc.rs".into()],
            vec![2, 1],
        )
        .unwrap();
        let result = extract_between_builtin(
            Value::StringArray(array),
            Value::String("runmat_".into()),
            Value::String(".rs".into()),
            Vec::new(),
        )
        .expect("extractBetween");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data, vec!["accel".to_string(), "gc".to_string()]);
                assert_eq!(sa.shape, vec![2, 1]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn extractBetween_char_array_rows() {
        let chars = CharArray::new(
            "GPUAccelerateIgnition".chars().collect(),
            1,
            "GPUAccelerateIgnition".len(),
        )
        .unwrap();
        let result = extract_between_builtin(
            Value::CharArray(chars),
            Value::String("GPU".into()),
            Value::String("tion".into()),
            Vec::new(),
        )
        .expect("extractBetween");
        match result {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 1);
                let text: String = out.data.iter().collect();
                assert_eq!(text.trim_end(), "AccelerateIgni");
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn extractBetween_cell_array_preserves_types() {
        let cell = CellArray::new(
            vec![
                Value::CharArray(CharArray::new_row("A[B]C")),
                Value::String("Planner<GPU>".into()),
            ],
            1,
            2,
        )
        .unwrap();
        let result = extract_between_builtin(
            Value::Cell(cell),
            Value::String("[".into()),
            Value::String("]".into()),
            Vec::new(),
        )
        .expect("extractBetween");
        match result {
            Value::Cell(out) => {
                let first = out.get(0, 0).unwrap();
                let second = out.get(0, 1).unwrap();
                assert_eq!(first, Value::CharArray(CharArray::new_row("B")));
                assert_eq!(second, Value::String(String::new()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn extractBetween_missing_string_propagates() {
        let strings = StringArray::new(vec!["<missing>".into()], vec![1, 1]).unwrap();
        let result = extract_between_builtin(
            Value::StringArray(strings),
            Value::String("[".into()),
            Value::String("]".into()),
            Vec::new(),
        )
        .expect("extractBetween");
        assert_eq!(
            result,
            Value::StringArray(StringArray::new(vec!["<missing>".into()], vec![1, 1]).unwrap())
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn extractBetween_position_type_error() {
        let err = extract_between_builtin(
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
    fn extractBetween_mixed_boundary_error() {
        let err = extract_between_builtin(
            Value::String("abc".into()),
            Value::String("a".into()),
            Value::Num(3.0),
            Vec::new(),
        )
        .unwrap_err();
        assert_eq!(err.to_string(), BOUNDARY_TYPE_ERROR);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn extractBetween_numeric_tensor_broadcast() {
        let text = StringArray::new(vec!["abcd".into(), "wxyz".into()], vec![2, 1]).unwrap();
        let start = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let stop = Tensor::new(vec![3.0, 4.0], vec![2, 1]).unwrap();
        let result = extract_between_builtin(
            Value::StringArray(text),
            Value::Tensor(start),
            Value::Tensor(stop),
            Vec::new(),
        )
        .expect("extractBetween");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data, vec!["abc".to_string(), "xyz".to_string()]);
                assert_eq!(sa.shape, vec![2, 1]);
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn extractBetween_option_invalid_value() {
        let err = extract_between_builtin(
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
    fn extractBetween_option_name_error() {
        let err = extract_between_builtin(
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
    fn extractBetween_option_pair_error() {
        let err = extract_between_builtin(
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
    fn extractBetween_missing_boundary_propagates() {
        let result = extract_between_builtin(
            Value::String("Planner<GPU>".into()),
            Value::String("<missing>".into()),
            Value::String(">".into()),
            Vec::new(),
        )
        .expect("extractBetween");
        assert_eq!(result, Value::String("<missing>".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn extractBetween_cell_boundary_arguments() {
        let text = CellArray::new(vec![Value::String("A<GPU>".into())], 1, 1).unwrap();
        let start = CellArray::new(vec![Value::CharArray(CharArray::new_row("<"))], 1, 1).unwrap();
        let stop = CellArray::new(vec![Value::CharArray(CharArray::new_row(">"))], 1, 1).unwrap();
        let result = extract_between_builtin(
            Value::Cell(text),
            Value::Cell(start),
            Value::Cell(stop),
            Vec::new(),
        )
        .expect("extractBetween");
        match result {
            Value::Cell(out) => {
                let value = out.get(0, 0).unwrap();
                assert_eq!(value, Value::String("GPU".into()));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn extractBetween_doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
