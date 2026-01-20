//! MATLAB-compatible `strcat` builtin with GPU-aware semantics for RunMat.

use runmat_builtins::{CellArray, CharArray, StringArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};
use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::common::{char_row_to_string_slice, is_missing_string};
use crate::{
    build_runtime_error, gather_if_needed, make_cell_with_shape, BuiltinResult, RuntimeError,
};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "strcat",
        builtin_path = "crate::builtins::strings::transform::strcat"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "strcat"
category: "strings/transform"
keywords: ["strcat", "string concatenation", "character arrays", "cell arrays", "trailing spaces"]
summary: "Concatenate text inputs element-wise with MATLAB-compatible trimming and implicit expansion."
references:
  - https://www.mathworks.com/help/matlab/ref/strcat.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "matlab"
  notes: "Executes on the CPU; GPU-resident inputs are gathered before concatenation so trimming semantics match MATLAB."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 8
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::strings::transform::strcat::tests"
  integration: "builtins::strings::transform::strcat::tests::strcat_cell_array_trims_trailing_spaces"
---

# What does the `strcat` function do in MATLAB / RunMat?
`strcat` horizontally concatenates text inputs element-wise. It accepts string arrays, character arrays,
character vectors, and cell arrays of character vectors, applying MATLAB's implicit expansion rules to
match array sizes.

## How does the `strcat` function behave in MATLAB / RunMat?
- Inputs are concatenated element-wise. Scalars expand across arrays of matching dimensions using MATLAB's
  implicit expansion rules.
- When at least one input is a string array (or string scalar), the result is a string array. `<missing>`
  values propagate, so any missing operand yields a missing result for that element.
- When no string arrays are present but any input is a cell array of character vectors, the result is a cell
  array whose elements are character vectors.
- Otherwise, the result is a character array. For character inputs, `strcat` removes trailing space characters
  from each operand **before** concatenating.
- Cell array elements must be character vectors (or string scalars). Mixing cell arrays with unsupported
  content raises a MATLAB-compatible error.
- Empty inputs broadcast naturally: an operand with a zero-length dimension yields an empty output after
  broadcasting.

## `strcat` Function GPU Execution Behaviour
RunMat currently performs text concatenation on the CPU. When any operand resides on the GPU, the runtime
gathers it to host memory before applying MATLAB-compatible trimming and concatenation rules. Providers do
not need to implement device kernels for this builtin today.

## GPU residency in RunMat (Do I need `gpuArray`?)
No. String manipulation runs on the CPU. If intermediate values are on the GPU, RunMat gathers them
automatically so you can call `strcat` without extra residency management.

## Examples of using the `strcat` function in MATLAB / RunMat

### Concatenate string scalars element-wise
```matlab
greeting = strcat("Run", "Mat");
```
Expected output:
```matlab
greeting = "RunMat"
```

### Concatenate a string scalar with a string array
```matlab
names = ["Ignition", "Turbine", "Accelerate"];
tagged = strcat("runmat-", names);
```
Expected output:
```matlab
tagged = 1×3 string
    "runmat-Ignition"    "runmat-Turbine"    "runmat-Accelerate"
```

### Concatenate character arrays while trimming trailing spaces
```matlab
A = char("GPU ", "Planner");
B = char("Accel", " Stage ");
result = strcat(A, B);
```
Expected output:
```matlab
result =

  2×11 char array

    'GPUAccel'
    'PlannerStage'
```

### Concatenate cell arrays of character vectors
```matlab
C = {'Run ', 'Plan '; 'Fuse ', 'Cache '};
suffix = {'Mat', 'Ops'; 'Kernels', 'Stats'};
combined = strcat(C, suffix);
```
Expected output:
```matlab
combined = 2×2 cell
    {'RunMat'}    {'PlanOps'}
    {'FuseKernels'}    {'CacheStats'}
```

### Propagate missing strings during concatenation
```matlab
values = [string(missing) "ready"];
out = strcat("job-", values);
```
Expected output:
```matlab
out = 1×2 string
    <missing>    "job-ready"
```

### Broadcast a scalar character vector across a character array
```matlab
labels = char("core", "runtime", "planner");
prefixed = strcat("runmat-", labels);
```
Expected output:
```matlab
prefixed =

  3×11 char array

    'runmat-core'
    'runmat-runtime'
    'runmat-planner'
```

## FAQ

### Does `strcat` remove spaces between words?
No. `strcat` only strips trailing **space characters** from character inputs before concatenating. Spaces in
the middle of a string remain untouched. To insert separators explicitly, concatenate the desired delimiter
or use `join`.

### How are missing strings handled?
Missing string scalars (`string(missing)`) propagate. If any operand is missing for a specific element, the
resulting element is `<missing>`.

### What happens when I mix strings and character arrays?
The output is a string array. Character inputs are converted to strings (after trimming trailing spaces) and
combined element-wise with the string operands.

### Can I concatenate cell arrays with string arrays?
Yes. Inputs are implicitly converted to strings when any operand is a string array, so the result is a string
array. Cell array elements must still contain character vectors (or scalar strings).

### What if I pass numeric or logical inputs?
`strcat` only accepts strings, character arrays, character vectors, or cell arrays of character vectors.
Passing unsupported types raises a MATLAB-compatible error.

### How are empty inputs treated?
Dimensions with length zero propagate through implicit expansion. For example, concatenating with an empty
string array returns an empty array with the broadcasted shape.

## See Also
[string](./string), [plus](./plus) (string concatenation with operator overloading),
[join](./join), [cellstr](./cellstr)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/strings/transform/strcat.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/transform/strcat.rs)
- Found an issue? Please [open a GitHub issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::transform::strcat")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "strcat",
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
    notes: "Executes on the CPU with trailing-space trimming; GPU inputs are gathered before concatenation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::transform::strcat")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "strcat",
    shape: ShapeRequirements::BroadcastCompatible,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "String concatenation runs on the host and is not eligible for fusion.",
};

const BUILTIN_NAME: &str = "strcat";
const ERROR_NOT_ENOUGH_INPUTS: &str = "strcat: not enough input arguments";
const ERROR_INVALID_INPUT: &str =
    "strcat: inputs must be strings, character arrays, or cell arrays of character vectors";
const ERROR_INVALID_CELL_ELEMENT: &str =
    "strcat: cell array elements must be character vectors or string scalars";

fn runtime_error_for(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn map_flow(err: RuntimeError) -> RuntimeError {
    map_control_flow_with_builtin(err, BUILTIN_NAME)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OperandKind {
    String,
    Cell,
    Char,
}

#[derive(Clone)]
struct TextElement {
    text: String,
    missing: bool,
}

#[derive(Clone)]
struct TextOperand {
    data: Vec<TextElement>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    kind: OperandKind,
}

impl TextOperand {
    fn from_value(value: Value) -> BuiltinResult<Self> {
        match value {
            Value::String(s) => Ok(Self::from_string_scalar(s)),
            Value::StringArray(sa) => Ok(Self::from_string_array(sa)),
            Value::CharArray(ca) => Self::from_char_array(&ca),
            Value::Cell(ca) => Self::from_cell_array(&ca),
            _ => Err(runtime_error_for(ERROR_INVALID_INPUT)),
        }
    }

    fn from_string_scalar(text: String) -> Self {
        let missing = is_missing_string(&text);
        Self {
            data: vec![TextElement { text, missing }],
            shape: vec![1, 1],
            strides: vec![1, 1],
            kind: OperandKind::String,
        }
    }

    fn from_string_array(array: StringArray) -> Self {
        let missing_flags: Vec<bool> = array.data.iter().map(|s| is_missing_string(s)).collect();
        let data = array
            .data
            .into_iter()
            .zip(missing_flags)
            .map(|(text, missing)| TextElement { text, missing })
            .collect();
        let shape = array.shape.clone();
        let strides = compute_strides(&shape);
        Self {
            data,
            shape,
            strides,
            kind: OperandKind::String,
        }
    }

    fn from_char_array(array: &CharArray) -> BuiltinResult<Self> {
        let rows = array.rows;
        let cols = array.cols;
        let mut elements = Vec::with_capacity(rows);
        for row in 0..rows {
            let text = char_row_to_string_slice(&array.data, cols, row);
            let trimmed = trim_trailing_spaces(&text);
            elements.push(TextElement {
                text: trimmed,
                missing: false,
            });
        }
        let shape = vec![rows, 1];
        let strides = compute_row_major_strides(&shape);
        Ok(Self {
            data: elements,
            shape,
            strides,
            kind: OperandKind::Char,
        })
    }

    fn from_cell_array(array: &CellArray) -> BuiltinResult<Self> {
        let total = array.data.len();
        let mut elements = Vec::with_capacity(total);
        for handle in &array.data {
            let text_element = cell_element_to_text(handle)?;
            elements.push(text_element);
        }
        let shape = array.shape.clone();
        let strides = compute_row_major_strides(&shape);
        Ok(Self {
            data: elements,
            shape,
            strides,
            kind: OperandKind::Cell,
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OutputKind {
    Char,
    Cell,
    String,
}

impl OutputKind {
    fn update(self, operand_kind: OperandKind) -> Self {
        match (self, operand_kind) {
            (_, OperandKind::String) => OutputKind::String,
            (OutputKind::String, _) => OutputKind::String,
            (OutputKind::Cell, _) => OutputKind::Cell,
            (_, OperandKind::Cell) => OutputKind::Cell,
            _ => self,
        }
    }
}

fn trim_trailing_spaces(text: &str) -> String {
    text.trim_end_matches(|ch: char| ch.is_ascii_whitespace())
        .to_string()
}

fn compute_row_major_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut strides = vec![0usize; shape.len()];
    let mut stride = 1usize;
    for dim in (0..shape.len()).rev() {
        strides[dim] = stride;
        let extent = shape[dim].max(1);
        stride = stride.saturating_mul(extent);
    }
    strides
}

fn column_major_coords(mut index: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut coords = Vec::with_capacity(shape.len());
    for &extent in shape {
        if extent == 0 {
            coords.push(0);
        } else {
            coords.push(index % extent);
            index /= extent;
        }
    }
    coords
}

fn row_major_index(coords: &[usize], shape: &[usize]) -> usize {
    if coords.is_empty() {
        return 0;
    }
    let mut index = 0usize;
    let mut stride = 1usize;
    for dim in (0..coords.len()).rev() {
        let extent = shape[dim].max(1);
        index += coords[dim] * stride;
        stride = stride.saturating_mul(extent);
    }
    index
}

fn cell_element_to_text(value: &Value) -> BuiltinResult<TextElement> {
    match value {
        Value::String(s) => Ok(TextElement {
            text: s.clone(),
            missing: is_missing_string(s),
        }),
        Value::StringArray(sa) if sa.data.len() == 1 => {
            let text = sa.data[0].clone();
            Ok(TextElement {
                missing: is_missing_string(&text),
                text,
            })
        }
        Value::CharArray(ca) if ca.rows <= 1 => {
            let text = if ca.rows == 0 {
                String::new()
            } else {
                char_row_to_string_slice(&ca.data, ca.cols, 0)
            };
            Ok(TextElement {
                text: trim_trailing_spaces(&text),
                missing: false,
            })
        }
        Value::CharArray(_) => Err(runtime_error_for(ERROR_INVALID_CELL_ELEMENT)),
        _ => Err(runtime_error_for(ERROR_INVALID_CELL_ELEMENT)),
    }
}

#[runtime_builtin(
    name = "strcat",
    category = "strings/transform",
    summary = "Concatenate strings, character arrays, or cell arrays of character vectors element-wise.",
    keywords = "strcat,string concatenation,character arrays,cell arrays",
    accel = "sink",
    builtin_path = "crate::builtins::strings::transform::strcat"
)]
fn strcat_builtin(rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.is_empty() {
        return Err(runtime_error_for(ERROR_NOT_ENOUGH_INPUTS));
    }

    let mut operands = Vec::with_capacity(rest.len());
    let mut output_kind = OutputKind::Char;

    for value in rest {
        let gathered = gather_if_needed(&value).map_err(map_flow)?;
        let operand = TextOperand::from_value(gathered)?;
        output_kind = output_kind.update(operand.kind);
        operands.push(operand);
    }

    let mut output_shape = operands
        .first()
        .map(|op| op.shape.clone())
        .unwrap_or_else(|| vec![1, 1]);
    for operand in operands.iter().skip(1) {
        output_shape = broadcast_shapes(BUILTIN_NAME, &output_shape, &operand.shape)
            .map_err(runtime_error_for)?;
    }

    let total_len: usize = output_shape.iter().product();
    let mut concatenated = Vec::with_capacity(total_len);

    for linear in 0..total_len {
        let mut buffer = String::new();
        let mut any_missing = false;
        for operand in &operands {
            let idx = broadcast_index(linear, &output_shape, &operand.shape, &operand.strides);
            let element = &operand.data[idx];
            if output_kind == OutputKind::String && element.missing {
                any_missing = true;
                continue;
            }
            buffer.push_str(&element.text);
        }
        if matches!(output_kind, OutputKind::String) && any_missing {
            concatenated.push(String::from("<missing>"));
        } else {
            concatenated.push(buffer);
        }
    }

    match output_kind {
        OutputKind::String => build_string_output(concatenated, &output_shape),
        OutputKind::Cell => build_cell_output(concatenated, &output_shape),
        OutputKind::Char => build_char_output(concatenated),
    }
}

fn build_string_output(data: Vec<String>, shape: &[usize]) -> BuiltinResult<Value> {
    if data.is_empty() {
        let array = StringArray::new(data, shape.to_vec())
            .map_err(|e| runtime_error_for(format!("{BUILTIN_NAME}: {e}")))?;
        return Ok(Value::StringArray(array));
    }

    let is_scalar = shape.is_empty() || shape.iter().all(|&dim| dim == 1);
    if is_scalar {
        return Ok(Value::String(data[0].clone()));
    }

    let array = StringArray::new(data, shape.to_vec())
        .map_err(|e| runtime_error_for(format!("{BUILTIN_NAME}: {e}")))?;
    Ok(Value::StringArray(array))
}

fn build_cell_output(mut data: Vec<String>, shape: &[usize]) -> BuiltinResult<Value> {
    if data.is_empty() {
        return make_cell_with_shape(Vec::new(), shape.to_vec())
            .map_err(|e| runtime_error_for(format!("{BUILTIN_NAME}: {e}")));
    }
    if shape.len() > 1 {
        let mut reordered = vec![String::new(); data.len()];
        for (cm_index, text) in data.into_iter().enumerate() {
            let coords = column_major_coords(cm_index, shape);
            let rm_index = row_major_index(&coords, shape);
            reordered[rm_index] = text;
        }
        data = reordered;
    }
    let mut values = Vec::with_capacity(data.len());
    for text in data {
        let char_array = CharArray::new_row(&text);
        values.push(Value::CharArray(char_array));
    }
    make_cell_with_shape(values, shape.to_vec())
        .map_err(|e| runtime_error_for(format!("{BUILTIN_NAME}: {e}")))
}

fn build_char_output(data: Vec<String>) -> BuiltinResult<Value> {
    let rows = data.len();
    if rows == 0 {
        let array = CharArray::new(Vec::new(), 0, 0)
            .map_err(|e| runtime_error_for(format!("{BUILTIN_NAME}: {e}")))?;
        return Ok(Value::CharArray(array));
    }

    let max_cols = data.iter().map(|s| s.chars().count()).max().unwrap_or(0);
    let mut chars = Vec::with_capacity(rows * max_cols);
    for text in data {
        let mut row_chars: Vec<char> = text.chars().collect();
        if row_chars.len() < max_cols {
            row_chars.resize(max_cols, ' ');
        }
        chars.extend(row_chars.into_iter());
    }
    let array = CharArray::new(chars, rows, max_cols)
        .map_err(|e| runtime_error_for(format!("{BUILTIN_NAME}: {e}")))?;
    Ok(Value::CharArray(array))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    #[cfg(feature = "wgpu")]
    use runmat_builtins::Tensor;
    use runmat_builtins::{CellArray, CharArray, IntValue, StringArray};

    use crate::builtins::common::test_support;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_string_scalar_concatenation() {
        let result = strcat_builtin(vec![
            Value::String("Run".into()),
            Value::String("Mat".into()),
        ])
        .expect("strcat");
        assert_eq!(result, Value::String("RunMat".into()));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_string_array_broadcasts_scalar() {
        let array = StringArray::new(vec!["core".into(), "runtime".into()], vec![1, 2]).unwrap();
        let result = strcat_builtin(vec![
            Value::String("runmat-".into()),
            Value::StringArray(array),
        ])
        .expect("strcat");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![1, 2]);
                assert_eq!(
                    sa.data,
                    vec![String::from("runmat-core"), String::from("runmat-runtime")]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_char_array_multiple_rows_concatenates_per_row() {
        let first = CharArray::new(vec!['A', ' ', 'B', 'C'], 2, 2).expect("char");
        let second = CharArray::new(vec!['X', 'Y', 'Z', ' '], 2, 2).expect("char");
        let result = strcat_builtin(vec![Value::CharArray(first), Value::CharArray(second)])
            .expect("strcat");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 2);
                assert_eq!(ca.cols, 3);
                let expected: Vec<char> = vec!['A', 'X', 'Y', 'B', 'C', 'Z'];
                assert_eq!(ca.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_char_array_trims_trailing_spaces() {
        let first = CharArray::new_row("GPU ");
        let second = CharArray::new_row(" Accel  ");
        let result = strcat_builtin(vec![Value::CharArray(first), Value::CharArray(second)])
            .expect("strcat");
        match result {
            Value::CharArray(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 9);
                let expected: Vec<char> = "GPU Accel".chars().collect();
                assert_eq!(ca.data, expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_mixed_char_and_string_returns_string_array() {
        let prefixes = CharArray::new(vec!['A', ' ', 'B', ' '], 2, 2).expect("char");
        let suffixes =
            StringArray::new(vec!["core".into(), "runtime".into()], vec![1, 2]).expect("strings");
        let result = strcat_builtin(vec![
            Value::CharArray(prefixes),
            Value::StringArray(suffixes),
        ])
        .expect("strcat");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![2, 2]);
                assert_eq!(
                    sa.data,
                    vec![
                        "Acore".to_string(),
                        "Bcore".to_string(),
                        "Aruntime".to_string(),
                        "Bruntime".to_string()
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_cell_array_trims_trailing_spaces() {
        let cell = make_cell_with_shape(
            vec![
                Value::CharArray(CharArray::new_row("Run ")),
                Value::CharArray(CharArray::new_row("Mat ")),
            ],
            vec![1, 2],
        )
        .expect("cell");
        let suffix = Value::CharArray(CharArray::new_row("Core "));
        let result = strcat_builtin(vec![cell, suffix]).expect("strcat");
        match result {
            Value::Cell(ca) => {
                assert_eq!(ca.shape, vec![1, 2]);
                let first: &Value = &ca.data[0];
                let second: &Value = &ca.data[1];
                match (first, second) {
                    (Value::CharArray(a), Value::CharArray(b)) => {
                        assert_eq!(a.data, "RunCore".chars().collect::<Vec<char>>());
                        assert_eq!(b.data, "MatCore".chars().collect::<Vec<char>>());
                    }
                    other => panic!("unexpected cell contents {other:?}"),
                }
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_cell_array_two_by_two_preserves_row_major_order() {
        let cell = make_cell_with_shape(
            vec![
                Value::CharArray(CharArray::new_row("Top ")),
                Value::CharArray(CharArray::new_row("Right ")),
                Value::CharArray(CharArray::new_row("Bottom ")),
                Value::CharArray(CharArray::new_row("Last ")),
            ],
            vec![2, 2],
        )
        .expect("cell");
        let suffix = Value::CharArray(CharArray::new_row("X"));
        let result = strcat_builtin(vec![cell, suffix]).expect("strcat");
        match result {
            Value::Cell(ca) => {
                assert_eq!(ca.shape, vec![2, 2]);
                let v00 = ca.get(0, 0).expect("cell (0,0)");
                let v01 = ca.get(0, 1).expect("cell (0,1)");
                let v10 = ca.get(1, 0).expect("cell (1,0)");
                let v11 = ca.get(1, 1).expect("cell (1,1)");
                match (v00, v01, v10, v11) {
                    (
                        Value::CharArray(a),
                        Value::CharArray(b),
                        Value::CharArray(c),
                        Value::CharArray(d),
                    ) => {
                        assert_eq!(a.data, "TopX".chars().collect::<Vec<char>>());
                        assert_eq!(b.data, "RightX".chars().collect::<Vec<char>>());
                        assert_eq!(c.data, "BottomX".chars().collect::<Vec<char>>());
                        assert_eq!(d.data, "LastX".chars().collect::<Vec<char>>());
                    }
                    other => panic!("unexpected cell contents {other:?}"),
                }
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_missing_strings_propagate() {
        let array = StringArray::new(
            vec![String::from("<missing>"), String::from("ready")],
            vec![1, 2],
        )
        .unwrap();
        let result = strcat_builtin(vec![
            Value::String("job-".into()),
            Value::StringArray(array),
        ])
        .expect("strcat");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.data[0], "<missing>");
                assert_eq!(sa.data[1], "job-ready");
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_empty_dimension_returns_empty_array() {
        let empty = StringArray::new(Vec::<String>::new(), vec![0, 2]).expect("string array");
        let result = strcat_builtin(vec![
            Value::StringArray(empty),
            Value::String("prefix".into()),
        ])
        .expect("strcat");
        match result {
            Value::StringArray(sa) => {
                assert_eq!(sa.shape, vec![0, 2]);
                assert!(sa.data.is_empty());
            }
            other => panic!("expected empty string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_errors_on_invalid_input_type() {
        let err = strcat_builtin(vec![Value::Int(IntValue::I32(4))]).expect_err("expected error");
        assert!(err.to_string().contains("inputs must be strings"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_errors_on_mismatched_sizes() {
        let left = CharArray::new(vec!['A', 'B'], 2, 1).expect("char");
        let right = CharArray::new(vec!['C', 'D', 'E'], 3, 1).expect("char");
        let err = strcat_builtin(vec![Value::CharArray(left), Value::CharArray(right)])
            .expect_err("expected broadcast error");
        let err_text = err.to_string();
        assert!(
            err_text.contains("size mismatch"),
            "unexpected error text: {err_text}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_errors_on_invalid_cell_element() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).expect("cell");
        let err = strcat_builtin(vec![Value::Cell(cell)]).expect_err("expected error");
        assert!(err
            .to_string()
            .contains("cell array elements must be character vectors"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_errors_on_empty_argument_list() {
        let err = strcat_builtin(Vec::new()).expect_err("expected error");
        assert_eq!(err.to_string(), ERROR_NOT_ENOUGH_INPUTS);
    }

    #[cfg(feature = "wgpu")]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcat_gpu_operand_still_errors_on_type() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0], vec![1, 2]).expect("tensor");
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let err = strcat_builtin(vec![Value::GpuTensor(handle)]).expect_err("expected error");
            assert!(err.to_string().contains("inputs must be strings"));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_compile() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
