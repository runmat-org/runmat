//! MATLAB-compatible `cellstr` builtin implemented for the modern RunMat runtime.

use runmat_builtins::{CellArray, CharArray, StringArray, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::dispatcher::gather_if_needed;
use crate::{make_cell, make_cell_with_shape};

const ERR_INPUT_NOT_TEXT: &str =
    "cellstr: input must be a character array, string array, or cell array of character vectors";
const ERR_CELL_CONTENT_NOT_TEXT: &str =
    "cellstr: cell array elements must be character vectors or string scalars";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "cellstr",
        builtin_path = "crate::builtins::cells::core::cellstr"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "cellstr"
category: "cells/core"
keywords: ["cellstr", "cell array", "character vectors", "string conversion", "text", "gpu fallback"]
summary: "Convert character arrays, string arrays, or cell arrays of text into cell arrays of character vectors with MATLAB-compatible trimming semantics."
references:
  - https://www.mathworks.com/help/matlab/ref/cellstr.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs entirely on the host. GPU-resident inputs are gathered automatically and the result is always a host cell array."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::cells::core::cellstr::tests"
  integration: "builtins::cells::core::cellstr::tests::doc_examples_present"
---

# What does the `cellstr` function do in MATLAB / RunMat?
`cellstr` converts character arrays, string arrays, or cell arrays of character vectors into a
cell array whose elements are character vectors (1-by-N char arrays). Trailing spaces in character
array rows are trimmed to match MATLAB behaviour.

## How does the `cellstr` function behave in MATLAB / RunMat?
- `cellstr(A)` with a character array `A` returns an `m × 1` cell array whose entries correspond to
  the rows of `A`, with trailing spaces removed from each row.
- `cellstr(S)` with a string array `S` returns a cell array of the same size; each string scalar is
  converted to a character vector.
- `cellstr(C)` with a cell array `C` normalises the contents so that every element becomes a
  character vector. String scalars are converted, existing character vectors are preserved, and any
  other type triggers an error just like MATLAB.
- String arrays of any dimensionality preserve their exact shape. Ordering follows MATLAB's
  column-major semantics so downstream indexing sees the expected elements.
- Empty character arrays result in empty cell arrays (for example, `0 × n` char arrays map to
  `0 × 1` cells). Empty strings become `""` character vectors stored in single-cell outputs.
- Missing string scalars are rendered as the literal `'<missing>'`, matching MATLAB's textual
  representation.
- Inputs that are not text (numeric tensors, logical arrays, structs, GPU tensors, etc.) raise a
  MATLAB-compatible usage error.
- Cell elements must already be text. Any numeric, logical, or GPU value inside the cell raises an
  error so that downstream code never sees non-text rows.

## `cellstr` Function GPU Execution Behaviour
`cellstr` is a host-only builtin. If the input or any nested cell elements contain GPU tensors,
RunMat gathers them to the host before conversion (including nested elements inside cell arrays).
The resulting cell array is always allocated in host memory. No provider hooks are required at this
stage, but the GPU metadata keeps the runtime aware that residency does not propagate past this
builtin.

## Examples of using the `cellstr` function in MATLAB / RunMat

### Converting a character matrix into a column cell array
```matlab
A = ['apple '; 'berry '; 'citrus'];
C = cellstr(A);
```
Expected output:
```matlab
C =
  3x1 cell array
    {'apple'}
    {'berry'}
    {'citrus'}
```

### Removing trailing spaces automatically
```matlab
words = ['a   '; 'b   '; 'c   '];
C = cellstr(words);
```
Expected output:
```matlab
C =
  3x1 cell array
    {'a'}
    {'b'}
    {'c'}
```

### Converting a string array while preserving shape
```matlab
S = ["north" "south"; "east" "west"];
C = cellstr(S);
```
Expected output:
```matlab
C =
  2x2 cell array
    {'north'}    {'south'}
    {'east'}     {'west'}
```

### Normalising a cell array that mixes strings and character vectors
```matlab
C = {"left", 'right'};
out = cellstr(C);
```
Expected output:
```matlab
out =
  1x2 cell array
    {'left'}    {'right'}
```

### Handling empty character arrays
```matlab
emptyChars = char.empty(0, 5);
C = cellstr(emptyChars);
size(C)
```
Expected output:
```matlab
ans =
     0     1
```

### Converting a single string scalar
```matlab
single = "RunMat";
C = cellstr(single);
```
Expected output:
```matlab
C =
  1x1 cell array
    {'RunMat'}
```

### Validating non-text inputs
```matlab
try
    cellstr(42);
catch ME
    disp(ME.message)
end
```
Expected output:
```matlab
cellstr: input must be a character array, string array, or cell array of character vectors
```

## GPU residency in RunMat (Do I need `gpuArray`?)
No. `cellstr` executes on the host. If the input originates from the GPU (for example, a cell array
whose elements were gathered lazily), RunMat performs the gather automatically before formatting
the text. The output always resides on the CPU heap, mirroring MATLAB where cell arrays of text are
CPU-managed containers.

## FAQ

### Does `cellstr` trim whitespace inside the text?
No. Only trailing spaces added by rectangular character arrays are removed. Interior spaces and
tabs are preserved exactly as supplied.

### What happens when a cell element is not text?
`cellstr` raises an error: every element must be a character vector or string scalar. This matches
MATLAB and prevents silent conversion of unsupported data such as numeric arrays.

### Can the output stay on the GPU?
Not yet. The resulting cell array always lives on the host. Future versions may add GPU-backed cell
containers, in which case the GPU metadata in this builtin will be updated accordingly without
breaking user code.

### Are string arrays with missing values supported?
Yes. Missing string scalars (rendered as `<missing>`) convert to the character vector `'<missing>'`
in the output cell array.

### How are empty inputs handled?
Empty character arrays become empty cell arrays (`0×1`). Empty string arrays return empty cell
arrays matching their size. Empty strings become single cells that contain a 1×0 character vector.

### Does `cellstr` copy existing cell arrays?
Yes. The builtin returns a fresh cell array where each element is normalised to a character vector.
Elements that are already character vectors are cloned so that downstream code can modify the
result without mutating the source cell.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::cells::core::cellstr")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cellstr",
    op_kind: GpuOpKind::Custom("text-convert"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Host-only text conversion. Inputs originating on the GPU are gathered before processing, and the output is always a host cell array.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::cells::core::cellstr")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cellstr",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "Terminates fusion because the result is a host-resident cell array of character vectors.",
};

#[runtime_builtin(
    name = "cellstr",
    category = "cells/core",
    summary = "Convert text to a cell array of character vectors.",
    keywords = "cellstr,text,character,string,conversion",
    accel = "gather",
    builtin_path = "crate::builtins::cells::core::cellstr"
)]
fn cellstr_builtin(value: Value) -> Result<Value, String> {
    let host = gather_if_needed(&value).map_err(|e| format!("cellstr: {e}"))?;
    match host {
        Value::CharArray(ca) => cellstr_from_char_array(ca),
        Value::StringArray(sa) => cellstr_from_string_array(sa),
        Value::String(text) => cellstr_from_string(text),
        Value::Cell(cell) => cellstr_from_cell(cell),
        Value::LogicalArray(_)
        | Value::Bool(_)
        | Value::Int(_)
        | Value::Num(_)
        | Value::Tensor(_)
        | Value::Complex(_, _)
        | Value::ComplexTensor(_)
        | Value::Struct(_)
        | Value::Object(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::ClassRef(_)
        | Value::MException(_) => Err(ERR_INPUT_NOT_TEXT.to_string()),
        Value::GpuTensor(_) => {
            Err("cellstr: input must be gathered to the host before conversion".to_string())
        }
    }
}

fn cellstr_from_string(text: String) -> Result<Value, String> {
    let row = Value::CharArray(CharArray::new_row(&text));
    make_cell(vec![row], 1, 1).map_err(|e| format!("cellstr: {e}"))
}

fn cellstr_from_char_array(ca: CharArray) -> Result<Value, String> {
    let rows = ca.rows;
    let cols = ca.cols;
    if rows == 0 {
        return make_cell(Vec::new(), 0, 1).map_err(|e| format!("cellstr: {e}"));
    }
    let mut values = Vec::with_capacity(rows);
    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let slice = &ca.data[start..end];
        let trimmed = trim_trailing_spaces(slice);
        values.push(Value::CharArray(CharArray::new_row(&trimmed)));
    }
    make_cell(values, rows, 1).map_err(|e| format!("cellstr: {e}"))
}

fn cellstr_from_string_array(sa: StringArray) -> Result<Value, String> {
    let shape = if sa.shape.is_empty() {
        vec![sa.rows.max(1), sa.cols.max(1)]
    } else {
        sa.shape.clone()
    };
    let total = shape.iter().product::<usize>();
    if total == 0 {
        return make_cell_with_shape(Vec::new(), shape).map_err(|e| format!("cellstr: {e}"));
    }
    if total != sa.data.len() {
        return Err("cellstr: internal string array shape mismatch".to_string());
    }
    let mut values = Vec::with_capacity(total);
    for row_major in 0..total {
        let coords = linear_to_multi_row_major(row_major, &shape);
        let column_major = multi_to_linear_column_major(&coords, &shape);
        let text = sa.data[column_major].clone();
        values.push(Value::CharArray(CharArray::new_row(&text)));
    }
    make_cell_with_shape(values, shape).map_err(|e| format!("cellstr: {e}"))
}

fn cellstr_from_cell(cell: CellArray) -> Result<Value, String> {
    let mut values = Vec::with_capacity(cell.data.len());
    for ptr in &cell.data {
        let element = unsafe { &*ptr.as_raw() };
        let gathered = gather_if_needed(element).map_err(|e| format!("cellstr: {e}"))?;
        values.push(coerce_to_char_vector(gathered)?);
    }
    make_cell_with_shape(values, cell.shape.clone()).map_err(|e| format!("cellstr: {e}"))
}

fn coerce_to_char_vector(value: Value) -> Result<Value, String> {
    match value {
        Value::CharArray(ca) => {
            if ca.rows == 1 || (ca.rows == 0 && ca.cols == 0) {
                Ok(Value::CharArray(ca))
            } else {
                Err(ERR_CELL_CONTENT_NOT_TEXT.to_string())
            }
        }
        Value::String(text) => Ok(Value::CharArray(CharArray::new_row(&text))),
        Value::StringArray(sa) => {
            if sa.data.len() == 1 {
                Ok(Value::CharArray(CharArray::new_row(&sa.data[0])))
            } else {
                Err(ERR_CELL_CONTENT_NOT_TEXT.to_string())
            }
        }
        Value::Num(_)
        | Value::Int(_)
        | Value::Bool(_)
        | Value::Tensor(_)
        | Value::LogicalArray(_)
        | Value::Complex(_, _)
        | Value::ComplexTensor(_)
        | Value::GpuTensor(_) => Err(ERR_CELL_CONTENT_NOT_TEXT.to_string()),
        Value::Cell(_) | Value::Struct(_) | Value::Object(_) | Value::HandleObject(_) => {
            Err(ERR_CELL_CONTENT_NOT_TEXT.to_string())
        }
        other => Err(format!("cellstr: unsupported cell element {other:?}")),
    }
}

fn trim_trailing_spaces(chars: &[char]) -> String {
    let mut end = chars.len();
    while end > 0 && chars[end - 1] == ' ' {
        end -= 1;
    }
    chars[..end].iter().collect()
}

fn linear_to_multi_row_major(mut index: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut coords = vec![0usize; shape.len()];
    for (dim, &extent) in shape.iter().enumerate().rev() {
        if extent == 0 {
            coords[dim] = 0;
        } else {
            coords[dim] = index % extent;
            index /= extent;
        }
    }
    coords
}

fn multi_to_linear_column_major(coords: &[usize], shape: &[usize]) -> usize {
    let mut stride = 1usize;
    let mut index = 0usize;
    for (dim, &coord) in coords.iter().enumerate() {
        let extent = shape[dim];
        if extent == 0 {
            return 0;
        }
        index += coord * stride;
        stride *= extent;
    }
    index
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;

    fn cell_to_strings(cell: &CellArray) -> Vec<String> {
        cell.data
            .iter()
            .map(|ptr| match unsafe { &*ptr.as_raw() } {
                Value::CharArray(ca) => ca.data.iter().collect(),
                other => panic!("expected CharArray in cell, found {other:?}"),
            })
            .collect()
    }

    #[test]
    fn converts_char_matrix_and_trims() {
        let data: Vec<char> = vec!['c', 'a', 't', ' ', 'd', 'o', 'g', ' ', 'f', 'o', 'x', ' '];
        let ca = CharArray::new(data, 3, 4).expect("char array");
        let result = cellstr_builtin(Value::CharArray(ca)).expect("cellstr");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 3);
                assert_eq!(cell.cols, 1);
                let rows = cell_to_strings(&cell);
                assert_eq!(
                    rows,
                    vec!["cat".to_string(), "dog".to_string(), "fox".to_string()]
                );
            }
            other => panic!("expected cell result, got {other:?}"),
        }
    }

    #[test]
    fn converts_string_array_with_shape() {
        let data = vec![
            "north".to_string(),
            "south".to_string(),
            "east".to_string(),
            "west".to_string(),
        ];
        let sa = StringArray::new(data, vec![2, 2]).expect("string array");
        let result = cellstr_builtin(Value::StringArray(sa)).expect("cellstr");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 2);
                assert_eq!(cell.cols, 2);
                let rows = cell_to_strings(&cell);
                assert_eq!(
                    rows,
                    vec![
                        "north".to_string(),
                        "east".to_string(),
                        "south".to_string(),
                        "west".to_string(),
                    ]
                );
            }
            other => panic!("expected cell result, got {other:?}"),
        }
    }

    #[test]
    fn converts_string_scalar() {
        let result = cellstr_builtin(Value::String("RunMat".to_string())).expect("cellstr");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 1);
                let rows = cell_to_strings(&cell);
                assert_eq!(rows, vec!["RunMat".to_string()]);
            }
            other => panic!("expected cell result, got {other:?}"),
        }
    }

    #[test]
    fn normalises_cell_elements() {
        let alpha = Value::CharArray(CharArray::new_row("alpha"));
        let beta = Value::String("beta".to_string());
        let cell = crate::make_cell(vec![alpha, beta], 1, 2).expect("cell");
        let result = cellstr_builtin(cell).expect("cellstr");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 2);
                let rows = cell_to_strings(&cell);
                assert_eq!(rows, vec!["alpha".to_string(), "beta".to_string()]);
            }
            other => panic!("expected cell result, got {other:?}"),
        }
    }

    #[test]
    fn rejects_non_text_cell_element() {
        let cell = crate::make_cell(vec![Value::Num(1.0)], 1, 1).expect("cell");
        let err = cellstr_builtin(cell).expect_err("expected error");
        assert!(err.contains("cell array elements must be"));
    }

    #[test]
    fn rejects_multirow_char_element() {
        let ca = CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).expect("char array");
        let cell = crate::make_cell(vec![Value::CharArray(ca)], 1, 1).expect("cell");
        let err = cellstr_builtin(cell).expect_err("expected error");
        assert!(err.contains("cell array elements must be"));
    }

    #[test]
    fn rejects_non_text_input() {
        let err = cellstr_builtin(Value::Num(std::f64::consts::PI)).expect_err("expected error");
        assert!(err.contains("input must be"));
    }

    #[test]
    fn handles_empty_char_array() {
        let ca = CharArray::new(Vec::new(), 0, 5).expect("empty char");
        let result = cellstr_builtin(Value::CharArray(ca)).expect("cellstr");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 0);
                assert_eq!(cell.cols, 1);
                assert!(cell.data.is_empty());
            }
            other => panic!("expected cell result, got {other:?}"),
        }
    }

    #[test]
    fn char_row_of_spaces_becomes_empty_vector() {
        let ca = CharArray::new(vec![' '; 3], 1, 3).expect("char array");
        let result = cellstr_builtin(Value::CharArray(ca)).expect("cellstr");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 1);
                match unsafe { &*cell.data[0].as_raw() } {
                    Value::CharArray(row) => {
                        assert_eq!(row.rows, 1);
                        assert_eq!(row.cols, 0);
                        assert!(row.data.is_empty());
                    }
                    other => panic!("expected CharArray, got {other:?}"),
                }
            }
            other => panic!("expected cell result, got {other:?}"),
        }
    }

    #[test]
    fn cell_elements_preserve_trailing_spaces() {
        let ca = CharArray::new(vec!['a', ' ', ' '], 1, 3).expect("char array");
        let cell = crate::make_cell(vec![Value::CharArray(ca.clone())], 1, 1).expect("cell");
        let result = cellstr_builtin(cell).expect("cellstr");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 1);
                match unsafe { &*cell.data[0].as_raw() } {
                    Value::CharArray(row) => {
                        assert_eq!(row.rows, ca.rows);
                        assert_eq!(row.cols, ca.cols);
                        assert_eq!(row.data, ca.data);
                    }
                    other => panic!("expected CharArray, got {other:?}"),
                }
            }
            other => panic!("expected cell result, got {other:?}"),
        }
    }

    #[test]
    fn string_array_missing_value_converts() {
        let sa = StringArray::new(vec!["<missing>".to_string()], vec![1, 1]).expect("string array");
        let result = cellstr_builtin(Value::StringArray(sa)).expect("cellstr");
        match result {
            Value::Cell(cell) => {
                let rows = cell_to_strings(&cell);
                assert_eq!(rows, vec!["<missing>".to_string()]);
            }
            other => panic!("expected cell result, got {other:?}"),
        }
    }

    #[test]
    fn empty_string_array_produces_empty_cell_shape() {
        let sa = StringArray::new(Vec::new(), vec![0, 2]).expect("string array");
        let result = cellstr_builtin(Value::StringArray(sa)).expect("cellstr");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 0);
                assert_eq!(cell.cols, 2);
                assert!(cell.data.is_empty());
            }
            other => panic!("expected cell result, got {other:?}"),
        }
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
