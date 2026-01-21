//! MATLAB-compatible `strfind` builtin for RunMat.

use runmat_builtins::{CellArray, Tensor, Value};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};

use super::text_utils::{value_to_owned_string, TextCollection, TextElement};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "strfind",
        builtin_path = "crate::builtins::strings::search::strfind"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "strfind"
category: "strings/search"
keywords: ["strfind", "substring", "index", "positions", "forcecelloutput"]
summary: "Locate the starting indices of pattern matches in text inputs, returning doubles or cell arrays like MATLAB."
references:
  - https://www.mathworks.com/help/matlab/ref/strfind.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "matlab"
  notes: "Executes on the CPU. When inputs reside on the GPU, RunMat gathers them automatically before searching."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::strings::search::strfind::tests"
  integration: "builtins::strings::search::strfind::tests::strfind_subject_cell_scalar_returns_cell"
---

# What does the `strfind` function do in MATLAB / RunMat?
`k = strfind(str, pattern)` returns the starting indices of every occurrence of `pattern`
inside `str`. The builtin mirrors MATLAB semantics for character vectors, string scalars,
string arrays, and cell arrays of character vectors.

## How does the `strfind` function behave in MATLAB / RunMat?
- Accepts text inputs as string scalars/arrays, character vectors/arrays, or cell arrays of
  character vectors. Mixed combinations are permitted.
- Applies MATLAB-style implicit expansion when either argument is non-scalar.
- Returns a numeric row vector when both inputs are scalar text (character vectors or string
  scalars) and `'ForceCellOutput'` is `false`. If either input is a cell array or the broadcasted
  result contains multiple elements, the output is a cell array with the broadcasted shape, and
  each cell contains a row vector of double indices.
- Pattern matches are case-sensitive and may overlap. For example, `strfind("aaaa","aa")`
  yields `[1 2 3]`.
- An empty pattern matches the boundaries between characters, producing indices
  `1:length(str)+1`. Missing strings (`<missing>`) never match any pattern.
- Specify `'ForceCellOutput', true` to always obtain a cell array, even for scalar inputs.

## `strfind` Function GPU Execution Behaviour
`strfind` performs substring searches on the host CPU. When its inputs currently live on the
GPU (for example, after other accelerated operations), RunMat gathers them automatically before
executing the search so the behaviour matches MATLAB exactly. No provider hooks are required.

## Examples of using the `strfind` function in MATLAB / RunMat

### Find substring positions in a character vector
```matlab
idx = strfind('abracadabra', 'abra');
```
Expected output:
```matlab
idx = [1 8];
```

### Locate overlapping matches
```matlab
idx = strfind("aaaa", "aa");
```
Expected output:
```matlab
idx = [1 2 3];
```

### Return matches for each element of a string array
```matlab
words = ["hydrogen"; "helium"; "lithium"];
idx = strfind(words, "i");
```
Expected output:
```matlab
idx = 3×1 cell array
    {[]}
    {[4]}
    {[2 5]}
```

### Search with multiple patterns against one subject
```matlab
idx = strfind("saturn", ["sat", "turn", "moon"]);
```
Expected output:
```matlab
idx = 1×3 cell array
    {[1]}    {[3]}    {[]}
```

### Force cell output for scalar inputs
```matlab
idx = strfind('mission', 's', 'ForceCellOutput', true);
```
Expected output:
```matlab
idx = 1×1 cell array
    {[3 4]}
```

### Handle empty patterns and missing strings
```matlab
idxEmpty = strfind("abc", "");
idxMissing = strfind("<missing>", "abc");
```
Expected output:
```matlab
idxEmpty = [1 2 3 4];
idxMissing = [];
```

## GPU residency in RunMat (Do I need `gpuArray`?)

You usually do NOT need to call `gpuArray` yourself in RunMat (unlike MATLAB).

`strfind` always executes on the host, but if you pass in GPU-resident values RunMat gathers
them automatically before performing the search. This keeps the results identical to MATLAB
while still allowing upstream computations to benefit from acceleration.

## FAQ

### What types can I pass to `strfind`?
Use string scalars/arrays, character vectors/arrays, or cell arrays of character vectors for
either argument. Mixed combinations are allowed; MATLAB-style implicit expansion aligns the
inputs automatically.

### How are the results formatted?
When both inputs are scalar text (character vectors or string scalars) and `'ForceCellOutput'`
is `false`, `strfind` returns a numeric row vector with the starting indices. In all other cases
— for example, if either input is a cell array or the broadcasted size exceeds one element —
the result is a cell array whose shape matches the broadcasted size, with each cell containing
a row vector of doubles.

### Do matches overlap?
Yes. The builtin considers every occurrence of the pattern, including overlapping matches.

### What happens with empty patterns?
An empty pattern matches the gaps between characters, so the indices `1:length(str)+1` are
returned for non-missing text. When the subject is missing, the result is an empty array.

### How do I always get a cell array?
Supply `'ForceCellOutput', true`. This mirrors MATLAB and is useful when you want consistent
cell outputs regardless of input sizes.

### Does `strfind` support case-insensitive matching?
No. `strfind` is case-sensitive like MATLAB. Use `contains`, `startsWith`, or the regular
expression functions when you need case-insensitive searches.

## See Also
[`contains`](./contains), [`startsWith`](./startswith), [`endsWith`](./endswith), [`regexp`](./regexp)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/strings/search/strfind.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/search/strfind.rs)
- Found a bug? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::search::strfind")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "strfind",
    op_kind: GpuOpKind::Custom("string-search"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Executes entirely on the host; GPU-resident inputs are gathered before substring matching.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::search::strfind")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "strfind",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Text operation; not eligible for fusion and materialises host-side numeric or cell outputs.",
};

const BUILTIN_NAME: &str = "strfind";

#[runtime_builtin(
    name = "strfind",
    category = "strings/search",
    summary = "Return the starting indices of pattern matches in text inputs.",
    keywords = "strfind,substring,index,positions,string search",
    accel = "sink",
    builtin_path = "crate::builtins::strings::search::strfind"
)]
async fn strfind_builtin(
    text: Value,
    pattern: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let text = gather_if_needed_async(&text).await?;
    let pattern = gather_if_needed_async(&pattern).await?;
    let force_cell_output = parse_force_cell_output(&rest)?;

    let subject = TextCollection::from_subject(BUILTIN_NAME, text)?;
    let patterns = TextCollection::from_pattern(BUILTIN_NAME, pattern)?;

    evaluate_strfind(&subject, &patterns, force_cell_output)
}

fn evaluate_strfind(
    subject: &TextCollection,
    patterns: &TextCollection,
    force_cell_output: bool,
) -> BuiltinResult<Value> {
    let output_shape =
        broadcast_shapes(BUILTIN_NAME, &subject.shape, &patterns.shape).map_err(strfind_error)?;
    let total = tensor::element_count(&output_shape);
    let return_cell = force_cell_output || subject.is_cell || patterns.is_cell || total != 1;

    let subject_strides = compute_strides(&subject.shape);
    let pattern_strides = compute_strides(&patterns.shape);

    let mut matches: Vec<Vec<usize>> = Vec::with_capacity(total);
    for linear in 0..total {
        let subject_idx = broadcast_index(linear, &output_shape, &subject.shape, &subject_strides);
        let pattern_idx = broadcast_index(linear, &output_shape, &patterns.shape, &pattern_strides);
        let result = match (
            &subject.elements[subject_idx],
            &patterns.elements[pattern_idx],
        ) {
            (TextElement::Missing, _) => Vec::new(),
            (_, TextElement::Missing) => Vec::new(),
            (TextElement::Text(text), TextElement::Text(pattern)) => {
                find_indices(text, pattern.as_str())
            }
        };
        matches.push(result);
    }

    if !return_cell {
        let indices = matches.into_iter().next().unwrap_or_default();
        return indices_to_numeric_value(&indices);
    }

    indices_to_cell(matches, &output_shape)
}

fn find_indices(text: &str, pattern: &str) -> Vec<usize> {
    if pattern.is_empty() {
        let len = text.chars().count();
        return (0..=len).map(|pos| pos + 1).collect();
    }

    let text_chars: Vec<char> = text.chars().collect();
    let pattern_chars: Vec<char> = pattern.chars().collect();
    let text_len = text_chars.len();
    let pattern_len = pattern_chars.len();

    if pattern_len == 0 || pattern_len > text_len {
        return Vec::new();
    }

    let mut indices = Vec::new();
    for start in 0..=text_len - pattern_len {
        if &text_chars[start..start + pattern_len] == pattern_chars.as_slice() {
            indices.push(start + 1);
        }
    }
    indices
}

fn indices_to_numeric_value(indices: &[usize]) -> BuiltinResult<Value> {
    let data = indices.iter().map(|&pos| pos as f64).collect::<Vec<_>>();
    let cols = indices.len();
    Tensor::new(data, vec![1, cols])
        .map(Value::Tensor)
        .map_err(|e| strfind_error(format!("{BUILTIN_NAME}: {e}")))
}

fn indices_to_tensor(indices: &[usize]) -> BuiltinResult<Value> {
    Tensor::new(
        indices.iter().map(|&pos| pos as f64).collect::<Vec<_>>(),
        vec![1, indices.len()],
    )
    .map(Value::Tensor)
    .map_err(|e| strfind_error(format!("{BUILTIN_NAME}: {e}")))
}

fn indices_to_cell(matches: Vec<Vec<usize>>, shape: &[usize]) -> BuiltinResult<Value> {
    let total = matches.len();
    if total == 0 {
        let (rows, cols) = shape_to_rows_cols(shape);
        return CellArray::new(Vec::new(), rows, cols)
            .map(Value::Cell)
            .map_err(|e| strfind_error(format!("{BUILTIN_NAME}: {e}")));
    }

    let (rows, cols) = shape_to_rows_cols(shape);
    if rows * cols != total {
        return Err(strfind_error(
            "strfind: internal size mismatch while constructing cell output",
        ));
    }

    let mut values = Vec::with_capacity(total);
    for row in 0..rows {
        for col in 0..cols {
            let column_major_idx = row + rows * col;
            let indices = matches
                .get(column_major_idx)
                .ok_or_else(|| strfind_error("strfind: internal indexing error"))?;
            let cell_value = indices_to_tensor(indices)?;
            values.push(cell_value);
        }
    }

    CellArray::new(values, rows, cols)
        .map(Value::Cell)
        .map_err(|e| strfind_error(format!("{BUILTIN_NAME}: {e}")))
}

fn shape_to_rows_cols(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => (shape[0], 1),
        _ => {
            let rows = shape[0];
            let cols = shape[1..]
                .iter()
                .copied()
                .fold(1usize, |acc, dim| acc.saturating_mul(dim));
            (rows, cols)
        }
    }
}

fn strfind_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

fn parse_force_cell_output(rest: &[Value]) -> BuiltinResult<bool> {
    if rest.is_empty() {
        return Ok(false);
    }
    if !rest.len().is_multiple_of(2) {
        return Err(strfind_error(
            "strfind: expected name-value pairs after the pattern (e.g., 'ForceCellOutput', true)",
        ));
    }

    let mut force_cell = None;
    for pair in rest.chunks(2) {
        let name = value_to_owned_string(&pair[0])
            .ok_or_else(|| strfind_error("strfind: option names must be text scalars"))?;
        if !name.eq_ignore_ascii_case("forcecelloutput") {
            return Err(strfind_error(format!(
                "strfind: unknown option '{name}'; supported option is 'ForceCellOutput'"
            )));
        }
        let value = parse_bool_like(&pair[1])?;
        force_cell = Some(value);
    }
    force_cell.ok_or_else(|| {
        strfind_error(
            "strfind: expected 'ForceCellOutput' option when providing name-value arguments",
        )
    })
}

fn parse_bool_like(value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(b) => Ok(*b),
        Value::Int(i) => Ok(!i.is_zero()),
        Value::Num(n) => {
            if !n.is_finite() {
                Err(strfind_error(
                    "strfind: option values must be finite numeric scalars",
                ))
            } else {
                Ok(*n != 0.0)
            }
        }
        Value::LogicalArray(array) => {
            if array.data.len() != 1 {
                Err(strfind_error(format!(
                    "strfind: option values must be scalar logicals (received {} elements)",
                    array.data.len()
                )))
            } else {
                Ok(array.data[0] != 0)
            }
        }
        Value::Tensor(tensor) => {
            if tensor.data.len() != 1 {
                Err(strfind_error(format!(
                    "strfind: option values must be scalar numeric values (received {} elements)",
                    tensor.data.len()
                )))
            } else if !tensor.data[0].is_finite() {
                Err(strfind_error(
                    "strfind: option values must be finite numeric scalars",
                ))
            } else {
                Ok(tensor.data[0] != 0.0)
            }
        }
        other => value_to_owned_string(other)
            .ok_or_else(|| {
                strfind_error("strfind: option values must be logical or numeric scalars")
            })
            .and_then(|text| match text.trim().to_ascii_lowercase().as_str() {
                "true" | "on" | "1" => Ok(true),
                "false" | "off" | "0" => Ok(false),
                _ => Err(strfind_error(format!(
                    "strfind: invalid value '{text}' for 'ForceCellOutput'; expected true or false"
                ))),
            }),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{CellArray, CharArray, StringArray, Tensor};

    fn run_strfind(text: Value, pattern: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(strfind_builtin(text, pattern, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_single_match_returns_row_vector() {
        let result = run_strfind(
            Value::String("saturn".into()),
            Value::String("sat".into()),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 1]);
                assert_eq!(tensor.data, vec![1.0]);
            }
            other => panic!("expected 1x1 tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_char_vector_matches() {
        let result = run_strfind(
            Value::CharArray(CharArray::new_row("abracadabra")),
            Value::CharArray(CharArray::new_row("abra")),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert_eq!(tensor.data, vec![1.0, 8.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_overlapping_matches() {
        let result = run_strfind(
            Value::String("aaaa".into()),
            Value::String("aa".into()),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 3]);
                assert_eq!(tensor.data, vec![1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_empty_pattern_returns_boundaries() {
        let result = run_strfind(
            Value::String("abc".into()),
            Value::String("".into()),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 4]);
                assert_eq!(tensor.data, vec![1.0, 2.0, 3.0, 4.0]);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_string_array_returns_cell() {
        let strings = StringArray::new(
            vec!["hydrogen".into(), "helium".into(), "lithium".into()],
            vec![3, 1],
        )
        .unwrap();
        let result = run_strfind(
            Value::StringArray(strings),
            Value::String("i".into()),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 3);
                assert_eq!(cell.cols, 1);
                let first = cell.get(0, 0).unwrap();
                let second = cell.get(1, 0).unwrap();
                let third = cell.get(2, 0).unwrap();
                match first {
                    Value::Tensor(tensor) => assert!(tensor.data.is_empty()),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
                match second {
                    Value::Tensor(tensor) => assert_eq!(tensor.data, vec![4.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
                match third {
                    Value::Tensor(tensor) => assert_eq!(tensor.data, vec![2.0, 5.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
            }
            other => panic!("expected cell result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_pattern_array_returns_cell() {
        let patterns =
            StringArray::new(vec!["sat".into(), "turn".into(), "moon".into()], vec![1, 3]).unwrap();
        let result = run_strfind(
            Value::String("saturn".into()),
            Value::StringArray(patterns),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 3);
                let first = cell.get(0, 0).unwrap();
                let second = cell.get(0, 1).unwrap();
                let third = cell.get(0, 2).unwrap();
                match first {
                    Value::Tensor(tensor) => assert_eq!(tensor.data, vec![1.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
                match second {
                    Value::Tensor(tensor) => assert_eq!(tensor.data, vec![3.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
                match third {
                    Value::Tensor(tensor) => assert!(tensor.data.is_empty()),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
            }
            other => panic!("expected cell result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_force_cell_output_name_value() {
        let result = run_strfind(
            Value::CharArray(CharArray::new_row("mission")),
            Value::CharArray(CharArray::new_row("s")),
            vec![Value::String("ForceCellOutput".into()), Value::Bool(true)],
        )
        .expect("strfind");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 1);
                match cell.get(0, 0).unwrap() {
                    Value::Tensor(tensor) => assert_eq!(tensor.data, vec![3.0, 4.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
            }
            other => panic!("expected cell output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_force_cell_output_numeric_value() {
        let result = run_strfind(
            Value::String("mission".into()),
            Value::String("s".into()),
            vec![Value::String("ForceCellOutput".into()), Value::Num(1.0)],
        )
        .expect("strfind");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 1);
                match cell.get(0, 0).unwrap() {
                    Value::Tensor(tensor) => assert_eq!(tensor.data, vec![3.0, 4.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
            }
            other => panic!("expected cell output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_force_cell_output_off_string() {
        let result = run_strfind(
            Value::String("mission".into()),
            Value::String("s".into()),
            vec![
                Value::String("ForceCellOutput".into()),
                Value::String("off".into()),
            ],
        )
        .expect("strfind");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 2]);
                assert_eq!(tensor.data, vec![3.0, 4.0]);
            }
            other => panic!("expected numeric tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_force_cell_output_non_scalar_error() {
        let option_value =
            Tensor::new(vec![1.0, 0.0], vec![1, 2]).expect("tensor construction for test");
        let err = run_strfind(
            Value::String("mission".into()),
            Value::String("s".into()),
            vec![
                Value::String("ForceCellOutput".into()),
                Value::Tensor(option_value),
            ],
        )
        .expect_err("strfind should error for non-scalar ForceCellOutput");
        assert!(
            err.to_string().contains("scalar"),
            "unexpected error message for non-scalar option: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_force_cell_output_missing_value_error() {
        let err = run_strfind(
            Value::String("mission".into()),
            Value::String("s".into()),
            vec![Value::String("ForceCellOutput".into())],
        )
        .expect_err("strfind should error when ForceCellOutput value missing");
        assert!(
            err.to_string().contains("name-value pairs"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_subject_cell_scalar_returns_cell() {
        let subject = CellArray::new(vec![Value::from("needle")], 1, 1).expect("cell construction");
        let result = run_strfind(
            Value::Cell(subject),
            Value::String("needle".into()),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 1);
                match cell.get(0, 0).unwrap() {
                    Value::Tensor(tensor) => assert_eq!(tensor.data, vec![1.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
            }
            other => panic!("expected cell output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_pattern_cell_scalar_returns_cell() {
        let pattern = CellArray::new(vec![Value::from("needle")], 1, 1).expect("cell construction");
        let result = run_strfind(
            Value::String("needle".into()),
            Value::Cell(pattern),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 1);
                assert_eq!(cell.cols, 1);
                match cell.get(0, 0).unwrap() {
                    Value::Tensor(tensor) => assert_eq!(tensor.data, vec![1.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
            }
            other => panic!("expected cell output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_missing_subject_returns_empty() {
        let result = run_strfind(
            Value::String("<missing>".into()),
            Value::String("abc".into()),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 0]);
                assert!(tensor.data.is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_missing_pattern_returns_empty_vector() {
        let patterns =
            StringArray::new(vec!["<missing>".into()], vec![1, 1]).expect("string array creation");
        let result = run_strfind(
            Value::String("planet".into()),
            Value::StringArray(patterns),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape, vec![1, 0]);
                assert!(tensor.data.is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_char_matrix_rows() {
        let data = vec!['c', 'a', 't', 'a', 'd', 'a', 'd', 'o', 'g'];
        let array = CharArray::new(data, 3, 3).unwrap();
        let result = run_strfind(
            Value::CharArray(array),
            Value::CharArray(CharArray::new_row("d")),
            Vec::new(),
        )
        .expect("strfind");
        match result {
            Value::Cell(cell) => {
                assert_eq!(cell.rows, 3);
                assert_eq!(cell.cols, 1);
                match cell.get(0, 0).unwrap() {
                    Value::Tensor(tensor) => assert!(tensor.data.is_empty()),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
                match cell.get(1, 0).unwrap() {
                    Value::Tensor(tensor) => assert_eq!(tensor.data, vec![2.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
                match cell.get(2, 0).unwrap() {
                    Value::Tensor(tensor) => assert_eq!(tensor.data, vec![1.0]),
                    other => panic!("expected tensor inside cell, got {other:?}"),
                }
            }
            other => panic!("expected cell output, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strfind_invalid_option_name_errors() {
        let err = run_strfind(
            Value::String("abc".into()),
            Value::String("a".into()),
            vec![Value::String("IgnoreCase".into()), Value::Bool(true)],
        )
        .expect_err("strfind should error");
        assert!(
            err.to_string().contains("unknown option"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
