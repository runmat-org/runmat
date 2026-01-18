//! MATLAB-compatible `regexpi` builtin for RunMat.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::strings::regex::regexp::{self, RegexpEvaluation};
use crate::{build_runtime_error, make_cell, BuiltinResult, RuntimeError};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "regexpi",
        builtin_path = "crate::builtins::strings::regex::regexpi"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "regexpi"
category: "strings/regex"
keywords: ["regexpi", "regular expression", "ignore case", "pattern", "match"]
summary: "Perform case-insensitive regular expression matching with MATLAB-compatible outputs."
references:
  - https://www.mathworks.com/help/matlab/ref/regexpi.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "none"
  notes: "Runs on the CPU. When inputs reside on the GPU, RunMat gathers them before matching and returns host-side containers."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 0
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::strings::regex::regexpi::tests"
  integration: "builtins::strings::regex::regexpi::tests::regexpi_builtin_match_output"
---

# What does the `regexpi` function do in MATLAB / RunMat?
`regexpi(text, pattern)` evaluates regular expression matches while ignoring case by default.
Outputs mirror MATLAB: you can retrieve 1-based match indices, substrings, capture tokens, token
extents, named tokens, or the text split around matches. Flags such as `'once'`, `'tokens'`,
`'match'`, `'split'`, `'tokenExtents'`, `'names'`, `'emptymatch'`, and `'forceCellOutput'`
are supported, together with case toggles (`'ignorecase'`, `'matchcase'`) and newline behaviour
(`'dotall'`, `'dotExceptNewline'`, `'lineanchors'`).

## How does the `regexpi` function behave in MATLAB / RunMat?
- Case-insensitive matching is the default; include `'matchcase'` when you need case-sensitive
  behaviour.
- With one output, `regexpi` returns a numeric row vector of 1-based match start indices.
- With multiple outputs, the default order is match starts, match ends, matched substrings.
- When the input is a string array or cell array of character vectors, outputs are cell arrays whose
  shape matches the input container.
- `'forceCellOutput'` forces cell outputs even for scalar inputs, matching MATLAB semantics.
- `'once'` limits each element to its first match, influencing every requested output.
- `'emptymatch','allow'` keeps zero-length matches; `'emptymatch','remove'` is the default filter.
- Named tokens (using `(?<name>...)`) return scalar struct values per match when `'names'` is
  requested. Unmatched names resolve to empty strings for MATLAB compatibility.

## `regexpi` Function GPU Execution Behaviour
`regexpi` executes entirely on the CPU. If inputs or previously computed intermediates are resident
on the GPU, RunMat gathers the necessary data before evaluation and returns host-side outputs.
Acceleration providers do not offer specialised hooks today; computed tensors remain on the host
unless explicit GPU transfers are requested later.

## Examples of using the `regexpi` function in MATLAB / RunMat

### Finding indices regardless of case
```matlab
idx = regexpi('Abracadabra', 'a');
```
Expected output:
```matlab
idx =
     1     4     6     8    11
```

### Returning matched substrings ignoring case
```matlab
matches = regexpi('abcXYZ123', '[a-z]{3}', 'match');
```
Expected output:
```matlab
matches =
  1×2 cell array
    {'abc'}    {'XYZ'}
```

### Extracting capture tokens case-insensitively
```matlab
tokens = regexpi('ID:AB12', '(?<prefix>[a-z]+)(?<digits>\d+)', 'tokens');
first = tokens{1}{1};
second = tokens{1}{2};
```
Expected output:
```matlab
first =
    'AB'
second =
    '12'
```

### Limiting `regexpi` to the first match
```matlab
firstMatch = regexpi('aXaXaX', 'ax', 'match', 'once');
```
Expected output:
```matlab
firstMatch =
    'aX'
```

### Splitting a string array without worrying about letter case
```matlab
parts = regexpi(["Color:Red"; "COLOR:Blue"], 'color:', 'split');
```
Expected output:
```matlab
parts =
  2×1 cell array
    {1×2 cell}
    {1×2 cell}

parts{2}{2}
ans =
    'Blue'
```

### Enforcing case-sensitive matches with `'matchcase'`
```matlab
idx = regexpi('CaseTest', 'case', 'matchcase');
```
Expected output:
```matlab
idx =
     []
```

## FAQ

### How are the outputs ordered when I request several?
If you do not specify explicit output flags, the default order is match starts, match ends, and
matched substrings—identical to MATLAB. Providing flags such as `'match'` or `'tokens'` returns only
the requested outputs.

### Can I make `regexpi` behave like `regexp` with case sensitivity?
Yes. Include the `'matchcase'` flag to disable the default case-insensitive mode. You can also pass
`'ignorecase'` explicitly to emphasise the default.

### Does `regexpi` support string arrays and cell arrays?
Yes. Outputs mirror the input container shape, and each element stores results for the corresponding
string or character vector.

### How do zero-length matches behave?
By default (`'emptymatch','remove'`), zero-length matches are omitted. Use
`'emptymatch','allow'` to keep them, which is helpful when inspecting optional pattern components.

### Does `regexpi` run on the GPU?
No. All matching occurs on the CPU. RunMat gathers GPU-resident inputs before processing and leaves
outputs on the host. Explicit `gpuArray` calls are required if you want to move the results back to
the GPU.

### Are named tokens supported?
Yes. Use the `(?<name>...)` syntax and request the `'names'` output flag. Each match produces a
scalar struct with fields for every named group.

### What happens with `'once'`?
`'once'` restricts each input element to the first match. All requested outputs honour that limit,
returning scalars instead of per-match cells.

### Can I keep scalar outputs in cells?
Yes. Pass `'forceCellOutput'` to wrap even scalar results in cells, which is useful when writing code
that must treat scalar and array inputs uniformly.

## See Also
`regexp`, `regexprep`, `contains`, `split`, `strfind`

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/strings/regex/regexpi.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/regex/regexpi.rs)
- Found a behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::regex::regexpi")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "regexpi",
    op_kind: GpuOpKind::Custom("regex"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Executes on the CPU; GPU inputs are gathered before evaluation and results stay on the host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::regex::regexpi")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "regexpi",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Control-flow-heavy regex evaluation is not eligible for fusion.",
};

const BUILTIN_NAME: &str = "regexpi";

fn runtime_error_for(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

/// Evaluate `regexpi` with MATLAB-compatible defaults and return the shared regex evaluation handle.
pub fn evaluate(
    subject: Value,
    pattern: Value,
    rest: &[Value],
) -> BuiltinResult<RegexpEvaluation> {
    let options = build_options(rest);
    regexp::evaluate_with(BUILTIN_NAME, subject, pattern, &options)
}

#[runtime_builtin(
    name = "regexpi",
    category = "strings/regex",
    summary = "Case-insensitive regular expression matching with MATLAB-compatible outputs.",
    keywords = "regexpi,regex,pattern,ignorecase,match",
    accel = "sink",
    builtin_path = "crate::builtins::strings::regex::regexpi"
)]
fn regexpi_builtin(subject: Value, pattern: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let evaluation = evaluate(subject, pattern, &rest)?;
    let mut outputs = evaluation.outputs_for_single()?;
    if outputs.is_empty() {
        return Ok(Value::Num(0.0));
    }
    if outputs.len() == 1 {
        Ok(outputs.remove(0))
    } else {
        let len = outputs.len();
        make_cell(outputs, 1, len)
            .map_err(|err| runtime_error_for(format!("{BUILTIN_NAME}: {err}")))
    }
}

fn build_options(rest: &[Value]) -> Vec<Value> {
    let mut options: Vec<Value> = rest.to_vec();
    if !has_case_directive(rest) {
        options.push(Value::String("ignorecase".into()));
    }
    options
}

fn has_case_directive(values: &[Value]) -> bool {
    values.iter().any(|value| {
        matches!(
            option_name(value).as_deref(),
            Some("ignorecase") | Some("matchcase")
        )
    })
}

fn option_name(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.to_ascii_lowercase()),
        Value::StringArray(sa) if sa.data.len() == 1 => Some(sa.data[0].to_ascii_lowercase()),
        Value::CharArray(ca) if ca.rows == 1 => {
            Some(ca.data.iter().collect::<String>().to_ascii_lowercase())
        }
        _ => None,
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use runmat_builtins::{CellArray, StringArray};

    use crate::builtins::common::test_support;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexpi_default_is_case_insensitive() {
        let eval = evaluate(
            Value::String("Abracadabra".into()),
            Value::String("a".into()),
            &[],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Tensor(t) => {
                assert_eq!(t.data, vec![1.0, 4.0, 6.0, 8.0, 11.0]);
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexpi_match_output_ignores_case() {
        let eval = evaluate(
            Value::String("abcXYZ123".into()),
            Value::String("[a-z]+".into()),
            &[Value::String("match".into())],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Cell(ca) => {
                assert_eq!(ca.data.len(), 1);
                let first = unsafe { &*ca.data[0].as_raw() };
                assert_eq!(first, &Value::String("abcXYZ".into()));
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexpi_matchcase_overrides_default() {
        let eval = evaluate(
            Value::String("CaseTest".into()),
            Value::String("case".into()),
            &[Value::String("matchcase".into())],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Tensor(t) => assert!(t.data.is_empty()),
            Value::Num(n) => assert_eq!(*n, 0.0),
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexpi_builtin_match_output() {
        let result = regexpi_builtin(
            Value::String("FooBarBaz".into()),
            Value::String("bar".into()),
            vec![Value::String("match".into())],
        )
        .unwrap();
        match result {
            Value::Cell(ca) => {
                assert_eq!(ca.data.len(), 1);
                let entry = unsafe { &*ca.data[0].as_raw() };
                assert_eq!(entry, &Value::String("Bar".into()));
            }
            other => panic!("unexpected builtin output {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexpi_tokens_once_returns_structured_cells() {
        let eval = evaluate(
            Value::String("ID:AB12".into()),
            Value::String("(?<prefix>[a-z]+)(?<digits>\\d+)".into()),
            &[
                Value::String("tokens".into()),
                Value::String("names".into()),
                Value::String("once".into()),
            ],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 2);
        match &outputs[0] {
            Value::Cell(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 2);
                let first = unsafe { &*ca.data[0].as_raw() };
                let second = unsafe { &*ca.data[1].as_raw() };
                assert_eq!(first, &Value::String("AB".into()));
                assert_eq!(second, &Value::String("12".into()));
            }
            other => panic!("unexpected tokens output {other:?}"),
        }
        match &outputs[1] {
            Value::Struct(st) => {
                assert_eq!(st.fields.len(), 2);
                assert_eq!(st.fields.get("prefix"), Some(&Value::String("AB".into())));
                assert_eq!(st.fields.get("digits"), Some(&Value::String("12".into())));
            }
            other => panic!("unexpected names output {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexpi_force_cell_output_for_scalar_subject() {
        let eval = evaluate(
            Value::String("Hello".into()),
            Value::String("l".into()),
            &[
                Value::String("forcecelloutput".into()),
                Value::String("match".into()),
            ],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Cell(ca) => {
                assert_eq!(ca.rows, 1);
                assert_eq!(ca.cols, 1);
                let cell = unsafe { &*ca.data[0].as_raw() };
                match cell {
                    Value::Cell(inner) => {
                        assert_eq!(inner.data.len(), 2);
                        let first = unsafe { &*inner.data[0].as_raw() };
                        let second = unsafe { &*inner.data[1].as_raw() };
                        assert_eq!(first, &Value::String("l".into()));
                        assert_eq!(second, &Value::String("l".into()));
                    }
                    other => panic!("unexpected nested value {other:?}"),
                }
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexpi_token_extents_provide_indices() {
        let eval = evaluate(
            Value::String("ID:AB12".into()),
            Value::String("([A-Z]+)(\\d+)".into()),
            &[Value::String("tokenExtents".into())],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Cell(ca) => {
                assert_eq!(ca.data.len(), 1);
                let matrix = unsafe { &*ca.data[0].as_raw() };
                match matrix {
                    Value::Tensor(t) => {
                        assert_eq!(t.shape, vec![2, 2]);
                        assert_eq!(t.data, vec![4.0, 6.0, 5.0, 7.0]);
                    }
                    other => panic!("expected tensor for token extents, got {other:?}"),
                }
            }
            other => panic!("unexpected token extents output {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexpi_split_returns_segments() {
        let eval = evaluate(
            Value::String("Red,Green,BLUE".into()),
            Value::String(",".into()),
            &[Value::String("split".into())],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Cell(ca) => {
                assert_eq!(ca.data.len(), 3);
                let parts: Vec<String> = ca
                    .data
                    .iter()
                    .map(|ptr| match unsafe { &*ptr.as_raw() } {
                        Value::String(s) => s.clone(),
                        other => panic!("expected string split part, got {other:?}"),
                    })
                    .collect();
                assert_eq!(parts, vec!["Red", "Green", "BLUE"]);
            }
            other => panic!("unexpected split output {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexpi_emptymatch_allow_keeps_zero_length_matches() {
        let eval = evaluate(
            Value::String("aba".into()),
            Value::String("b*".into()),
            &[
                Value::String("emptymatch".into()),
                Value::String("allow".into()),
            ],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Tensor(t) => assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0]),
            other => panic!("expected tensor with match indices, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexpi_string_array_preserves_shape() {
        let array =
            StringArray::new(vec!["OneTwo".into(), "THREEfour".into()], vec![2, 1]).unwrap();
        let eval = evaluate(
            Value::StringArray(array),
            Value::String("[a-z]+".into()),
            &[Value::String("match".into())],
        )
        .unwrap();
        let outputs = eval.outputs_for_single().unwrap();
        assert_eq!(outputs.len(), 1);
        match &outputs[0] {
            Value::Cell(ca) => {
                assert_eq!(ca.rows, 2);
                assert_eq!(ca.cols, 1);
                assert_eq!(ca.data.len(), 2);
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn regexpi_cell_array_inputs_require_all_strings() {
        let handles = vec![
            unsafe {
                runmat_gc_api::GcPtr::from_raw(Box::into_raw(Box::new(Value::String("A".into()))))
            },
            unsafe { runmat_gc_api::GcPtr::from_raw(Box::into_raw(Box::new(Value::Num(1.0)))) },
        ];
        let cell = CellArray::new_handles(handles, 2, 1).unwrap();
        let err = evaluate(Value::Cell(cell), Value::String("a".into()), &[])
            .err()
            .expect("expected regexpi to reject non-text cell elements");
        let message = err.message().to_string();
        assert!(
            message.contains("cell array elements"),
            "unexpected error message: {message}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
