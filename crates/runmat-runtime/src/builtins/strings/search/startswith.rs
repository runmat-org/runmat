//! MATLAB-compatible `startsWith` builtin for RunMat.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult};

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};

use super::text_utils::{logical_result, parse_ignore_case, TextCollection, TextElement};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "startsWith",
        builtin_path = "crate::builtins::strings::search::startswith"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "startsWith"
category: "strings/search"
keywords: ["startswith", "prefix", "string array", "ignorecase", "text search"]
summary: "Check whether text inputs start with specific patterns using MATLAB-compatible broadcasting and case handling."
references:
  - https://www.mathworks.com/help/matlab/ref/startswith.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "matlab"
  notes: "Runs on the CPU; GPU-resident inputs are gathered automatically so behaviour matches MATLAB."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::strings::search::startswith::tests"
  integration: "builtins::strings::search::startswith::tests::startswith_cell_array_patterns"
---

# What does the `startsWith` function do in MATLAB / RunMat?
`startsWith(str, pattern)` returns a logical result indicating whether each element of `str`
begins with the corresponding text in `pattern`. The builtin supports string arrays, character
vectors/arrays, and cell arrays of character vectors, mirroring MATLAB's implicit expansion semantics.

## How does the `startsWith` function behave in MATLAB / RunMat?
- Accepts text inputs as string scalars/arrays, character vectors/arrays, or cell arrays of character vectors.
- Accepts patterns in the same formats; scalar inputs expand across the other argument following MATLAB broadcast rules.
- Missing strings (`<missing>`) never match any pattern.
- Empty patterns (`""` or `''`) always match non-missing text elements.
- Patterns that are `<missing>` never match and therefore return `false`.
- Character arrays treat each row as an independent element; zero-row character arrays yield empty outputs.
- The optional `IgnoreCase` flag can be supplied either as a trailing scalar or via the `'IgnoreCase', value` name-value pair. It accepts logical/numeric scalars and the strings `'on'`, `'off'`, `'true'`, and `'false'` (default is case-sensitive).
- Returns a logical scalar when the broadcasted size is one element, otherwise returns a logical array.

## `startsWith` Function GPU Execution Behaviour
`startsWith` performs host-side prefix comparison. When inputs currently live on the GPU, RunMat gathers
them back to the host before evaluation so the behaviour is identical to MATLAB. No acceleration provider
hooks are required for this builtin.

## Examples of using the `startsWith` function in MATLAB / RunMat

### Determine whether a string starts with a prefix
```matlab
tf = startsWith("RunMat Accelerate", "RunMat");
```
Expected output:
```matlab
tf = logical
   1
```

### Perform a case-insensitive prefix check
```matlab
tf = startsWith("RunMat", "run", 'IgnoreCase', true);
```
Expected output:
```matlab
tf = logical
   1
```

### Ignore case using the legacy logical flag
```matlab
tf = startsWith("RUNMAT", "run", true);
```
Expected output:
```matlab
tf = logical
   1
```

### Apply a scalar prefix to every element of a string array
```matlab
labels = ["alpha" "beta" "gamma"];
tf = startsWith(labels, "a");
```
Expected output:
```matlab
tf = 1×3 logical array
   1   0   0
```

### Match element-wise prefixes with implicit expansion
```matlab
names = ["hydrogen"; "helium"; "lithium"];
prefixes = ["hyd"; "hel"; "lit"];
tf = startsWith(names, prefixes);
```
Expected output:
```matlab
tf = 3×1 logical array
   1
   1
   1
```

### Test prefixes for a cell array of character vectors
```matlab
C = {'Mercury', 'Venus', 'Mars'};
tf = startsWith(C, 'M');
```
Expected output:
```matlab
tf = 1×3 logical array
   1   0   1
```

### Provide multiple prefixes as a column vector
```matlab
tf = startsWith("saturn", ['s'; 'n'; 'x']);
```
Expected output:
```matlab
tf = 3×1 logical array
   1
   0
   0
```

### Handle empty and missing values
```matlab
texts = ["", "<missing>"];
tf = startsWith(texts, "");
```
Expected output:
```matlab
tf = 1×2 logical array
   1   0
```

## GPU residency in RunMat (Do I need `gpuArray`?)

You usually do NOT need to call `gpuArray` yourself in RunMat (unlike MATLAB).

`startsWith` always executes on the host, but RunMat's runtime automatically gathers any GPU-resident inputs
before evaluating the prefix check. Because the builtin registers a `ResidencyPolicy::GatherImmediately`,
the planner gathers device handles eagerly and computes the logical result on the CPU. You may still call
`gpuArray` manually for compatibility with MATLAB code; the runtime gathers the inputs just in time, so
results match MATLAB precisely.

## FAQ

### What types can I pass to `startsWith`?
Use string scalars/arrays, character vectors/arrays, or cell arrays of character vectors for both arguments.
Mixed combinations are accepted, and RunMat performs MATLAB-style implicit expansion when the array sizes differ.

### How do I ignore letter case?
Supply `'IgnoreCase', true` (or `'on'`) after the pattern argument. The option is case-insensitive, so `'ignorecase'`
also works. The default is `false`, matching MATLAB.

### What happens with empty patterns?
Empty patterns (`""` or `''`) always match non-missing text elements. When the text element is missing (`<missing>`),
the result is `false`.

### Can I provide multiple prefixes at once?
Yes. Provide `pattern` as a string array, character array, or cell array of character vectors. RunMat applies implicit
expansion so that scalar inputs expand across the other argument automatically.

### How are missing strings treated?
Missing string scalars (displayed as `<missing>`) never match any pattern and produce `false` in the result. Use
`ismissing` if you need to handle missing values separately.

### Does `startsWith` run on the GPU?
No. The builtin executes on the CPU. If inputs reside on the GPU (for example, after other accelerated operations),
RunMat gathers them automatically so behaviour matches MATLAB.

### Does `startsWith` preserve the input shape?
Yes. The output is a logical array whose shape reflects the MATLAB-style implicit expansion result. When that shape
contains exactly one element, the builtin returns a logical scalar.

## See Also
[`contains`](./contains), [`endsWith`](./endsWith), [`regexp`](./regexp), [`regexpi`](./regexpi)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/strings/search/startswith.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/search/startswith.rs)
- Found a bug? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::search::startswith")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "startsWith",
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
    notes: "Executes entirely on the host; inputs are gathered from the GPU before evaluating prefix checks.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::strings::search::startswith"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "startsWith",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Text operation; not eligible for fusion and materialises host logical results.",
};

const BUILTIN_NAME: &str = "startsWith";

#[runtime_builtin(
    name = "startsWith",
    category = "strings/search",
    summary = "Return logical values indicating whether text inputs start with specific patterns.",
    keywords = "startswith,prefix,text,ignorecase,search",
    accel = "sink",
    builtin_path = "crate::builtins::strings::search::startswith"
)]
async fn startswith_builtin(
    text: Value,
    pattern: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let text = gather_if_needed_async(&text).await?;
    let pattern = gather_if_needed_async(&pattern).await?;
    let ignore_case = parse_ignore_case(BUILTIN_NAME, &rest)?;
    let subject = TextCollection::from_subject(BUILTIN_NAME, text)?;
    let patterns = TextCollection::from_pattern(BUILTIN_NAME, pattern)?;
    evaluate_startswith(&subject, &patterns, ignore_case)
}

fn evaluate_startswith(
    subject: &TextCollection,
    patterns: &TextCollection,
    ignore_case: bool,
) -> BuiltinResult<Value> {
    let output_shape = broadcast_shapes(BUILTIN_NAME, &subject.shape, &patterns.shape)
        .map_err(|err| build_runtime_error(err).with_builtin(BUILTIN_NAME).build())?;
    let total = tensor::element_count(&output_shape);
    if total == 0 {
        return logical_result(BUILTIN_NAME, Vec::new(), output_shape);
    }

    let subject_strides = compute_strides(&subject.shape);
    let pattern_strides = compute_strides(&patterns.shape);
    let subject_lower = if ignore_case {
        Some(subject.lowercased())
    } else {
        None
    };
    let pattern_lower = if ignore_case {
        Some(patterns.lowercased())
    } else {
        None
    };

    let mut data = Vec::with_capacity(total);
    for linear in 0..total {
        let subject_idx = broadcast_index(linear, &output_shape, &subject.shape, &subject_strides);
        let pattern_idx = broadcast_index(linear, &output_shape, &patterns.shape, &pattern_strides);
        let value = match (
            &subject.elements[subject_idx],
            &patterns.elements[pattern_idx],
        ) {
            (TextElement::Missing, _) => false,
            (_, TextElement::Missing) => false,
            (TextElement::Text(text), TextElement::Text(pattern)) => {
                if pattern.is_empty() {
                    true
                } else if ignore_case {
                    let lowered_subject = subject_lower
                        .as_ref()
                        .and_then(|vec| vec[subject_idx].as_deref())
                        .expect("lowercase subject available");
                    let lowered_pattern = pattern_lower
                        .as_ref()
                        .and_then(|vec| vec[pattern_idx].as_deref())
                        .expect("lowercase pattern available");
                    lowered_subject.starts_with(lowered_pattern)
                } else {
                    text.starts_with(pattern.as_str())
                }
            }
        };
        data.push(if value { 1 } else { 0 });
    }
    logical_result(BUILTIN_NAME, data, output_shape)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{CellArray, CharArray, IntValue, LogicalArray, StringArray, Tensor};

    fn run_startswith(text: Value, pattern: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(startswith_builtin(text, pattern, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_string_scalar_true() {
        let result = run_startswith(
            Value::String("RunMat".into()),
            Value::String("Run".into()),
            Vec::new(),
        )
        .expect("startsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_string_scalar_false() {
        let result = run_startswith(
            Value::String("RunMat".into()),
            Value::String("Mat".into()),
            Vec::new(),
        )
        .expect("startsWith");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_ignore_case_option() {
        let result = run_startswith(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![Value::String("IgnoreCase".into()), Value::Bool(true)],
        )
        .expect("startsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_string_array_scalar_pattern() {
        let array = StringArray::new(
            vec!["alpha".into(), "beta".into(), "gamma".into()],
            vec![3, 1],
        )
        .unwrap();
        let result = run_startswith(
            Value::StringArray(array),
            Value::String("a".into()),
            Vec::new(),
        )
        .expect("startsWith");
        let expected = LogicalArray::new(vec![1, 0, 0], vec![3, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_elementwise_patterns() {
        let subjects = StringArray::new(
            vec!["hydrogen".into(), "helium".into(), "lithium".into()],
            vec![3, 1],
        )
        .unwrap();
        let patterns =
            StringArray::new(vec!["hyd".into(), "hel".into(), "lit".into()], vec![3, 1]).unwrap();
        let result = run_startswith(
            Value::StringArray(subjects),
            Value::StringArray(patterns),
            Vec::new(),
        )
        .expect("startsWith");
        let expected = LogicalArray::new(vec![1, 1, 1], vec![3, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_broadcast_pattern_column_vector() {
        let patterns = CharArray::new(vec!['s', 'n', 'x'], 3, 1).unwrap();
        let result = run_startswith(
            Value::String("saturn".into()),
            Value::CharArray(patterns),
            Vec::new(),
        )
        .expect("startsWith char");
        let expected = LogicalArray::new(vec![1, 0, 0], vec![3, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_cell_array_patterns() {
        let cell = CellArray::new(
            vec![
                Value::from("Mercury"),
                Value::from("Venus"),
                Value::from("Mars"),
            ],
            1,
            3,
        )
        .unwrap();
        let result = run_startswith(Value::Cell(cell), Value::String("M".into()), Vec::new())
            .expect("startsWith");
        let expected = LogicalArray::new(vec![1, 0, 1], vec![1, 3]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_missing_strings_false() {
        let array = StringArray::new(vec!["<missing>".into()], vec![1, 1]).unwrap();
        let result = run_startswith(
            Value::StringArray(array),
            Value::String("a".into()),
            Vec::new(),
        )
        .expect("startsWith");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_empty_pattern_true() {
        let result = run_startswith(
            Value::String("foo".into()),
            Value::String("".into()),
            Vec::new(),
        )
        .expect("startsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_invalid_option_name() {
        let err = run_startswith(
            Value::String("foo".into()),
            Value::String("f".into()),
            vec![Value::String("IgnoreCases".into()), Value::Bool(true)],
        )
        .unwrap_err();
        assert!(err.to_string().contains("unknown option"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_ignore_case_string_flag() {
        let result = run_startswith(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::String("on".into()),
            ],
        )
        .expect("startsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_ignore_case_numeric_flag() {
        let result = run_startswith(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::Int(IntValue::I32(0)),
            ],
        )
        .expect("startsWith");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_ignore_case_positional_value() {
        let result = run_startswith(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![Value::Bool(true)],
        )
        .expect("startsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_ignore_case_logical_array_value() {
        let logical = LogicalArray::new(vec![1], vec![1, 1]).unwrap();
        let result = run_startswith(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::LogicalArray(logical),
            ],
        )
        .expect("startsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_ignore_case_tensor_value() {
        let tensor = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let result = run_startswith(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![Value::String("IgnoreCase".into()), Value::Tensor(tensor)],
        )
        .expect("startsWith");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_ignore_case_invalid_value() {
        let err = run_startswith(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::String("maybe".into()),
            ],
        )
        .unwrap_err();
        assert!(err.to_string().contains("invalid value"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_ignore_case_logical_array_invalid_size() {
        let logical = LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap();
        let err = run_startswith(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::LogicalArray(logical),
            ],
        )
        .unwrap_err();
        assert!(err.to_string().contains("scalar logicals"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_ignore_case_numeric_nan_invalid() {
        let err = run_startswith(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![Value::Num(f64::NAN)],
        )
        .unwrap_err();
        assert!(err.to_string().contains("finite scalar"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_ignore_case_missing_value() {
        let err = run_startswith(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![Value::String("IgnoreCase".into())],
        )
        .unwrap_err();
        assert!(err
            .to_string()
            .contains("expected a value after 'IgnoreCase'"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_mismatched_shapes_error() {
        let text = StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap();
        let pattern =
            StringArray::new(vec!["a".into(), "b".into(), "c".into()], vec![3, 1]).unwrap();
        let err = run_startswith(
            Value::StringArray(text),
            Value::StringArray(pattern),
            Vec::new(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("size mismatch"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_invalid_subject_type() {
        let err =
            run_startswith(Value::Num(1.0), Value::String("a".into()), Vec::new()).unwrap_err();
        assert!(err.to_string().contains("first argument must be text"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_invalid_pattern_type() {
        let err =
            run_startswith(Value::String("foo".into()), Value::Num(1.0), Vec::new()).unwrap_err();
        assert!(
            err.to_string().contains("pattern must be text"),
            "expected pattern type error, got: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_cell_invalid_element_error() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let err =
            run_startswith(Value::Cell(cell), Value::String("a".into()), Vec::new()).unwrap_err();
        assert!(err.to_string().contains("cell array elements"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_zero_sized_inputs() {
        let subjects = StringArray::new(Vec::<String>::new(), vec![0, 1]).unwrap();
        let result = run_startswith(
            Value::StringArray(subjects),
            Value::String("a".into()),
            Vec::new(),
        )
        .expect("startsWith");
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![0, 1]);
                assert!(array.data.is_empty());
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn startswith_missing_pattern_false() {
        let result = run_startswith(
            Value::String("alpha".into()),
            Value::String("<missing>".into()),
            Vec::new(),
        )
        .expect("startsWith");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
