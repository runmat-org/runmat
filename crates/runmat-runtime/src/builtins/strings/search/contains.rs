//! MATLAB-compatible `contains` builtin for RunMat.

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
        name = "contains",
        builtin_path = "crate::builtins::strings::search::contains"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "contains"
category: "strings/search"
keywords: ["contains", "substring", "text search", "ignorecase", "string array"]
summary: "Determine whether text inputs contain specific patterns with MATLAB-compatible implicit expansion and case handling."
references:
  - https://www.mathworks.com/help/matlab/ref/contains.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "matlab"
  notes: "Runs on the CPU; when inputs reside on the GPU RunMat gathers them automatically so behaviour matches MATLAB."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::strings::search::contains::tests"
  integration: "builtins::strings::search::contains::tests::contains_cell_array_patterns"
---

# What does the `contains` function do in MATLAB / RunMat?
`contains(str, pattern)` returns a logical result indicating whether each element of `str` contains
the corresponding text in `pattern`. The builtin supports string arrays, character vectors/arrays,
and cell arrays of character vectors, honouring MATLAB's implicit expansion semantics.

## How does the `contains` function behave in MATLAB / RunMat?
- Accepts text inputs as string scalars/arrays, character scalars/arrays, or cell arrays of character vectors.
- Accepts patterns in the same formats. Either input may be scalar; when sizes differ MATLAB-style implicit expansion applies.
- Missing string scalars (`<missing>`) never match any pattern.
- Empty pattern values (`""` or `''`) always match non-missing text elements.
- The optional `'IgnoreCase', true|false` name-value pair controls case sensitivity (default is case-sensitive).
- Returns a logical scalar when the broadcasted size is one element, otherwise returns a logical array with the broadcasted shape.

## `contains` Function GPU Execution Behaviour
`contains` performs host-side text comparisons. When invoked with data that currently resides on the GPU,
RunMat gathers the inputs before executing the search so that results match MATLAB exactly. No provider hooks
are required for this builtin.

## Examples of using the `contains` function in MATLAB / RunMat

### Check if a string contains a substring
```matlab
tf = contains("RunMat Accelerate", "Accelerate");
```
Expected output:
```matlab
tf = logical
   1
```

### Perform a case-insensitive search
```matlab
tf = contains("RunMat", "run", 'IgnoreCase', true);
```
Expected output:
```matlab
tf = logical
   1
```

### Apply a scalar pattern to every element of a string array
```matlab
labels = ["alpha" "beta" "gamma"];
tf = contains(labels, "a");
```
Expected output:
```matlab
tf = 1×3 logical array
   1   1   1
```

### Match element-wise patterns with implicit expansion
```matlab
names = ["hydrogen"; "helium"; "lithium"];
patterns = ["gen"; "ium"; "iron"];
tf = contains(names, patterns);
```
Expected output:
```matlab
tf = 3×1 logical array
   1
   1
   0
```

### Search a cell array of character vectors
```matlab
C = {'Mercury', 'Venus', 'Mars'};
tf = contains(C, 'us');
```
Expected output:
```matlab
tf = 1×3 logical array
   0   1   0
```

### Provide multiple patterns as a column vector
```matlab
tf = contains("saturn", ['s'; 'n'; 'x']);
```
Expected output:
```matlab
tf = 3×1 logical array
   1
   1
   0
```

### Handle empty and missing text values
```matlab
texts = ["", "<missing>"];
tf = contains(texts, "");
```
Expected output:
```matlab
tf = 1×2 logical array
   1   0
```

## GPU residency in RunMat (Do I need `gpuArray`?)

You usually do NOT need to call `gpuArray` yourself in RunMat (unlike MATLAB).

`contains` always executes on the host, but RunMat's runtime automatically gathers any GPU-resident inputs
before evaluating the text search so behaviour is identical to MATLAB. You can still call `gpuArray` manually
for compatibility with legacy MATLAB code; the runtime will gather those inputs just in time when `contains`
runs.

## FAQ

### What types can I pass to `contains`?
Use string scalars/arrays, character vectors/arrays, or cell arrays of character vectors for both the text and the pattern. Mixed combinations are accepted, and RunMat performs MATLAB-style implicit expansion when the array sizes differ.

### How do I ignore letter case?
Specify `'IgnoreCase', true` (or `'on'`) after the pattern argument. The option is case-insensitive, so `'ignorecase'` also works. The default is `false`, matching MATLAB.

### What happens with empty patterns?
Empty patterns (`""` or `''`) always match non-missing text elements. When the text element is missing (`<missing>`), the result is `false`.

### Can I search for multiple patterns at once?
Yes. Provide `pattern` as a string array, character array, or cell array of character vectors. RunMat applies implicit expansion so that scalar inputs expand across the other argument automatically.

### How are missing strings treated?
Missing string scalars (displayed as `<missing>`) never match any pattern and produce `false` in the result. Use `ismissing` if you need to separate missing values before calling `contains`.

### Does `contains` run on the GPU?
No. The builtin executes on the CPU. If inputs reside on the GPU (for example, after other accelerated operations), RunMat gathers them automatically so behaviour matches MATLAB.

### Does `contains` preserve the input shape?
Yes. The output is a logical array whose shape is the implicit-expansion result of the input shapes. When the broadcasted shape has exactly one element, the builtin returns a logical scalar.

## See Also
`startsWith`, `endsWith`, [`regexp`](../regex/regexp.rs), [`regexpi`](../regex/regexpi.rs)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/strings/search/contains.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/search/contains.rs)
- Found a bug? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::search::contains")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "contains",
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
    notes: "Executes entirely on the host; inputs are gathered from the GPU before performing substring checks.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::search::contains")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "contains",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Text operation; not eligible for fusion and materialises host logical results.",
};

const BUILTIN_NAME: &str = "contains";

#[runtime_builtin(
    name = "contains",
    category = "strings/search",
    summary = "Return logical values indicating whether text inputs contain specific patterns.",
    keywords = "contains,substring,text,ignorecase,search",
    accel = "sink",
    builtin_path = "crate::builtins::strings::search::contains"
)]
async fn contains_builtin(
    text: Value,
    pattern: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    let text = gather_if_needed_async(&text).await?;
    let pattern = gather_if_needed_async(&pattern).await?;
    let ignore_case = parse_ignore_case(BUILTIN_NAME, &rest)?;
    let subject = TextCollection::from_subject(BUILTIN_NAME, text)?;
    let patterns = TextCollection::from_pattern(BUILTIN_NAME, pattern)?;
    evaluate_contains(&subject, &patterns, ignore_case)
}

fn evaluate_contains(
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
                    lowered_subject.contains(lowered_pattern)
                } else {
                    text.contains(pattern.as_str())
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
    use runmat_builtins::{CellArray, CharArray, IntValue, LogicalArray, StringArray};

    fn run_contains(text: Value, pattern: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        futures::executor::block_on(contains_builtin(text, pattern, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_string_scalar_true() {
        let result = run_contains(
            Value::String("RunMat".into()),
            Value::String("Run".into()),
            Vec::new(),
        )
        .expect("contains");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_string_scalar_false() {
        let result = run_contains(
            Value::String("RunMat".into()),
            Value::String("forge".into()),
            Vec::new(),
        )
        .expect("contains");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_ignore_case_option() {
        let result = run_contains(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![Value::String("IgnoreCase".into()), Value::Bool(true)],
        )
        .expect("contains");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_string_array_scalar_pattern() {
        let array = StringArray::new(
            vec!["alpha".into(), "beta".into(), "gamma".into()],
            vec![3, 1],
        )
        .unwrap();
        let result = run_contains(
            Value::StringArray(array),
            Value::String("a".into()),
            Vec::new(),
        )
        .expect("contains");
        let expected = LogicalArray::new(vec![1, 1, 1], vec![3, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_elementwise_patterns() {
        let subjects = StringArray::new(
            vec!["hydrogen".into(), "helium".into(), "lithium".into()],
            vec![3, 1],
        )
        .unwrap();
        let patterns =
            StringArray::new(vec!["gen".into(), "ium".into(), "iron".into()], vec![3, 1]).unwrap();
        let result = run_contains(
            Value::StringArray(subjects),
            Value::StringArray(patterns),
            Vec::new(),
        )
        .expect("contains");
        let expected = LogicalArray::new(vec![1, 1, 0], vec![3, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_broadcast_pattern_column_vector() {
        let patterns = CharArray::new(vec!['s', 'n', 'x'], 3, 1).unwrap();
        let result = run_contains(
            Value::String("saturn".into()),
            Value::CharArray(patterns),
            Vec::new(),
        )
        .expect("contains char");
        let expected = LogicalArray::new(vec![1, 1, 0], vec![3, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_cell_array_patterns() {
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
        let result = run_contains(Value::Cell(cell), Value::String("us".into()), Vec::new())
            .expect("contains");
        let expected = LogicalArray::new(vec![0, 1, 0], vec![1, 3]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_missing_strings_false() {
        let array = StringArray::new(vec!["<missing>".into()], vec![1, 1]).unwrap();
        let result = run_contains(
            Value::StringArray(array),
            Value::String("a".into()),
            Vec::new(),
        )
        .expect("contains");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_empty_pattern_true() {
        let result = run_contains(
            Value::String("foo".into()),
            Value::String("".into()),
            Vec::new(),
        )
        .expect("contains");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_invalid_option_name() {
        let err = run_contains(
            Value::String("foo".into()),
            Value::String("f".into()),
            vec![Value::String("IgnoreCases".into()), Value::Bool(true)],
        )
        .unwrap_err();
        assert!(err.to_string().contains("unknown option"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_ignore_case_string_flag() {
        let result = run_contains(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::String("on".into()),
            ],
        )
        .expect("contains");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_ignore_case_numeric_flag() {
        let result = run_contains(
            Value::String("RunMat".into()),
            Value::String("run".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::Int(IntValue::I32(0)),
            ],
        )
        .expect("contains");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_ignore_case_invalid_value() {
        let err = run_contains(
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
    fn contains_ignore_case_missing_value() {
        let err = run_contains(
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
    fn contains_mismatched_shapes_error() {
        let text = StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap();
        let pattern =
            StringArray::new(vec!["a".into(), "b".into(), "c".into()], vec![3, 1]).unwrap();
        let err = run_contains(
            Value::StringArray(text),
            Value::StringArray(pattern),
            Vec::new(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("size mismatch"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_invalid_subject_type() {
        let err = run_contains(Value::Num(1.0), Value::String("a".into()), Vec::new()).unwrap_err();
        assert!(err.to_string().contains("first argument must be text"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_invalid_pattern_type() {
        let err =
            run_contains(Value::String("foo".into()), Value::Num(1.0), Vec::new()).unwrap_err();
        assert!(err.to_string().contains("pattern must be text"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_cell_invalid_element_error() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let err =
            run_contains(Value::Cell(cell), Value::String("a".into()), Vec::new()).unwrap_err();
        assert!(err.to_string().contains("cell array elements"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn contains_zero_sized_inputs() {
        let subjects = StringArray::new(Vec::<String>::new(), vec![0, 1]).unwrap();
        let result = run_contains(
            Value::StringArray(subjects),
            Value::String("a".into()),
            Vec::new(),
        )
        .expect("contains");
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
    fn contains_missing_pattern_false() {
        let result = run_contains(
            Value::String("alpha".into()),
            Value::String("<missing>".into()),
            Vec::new(),
        )
        .expect("contains");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
