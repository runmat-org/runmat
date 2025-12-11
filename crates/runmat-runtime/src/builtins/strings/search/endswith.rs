//! MATLAB-compatible `endsWith` builtin for RunMat.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::gather_if_needed;

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};

use super::text_utils::{logical_result, parse_ignore_case, TextCollection, TextElement};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "endsWith",
        wasm_path = "crate::builtins::strings::search::endswith"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "endsWith"
category: "strings/search"
keywords: ["endswith", "suffix", "string array", "ignorecase", "text search"]
summary: "Check whether text inputs end with specific patterns using MATLAB-compatible broadcasting and case handling."
references:
  - https://www.mathworks.com/help/matlab/ref/endswith.html
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
  unit: "builtins::strings::search::endswith::tests"
  integration: "builtins::strings::search::endswith::tests::endswith_cell_array_patterns"
---

# What does the `endsWith` function do in MATLAB / RunMat?
`endsWith(str, pattern)` returns a logical result indicating whether each element of `str`
ends with the corresponding text in `pattern`. The builtin supports string arrays, character
vectors/arrays, and cell arrays of character vectors, mirroring MATLAB's implicit expansion semantics.

## How does the `endsWith` function behave in MATLAB / RunMat?
- Accepts text inputs as string scalars/arrays, character vectors/arrays, or cell arrays of character vectors.
- Accepts patterns in the same formats; scalar inputs expand across the other argument following MATLAB broadcast rules.
- Missing string scalars (`missing`, displayed as `<missing>`) never match any pattern.
- Empty patterns (`""` or `''`) always match non-missing text elements.
- Patterns that are `<missing>` never match and therefore return `false`.
- Character arrays treat each row as an independent element; zero-row character arrays yield empty outputs.
- The optional `IgnoreCase` flag can be supplied either as a trailing scalar or via the `'IgnoreCase', value` name-value pair. It accepts logical/numeric scalars and the strings `'on'`, `'off'`, `'true'`, and `'false'` (default is case-sensitive).
- Returns a logical scalar when the broadcasted size is one element, otherwise returns a logical array.

## `endsWith` Function GPU Execution Behaviour
`endsWith` performs host-side suffix comparison. When inputs currently live on the GPU, RunMat gathers
them back to the host before evaluation so the behaviour is identical to MATLAB. No acceleration provider
hooks are required for this builtin.

## Examples of using the `endsWith` function in MATLAB / RunMat

### Check if a filename ends with an extension
```matlab
tf = endsWith("report.pdf", ".pdf");
```
Expected output:
```matlab
tf = logical
   1
```

### Perform a case-insensitive suffix test
```matlab
tf = endsWith("RunMat", "MAT", 'IgnoreCase', true);
```
Expected output:
```matlab
tf = logical
   1
```

### Apply a scalar suffix to every element of a string array
```matlab
labels = ["alpha" "beta" "gamma"];
tf = endsWith(labels, "ma");
```
Expected output:
```matlab
tf = 1×3 logical array
   0   0   1
```

### Use element-wise suffixes with implicit expansion
```matlab
names = ["hydrogen"; "helium"; "lithium"];
suffixes = ["gen"; "ium"; "ium"];
tf = endsWith(names, suffixes);
```
Expected output:
```matlab
tf = 3×1 logical array
   1
   1
   1
```

### Test suffixes for a cell array of character vectors
```matlab
C = {'Mercury', 'Venus', 'Mars'};
tf = endsWith(C, 's');
```
Expected output:
```matlab
tf = 1×3 logical array
   0   1   1
```

### Provide multiple suffixes as a column vector
```matlab
tf = endsWith("saturn", ['n'; 'x'; 'r']);
```
Expected output:
```matlab
tf = 3×1 logical array
   1
   0
   0
```

### Handle empty patterns and missing values
```matlab
texts = ["", missing];
tf = endsWith(texts, "");
```
Expected output:
```matlab
tf = 1×2 logical array
   1   0
```

## GPU residency in RunMat (Do I need `gpuArray`?)

You usually do NOT need to call `gpuArray` yourself in RunMat (unlike MATLAB).

`endsWith` always executes on the host, but RunMat's runtime automatically gathers any GPU-resident inputs
before evaluating the suffix check. Because the builtin registers a `ResidencyPolicy::GatherImmediately`,
the planner gathers device handles eagerly and computes the logical result on the CPU. You may still call
`gpuArray` manually for compatibility with MATLAB code; the runtime gathers the inputs just in time, so
results match MATLAB precisely.

## FAQ

### What types can I pass to `endsWith`?
Use string scalars/arrays, character vectors/arrays, or cell arrays of character vectors for both arguments.
Mixed combinations are accepted, and RunMat performs MATLAB-style implicit expansion when the array sizes differ.

### How do I ignore letter case?
Supply `'IgnoreCase', true` (or `'on'`) after the pattern argument. The option is case-insensitive, so `'ignorecase'`
also works. The default is `false`, matching MATLAB.

### What happens with empty patterns?
Empty patterns (`""` or `''`) always match non-missing text elements. When the text element is missing (`<missing>`),
the result is `false`.

### Can I provide multiple suffixes at once?
Yes. Provide `pattern` as a string array, character array, or cell array of character vectors. RunMat applies implicit
expansion so that scalar inputs expand across the other argument automatically.

### How are missing strings treated?
Missing string scalars (displayed as `<missing>`) never match any pattern and produce `false` in the result. Use
`ismissing` if you need to handle missing values separately.

### Does `endsWith` run on the GPU?
No. The builtin executes on the CPU. If inputs reside on the GPU (for example, after other accelerated operations),
RunMat gathers them automatically so behaviour matches MATLAB.

### Does `endsWith` preserve the input shape?
Yes. The output is a logical array whose shape reflects the MATLAB-style implicit expansion result. When that shape
contains exactly one element, the builtin returns a logical scalar.

## See Also
[`contains`](./contains), [`startsWith`](./startswith), [`regexp`](../regex/regexp), [`regexpi`](../regex/regexpi)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/strings/search/endswith.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/search/endswith.rs)
- Found a bug? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(wasm_path = "crate::builtins::strings::search::endswith")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "endsWith",
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
    notes: "Executes entirely on the host; inputs are gathered from the GPU before evaluating suffix checks.",
};

#[runmat_macros::register_fusion_spec(wasm_path = "crate::builtins::strings::search::endswith")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "endsWith",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Text operation; not eligible for fusion and materialises host logical results.",
};

#[runtime_builtin(
    name = "endsWith",
    category = "strings/search",
    summary = "Return logical values indicating whether text inputs end with specific patterns.",
    keywords = "endswith,suffix,text,ignorecase,search",
    accel = "sink",
    wasm_path = "crate::builtins::strings::search::endswith"
)]
fn endswith_builtin(text: Value, pattern: Value, rest: Vec<Value>) -> Result<Value, String> {
    let text = gather_if_needed(&text).map_err(|e| format!("endsWith: {e}"))?;
    let pattern = gather_if_needed(&pattern).map_err(|e| format!("endsWith: {e}"))?;
    let mut option_args = Vec::with_capacity(rest.len());
    for value in rest {
        option_args.push(gather_if_needed(&value).map_err(|e| format!("endsWith: {e}"))?);
    }
    let ignore_case = parse_ignore_case("endsWith", &option_args)?;
    let subject = TextCollection::from_subject("endsWith", text)?;
    let patterns = TextCollection::from_pattern("endsWith", pattern)?;
    evaluate_endswith(&subject, &patterns, ignore_case)
}

fn evaluate_endswith(
    subject: &TextCollection,
    patterns: &TextCollection,
    ignore_case: bool,
) -> Result<Value, String> {
    let output_shape = broadcast_shapes("endsWith", &subject.shape, &patterns.shape)?;
    let total = tensor::element_count(&output_shape);
    if total == 0 {
        return logical_result("endsWith", Vec::new(), output_shape);
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
                    lowered_subject.ends_with(lowered_pattern)
                } else {
                    text.ends_with(pattern.as_str())
                }
            }
        };
        data.push(if value { 1 } else { 0 });
    }
    logical_result("endsWith", data, output_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{CellArray, CharArray, IntValue, LogicalArray, StringArray, Tensor};

    #[test]
    fn endswith_string_scalar_true() {
        let result = endswith_builtin(
            Value::String("RunMat".into()),
            Value::String("Mat".into()),
            Vec::new(),
        )
        .expect("endsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn endswith_string_scalar_false() {
        let result = endswith_builtin(
            Value::String("RunMat".into()),
            Value::String("Run".into()),
            Vec::new(),
        )
        .expect("endsWith");
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn endswith_ignore_case_option() {
        let result = endswith_builtin(
            Value::String("RunMat".into()),
            Value::String("mat".into()),
            vec![Value::String("IgnoreCase".into()), Value::Bool(true)],
        )
        .expect("endsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn endswith_string_array_scalar_pattern() {
        let array = StringArray::new(
            vec!["alpha".into(), "beta".into(), "gamma".into()],
            vec![3, 1],
        )
        .unwrap();
        let result = endswith_builtin(
            Value::StringArray(array),
            Value::String("a".into()),
            Vec::new(),
        )
        .expect("endsWith");
        let expected = LogicalArray::new(vec![1, 1, 1], vec![3, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[test]
    fn endswith_elementwise_patterns() {
        let subjects = StringArray::new(
            vec!["hydrogen".into(), "helium".into(), "lithium".into()],
            vec![3, 1],
        )
        .unwrap();
        let patterns =
            StringArray::new(vec!["gen".into(), "ium".into(), "ium".into()], vec![3, 1]).unwrap();
        let result = endswith_builtin(
            Value::StringArray(subjects),
            Value::StringArray(patterns),
            Vec::new(),
        )
        .expect("endsWith");
        let expected = LogicalArray::new(vec![1, 1, 1], vec![3, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[test]
    fn endswith_broadcast_pattern_column_vector() {
        let patterns = CharArray::new(vec!['n', 'x', 'r'], 3, 1).unwrap();
        let result = endswith_builtin(
            Value::String("saturn".into()),
            Value::CharArray(patterns),
            Vec::new(),
        )
        .expect("endsWith char");
        let expected = LogicalArray::new(vec![1, 0, 0], vec![3, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[test]
    fn endswith_cell_array_patterns() {
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
        let result = endswith_builtin(Value::Cell(cell), Value::String("s".into()), Vec::new())
            .expect("endsWith");
        let expected = LogicalArray::new(vec![0, 1, 1], vec![1, 3]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[test]
    fn endswith_missing_strings_false() {
        let array = StringArray::new(vec!["<missing>".into()], vec![1, 1]).unwrap();
        let result = endswith_builtin(
            Value::StringArray(array),
            Value::String("a".into()),
            Vec::new(),
        )
        .expect("endsWith");
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn endswith_empty_pattern_true() {
        let result = endswith_builtin(
            Value::String("foo".into()),
            Value::String("".into()),
            Vec::new(),
        )
        .expect("endsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn endswith_invalid_option_name() {
        let err = endswith_builtin(
            Value::String("foo".into()),
            Value::String("o".into()),
            vec![Value::String("IgnoreCases".into()), Value::Bool(true)],
        )
        .unwrap_err();
        assert!(err.contains("unknown option"));
    }

    #[test]
    fn endswith_ignore_case_string_flag() {
        let result = endswith_builtin(
            Value::String("RunMat".into()),
            Value::String("mat".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::String("on".into()),
            ],
        )
        .expect("endsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn endswith_ignore_case_numeric_flag() {
        let result = endswith_builtin(
            Value::String("RunMat".into()),
            Value::String("mat".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::Int(IntValue::I32(0)),
            ],
        )
        .expect("endsWith");
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn endswith_ignore_case_positional_value() {
        let result = endswith_builtin(
            Value::String("RunMat".into()),
            Value::String("mat".into()),
            vec![Value::Bool(true)],
        )
        .expect("endsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn endswith_ignore_case_logical_array_value() {
        let logical = LogicalArray::new(vec![1], vec![1, 1]).unwrap();
        let result = endswith_builtin(
            Value::String("RunMat".into()),
            Value::String("mat".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::LogicalArray(logical),
            ],
        )
        .expect("endsWith");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn endswith_ignore_case_tensor_value() {
        let tensor = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let result = endswith_builtin(
            Value::String("RunMat".into()),
            Value::String("mat".into()),
            vec![Value::String("IgnoreCase".into()), Value::Tensor(tensor)],
        )
        .expect("endsWith");
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn endswith_ignore_case_gpu_tensor_flag() {
        test_support::with_test_provider(|provider| {
            let data = [1.0];
            let shape = [1usize, 1usize];
            let handle = provider
                .upload(&HostTensorView {
                    data: &data,
                    shape: &shape,
                })
                .expect("upload");
            let result = endswith_builtin(
                Value::String("RunMat".into()),
                Value::String("mat".into()),
                vec![
                    Value::String("IgnoreCase".into()),
                    Value::GpuTensor(handle.clone()),
                ],
            )
            .expect("endsWith");
            assert_eq!(result, Value::Bool(true));
            provider.free(&handle).expect("free gpu flag");
        });
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn endswith_ignore_case_gpu_tensor_flag_wgpu() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };

        if register_wgpu_provider(WgpuProviderOptions::default()).is_err() {
            // Skip when wgpu backend cannot initialise on this machine.
            return;
        }
        let Some(provider) = runmat_accelerate_api::provider() else {
            return;
        };
        let data = [1.0];
        let shape = [1usize, 1usize];
        let handle = provider
            .upload(&HostTensorView {
                data: &data,
                shape: &shape,
            })
            .expect("upload");
        let result = endswith_builtin(
            Value::String("RunMat".into()),
            Value::String("mat".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::GpuTensor(handle.clone()),
            ],
        )
        .expect("endsWith");
        assert_eq!(result, Value::Bool(true));
        let _ = provider.free(&handle);
    }

    #[test]
    fn endswith_ignore_case_invalid_value() {
        let err = endswith_builtin(
            Value::String("RunMat".into()),
            Value::String("mat".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::String("maybe".into()),
            ],
        )
        .unwrap_err();
        assert!(err.contains("invalid value"));
    }

    #[test]
    fn endswith_ignore_case_logical_array_invalid_size() {
        let logical = LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap();
        let err = endswith_builtin(
            Value::String("RunMat".into()),
            Value::String("mat".into()),
            vec![
                Value::String("IgnoreCase".into()),
                Value::LogicalArray(logical),
            ],
        )
        .unwrap_err();
        assert!(err.contains("scalar logicals"));
    }

    #[test]
    fn endswith_ignore_case_numeric_nan_invalid() {
        let err = endswith_builtin(
            Value::String("RunMat".into()),
            Value::String("mat".into()),
            vec![Value::Num(f64::NAN)],
        )
        .unwrap_err();
        assert!(err.contains("finite scalar"));
    }

    #[test]
    fn endswith_ignore_case_missing_value() {
        let err = endswith_builtin(
            Value::String("RunMat".into()),
            Value::String("mat".into()),
            vec![Value::String("IgnoreCase".into())],
        )
        .unwrap_err();
        assert!(err.contains("expected a value after 'IgnoreCase'"));
    }

    #[test]
    fn endswith_mismatched_shapes_error() {
        let text = StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap();
        let pattern =
            StringArray::new(vec!["a".into(), "b".into(), "c".into()], vec![3, 1]).unwrap();
        let err = endswith_builtin(
            Value::StringArray(text),
            Value::StringArray(pattern),
            Vec::new(),
        )
        .unwrap_err();
        assert!(err.contains("size mismatch"));
    }

    #[test]
    fn endswith_invalid_subject_type() {
        let err =
            endswith_builtin(Value::Num(1.0), Value::String("a".into()), Vec::new()).unwrap_err();
        assert!(err.contains("first argument must be text"));
    }

    #[test]
    fn endswith_invalid_pattern_type() {
        let err =
            endswith_builtin(Value::String("foo".into()), Value::Num(1.0), Vec::new()).unwrap_err();
        assert!(
            err.contains("pattern must be text"),
            "expected pattern type error, got: {err}"
        );
    }

    #[test]
    fn endswith_cell_invalid_element_error() {
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let err =
            endswith_builtin(Value::Cell(cell), Value::String("a".into()), Vec::new()).unwrap_err();
        assert!(err.contains("cell array elements"));
    }

    #[test]
    fn endswith_zero_sized_inputs() {
        let subjects = StringArray::new(Vec::<String>::new(), vec![0, 1]).unwrap();
        let result = endswith_builtin(
            Value::StringArray(subjects),
            Value::String("a".into()),
            Vec::new(),
        )
        .expect("endsWith");
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![0, 1]);
                assert!(array.data.is_empty());
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[test]
    fn endswith_missing_pattern_false() {
        let result = endswith_builtin(
            Value::String("alpha".into()),
            Value::String("<missing>".into()),
            Vec::new(),
        )
        .expect("endsWith");
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
