//! MATLAB-compatible `strncmp` builtin for RunMat.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::strings::search::text_utils::{logical_result, TextCollection, TextElement};
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{gather_if_needed, register_builtin_fusion_spec, register_builtin_gpu_spec};

const FN_NAME: &str = "strncmp";

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "strncmp"
category: "strings/core"
keywords: ["strncmp", "string compare", "prefix", "text equality", "case-sensitive"]
summary: "Compare text inputs for equality up to N leading characters with MATLAB-compatible broadcasting."
references:
  - https://www.mathworks.com/help/matlab/ref/strncmp.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "matlab"
  notes: "Executes on the CPU. GPU-resident inputs are gathered automatically so prefix comparisons match MATLAB exactly."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 3
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::strings::core::strncmp::tests"
  integration: "builtins::strings::core::strncmp::tests::strncmp_char_array_rows"
---

# What does the `strncmp` function do in MATLAB / RunMat?
`strncmp(A, B, N)` compares text values element-wise and returns logical `true` when the first `N`
characters of the corresponding elements are identical. Comparisons are case-sensitive and respect
MATLAB's implicit expansion rules across strings, character arrays, and cell arrays of character vectors.

## How does the `strncmp` function behave in MATLAB / RunMat?
- **Accepted text types**: String scalars/arrays, character vectors or character arrays, and cell arrays of character vectors.
- **Scalar `N` requirement**: The third argument must evaluate to a finite, nonnegative integer scalar. Numeric, logical, and scalar tensor/logical-array values are accepted when they convert cleanly.
- **Implicit expansion**: Scalars expand to match the size of the other operand before comparison.
- **Character arrays**: Rows are treated as independent character vectors. Each row is compared against the other operand and the result is returned as a column vector.
- **Unicode-aware comparisons**: Prefixes are counted in MATLAB characters (Unicode scalar values), so multi-byte UTF-8 sequences are handled transparently.
- **Prefix length semantics**: If `N` is `0`, every comparison evaluates to `true`. If either text element is shorter than `N`, it must match exactly up to the end of the shorter value and the longer value must also end within `N` characters to be considered equal.
- **Missing strings**: Any comparison involving a missing string returns `false` unless `N == 0`.
- **Result type**: Scalar comparisons return logical scalars. Array comparisons return logical arrays that follow MATLAB's column-major ordering.

## `strncmp` Function GPU Execution Behaviour
`strncmp` is registered as an acceleration **sink**. When any operand resides on the GPU, RunMat gathers
all inputs back to host memory before performing the comparison so that the behaviour matches MATLAB
exactly. The logical result is always returned on the host. Providers do not need to supply specialised kernels.

## Examples of using the `strncmp` function in MATLAB / RunMat

### Checking whether two strings share a prefix
```matlab
tf = strncmp("RunMat", "Runway", 3);
```
Expected output:
```matlab
tf = logical
   1
```

### Comparing string arrays with implicit expansion
```matlab
names = ["north" "south" "east"];
tf = strncmp(names, "no", 2);
```
Expected output:
```matlab
tf = 1×3 logical array
   1   0   0
```

### Comparing rows of a character array
```matlab
animals = char("cat", "camel", "cow");
tf = strncmp(animals, "ca", 2);
```
Expected output:
```matlab
tf = 3×1 logical array
   1
   1
   0
```

### Comparing cell arrays element-wise
```matlab
C1 = {'red', 'green', 'blue'};
C2 = {'rose', 'grey', 'black'};
tf = strncmp(C1, C2, 2);
```
Expected output:
```matlab
tf = 1×3 logical array
   1   0   0
```

### Handling zero-length comparisons
```matlab
tf = strncmp("alpha", "omega", 0);
```
Expected output:
```matlab
tf = logical
   1
```

## GPU residency in RunMat (Do I need `gpuArray`?)
No. If you pass GPU-resident data, RunMat automatically gathers it to host memory before running `strncmp`.
The builtin is an acceleration sink and always returns host logical outputs. Explicit `gpuArray` / `gather`
calls are only required for compatibility with legacy MATLAB workflows.

## FAQ

### What argument types does `strncmp` accept?
String arrays, character vectors/arrays, and cell arrays of character vectors. Mixed combinations are converted automatically. The third argument `N` must be a nonnegative integer scalar.

### Is the comparison case-sensitive?
Yes. Use `strncmpi` if you need a case-insensitive prefix comparison.

### What happens when `N` is zero?
The builtin returns `true` for every element because zero leading characters are compared.

### How are shorter strings handled when `N` is larger than their length?
The shorter value must match the longer value exactly for its entire length, and the longer value must not have additional characters within the first `N` positions. Otherwise the comparison returns `false`.

### How are missing string values treated?
Any comparison that involves a missing string returns `false`, except when `N` is zero (because no characters are compared).

### Does `strncmp` produce logical results?
Yes. Scalar comparisons yield logical scalars; array inputs produce logical arrays that follow MATLAB’s column-major ordering.

## See Also
[strcmp](./strcmp), [strcmpi](./strcmpi), [contains](../../search/contains), [startswith](../../search/startswith), [strlength](./strlength)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/strings/core/strncmp.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/strncmp.rs)
- Found a bug? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "strncmp",
    op_kind: GpuOpKind::Custom("string-prefix-compare"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Performs host-side prefix comparisons; GPU inputs are gathered before evaluation.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "strncmp",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Produces logical host results and is not eligible for GPU fusion.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("strncmp", DOC_MD);

#[runtime_builtin(
    name = "strncmp",
    category = "strings/core",
    summary = "Compare text inputs for equality up to N leading characters (case-sensitive).",
    keywords = "strncmp,string compare,prefix,text equality",
    accel = "sink"
)]
fn strncmp_builtin(a: Value, b: Value, n: Value) -> Result<Value, String> {
    let a = gather_if_needed(&a).map_err(|e| format!("{FN_NAME}: {e}"))?;
    let b = gather_if_needed(&b).map_err(|e| format!("{FN_NAME}: {e}"))?;
    let n = gather_if_needed(&n).map_err(|e| format!("{FN_NAME}: {e}"))?;

    let limit = parse_prefix_length(n)?;
    let left = TextCollection::from_argument(FN_NAME, a, "first argument")?;
    let right = TextCollection::from_argument(FN_NAME, b, "second argument")?;
    evaluate_strncmp(&left, &right, limit)
}

fn evaluate_strncmp(
    left: &TextCollection,
    right: &TextCollection,
    limit: usize,
) -> Result<Value, String> {
    let shape = broadcast_shapes(FN_NAME, &left.shape, &right.shape)?;
    let total = tensor::element_count(&shape);
    if total == 0 {
        return logical_result(FN_NAME, Vec::new(), shape);
    }

    let left_strides = compute_strides(&left.shape);
    let right_strides = compute_strides(&right.shape);
    let mut data = Vec::with_capacity(total);

    for linear in 0..total {
        let li = broadcast_index(linear, &shape, &left.shape, &left_strides);
        let ri = broadcast_index(linear, &shape, &right.shape, &right_strides);
        let equal = if limit == 0 {
            true
        } else {
            match (&left.elements[li], &right.elements[ri]) {
                (TextElement::Missing, _) | (_, TextElement::Missing) => false,
                (TextElement::Text(lhs), TextElement::Text(rhs)) => prefix_equal(lhs, rhs, limit),
            }
        };
        data.push(if equal { 1 } else { 0 });
    }

    logical_result(FN_NAME, data, shape)
}

fn prefix_equal(lhs: &str, rhs: &str, limit: usize) -> bool {
    if limit == 0 {
        return true;
    }
    let mut lhs_iter = lhs.chars();
    let mut rhs_iter = rhs.chars();
    let mut compared = 0usize;

    while compared < limit {
        let left_char = lhs_iter.next();
        let right_char = rhs_iter.next();
        match (left_char, right_char) {
            (Some(lc), Some(rc)) => {
                if lc != rc {
                    return false;
                }
            }
            (None, Some(_)) | (Some(_), None) => {
                return false;
            }
            (None, None) => {
                return true;
            }
        }
        compared += 1;
    }

    true
}

fn parse_prefix_length(value: Value) -> Result<usize, String> {
    match value {
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw < 0 {
                return Err(format!(
                    "{FN_NAME}: prefix length must be a nonnegative integer"
                ));
            }
            Ok(raw as usize)
        }
        Value::Num(n) => parse_prefix_length_from_float(n),
        Value::Bool(b) => Ok(if b { 1 } else { 0 }),
        Value::Tensor(tensor) => {
            if tensor.data.len() != 1 {
                return Err(format!(
                    "{FN_NAME}: prefix length must be a nonnegative integer scalar"
                ));
            }
            parse_prefix_length_from_float(tensor.data[0])
        }
        Value::LogicalArray(array) => {
            if array.data.len() != 1 {
                return Err(format!(
                    "{FN_NAME}: prefix length must be a nonnegative integer scalar"
                ));
            }
            Ok(if array.data[0] != 0 { 1 } else { 0 })
        }
        other => Err(format!(
            "{FN_NAME}: prefix length must be a nonnegative integer scalar, received {other:?}"
        )),
    }
}

fn parse_prefix_length_from_float(value: f64) -> Result<usize, String> {
    if !value.is_finite() {
        return Err(format!(
            "{FN_NAME}: prefix length must be a finite nonnegative integer"
        ));
    }
    if value < 0.0 {
        return Err(format!(
            "{FN_NAME}: prefix length must be a nonnegative integer"
        ));
    }
    let rounded = value.round();
    if (rounded - value).abs() > f64::EPSILON {
        return Err(format!(
            "{FN_NAME}: prefix length must be a nonnegative integer"
        ));
    }
    if rounded > (usize::MAX as f64) {
        return Err(format!(
            "{FN_NAME}: prefix length exceeds the maximum supported size"
        ));
    }
    Ok(rounded as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "doc_export")]
    use crate::builtins::common::test_support;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use runmat_builtins::{CellArray, CharArray, IntValue, LogicalArray, StringArray, Tensor};

    #[test]
    fn strncmp_exact_prefix_true() {
        let result = strncmp_builtin(
            Value::String("RunMat".into()),
            Value::String("Runway".into()),
            Value::Int(IntValue::I32(3)),
        )
        .expect("strncmp");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn strncmp_mismatch_within_prefix_false() {
        let result = strncmp_builtin(
            Value::String("RunMat".into()),
            Value::String("Runway".into()),
            Value::Int(IntValue::I32(4)),
        )
        .expect("strncmp");
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn strncmp_longer_string_after_prefix_false() {
        let result = strncmp_builtin(
            Value::String("cat".into()),
            Value::String("cater".into()),
            Value::Int(IntValue::I32(4)),
        )
        .expect("strncmp");
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn strncmp_zero_length_always_true() {
        let result = strncmp_builtin(
            Value::String("alpha".into()),
            Value::String("omega".into()),
            Value::Num(0.0),
        )
        .expect("strncmp");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn strncmp_prefix_length_bool_true_compares_first_character() {
        let result = strncmp_builtin(
            Value::String("alpha".into()),
            Value::String("array".into()),
            Value::Bool(true),
        )
        .expect("strncmp");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn strncmp_prefix_length_bool_false_treated_as_zero() {
        let result = strncmp_builtin(
            Value::String("alpha".into()),
            Value::String("omega".into()),
            Value::Bool(false),
        )
        .expect("strncmp");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn strncmp_prefix_length_logical_array_scalar() {
        let logical = LogicalArray::new(vec![1], vec![1]).unwrap();
        let result = strncmp_builtin(
            Value::String("beta".into()),
            Value::String("theta".into()),
            Value::LogicalArray(logical),
        )
        .expect("strncmp");
        assert_eq!(result, Value::Bool(false));
    }

    #[test]
    fn strncmp_prefix_length_tensor_scalar_double() {
        let limit = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let result = strncmp_builtin(
            Value::String("gamma".into()),
            Value::String("gamut".into()),
            Value::Tensor(limit),
        )
        .expect("strncmp");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn strncmp_char_array_rows() {
        let chars = CharArray::new(
            vec![
                'c', 'a', 't', ' ', ' ', 'c', 'a', 'm', 'e', 'l', 'c', 'o', 'w', ' ', ' ',
            ],
            3,
            5,
        )
        .unwrap();
        let result = strncmp_builtin(
            Value::CharArray(chars),
            Value::String("ca".into()),
            Value::Int(IntValue::I32(2)),
        )
        .expect("strncmp");
        let expected = LogicalArray::new(vec![1, 1, 0], vec![3, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[test]
    fn strncmp_cell_arrays_broadcast() {
        let left = CellArray::new(
            vec![
                Value::from("red"),
                Value::from("green"),
                Value::from("blue"),
            ],
            1,
            3,
        )
        .unwrap();
        let right = CellArray::new(
            vec![
                Value::from("rose"),
                Value::from("gray"),
                Value::from("black"),
            ],
            1,
            3,
        )
        .unwrap();
        let result = strncmp_builtin(
            Value::Cell(left),
            Value::Cell(right),
            Value::Int(IntValue::I32(2)),
        )
        .expect("strncmp");
        let expected = LogicalArray::new(vec![0, 1, 1], vec![1, 3]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[test]
    fn strncmp_string_array_broadcast_scalar() {
        let strings = StringArray::new(
            vec!["north".into(), "south".into(), "east".into()],
            vec![1, 3],
        )
        .unwrap();
        let result = strncmp_builtin(
            Value::StringArray(strings),
            Value::String("no".into()),
            Value::Int(IntValue::I32(2)),
        )
        .expect("strncmp");
        let expected = LogicalArray::new(vec![1, 0, 0], vec![1, 3]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[test]
    fn strncmp_missing_string_false_when_prefix_positive() {
        let strings =
            StringArray::new(vec!["<missing>".into(), "value".into()], vec![1, 2]).unwrap();
        let result = strncmp_builtin(
            Value::StringArray(strings),
            Value::String("val".into()),
            Value::Int(IntValue::I32(3)),
        )
        .expect("strncmp");
        let expected = LogicalArray::new(vec![0, 1], vec![1, 2]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[test]
    fn strncmp_missing_zero_length_true() {
        let strings = StringArray::new(vec!["<missing>".into()], vec![1, 1]).unwrap();
        let result = strncmp_builtin(
            Value::StringArray(strings),
            Value::String("anything".into()),
            Value::Int(IntValue::I32(0)),
        )
        .expect("strncmp");
        assert_eq!(result, Value::Bool(true));
    }

    #[test]
    fn strncmp_size_mismatch_error() {
        let left = StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap();
        let right = StringArray::new(vec!["a".into(), "b".into(), "c".into()], vec![3, 1]).unwrap();
        let err = strncmp_builtin(
            Value::StringArray(left),
            Value::StringArray(right),
            Value::Int(IntValue::I32(1)),
        )
        .expect_err("size mismatch");
        assert!(err.contains("size mismatch"));
    }

    #[test]
    fn strncmp_invalid_length_type_errors() {
        let err = strncmp_builtin(
            Value::String("abc".into()),
            Value::String("abc".into()),
            Value::String("3".into()),
        )
        .expect_err("invalid prefix length");
        assert!(err.contains("prefix length"));
    }

    #[test]
    fn strncmp_negative_length_errors() {
        let err = strncmp_builtin(
            Value::String("abc".into()),
            Value::String("abc".into()),
            Value::Num(-1.0),
        )
        .expect_err("negative length");
        assert!(err.to_ascii_lowercase().contains("nonnegative"));
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn strncmp_prefix_length_from_gpu_tensor() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };
        use runmat_accelerate_api::HostTensorView;

        let provider = match register_wgpu_provider(WgpuProviderOptions::default()) {
            Ok(provider) => provider,
            Err(_) => return,
        };
        let tensor = Tensor::new(vec![3.0], vec![1, 1]).unwrap();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload prefix length to GPU");
        let result = strncmp_builtin(
            Value::String("delta".into()),
            Value::String("deluge".into()),
            Value::GpuTensor(handle.clone()),
        )
        .expect("strncmp");
        assert_eq!(result, Value::Bool(true));
        let _ = provider.free(&handle);
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
