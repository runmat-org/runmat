//! MATLAB-compatible `strcmp` builtin for RunMat.

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::strings::search::text_utils::{logical_result, TextCollection, TextElement};
use crate::gather_if_needed;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "strcmp",
        builtin_path = "crate::builtins::strings::core::strcmp"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "strcmp"
category: "strings/core"
keywords: ["strcmp", "string compare", "text equality", "cell array", "character vector"]
summary: "Compare text inputs for exact equality with MATLAB-compatible implicit expansion across text types."
references:
  - https://www.mathworks.com/help/matlab/ref/strcmp.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "matlab"
  notes: "Executes on the CPU; GPU-resident inputs are gathered automatically so results match MATLAB behaviour exactly."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::strings::core::strcmp::tests"
  integration: "builtins::strings::core::strcmp::tests::strcmp_cell_array_scalar"
---

# What does the `strcmp` function do in MATLAB / RunMat?
`strcmp(a, b)` returns logical `true` when two pieces of text match exactly and `false` otherwise.
It accepts string arrays, character vectors/arrays, and cell arrays of character vectors, mirroring MATLAB semantics.

## How does the `strcmp` function behave in MATLAB / RunMat?
- **Accepted text types**: Works with string scalars/arrays, character vectors or matrices created with `char`, and cell arrays of character vectors. Mixed combinations are converted automatically, matching MATLAB.
- **Implicit expansion**: Scalar inputs expand to the shape of the other operand, producing element-wise comparisons for higher-dimensional arrays.
- **Character arrays**: Rows are compared individually. The result is a column vector whose length equals the number of rows in the character array.
- **Cell arrays**: Each cell is treated as a text scalar. Scalar cell arrays expand across the other operand before comparison.
- **Missing strings**: String elements equal to `missing` compare unequal to everything, including other `missing` values.
- **Result form**: A single logical scalar is returned for scalar comparisons; otherwise you receive a logical array using column-major MATLAB layout.
- **Case sensitivity**: Matching is case-sensitive. Use `strcmpi` for case-insensitive comparisons.

## `strcmp` Function GPU Execution Behaviour
`strcmp` is registered as an acceleration sink. When either input resides on the GPU, RunMat gathers both
operands back to host memory before comparing them so the results match MATLAB exactly. Providers do not
need to implement custom kernels, and the logical result is always returned on the CPU.

## Examples of using the `strcmp` function in MATLAB / RunMat

### Compare Two Equal Strings
```matlab
tf = strcmp("RunMat", "RunMat");
```
Expected output:
```matlab
tf = logical
   1
```

### Compare String Array With Scalar Text
```matlab
names = ["red" "green" "blue"];
tf = strcmp(names, "green");
```
Expected output:
```matlab
tf = 1×3 logical array
   0   1   0
```

### Compare Character Array Rows
```matlab
labels = char("cat", "dog", "cat");
tf = strcmp(labels, "cat");
```
Expected output:
```matlab
tf = 3×1 logical array
   1
   0
   1
```

### Compare Two Cell Arrays Of Character Vectors
```matlab
C1 = {'apple', 'pear', 'grape'};
C2 = {'apple', 'peach', 'grape'};
tf = strcmp(C1, C2);
```
Expected output:
```matlab
tf = 1×3 logical array
   1   0   1
```

### Handle Missing Strings
```matlab
vals = ["alpha" missing];
tf = strcmp(vals, "alpha");
```
Expected output:
```matlab
tf = 1×2 logical array
   1   0
```

### Implicit Expansion With Column Vector Text
```matlab
patterns = char("north", "south");
tf = strcmp(patterns, ["north" "east"]);
```
Expected output:
```matlab
tf = 2×2 logical array
   1   0
   0   0
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You normally do **not** need to call `gpuArray`. If you do, RunMat gathers the operands before computing `strcmp`
so the output matches MATLAB. The result always lives on the host because this builtin inspects text data.

## FAQ

### What types can I pass to `strcmp`?
Use string arrays, character vectors/arrays, or cell arrays of character vectors. Mixed combinations are accepted and follow MATLAB's implicit expansion rules.

### Does `strcmp` ignore letter case?
No. `strcmp` is case-sensitive. Use `strcmpi` for case-insensitive comparisons.

### What happens when the inputs contain missing strings?
Missing string scalars compare unequal to every value (including other missing strings), so the result is `false`.

### Can `strcmp` compare matrices of characters?
Yes. Character arrays compare row-by-row, returning a column vector whose entries tell you whether each row matches.

### Does `strcmp` return numeric or logical results?
It returns logical results. Scalars become logical scalars (`Value::Bool`), while arrays are returned as logical arrays.

## See Also
[string](./string), [char](./char), [contains](../../search/contains), [startswith](../../search/startswith), [strlength](./strlength)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/strings/core/strcmp.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/strcmp.rs)
- Found a bug? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::core::strcmp")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "strcmp",
    op_kind: GpuOpKind::Custom("string-compare"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Performs host-side text comparisons; GPU operands are gathered automatically before evaluation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::core::strcmp")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "strcmp",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Produces logical results on the host; not eligible for GPU fusion.",
};

#[runtime_builtin(
    name = "strcmp",
    category = "strings/core",
    summary = "Compare text inputs for exact matches (case-sensitive).",
    keywords = "strcmp,string compare,text equality",
    accel = "sink",
    builtin_path = "crate::builtins::strings::core::strcmp"
)]
fn strcmp_builtin(a: Value, b: Value) -> Result<Value, String> {
    let a = gather_if_needed(&a).map_err(|e| format!("strcmp: {e}"))?;
    let b = gather_if_needed(&b).map_err(|e| format!("strcmp: {e}"))?;
    let left = TextCollection::from_argument("strcmp", a, "first argument")?;
    let right = TextCollection::from_argument("strcmp", b, "second argument")?;
    evaluate_strcmp(&left, &right)
}

fn evaluate_strcmp(left: &TextCollection, right: &TextCollection) -> Result<Value, String> {
    let shape = broadcast_shapes("strcmp", &left.shape, &right.shape)?;
    let total = tensor::element_count(&shape);
    if total == 0 {
        return logical_result("strcmp", Vec::new(), shape);
    }
    let left_strides = compute_strides(&left.shape);
    let right_strides = compute_strides(&right.shape);
    let mut data = Vec::with_capacity(total);
    for linear in 0..total {
        let li = broadcast_index(linear, &shape, &left.shape, &left_strides);
        let ri = broadcast_index(linear, &shape, &right.shape, &right_strides);
        let equal = match (&left.elements[li], &right.elements[ri]) {
            (TextElement::Missing, _) => false,
            (_, TextElement::Missing) => false,
            (TextElement::Text(lhs), TextElement::Text(rhs)) => lhs == rhs,
        };
        data.push(if equal { 1 } else { 0 });
    }
    logical_result("strcmp", data, shape)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{CellArray, CharArray, LogicalArray, StringArray};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_string_scalar_true() {
        let result = strcmp_builtin(
            Value::String("RunMat".into()),
            Value::String("RunMat".into()),
        )
        .expect("strcmp");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_string_scalar_false() {
        let result = strcmp_builtin(
            Value::String("RunMat".into()),
            Value::String("runmat".into()),
        )
        .expect("strcmp");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_string_array_broadcast_scalar() {
        let array = StringArray::new(
            vec!["red".into(), "green".into(), "blue".into()],
            vec![1, 3],
        )
        .unwrap();
        let result =
            strcmp_builtin(Value::StringArray(array), Value::String("green".into())).expect("cmp");
        let expected = LogicalArray::new(vec![0, 1, 0], vec![1, 3]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_char_array_row_compare() {
        let chars = CharArray::new(vec!['c', 'a', 't', 'd', 'o', 'g'], 2, 3).unwrap();
        let result =
            strcmp_builtin(Value::CharArray(chars), Value::String("cat".into())).expect("cmp");
        let expected = LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_char_array_to_char_array() {
        let left = CharArray::new(vec!['a', 'b', 'c', 'd'], 2, 2).unwrap();
        let right = CharArray::new(vec!['a', 'b', 'x', 'y'], 2, 2).unwrap();
        let result =
            strcmp_builtin(Value::CharArray(left), Value::CharArray(right)).expect("strcmp");
        let expected = LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_cell_array_scalar() {
        let cell = CellArray::new(
            vec![
                Value::from("apple"),
                Value::from("pear"),
                Value::from("grape"),
            ],
            1,
            3,
        )
        .unwrap();
        let result =
            strcmp_builtin(Value::Cell(cell), Value::String("grape".into())).expect("strcmp");
        let expected = LogicalArray::new(vec![0, 0, 1], vec![1, 3]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_cell_array_to_cell_array_broadcasts() {
        let left = CellArray::new(vec![Value::from("red"), Value::from("blue")], 2, 1).unwrap();
        let right = CellArray::new(vec![Value::from("red")], 1, 1).unwrap();
        let result = strcmp_builtin(Value::Cell(left), Value::Cell(right)).expect("strcmp");
        let expected = LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_string_array_multi_dimensional_broadcast() {
        let left = StringArray::new(vec!["north".into(), "south".into()], vec![2, 1]).unwrap();
        let right = StringArray::new(
            vec!["north".into(), "east".into(), "south".into()],
            vec![1, 3],
        )
        .unwrap();
        let result =
            strcmp_builtin(Value::StringArray(left), Value::StringArray(right)).expect("strcmp");
        let expected = LogicalArray::new(vec![1, 0, 0, 0, 0, 1], vec![2, 3]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_char_array_trailing_space_is_not_equal() {
        let chars = CharArray::new(vec!['c', 'a', 't', ' '], 1, 4).unwrap();
        let result =
            strcmp_builtin(Value::CharArray(chars), Value::String("cat".into())).expect("strcmp");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_char_array_empty_rows_returns_empty() {
        let chars = CharArray::new(Vec::new(), 0, 0).unwrap();
        let result = strcmp_builtin(Value::CharArray(chars), Value::String("anything".into()))
            .expect("strcmp");
        match result {
            Value::LogicalArray(array) => {
                assert_eq!(array.shape, vec![0, 1]);
                assert!(array.data.is_empty());
            }
            other => panic!("expected empty logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_missing_strings_compare_false() {
        let strings = StringArray::new(vec!["<missing>".into()], vec![1, 1]).unwrap();
        let result = strcmp_builtin(
            Value::StringArray(strings.clone()),
            Value::StringArray(strings),
        )
        .expect("strcmp");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_missing_string_false() {
        let array = StringArray::new(vec!["alpha".into(), "<missing>".into()], vec![1, 2]).unwrap();
        let result =
            strcmp_builtin(Value::StringArray(array), Value::String("alpha".into())).expect("cmp");
        let expected = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_size_mismatch_error() {
        let left = StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap();
        let right = StringArray::new(vec!["a".into(), "b".into(), "c".into()], vec![3, 1]).unwrap();
        let err = strcmp_builtin(Value::StringArray(left), Value::StringArray(right))
            .expect_err("size mismatch");
        assert!(err.contains("size mismatch"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmp_invalid_argument_type() {
        let err =
            strcmp_builtin(Value::Num(1.0), Value::String("a".into())).expect_err("invalid type");
        assert!(err.contains("first argument must be text"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
