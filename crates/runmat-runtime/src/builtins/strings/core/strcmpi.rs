//! MATLAB-compatible `strcmpi` builtin for RunMat (case-insensitive string comparison).

use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::builtins::common::broadcast::{broadcast_index, broadcast_shapes, compute_strides};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::common::map_control_flow_with_builtin;
use crate::builtins::strings::search::text_utils::{logical_result, TextCollection, TextElement};
use crate::{build_runtime_error, gather_if_needed, BuiltinResult, RuntimeControlFlow};

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "strcmpi",
        builtin_path = "crate::builtins::strings::core::strcmpi"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "strcmpi"
category: "strings/core"
keywords: ["strcmpi", "case insensitive compare", "string compare", "text equality", "cell array"]
summary: "Compare text inputs for equality without considering letter case, matching MATLAB's `strcmpi` semantics."
references:
  - https://www.mathworks.com/help/matlab/ref/strcmpi.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: []
  broadcasting: "matlab"
  notes: "Runs on the host; GPU inputs are gathered automatically before comparison so results match MATLAB exactly."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 2
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::strings::core::strcmpi::tests"
  integration: "builtins::strings::core::strcmpi::tests::strcmpi_cell_array_scalar_casefold"
---

# What does the `strcmpi` function do in MATLAB / RunMat?
`strcmpi(a, b)` compares text values without considering letter case. It returns logical `true` when inputs match case-insensitively and `false` otherwise. Supported text types mirror MATLAB: string scalars/arrays, character vectors and arrays, and cell arrays filled with character vectors.

## How does the `strcmpi` function behave in MATLAB / RunMat?
- **Case-insensitive**: Letter case is ignored. `"RunMat"`, `"runmat"`, and `"RUNMAT"` all compare equal.
- **Accepted text types**: Works with string arrays, character vectors or matrices, and cell arrays of character vectors. Mixed combinations are normalised automatically.
- **Implicit expansion**: Scalar operands expand against array operands so element-wise comparisons follow MATLAB broadcasting rules.
- **Character arrays**: Rows are compared individually. Results are column vectors whose length equals the number of rows in the character array.
- **Cell arrays**: Each cell is treated as a text scalar. Scalar cells expand across the other operand before comparison.
- **Missing strings**: Elements whose value is `missing` (`"<missing>"`) always compare unequal, even to other missing values, matching MATLAB.
- **Result form**: Scalar comparisons return logical scalars. Otherwise the result is a logical array that matches the broadcast shape.

## `strcmpi` Function GPU Execution Behaviour
`strcmpi` is registered as an acceleration sink. When either operand resides on the GPU, RunMat gathers both to host memory before comparing them. This keeps behaviour identical to MATLAB and avoids requiring backend-specific kernels. The logical result is produced on the CPU and never remains GPU-resident.

## Examples of using the `strcmpi` function in MATLAB / RunMat

### Compare Two Strings Ignoring Case
```matlab
tf = strcmpi("RunMat", "runmat");
```
Expected output:
```matlab
tf = logical
   1
```

### Find Case-Insensitive Matches Inside a String Array
```matlab
colors = ["Red" "GREEN" "blue"];
mask = strcmpi(colors, "green");
```
Expected output:
```matlab
mask = 1×3 logical array
   0   1   0
```

### Compare Character Array Rows Without Case Sensitivity
```matlab
animals = char("Cat", "DOG", "cat");
tf = strcmpi(animals, "cAt");
```
Expected output:
```matlab
tf = 3×1 logical array
   1
   0
   1
```

### Compare Cell Arrays Of Character Vectors Case-Insensitively
```matlab
C1 = {'north', 'East', 'SOUTH'};
C2 = {'NORTH', 'east', 'west'};
tf = strcmpi(C1, C2);
```
Expected output:
```matlab
tf = 1×3 logical array
   1   1   0
```

### Broadcast A String Scalar Against A Character Matrix
```matlab
patterns = char("alpha", "BETA");
tf = strcmpi(patterns, ["ALPHA" "beta"]);
```
Expected output:
```matlab
tf = 2×2 logical array
   1   0
   0   1
```

### Handle Missing Strings In Case-Insensitive Comparisons
```matlab
values = ["Active" missing];
mask = strcmpi(values, "active");
```
Expected output:
```matlab
mask = 1×2 logical array
   1   0
```

## GPU residency in RunMat (Do I need `gpuArray`?)
You rarely need to call `gpuArray` manually. When inputs already live on the GPU, RunMat gathers them before calling `strcmpi`, then returns a logical result on the host. This matches MATLAB’s behaviour while keeping the runtime simple. Subsequent GPU-aware code can explicitly transfer results back if needed.

## FAQ

### Does `strcmpi` support string arrays, character arrays, and cell arrays?
Yes. All MATLAB-supported text containers are accepted, and mixed combinations are handled automatically with implicit expansion.

### Is whitespace significant when comparing character arrays?
Yes. Trailing spaces or different lengths make rows unequal, just like `strcmp`. Use `strtrim` or `strip` if you need to ignore whitespace.

### Do missing strings compare equal?
No. The `missing` sentinel compares unequal to all values, including another missing string scalar.

### Can I compare complex numbers or numeric arrays with `strcmpi`?
No. Both arguments must contain text. Numeric inputs produce a descriptive MATLAB-style error.

### How are GPU inputs handled?
They are gathered to the CPU automatically before comparison. Providers do not need to implement additional kernels for `strcmpi`.

### What is returned when both inputs are scalars?
A logical scalar is returned (`true` or `false`). For non-scalar shapes, a logical array that mirrors the broadcast dimensions is produced.

## See Also
[strcmp](./strcmp), [contains](./contains), [startswith](./startswith), [endswith](./endswith), [strip](./strip)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/strings/core/strcmpi.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/strings/core/strcmpi.rs)
- Found a bug? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::strings::core::strcmpi")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "strcmpi",
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
    notes: "Runs entirely on the CPU; GPU operands are gathered before comparison.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::strings::core::strcmpi")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "strcmpi",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Produces logical host results; not eligible for GPU fusion.",
};

#[allow(dead_code)]
fn strcmpi_flow(message: impl Into<String>) -> RuntimeControlFlow {
    build_runtime_error(message).with_builtin("strcmpi").build().into()
}

fn remap_strcmpi_flow(flow: RuntimeControlFlow) -> RuntimeControlFlow {
    map_control_flow_with_builtin(flow, "strcmpi")
}

#[runtime_builtin(
    name = "strcmpi",
    category = "strings/core",
    summary = "Compare text inputs for equality without considering case.",
    keywords = "strcmpi,string compare,text equality",
    accel = "sink",
    builtin_path = "crate::builtins::strings::core::strcmpi"
)]
fn strcmpi_builtin(a: Value, b: Value) -> crate::BuiltinResult<Value> {
    let a = gather_if_needed(&a).map_err(remap_strcmpi_flow)?;
    let b = gather_if_needed(&b).map_err(remap_strcmpi_flow)?;
    let left = TextCollection::from_argument("strcmpi", a, "first argument")?;
    let right = TextCollection::from_argument("strcmpi", b, "second argument")?;
    evaluate_strcmpi(&left, &right)
}

fn evaluate_strcmpi(left: &TextCollection, right: &TextCollection) -> BuiltinResult<Value> {
    let shape = broadcast_shapes("strcmpi", &left.shape, &right.shape)?;
    let total = tensor::element_count(&shape);
    if total == 0 {
        return logical_result("strcmpi", Vec::new(), shape);
    }
    let left_strides = compute_strides(&left.shape);
    let right_strides = compute_strides(&right.shape);
    let left_lower = left.lowercased();
    let right_lower = right.lowercased();
    let mut data = Vec::with_capacity(total);
    for linear in 0..total {
        let li = broadcast_index(linear, &shape, &left.shape, &left_strides);
        let ri = broadcast_index(linear, &shape, &right.shape, &right_strides);
        let equal = match (&left.elements[li], &right.elements[ri]) {
            (TextElement::Missing, _) => false,
            (_, TextElement::Missing) => false,
            (TextElement::Text(_), TextElement::Text(_)) => {
                match (&left_lower[li], &right_lower[ri]) {
                    (Some(lhs), Some(rhs)) => lhs == rhs,
                    _ => false,
                }
            }
        };
        data.push(if equal { 1 } else { 0 });
    }
    logical_result("strcmpi", data, shape)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::RuntimeControlFlow;
    use runmat_builtins::{CellArray, CharArray, LogicalArray, StringArray};

    fn error_message(flow: RuntimeControlFlow) -> String {
        match flow {
            RuntimeControlFlow::Error(err) => err.message().to_string(),
            RuntimeControlFlow::Suspend(_) => panic!("unexpected suspension"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_string_scalar_true_ignores_case() {
        let result = strcmpi_builtin(
            Value::String("RunMat".into()),
            Value::String("runmat".into()),
        )
        .expect("strcmpi");
        assert_eq!(result, Value::Bool(true));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_string_scalar_false_when_text_differs() {
        let result = strcmpi_builtin(
            Value::String("RunMat".into()),
            Value::String("runtime".into()),
        )
        .expect("strcmpi");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_string_array_broadcast_scalar_case_insensitive() {
        let array = StringArray::new(
            vec!["red".into(), "green".into(), "blue".into()],
            vec![1, 3],
        )
        .unwrap();
        let result = strcmpi_builtin(Value::StringArray(array), Value::String("GREEN".into()))
            .expect("strcmpi");
        let expected = LogicalArray::new(vec![0, 1, 0], vec![1, 3]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_char_array_row_compare_casefold() {
        let chars = CharArray::new(vec!['c', 'a', 't', 'D', 'O', 'G'], 2, 3).unwrap();
        let result =
            strcmpi_builtin(Value::CharArray(chars), Value::String("CaT".into())).expect("cmp");
        let expected = LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_char_array_to_char_array_casefold() {
        let left = CharArray::new(vec!['A', 'b', 'C', 'd'], 2, 2).unwrap();
        let right = CharArray::new(vec!['a', 'B', 'x', 'Y'], 2, 2).unwrap();
        let result =
            strcmpi_builtin(Value::CharArray(left), Value::CharArray(right)).expect("strcmpi");
        let expected = LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_cell_array_scalar_casefold() {
        let cell = CellArray::new(
            vec![
                Value::from("North"),
                Value::from("east"),
                Value::from("South"),
            ],
            1,
            3,
        )
        .unwrap();
        let result =
            strcmpi_builtin(Value::Cell(cell), Value::String("EAST".into())).expect("strcmpi");
        let expected = LogicalArray::new(vec![0, 1, 0], vec![1, 3]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_cell_array_vs_cell_array_broadcast() {
        let left = CellArray::new(vec![Value::from("North"), Value::from("East")], 1, 2).unwrap();
        let right = CellArray::new(vec![Value::from("north")], 1, 1).unwrap();
        let result = strcmpi_builtin(Value::Cell(left), Value::Cell(right)).expect("strcmpi");
        let expected = LogicalArray::new(vec![1, 0], vec![1, 2]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_string_array_multi_dimensional_broadcast() {
        let left = StringArray::new(vec!["north".into(), "south".into()], vec![2, 1]).unwrap();
        let right = StringArray::new(
            vec!["NORTH".into(), "EAST".into(), "SOUTH".into()],
            vec![1, 3],
        )
        .unwrap();
        let result =
            strcmpi_builtin(Value::StringArray(left), Value::StringArray(right)).expect("strcmpi");
        let expected = LogicalArray::new(vec![1, 0, 0, 0, 0, 1], vec![2, 3]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_missing_strings_compare_false() {
        let strings = StringArray::new(vec!["<missing>".into()], vec![1, 1]).unwrap();
        let result = strcmpi_builtin(
            Value::StringArray(strings.clone()),
            Value::StringArray(strings),
        )
        .expect("strcmpi");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_char_array_trailing_space_not_equal() {
        let chars = CharArray::new(vec!['c', 'a', 't', ' '], 1, 4).unwrap();
        let result =
            strcmpi_builtin(Value::CharArray(chars), Value::String("cat".into())).expect("strcmpi");
        assert_eq!(result, Value::Bool(false));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_size_mismatch_error() {
        let left = StringArray::new(vec!["a".into(), "b".into()], vec![2, 1]).unwrap();
        let right = StringArray::new(vec!["a".into(), "b".into(), "c".into()], vec![3, 1]).unwrap();
        let err = error_message(
            strcmpi_builtin(Value::StringArray(left), Value::StringArray(right))
                .expect_err("size mismatch"),
        );
        assert!(err.contains("size mismatch"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_invalid_argument_type() {
        let err = error_message(
            strcmpi_builtin(Value::Num(1.0), Value::String("a".into())).expect_err("invalid type"),
        );
        assert!(err.contains("first argument must be text"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_cell_array_invalid_element_errors() {
        let cell = CellArray::new(vec![Value::Num(42.0)], 1, 1).unwrap();
        let err = error_message(
            strcmpi_builtin(Value::Cell(cell), Value::String("test".into()))
                .expect_err("cell element type"),
        );
        assert!(err.contains("cell array elements must be character vectors or string scalars"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn strcmpi_empty_char_array_returns_empty() {
        let chars = CharArray::new(Vec::<char>::new(), 0, 3).unwrap();
        let result = strcmpi_builtin(Value::CharArray(chars), Value::String("anything".into()))
            .expect("cmp");
        let expected = LogicalArray::new(Vec::<u8>::new(), vec![0, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn strcmpi_with_wgpu_provider_matches_expected() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let names = StringArray::new(vec!["North".into(), "south".into()], vec![2, 1]).unwrap();
        let comparison = StringArray::new(vec!["north".into()], vec![1, 1]).unwrap();
        let result = strcmpi_builtin(Value::StringArray(names), Value::StringArray(comparison))
            .expect("strcmpi");
        let expected = LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap();
        assert_eq!(result, Value::LogicalArray(expected));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
