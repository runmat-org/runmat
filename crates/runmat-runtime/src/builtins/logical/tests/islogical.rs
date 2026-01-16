//! MATLAB-compatible `islogical` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::{build_runtime_error, BuiltinResult, RuntimeControlFlow};
use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "islogical",
        builtin_path = "crate::builtins::logical::tests::islogical"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "islogical"
category: "logical/tests"
keywords: ["islogical", "logical type", "boolean predicate", "gpuArray logical", "MATLAB islogical"]
summary: "Return true when a value is stored as MATLAB logical data (scalar or array)."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Checks provider metadata via `logical_islogical`; falls back to host inspection when unavailable."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::logical::tests::islogical::tests"
  integration: "builtins::logical::tests::islogical::tests::gpu_handles_use_metadata_when_available"
  gpu: "builtins::logical::tests::islogical::tests::islogical_wgpu_elem_eq_sets_metadata"
  doc: "builtins::logical::tests::islogical::tests::doc_examples_present"
---

# What does the `islogical` function do in MATLAB / RunMat?
`tf = islogical(x)` returns a logical scalar that is `true` when `x` is stored as MATLAB
logical data (either a scalar logical or an N-D logical array). All other value kinds,
including doubles, integers, characters, structs, and objects, return `false`.

## How does the `islogical` function behave in MATLAB / RunMat?
- Logical scalars (`true`, `false`) return `true`.
- Instances of `logical` arrays return `true`, regardless of dimensionality.
- Numeric, complex, string, character, struct, cell, and object inputs return `false`.
- `gpuArray` values return `true` when their residency metadata marks them as logical; the
  builtin never gathers device buffers just to answer the query unless metadata is absent.
- The result is always a logical scalar.

## Examples of using the `islogical` function in MATLAB / RunMat

### Checking a logical scalar

```matlab
flag = islogical(true);
```

Expected output:

```matlab
flag =
     1
```

### Verifying a logical array constructed with `logical`

```matlab
mask = logical([1 0 1 0]);
is_mask_logical = islogical(mask);
```

Expected output:

```matlab
is_mask_logical =
     1
```

### Numeric arrays are not logical

```matlab
values = [1 2 3];
is_logical = islogical(values);
```

Expected output:

```matlab
is_logical =
     0
```

### Character arrays return false

```matlab
chars = ['R' 'u' 'n'];
is_char_logical = islogical(chars);
```

Expected output:

```matlab
is_char_logical =
     0
```

### gpuArray comparisons yield logical gpuArrays

```matlab
G = gpuArray([1 2 3]);
mask = G > 1;
tf = islogical(mask);
```

Expected output:

```matlab
tf =
     1
```

### Containers and structs are not logical

```matlab
items = {true, 1};
person = struct("name", "Ada");
tf_cell = islogical(items);
tf_struct = islogical(person);
```

Expected output:

```matlab
tf_cell =
     0

tf_struct =
     0
```

## `islogical` Function GPU Execution Behaviour
When RunMat Accelerate is active, `islogical` first asks the registered acceleration provider
for the `logical_islogical` hook. Providers that implement the hook respond using device
metadata without copying data back to the CPU. If the hook is missing, RunMat consults its
own residency metadata and only gathers device buffers as a last resort, ensuring that the
builtin always succeeds while minimizing host↔device transfers.

## GPU residency in RunMat (Do I need `gpuArray`?)
You typically do **not** need to call `gpuArray` manually. RunMat's auto-offload planner keeps
logical masks resident on the GPU when downstream expressions benefit from device execution.
Explicit `gpuArray` and `gather` calls remain available for compatibility with MATLAB code
that manages residency explicitly.

## FAQ

### Does `islogical` ever return an array?
No. The builtin always returns a logical scalar, even when the input is an array.

### Does `islogical` convert numeric data to logical?
No. It only reports whether the storage is already logical. Use the `logical` builtin to
convert numeric data explicitly.

### How does `islogical` behave with `gpuArray` inputs?
If the provider exposes `logical_islogical`, the check happens entirely on the GPU. When the
hook is absent, RunMat consults residency metadata and, if needed, gathers the value to host
memory to inspect its storage kind.

### Are character or string arrays considered logical?
No. Character vectors, string scalars, and string arrays are not logical values and therefore
return `false`.

### Do cell arrays or structs ever count as logical?
No. Containers such as cell arrays, structs, tables, and objects always return `false`.

### Does a logical `gpuArray` gathered back to the host stay logical?
Yes. When residency metadata marks the handle as logical, gathering produces a host
`logical` array, and `islogical` continues to report `true`.

## See Also
[isreal](./isreal), [isfinite](./isfinite), [isnan](./isnan), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- The full source code for the implementation of the `islogical` function is available at: [`crates/runmat-runtime/src/builtins/logical/tests/islogical.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/tests/islogical.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::tests::islogical")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "islogical",
    op_kind: GpuOpKind::Custom("metadata"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("logical_islogical")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Reads provider metadata when `logical_islogical` is implemented; otherwise consults runtime residency tracking and, as a last resort, gathers once to the host.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::tests::islogical")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "islogical",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Metadata query that executes outside of fusion pipelines.",
};

const BUILTIN_NAME: &str = "islogical";
const IDENTIFIER_INTERNAL: &str = "RunMat:islogical:InternalError";

#[runtime_builtin(
    name = "islogical",
    category = "logical/tests",
    summary = "Return true when a value uses logical storage.",
    keywords = "islogical,logical,bool,gpu",
    accel = "metadata",
    builtin_path = "crate::builtins::logical::tests::islogical"
)]
fn islogical_builtin(value: Value) -> BuiltinResult<Value> {
    match value {
        Value::GpuTensor(handle) => islogical_gpu(handle),
        other => islogical_host(other),
    }
}

fn islogical_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(flag) = provider.logical_islogical(&handle) {
            return Ok(Value::Bool(flag));
        }
    }

    if runmat_accelerate_api::handle_is_logical(&handle) {
        return Ok(Value::Bool(true));
    }

    let gpu_value = Value::GpuTensor(handle.clone());
    let gathered = gpu_helpers::gather_value(&gpu_value)
        .map_err(|err| internal_error(format!("islogical: {err}")))?;
    islogical_host(gathered)
}

fn islogical_host(value: Value) -> BuiltinResult<Value> {
    let flag = matches!(value, Value::Bool(_) | Value::LogicalArray(_));
    match value {
        Value::GpuTensor(_) => Err(internal_error(
            "islogical: internal error, GPU value reached host path",
        )),
        _ => Ok(Value::Bool(flag)),
    }
}

fn internal_error(message: impl Into<String>) -> RuntimeControlFlow {
    build_runtime_error(message)
        .with_identifier(IDENTIFIER_INTERNAL)
        .with_builtin(BUILTIN_NAME)
        .build()
        .into()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_backend;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{CharArray, LogicalArray, Tensor, Value};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_scalars_report_true() {
        assert_eq!(
            islogical_builtin(Value::Bool(true)).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            islogical_builtin(Value::Bool(false)).unwrap(),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logical_arrays_report_true() {
        let array = LogicalArray::new(vec![1, 0, 1], vec![3, 1]).unwrap();
        assert_eq!(
            islogical_builtin(Value::LogicalArray(array)).unwrap(),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_values_report_false() {
        let tensor = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        assert_eq!(
            islogical_builtin(Value::Tensor(tensor)).unwrap(),
            Value::Bool(false)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn string_values_report_false() {
        assert_eq!(
            islogical_builtin(Value::String("runmat".to_string())).unwrap(),
            Value::Bool(false)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn char_arrays_report_false() {
        let chars = CharArray::new("rm".chars().collect(), 1, 2).unwrap();
        assert_eq!(
            islogical_builtin(Value::CharArray(chars)).unwrap(),
            Value::Bool(false)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn structures_and_cells_report_false() {
        let struct_value = runmat_builtins::StructValue::new();
        let cell = runmat_builtins::CellArray::new(vec![Value::Bool(true)], 1, 1).unwrap();
        assert_eq!(
            islogical_builtin(Value::Struct(struct_value)).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            islogical_builtin(Value::Cell(cell)).unwrap(),
            Value::Bool(false)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_handles_use_metadata_when_available() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 0.0, 1.0], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let logical_value = gpu_helpers::logical_gpu_value(handle.clone());
            let result = islogical_builtin(logical_value).expect("islogical");
            assert_eq!(result, Value::Bool(true));
            provider.free(&handle).ok();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_numeric_handles_report_false() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![2.0, 4.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = islogical_builtin(Value::GpuTensor(handle.clone())).expect("islogical");
            assert_eq!(result, Value::Bool(false));
            provider.free(&handle).ok();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn islogical_wgpu_elem_eq_sets_metadata() {
        let _ = wgpu_backend::register_wgpu_provider(wgpu_backend::WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let lhs = vec![1.0, 0.0, 1.0, 0.0];
        let rhs = vec![1.0, 0.0, 0.0, 0.0];
        let shape = vec![4, 1];
        let lhs_view = HostTensorView {
            data: &lhs,
            shape: &shape,
        };
        let rhs_view = HostTensorView {
            data: &rhs,
            shape: &shape,
        };
        let lhs_handle = provider.upload(&lhs_view).expect("upload lhs");
        let rhs_handle = provider.upload(&rhs_view).expect("upload rhs");
        let mask_handle = provider.elem_eq(&lhs_handle, &rhs_handle).expect("elem_eq");

        let result = islogical_builtin(Value::GpuTensor(mask_handle.clone())).expect("islogical");
        assert_eq!(result, Value::Bool(true));

        provider.free(&mask_handle).ok();
        provider.free(&lhs_handle).ok();
        provider.free(&rhs_handle).ok();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn islogical_wgpu_numeric_stays_false() {
        let _ = wgpu_backend::register_wgpu_provider(wgpu_backend::WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let data = vec![0.25, 0.5, 0.75];
        let shape = vec![3, 1];
        let view = HostTensorView {
            data: &data,
            shape: &shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let result = islogical_builtin(Value::GpuTensor(handle.clone())).expect("islogical");
        assert_eq!(result, Value::Bool(false));
        provider.free(&handle).ok();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
