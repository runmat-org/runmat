//! MATLAB-compatible `isnumeric` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "isnumeric",
        builtin_path = "crate::builtins::logical::tests::isnumeric"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "isnumeric"
category: "logical/tests"
keywords: ["isnumeric", "numeric type", "type predicate", "gpuArray isnumeric", "MATLAB isnumeric"]
summary: "Return true when a value is stored as numeric data (real or complex)."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Queries device metadata; falls back to runtime residency tracking when provider hooks are absent."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::logical::tests::isnumeric::tests"
  integration: "builtins::logical::tests::isnumeric::tests::gpu_numeric_and_logical_handles"
  gpu: "builtins::logical::tests::isnumeric::tests::isnumeric_wgpu_handles_respect_metadata"
  doc: "builtins::logical::tests::isnumeric::tests::doc_examples_present"
---

# What does the `isnumeric` function do in MATLAB / RunMat?
`tf = isnumeric(x)` returns a logical scalar that is `true` when `x` stores numeric data and
`false` otherwise. Numeric data includes doubles, singles, integers, and complex numbers, as
well as dense numeric arrays that live on the CPU or GPU.

## How does the `isnumeric` function behave in MATLAB / RunMat?
- Every built-in numeric class (`double`, `single`, signed/unsigned integer types) returns
  `true`, including complex scalars.
- Real and complex numeric arrays return `true` regardless of dimensionality or residency on
  the CPU or GPU.
- `gpuArray` values rely on provider metadata: numeric handles return `true`, while logical
  masks constructed on the GPU return `false`.
- Logical values, characters, strings, tables, cell arrays, structs, objects, and function
  handles return `false`.
- The result is always a logical scalar.

## Examples of using the `isnumeric` function in MATLAB / RunMat

### Checking if a scalar double is numeric

```matlab
tf = isnumeric(42);
```

Expected output:

```matlab
tf =
     1
```

### Detecting numeric matrices

```matlab
A = [1 2 3; 4 5 6];
tf = isnumeric(A);
```

Expected output:

```matlab
tf =
     1
```

### Testing complex numbers for numeric storage

```matlab
z = 1 + 2i;
tf = isnumeric(z);
```

Expected output:

```matlab
tf =
     1
```

### Logical arrays are not numeric

```matlab
mask = logical([1 0 1]);
tf = isnumeric(mask);
```

Expected output:

```matlab
tf =
     0
```

### Character vectors and strings return false

```matlab
chars = ['R' 'u' 'n'];
name = "RunMat";
tf_chars = isnumeric(chars);
tf_string = isnumeric(name);
```

Expected output:

```matlab
tf_chars =
     0

tf_string =
     0
```

### Evaluating `gpuArray` inputs

```matlab
G = gpuArray(rand(4));
mask = G > 0.5;
tf_numeric = isnumeric(G);
tf_mask = isnumeric(mask);
```

Expected output:

```matlab
tf_numeric =
     1

tf_mask =
     0
```

## `isnumeric` Function GPU Execution Behaviour
When RunMat Accelerate is active, `isnumeric` first checks provider metadata via the
`logical_islogical` hook to determine whether a `gpuArray` handle was created as a logical
mask. Providers that supply the hook can answer the query without copying data back to the
host. When the hook is absent, RunMat consults its residency metadata and only gathers the
value to host memory when the residency tag is missing, ensuring the builtin always succeeds.

## GPU residency in RunMat (Do I need `gpuArray`?)
You generally do **not** need to call `gpuArray` manually. RunMat's auto-offload planner keeps
numeric tensors on the GPU across fused expressions whenever that improves performance.
Explicit `gpuArray` and `gather` calls remain available for compatibility with MATLAB scripts
that manage residency themselves.

## FAQ

### Does `isnumeric` ever return an array?
No. The builtin always returns a logical scalar, even when the input is an array.

### Are complex tensors considered numeric?
Yes. Real and complex tensors both return `true`, matching MATLAB semantics.

### Does `isnumeric` gather GPU data back to the host?
Only when residency metadata is unavailable. Providers that expose type metadata let RunMat
answer the query without host↔device transfers.

### Do logical masks return `true`?
No. Logical scalars and logical arrays return `false`. Use `islogical` if you need to detect
logical storage explicitly.

### What about character vectors or string arrays?
They return `false`, just like in MATLAB. Characters and strings are text types rather than
numeric arrays.

### Do cell arrays or structs ever count as numeric?
No. Containers and objects always return `false`.

### Is there a difference between CPU and GPU numeric arrays?
No. Both host and device numeric arrays return `true`; only logical GPU handles report `false`.

## See Also
[islogical](./islogical), [isreal](./isreal), [isfinite](./isfinite), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- The full source code for the implementation of the `isnumeric` function is available at: [`crates/runmat-runtime/src/builtins/logical/tests/isnumeric.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/logical/tests/isnumeric.rs)
- Found a bug or behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::logical::tests::isnumeric")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isnumeric",
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
        "Uses provider metadata to distinguish logical gpuArrays from numeric ones; otherwise falls back to runtime residency tracking.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::logical::tests::isnumeric")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isnumeric",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Type check executed outside fusion; planners treat it as a scalar metadata query.",
};

#[runtime_builtin(
    name = "isnumeric",
    category = "logical/tests",
    summary = "Return true when a value is stored as numeric data.",
    keywords = "isnumeric,numeric,type,gpu",
    accel = "metadata",
    builtin_path = "crate::builtins::logical::tests::isnumeric"
)]
fn isnumeric_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => isnumeric_gpu(handle),
        other => Ok(Value::Bool(isnumeric_value(&other))),
    }
}

fn isnumeric_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(flag) = provider.logical_islogical(&handle) {
            return Ok(Value::Bool(!flag));
        }
    }

    if runmat_accelerate_api::handle_is_logical(&handle) {
        return Ok(Value::Bool(false));
    }

    // Fall back to gathering only when metadata is unavailable.
    let gpu_value = Value::GpuTensor(handle.clone());
    if let Ok(gathered) = gpu_helpers::gather_value(&gpu_value) {
        return isnumeric_host(gathered);
    }

    Ok(Value::Bool(true))
}

fn isnumeric_host(value: Value) -> Result<Value, String> {
    if matches!(value, Value::GpuTensor(_)) {
        return Err("isnumeric: internal error, GPU value reached host path".to_string());
    }
    Ok(Value::Bool(isnumeric_value(&value)))
}

fn isnumeric_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Num(_)
            | Value::Int(_)
            | Value::Complex(_, _)
            | Value::Tensor(_)
            | Value::ComplexTensor(_)
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        CellArray, CharArray, Closure, ComplexTensor, HandleRef, IntValue, Listener, LogicalArray,
        MException, ObjectInstance, StringArray, StructValue, Tensor,
    };
    use runmat_gc_api::GcPtr;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_scalars_return_true() {
        assert_eq!(
            isnumeric_builtin(Value::Num(3.5)).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            isnumeric_builtin(Value::Int(IntValue::I16(7))).unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            isnumeric_builtin(Value::Complex(1.0, -2.0)).unwrap(),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn numeric_tensors_return_true() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        assert_eq!(
            isnumeric_builtin(Value::Tensor(tensor)).unwrap(),
            Value::Bool(true)
        );

        let complex = ComplexTensor::new(vec![(1.0, 2.0), (3.0, 4.0)], vec![2, 1]).unwrap();
        assert_eq!(
            isnumeric_builtin(Value::ComplexTensor(complex)).unwrap(),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn non_numeric_types_return_false() {
        assert_eq!(
            isnumeric_builtin(Value::Bool(true)).unwrap(),
            Value::Bool(false)
        );

        let logical = LogicalArray::new(vec![1, 0], vec![2, 1]).unwrap();
        assert_eq!(
            isnumeric_builtin(Value::LogicalArray(logical)).unwrap(),
            Value::Bool(false)
        );

        let chars = CharArray::new("rm".chars().collect(), 1, 2).unwrap();
        assert_eq!(
            isnumeric_builtin(Value::CharArray(chars)).unwrap(),
            Value::Bool(false)
        );

        assert_eq!(
            isnumeric_builtin(Value::String("runmat".into())).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            isnumeric_builtin(Value::Struct(StructValue::new())).unwrap(),
            Value::Bool(false)
        );
        let string_array =
            StringArray::new(vec!["foo".into(), "bar".into()], vec![1, 2]).expect("string array");
        assert_eq!(
            isnumeric_builtin(Value::StringArray(string_array)).unwrap(),
            Value::Bool(false)
        );
        let cell =
            CellArray::new(vec![Value::Num(1.0), Value::Bool(false)], 1, 2).expect("cell array");
        assert_eq!(
            isnumeric_builtin(Value::Cell(cell)).unwrap(),
            Value::Bool(false)
        );
        let object = ObjectInstance::new("runmat.MockObject".into());
        assert_eq!(
            isnumeric_builtin(Value::Object(object)).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            isnumeric_builtin(Value::FunctionHandle("runmat_fun".into())).unwrap(),
            Value::Bool(false)
        );
        let closure = Closure {
            function_name: "anon".into(),
            captures: vec![Value::Num(1.0)],
        };
        assert_eq!(
            isnumeric_builtin(Value::Closure(closure)).unwrap(),
            Value::Bool(false)
        );
        let handle = HandleRef {
            class_name: "runmat.Handle".into(),
            target: GcPtr::null(),
            valid: true,
        };
        assert_eq!(
            isnumeric_builtin(Value::HandleObject(handle)).unwrap(),
            Value::Bool(false)
        );
        let listener = Listener {
            id: 1,
            target: GcPtr::null(),
            event_name: "changed".into(),
            callback: GcPtr::null(),
            enabled: true,
            valid: true,
        };
        assert_eq!(
            isnumeric_builtin(Value::Listener(listener)).unwrap(),
            Value::Bool(false)
        );
        assert_eq!(
            isnumeric_builtin(Value::ClassRef("pkg.Class".into())).unwrap(),
            Value::Bool(false)
        );
        let mex = MException::new("MATLAB:mock".into(), "message".into());
        assert_eq!(
            isnumeric_builtin(Value::MException(mex)).unwrap(),
            Value::Bool(false)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gpu_numeric_and_logical_handles() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let numeric_handle = provider.upload(&view).expect("upload");
            let numeric = isnumeric_builtin(Value::GpuTensor(numeric_handle.clone())).unwrap();
            assert_eq!(numeric, Value::Bool(true));

            let logical_value = gpu_helpers::logical_gpu_value(numeric_handle.clone());
            let logical = isnumeric_builtin(logical_value).unwrap();
            assert_eq!(logical, Value::Bool(false));

            runmat_accelerate_api::clear_handle_logical(&numeric_handle);
            provider.free(&numeric_handle).ok();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn isnumeric_wgpu_handles_respect_metadata() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![4, 1];
        let view = HostTensorView {
            data: &data,
            shape: &shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let numeric = isnumeric_builtin(Value::GpuTensor(handle.clone())).unwrap();
        assert_eq!(numeric, Value::Bool(true));

        let logical_value = gpu_helpers::logical_gpu_value(handle.clone());
        let logical = isnumeric_builtin(logical_value).unwrap();
        assert_eq!(logical, Value::Bool(false));

        runmat_accelerate_api::clear_handle_logical(&handle);
        provider.free(&handle).ok();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
