//! MATLAB-compatible `isreal` builtin with GPU-aware semantics for RunMat.
//!
//! This predicate reports whether a value is stored without an imaginary
//! component. Unlike `isfinite`/`isnan`, it returns a single logical scalar.

use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
#[cfg(feature = "doc_export")]
#[runmat_macros::register_doc_text(name = "isreal")]
pub const DOC_MD: &str = r#"---
title: "isreal"
category: "logical/tests"
keywords: ["isreal", "complex storage", "real array", "gpuArray isreal", "MATLAB isreal"]
summary: "Return true when a value uses real storage without an imaginary component."
references: []
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Queries provider metadata via `logical_isreal`; falls back to a host-side check when a backend does not expose the hook."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::logical::tests::isreal::tests"
  integration: "builtins::logical::tests::isreal::tests::isreal_gpu_roundtrip"
---

# What does the `isreal` function do in MATLAB / RunMat?
`tf = isreal(x)` returns a logical scalar that is `true` when the value `x` is stored without an imaginary component, and `false` otherwise. It mirrors MATLAB's storage-centric definition: any complex storage (even with zero-valued imaginary parts) is reported as not real.

## How does the `isreal` function behave in MATLAB / RunMat?
- Real numeric scalars and dense tensors return `true`.
- Logical arrays, `duration`, `calendarDuration`, and character arrays always return `true`.
- Complex scalars and `ComplexTensor` values return `false`, even when every imaginary component equals zero, because the data uses complex storage.
- `string`, `table`, `cell`, `struct`, `datetime`, `function_handle`, and object values always return `false`, matching MATLAB's documented rules.
- The return value is always a logical scalar; `isreal` does **not** produce per-element masks.

## Examples of using the `isreal` function in MATLAB / RunMat

### Checking if a real-valued matrix is stored as real

```matlab
A = [7 3 2; 2 1 12; 52 108 78];
tf = isreal(A);
```

Expected output:

```matlab
tf =
     1
```

### Detecting complex entries inside an array

```matlab
B = [1 3+4i 2; 2i 1 12];
tf = isreal(B);
```

Expected output:

```matlab
tf =
     0
```

### Complex storage with zero-valued imaginary parts still reports false

```matlab
C = complex(12);   % Stored as complex double with 0 imaginary part
tf = isreal(C);
```

Expected output:

```matlab
tf =
     0
```

### Logical and character data are considered real

```matlab
mask = logical([1 0 1]);
chars = ['R' 'u' 'n'];
tf_mask = isreal(mask);
tf_chars = isreal(chars);
```

Expected output:

```matlab
tf_mask =
     1

tf_chars =
     1
```

### Strings, cells, and structs are never real

```matlab
txt = "RunMat";
vec = {1, 2};
person = struct("name", "Ada");
tf_txt = isreal(txt);
tf_vec = isreal(vec);
tf_person = isreal(person);
```

Expected output:

```matlab
tf_txt =
     0

tf_vec =
     0

tf_person =
     0
```

### Querying `gpuArray` storage without gathering data

```matlab
G = gpuArray(rand(1024, 1024));
tf_gpu = isreal(G);
```

Expected output:

```matlab
tf_gpu =
     1
```

## `isreal` Function GPU Execution Behaviour
When RunMat Accelerate is active, the runtime asks the registered provider for the `logical_isreal` hook. A provider can answer the query using device metadata without downloading the tensor. If the hook is unavailable, RunMat gathers the value once and performs the storage check on the host, so the builtin always returns a result.

## GPU residency in RunMat (Do I need `gpuArray`?)
Normally you do **not** need to call `gpuArray` explicitly. RunMat's auto-offload planner tracks when tensors already reside on the GPU and keeps them there. `isreal` simply inspects storage metadata and returns a host logical scalar, gathering device data only as a fallback.

## FAQ

### Does `isreal` inspect each element of the array?
No. It is a storage-level predicate. Use `imag(A) == 0` or `A == real(A)` if you need per-element checks.

### Why does `isreal` return false for `complex(5)` or `1 + 0i`?
Those expressions allocate complex storage even though the imaginary part is zero. MATLAB and RunMat both treat that storage as complex, so `isreal` returns `false`.

### What about logical arrays, durations, or character data?
They always return `true` because those types never allocate an imaginary component.

### Why do strings, cells, structs, or objects return false?
MATLAB documents that `isreal` returns `false` for those container and object types. RunMat follows the same rules for compatibility.

### Will `isreal` trigger GPU computation or kernel launches?
No. It is a metadata query. Providers can answer it without dispatching kernels, and the runtime falls back to a single host download only when necessary.

### How can I check whether each element is real on the GPU?
Use elementwise predicates such as `imag`/`real` combined with comparison (`imag(A) == 0`) or express the logic with fused operations. `isreal` is intentionally scalar.

## See Also
[isfinite](./isfinite), [isinf](./isinf), [isnan](./isnan), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "isreal",
    op_kind: GpuOpKind::Custom("storage-check"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("logical_isreal")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Queries provider metadata when `logical_isreal` is available; otherwise gathers once and inspects host storage.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "isreal",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Scalar metadata predicate that remains outside fusion graphs.",
};

#[runtime_builtin(
    name = "isreal",
    category = "logical/tests",
    summary = "Return true when a value uses real storage without an imaginary component.",
    keywords = "isreal,real,complex,gpu,logical",
    accel = "metadata"
)]
fn isreal_builtin(value: Value) -> Result<Value, String> {
    match value {
        Value::GpuTensor(handle) => isreal_gpu(handle),
        other => isreal_host(other),
    }
}

fn isreal_gpu(handle: GpuTensorHandle) -> Result<Value, String> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(flag) = provider.logical_isreal(&handle) {
            return Ok(Value::Bool(flag));
        }
    }

    let gpu_value = Value::GpuTensor(handle);
    let gathered = gpu_helpers::gather_value(&gpu_value)?;
    isreal_host(gathered)
}

fn isreal_host(value: Value) -> Result<Value, String> {
    let flag = match value {
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => true,
        Value::Tensor(_) => true,
        Value::LogicalArray(_) => true,
        Value::CharArray(_) => true,
        Value::Complex(_, _) => false,
        Value::ComplexTensor(_) => false,
        Value::String(_) => false,
        Value::StringArray(_) => false,
        Value::Struct(_) => false,
        Value::Cell(_) => false,
        Value::Object(_) => false,
        Value::HandleObject(_) => false,
        Value::Listener(_) => false,
        Value::FunctionHandle(_) => false,
        Value::Closure(_) => false,
        Value::ClassRef(_) => false,
        Value::MException(_) => false,
        Value::GpuTensor(_) => {
            return Err("isreal: internal error, GPU value reached host path".to_string());
        }
    };
    Ok(Value::Bool(flag))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_builtins::{
        CellArray, CharArray, Closure, ComplexTensor, HandleRef, Listener, LogicalArray,
        MException, ObjectInstance, StructValue, Tensor,
    };
    use runmat_gc_api::GcPtr;

    #[test]
    fn isreal_reports_true_for_real_scalars() {
        let real = isreal_builtin(Value::Num(42.0)).expect("isreal");
        let integer = isreal_builtin(Value::from(5_i32)).expect("isreal");
        let boolean = isreal_builtin(Value::Bool(false)).expect("isreal");
        assert_eq!(real, Value::Bool(true));
        assert_eq!(integer, Value::Bool(true));
        assert_eq!(boolean, Value::Bool(true));
    }

    #[test]
    fn isreal_rejects_complex_storage_even_with_zero_imaginary_part() {
        let complex = isreal_builtin(Value::Complex(3.0, 4.0)).expect("isreal");
        let complex_zero_imag = isreal_builtin(Value::Complex(12.0, 0.0)).expect("isreal");
        let complex_tensor = ComplexTensor::new(vec![(1.0, 0.0), (2.0, -1.0)], vec![2, 1]).unwrap();
        let tensor_flag = isreal_builtin(Value::ComplexTensor(complex_tensor)).expect("isreal");
        assert_eq!(complex, Value::Bool(false));
        assert_eq!(complex_zero_imag, Value::Bool(false));
        assert_eq!(tensor_flag, Value::Bool(false));
    }

    #[test]
    fn isreal_handles_array_and_container_types() {
        let tensor = Tensor::new(vec![1.0, -2.0, 3.5], vec![3, 1]).unwrap();
        let logical = LogicalArray::new(vec![1, 0, 1], vec![3, 1]).unwrap();
        let chars = CharArray::new_row("RunMat");
        let string_flag = isreal_builtin(Value::from("RunMat")).expect("isreal");
        let string_array =
            runmat_builtins::StringArray::new(vec!["a".into(), "b".into()], vec![2]).unwrap();
        let cell = CellArray::new(vec![Value::Num(1.0)], 1, 1).unwrap();
        let mut fields = StructValue::new();
        fields.fields.insert("name".into(), Value::from("Ada"));
        let object = ObjectInstance::new("RunMat.Object".into());

        let tensor_flag = isreal_builtin(Value::Tensor(tensor)).expect("isreal");
        let logical_flag = isreal_builtin(Value::LogicalArray(logical)).expect("isreal");
        let char_flag = isreal_builtin(Value::CharArray(chars)).expect("isreal");
        let string_array_flag =
            isreal_builtin(Value::StringArray(string_array)).expect("isreal string array");
        let cell_flag = isreal_builtin(Value::Cell(cell)).expect("isreal cell");
        let struct_flag = isreal_builtin(Value::Struct(fields)).expect("isreal struct");
        let object_flag = isreal_builtin(Value::Object(object)).expect("isreal object");

        assert_eq!(tensor_flag, Value::Bool(true));
        assert_eq!(logical_flag, Value::Bool(true));
        assert_eq!(char_flag, Value::Bool(true));
        assert_eq!(string_flag, Value::Bool(false));
        assert_eq!(string_array_flag, Value::Bool(false));
        assert_eq!(cell_flag, Value::Bool(false));
        assert_eq!(struct_flag, Value::Bool(false));
        assert_eq!(object_flag, Value::Bool(false));
    }

    #[test]
    fn isreal_handles_function_and_handle_like_types() {
        let function_flag =
            isreal_builtin(Value::FunctionHandle("runmat_builtin".into())).expect("isreal fn");
        let closure_flag = isreal_builtin(Value::Closure(Closure {
            function_name: "anon".into(),
            captures: vec![Value::Num(1.0)],
        }))
        .expect("isreal closure");
        let handle_flag = isreal_builtin(Value::HandleObject(HandleRef {
            class_name: "MockHandle".into(),
            target: GcPtr::null(),
            valid: true,
        }))
        .expect("isreal handle");
        let listener_flag = isreal_builtin(Value::Listener(Listener {
            id: 42,
            target: GcPtr::null(),
            event_name: "changed".into(),
            callback: GcPtr::null(),
            enabled: true,
            valid: true,
        }))
        .expect("isreal listener");
        let class_ref_flag =
            isreal_builtin(Value::ClassRef("pkg.Class".into())).expect("isreal classref");
        let mex_flag = isreal_builtin(Value::MException(MException::new(
            "MATLAB:mock".into(),
            "message".into(),
        )))
        .expect("isreal mexception");

        assert_eq!(function_flag, Value::Bool(false));
        assert_eq!(closure_flag, Value::Bool(false));
        assert_eq!(handle_flag, Value::Bool(false));
        assert_eq!(listener_flag, Value::Bool(false));
        assert_eq!(class_ref_flag, Value::Bool(false));
        assert_eq!(mex_flag, Value::Bool(false));
    }

    #[test]
    fn isreal_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = isreal_builtin(Value::GpuTensor(handle)).expect("isreal gpu");
            assert_eq!(result, Value::Bool(true));
        });
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn isreal_wgpu_provider_reports_true() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );
        let tensor = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = runmat_accelerate_api::provider()
            .unwrap()
            .upload(&view)
            .expect("upload");
        let result = isreal_builtin(Value::GpuTensor(handle)).expect("isreal gpu");
        assert_eq!(result, Value::Bool(true));
    }
}
