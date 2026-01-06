//! MATLAB-compatible `gather` builtin with provider-aware semantics.

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::make_cell;
use runmat_builtins::Value;
use runmat_macros::runtime_builtin;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = "gather",
        builtin_path = "crate::builtins::acceleration::gpu::gather"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "gather"
category: "acceleration/gpu"
keywords: ["gather", "gpuArray", "download", "host copy", "accelerate", "residency"]
summary: "Transfer gpuArray data back to host memory, recursively handling cells, structs, and objects."
references:
  - https://www.mathworks.com/help/parallel-computing/gather.html
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Executes on the CPU; gpuArray inputs are downloaded through the provider's `download` hook and residency metadata is cleared so planners know the value now lives on the host."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::acceleration::gpu::gather::tests"
  integration: "builtins::acceleration::gpu::gather::tests::gather_downloads_gpu_tensor"
  wgpu: "builtins::acceleration::gpu::gather::tests::gather_wgpu_provider_roundtrip"
---

# What does the `gather` function do in MATLAB / RunMat?
`gather(X)` copies data that resides on the GPU or in another distributed storage back into host
memory. In RunMat, this means turning `gpuArray` handles into dense MATLAB values while leaving input
values that are already on the CPU unchanged.

## How does the `gather` function behave in MATLAB / RunMat?
- Accepts any MATLAB value. Non-GPU inputs (numbers, logicals, structs, strings, etc.) pass through
  untouched, so `gather` is safe to call unconditionally at API boundaries.
- Downloads gpuArray tensors via the active acceleration provider, producing dense double-precision
  matrices. Logical gpuArray inputs return logical arrays with MATLAB-compatible 0/1 encoding.
- Recursively descends into cells, structs, and objects, gathering every nested gpuArray handle. This
  mirrors MATLAB's behaviour when you gather composite data structures.
- Clears residency metadata so the auto-offload planner treats the gathered value as host-resident.
- Supports multiple inputs: in a single-output context it returns a `1×N` cell array preserving the
  original order; in a multi-output assignment the number of inputs and outputs must match, mirroring
  MATLAB's requirement.
- Raises `gather: no acceleration provider registered` when you attempt to download gpuArray data
  without an active provider, and propagates provider-specific download errors verbatim.

## `gather` Function GPU Execution Behaviour
`gather` itself runs on the CPU. When the input contains gpuArray handles, the builtin calls the
provider's `download` hook to retrieve a `HostTensorOwned` view, converts the result into MATLAB data,
and clears residency via `runmat_accelerate_api::clear_residency`. If the provider does not implement
`download`, the builtin surfaces the provider error so you know the backend must be extended. When the
input is already on the host, no provider work is required.

## GPU residency in RunMat (Do I need `gpuArray`?)
RunMat's auto-offload planner keeps tensors on the GPU until a builtin marked as a sink (such as
`gather`, plotting functions, or I/O) requests host access. You usually call `gather` at API
boundaries, for example to log results or hand them to CPU-only libraries. If the upstream computation
never leaves the GPU, you can omit `gather` and keep chaining gpu-aware builtins.

## Examples of using the `gather` function in MATLAB / RunMat

### Converting a gpuArray back to host memory
```matlab
G = gpuArray([1 2 3; 4 5 6]);
H = gather(G);
```
Expected output:
```matlab
H =
     1     2     3
     4     5     6
```

### Gathering data that is already on the CPU
```matlab
x = [10 20 30];
y = gather(x);
```
Expected output:
```matlab
y =
    10    20    30
```

### Preserving logical values when gathering
```matlab
mask = gpuArray(logical([1 0 1 0]));
hostMask = gather(mask);
```
Expected output:
```matlab
hostMask =
  1×4 logical array
   1   0   1   0
```

### Gathering gpuArray values stored inside a cell array
```matlab
C = {gpuArray([1 2]), 42};
hostC = gather(C);
```
Expected output:
```matlab
hostC =
  1×2 cell array
    {[1 2]}    {[42]}
```

### Gathering struct fields that live on the GPU
```matlab
S.data = gpuArray(magic(3));
S.label = "gpu result";
S_host = gather(S);
```
Expected output:
```matlab
S_host =
  struct with fields:
     data: [3×3 double]
    label: "gpu result"
```

### Gathering multiple gpuArrays into one cell result
```matlab
A = gpuArray(eye(3));
B = gpuArray(ones(3));
cellOut = gather(A, B);
```
Expected output:
```matlab
cellOut =
  1×2 cell array
    {[3×3 double]}    {[3×3 double]}
```

### Gathering results at the end of a GPU pipeline
```matlab
A = gpuArray(rand(1024, 1));
B = sin(A) .* 5;
result = gather(B);
```
Expected output (first three elements shown):
```matlab
result(1:3) =
    4.1377
    2.4884
    0.1003
```

## FAQ

### Does `gather` modify the original gpuArray?
No. `gather` returns a host-side copy. The original gpuArray value remains valid and continues to
reside on the GPU until it goes out of scope.

### What happens if the input does not live on the GPU?
Nothing changes—the value is returned as-is. This makes `gather` safe to sprinkle into code paths
that may or may not run on the GPU.

### How are logical gpuArray values represented after gathering?
Logical handles are tagged during `gpuArray` creation. `gather` reads that metadata and produces a
MATLAB logical array with the same shape, ensuring comparisons like `isa(result, 'logical')` behave
as expected.

### Does `gather` recurse into cells, structs, and objects?
Yes. Every nested gpuArray handle inside a cell array, struct field, or object property is downloaded
and replaced with host data.

### What happens when I pass multiple inputs but capture a single output?
RunMat follows MATLAB: it gathers each input and returns a `1×N` cell array so you can unpack values
later. In multi-output assignments you must request the same number of outputs as inputs.

### What if no acceleration provider is registered?
RunMat raises `gather: no acceleration provider registered` when you attempt to gather a gpuArray
without an active provider. Register a provider (for example, via `runmat-accelerate`) before calling
`gather`.

### Does `gather` free GPU memory automatically?
No. The gpuArray remains on the device. Free the handle explicitly (by clearing the variable) if you
no longer need it.

## See Also
[gpuArray](./gpuarray), [gpuDevice](./gpudevice), [sum](./sum), [mean](./mean)

## Source & Feedback
- Source: [`crates/runmat-runtime/src/builtins/acceleration/gpu/gather.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/acceleration/gpu/gather.rs)
- Found a bug? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::acceleration::gpu::gather")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "gather",
    op_kind: GpuOpKind::Custom("gather"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("download")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Downloads gpuArray handles via the provider's `download` hook and clears residency metadata; host inputs pass through unchanged.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::acceleration::gpu::gather")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "gather",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Acts as a residency sink for fusion planning; always materialises host data and clears gpuArray residency tracking.",
};

#[runtime_builtin(
    name = "gather",
    category = "acceleration/gpu",
    summary = "Bring gpuArray data back to host memory.",
    keywords = "gather,gpuArray,accelerate,download",
    accel = "sink",
    builtin_path = "crate::builtins::acceleration::gpu::gather"
)]
fn gather_builtin(args: Vec<Value>) -> Result<Value, String> {
    let eval = evaluate(&args)?;
    let len = eval.len();
    if len == 1 {
        Ok(eval.into_first())
    } else {
        let outputs = eval.into_outputs();
        make_cell(outputs, 1, len)
    }
}

/// Combined gather result used by single- and multi-output call sites.
#[derive(Debug, Clone)]
pub struct GatherResult {
    outputs: Vec<Value>,
}

impl GatherResult {
    fn new(outputs: Vec<Value>) -> Self {
        Self { outputs }
    }

    /// Number of gathered outputs.
    pub fn len(&self) -> usize {
        self.outputs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.outputs.is_empty()
    }

    /// Borrowed slice of outputs (in call-order).
    pub fn outputs(&self) -> &[Value] {
        &self.outputs
    }

    /// Consume the result, yielding all outputs.
    pub fn into_outputs(self) -> Vec<Value> {
        self.outputs
    }

    /// Consume the result, yielding the first output (requires at least one input).
    pub fn into_first(self) -> Value {
        self.outputs
            .into_iter()
            .next()
            .expect("gather requires at least one input")
    }
}

/// Evaluate `gather` for arbitrary argument lists and return all outputs.
pub fn evaluate(args: &[Value]) -> Result<GatherResult, String> {
    if args.is_empty() {
        return Err("gather: not enough input arguments".to_string());
    }
    let mut outputs = Vec::with_capacity(args.len());
    for value in args {
        outputs.push(gather_argument(value)?);
    }
    Ok(GatherResult::new(outputs))
}

fn gather_argument(value: &Value) -> Result<Value, String> {
    match crate::dispatcher::gather_if_needed(value) {
        Ok(val) => Ok(val),
        Err(err) => {
            if err.trim_start().to_ascii_lowercase().starts_with("gather:") {
                Err(err)
            } else {
                Err(format!("gather: {err}"))
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{CellArray, StructValue, Tensor};

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gather_passes_through_host_values() {
        let value = Value::Num(42.0);
        let result = gather_builtin(vec![value.clone()]).expect("gather");
        assert_eq!(result, value);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gather_downloads_gpu_tensor() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = gather_builtin(vec![Value::GpuTensor(handle)]).expect("gather");
            match result {
                Value::Tensor(host) => {
                    assert_eq!(host.shape, tensor.shape);
                    assert_eq!(host.data, tensor.data);
                }
                other => panic!("expected tensor result, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gather_preserves_logical_gpu_tensors() {
        test_support::with_test_provider(|provider| {
            let data = vec![0.0, 1.0, 1.0, 0.0];
            let tensor = Tensor::new(data.clone(), vec![2, 2]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            runmat_accelerate_api::set_handle_logical(&handle, true);
            let result = gather_builtin(vec![Value::GpuTensor(handle)]).expect("gather");
            match result {
                Value::LogicalArray(logical) => {
                    assert_eq!(logical.shape, vec![2, 2]);
                    assert_eq!(logical.data, vec![0, 1, 1, 0]);
                }
                other => panic!("expected logical array, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gather_recurses_into_cells() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![7.0, 8.0], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let cell = CellArray::new(vec![Value::GpuTensor(handle), Value::from("host")], 1, 2)
                .expect("cell");
            let result = gather_builtin(vec![Value::Cell(cell)]).expect("gather");
            let Value::Cell(gathered) = result else {
                panic!("expected cell result");
            };
            let first = gathered.get(0, 0).expect("first element");
            match first {
                Value::Tensor(t) => {
                    assert_eq!(t.shape, vec![2, 1]);
                    assert_eq!(t.data, tensor.data);
                }
                other => panic!("expected tensor in cell, got {other:?}"),
            }
            let second = gathered.get(0, 1).expect("second element");
            assert_eq!(second, Value::from("host"));
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gather_recurses_into_structs() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![3.5, -1.25], vec![2, 1]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let mut st = StructValue::new();
            st.insert("data", Value::GpuTensor(handle));
            st.insert("label", Value::from("gpu result"));

            let result = gather_builtin(vec![Value::Struct(st)]).expect("gather");
            let Value::Struct(gathered) = result else {
                panic!("expected struct result");
            };
            let Some(Value::Tensor(host)) = gathered.fields.get("data") else {
                panic!("missing tensor field");
            };
            assert_eq!(host.shape, vec![2, 1]);
            assert_eq!(host.data, tensor.data);
            let Some(Value::String(label)) = gathered.fields.get("label") else {
                panic!("missing label");
            };
            assert_eq!(label, "gpu result");
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gather_returns_cell_for_multiple_inputs() {
        let result =
            gather_builtin(vec![Value::Num(1.0), Value::from("two")]).expect("gather cell");
        let Value::Cell(cell) = result else {
            panic!("expected cell for multiple inputs");
        };
        assert_eq!(cell.rows, 1);
        assert_eq!(cell.cols, 2);
        assert_eq!(cell.get(0, 0).unwrap(), Value::Num(1.0));
        assert_eq!(cell.get(0, 1).unwrap(), Value::from("two"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn evaluate_returns_outputs_in_order() {
        let eval =
            evaluate(&[Value::Num(5.0), Value::Bool(true), Value::from("hello")]).expect("eval");
        assert_eq!(eval.len(), 3);
        assert_eq!(eval.outputs()[0], Value::Num(5.0));
        assert_eq!(eval.outputs()[1], Value::Bool(true));
        assert_eq!(eval.outputs()[2], Value::from("hello"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn gather_requires_at_least_one_argument() {
        let err = gather_builtin(Vec::new()).expect_err("expected error");
        assert_eq!(err, "gather: not enough input arguments");
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn gather_wgpu_provider_roundtrip() {
        use runmat_accelerate_api::AccelProvider;

        match runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) {
            Ok(provider) => {
                let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
                let view = HostTensorView {
                    data: &tensor.data,
                    shape: &tensor.shape,
                };
                let handle = provider.upload(&view).expect("upload");
                let eval = evaluate(&[Value::GpuTensor(handle.clone())]).expect("evaluate");
                let outputs = eval.into_outputs();
                assert_eq!(outputs.len(), 1);
                match outputs.into_iter().next().unwrap() {
                    Value::Tensor(host) => {
                        assert_eq!(host.shape, tensor.shape);
                        assert_eq!(host.data, tensor.data);
                    }
                    other => panic!("expected tensor value, got {other:?}"),
                }
                let _ = provider.free(&handle);
            }
            Err(err) => {
                tracing::warn!("Skipping gather_wgpu_provider_roundtrip: {err}");
            }
        }
        // Restore the simple provider so subsequent tests see a predictable backend.
        runmat_accelerate::simple_provider::register_inprocess_provider();
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
