//! MATLAB-compatible `ctranspose` builtin with GPU-aware semantics for RunMat.
//!
//! This module mirrors MATLAB's conjugate-transpose operator (`A'`) across numeric,
//! logical, string, char, and cell arrays while integrating with RunMat Accelerate to
//! preserve GPU residency whenever possible.

use crate::builtins::array::shape::permute::{
    permute_complex_tensor, permute_logical_array, permute_string_array, permute_tensor,
};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use log::warn;
use runmat_accelerate_api::{GpuTensorHandle, HostTensorView};
use runmat_builtins::{
    CellArray, CharArray, ComplexTensor, LogicalArray, StringArray, Tensor, Value,
};
use runmat_macros::runtime_builtin;

const NAME: &str = "ctranspose";

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(
        name = NAME,
        builtin_path = "crate::builtins::math::linalg::ops::ctranspose"
    )
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "ctranspose"
category: "math/linalg/ops"
keywords: ["ctranspose", "conjugate transpose", "hermitian", "gpu", "matrix transpose"]
summary: "Swap the first two dimensions of arrays and conjugate complex values."
references: ["https://www.mathworks.com/help/matlab/ref/ctranspose.html"]
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "none"
  notes: "Invokes provider transpose/permute plus unary_conj hooks when available; otherwise gathers to host, applies the conjugate transpose, and re-uploads."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::linalg::ops::ctranspose::tests"
  integration: "builtins::math::linalg::ops::ctranspose::tests::ctranspose_gpu_roundtrip"
  gpu: "builtins::math::linalg::ops::ctranspose::tests::ctranspose_wgpu_matches_cpu"
  doc: "builtins::math::linalg::ops::ctranspose::tests::doc_examples_present"
---

# What does the `ctranspose` function do in MATLAB / RunMat?
`B = ctranspose(A)` (or `B = A'`) flips the first two dimensions of `A` **and** conjugates
complex values. Real-valued inputs therefore behave like `transpose`, while complex inputs
receive Hermitian conjugation.

## How does the `ctranspose` function behave in MATLAB / RunMat?
- Works for scalars, vectors, matrices, and N-D arrays; only the first two axes are swapped.
- Real numeric, logical, and character data are not changed by conjugation.
- Complex values receive element-wise conjugation after the transpose (`(a + bi)' = a - bi`).
- Character arrays and cell arrays preserve their types; `ctranspose` simply rearranges entries.
- String scalars are passed through unchanged; string arrays transpose like MATLAB.
- Empty arrays and singleton dimensions follow MATLAB's column-major semantics.

## `ctranspose` Function GPU Execution Behaviour
When a tensor resides on the GPU, RunMat first asks the active acceleration provider to execute
the transpose and conjugation in-device:

- **Provider support:** If the backend exposes both `transpose` (or `permute`) and
  `unary_conj`, the entire operation happens on the device without a gather.
- **Partial hooks:** If the transpose succeeds but conjugation fails, RunMat falls back to the
  host path while logging a warning so users know their backend is incomplete.
- **No hooks:** RunMat gathers the tensor, applies the conjugate transpose on the CPU, and
  re-uploads the result when possible so downstream kernels can keep running on the GPU.

Current providers operate on real double-precision tensors; complex GPU tensors will gather to
the host until native complex layouts are implemented.

## Examples of using the `ctranspose` function in MATLAB / RunMat

### Conjugate transpose of a complex matrix
```matlab
Z = [1+2i 3-4i; 5+0i 6-7i];
H = ctranspose(Z);
```
Expected output:
```matlab
H =
   1 - 2i   5 - 0i
   3 + 4i   6 + 7i
```

### Conjugate transpose of a real matrix equals the plain transpose
```matlab
A = [1 2 3; 4 5 6];
B = ctranspose(A);
```
Expected output:
```matlab
B =
     1     4
     2     5
     3     6
```

### Conjugate transpose turns row vectors into column vectors
```matlab
row = [1-2i, 3+4i, 5];
col = ctranspose(row);
size(col)
```
Expected output:
```matlab
ans = [3 1]
```

### Conjugate transpose of a complex scalar
```matlab
z = 2 + 3i;
result = ctranspose(z);
```
Expected output:
```matlab
result = 2 - 3i;
```

### Conjugate transpose of text data preserves characters
```matlab
C = ['r' 'u' 'n'; 'm' 'a' 't'];
CT = ctranspose(C);
```
Expected output:
```matlab
CT =
    'rm'
    'ua'
    'nt'
```

### Conjugate transpose of a gpuArray without leaving the device
```matlab
G = gpuArray(rand(1024, 64) + 1i * rand(1024, 64));
GT = ctranspose(G);
```
`GT` stays on the GPU when the provider implements the needed hooks; otherwise RunMat gathers,
applies the conjugate transpose, and uploads the result transparently.

## GPU residency in RunMat (Do I need `gpuArray`?)
No additional residency management is required. If the planner keeps your data on the GPU,
`ctranspose` honours that residency and either executes on the device (when hooks are present) or
performs a gather/transpose/upload round-trip automatically.

## FAQ
1. **How is `ctranspose` different from `transpose`?**  
   `transpose` (`A.'`) swaps dimensions without conjugation; `ctranspose` (`A'`) also conjugates
   complex values. For purely real data they are identical.
2. **Does `ctranspose` change logical or character arrays?**  
   Only their layout changes. Values remain logical or character, and conjugation has no effect.
3. **What about higher-dimensional arrays?**  
   Only the first two axes are swapped; trailing dimensions stay in-place, matching MATLAB.
4. **Does the result share storage with the input?**  
   No. `ctranspose` materialises a fresh array, although fusion may eliminate the copy in
   optimised pipelines.
5. **How are complex tensors handled on the GPU today?**  
   Complex gpuArray support is still in flight. When complex data appears, RunMat gathers to the
   host, applies the conjugate transpose, and re-uploads if needed.
6. **Will `ctranspose` fuse with neighbouring kernels?**  
   Conjugate transposes currently act as fusion boundaries so that shape changes are visible to
   downstream kernels.
7. **Can I rely on `ctranspose` inside linear-algebra routines (e.g., Hermitian products)?**  
   Yes. The builtin mirrors MATLAB semantics precisely and is safe to use inside idioms like
   `A' * A`.
8. **What error do I get for unsupported types?**  
   Non-numeric objects (e.g., structs) raise `ctranspose: unsupported input type ...`, matching
   MATLAB's strict type checks.

## See Also
[transpose](./transpose), [conj](./conj), [mtimes](./mtimes), [gpuArray](./gpuarray), [gather](./gather)

## Source & Feedback
- Implementation: [`crates/runmat-runtime/src/builtins/math/linalg/ops/ctranspose.rs`](https://github.com/runmat-org/runmat/blob/main/crates/runmat-runtime/src/builtins/math/linalg/ops/ctranspose.rs)
- Found a behavioural difference? Please [open an issue](https://github.com/runmat-org/runmat/issues/new/choose) with details and a minimal repro.
"#;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::linalg::ops::ctranspose")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Transpose,
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("transpose"),
        ProviderHook::Custom("permute"),
        ProviderHook::Unary { name: "unary_conj" },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Uses provider transpose/permute hooks followed by unary_conj; falls back to host conjugate transpose when either hook is missing.",
};

fn builtin_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message).with_builtin(NAME).build()
}

fn map_control_flow(err: RuntimeError) -> RuntimeError {
    let mut builder = build_runtime_error(err.message()).with_builtin(NAME);
    if let Some(identifier) = err.identifier() {
        builder = builder.with_identifier(identifier.to_string());
    }
    if let Some(task_id) = err.context.task_id.clone() {
        builder = builder.with_task_id(task_id);
    }
    if !err.context.call_stack.is_empty() {
        builder = builder.with_call_stack(err.context.call_stack.clone());
    }
    if let Some(phase) = err.context.phase.clone() {
        builder = builder.with_phase(phase);
    }
    builder.with_source(err).build()
}

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::linalg::ops::ctranspose"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Conjugate transposes act as fusion boundaries so downstream kernels observe the updated layout.",
};

#[runtime_builtin(
    name = "ctranspose",
    category = "math/linalg/ops",
    summary = "Swap the first two dimensions of arrays and conjugate complex values.",
    keywords = "ctranspose,conjugate transpose,hermitian,gpu",
    accel = "transpose",
    builtin_path = "crate::builtins::math::linalg::ops::ctranspose"
)]
async fn ctranspose_builtin(mut args: Vec<Value>) -> BuiltinResult<Value> {
    let value = match args.len() {
        0 => return Err(builtin_error("ctranspose: missing input argument")),
        1 => args.remove(0),
        _ => return Err(builtin_error("ctranspose: too many input arguments")),
    };
    match value {
        Value::GpuTensor(handle) => ctranspose_gpu(handle).await,
        Value::Complex(re, im) => ctranspose_complex_scalar(re, im),
        Value::ComplexTensor(ct) => ctranspose_complex_tensor(ct),
        Value::Tensor(t) => Ok(tensor::tensor_into_value(ctranspose_tensor(t)?)),
        Value::LogicalArray(la) => Ok(Value::LogicalArray(ctranspose_logical_array(la)?)),
        Value::CharArray(ca) => Ok(Value::CharArray(ctranspose_char_array(ca)?)),
        Value::StringArray(sa) => Ok(Value::StringArray(ctranspose_string_array(sa)?)),
        Value::Cell(ca) => Ok(Value::Cell(ctranspose_cell_array(ca)?)),
        Value::Num(n) => Ok(Value::Num(n)),
        Value::Int(i) => Ok(Value::Int(i)),
        Value::Bool(b) => Ok(Value::Bool(b)),
        Value::String(s) => Ok(Value::String(s)),
        other => Err(builtin_error(format!(
            "ctranspose: unsupported input type {other:?}"
        ))),
    }
}

fn ctranspose_tensor(tensor: Tensor) -> BuiltinResult<Tensor> {
    let rank = tensor.shape.len();
    if rank <= 2 {
        ctranspose_tensor_matrix(&tensor)
    } else {
        let order = ctranspose_order(rank);
        permute_tensor(NAME, tensor, &order)
    }
}

fn ctranspose_complex_tensor(ct: ComplexTensor) -> BuiltinResult<Value> {
    let rank = ct.shape.len();
    if rank == 0 {
        return ctranspose_complex_tensor_value(ct);
    }
    if rank <= 2 {
        let data = ctranspose_complex_matrix(&ct);
        let shape = vec![ct.cols, ct.rows];
        let transposed = ComplexTensor::new(data, shape.clone())
            .map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
        ctranspose_complex_tensor_value(transposed)
    } else {
        let order = ctranspose_order(rank);
        let permuted = permute_complex_tensor(NAME, ct, &order)?;
        ctranspose_complex_tensor_value(permuted)
    }
}

fn ctranspose_complex_tensor_value(ct: ComplexTensor) -> BuiltinResult<Value> {
    let shape = ct.shape.clone();
    let data = ct.data;
    let mut all_real = true;
    let mut conj_data = Vec::with_capacity(data.len());
    for (re, im) in data {
        let imag = -im;
        if imag != 0.0 || imag.is_nan() {
            all_real = false;
        }
        conj_data.push((re, imag));
    }
    if all_real {
        let real: Vec<f64> = conj_data.iter().map(|(re, _)| *re).collect();
        let tensor = Tensor::new(real, shape).map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
        Ok(tensor::tensor_into_value(tensor))
    } else {
        let tensor = ComplexTensor::new(conj_data, shape)
            .map_err(|e| builtin_error(format!("{NAME}: {e}")))?;
        Ok(Value::ComplexTensor(tensor))
    }
}

fn ctranspose_complex_scalar(re: f64, im: f64) -> BuiltinResult<Value> {
    let imag = -im;
    if imag == 0.0 && !imag.is_nan() {
        Ok(Value::Num(re))
    } else {
        Ok(Value::Complex(re, imag))
    }
}

fn ctranspose_logical_array(la: LogicalArray) -> BuiltinResult<LogicalArray> {
    let rank = la.shape.len();
    if rank == 0 {
        return Ok(la);
    }
    if rank <= 2 {
        let rows = la.shape.first().copied().unwrap_or(1);
        let cols = if rank >= 2 {
            la.shape.get(1).copied().unwrap_or(1)
        } else {
            1
        };
        let mut out = vec![0u8; la.data.len()];
        for i in 0..rows {
            for j in 0..cols {
                let src = i + j * rows;
                let dst = j + i * cols;
                if src < la.data.len() && dst < out.len() {
                    out[dst] = la.data[src];
                }
            }
        }
        let new_shape = vec![cols, rows];
        LogicalArray::new(out, new_shape).map_err(|e| builtin_error(format!("{NAME}: {e}")))
    } else {
        let order = ctranspose_order(rank);
        permute_logical_array(NAME, la, &order)
    }
}

fn ctranspose_char_array(ca: CharArray) -> BuiltinResult<CharArray> {
    let rows = ca.rows;
    let cols = ca.cols;
    if ca.data.is_empty() {
        return CharArray::new(Vec::new(), cols, rows)
            .map_err(|e| builtin_error(format!("{NAME}: {e}")));
    }
    let mut out = vec!['\0'; ca.data.len()];
    for r in 0..rows {
        for c in 0..cols {
            let src = r * cols + c;
            let dst = c * rows + r;
            if src < ca.data.len() && dst < out.len() {
                out[dst] = ca.data[src];
            }
        }
    }
    CharArray::new(out, cols, rows).map_err(|e| builtin_error(format!("{NAME}: {e}")))
}

fn ctranspose_string_array(sa: StringArray) -> BuiltinResult<StringArray> {
    let rank = sa.shape.len();
    if rank == 0 {
        return Ok(sa);
    }
    if rank <= 2 {
        let rows = sa.rows;
        let cols = sa.cols;
        let mut out = vec![String::new(); sa.data.len()];
        for r in 0..rows {
            for c in 0..cols {
                let src = r + c * rows;
                let dst = c + r * cols;
                if src < sa.data.len() && dst < out.len() {
                    out[dst] = sa.data[src].clone();
                }
            }
        }
        let new_shape = if rank >= 2 {
            let mut shape = sa.shape.clone();
            if shape.len() >= 2 {
                shape.swap(0, 1);
            }
            shape
        } else {
            vec![cols, rows]
        };
        StringArray::new(out, new_shape).map_err(|e| builtin_error(format!("{NAME}: {e}")))
    } else {
        let order = ctranspose_order(rank);
        permute_string_array(NAME, sa, &order)
    }
}

fn ctranspose_cell_array(ca: CellArray) -> BuiltinResult<CellArray> {
    let rows = ca.rows;
    let cols = ca.cols;
    let mut out = Vec::with_capacity(ca.data.len());
    for c in 0..cols {
        for r in 0..rows {
            let idx = r * cols + c;
            out.push(ca.data[idx].clone());
        }
    }
    CellArray::new_handles(out, cols, rows).map_err(|e| builtin_error(format!("{NAME}: {e}")))
}

async fn ctranspose_gpu(handle: GpuTensorHandle) -> BuiltinResult<Value> {
    let rank = handle.shape.len();
    if rank == 0 {
        return Ok(Value::GpuTensor(handle));
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        let mut transposed: Option<GpuTensorHandle> = None;
        if rank <= 2 {
            match provider.transpose(&handle) {
                Ok(out) => transposed = Some(out),
                Err(err) => {
                    let info = provider.device_info_struct();
                    warn!(
                        "ctranspose: provider {} (backend: {}) missing transpose hook; falling back ({err})",
                        info.name,
                        info.backend.as_deref().unwrap_or("unknown")
                    );
                }
            }
        } else {
            let order = ctranspose_order(rank);
            let zero_based: Vec<usize> = order.iter().map(|&idx| idx - 1).collect();
            match provider.permute(&handle, &zero_based) {
                Ok(out) => transposed = Some(out),
                Err(err) => {
                    let info = provider.device_info_struct();
                    warn!(
                        "ctranspose: provider {} (backend: {}) missing permute hook; falling back ({err})",
                        info.name,
                        info.backend.as_deref().unwrap_or("unknown")
                    );
                }
            }
        }

        if let Some(transposed_handle) = transposed {
            match provider.unary_conj(&transposed_handle) {
                Ok(conjugated) => {
                    if let Some(info) =
                        runmat_accelerate_api::handle_transpose_info(&transposed_handle)
                    {
                        runmat_accelerate_api::record_handle_transpose(
                            &conjugated,
                            info.base_rows,
                            info.base_cols,
                        );
                    }
                    return Ok(Value::GpuTensor(conjugated));
                }
                Err(err) => {
                    let info = provider.device_info_struct();
                    warn!(
                        "ctranspose: provider {} (backend: {}) missing unary_conj hook; falling back ({err})",
                        info.name,
                        info.backend.as_deref().unwrap_or("unknown")
                    );
                }
            }
        }
    }

    let host = gpu_helpers::gather_tensor_async(&handle)
        .await
        .map_err(map_control_flow)?;
    let transposed = ctranspose_tensor(host)?;
    if let Some(provider) = runmat_accelerate_api::provider() {
        let view = HostTensorView {
            data: &transposed.data,
            shape: &transposed.shape,
        };
        match provider.upload(&view) {
            Ok(uploaded) => return Ok(Value::GpuTensor(uploaded)),
            Err(err) => warn!(
                "ctranspose: re-upload after host fallback failed; returning host tensor ({err})"
            ),
        }
    }
    Ok(tensor::tensor_into_value(transposed))
}

fn ctranspose_order(rank: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (1..=rank.max(2)).collect();
    if order.len() >= 2 {
        order.swap(0, 1);
    }
    if order.len() > rank && rank < 2 {
        order.truncate(rank.max(2));
    }
    order
}

fn ctranspose_tensor_matrix(tensor: &Tensor) -> BuiltinResult<Tensor> {
    let rows = tensor.rows();
    let cols = tensor.cols();
    if tensor.data.is_empty() {
        return Tensor::new(Vec::new(), vec![cols, rows])
            .map_err(|e| builtin_error(format!("{NAME}: {e}")));
    }
    let mut out = vec![0.0; tensor.data.len()];
    for r in 0..rows {
        for c in 0..cols {
            let src = r + c * rows;
            let dst = c + r * cols;
            if src < tensor.data.len() && dst < out.len() {
                out[dst] = tensor.data[src];
            }
        }
    }
    Tensor::new(out, vec![cols, rows]).map_err(|e| builtin_error(format!("{NAME}: {e}")))
}

fn ctranspose_complex_matrix(ct: &ComplexTensor) -> Vec<(f64, f64)> {
    let rows = ct.rows;
    let cols = ct.cols;
    if ct.data.is_empty() {
        return Vec::new();
    }
    let mut out = vec![(0.0, 0.0); ct.data.len()];
    for r in 0..rows {
        for c in 0..cols {
            let src = r + c * rows;
            let dst = c + r * cols;
            if src < ct.data.len() && dst < out.len() {
                out[dst] = ct.data[src];
            }
        }
    }
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::array::shape::permute::permute_tensor;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    fn unwrap_error(err: crate::RuntimeError) -> crate::RuntimeError {
        err
    }
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider as wgpu_backend;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, LogicalArray, StringArray, StructValue, Tensor};

    fn call_ctranspose(value: Value) -> BuiltinResult<Value> {
        block_on(super::ctranspose_builtin(vec![value]))
    }

    fn tensor(data: &[f64], shape: &[usize]) -> Tensor {
        Tensor::new(data.to_vec(), shape.to_vec()).unwrap()
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ctranspose_real_matrix_matches_transpose() {
        let input = tensor(&[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[2, 3]);
        let value = call_ctranspose(Value::Tensor(input)).expect("ctranspose");
        match value {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![3, 2]);
                assert_eq!(out.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ctranspose_complex_matrix_conjugates() {
        let data = vec![(1.0, 2.0), (3.0, -4.0), (5.0, 0.0), (6.0, -7.0)];
        let ct = ComplexTensor::new(data, vec![2, 2]).unwrap();
        let value = call_ctranspose(Value::ComplexTensor(ct)).expect("ctranspose");
        match value {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(
                    out.data,
                    vec![(1.0, -2.0), (5.0, -0.0), (3.0, 4.0), (6.0, 7.0)]
                );
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ctranspose_complex_tensor_realises_real_when_imag_zero() {
        let data = vec![(1.0, 0.0), (2.0, -0.0)];
        let ct = ComplexTensor::new(data, vec![1, 2]).unwrap();
        let value = call_ctranspose(Value::ComplexTensor(ct)).expect("ctranspose");
        match value {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 1]);
                assert_eq!(out.data, vec![1.0, 2.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ctranspose_complex_scalar() {
        let result = call_ctranspose(Value::Complex(2.0, 3.0)).expect("ctranspose");
        match result {
            Value::Complex(re, im) => {
                assert!((re - 2.0).abs() < 1e-12);
                assert!((im + 3.0).abs() < 1e-12);
            }
            other => panic!("expected complex scalar, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ctranspose_logical_mask() {
        let la = LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap();
        let value = call_ctranspose(Value::LogicalArray(la)).expect("ctranspose");
        match value {
            Value::LogicalArray(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(out.data, vec![1, 0, 0, 1]);
            }
            other => panic!("expected logical array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ctranspose_char_matrix() {
        let ca = CharArray::new("runmat".chars().collect(), 2, 3).unwrap();
        let value = call_ctranspose(Value::CharArray(ca)).expect("ctranspose");
        match value {
            Value::CharArray(out) => {
                assert_eq!(out.rows, 3);
                assert_eq!(out.cols, 2);
                assert_eq!(out.data, vec!['r', 'm', 'u', 'a', 'n', 't']);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ctranspose_string_array_transposes() {
        let data = vec![
            "r0c0".to_string(),
            "r1c0".to_string(),
            "r0c1".to_string(),
            "r1c1".to_string(),
        ];
        let sa = StringArray::new(data, vec![2, 2]).unwrap();
        let value = call_ctranspose(Value::StringArray(sa)).expect("ctranspose");
        match value {
            Value::StringArray(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert_eq!(
                    out.data,
                    vec![
                        "r0c0".to_string(),
                        "r0c1".to_string(),
                        "r1c0".to_string(),
                        "r1c1".to_string()
                    ]
                );
            }
            other => panic!("expected string array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ctranspose_cell_array() {
        let cells = vec![
            Value::from(1),
            Value::from(2),
            Value::from(3),
            Value::from(4),
        ];
        let cell_array = CellArray::new(cells, 2, 2).unwrap();
        let value = call_ctranspose(Value::Cell(cell_array)).expect("ctranspose");
        match value {
            Value::Cell(out) => {
                assert_eq!(out.rows, 2);
                assert_eq!(out.cols, 2);
                let v00 = out.get(0, 0).unwrap();
                let v01 = out.get(0, 1).unwrap();
                let v10 = out.get(1, 0).unwrap();
                let v11 = out.get(1, 1).unwrap();
                assert_eq!(v00, Value::from(1));
                assert_eq!(v01, Value::from(3));
                assert_eq!(v10, Value::from(2));
                assert_eq!(v11, Value::from(4));
            }
            other => panic!("expected cell array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ctranspose_scalar_types_identity() {
        assert_eq!(
            call_ctranspose(Value::Num(std::f64::consts::PI)).unwrap(),
            Value::Num(std::f64::consts::PI)
        );
        assert_eq!(
            call_ctranspose(Value::Int(IntValue::I32(5))).unwrap(),
            Value::Int(IntValue::I32(5))
        );
        assert_eq!(
            call_ctranspose(Value::Bool(true)).unwrap(),
            Value::Bool(true)
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ctranspose_tensor_swaps_first_two_dims_for_nd() {
        let data: Vec<f64> = (1..=12).map(|n| n as f64).collect();
        let input = tensor(&data, &[2, 3, 2]);
        let expected = permute_tensor(NAME, input.clone(), &[2, 1, 3]).unwrap();
        let value = call_ctranspose(Value::Tensor(input)).expect("ctranspose");
        match value {
            Value::Tensor(out) => {
                assert_eq!(out.shape, expected.shape);
                assert_eq!(out.data, expected.data);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ctranspose_struct_unsupported() {
        let mut st = StructValue::new();
        st.fields.insert("field".to_string(), Value::Num(1.0));
        let err = unwrap_error(call_ctranspose(Value::Struct(st)).unwrap_err());
        assert!(err.message().contains("unsupported input type"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ctranspose_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let t = tensor(&[1.0, 4.0, 2.0, 5.0], &[2, 2]);
            let view = HostTensorView {
                data: &t.data,
                shape: &t.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = call_ctranspose(Value::GpuTensor(handle)).expect("ctranspose");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![2, 2]);
            assert_eq!(gathered.data, vec![1.0, 2.0, 4.0, 5.0]);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn ctranspose_wgpu_matches_cpu() {
        let _ = wgpu_backend::register_wgpu_provider(wgpu_backend::WgpuProviderOptions::default());
        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let data: Vec<f64> = (1..=12).map(|n| n as f64).collect();
        let tensor = Tensor::new(data, vec![3, 4]).expect("tensor");
        let cpu_value = call_ctranspose(Value::Tensor(tensor.clone())).expect("cpu");
        let cpu_tensor = match cpu_value {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };

        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value = call_ctranspose(Value::GpuTensor(handle)).expect("gpu");
        let gathered = test_support::gather(gpu_value).expect("gather");
        assert_eq!(gathered.shape, cpu_tensor.shape);
        assert_eq!(gathered.data, cpu_tensor.data);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
