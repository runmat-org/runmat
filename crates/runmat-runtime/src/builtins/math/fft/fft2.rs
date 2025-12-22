//! MATLAB-compatible `fft2` builtin with GPU-aware semantics for RunMat.

use super::common::{parse_length, tensor_to_complex_tensor, value_to_complex_tensor};
use super::fft::{fft_complex_tensor, fft_download_gpu_result};
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg(feature = "doc_export")]
use crate::register_builtin_doc_text;
use crate::{register_builtin_fusion_spec, register_builtin_gpu_spec};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_builtins::{ComplexTensor, Value};
use runmat_macros::runtime_builtin;

#[cfg(feature = "doc_export")]
pub const DOC_MD: &str = r#"---
title: "fft2"
category: "math/fft"
keywords: ["fft2", "2d fft", "two dimensional fourier transform", "image frequency analysis", "gpu"]
summary: "Compute the two-dimensional discrete Fourier transform (DFT) of numeric or complex data."
references:
  - title: "MATLAB fft2 documentation"
    url: "https://www.mathworks.com/help/matlab/ref/fft2.html"
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Runs two provider-backed fft_dim passes when available; otherwise the runtime gathers to the host."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::fft::fft2::tests"
  integration: "builtins::math::fft::fft2::tests::fft2_gpu_roundtrip_matches_cpu"
---

# What does the `fft2` function do in MATLAB / RunMat?
`fft2(X)` computes the two-dimensional discrete Fourier transform (DFT) of `X`. It is equivalent to
applying `fft` along the first dimension and then along the second dimension, preserving MATLAB’s
column-major ordering and output shape semantics.

## How does the `fft2` function behave in MATLAB / RunMat?
- `fft2(X)` transforms along the first and second dimensions that have size greater than one.
- `fft2(X, M, N)` zero-pads or truncates `X` to `M` rows and `N` columns before evaluating the 2-D transform.
- `fft2(X, SIZE)` accepts a two-element vector (or scalar) specifying the transform lengths.
- Real inputs produce complex outputs; complex inputs are transformed element-wise with no additional conversion.
- Higher-dimensional inputs are transformed slice-by-slice across trailing dimensions, matching MATLAB behaviour.
- Empty dimensions yield empty outputs; zero padding with `0` produces a zero-sized complex tensor.
- GPU arrays execute on-device when the provider advertises the `fft_dim` hook; otherwise RunMat gathers the data and
  performs the transform on the host using `rustfft`.

## Examples of using the `fft2` function in MATLAB / RunMat

### Computing the 2-D FFT of a small matrix
```matlab
X = [1 2; 3 4];
Y = fft2(X);
```
Expected output:
```matlab
Y =
    10 + 0i   -2 + 0i
    -4 + 0i    0 + 0i
```

### Zero-padding an image patch before `fft2`
```matlab
patch = [1 0 1; 0 1 0; 1 0 1];
F = fft2(patch, 8, 8);
```
`F` now contains an 8×8 spectrum suitable for convolution in the frequency domain.

### Specifying transform lengths with a size vector
```matlab
X = rand(4, 6);
F = fft2(X, [8 4]);   % pad rows to 8 and truncate columns to 4
```

### Using `fft2` on `gpuArray` data
```matlab
G = gpuArray(rand(256, 256));
F = fft2(G);
R = gather(F);
```
When the active provider implements `fft_dim`, both passes remain on the GPU. Otherwise RunMat gathers `G`,
executes the transform with `rustfft`, and returns a host-resident complex tensor.

### Applying `fft2` to each slice of a 3-D volume
```matlab
V = rand(64, 64, 10);
spectra = fft2(V);
```
Each `64×64` slice along the third dimension receives its own two-dimensional FFT.

### Verifying `fft2` against sequential `fft` calls
```matlab
X = rand(5, 7);
sequential = fft(fft(X, [], 1), [], 2);
direct = fft2(X);
```
`direct` and `sequential` produce the same complex data (up to round-off).

## GPU residency in RunMat (Do I need `gpuArray`?)

You usually do NOT need to call `gpuArray` manually. The fusion planner and native acceleration layer keep tensors on
the GPU when a provider offers FFT kernels. If the provider lacks `fft_dim`, RunMat gathers inputs, evaluates the FFT
pair on the host, and returns a MATLAB-compatible complex tensor. You can still use `gpuArray` for explicit residency
control, particularly when interoperating with MATLAB code that expects it.

## FAQ

1. **Is `fft2(X)` the same as `fft(fft(X, [], 1), [], 2)`?**  
   Yes. RunMat literally performs the two sequential transforms so the results match MATLAB exactly.

2. **How do zero-length transform sizes behave?**  
   Passing `0` for either `M` or `N` produces a complex tensor with zero elements along that dimension.

3. **Can I use a single scalar for the size argument?**  
   Yes. `fft2(X, K)` is shorthand for `fft2(X, [K K])`, padding or truncating both dimensions to `K`.

4. **What happens when `X` has more than two dimensions?**  
   RunMat applies `fft2` to every 2-D slice defined by the first two dimensions, leaving higher dimensions untouched.

5. **Do I get complex outputs for real inputs?**  
   Always. Even when the imaginary parts are zero, outputs are stored as complex tensors to mirror MATLAB semantics.

6. **Will `fft2` run on the GPU automatically?**  
   Yes if the active provider implements `fft_dim`. Otherwise RunMat gathers to the host and performs the transform with `rustfft`.

7. **Does `fft2` normalise the output?**  
   No. Like MATLAB, the forward FFT leaves scaling untouched; use `ifft2` for the inverse with `1/(M*N)` scaling.

8. **Can I mix `[]` with explicit sizes (e.g., `fft2(X, [], 128)`)?**  
   Yes. Passing `[]` leaves that dimension unchanged while applying the specified size to the other dimension.

## See Also
[fft](./fft), [ifft](./ifft), [fftshift](./fftshift), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- Full source: `crates/runmat-runtime/src/builtins/math/fft/fft2.rs`
- Found an issue? [Open a ticket](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fft2",
    op_kind: GpuOpKind::Custom("fft2"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("fft_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Performs two sequential `fft_dim` passes (dimensions 0 and 1); falls back to host execution when the hook is missing.",
};

register_builtin_gpu_spec!(GPU_SPEC);

pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fft2",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "fft2 terminates fusion plans; fused kernels are not generated for multi-dimensional FFTs.",
};

register_builtin_fusion_spec!(FUSION_SPEC);

#[cfg(feature = "doc_export")]
register_builtin_doc_text!("fft2", DOC_MD);

#[runtime_builtin(
    name = "fft2",
    category = "math/fft",
    summary = "Compute the two-dimensional discrete Fourier transform (DFT) of numeric or complex data.",
    keywords = "fft2,2d fft,two-dimensional fourier transform,gpu"
)]
fn fft2_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let lengths = parse_fft2_arguments(&rest)?;
    match value {
        Value::GpuTensor(handle) => fft2_gpu(handle, lengths),
        other => fft2_host(other, lengths),
    }
}

fn fft2_host(value: Value, lengths: (Option<usize>, Option<usize>)) -> Result<Value, String> {
    let tensor = value_to_complex_tensor(value, "fft2")?;
    let transformed = fft2_complex_tensor(tensor, lengths)?;
    Ok(complex_tensor_into_value(transformed))
}

fn fft2_gpu(
    handle: GpuTensorHandle,
    lengths: (Option<usize>, Option<usize>),
) -> Result<Value, String> {
    if matches!(lengths.0, Some(0)) || matches!(lengths.1, Some(0)) {
        return fft2_gpu_fallback(handle, lengths);
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(first) = provider.fft_dim(&handle, lengths.0, 0) {
            match provider.fft_dim(&first, lengths.1, 1) {
                Ok(second) => {
                    if first.buffer_id != second.buffer_id {
                        provider.free(&first).ok();
                        runmat_accelerate_api::clear_residency(&first);
                    }
                    let complex = fft2_download_gpu_result(provider, &second)?;
                    return Ok(complex_tensor_into_value(complex));
                }
                Err(_) => {
                    let partial = fft2_download_gpu_result(provider, &first)?;
                    let completed = fft_complex_tensor(partial, lengths.1, Some(2))?;
                    return Ok(complex_tensor_into_value(completed));
                }
            }
        }
    }

    fft2_gpu_fallback(handle, lengths)
}

fn fft2_download_gpu_result(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> Result<ComplexTensor, String> {
    fft_download_gpu_result(provider, handle).map_err(|e| e.replace("fft", "fft2"))
}

fn fft2_gpu_fallback(
    handle: GpuTensorHandle,
    lengths: (Option<usize>, Option<usize>),
) -> Result<Value, String> {
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    let complex = tensor_to_complex_tensor(tensor, "fft2")?;
    let transformed = fft2_complex_tensor(complex, lengths)?;
    Ok(complex_tensor_into_value(transformed))
}

fn fft2_complex_tensor(
    tensor: ComplexTensor,
    lengths: (Option<usize>, Option<usize>),
) -> Result<ComplexTensor, String> {
    let (len_rows, len_cols) = lengths;
    let first = fft_complex_tensor(tensor, len_rows, Some(1))?;
    fft_complex_tensor(first, len_cols, Some(2))
}

fn parse_fft2_arguments(args: &[Value]) -> Result<(Option<usize>, Option<usize>), String> {
    match args.len() {
        0 => Ok((None, None)),
        1 => parse_fft2_single(&args[0]),
        2 => {
            let rows = parse_length(&args[0], "fft2")?;
            let cols = parse_length(&args[1], "fft2")?;
            Ok((rows, cols))
        }
        _ => Err("fft2: expected fft2(X), fft2(X, M, N), or fft2(X, SIZE)".to_string()),
    }
}

fn parse_fft2_single(value: &Value) -> Result<(Option<usize>, Option<usize>), String> {
    match value {
        Value::Tensor(tensor) => parse_length_pair(&tensor.data, "fft2"),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(logical)?;
            parse_length_pair(&tensor.data, "fft2")
        }
        Value::Num(_) | Value::Int(_) => {
            let len = parse_length(value, "fft2")?;
            Ok((len, len))
        }
        Value::Complex(re, im) => {
            if im.abs() > f64::EPSILON {
                return Err("fft2: transform lengths must be real-valued".to_string());
            }
            let scalar = Value::Num(*re);
            let len = parse_length(&scalar, "fft2")?;
            Ok((len, len))
        }
        Value::ComplexTensor(_) => Err("fft2: size vector must contain real values".to_string()),
        Value::GpuTensor(_) => {
            Err("fft2: size vector must be numeric and host-resident".to_string())
        }
        Value::Bool(_) => Err("fft2: transform lengths must be numeric".to_string()),
        Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_)
        | Value::Cell(_)
        | Value::Struct(_)
        | Value::FunctionHandle(_)
        | Value::Closure(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::Object(_)
        | Value::ClassRef(_)
        | Value::MException(_) => Err("fft2: transform lengths must be numeric".to_string()),
        Value::Symbolic(_) => {
            Err("fft2: symbolic input not supported, use numeric values".to_string())
        }
    }
}

fn parse_length_pair(
    data: &[f64],
    builtin: &str,
) -> Result<(Option<usize>, Option<usize>), String> {
    match data.len() {
        0 => Ok((None, None)),
        1 => {
            let scalar = Value::Num(data[0]);
            let len = parse_length(&scalar, builtin)?;
            Ok((len, len))
        }
        2 => {
            let first = Value::Num(data[0]);
            let second = Value::Num(data[1]);
            let len_rows = parse_length(&first, builtin)?;
            let len_cols = parse_length(&second, builtin)?;
            Ok((len_rows, len_cols))
        }
        _ => Err(format!(
            "{builtin}: size vector must contain at most two elements"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, Tensor};

    fn approx_eq(a: (f64, f64), b: (f64, f64), tol: f64) -> bool {
        (a.0 - b.0).abs() <= tol && (a.1 - b.1).abs() <= tol
    }

    #[test]
    fn fft2_matches_sequential_fft() {
        let tensor = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let result = fft2_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("fft2");
        let sequential = {
            let complex = value_to_complex_tensor(Value::Tensor(tensor), "fft2").unwrap();
            let first = fft_complex_tensor(complex, None, Some(1)).unwrap();
            fft_complex_tensor(first, None, Some(2)).unwrap()
        };
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, sequential.shape);
                for (lhs, rhs) in out.data.iter().zip(sequential.data.iter()) {
                    assert!(approx_eq(*lhs, *rhs, 1e-12));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn fft2_accepts_scalar_length() {
        let tensor = Tensor::new((0..9).map(|v| v as f64).collect(), vec![3, 3]).unwrap();
        let result = fft2_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::Int(IntValue::I32(4))],
        )
        .expect("fft2");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![4, 4]);
                assert_eq!(out.data.len(), 16);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn fft2_accepts_size_vector() {
        let tensor = Tensor::new((0..6).map(|v| v as f64).collect(), vec![2, 3]).unwrap();
        let size = Tensor::new(vec![4.0, 2.0], vec![1, 2]).unwrap();
        let result =
            fft2_builtin(Value::Tensor(tensor.clone()), vec![Value::Tensor(size)]).expect("fft2");
        match result {
            Value::ComplexTensor(out) => assert_eq!(out.shape, vec![4, 2]),
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn fft2_accepts_empty_length_vector() {
        let tensor = Tensor::new((0..6).map(|v| v as f64).collect(), vec![2, 3]).unwrap();
        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let result =
            fft2_builtin(Value::Tensor(tensor.clone()), vec![Value::Tensor(empty)]).expect("fft2");
        match result {
            Value::ComplexTensor(out) => assert_eq!(out.shape, tensor.shape),
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn fft2_zero_length_returns_empty() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = fft2_builtin(
            Value::Tensor(tensor),
            vec![Value::Int(IntValue::I32(0)), Value::Int(IntValue::I32(3))],
        )
        .expect("fft2");
        match result {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, vec![0, 3]);
                assert!(out.data.is_empty());
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn fft2_gpu_roundtrip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new((0..8).map(|v| v as f64).collect(), vec![2, 4]).unwrap();
            let view = HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu = fft2_builtin(Value::GpuTensor(handle), Vec::new()).expect("fft2 gpu");
            let cpu = fft2_builtin(Value::Tensor(tensor), Vec::new()).expect("fft2 cpu");
            match (gpu, cpu) {
                (Value::ComplexTensor(g), Value::ComplexTensor(c)) => {
                    assert_eq!(g.shape, c.shape);
                    let tol = 1e-10;
                    for (lhs, rhs) in g.data.iter().zip(c.data.iter()) {
                        assert!(approx_eq(*lhs, *rhs, tol), "{lhs:?} vs {rhs:?}");
                    }
                }
                other => panic!("unexpected results {other:?}"),
            }
        });
    }

    #[test]
    fn fft2_rejects_size_vector_with_more_than_two_entries() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let size = Tensor::new(vec![4.0, 2.0, 1.0], vec![1, 3]).unwrap();
        let err = fft2_builtin(Value::Tensor(tensor), vec![Value::Tensor(size)]).unwrap_err();
        assert!(err.contains("fft2"));
        assert!(err.contains("two elements"));
    }

    #[test]
    fn fft2_rejects_boolean_length_argument() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = fft2_builtin(Value::Tensor(tensor), vec![Value::Bool(true)]).unwrap_err();
        assert!(err.contains("numeric"));
    }

    #[test]
    fn fft2_accepts_mixed_empty_and_length_arguments() {
        let tensor = Tensor::new((0..6).map(|v| v as f64).collect(), vec![2, 3]).unwrap();
        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let result = fft2_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::Tensor(empty), Value::Int(IntValue::I32(4))],
        )
        .expect("fft2");
        match result {
            Value::ComplexTensor(out) => assert_eq!(out.shape, vec![2, 4]),
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn fft2_rejects_excess_arguments() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let err = fft2_builtin(
            Value::Tensor(tensor),
            vec![
                Value::Int(IntValue::I32(2)),
                Value::Int(IntValue::I32(2)),
                Value::Int(IntValue::I32(2)),
            ],
        )
        .unwrap_err();
        assert!(err.contains("fft2"));
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn fft2_wgpu_matches_cpu() {
        let provider = match std::panic::catch_unwind(|| {
            runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider()
        }) {
            Ok(Ok(Some(provider))) => provider,
            _ => return,
        };

        let tensor = Tensor::new((0..16).map(|v| v as f64).collect(), vec![4, 4]).expect("tensor");
        let tensor_cpu = tensor.clone();
        let view = HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let gpu_value =
            fft2_builtin(Value::GpuTensor(handle.clone()), Vec::new()).expect("fft2 gpu");
        let cpu_value = fft2_builtin(Value::Tensor(tensor_cpu), Vec::new()).expect("fft2 cpu");
        let gpu_ct = value_to_complex_tensor(gpu_value, "fft2").expect("gpu complex tensor");
        let cpu_ct = value_to_complex_tensor(cpu_value, "fft2").expect("cpu complex tensor");
        assert_eq!(gpu_ct.shape, cpu_ct.shape);
        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-10,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (lhs, rhs) in gpu_ct.data.iter().zip(cpu_ct.data.iter()) {
            assert!(approx_eq(*lhs, *rhs, tol), "{lhs:?} vs {rhs:?}");
        }
        provider.free(&handle).ok();
        runmat_accelerate_api::clear_residency(&handle);
    }

    #[test]
    #[cfg(feature = "doc_export")]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
