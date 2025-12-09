//! MATLAB-compatible `fft` builtin with GPU-aware semantics for RunMat.

use super::common::{
    default_dimension, host_to_complex_tensor, parse_length, tensor_to_complex_tensor,
    trim_trailing_ones, value_to_complex_tensor,
};
use num_complex::Complex;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle};
use runmat_builtins::{ComplexTensor, Value};
use runmat_macros::runtime_builtin;
use rustfft::FftPlanner;
use std::sync::Arc;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
#[cfg(feature = "doc_export")]
#[runmat_macros::register_doc_text(name = "fft")]
pub const DOC_MD: &str = r#"---
title: "fft"
category: "math/fft"
keywords: ["fft", "fourier transform", "complex", "zero padding", "gpu"]
summary: "Compute the discrete Fourier transform (DFT) of vectors, matrices, or N-D tensors."
references:
  - title: "MATLAB fft documentation"
    url: "https://www.mathworks.com/help/matlab/ref/fft.html"
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Falls back to host execution when the active acceleration provider does not expose an FFT hook."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::fft::fft::tests"
  integration: "builtins::math::fft::fft::tests::fft_gpu_roundtrip_matches_cpu"
---

# What does the `fft` function do in MATLAB / RunMat?
`fft(X)` computes the discrete Fourier transform (DFT) of the input data. When `X` is a vector,
`fft` returns the frequency-domain representation of the vector. When `X` is a matrix or an
N-D tensor, the transform is applied along the first non-singleton dimension unless another
dimension is specified.

## How does the `fft` function behave in MATLAB / RunMat?
- `fft(X)` transforms along the first dimension whose size is greater than 1.
- `fft(X, n)` zero-pads or truncates `X` to length `n` before transforming along the default dimension.
- `fft(X, n, dim)` applies the transform along dimension `dim`.
- Real inputs produce complex outputs; complex inputs are handled element-wise with no additional conversion.
- Empty inputs remain empty; zero-padding with `n` produces zero-valued spectra.
- GPU arrays are gathered to the host when the selected provider has no FFT implementation.

## Examples of using the `fft` function in MATLAB / RunMat

### Computing the FFT of a real time-domain vector
```matlab
x = [1 2 3 4];
Y = fft(x);
```
Expected output (RunMat prints complex numbers with `a + bi` formatting):
```matlab
Y =
  Columns 1 through 4
   10 + 0i  -2 + 2i  -2 + 0i  -2 - 2i
```

### Applying fft column-wise to a matrix
```matlab
A = [1 2 3; 4 5 6];
F = fft(A);
```
Expected output:
```matlab
F =
   5 + 0i   7 + 0i   9 + 0i
  -3 + 3i  -3 + 3i  -3 + 3i
```

### Zero-padding before the FFT
```matlab
x = [1 2 3];
Y = fft(x, 5);
```
The transform is computed on a length-5 sequence `[1 2 3 0 0]`, producing five complex frequency bins.

### Selecting the transform dimension for a row vector
```matlab
x = [1 2 3 4];
Y = fft(x, [], 2);
```
`Y` matches `fft(x)` because the transform is applied along dimension 2 (the row).

### FFT of a complex-valued signal
```matlab
t = 0:3;
x = exp(1i * pi/2 * t);
Y = fft(x);
```
The complex sinusoid is mapped to a single non-zero frequency bin at the expected location.

### FFT with gpuArray inputs
```matlab
g = gpuArray(rand(1, 1024));  % Residency is on the GPU
G = fft(g);                   % Falls back to host if provider FFT hooks are unavailable
result = gather(G);
```
RunMat gathers the data from the device and performs the transform on the host unless the active
provider advertises an FFT implementation. When the WGPU provider handles the FFT, the kernel executes
on the device but the result is downloaded immediately so the builtin can return a MATLAB-compatible
`ComplexTensor`.

## FAQ

### Does `fft` always return complex values?
Yes. Even when the imaginary part is zero, the result is stored as a complex array to match MATLAB semantics.

### What happens if I pass `[]` as the second argument?
Passing `[]` leaves the transform length unchanged. This is equivalent to omitting the `n` parameter.

### Can I transform along a dimension larger than the current rank?
Yes. RunMat automatically treats trailing dimensions as length-1 and will create the requested dimension on output.

### How does zero-padding work?
When `n` is larger than the size of `X` along the transform dimension, RunMat pads with zeros before evaluating the FFT.

### What precision is used for the FFT?
RunMat computes FFTs in double precision on the host. Providers may use single or double precision depending on device capabilities.

### Will RunMat run the FFT on my GPU automatically?
When a provider installs an FFT hook, RunMat executes on the GPU. Otherwise, the runtime gathers the data and performs the transform on the CPU.

### Is inverse FFT (`ifft`) available?
`ifft` will be provided in a companion builtin. Until then, you can recover a time-domain signal by dividing by the length and taking the complex conjugate manually.

### How do I compute multi-dimensional FFTs?
Call `fft` repeatedly along each dimension (`fft(fft(X, [], 1), [], 2)` for a 2-D FFT). Future releases will add dedicated helpers.

### Does `fft` support complex strides or non-unit sampling intervals?
`fft` assumes unit spacing. You can multiply the result by appropriate phase factors to account for custom sampling intervals.

## See Also
[ifft](./ifft), [fftshift](./fftshift), [abs](../elementwise/abs), [angle](../elementwise/angle), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- Full source: `crates/runmat-runtime/src/builtins/math/fft/fft.rs`
- Found an issue? [Open a ticket](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fft",
    op_kind: GpuOpKind::Custom("fft"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("fft_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers should implement `fft_dim` to transform along an arbitrary dimension; the runtime gathers to host when unavailable.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fft",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "FFT participates in fusion plans only as a boundary; no fused kernels are generated today.",
};

#[runtime_builtin(
    name = "fft",
    category = "math/fft",
    summary = "Compute the discrete Fourier transform (DFT) of numeric or complex data.",
    keywords = "fft,fourier transform,complex,gpu"
)]
fn fft_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let (length, dimension) = parse_arguments(&rest)?;
    match value {
        Value::GpuTensor(handle) => fft_gpu(handle, length, dimension),
        other => fft_host(other, length, dimension),
    }
}

fn fft_host(
    value: Value,
    length: Option<usize>,
    dimension: Option<usize>,
) -> Result<Value, String> {
    let tensor = value_to_complex_tensor(value, "fft")?;
    let transformed = fft_complex_tensor(tensor, length, dimension)?;
    Ok(complex_tensor_into_value(transformed))
}

fn fft_gpu(
    handle: GpuTensorHandle,
    length: Option<usize>,
    dimension: Option<usize>,
) -> Result<Value, String> {
    let mut shape = if handle.shape.is_empty() {
        vec![1]
    } else {
        handle.shape.clone()
    };

    let dim_one_based = match dimension {
        Some(0) => return Err("fft: dimension must be >= 1".to_string()),
        Some(dim) => dim,
        None => default_dimension(&shape),
    };

    let dim_index = dim_one_based - 1;
    while shape.len() <= dim_index {
        shape.push(1);
    }
    let current_len = shape[dim_index];
    let target_len = length.unwrap_or(current_len);

    if target_len == 0 {
        let tensor = gpu_helpers::gather_tensor(&handle)?;
        let complex = tensor_to_complex_tensor(tensor, "fft")?;
        let transformed = fft_complex_tensor(complex, length, dimension)?;
        return Ok(complex_tensor_into_value(transformed));
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(out) = provider.fft_dim(&handle, length, dim_index) {
            let complex = fft_download_gpu_result(provider, &out)?;
            return Ok(complex_tensor_into_value(complex));
        }
    }

    let tensor = gpu_helpers::gather_tensor(&handle)?;
    let complex = tensor_to_complex_tensor(tensor, "fft")?;
    let transformed = fft_complex_tensor(complex, length, dimension)?;
    Ok(complex_tensor_into_value(transformed))
}

pub(super) fn fft_download_gpu_result(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> Result<ComplexTensor, String> {
    let host = provider.download(handle).map_err(|e| format!("fft: {e}"))?;
    provider.free(handle).ok();
    runmat_accelerate_api::clear_residency(handle);
    host_to_complex_tensor(host, "fft")
}

fn parse_arguments(args: &[Value]) -> Result<(Option<usize>, Option<usize>), String> {
    match args.len() {
        0 => Ok((None, None)),
        1 => {
            let len = parse_length(&args[0], "fft")?;
            Ok((len, None))
        }
        2 => {
            let len = parse_length(&args[0], "fft")?;
            let dim = Some(tensor::parse_dimension(&args[1], "fft")?);
            Ok((len, dim))
        }
        _ => Err("fft: expected fft(X), fft(X, N), or fft(X, N, DIM)".to_string()),
    }
}

pub(super) fn fft_complex_tensor(
    mut tensor: ComplexTensor,
    length: Option<usize>,
    dimension: Option<usize>,
) -> Result<ComplexTensor, String> {
    if tensor.shape.is_empty() {
        tensor.shape = vec![tensor.data.len()];
        tensor.rows = tensor.shape.first().copied().unwrap_or(1);
        tensor.cols = tensor.shape.get(1).copied().unwrap_or(1);
    }

    let mut shape = tensor.shape.clone();
    let origin_rank = shape.len();
    let dim = match dimension {
        Some(0) => return Err("fft: dimension must be >= 1".to_string()),
        Some(dim) => dim - 1,
        None => default_dimension(&shape) - 1,
    };

    while shape.len() <= dim {
        shape.push(1);
    }

    let current_len = shape[dim];
    let target_len = length.unwrap_or(current_len);

    if target_len == 0 {
        let mut out_shape = shape;
        out_shape[dim] = 0;
        trim_trailing_ones(&mut out_shape, origin_rank);
        return ComplexTensor::new(Vec::<(f64, f64)>::new(), out_shape)
            .map_err(|e| format!("fft: {e}"));
    }

    let inner_stride = shape[..dim]
        .iter()
        .copied()
        .fold(1usize, |acc, dim| acc.saturating_mul(dim));
    let outer_stride = shape[dim + 1..]
        .iter()
        .copied()
        .fold(1usize, |acc, dim| acc.saturating_mul(dim));
    let num_slices = inner_stride.saturating_mul(outer_stride);

    let input = tensor
        .data
        .into_iter()
        .map(|(re, im)| Complex::new(re, im))
        .collect::<Vec<_>>();

    if num_slices == 0 {
        let mut out_shape = shape;
        out_shape[dim] = target_len;
        trim_trailing_ones(&mut out_shape, origin_rank);
        let data = vec![(0.0, 0.0); 0];
        return ComplexTensor::new(data, out_shape).map_err(|e| format!("fft: {e}"));
    }

    let output_len = target_len.saturating_mul(num_slices);
    let mut output = vec![Complex::new(0.0, 0.0); output_len];

    let mut planner = FftPlanner::<f64>::new();
    let fft_plan: Option<Arc<dyn rustfft::Fft<f64>>> = if target_len > 1 {
        Some(planner.plan_fft_forward(target_len))
    } else {
        None
    };

    let copy_len = current_len.min(target_len);
    let mut buffer = vec![Complex::new(0.0, 0.0); target_len];

    for outer in 0..outer_stride {
        let base_in = outer.saturating_mul(current_len.saturating_mul(inner_stride));
        let base_out = outer.saturating_mul(target_len.saturating_mul(inner_stride));
        for inner in 0..inner_stride {
            buffer.iter_mut().for_each(|c| *c = Complex::new(0.0, 0.0));
            for (k, slot) in buffer.iter_mut().enumerate().take(copy_len) {
                let src_idx = base_in + inner + k * inner_stride;
                if src_idx < input.len() {
                    *slot = input[src_idx];
                }
            }
            if target_len > 1 {
                if let Some(plan) = &fft_plan {
                    plan.process(&mut buffer);
                }
            }
            for (k, value) in buffer.iter().enumerate().take(target_len) {
                let dst_idx = base_out + inner + k * inner_stride;
                if dst_idx < output.len() {
                    output[dst_idx] = *value;
                }
            }
        }
    }

    let mut out_shape = shape;
    out_shape[dim] = target_len;
    trim_trailing_ones(&mut out_shape, origin_rank.max(dim + 1));

    let data = output.into_iter().map(|c| (c.re, c.im)).collect::<Vec<_>>();
    ComplexTensor::new(data, out_shape).map_err(|e| format!("fft: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use num_complex::Complex;
    use runmat_builtins::{ComplexTensor as HostComplexTensor, IntValue, Tensor};
    use rustfft::FftPlanner;

    fn approx_eq(a: (f64, f64), b: (f64, f64), tol: f64) -> bool {
        (a.0 - b.0).abs() <= tol && (a.1 - b.1).abs() <= tol
    }

    fn value_as_complex_tensor(value: Value) -> HostComplexTensor {
        match value {
            Value::ComplexTensor(tensor) => tensor,
            Value::Complex(re, im) => HostComplexTensor::new(vec![(re, im)], vec![1, 1]).unwrap(),
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn fft_real_vector() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let result = fft_host(Value::Tensor(tensor), None, None).expect("fft");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![4]);
                let expected = [(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)];
                for (idx, val) in ct.data.iter().enumerate() {
                    assert!(
                        approx_eq(*val, expected[idx], 1e-12),
                        "idx {idx} {:?} ~= {:?}",
                        val,
                        expected[idx]
                    );
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn fft_matrix_default_dimension() {
        let tensor = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let result = fft_host(Value::Tensor(tensor), None, None).expect("fft");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 3]);
                let expected = [
                    (5.0, 0.0),
                    (-3.0, 0.0),
                    (7.0, 0.0),
                    (-3.0, 0.0),
                    (9.0, 0.0),
                    (-3.0, 0.0),
                ];
                for (idx, val) in ct.data.iter().enumerate() {
                    assert!(approx_eq(*val, expected[idx], 1e-12));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn fft_zero_padding_with_length_argument() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let result =
            fft_host(Value::Tensor(tensor), Some(5), None).expect("fft with explicit length");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![5]);
                assert!(approx_eq(ct.data[0], (6.0, 0.0), 1e-12));
                assert_eq!(ct.data.len(), 5);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn fft_empty_length_argument_defaults_to_input_length() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let baseline =
            fft_builtin(Value::Tensor(tensor.clone()), Vec::new()).expect("baseline fft");
        let empty = Tensor::new(Vec::<f64>::new(), vec![0]).unwrap();
        let result = fft_builtin(
            Value::Tensor(tensor),
            vec![Value::Tensor(empty), Value::Int(IntValue::I32(1))],
        )
        .expect("fft with empty length");
        let base_ct = value_as_complex_tensor(baseline);
        let result_ct = value_as_complex_tensor(result);
        assert_eq!(base_ct.shape, result_ct.shape);
        assert_eq!(base_ct.data.len(), result_ct.data.len());
        for (idx, (a, b)) in base_ct.data.iter().zip(result_ct.data.iter()).enumerate() {
            assert!(
                approx_eq(*a, *b, 1e-12),
                "mismatch at index {idx}: {:?} vs {:?}",
                a,
                b
            );
        }
    }

    #[test]
    fn fft_truncates_when_length_smaller() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let result =
            fft_host(Value::Tensor(tensor), Some(2), None).expect("fft with truncation length");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2]);
                let expected = [(3.0, 0.0), (-1.0, 0.0)];
                for (idx, val) in ct.data.iter().enumerate() {
                    assert!(approx_eq(*val, expected[idx], 1e-12));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn fft_zero_length_returns_empty_tensor() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let result = fft_host(Value::Tensor(tensor), Some(0), None).expect("fft with zero length");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![0]);
                assert!(ct.data.is_empty());
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn fft_complex_input_preserves_imaginary_components() {
        let tensor =
            HostComplexTensor::new(vec![(1.0, 1.0), (0.0, -1.0), (2.0, 0.5)], vec![3]).unwrap();
        let result =
            fft_host(Value::ComplexTensor(tensor.clone()), None, None).expect("fft complex");
        let mut expected = tensor
            .data
            .iter()
            .map(|(re, im)| Complex::new(*re, *im))
            .collect::<Vec<_>>();
        FftPlanner::<f64>::new()
            .plan_fft_forward(expected.len())
            .process(&mut expected);
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![3]);
                assert_eq!(ct.data.len(), 3);
                for (idx, val) in ct.data.iter().enumerate() {
                    let exp = expected[idx];
                    assert!(approx_eq(*val, (exp.re, exp.im), 1e-12));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn fft_row_vector_dimension_two() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let result = fft_host(Value::Tensor(tensor), None, Some(2)).expect("fft along dimension 2");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 4]);
                let expected = [(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)];
                for (idx, val) in ct.data.iter().enumerate() {
                    assert!(approx_eq(*val, expected[idx], 1e-12));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn fft_dimension_extends_rank() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let original = tensor.clone();
        let result =
            fft_host(Value::Tensor(tensor), None, Some(3)).expect("fft with extra dimension");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 4, 1]);
                assert_eq!(ct.data.len(), original.data.len());
                for (idx, (re, im)) in ct.data.iter().enumerate() {
                    assert!(approx_eq((*re, *im), (original.data[idx], 0.0), 1e-12));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn fft_dimension_extends_rank_with_padding() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let original = tensor.clone();
        let result = fft_host(Value::Tensor(tensor), Some(4), Some(3))
            .expect("fft with padded third dimension");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![1, 4, 4]);
                let mut expected = Vec::with_capacity(16);
                for _depth in 0..4 {
                    for &value in &original.data {
                        expected.push((value, 0.0));
                    }
                }
                assert_eq!(ct.data.len(), expected.len());
                for (idx, (actual, expected)) in ct.data.iter().zip(expected.iter()).enumerate() {
                    assert!(
                        approx_eq(*actual, *expected, 1e-12),
                        "idx {idx}: {:?} != {:?}",
                        actual,
                        expected
                    );
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn fft_rejects_non_numeric_length() {
        assert!(parse_arguments(&[Value::Bool(true)]).is_err());
    }

    #[test]
    fn fft_rejects_negative_length() {
        let err = parse_arguments(&[Value::Num(-1.0)]).unwrap_err();
        assert!(err.contains("length must be non-negative"));
    }

    #[test]
    fn fft_rejects_fractional_length() {
        let err = parse_arguments(&[Value::Num(1.5)]).unwrap_err();
        assert!(err.contains("length must be an integer"));
    }

    #[test]
    fn fft_rejects_dimension_zero() {
        let err = parse_arguments(&[Value::Num(4.0), Value::Int(IntValue::I32(0))]).unwrap_err();
        assert!(err.contains("dimension must be >= 1"));
    }

    #[test]
    fn fft_gpu_roundtrip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu = fft_builtin(Value::GpuTensor(handle.clone()), Vec::new()).expect("fft");
            let cpu = fft_builtin(Value::Tensor(tensor), Vec::new()).expect("fft");
            let gpu_host = value_as_complex_tensor(gpu);
            let cpu_host = value_as_complex_tensor(cpu);
            assert_eq!(gpu_host.shape, cpu_host.shape);
            for (a, b) in gpu_host.data.iter().zip(cpu_host.data.iter()) {
                assert!(approx_eq(*a, *b, 1e-12));
            }
            provider.free(&handle).ok();
        });
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn fft_wgpu_matches_cpu() {
        if let Some(provider) = runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider()
            .expect("wgpu provider")
        {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
            let tensor_cpu = tensor.clone();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.data,
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu = fft_builtin(Value::GpuTensor(handle.clone()), Vec::new()).expect("gpu fft");
            let cpu = fft_builtin(Value::Tensor(tensor_cpu), Vec::new()).expect("cpu fft");
            let gpu_ct = value_as_complex_tensor(gpu);
            let cpu_ct = value_as_complex_tensor(cpu);
            assert_eq!(gpu_ct.shape, cpu_ct.shape);
            for (a, b) in gpu_ct.data.iter().zip(cpu_ct.data.iter()) {
                assert!(approx_eq(*a, *b, 1e-9));
            }
            provider.free(&handle).ok();
        }
    }

    #[test]
    fn doc_examples_present() {
        let blocks = test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }
}
