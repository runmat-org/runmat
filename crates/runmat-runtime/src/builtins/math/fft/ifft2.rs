//! MATLAB-compatible `ifft2` builtin with GPU-aware semantics for RunMat.

use super::common::{
    host_to_complex_tensor, parse_length, tensor_to_complex_tensor, value_to_complex_tensor,
};
use super::ifft::ifft_complex_tensor;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, HostTensorOwned};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;

#[cfg_attr(
    feature = "doc_export",
    runmat_macros::register_doc_text(name = "ifft2")
)]
#[cfg_attr(not(feature = "doc_export"), allow(dead_code))]
pub const DOC_MD: &str = r#"---
title: "ifft2"
category: "math/fft"
keywords: ["ifft2", "inverse 2d fft", "image reconstruction", "symmetric", "gpu"]
summary: "Compute the two-dimensional inverse discrete Fourier transform (IDFT) of numeric or complex data."
references:
  - title: "MATLAB ifft2 documentation"
    url: "https://www.mathworks.com/help/matlab/ref/ifft2.html"
gpu_support:
  elementwise: false
  reduction: false
  precisions: ["f32", "f64"]
  broadcasting: "matlab"
  notes: "Runs two provider-backed ifft_dim passes when available; otherwise RunMat gathers to the host and performs the transform with rustfft."
fusion:
  elementwise: false
  reduction: false
  max_inputs: 1
  constants: "inline"
requires_feature: null
tested:
  unit: "builtins::math::fft::ifft2::tests"
  integration: "builtins::math::fft::ifft2::tests::ifft2_gpu_roundtrip_matches_cpu"
---

# What does the `ifft2` function do in MATLAB / RunMat?
`ifft2(X)` computes the two-dimensional inverse discrete Fourier transform (IDFT) of `X`. It
undoes the effect of `fft2` by applying `ifft` along the first dimension and then along the
second dimension, preserving MATLAB’s column-major semantics.

## How does the `ifft2` function behave in MATLAB / RunMat?
- `ifft2(X)` transforms along the first two dimensions whose sizes exceed one.
- `ifft2(X, M, N)` zero-pads or truncates the spectrum to `M` rows and `N` columns before the inverse.
- `ifft2(X, SIZE)` accepts a scalar or two-element vector describing the transform lengths.
- `ifft2(..., 'symmetric')` discards tiny imaginary parts and returns a real matrix when the
  spectrum is conjugate-symmetric. `'nonsymmetric'` keeps the complex result.
- Higher-dimensional inputs are processed slice-by-slice across trailing dimensions.
- Empty sizes and zero padding mirror MATLAB behaviour, producing empty outputs when any requested length is zero.

## Examples of using the `ifft2` function in MATLAB / RunMat

### Reconstructing an image patch from its 2-D spectrum
```matlab
F = [10  -2  0  -2;
     -4   0  0   0];
x = ifft2(F);
```
Expected output:
```matlab
x =
    1.0000    2.0000
    3.0000    4.0000
```

### Zero-padding before the inverse transform
```matlab
F = fft2([1 0; 0 1], 4, 4);
spatial = ifft2(F, 4, 4);
```
`spatial` is a `4×4` matrix representing the zero-padded impulse response.

### Supplying transform lengths with a size vector
```matlab
Y = fft2(rand(3,4));
X = ifft2(Y, [5 2]);   % pad rows to 5, truncate columns to 2
```

### Forcing a real-valued result with `'symmetric'`
```matlab
F = fft2([1 2; 3 4]);
realImage = ifft2(F, 'symmetric');
```
`realImage` equals the original matrix and has no residual imaginary values.

### Running `ifft2` on `gpuArray` inputs
```matlab
G = gpuArray(fft2(peaks(64)));
spatial = ifft2(G);
result = gather(spatial);
```
When the provider exposes `ifft_dim`, both passes execute on the GPU. Otherwise RunMat gathers `G`
and finishes on the host transparently.

### Recovering each slice of a volume
```matlab
spectra = fft2(rand(16, 16, 10));
volume = ifft2(spectra);
```
Every `16×16` slice along the third dimension is reconstructed independently.

## GPU residency in RunMat (Do I need `gpuArray`?)

You usually do **not** need to call `gpuArray` manually. RunMat’s native acceleration layer keeps
intermediate tensors on the GPU whenever the provider implements `ifft_dim`. If the provider lacks
that hook (or detects an unsupported length), RunMat gathers the input once, executes the inverse
transform on the CPU with `rustfft`, and returns a MATLAB-compatible result automatically.

## FAQ

1. **Is `ifft2(X)` equivalent to `ifft(ifft(X, [], 1), [], 2)`?**  
   Yes. RunMat literally performs two sequential 1-D inverse transforms so the behaviour matches MATLAB exactly.

2. **How do zero-length sizes behave?**  
   Passing `0` for either transform length produces an output with zero elements along that dimension, just like MATLAB.

3. **Can I mix `[]` with explicit sizes (e.g., `ifft2(X, [], 64)`)?**  
   Yes. `[]` leaves that dimension unchanged while the other argument controls padding or truncation.

4. **What does the `'symmetric'` flag do?**  
   It tells RunMat to coerce the result to real values, assuming the spectrum is conjugate-symmetric. Imaginary parts are dropped.

5. **What happens when the input is real-valued?**  
   RunMat promotes the data to complex with zero imaginary parts before applying the inverse. The output can still be coerced to real with `'symmetric'`.

6. **Will `ifft2` run on the GPU automatically?**  
   Yes when the active provider exposes `ifft_dim`. Otherwise the runtime gathers to the host and evaluates the inverse using `rustfft`.

7. **Does `ifft2` normalise the result?**  
   Yes. The builtin divides by the product of the transform lengths so that `ifft2(fft2(X))` reproduces `X`.

8. **Can I pass `gpuArray` size vectors or symmetry flags?**  
   No. Length arguments must be host scalars or vectors, and the symmetry flag must be a host string, mirroring MATLAB’s restrictions.

9. **How are higher-dimensional arrays handled?**  
   Transformations are applied to every 2-D slice defined by the first two dimensions; trailing dimensions are preserved unchanged.

10. **Does `'nonsymmetric'` change the result?**  
    It simply states the default behaviour (return complex outputs) but is accepted for MATLAB compatibility.

## See Also
[fft2](./fft2), [ifft](./ifft), [fft](./fft), [gpuArray](../../acceleration/gpu/gpuArray), [gather](../../acceleration/gpu/gather)

## Source & Feedback
- Full source: `crates/runmat-runtime/src/builtins/math/fft/ifft2.rs`
- Found an issue? [Open a ticket](https://github.com/runmat-org/runmat/issues/new/choose) with a minimal reproduction.
"#;

#[runmat_macros::register_gpu_spec]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ifft2",
    op_kind: GpuOpKind::Custom("ifft2"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("ifft_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes:
        "Performs two sequential `ifft_dim` passes (dimensions 0 and 1); falls back to host execution when the hook is missing.",
};

#[runmat_macros::register_fusion_spec]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ifft2",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes:
        "ifft2 terminates fusion plans; fused kernels are not generated for multi-dimensional inverse FFTs.",
};

#[runtime_builtin(
    name = "ifft2",
    category = "math/fft",
    summary = "Compute the two-dimensional inverse discrete Fourier transform (IDFT) of numeric or complex data.",
    keywords = "ifft2,inverse fft,image reconstruction,gpu"
)]
fn ifft2_builtin(value: Value, rest: Vec<Value>) -> Result<Value, String> {
    let ((len_rows, len_cols), symmetric) = parse_ifft2_arguments(&rest)?;
    match value {
        Value::GpuTensor(handle) => ifft2_gpu(handle, (len_rows, len_cols), symmetric),
        other => ifft2_host(other, (len_rows, len_cols), symmetric),
    }
}

fn ifft2_host(
    value: Value,
    lengths: (Option<usize>, Option<usize>),
    symmetric: bool,
) -> Result<Value, String> {
    let tensor = value_to_complex_tensor(value, "ifft2")?;
    let transformed = ifft2_complex_tensor(tensor, lengths)?;
    finalize_ifft2_output(transformed, symmetric)
}

fn ifft2_gpu(
    handle: GpuTensorHandle,
    lengths: (Option<usize>, Option<usize>),
    symmetric: bool,
) -> Result<Value, String> {
    if matches!(lengths.0, Some(0)) || matches!(lengths.1, Some(0)) {
        return ifft2_gpu_fallback(handle, lengths, symmetric);
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(first) = provider.ifft_dim(&handle, lengths.0, 0) {
            match provider.ifft_dim(&first, lengths.1, 1) {
                Ok(second) => {
                    if first.buffer_id != second.buffer_id {
                        provider.free(&first).ok();
                        runmat_accelerate_api::clear_residency(&first);
                    }
                    let complex = ifft2_download_gpu_result(provider, &second)?;
                    return finalize_ifft2_output(complex, symmetric);
                }
                Err(_) => {
                    let partial = ifft2_download_gpu_result(provider, &first)?;
                    let completed = ifft_complex_tensor(partial, lengths.1, Some(2))?;
                    return finalize_ifft2_output(completed, symmetric);
                }
            }
        }
    }

    ifft2_gpu_fallback(handle, lengths, symmetric)
}

fn ifft2_download_gpu_result(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> Result<ComplexTensor, String> {
    let host = provider
        .download(handle)
        .map_err(|e| format!("ifft2: {e}"))?;
    provider.free(handle).ok();
    runmat_accelerate_api::clear_residency(handle);
    host_to_complex_tensor(host, "ifft2")
}

fn ifft2_gpu_fallback(
    handle: GpuTensorHandle,
    lengths: (Option<usize>, Option<usize>),
    symmetric: bool,
) -> Result<Value, String> {
    let tensor = gpu_helpers::gather_tensor(&handle)?;
    let complex = if tensor.shape.last() == Some(&2) {
        let host = HostTensorOwned {
            data: tensor.data,
            shape: tensor.shape,
        };
        host_to_complex_tensor(host, "ifft2")?
    } else {
        tensor_to_complex_tensor(tensor, "ifft2")?
    };
    let transformed = ifft2_complex_tensor(complex, lengths)?;
    finalize_ifft2_output(transformed, symmetric)
}

fn ifft2_complex_tensor(
    tensor: ComplexTensor,
    lengths: (Option<usize>, Option<usize>),
) -> Result<ComplexTensor, String> {
    let (len_rows, len_cols) = lengths;
    let first = ifft_complex_tensor(tensor, len_rows, Some(1))?;
    ifft_complex_tensor(first, len_cols, Some(2))
}

fn finalize_ifft2_output(tensor: ComplexTensor, symmetric: bool) -> Result<Value, String> {
    if symmetric {
        complex_tensor_to_real_value(tensor, "ifft2")
    } else {
        Ok(complex_tensor_into_value(tensor))
    }
}

fn complex_tensor_to_real_value(tensor: ComplexTensor, builtin: &str) -> Result<Value, String> {
    let data = tensor.data.iter().map(|(re, _)| *re).collect::<Vec<_>>();
    let real = Tensor::new(data, tensor.shape.clone()).map_err(|e| format!("{builtin}: {e}"))?;
    Ok(Value::Tensor(real))
}

type LengthPair = (Option<usize>, Option<usize>);
type LengthsAndSymmetry = (LengthPair, bool);

fn parse_ifft2_arguments(args: &[Value]) -> Result<LengthsAndSymmetry, String> {
    if args.is_empty() {
        return Ok(((None, None), false));
    }

    let (maybe_flag, rem) = split_symflag(args)?;
    let mut symmetry = false;
    if let Some(flag) = maybe_flag {
        symmetry = flag;
    }

    let lengths = match rem.len() {
        0 => (None, None),
        1 => parse_ifft2_single(&rem[0])?,
        2 => {
            let rows = parse_length(&rem[0], "ifft2")?;
            let cols = parse_length(&rem[1], "ifft2")?;
            (rows, cols)
        }
        _ => {
            return Err(
                "ifft2: expected ifft2(X), ifft2(X, M, N), or ifft2(X, SIZE[, symflag])"
                    .to_string(),
            )
        }
    };

    Ok((lengths, symmetry))
}

fn split_symflag(args: &[Value]) -> Result<(Option<bool>, &[Value]), String> {
    if let Some((last, rest)) = args.split_last() {
        if let Some(flag) = parse_symflag(last)? {
            // Ensure no earlier argument is also a symmetry flag.
            for value in rest {
                if parse_symflag(value)?.is_some() {
                    return Err("ifft2: symmetry flag must appear once at the end".to_string());
                }
            }
            return Ok((Some(flag), rest));
        }
    }

    // Validate that no argument except the last is a symmetry flag.
    for value in args {
        if parse_symflag(value)?.is_some() {
            return Err("ifft2: symmetry flag must appear as the final argument".to_string());
        }
    }

    Ok((None, args))
}

fn parse_ifft2_single(value: &Value) -> Result<(Option<usize>, Option<usize>), String> {
    match value {
        Value::Tensor(tensor) => parse_length_pair(&tensor.data, "ifft2"),
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(logical)?;
            parse_length_pair(&tensor.data, "ifft2")
        }
        Value::Num(_) | Value::Int(_) => {
            let len = parse_length(value, "ifft2")?;
            Ok((len, len))
        }
        Value::Complex(re, im) => {
            if im.abs() > f64::EPSILON {
                return Err("ifft2: transform lengths must be real-valued".to_string());
            }
            let scalar = Value::Num(*re);
            let len = parse_length(&scalar, "ifft2")?;
            Ok((len, len))
        }
        Value::ComplexTensor(_) => Err("ifft2: size vector must contain real values".to_string()),
        Value::GpuTensor(_) => {
            Err("ifft2: size vector must be numeric and host-resident".to_string())
        }
        Value::Bool(_) => Err("ifft2: transform lengths must be numeric".to_string()),
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
        | Value::MException(_) => Err("ifft2: transform lengths must be numeric".to_string()),
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

fn parse_symflag(value: &Value) -> Result<Option<bool>, String> {
    use std::borrow::Cow;

    let text: Option<Cow<'_, str>> = match value {
        Value::String(s) => Some(Cow::Borrowed(s.as_str())),
        Value::CharArray(ca) if ca.rows == 1 => {
            let collected: String = ca.data.iter().collect();
            Some(Cow::Owned(collected))
        }
        Value::StringArray(sa) if sa.data.len() == 1 => Some(Cow::Borrowed(sa.data[0].as_str())),
        _ => None,
    };

    let Some(text) = text else {
        return Ok(None);
    };

    let trimmed = text.trim();
    if trimmed.eq_ignore_ascii_case("symmetric") {
        Ok(Some(true))
    } else if trimmed.eq_ignore_ascii_case("nonsymmetric") {
        Ok(Some(false))
    } else {
        Err(format!("ifft2: unrecognized option '{trimmed}'"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{IntValue, Tensor as HostTensor};

    fn approx_eq(a: (f64, f64), b: (f64, f64), tol: f64) -> bool {
        (a.0 - b.0).abs() <= tol && (a.1 - b.1).abs() <= tol
    }

    fn fft2_of_tensor(tensor: &HostTensor) -> ComplexTensor {
        let complex = value_to_complex_tensor(Value::Tensor(tensor.clone()), "fft2").unwrap();
        let first = super::super::fft::fft_complex_tensor(complex, None, Some(1)).unwrap();
        super::super::fft::fft_complex_tensor(first, None, Some(2)).unwrap()
    }

    #[test]
    fn ifft2_inverts_known_fft2() {
        let tensor = HostTensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let value =
            ifft2_builtin(Value::ComplexTensor(spectrum.clone()), Vec::new()).expect("ifft2");
        match value {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, tensor.shape);
                for (idx, (re, im)) in out.data.iter().enumerate() {
                    assert!(approx_eq((*re, *im), (tensor.data[idx], 0.0), 1e-12));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn ifft2_symmetric_returns_real() {
        let tensor = HostTensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let value = ifft2_builtin(
            Value::ComplexTensor(spectrum.clone()),
            vec![Value::from("symmetric")],
        )
        .expect("ifft2 symmetric");
        match value {
            Value::Tensor(out) => {
                assert_eq!(out.shape, tensor.shape);
                assert_eq!(out.data, tensor.data);
            }
            other => panic!("expected real tensor, got {other:?}"),
        }
    }

    #[test]
    fn ifft2_accepts_nonsymmetric_flag() {
        let tensor = HostTensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let value = ifft2_builtin(
            Value::ComplexTensor(spectrum.clone()),
            vec![Value::from("nonsymmetric")],
        )
        .expect("ifft2 nonsymmetric");
        let result = value_to_complex_tensor(value, "ifft2").expect("complex output");
        assert_eq!(result.shape, tensor.shape);
        for (idx, (re, im)) in result.data.iter().enumerate() {
            assert!(approx_eq((*re, *im), (tensor.data[idx], 0.0), 1e-12));
        }
    }

    #[test]
    fn ifft2_accepts_scalar_length() {
        let tensor = HostTensor::new((0..9).map(|v| v as f64).collect(), vec![3, 3]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let value = ifft2_builtin(
            Value::ComplexTensor(spectrum),
            vec![Value::Int(IntValue::I32(4))],
        )
        .expect("ifft2");
        match value {
            Value::ComplexTensor(out) => assert_eq!(out.shape, vec![4, 4]),
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn ifft2_accepts_size_vector() {
        let tensor = HostTensor::new((0..6).map(|v| v as f64).collect(), vec![2, 3]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let size = HostTensor::new(vec![4.0, 2.0], vec![1, 2]).unwrap();
        let value = ifft2_builtin(Value::ComplexTensor(spectrum), vec![Value::Tensor(size)])
            .expect("ifft2");
        match value {
            Value::ComplexTensor(out) => assert_eq!(out.shape, vec![4, 2]),
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn ifft2_treats_empty_lengths_as_defaults() {
        let tensor = HostTensor::new((0..6).map(|v| v as f64).collect(), vec![2, 3]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let empty_rows = HostTensor::new(vec![], vec![0]).unwrap();
        let empty_cols = HostTensor::new(vec![], vec![0]).unwrap();
        let value = ifft2_builtin(
            Value::ComplexTensor(spectrum.clone()),
            vec![Value::Tensor(empty_rows), Value::Tensor(empty_cols)],
        )
        .expect("ifft2");
        match value {
            Value::ComplexTensor(out) => {
                assert_eq!(out.shape, tensor.shape);
                for (idx, (re, im)) in out.data.iter().enumerate() {
                    assert!(approx_eq((*re, *im), (tensor.data[idx], 0.0), 1e-12));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn ifft2_rejects_boolean_length() {
        let tensor = HostTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let err =
            ifft2_builtin(Value::ComplexTensor(spectrum), vec![Value::Bool(true)]).unwrap_err();
        assert!(err.contains("ifft2"));
        assert!(err.contains("numeric"));
    }

    #[test]
    fn ifft2_rejects_excess_arguments() {
        let tensor = HostTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let err = ifft2_builtin(
            Value::ComplexTensor(spectrum),
            vec![
                Value::Int(IntValue::I32(2)),
                Value::Int(IntValue::I32(2)),
                Value::Int(IntValue::I32(2)),
            ],
        )
        .unwrap_err();
        assert!(err.contains("ifft2"));
    }

    #[test]
    fn ifft2_zero_lengths_return_empty_result() {
        let tensor = HostTensor::new((0..6).map(|v| v as f64).collect(), vec![2, 3]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let value = ifft2_builtin(
            Value::ComplexTensor(spectrum),
            vec![Value::Int(IntValue::I32(0)), Value::Int(IntValue::I32(0))],
        )
        .expect("ifft2");
        match value {
            Value::ComplexTensor(out) => {
                assert!(out.data.is_empty());
                assert_eq!(out.shape, vec![0, 0]);
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn ifft2_gpu_roundtrip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let tensor = HostTensor::new((0..8).map(|v| v as f64).collect(), vec![2, 4]).unwrap();
            let spectrum = fft2_of_tensor(&tensor);

            let host_real_imag = spectrum
                .data
                .iter()
                .flat_map(|(re, im)| [*re, *im])
                .collect::<Vec<_>>();
            let mut shape = spectrum.shape.clone();
            shape.push(2);
            let view = HostTensorView {
                data: &host_real_imag,
                shape: &shape,
            };
            let handle = provider.upload(&view).expect("upload spectrum");

            let gpu = ifft2_builtin(Value::GpuTensor(handle), Vec::new()).expect("ifft2 gpu");
            let cpu = ifft2_builtin(Value::ComplexTensor(spectrum.clone()), Vec::new())
                .expect("ifft2 cpu");

            match (gpu, cpu) {
                (Value::ComplexTensor(g), Value::ComplexTensor(c)) => {
                    assert_eq!(g.shape, c.shape);
                    for (lhs, rhs) in g.data.iter().zip(c.data.iter()) {
                        assert!(approx_eq(*lhs, *rhs, 1e-10), "{lhs:?} vs {rhs:?}");
                    }
                }
                other => panic!("unexpected results {other:?}"),
            }
        });
    }

    #[test]
    fn ifft2_handles_row_and_column_lengths() {
        let tensor = HostTensor::new((0..12).map(|v| v as f64).collect(), vec![3, 4]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let value = ifft2_builtin(
            Value::ComplexTensor(spectrum),
            vec![Value::Int(IntValue::I32(5)), Value::Int(IntValue::I32(2))],
        )
        .expect("ifft2");
        match value {
            Value::ComplexTensor(out) => assert_eq!(out.shape, vec![5, 2]),
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn ifft2_rejects_unknown_symmetry_flag() {
        let err = parse_ifft2_arguments(&[Value::from("invalid")]).unwrap_err();
        assert!(err.contains("unrecognized option"));
    }

    #[test]
    fn ifft2_requires_symflag_last() {
        let tensor = HostTensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let err = ifft2_builtin(
            Value::ComplexTensor(spectrum),
            vec![Value::from("symmetric"), Value::Int(IntValue::I32(2))],
        )
        .unwrap_err();
        assert!(err.contains("symmetry flag must appear as the final argument"));
    }

    #[test]
    fn doc_examples_present() {
        let blocks = crate::builtins::common::test_support::doc_examples(DOC_MD);
        assert!(!blocks.is_empty());
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn ifft2_wgpu_matches_cpu() {
        let provider = match std::panic::catch_unwind(|| {
            runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider()
        }) {
            Ok(Ok(Some(provider))) => provider,
            _ => return,
        };

        let tensor = HostTensor::new((0..16).map(|v| v as f64).collect(), vec![4, 4]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let host_real_imag = spectrum
            .data
            .iter()
            .flat_map(|(re, im)| [*re, *im])
            .collect::<Vec<_>>();
        let mut shape = spectrum.shape.clone();
        shape.push(2);
        let view = HostTensorView {
            data: &host_real_imag,
            shape: &shape,
        };
        let handle = provider.upload(&view).expect("upload spectrum");

        let gpu_val =
            ifft2_builtin(Value::GpuTensor(handle.clone()), Vec::new()).expect("ifft2 gpu");
        let cpu_val = ifft2_builtin(Value::ComplexTensor(spectrum), Vec::new()).expect("ifft2 cpu");

        let gpu_ct = value_to_complex_tensor(gpu_val, "ifft2").expect("gpu complex tensor");
        let cpu_ct = value_to_complex_tensor(cpu_val, "ifft2").expect("cpu complex tensor");
        assert_eq!(gpu_ct.shape, cpu_ct.shape);

        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-10,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (lhs, rhs) in gpu_ct.data.iter().zip(cpu_ct.data.iter()) {
            assert!(approx_eq(*lhs, *rhs, tol), "{lhs:?} vs {rhs:?}");
        }
    }
}
