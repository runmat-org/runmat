//! MATLAB-compatible `ifft` builtin with GPU-aware semantics for RunMat.

use super::common::{
    default_dimension, host_to_complex_tensor, parse_length, trim_trailing_ones,
    value_to_complex_tensor,
};
use num_complex::Complex;
use runmat_accelerate_api::{AccelProvider, GpuTensorHandle, HostTensorOwned};
use runmat_builtins::{ComplexTensor, Tensor, Value};
use runmat_macros::runtime_builtin;
use rustfft::FftPlanner;

use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{
    gpu_helpers,
    shape::{is_scalar_shape, normalize_scalar_shape},
    tensor,
};
use crate::dispatcher::download_handle_async;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::fft::ifft")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ifft",
    op_kind: GpuOpKind::Custom("ifft"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("ifft_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers should expose `ifft_dim` (or reuse `fft_dim` with inverse scaling); when absent, the runtime gathers to the host and evaluates the inverse FFT in software.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::fft::ifft")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ifft",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Inverse FFT boundaries are not currently fused; fusion plans terminate before invoking `ifft`.",
};

const BUILTIN_NAME: &str = "ifft";

fn ifft_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "ifft",
    category = "math/fft",
    summary = "Inverse discrete Fourier transform with optional length, dimension, and symmetric flag.",
    keywords = "ifft,inverse fft,inverse fourier transform,symmetric,gpu",
    builtin_path = "crate::builtins::math::fft::ifft"
)]
async fn ifft_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let (length, dimension, symmetric) = parse_arguments(&rest).await?;
    match value {
        Value::GpuTensor(handle) => ifft_gpu(handle, length, dimension, symmetric).await,
        other => ifft_host(other, length, dimension, symmetric),
    }
}

fn ifft_host(
    value: Value,
    length: Option<usize>,
    dimension: Option<usize>,
    symmetric: bool,
) -> BuiltinResult<Value> {
    let tensor = value_to_complex_tensor(value, BUILTIN_NAME)?;
    let transformed = ifft_complex_tensor(tensor, length, dimension)?;
    finalize_ifft_output(transformed, symmetric)
}

async fn ifft_gpu(
    handle: GpuTensorHandle,
    length: Option<usize>,
    dimension: Option<usize>,
    symmetric: bool,
) -> BuiltinResult<Value> {
    let mut logical_shape = normalize_scalar_shape(&handle.shape);
    if logical_shape.last() == Some(&2) {
        logical_shape.pop();
        logical_shape = normalize_scalar_shape(&logical_shape);
    }

    let dim_one_based = match dimension {
        Some(0) => return Err(ifft_error("ifft: dimension must be >= 1")),
        Some(dim) => dim,
        None => default_dimension(&logical_shape),
    };
    let dim_index = dim_one_based - 1;

    while logical_shape.len() <= dim_index {
        logical_shape.push(1);
    }

    let current_len = logical_shape.get(dim_index).copied().unwrap_or(0);
    let target_len = length.unwrap_or(current_len);

    if let Some(provider) = runmat_accelerate_api::provider() {
        if target_len != 0 {
            if let Ok(out) = provider.ifft_dim(&handle, length, dim_index).await {
                let complex = ifft_download_gpu_result(provider, &out).await?;
                return finalize_ifft_output(complex, symmetric);
            }
        }

        let host = download_handle_async(provider, &handle)
            .await
            .map_err(|e| ifft_error(format!("ifft: {e}")))?;
        runmat_accelerate_api::clear_residency(&handle);
        let complex = host_to_complex_tensor(host, BUILTIN_NAME)?;
        let transformed = ifft_complex_tensor(complex, length, dimension)?;
        return finalize_ifft_output(transformed, symmetric);
    }

    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    let Tensor { data, shape, .. } = tensor;
    let host = HostTensorOwned { data, shape };
    let complex = host_to_complex_tensor(host, BUILTIN_NAME)?;
    let transformed = ifft_complex_tensor(complex, length, dimension)?;
    finalize_ifft_output(transformed, symmetric)
}

pub(super) fn ifft_complex_tensor(
    mut tensor: ComplexTensor,
    length: Option<usize>,
    dimension: Option<usize>,
) -> BuiltinResult<ComplexTensor> {
    if is_scalar_shape(&tensor.shape) {
        tensor.shape = normalize_scalar_shape(&tensor.shape);
        tensor.rows = tensor.shape.first().copied().unwrap_or(1);
        tensor.cols = tensor.shape.get(1).copied().unwrap_or(1);
    }

    let mut shape = tensor.shape.clone();
    let origin_rank = shape.len();
    let dim_index = match dimension {
        Some(0) => return Err(ifft_error("ifft: dimension must be >= 1")),
        Some(dim) => dim - 1,
        None => default_dimension(&shape) - 1,
    };

    while shape.len() <= dim_index {
        shape.push(1);
    }

    let current_len = shape[dim_index];
    let target_len = length.unwrap_or(current_len);

    if target_len == 0 {
        let mut out_shape = shape;
        out_shape[dim_index] = 0;
        trim_trailing_ones(&mut out_shape, origin_rank);
        return ComplexTensor::new(Vec::<(f64, f64)>::new(), out_shape)
            .map_err(|e| ifft_error(format!("ifft: {e}")));
    }

    let inner_stride = shape[..dim_index]
        .iter()
        .copied()
        .fold(1usize, |acc, dim| acc.saturating_mul(dim));
    let outer_stride = shape[dim_index + 1..]
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
        out_shape[dim_index] = target_len;
        trim_trailing_ones(&mut out_shape, origin_rank.max(dim_index + 1));
        return ComplexTensor::new(Vec::<(f64, f64)>::new(), out_shape)
            .map_err(|e| ifft_error(format!("ifft: {e}")));
    }

    let output_len = target_len.saturating_mul(num_slices);
    let mut output = vec![Complex::new(0.0, 0.0); output_len];

    let mut planner = FftPlanner::<f64>::new();
    let ifft_plan = if target_len > 1 {
        Some(planner.plan_fft_inverse(target_len))
    } else {
        None
    };

    let copy_len = current_len.min(target_len);
    let mut buffer = vec![Complex::new(0.0, 0.0); target_len];
    let scale = 1.0 / (target_len as f64);

    for outer in 0..outer_stride {
        let base_in = outer.saturating_mul(current_len.saturating_mul(inner_stride));
        let base_out = outer.saturating_mul(target_len.saturating_mul(inner_stride));
        for inner in 0..inner_stride {
            buffer.fill(Complex::new(0.0, 0.0));
            for (k, slot) in buffer.iter_mut().enumerate().take(copy_len) {
                let src_idx = base_in + inner + k * inner_stride;
                if src_idx < input.len() {
                    *slot = input[src_idx];
                }
            }
            if let Some(plan) = &ifft_plan {
                plan.process(&mut buffer);
            }
            for (k, value) in buffer.iter().enumerate().take(target_len) {
                let dst_idx = base_out + inner + k * inner_stride;
                if dst_idx < output.len() {
                    output[dst_idx] = *value * scale;
                }
            }
        }
    }

    let mut out_shape = shape;
    out_shape[dim_index] = target_len;
    trim_trailing_ones(&mut out_shape, origin_rank.max(dim_index + 1));

    let data = output.into_iter().map(|c| (c.re, c.im)).collect::<Vec<_>>();
    ComplexTensor::new(data, out_shape).map_err(|e| ifft_error(format!("ifft: {e}")))
}

fn finalize_ifft_output(tensor: ComplexTensor, symmetric: bool) -> BuiltinResult<Value> {
    if symmetric {
        complex_tensor_to_real_value(tensor, BUILTIN_NAME)
    } else {
        Ok(complex_tensor_into_value(tensor))
    }
}

async fn ifft_download_gpu_result(
    provider: &dyn AccelProvider,
    handle: &GpuTensorHandle,
) -> BuiltinResult<ComplexTensor> {
    let host = download_handle_async(provider, handle)
        .await
        .map_err(|e| ifft_error(format!("ifft: {e}")))?;
    provider.free(handle).ok();
    runmat_accelerate_api::clear_residency(handle);
    host_to_complex_tensor(host, BUILTIN_NAME)
}

fn complex_tensor_to_real_value(tensor: ComplexTensor, builtin: &str) -> BuiltinResult<Value> {
    let data = tensor.data.iter().map(|(re, _)| *re).collect::<Vec<_>>();
    let real = Tensor::new(data, tensor.shape.clone())
        .map_err(|e| ifft_error(format!("{builtin}: {e}")))?;
    Ok(Value::Tensor(real))
}

async fn parse_dimension_arg(value: &Value) -> BuiltinResult<usize> {
    match value {
        Value::Int(_) | Value::Num(_) => {
            tensor::dimension_from_value_async(value, BUILTIN_NAME, false)
                .await
                .map_err(ifft_error)?
                .ok_or_else(|| {
                    ifft_error(format!(
                        "{BUILTIN_NAME}: dimension must be numeric, got {value:?}"
                    ))
                })
        }
        _ => Err(ifft_error(format!(
            "{BUILTIN_NAME}: dimension must be numeric, got {value:?}"
        ))),
    }
}

async fn parse_arguments(args: &[Value]) -> BuiltinResult<(Option<usize>, Option<usize>, bool)> {
    match args.len() {
        0 => Ok((None, None, false)),
        1 => match parse_symflag(&args[0])? {
            Some(flag) => Ok((None, None, flag)),
            None => {
                let len = parse_length(&args[0], BUILTIN_NAME)?;
                Ok((len, None, false))
            }
        },
        2 => {
            let first_flag = parse_symflag(&args[0])?;
            let second_flag = parse_symflag(&args[1])?;
            if let Some(flag) = second_flag {
                if first_flag.is_some() {
                    return Err(ifft_error(
                        "ifft: symmetry flag must appear as the final argument",
                    ));
                }
                let len = parse_length(&args[0], BUILTIN_NAME)?;
                Ok((len, None, flag))
            } else if first_flag.is_some() {
                Err(ifft_error(
                    "ifft: symmetry flag must appear as the final argument",
                ))
            } else {
                let len = parse_length(&args[0], BUILTIN_NAME)?;
                let dim = Some(parse_dimension_arg(&args[1]).await?);
                Ok((len, dim, false))
            }
        }
        3 => {
            let first_flag = parse_symflag(&args[0])?;
            let second_flag = parse_symflag(&args[1])?;
            let third_flag = parse_symflag(&args[2])?;
            let symmetry = third_flag.ok_or_else(|| {
                ifft_error("ifft: expected 'symmetric' or 'nonsymmetric' as the final argument")
            })?;
            if first_flag.is_some() || second_flag.is_some() {
                return Err(ifft_error(
                    "ifft: symmetry flag may only appear once at the end",
                ));
            }
            let len = parse_length(&args[0], BUILTIN_NAME)?;
            let dim = Some(parse_dimension_arg(&args[1]).await?);
            Ok((len, dim, symmetry))
        }
        _ => Err(ifft_error(
            "ifft: expected ifft(X), ifft(X, N), ifft(X, N, DIM), or ifft(X, N, DIM, symflag)",
        )),
    }
}

fn parse_symflag(value: &Value) -> BuiltinResult<Option<bool>> {
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
        Err(ifft_error(format!("ifft: unrecognized option '{trimmed}'")))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use num_complex::Complex;
    use runmat_builtins::{ComplexTensor as HostComplexTensor, IntValue};

    fn approx_eq((a_re, a_im): (f64, f64), (b_re, b_im): (f64, f64), tol: f64) -> bool {
        (a_re - b_re).abs() <= tol && (a_im - b_im).abs() <= tol
    }

    fn error_message(error: crate::RuntimeError) -> String {
        error.message().to_string()
    }

    fn value_as_complex_tensor(value: Value) -> HostComplexTensor {
        match value {
            Value::ComplexTensor(t) => t,
            Value::Tensor(t) => {
                HostComplexTensor::new(t.data.into_iter().map(|re| (re, 0.0)).collect(), t.shape)
                    .unwrap()
            }
            Value::Num(n) => HostComplexTensor::new(vec![(n, 0.0)], vec![1, 1]).unwrap(),
            Value::Int(i) => HostComplexTensor::new(vec![(i.to_f64(), 0.0)], vec![1, 1]).unwrap(),
            other => panic!("unexpected value kind {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_inverts_known_fft() {
        let spectrum = HostComplexTensor::new(
            vec![(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)],
            vec![4],
        )
        .unwrap();
        let result = ifft_host(Value::ComplexTensor(spectrum), None, None, false).expect("ifft");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![4]);
                let expected = [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)];
                for (idx, actual) in ct.data.iter().enumerate() {
                    assert!(approx_eq(*actual, expected[idx], 1e-12));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_symmetric_returns_real_tensor() {
        let spectrum = HostComplexTensor::new(
            vec![(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)],
            vec![4],
        )
        .unwrap();
        let result =
            ifft_host(Value::ComplexTensor(spectrum), None, None, true).expect("ifft symmetric");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![4]);
                assert_eq!(t.data, vec![1.0, 2.0, 3.0, 4.0]);
            }
            other => panic!("expected real tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_zero_length_returns_empty_tensor() {
        let spectrum = HostComplexTensor::new(Vec::new(), vec![0]).unwrap();
        let result = ifft_host(Value::ComplexTensor(spectrum), Some(0), None, false)
            .expect("ifft zero length");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![0]);
                assert!(ct.data.is_empty());
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_dimension_argument_recovers_matrix() {
        let original = Tensor::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], vec![2, 3]).unwrap();
        let mut spectrum = Vec::with_capacity(original.data.len());
        let rows = original.shape[0];
        let cols = original.shape[1];
        for c in 0..cols {
            let mut column = Vec::with_capacity(rows);
            for r in 0..rows {
                column.push(Complex::new(original.data[r + c * rows], 0.0));
            }
            let mut fft = column.clone();
            FftPlanner::<f64>::new()
                .plan_fft_forward(rows)
                .process(&mut fft);
            for value in fft {
                spectrum.push((value.re, value.im));
            }
        }
        let freq = HostComplexTensor::new(spectrum, vec![2, 3]).unwrap();
        let result = ifft_host(Value::ComplexTensor(freq), None, Some(1), false).expect("ifft dim");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2, 3]);
                for (idx, (re, im)) in ct.data.iter().enumerate() {
                    assert!(approx_eq((*re, *im), (original.data[idx], 0.0), 1e-12));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_rejects_dimension_zero() {
        let err = error_message(
            block_on(parse_arguments(&[
                Value::Num(4.0),
                Value::Int(IntValue::I32(0)),
            ]))
            .unwrap_err(),
        );
        assert!(err.contains("dimension must be >= 1"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_rejects_unknown_string_option() {
        let err =
            error_message(block_on(parse_arguments(&[Value::from("invalidflag")])).unwrap_err());
        assert!(err.contains("unrecognized option"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_accepts_nonsymmetric_flag() {
        let (len, dim, symmetric) =
            block_on(parse_arguments(&[Value::from("nonsymmetric")])).expect("parse");
        assert!(len.is_none());
        assert!(dim.is_none());
        assert!(!symmetric);

        let spectrum = HostComplexTensor::new(
            vec![(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)],
            vec![4],
        )
        .unwrap();
        let result =
            ifft_host(Value::ComplexTensor(spectrum), None, None, symmetric).expect("ifft");
        match result {
            Value::ComplexTensor(ct) => assert_eq!(ct.shape, vec![4]),
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_symflag_requires_final_position() {
        let err = error_message(
            block_on(parse_arguments(&[
                Value::from("nonsymmetric"),
                Value::Num(4.0),
            ]))
            .unwrap_err(),
        );
        assert!(err.contains("symmetry flag must appear as the final argument"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_symflag_accepts_whitespace() {
        let (len, dim, symmetric) =
            block_on(parse_arguments(&[Value::from(" symmetric ")])).expect("parse");
        assert!(len.is_none());
        assert!(dim.is_none());
        assert!(symmetric);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_zero_padding_length_argument() {
        let spectrum = HostComplexTensor::new(vec![(4.0, 0.0)], vec![1]).unwrap();
        let result = ifft_host(Value::ComplexTensor(spectrum), Some(4), None, false).expect("ifft");
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![4]);
                for &(re, im) in &ct.data {
                    assert!((re - 1.0).abs() < 1e-12);
                    assert!(im.abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_truncates_when_length_is_smaller() {
        let spectrum = HostComplexTensor::new(
            vec![(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)],
            vec![4],
        )
        .unwrap();
        let result = ifft_host(Value::ComplexTensor(spectrum), Some(2), None, false).expect("ifft");
        let mut expected = vec![Complex::new(10.0, 0.0), Complex::new(-2.0, 2.0)];
        FftPlanner::<f64>::new()
            .plan_fft_inverse(2)
            .process(&mut expected);
        for value in &mut expected {
            *value /= 2.0;
        }
        match result {
            Value::ComplexTensor(ct) => {
                assert_eq!(ct.shape, vec![2]);
                for ((re, im), expected) in ct.data.iter().zip(expected.iter()) {
                    assert!(approx_eq((*re, *im), (expected.re, expected.im), 1e-12));
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_empty_length_with_symmetric_flag() {
        let empty = Tensor::new(Vec::new(), vec![0]).unwrap();
        let (len, dim, symmetric) = block_on(parse_arguments(&[
            Value::Tensor(empty),
            Value::from("symmetric"),
        ]))
        .expect("parse");
        assert!(len.is_none());
        assert!(dim.is_none());
        assert!(symmetric);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft_gpu_roundtrip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let spectrum = vec![10.0, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0, -2.0];
            let shape = vec![4, 2];
            let view = runmat_accelerate_api::HostTensorView {
                data: &spectrum,
                shape: &shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu = ifft_builtin(Value::GpuTensor(handle.clone()), Vec::new()).expect("ifft");
            let cpu_spectrum = HostComplexTensor::new(
                vec![(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)],
                vec![4],
            )
            .unwrap();
            let cpu = ifft_builtin(Value::ComplexTensor(cpu_spectrum), Vec::new()).expect("ifft");
            let gpu_ct = value_as_complex_tensor(gpu);
            let cpu_ct = value_as_complex_tensor(cpu);
            assert_eq!(gpu_ct.shape, cpu_ct.shape);
            for (a, b) in gpu_ct.data.iter().zip(cpu_ct.data.iter()) {
                assert!(approx_eq(*a, *b, 1e-12));
            }
            provider.free(&handle).ok();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn ifft_wgpu_matches_cpu() {
        if let Some(provider) = runmat_accelerate::backend::wgpu::provider::ensure_wgpu_provider()
            .expect("wgpu provider")
        {
            let spectrum = vec![10.0, 0.0, -2.0, 2.0, -2.0, 0.0, -2.0, -2.0];
            let shape = vec![4, 2];
            let view = runmat_accelerate_api::HostTensorView {
                data: &spectrum,
                shape: &shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let gpu = ifft_builtin(Value::GpuTensor(handle.clone()), Vec::new()).expect("gpu ifft");
            let cpu_spectrum = HostComplexTensor::new(
                vec![(10.0, 0.0), (-2.0, 2.0), (-2.0, 0.0), (-2.0, -2.0)],
                vec![4],
            )
            .unwrap();
            let cpu =
                ifft_builtin(Value::ComplexTensor(cpu_spectrum), Vec::new()).expect("cpu ifft");
            let gpu_ct = value_as_complex_tensor(gpu);
            let cpu_ct = value_as_complex_tensor(cpu);
            assert_eq!(gpu_ct.shape, cpu_ct.shape);
            for (a, b) in gpu_ct.data.iter().zip(cpu_ct.data.iter()) {
                assert!(approx_eq(*a, *b, 1e-9));
            }
            provider.free(&handle).ok();
        }
    }

    fn ifft_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::ifft_builtin(value, rest))
    }
}
