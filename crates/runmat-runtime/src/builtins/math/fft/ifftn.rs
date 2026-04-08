//! MATLAB-compatible `ifftn` builtin with GPU-aware semantics for RunMat.

use super::common::{host_to_complex_tensor, tensor_to_complex_tensor, value_to_complex_tensor};
use super::ifft::ifft_complex_tensor;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::builtins::math::fft::type_resolvers::ifftn_type;
use crate::{build_runtime_error, dispatcher::download_handle_async, BuiltinResult, RuntimeError};
use runmat_accelerate_api::{GpuTensorHandle, HostTensorOwned};
use runmat_builtins::{ComplexTensor, Value};

#[cfg(test)]
use runmat_builtins::Tensor;
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::fft::ifftn")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "ifftn",
    op_kind: GpuOpKind::Custom("ifftn"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("ifft_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Performs sequential `ifft_dim` passes along each transformed axis; falls back to host execution when unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::fft::ifftn")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "ifftn",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "ifftn terminates fusion plans; fused kernels are not generated for N-D inverse FFTs.",
};

const BUILTIN_NAME: &str = "ifftn";

fn ifftn_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "ifftn",
    category = "math/fft",
    summary = "Compute the N-dimensional inverse discrete Fourier transform (IDFT) of numeric or complex data.",
    keywords = "ifftn,inverse nd fft,n-dimensional inverse fourier transform,gpu",
    type_resolver(ifftn_type),
    builtin_path = "crate::builtins::math::fft::ifftn"
)]
async fn ifftn_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let sizes = parse_ifftn_sizes(&rest)?;
    match value {
        Value::GpuTensor(handle) => ifftn_gpu(handle, sizes).await,
        other => ifftn_host(other, sizes),
    }
}

fn ifftn_host(value: Value, sizes: Option<Vec<usize>>) -> BuiltinResult<Value> {
    let tensor = value_to_complex_tensor(value, BUILTIN_NAME)?;
    let transformed = ifftn_complex_tensor(tensor, sizes)?;
    Ok(complex_tensor_into_value(transformed))
}

async fn ifftn_gpu(handle: GpuTensorHandle, sizes: Option<Vec<usize>>) -> BuiltinResult<Value> {
    if let Some(ref spec) = sizes {
        if spec.iter().any(|&n| n == 0) {
            return ifftn_gpu_fallback(handle, sizes).await;
        }
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        let mut current = handle.clone();
        let mut ok = true;
        let mut logical_shape = current.shape.clone();
        if logical_shape.last() == Some(&2) {
            logical_shape.pop();
        }
        if logical_shape.is_empty() {
            logical_shape.push(1);
        }
        let axis_count = sizes
            .as_ref()
            .map(|v| v.len())
            .unwrap_or_else(|| logical_shape.len());

        for axis in 0..axis_count {
            let len = sizes.as_ref().and_then(|v| v.get(axis).copied());
            match provider.ifft_dim(&current, len, axis).await {
                Ok(next) => {
                    if current.buffer_id != next.buffer_id {
                        provider.free(&current).ok();
                        runmat_accelerate_api::clear_residency(&current);
                    }
                    current = next;
                }
                Err(_) => {
                    ok = false;
                    break;
                }
            }
        }

        if ok {
            let host = download_handle_async(provider, &current)
                .await
                .map_err(|e| ifftn_error(format!("ifftn: {e}")))?;
            provider.free(&current).ok();
            runmat_accelerate_api::clear_residency(&current);
            let complex = host_to_complex_tensor(host, BUILTIN_NAME)?;
            return Ok(complex_tensor_into_value(complex));
        }
    }

    ifftn_gpu_fallback(handle, sizes).await
}

async fn ifftn_gpu_fallback(
    handle: GpuTensorHandle,
    sizes: Option<Vec<usize>>,
) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let host = download_handle_async(provider, &handle)
            .await
            .map_err(|e| ifftn_error(format!("ifftn: {e}")))?;
        runmat_accelerate_api::clear_residency(&handle);
        let complex = host_to_complex_tensor(host, BUILTIN_NAME)?;
        let transformed = ifftn_complex_tensor(complex, sizes)?;
        return Ok(complex_tensor_into_value(transformed));
    }

    let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
    let complex = if tensor.shape.last() == Some(&2) {
        let host = HostTensorOwned {
            data: tensor.data,
            shape: tensor.shape,
        };
        host_to_complex_tensor(host, BUILTIN_NAME)?
    } else {
        tensor_to_complex_tensor(tensor, BUILTIN_NAME)?
    };
    let transformed = ifftn_complex_tensor(complex, sizes)?;
    Ok(complex_tensor_into_value(transformed))
}

fn ifftn_complex_tensor(
    tensor: ComplexTensor,
    sizes: Option<Vec<usize>>,
) -> BuiltinResult<ComplexTensor> {
    let mut out = tensor;
    let axis_count = sizes
        .as_ref()
        .map(|v| v.len())
        .unwrap_or_else(|| out.shape.len().max(1));

    for axis in 0..axis_count {
        let len = sizes.as_ref().and_then(|v| v.get(axis).copied());
        out = ifft_complex_tensor(out, len, Some(axis + 1))?;
    }
    Ok(out)
}

fn parse_ifftn_sizes(args: &[Value]) -> BuiltinResult<Option<Vec<usize>>> {
    match args.len() {
        0 => Ok(None),
        1 => parse_sizes_value(&args[0]).map(Some),
        _ => Err(ifftn_error("ifftn: expected ifftn(X) or ifftn(X, SIZE)")),
    }
}

fn parse_sizes_value(value: &Value) -> BuiltinResult<Vec<usize>> {
    match value {
        Value::Tensor(t) => parse_sizes_data(&t.data),
        Value::LogicalArray(logical) => {
            let t = tensor::logical_to_tensor(logical)
                .map_err(|e| ifftn_error(format!("ifftn: {e}")))?;
            parse_sizes_data(&t.data)
        }
        Value::Num(n) => parse_sizes_data(&[*n]),
        Value::Int(i) => parse_sizes_data(&[i.to_f64()]),
        Value::Complex(re, im) => {
            if im.abs() > f64::EPSILON {
                return Err(ifftn_error("ifftn: SIZE must be real-valued"));
            }
            parse_sizes_data(&[*re])
        }
        Value::ComplexTensor(_) => Err(ifftn_error("ifftn: SIZE must be real-valued")),
        _ => Err(ifftn_error("ifftn: SIZE must be numeric")),
    }
}

fn parse_sizes_data(data: &[f64]) -> BuiltinResult<Vec<usize>> {
    let mut out = Vec::with_capacity(data.len());
    for &v in data {
        if !v.is_finite() {
            return Err(ifftn_error("ifftn: SIZE values must be finite"));
        }
        if v < 0.0 {
            return Err(ifftn_error("ifftn: SIZE values must be non-negative"));
        }
        let rounded = v.round();
        if (rounded - v).abs() > f64::EPSILON {
            return Err(ifftn_error("ifftn: SIZE values must be integers"));
        }
        out.push(rounded as usize);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::math::fft::fft::fft_complex_tensor;

    #[test]
    fn ifftn_roundtrip_matches_input_real_part() {
        let input = Tensor::new((1..=8).map(|v| v as f64).collect(), vec![2, 2, 2]).unwrap();
        let complex = value_to_complex_tensor(Value::Tensor(input.clone()), BUILTIN_NAME).unwrap();
        let a = fft_complex_tensor(complex, None, Some(1)).unwrap();
        let b = fft_complex_tensor(a, None, Some(2)).unwrap();
        let freq = fft_complex_tensor(b, None, Some(3)).unwrap();
        let back = ifftn_complex_tensor(freq, None).unwrap();
        assert_eq!(back.shape, vec![2, 2, 2]);
        for (idx, (re, im)) in back.data.iter().enumerate() {
            assert!((*re - input.data[idx]).abs() < 1e-10);
            assert!(im.abs() < 1e-10);
        }
    }
}
