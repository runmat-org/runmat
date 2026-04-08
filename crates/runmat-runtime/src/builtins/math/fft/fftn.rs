//! MATLAB-compatible `fftn` builtin with GPU-aware semantics for RunMat.

use super::common::{
    download_provider_complex_tensor, gather_gpu_complex_tensor, parse_nd_sizes_value,
    transform_nd_complex_tensor, value_to_complex_tensor, TransformDirection,
};
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::math::fft::type_resolvers::fftn_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{ComplexTensor, Value};

#[cfg(test)]
use runmat_builtins::Tensor;
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::fft::fftn")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "fftn",
    op_kind: GpuOpKind::Custom("fftn"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("fft_dim")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Performs sequential `fft_dim` passes along each transformed axis; falls back to host execution when unavailable.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::fft::fftn")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "fftn",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "fftn terminates fusion plans; fused kernels are not generated for N-D FFTs.",
};

const BUILTIN_NAME: &str = "fftn";

fn fftn_error(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(BUILTIN_NAME)
        .build()
}

#[runtime_builtin(
    name = "fftn",
    category = "math/fft",
    summary = "Compute the N-dimensional discrete Fourier transform (DFT) of numeric or complex data.",
    keywords = "fftn,nd fft,n-dimensional fourier transform,gpu",
    type_resolver(fftn_type),
    builtin_path = "crate::builtins::math::fft::fftn"
)]
async fn fftn_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let sizes = parse_fftn_sizes(&rest)?;
    match value {
        Value::GpuTensor(handle) => fftn_gpu(handle, sizes).await,
        other => fftn_host(other, sizes),
    }
}

fn fftn_host(value: Value, sizes: Option<Vec<usize>>) -> BuiltinResult<Value> {
    let tensor = value_to_complex_tensor(value, BUILTIN_NAME)?;
    let transformed = fftn_complex_tensor(tensor, sizes)?;
    Ok(complex_tensor_into_value(transformed))
}

async fn fftn_gpu(handle: GpuTensorHandle, sizes: Option<Vec<usize>>) -> BuiltinResult<Value> {
    if let Some(ref spec) = sizes {
        if spec.iter().any(|&n| n == 0) {
            return fftn_gpu_fallback(handle, sizes).await;
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
            match provider.fft_dim(&current, len, axis).await {
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
            let complex = download_provider_complex_tensor(provider, &current, BUILTIN_NAME, true)
                .await?;
            return Ok(complex_tensor_into_value(complex));
        }
    }

    fftn_gpu_fallback(handle, sizes).await
}

async fn fftn_gpu_fallback(handle: GpuTensorHandle, sizes: Option<Vec<usize>>) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let complex = download_provider_complex_tensor(provider, &handle, BUILTIN_NAME, false).await?;
        let transformed = fftn_complex_tensor(complex, sizes)?;
        return Ok(complex_tensor_into_value(transformed));
    }

    let complex = gather_gpu_complex_tensor(&handle, BUILTIN_NAME).await?;
    let transformed = fftn_complex_tensor(complex, sizes)?;
    Ok(complex_tensor_into_value(transformed))
}

fn fftn_complex_tensor(tensor: ComplexTensor, sizes: Option<Vec<usize>>) -> BuiltinResult<ComplexTensor> {
    transform_nd_complex_tensor(tensor, sizes.as_deref(), TransformDirection::Forward, BUILTIN_NAME)
}

fn parse_fftn_sizes(args: &[Value]) -> BuiltinResult<Option<Vec<usize>>> {
    match args.len() {
        0 => Ok(None),
        1 => parse_sizes_value(&args[0]).map(Some),
        _ => Err(fftn_error("fftn: expected fftn(X) or fftn(X, SIZE)")),
    }
}

fn parse_sizes_value(value: &Value) -> BuiltinResult<Vec<usize>> {
    parse_nd_sizes_value(value, BUILTIN_NAME)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::math::fft::fft::fft_complex_tensor;

    #[test]
    fn fftn_matches_sequential_fft_on_3d() {
        let input = Tensor::new((1..=8).map(|v| v as f64).collect(), vec![2, 2, 2]).unwrap();
        let complex = value_to_complex_tensor(Value::Tensor(input), BUILTIN_NAME).unwrap();
        let got = fftn_complex_tensor(complex.clone(), None).unwrap();

        let a = fft_complex_tensor(complex, None, Some(1)).unwrap();
        let b = fft_complex_tensor(a, None, Some(2)).unwrap();
        let expect = fft_complex_tensor(b, None, Some(3)).unwrap();

        assert_eq!(got.shape, expect.shape);
        for (lhs, rhs) in got.data.iter().zip(expect.data.iter()) {
            assert!((lhs.0 - rhs.0).abs() < 1e-12);
            assert!((lhs.1 - rhs.1).abs() < 1e-12);
        }
    }
}
