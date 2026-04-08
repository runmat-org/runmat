//! MATLAB-compatible `ifftn` builtin with GPU-aware semantics for RunMat.

use super::common::{
    complex_tensor_to_real_value, download_provider_complex_tensor, gather_gpu_complex_tensor,
    parse_nd_sizes_value, parse_symflag, transform_nd_complex_tensor, value_to_complex_tensor,
    TransformDirection,
};
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::math::fft::type_resolvers::ifftn_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::GpuTensorHandle;
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
    let (sizes, symmetric) = parse_ifftn_arguments(&rest)?;
    match value {
        Value::GpuTensor(handle) => ifftn_gpu(handle, sizes, symmetric).await,
        other => ifftn_host(other, sizes, symmetric),
    }
}

fn ifftn_host(value: Value, sizes: Option<Vec<usize>>, symmetric: bool) -> BuiltinResult<Value> {
    let tensor = value_to_complex_tensor(value, BUILTIN_NAME)?;
    let transformed = ifftn_complex_tensor(tensor, sizes)?;
    finalize_ifftn_output(transformed, symmetric)
}

async fn ifftn_gpu(
    handle: GpuTensorHandle,
    sizes: Option<Vec<usize>>,
    symmetric: bool,
) -> BuiltinResult<Value> {
    if let Some(ref spec) = sizes {
        if spec.contains(&0) {
            return ifftn_gpu_fallback(handle, sizes, symmetric).await;
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
            if !symmetric {
                return Ok(Value::GpuTensor(current));
            }
            if let Ok(real) = provider.fft_extract_real(&current).await {
                provider.free(&current).ok();
                runmat_accelerate_api::clear_residency(&current);
                return Ok(Value::GpuTensor(real));
            }
            let complex =
                download_provider_complex_tensor(provider, &current, BUILTIN_NAME, true).await?;
            return finalize_ifftn_output(complex, true);
        }
    }

    ifftn_gpu_fallback(handle, sizes, symmetric).await
}

async fn ifftn_gpu_fallback(
    handle: GpuTensorHandle,
    sizes: Option<Vec<usize>>,
    symmetric: bool,
) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let complex =
            download_provider_complex_tensor(provider, &handle, BUILTIN_NAME, false).await?;
        let transformed = ifftn_complex_tensor(complex, sizes)?;
        return finalize_ifftn_output(transformed, symmetric);
    }

    let complex = gather_gpu_complex_tensor(&handle, BUILTIN_NAME).await?;
    let transformed = ifftn_complex_tensor(complex, sizes)?;
    finalize_ifftn_output(transformed, symmetric)
}

fn ifftn_complex_tensor(
    tensor: ComplexTensor,
    sizes: Option<Vec<usize>>,
) -> BuiltinResult<ComplexTensor> {
    transform_nd_complex_tensor(
        tensor,
        sizes.as_deref(),
        TransformDirection::Inverse,
        BUILTIN_NAME,
    )
}

fn finalize_ifftn_output(tensor: ComplexTensor, symmetric: bool) -> BuiltinResult<Value> {
    if symmetric {
        complex_tensor_to_real_value(tensor, BUILTIN_NAME)
    } else {
        Ok(complex_tensor_into_value(tensor))
    }
}

fn parse_ifftn_arguments(args: &[Value]) -> BuiltinResult<(Option<Vec<usize>>, bool)> {
    if args.is_empty() {
        return Ok((None, false));
    }

    let (symflag, rem) = split_symflag(args)?;
    let symmetric = symflag.unwrap_or(false);

    let sizes = match rem.len() {
        0 => None,
        1 => Some(parse_sizes_value(&rem[0])?),
        _ => {
            return Err(ifftn_error(
                "ifftn: expected ifftn(X), ifftn(X, SIZE), or ifftn(X, SIZE, symflag)",
            ))
        }
    };
    Ok((sizes, symmetric))
}

fn split_symflag(args: &[Value]) -> BuiltinResult<(Option<bool>, &[Value])> {
    if let Some((last, rest)) = args.split_last() {
        if let Some(flag) = parse_symflag(last, BUILTIN_NAME)? {
            for value in rest {
                if parse_symflag(value, BUILTIN_NAME)?.is_some() {
                    return Err(ifftn_error(
                        "ifftn: symmetry flag must appear once at the end",
                    ));
                }
            }
            return Ok((Some(flag), rest));
        }
    }

    for value in args {
        if parse_symflag(value, BUILTIN_NAME)?.is_some() {
            return Err(ifftn_error(
                "ifftn: symmetry flag must appear as the final argument",
            ));
        }
    }

    Ok((None, args))
}

fn parse_sizes_value(value: &Value) -> BuiltinResult<Vec<usize>> {
    parse_nd_sizes_value(value, BUILTIN_NAME)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::math::fft::fft::fft_complex_tensor;
    use futures::executor::block_on;

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

    #[test]
    fn ifftn_accepts_symmetric_flag() {
        let input = Tensor::new((1..=8).map(|v| v as f64).collect(), vec![2, 2, 2]).unwrap();
        let complex = value_to_complex_tensor(Value::Tensor(input.clone()), BUILTIN_NAME).unwrap();
        let a = fft_complex_tensor(complex, None, Some(1)).unwrap();
        let b = fft_complex_tensor(a, None, Some(2)).unwrap();
        let freq = fft_complex_tensor(b, None, Some(3)).unwrap();

        let result = block_on(ifftn_builtin(
            Value::ComplexTensor(freq),
            vec![Value::from("symmetric")],
        ))
        .expect("ifftn symmetric");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![2, 2, 2]);
                for (got, expected) in t.data.iter().zip(input.data.iter()) {
                    assert!((*got - *expected).abs() < 1e-10);
                }
            }
            other => panic!("expected real tensor, got {other:?}"),
        }
    }

    #[test]
    fn ifftn_requires_symflag_final_position() {
        let input = Tensor::new((1..=8).map(|v| v as f64).collect(), vec![2, 2, 2]).unwrap();
        let size = Tensor::new(vec![2.0, 2.0, 2.0], vec![1, 3]).unwrap();
        let err = block_on(ifftn_builtin(
            Value::Tensor(input),
            vec![Value::from("symmetric"), Value::Tensor(size)],
        ))
        .unwrap_err();
        assert!(err
            .message()
            .contains("symmetry flag must appear as the final argument"));
    }
}
