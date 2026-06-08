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
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, Value,
};

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

const FFTN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "N-D complex Fourier spectrum output.",
}];

const FFTN_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input array.",
}];

const FFTN_INPUTS_SIZE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input array.",
    },
    BuiltinParamDescriptor {
        name: "SIZE",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Transform sizes per dimension.",
    },
];

const FFTN_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Y = fftn(X)",
        inputs: &FFTN_INPUTS_CORE,
        outputs: &FFTN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = fftn(X, SIZE)",
        inputs: &FFTN_INPUTS_SIZE,
        outputs: &FFTN_OUTPUT,
    },
];

const FFTN_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTN.ARG_COUNT",
    identifier: Some("RunMat:fftn:ArgCount"),
    when: "More than two input arguments are supplied.",
    message: "fftn: invalid argument count",
};

const FFTN_ERROR_INVALID_SIZE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTN.INVALID_SIZE",
    identifier: Some("RunMat:fftn:InvalidSize"),
    when: "SIZE argument is invalid.",
    message: "fftn: invalid SIZE argument",
};

const FFTN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTN.INVALID_INPUT",
    identifier: Some("RunMat:fftn:InvalidInput"),
    when: "Input cannot be converted to supported numeric/complex domain.",
    message: "fftn: invalid input",
};

const FFTN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.FFTN.INTERNAL",
    identifier: Some("RunMat:fftn:Internal"),
    when: "FFTN execution or tensor shaping fails.",
    message: "fftn: internal error",
};

const FFTN_ERRORS: [BuiltinErrorDescriptor; 4] = [
    FFTN_ERROR_ARG_COUNT,
    FFTN_ERROR_INVALID_SIZE,
    FFTN_ERROR_INVALID_INPUT,
    FFTN_ERROR_INTERNAL,
];

pub const FFTN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &FFTN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &FFTN_ERRORS,
};

fn fftn_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    fftn_error_with_message(error.message, error)
}

fn fftn_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    fftn_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn fftn_error_with_source(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
    source: RuntimeError,
) -> RuntimeError {
    let mut builder = build_runtime_error(format!("{}: {}", error.message, detail.as_ref()))
        .with_builtin(BUILTIN_NAME)
        .with_source(source);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn fftn_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "fftn",
    category = "math/fft",
    summary = "Compute the N-dimensional discrete Fourier transform (DFT) of numeric or complex data.",
    keywords = "fftn,nd fft,n-dimensional fourier transform,gpu",
    type_resolver(fftn_type),
    descriptor(crate::builtins::math::fft::fftn::FFTN_DESCRIPTOR),
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
    let tensor = value_to_complex_tensor(value, BUILTIN_NAME).map_err(|source| {
        fftn_error_with_source(&FFTN_ERROR_INVALID_INPUT, "input conversion failed", source)
    })?;
    let transformed = fftn_complex_tensor(tensor, sizes)?;
    Ok(complex_tensor_into_value(transformed))
}

async fn fftn_gpu(handle: GpuTensorHandle, sizes: Option<Vec<usize>>) -> BuiltinResult<Value> {
    if let Some(ref spec) = sizes {
        if spec.contains(&0) {
            return fftn_gpu_fallback(handle, sizes).await;
        }
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        let mut current = handle.clone();
        let mut ok = true;
        let mut logical_shape = current.shape.clone();
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
            return Ok(Value::GpuTensor(current));
        }
    }

    fftn_gpu_fallback(handle, sizes).await
}

async fn fftn_gpu_fallback(
    handle: GpuTensorHandle,
    sizes: Option<Vec<usize>>,
) -> BuiltinResult<Value> {
    if let Some(provider) = runmat_accelerate_api::provider() {
        let complex = download_provider_complex_tensor(provider, &handle, BUILTIN_NAME, false)
            .await
            .map_err(|source| {
                fftn_error_with_source(
                    &FFTN_ERROR_INVALID_INPUT,
                    "provider download failed",
                    source,
                )
            })?;
        let transformed = fftn_complex_tensor(complex, sizes)?;
        return Ok(complex_tensor_into_value(transformed));
    }

    let complex = gather_gpu_complex_tensor(&handle, BUILTIN_NAME)
        .await
        .map_err(|source| {
            fftn_error_with_source(&FFTN_ERROR_INVALID_INPUT, "gpu gather failed", source)
        })?;
    let transformed = fftn_complex_tensor(complex, sizes)?;
    Ok(complex_tensor_into_value(transformed))
}

fn fftn_complex_tensor(
    tensor: ComplexTensor,
    sizes: Option<Vec<usize>>,
) -> BuiltinResult<ComplexTensor> {
    transform_nd_complex_tensor(
        tensor,
        sizes.as_deref(),
        TransformDirection::Forward,
        BUILTIN_NAME,
    )
    .map_err(|source| fftn_error_with_source(&FFTN_ERROR_INTERNAL, "transform failed", source))
}

fn parse_fftn_sizes(args: &[Value]) -> BuiltinResult<Option<Vec<usize>>> {
    match args.len() {
        0 => Ok(None),
        1 => parse_sizes_value(&args[0]).map(Some),
        _ => Err(fftn_error(&FFTN_ERROR_ARG_COUNT)),
    }
}

fn parse_sizes_value(value: &Value) -> BuiltinResult<Vec<usize>> {
    parse_nd_sizes_value(value, BUILTIN_NAME).map_err(|source| {
        fftn_error_with_detail(
            &FFTN_ERROR_INVALID_SIZE,
            format!("SIZE parse failed: {source}"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::math::fft::fft::fft_complex_tensor;
    use runmat_builtins::builtin_function_by_name;

    fn error_message(error: crate::RuntimeError) -> String {
        error.message().to_string()
    }

    fn error_identifier(error: &crate::RuntimeError) -> Option<&str> {
        error.identifier()
    }

    #[test]
    fn fftn_descriptor_signatures_and_errors() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("fftn builtin");
        let descriptor = builtin.descriptor.expect("fftn descriptor");
        let labels: Vec<&str> = descriptor.signatures.iter().map(|sig| sig.label).collect();
        assert!(labels.contains(&"Y = fftn(X)"));
        assert!(labels.contains(&"Y = fftn(X, SIZE)"));
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.FFTN.INVALID_SIZE"));
    }

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

    #[test]
    fn fftn_rejects_invalid_argument_count() {
        let err = parse_fftn_sizes(&[Value::Num(2.0), Value::Num(3.0)]).unwrap_err();
        assert_eq!(error_identifier(&err), FFTN_ERROR_ARG_COUNT.identifier);
        assert!(error_message(err).contains(FFTN_ERROR_ARG_COUNT.message));
    }

    #[test]
    fn fftn_rejects_invalid_size_argument() {
        let err = parse_fftn_sizes(&[Value::from("bad")]).unwrap_err();
        assert_eq!(error_identifier(&err), FFTN_ERROR_INVALID_SIZE.identifier);
        assert!(error_message(err).contains(FFTN_ERROR_INVALID_SIZE.message));
    }
}
