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
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, Value,
};

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

const IFFTN_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "N-D inverse FFT output.",
}];

const IFFTN_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input spectrum or signal.",
}];

const IFFTN_INPUTS_SIZE: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input spectrum or signal.",
    },
    BuiltinParamDescriptor {
        name: "SIZE",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Transform sizes per dimension.",
    },
];

const IFFTN_INPUTS_SYMFLAG: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input spectrum or signal.",
    },
    BuiltinParamDescriptor {
        name: "symflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"nonsymmetric\""),
        description: "Symmetry flag: \"symmetric\" or \"nonsymmetric\".",
    },
];

const IFFTN_INPUTS_SIZE_SYMFLAG: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input spectrum or signal.",
    },
    BuiltinParamDescriptor {
        name: "SIZE",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Transform sizes per dimension.",
    },
    BuiltinParamDescriptor {
        name: "symflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"nonsymmetric\""),
        description: "Symmetry flag: \"symmetric\" or \"nonsymmetric\".",
    },
];

const IFFTN_SIGNATURES: [BuiltinSignatureDescriptor; 4] = [
    BuiltinSignatureDescriptor {
        label: "Y = ifftn(X)",
        inputs: &IFFTN_INPUTS_CORE,
        outputs: &IFFTN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifftn(X, SIZE)",
        inputs: &IFFTN_INPUTS_SIZE,
        outputs: &IFFTN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifftn(X, symflag)",
        inputs: &IFFTN_INPUTS_SYMFLAG,
        outputs: &IFFTN_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifftn(X, SIZE, symflag)",
        inputs: &IFFTN_INPUTS_SIZE_SYMFLAG,
        outputs: &IFFTN_OUTPUT,
    },
];

const IFFTN_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTN.ARG_COUNT",
    identifier: Some("RunMat:ifftn:ArgCount"),
    when: "More than three input arguments are supplied.",
    message: "ifftn: invalid argument count",
};

const IFFTN_ERROR_INVALID_SIZE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTN.INVALID_SIZE",
    identifier: Some("RunMat:ifftn:InvalidSize"),
    when: "SIZE argument is invalid.",
    message: "ifftn: invalid SIZE argument",
};

const IFFTN_ERROR_INVALID_SYMFLAG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTN.INVALID_SYMFLAG",
    identifier: Some("RunMat:ifftn:InvalidSymflag"),
    when: "Symmetry flag is invalid or appears in an invalid position.",
    message: "ifftn: invalid symmetry flag usage",
};

const IFFTN_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTN.INVALID_INPUT",
    identifier: Some("RunMat:ifftn:InvalidInput"),
    when: "Input cannot be converted to supported numeric/complex domain.",
    message: "ifftn: invalid input",
};

const IFFTN_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFTN.INTERNAL",
    identifier: Some("RunMat:ifftn:Internal"),
    when: "IFFTN execution or tensor shaping fails.",
    message: "ifftn: internal error",
};

const IFFTN_ERRORS: [BuiltinErrorDescriptor; 5] = [
    IFFTN_ERROR_ARG_COUNT,
    IFFTN_ERROR_INVALID_SIZE,
    IFFTN_ERROR_INVALID_SYMFLAG,
    IFFTN_ERROR_INVALID_INPUT,
    IFFTN_ERROR_INTERNAL,
];

pub const IFFTN_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IFFTN_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IFFTN_ERRORS,
};

fn ifftn_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    ifftn_error_with_message(error.message, error)
}

fn ifftn_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    ifftn_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn ifftn_error_with_source(
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

fn ifftn_error_with_message(
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
    name = "ifftn",
    category = "math/fft",
    summary = "Compute the N-dimensional inverse discrete Fourier transform (IDFT) of numeric or complex data.",
    keywords = "ifftn,inverse nd fft,n-dimensional inverse fourier transform,gpu",
    type_resolver(ifftn_type),
    descriptor(crate::builtins::math::fft::ifftn::IFFTN_DESCRIPTOR),
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
    let tensor = value_to_complex_tensor(value, BUILTIN_NAME).map_err(|source| {
        ifftn_error_with_source(
            &IFFTN_ERROR_INVALID_INPUT,
            "input conversion failed",
            source,
        )
    })?;
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
            let complex = download_provider_complex_tensor(provider, &current, BUILTIN_NAME, true)
                .await
                .map_err(|source| {
                    ifftn_error_with_source(
                        &IFFTN_ERROR_INVALID_INPUT,
                        "provider download failed",
                        source,
                    )
                })?;
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
        let complex = download_provider_complex_tensor(provider, &handle, BUILTIN_NAME, false)
            .await
            .map_err(|source| {
                ifftn_error_with_source(
                    &IFFTN_ERROR_INVALID_INPUT,
                    "provider download failed",
                    source,
                )
            })?;
        let transformed = ifftn_complex_tensor(complex, sizes)?;
        return finalize_ifftn_output(transformed, symmetric);
    }

    let complex = gather_gpu_complex_tensor(&handle, BUILTIN_NAME)
        .await
        .map_err(|source| {
            ifftn_error_with_source(&IFFTN_ERROR_INVALID_INPUT, "gpu gather failed", source)
        })?;
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
    .map_err(|source| ifftn_error_with_source(&IFFTN_ERROR_INTERNAL, "transform failed", source))
}

fn finalize_ifftn_output(tensor: ComplexTensor, symmetric: bool) -> BuiltinResult<Value> {
    if symmetric {
        complex_tensor_to_real_value(tensor, BUILTIN_NAME).map_err(|source| {
            ifftn_error_with_source(
                &IFFTN_ERROR_INTERNAL,
                "real-value extraction failed",
                source,
            )
        })
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
        _ => return Err(ifftn_error(&IFFTN_ERROR_ARG_COUNT)),
    };
    Ok((sizes, symmetric))
}

fn split_symflag(args: &[Value]) -> BuiltinResult<(Option<bool>, &[Value])> {
    if let Some((last, rest)) = args.split_last() {
        if let Some(flag) = parse_symflag(last, BUILTIN_NAME).map_err(|source| {
            ifftn_error_with_source(&IFFTN_ERROR_INVALID_SYMFLAG, "symflag parse failed", source)
        })? {
            for value in rest {
                if parse_symflag(value, BUILTIN_NAME)
                    .map_err(|source| {
                        ifftn_error_with_source(
                            &IFFTN_ERROR_INVALID_SYMFLAG,
                            "symflag parse failed",
                            source,
                        )
                    })?
                    .is_some()
                {
                    return Err(ifftn_error_with_detail(
                        &IFFTN_ERROR_INVALID_SYMFLAG,
                        "symmetry flag must appear once at the end",
                    ));
                }
            }
            return Ok((Some(flag), rest));
        }
    }

    for value in args {
        if parse_symflag(value, BUILTIN_NAME)
            .map_err(|source| {
                ifftn_error_with_source(
                    &IFFTN_ERROR_INVALID_SYMFLAG,
                    "symflag parse failed",
                    source,
                )
            })?
            .is_some()
        {
            return Err(ifftn_error_with_detail(
                &IFFTN_ERROR_INVALID_SYMFLAG,
                "symmetry flag must appear as the final argument",
            ));
        }
    }

    Ok((None, args))
}

fn parse_sizes_value(value: &Value) -> BuiltinResult<Vec<usize>> {
    parse_nd_sizes_value(value, BUILTIN_NAME).map_err(|source| {
        ifftn_error_with_detail(
            &IFFTN_ERROR_INVALID_SIZE,
            format!("SIZE parse failed: {source}"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::math::fft::fft::fft_complex_tensor;
    use futures::executor::block_on;
    use runmat_builtins::builtin_function_by_name;

    fn error_message(error: crate::RuntimeError) -> String {
        error.message().to_string()
    }

    fn error_identifier(error: &crate::RuntimeError) -> Option<&str> {
        error.identifier()
    }

    #[test]
    fn ifftn_descriptor_signatures_and_errors() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("ifftn builtin");
        let descriptor = builtin.descriptor.expect("ifftn descriptor");
        let labels: Vec<&str> = descriptor.signatures.iter().map(|sig| sig.label).collect();
        assert!(labels.contains(&"Y = ifftn(X)"));
        assert!(labels.contains(&"Y = ifftn(X, SIZE)"));
        assert!(labels.contains(&"Y = ifftn(X, symflag)"));
        assert!(labels.contains(&"Y = ifftn(X, SIZE, symflag)"));
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.IFFTN.INVALID_SYMFLAG"));
    }

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
        assert_eq!(
            error_identifier(&err),
            IFFTN_ERROR_INVALID_SYMFLAG.identifier
        );
        assert!(error_message(err).contains(IFFTN_ERROR_INVALID_SYMFLAG.message));
    }

    #[test]
    fn ifftn_rejects_invalid_argument_count() {
        let err = parse_ifftn_arguments(&[
            Value::Num(2.0),
            Value::Num(2.0),
            Value::Num(2.0),
            Value::Num(2.0),
        ])
        .unwrap_err();
        assert_eq!(error_identifier(&err), IFFTN_ERROR_ARG_COUNT.identifier);
        assert!(error_message(err).contains(IFFTN_ERROR_ARG_COUNT.message));
    }

    #[test]
    fn ifftn_rejects_invalid_size_argument() {
        let err = parse_ifftn_arguments(&[Value::Bool(true)]).unwrap_err();
        assert_eq!(error_identifier(&err), IFFTN_ERROR_INVALID_SIZE.identifier);
        assert!(error_message(err).contains(IFFTN_ERROR_INVALID_SIZE.message));
    }
}
