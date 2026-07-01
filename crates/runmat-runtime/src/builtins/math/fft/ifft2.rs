//! MATLAB-compatible `ifft2` builtin with GPU-aware semantics for RunMat.

use super::common::{
    complex_tensor_to_real_value, download_provider_complex_tensor, gather_gpu_complex_tensor,
    parse_2d_lengths_from_data, parse_length, parse_symflag, transform_axes_complex_tensor,
    value_to_complex_tensor, TransformDirection,
};
use super::ifft::ifft_complex_tensor;
use crate::builtins::common::random_args::complex_tensor_into_value;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::math::fft::type_resolvers::ifft2_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use runmat_accelerate_api::GpuTensorHandle;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexTensor, Value,
};
use runmat_macros::runtime_builtin;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::fft::ifft2")]
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

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::fft::ifft2")]
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

const BUILTIN_NAME: &str = "ifft2";

const IFFT2_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Y",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "2-D inverse FFT output.",
}];

const IFFT2_INPUTS_CORE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input spectrum or signal.",
}];

const IFFT2_INPUTS_SIZE: [BuiltinParamDescriptor; 2] = [
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
        description: "Scalar N or two-element [M N] size vector.",
    },
];

const IFFT2_INPUTS_M_N: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input spectrum or signal.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Output row count for transform.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Output column count for transform.",
    },
];

const IFFT2_INPUTS_SYMFLAG: [BuiltinParamDescriptor; 2] = [
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

const IFFT2_INPUTS_SIZE_SYMFLAG: [BuiltinParamDescriptor; 3] = [
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
        description: "Scalar N or two-element [M N] size vector.",
    },
    BuiltinParamDescriptor {
        name: "symflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"nonsymmetric\""),
        description: "Symmetry flag: \"symmetric\" or \"nonsymmetric\".",
    },
];

const IFFT2_INPUTS_M_N_SYMFLAG: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input spectrum or signal.",
    },
    BuiltinParamDescriptor {
        name: "M",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Output row count for transform.",
    },
    BuiltinParamDescriptor {
        name: "N",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("[]"),
        description: "Output column count for transform.",
    },
    BuiltinParamDescriptor {
        name: "symflag",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"nonsymmetric\""),
        description: "Symmetry flag: \"symmetric\" or \"nonsymmetric\".",
    },
];

const IFFT2_SIGNATURES: [BuiltinSignatureDescriptor; 6] = [
    BuiltinSignatureDescriptor {
        label: "Y = ifft2(X)",
        inputs: &IFFT2_INPUTS_CORE,
        outputs: &IFFT2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifft2(X, SIZE)",
        inputs: &IFFT2_INPUTS_SIZE,
        outputs: &IFFT2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifft2(X, M, N)",
        inputs: &IFFT2_INPUTS_M_N,
        outputs: &IFFT2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifft2(X, symflag)",
        inputs: &IFFT2_INPUTS_SYMFLAG,
        outputs: &IFFT2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifft2(X, SIZE, symflag)",
        inputs: &IFFT2_INPUTS_SIZE_SYMFLAG,
        outputs: &IFFT2_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Y = ifft2(X, M, N, symflag)",
        inputs: &IFFT2_INPUTS_M_N_SYMFLAG,
        outputs: &IFFT2_OUTPUT,
    },
];

const IFFT2_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT2.ARG_COUNT",
    identifier: Some("RunMat:ifft2:ArgCount"),
    when: "More than four input arguments are supplied.",
    message: "ifft2: invalid argument count",
};

const IFFT2_ERROR_INVALID_LENGTH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT2.INVALID_LENGTH",
    identifier: Some("RunMat:ifft2:InvalidLength"),
    when: "Length/size arguments are invalid.",
    message: "ifft2: invalid transform length argument",
};

const IFFT2_ERROR_INVALID_SIZE_VECTOR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT2.INVALID_SIZE_VECTOR",
    identifier: Some("RunMat:ifft2:InvalidSizeVector"),
    when: "Single SIZE argument is invalid.",
    message: "ifft2: invalid size vector argument",
};

const IFFT2_ERROR_INVALID_SYMFLAG: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT2.INVALID_SYMFLAG",
    identifier: Some("RunMat:ifft2:InvalidSymflag"),
    when: "Symmetry flag is invalid or appears in an invalid position.",
    message: "ifft2: invalid symmetry flag usage",
};

const IFFT2_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT2.INVALID_INPUT",
    identifier: Some("RunMat:ifft2:InvalidInput"),
    when: "Input cannot be converted to supported numeric/complex domain.",
    message: "ifft2: invalid input",
};

const IFFT2_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.IFFT2.INTERNAL",
    identifier: Some("RunMat:ifft2:Internal"),
    when: "IFFT2 execution or tensor shaping fails.",
    message: "ifft2: internal error",
};

const IFFT2_ERRORS: [BuiltinErrorDescriptor; 6] = [
    IFFT2_ERROR_ARG_COUNT,
    IFFT2_ERROR_INVALID_LENGTH,
    IFFT2_ERROR_INVALID_SIZE_VECTOR,
    IFFT2_ERROR_INVALID_SYMFLAG,
    IFFT2_ERROR_INVALID_INPUT,
    IFFT2_ERROR_INTERNAL,
];

pub const IFFT2_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &IFFT2_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &IFFT2_ERRORS,
};

fn ifft2_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    ifft2_error_with_message(error.message, error)
}

fn ifft2_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    ifft2_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn ifft2_error_with_source(
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

fn ifft2_error_with_message(
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
    name = "ifft2",
    category = "math/fft",
    summary = "Compute two-dimensional inverse Fourier transforms.",
    keywords = "ifft2,inverse fft,image reconstruction,gpu",
    type_resolver(ifft2_type),
    descriptor(crate::builtins::math::fft::ifft2::IFFT2_DESCRIPTOR),
    builtin_path = "crate::builtins::math::fft::ifft2"
)]
async fn ifft2_builtin(value: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
    let ((len_rows, len_cols), symmetric) = parse_ifft2_arguments(&rest)?;
    match value {
        Value::GpuTensor(handle) => ifft2_gpu(handle, (len_rows, len_cols), symmetric).await,
        other => ifft2_host(other, (len_rows, len_cols), symmetric),
    }
}

fn ifft2_host(
    value: Value,
    lengths: (Option<usize>, Option<usize>),
    symmetric: bool,
) -> BuiltinResult<Value> {
    let tensor = value_to_complex_tensor(value, BUILTIN_NAME).map_err(|source| {
        ifft2_error_with_source(
            &IFFT2_ERROR_INVALID_INPUT,
            "input conversion failed",
            source,
        )
    })?;
    let transformed = ifft2_complex_tensor(tensor, lengths)?;
    finalize_ifft2_output(transformed, symmetric)
}

async fn ifft2_gpu(
    handle: GpuTensorHandle,
    lengths: (Option<usize>, Option<usize>),
    symmetric: bool,
) -> BuiltinResult<Value> {
    if matches!(lengths.0, Some(0)) || matches!(lengths.1, Some(0)) {
        return ifft2_gpu_fallback(handle, lengths, symmetric).await;
    }

    if let Some(provider) = runmat_accelerate_api::provider() {
        if let Ok(first) = provider.ifft_dim(&handle, lengths.0, 0).await {
            match provider.ifft_dim(&first, lengths.1, 1).await {
                Ok(second) => {
                    if first.buffer_id != second.buffer_id {
                        provider.free(&first).ok();
                        runmat_accelerate_api::clear_residency(&first);
                    }
                    if !symmetric {
                        return Ok(Value::GpuTensor(second));
                    }
                    if let Ok(real) = provider.fft_extract_real(&second).await {
                        provider.free(&second).ok();
                        runmat_accelerate_api::clear_residency(&second);
                        return Ok(Value::GpuTensor(real));
                    }
                    let complex =
                        download_provider_complex_tensor(provider, &second, BUILTIN_NAME, true)
                            .await
                            .map_err(|source| {
                                ifft2_error_with_source(
                                    &IFFT2_ERROR_INVALID_INPUT,
                                    "provider download failed",
                                    source,
                                )
                            })?;
                    return finalize_ifft2_output(complex, true);
                }
                Err(_) => {
                    let partial =
                        download_provider_complex_tensor(provider, &first, BUILTIN_NAME, true)
                            .await
                            .map_err(|source| {
                                ifft2_error_with_source(
                                    &IFFT2_ERROR_INVALID_INPUT,
                                    "provider download failed",
                                    source,
                                )
                            })?;
                    let completed = ifft_complex_tensor(partial, lengths.1, Some(2))?;
                    return finalize_ifft2_output(completed, symmetric);
                }
            }
        }
    }

    ifft2_gpu_fallback(handle, lengths, symmetric).await
}

async fn ifft2_gpu_fallback(
    handle: GpuTensorHandle,
    lengths: (Option<usize>, Option<usize>),
    symmetric: bool,
) -> BuiltinResult<Value> {
    let complex = gather_gpu_complex_tensor(&handle, BUILTIN_NAME)
        .await
        .map_err(|source| {
            ifft2_error_with_source(&IFFT2_ERROR_INVALID_INPUT, "gpu gather failed", source)
        })?;
    let transformed = ifft2_complex_tensor(complex, lengths)?;
    finalize_ifft2_output(transformed, symmetric)
}

fn ifft2_complex_tensor(
    tensor: ComplexTensor,
    lengths: (Option<usize>, Option<usize>),
) -> BuiltinResult<ComplexTensor> {
    let (len_rows, len_cols) = lengths;
    transform_axes_complex_tensor(
        tensor,
        &[len_rows, len_cols],
        TransformDirection::Inverse,
        BUILTIN_NAME,
    )
    .map_err(|source| ifft2_error_with_source(&IFFT2_ERROR_INTERNAL, "transform failed", source))
}

fn finalize_ifft2_output(tensor: ComplexTensor, symmetric: bool) -> BuiltinResult<Value> {
    if symmetric {
        complex_tensor_to_real_value(tensor, BUILTIN_NAME).map_err(|source| {
            ifft2_error_with_source(
                &IFFT2_ERROR_INTERNAL,
                "real-value extraction failed",
                source,
            )
        })
    } else {
        Ok(complex_tensor_into_value(tensor))
    }
}

type LengthPair = (Option<usize>, Option<usize>);
type LengthsAndSymmetry = (LengthPair, bool);

fn parse_ifft2_arguments(args: &[Value]) -> BuiltinResult<LengthsAndSymmetry> {
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
            let rows = parse_length(&rem[0], BUILTIN_NAME).map_err(|source| {
                ifft2_error_with_source(
                    &IFFT2_ERROR_INVALID_LENGTH,
                    "row-length parse failed",
                    source,
                )
            })?;
            let cols = parse_length(&rem[1], BUILTIN_NAME).map_err(|source| {
                ifft2_error_with_source(
                    &IFFT2_ERROR_INVALID_LENGTH,
                    "column-length parse failed",
                    source,
                )
            })?;
            (rows, cols)
        }
        _ => return Err(ifft2_error(&IFFT2_ERROR_ARG_COUNT)),
    };

    Ok((lengths, symmetry))
}

fn split_symflag(args: &[Value]) -> BuiltinResult<(Option<bool>, &[Value])> {
    if let Some((last, rest)) = args.split_last() {
        if let Some(flag) = parse_symflag(last, BUILTIN_NAME).map_err(|source| {
            ifft2_error_with_source(&IFFT2_ERROR_INVALID_SYMFLAG, "symflag parse failed", source)
        })? {
            // Ensure no earlier argument is also a symmetry flag.
            for value in rest {
                if parse_symflag(value, BUILTIN_NAME)
                    .map_err(|source| {
                        ifft2_error_with_source(
                            &IFFT2_ERROR_INVALID_SYMFLAG,
                            "symflag parse failed",
                            source,
                        )
                    })?
                    .is_some()
                {
                    return Err(ifft2_error_with_detail(
                        &IFFT2_ERROR_INVALID_SYMFLAG,
                        "symmetry flag must appear once at the end",
                    ));
                }
            }
            return Ok((Some(flag), rest));
        }
    }

    // Validate that no argument except the last is a symmetry flag.
    for value in args {
        if parse_symflag(value, BUILTIN_NAME)
            .map_err(|source| {
                ifft2_error_with_source(
                    &IFFT2_ERROR_INVALID_SYMFLAG,
                    "symflag parse failed",
                    source,
                )
            })?
            .is_some()
        {
            return Err(ifft2_error_with_detail(
                &IFFT2_ERROR_INVALID_SYMFLAG,
                "symmetry flag must appear as the final argument",
            ));
        }
    }

    Ok((None, args))
}

fn parse_ifft2_single(value: &Value) -> BuiltinResult<(Option<usize>, Option<usize>)> {
    match value {
        Value::Tensor(tensor) => {
            parse_2d_lengths_from_data(&tensor.data, BUILTIN_NAME).map_err(|source| {
                ifft2_error_with_detail(
                    &IFFT2_ERROR_INVALID_SIZE_VECTOR,
                    format!("size vector parse failed: {source}"),
                )
            })
        }
        Value::LogicalArray(logical) => {
            let tensor = tensor::logical_to_tensor(logical).map_err(|source| {
                ifft2_error_with_detail(
                    &IFFT2_ERROR_INVALID_SIZE_VECTOR,
                    format!("logical size-vector conversion failed: {source}"),
                )
            })?;
            parse_2d_lengths_from_data(&tensor.data, BUILTIN_NAME).map_err(|source| {
                ifft2_error_with_detail(
                    &IFFT2_ERROR_INVALID_SIZE_VECTOR,
                    format!("size vector parse failed: {source}"),
                )
            })
        }
        Value::Num(_) | Value::Int(_) => {
            let len = parse_length(value, BUILTIN_NAME).map_err(|source| {
                ifft2_error_with_source(&IFFT2_ERROR_INVALID_LENGTH, "length parse failed", source)
            })?;
            Ok((len, len))
        }
        Value::Complex(re, im) => {
            if im.abs() > f64::EPSILON {
                return Err(ifft2_error(&IFFT2_ERROR_INVALID_LENGTH));
            }
            let scalar = Value::Num(*re);
            let len = parse_length(&scalar, BUILTIN_NAME).map_err(|source| {
                ifft2_error_with_source(&IFFT2_ERROR_INVALID_LENGTH, "length parse failed", source)
            })?;
            Ok((len, len))
        }
        Value::ComplexTensor(_) => Err(ifft2_error(&IFFT2_ERROR_INVALID_SIZE_VECTOR)),
        Value::GpuTensor(_) => Err(ifft2_error(&IFFT2_ERROR_INVALID_SIZE_VECTOR)),
        Value::Bool(_) => Err(ifft2_error(&IFFT2_ERROR_INVALID_LENGTH)),
        Value::String(_)
        | Value::StringArray(_)
        | Value::CharArray(_)
        | Value::Symbolic(_)
        | Value::SymbolicArray(_)
        | Value::SparseTensor(_)
        | Value::Cell(_)
        | Value::Struct(_)
        | Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_)
        | Value::HandleObject(_)
        | Value::Listener(_)
        | Value::Object(_)
        | Value::ClassRef(_)
        | Value::MException(_)
        | Value::OutputList(_) => Err(ifft2_error(&IFFT2_ERROR_INVALID_LENGTH)),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use crate::builtins::math::fft::common;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        builtin_function_by_name, IntValue, ResolveContext, Tensor as HostTensor, Type,
    };

    fn approx_eq(a: (f64, f64), b: (f64, f64), tol: f64) -> bool {
        (a.0 - b.0).abs() <= tol && (a.1 - b.1).abs() <= tol
    }

    fn error_message(error: crate::RuntimeError) -> String {
        error.message().to_string()
    }

    fn error_identifier(error: &crate::RuntimeError) -> Option<&str> {
        error.identifier()
    }

    fn fft2_of_tensor(tensor: &HostTensor) -> ComplexTensor {
        let complex = value_to_complex_tensor(Value::Tensor(tensor.clone()), "fft2").unwrap();
        let first = super::super::fft::fft_complex_tensor(complex, None, Some(1)).unwrap();
        super::super::fft::fft_complex_tensor(first, None, Some(2)).unwrap()
    }

    fn value_to_host_complex(value: Value) -> ComplexTensor {
        match value {
            Value::ComplexTensor(ct) => ct,
            Value::GpuTensor(handle) => {
                let provider = runmat_accelerate_api::provider_for_handle(&handle)
                    .or_else(runmat_accelerate_api::provider)
                    .expect("provider for gpu handle");
                let host = block_on(provider.download(&handle)).expect("download gpu ifft2 output");
                common::host_to_complex_tensor(host, BUILTIN_NAME).expect("decode gpu complex")
            }
            other => panic!("expected complex value, got {other:?}"),
        }
    }

    #[test]
    fn ifft2_type_pads_rank() {
        let out = ifft2_type(
            &[Type::Tensor {
                shape: Some(vec![Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(1)])
            }
        );
    }

    #[test]
    fn ifft2_descriptor_signatures_and_errors() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("ifft2 builtin");
        let descriptor = builtin.descriptor.expect("ifft2 descriptor");
        let labels: Vec<&str> = descriptor.signatures.iter().map(|sig| sig.label).collect();
        assert!(labels.contains(&"Y = ifft2(X)"));
        assert!(labels.contains(&"Y = ifft2(X, SIZE)"));
        assert!(labels.contains(&"Y = ifft2(X, M, N)"));
        assert!(labels.contains(&"Y = ifft2(X, symflag)"));
        assert!(labels.contains(&"Y = ifft2(X, SIZE, symflag)"));
        assert!(labels.contains(&"Y = ifft2(X, M, N, symflag)"));
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.IFFT2.INVALID_SYMFLAG"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft2_rejects_boolean_length() {
        let tensor = HostTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let err =
            ifft2_builtin(Value::ComplexTensor(spectrum), vec![Value::Bool(true)]).unwrap_err();
        assert_eq!(
            error_identifier(&err),
            IFFT2_ERROR_INVALID_LENGTH.identifier
        );
        assert!(error_message(err).contains(IFFT2_ERROR_INVALID_LENGTH.message));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
        assert_eq!(error_identifier(&err), IFFT2_ERROR_ARG_COUNT.identifier);
        assert!(error_message(err).contains(IFFT2_ERROR_ARG_COUNT.message));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft2_gpu_roundtrip_matches_cpu() {
        test_support::with_test_provider(|provider| {
            let tensor = HostTensor::new((0..8).map(|v| v as f64).collect(), vec![2, 4]).unwrap();
            let spectrum = fft2_of_tensor(&tensor);
            let view = HostTensorView {
                data: &spectrum
                    .data
                    .iter()
                    .flat_map(|(re, im)| [*re, *im])
                    .collect::<Vec<_>>(),
                shape: &[2, 4, 2],
            };
            let raw = provider.upload(&view).expect("upload spectrum");
            let second = runmat_accelerate_api::GpuTensorHandle {
                shape: spectrum.shape.clone(),
                device_id: raw.device_id,
                buffer_id: raw.buffer_id,
            };
            runmat_accelerate_api::set_handle_storage(
                &second,
                runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
            );

            let gpu =
                ifft2_builtin(Value::GpuTensor(second.clone()), Vec::new()).expect("ifft2 gpu");
            let cpu = ifft2_builtin(Value::ComplexTensor(spectrum.clone()), Vec::new())
                .expect("ifft2 cpu");

            let g = value_to_host_complex(gpu);
            let c = value_to_host_complex(cpu);
            assert_eq!(g.shape, c.shape);
            for (lhs, rhs) in g.data.iter().zip(c.data.iter()) {
                assert!(approx_eq(*lhs, *rhs, 1e-10), "{lhs:?} vs {rhs:?}");
            }
            provider.free(&raw).ok();
            provider.free(&second).ok();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft2_rejects_unknown_symmetry_flag() {
        let err = parse_ifft2_arguments(&[Value::from("invalid")]).unwrap_err();
        assert_eq!(
            error_identifier(&err),
            IFFT2_ERROR_INVALID_SYMFLAG.identifier
        );
        assert!(error_message(err).contains(IFFT2_ERROR_INVALID_SYMFLAG.message));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn ifft2_requires_symflag_last() {
        let tensor = HostTensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let spectrum = fft2_of_tensor(&tensor);
        let err = ifft2_builtin(
            Value::ComplexTensor(spectrum),
            vec![Value::from("symmetric"), Value::Int(IntValue::I32(2))],
        )
        .unwrap_err();
        assert_eq!(
            error_identifier(&err),
            IFFT2_ERROR_INVALID_SYMFLAG.identifier
        );
        assert!(error_message(err).contains(IFFT2_ERROR_INVALID_SYMFLAG.message));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
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
        let view = HostTensorView {
            data: &host_real_imag,
            shape: &[4, 4, 2],
        };
        let raw = provider.upload(&view).expect("upload spectrum");
        let second = runmat_accelerate_api::GpuTensorHandle {
            shape: spectrum.shape.clone(),
            device_id: raw.device_id,
            buffer_id: raw.buffer_id,
        };
        runmat_accelerate_api::set_handle_storage(
            &second,
            runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
        );

        let gpu_val =
            ifft2_builtin(Value::GpuTensor(second.clone()), Vec::new()).expect("ifft2 gpu");
        let cpu_val = ifft2_builtin(Value::ComplexTensor(spectrum), Vec::new()).expect("ifft2 cpu");

        let gpu_ct = value_to_host_complex(gpu_val);
        let cpu_ct = value_to_host_complex(cpu_val);
        assert_eq!(gpu_ct.shape, cpu_ct.shape);

        let tol = match provider.precision() {
            runmat_accelerate_api::ProviderPrecision::F64 => 1e-10,
            runmat_accelerate_api::ProviderPrecision::F32 => 1e-5,
        };
        for (lhs, rhs) in gpu_ct.data.iter().zip(cpu_ct.data.iter()) {
            assert!(approx_eq(*lhs, *rhs, tol), "{lhs:?} vs {rhs:?}");
        }
        provider.free(&raw).ok();
        provider.free(&second).ok();
    }

    fn ifft2_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::ifft2_builtin(value, rest))
    }
}
