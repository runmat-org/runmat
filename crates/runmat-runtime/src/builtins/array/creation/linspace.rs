//! MATLAB-compatible `linspace` builtin with GPU-aware semantics for RunMat.

use log::trace;
use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinIntegerBackendRule,
    BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
    BuiltinOutputMode, BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType,
    BuiltinSignatureDescriptor, ComplexTensor, IntValue, NumericDType, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::build_runtime_error;
use crate::builtins::array::type_resolvers::row_vector_type;
use crate::builtins::common::residency::{sequence_gpu_preference, SequenceIntent};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use runmat_builtins::ResolveContext;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::linspace")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "linspace",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("linspace")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may generate sequences directly; the runtime uploads host-generated data when hooks are absent.",
};

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_builtin("linspace")
        .build()
}

fn linspace_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    linspace_error_with_message(error.message, error)
}

fn linspace_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> crate::RuntimeError {
    linspace_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn linspace_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("linspace");
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::linspace")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "linspace",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Sequence generation is treated as a sink and is not fused with other operations.",
};

fn linspace_type(_args: &[Type], ctx: &ResolveContext) -> Type {
    row_vector_type(ctx)
}

const LINSPACE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Row vector of linearly spaced values.",
}];

const LINSPACE_SIG_2_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "start",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Starting value.",
    },
    BuiltinParamDescriptor {
        name: "stop",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Ending value.",
    },
];

const LINSPACE_SIG_3_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "start",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Starting value.",
    },
    BuiltinParamDescriptor {
        name: "stop",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Ending value.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("100"),
        description: "Number of points.",
    },
];

const LINSPACE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "x = linspace(start, stop)",
        inputs: &LINSPACE_SIG_2_INPUTS,
        outputs: &LINSPACE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "x = linspace(start, stop, n)",
        inputs: &LINSPACE_SIG_3_INPUTS,
        outputs: &LINSPACE_OUTPUT,
    },
];

const LINSPACE_INTEGER_ENDPOINT_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "start",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer endpoints are rejected; linspace endpoints must be single or double.",
    },
    BuiltinIntegerInputCapability {
        name: "stop",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "Integer endpoints are rejected; linspace endpoints must be single or double.",
    },
];

const LINSPACE_INTEGER_COUNT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "All eight integer classes are accepted as exact structural counts.",
    }];

pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "x = linspace(integer_start, integer_stop[, n])",
        inputs: &LINSPACE_INTEGER_ENDPOINT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::ScalarOnly,
        notes:
            "Host and resident integer endpoints are rejected before numeric sequence generation.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "x = linspace(start, stop, integer_n)",
        inputs: &LINSPACE_INTEGER_COUNT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The count is read exactly; output class follows the floating-point endpoints.",
    },
];

const LINSPACE_ERROR_ARG_COUNT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINSPACE.ARG_COUNT",
    identifier: None,
    when: "More than three input arguments are provided.",
    message: "linspace: expected at most three input arguments",
};

const LINSPACE_ERROR_COUNT_NOT_SCALAR: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINSPACE.COUNT_NOT_SCALAR",
    identifier: None,
    when: "The count argument is not a numeric scalar value.",
    message: "linspace: number of points must be a scalar",
};

const LINSPACE_ERROR_COUNT_NOT_FINITE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINSPACE.COUNT_NOT_FINITE",
    identifier: None,
    when: "The count argument is infinite.",
    message: "linspace: number of points must not be infinite",
};

const LINSPACE_ERROR_COUNT_TOO_LARGE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.LINSPACE.COUNT_TOO_LARGE",
    identifier: None,
    when: "The count argument exceeds platform limits.",
    message: "linspace: number of points is too large for this platform",
};

const LINSPACE_ERRORS: [BuiltinErrorDescriptor; 4] = [
    LINSPACE_ERROR_ARG_COUNT,
    LINSPACE_ERROR_COUNT_NOT_SCALAR,
    LINSPACE_ERROR_COUNT_NOT_FINITE,
    LINSPACE_ERROR_COUNT_TOO_LARGE,
];

pub const LINSPACE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LINSPACE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LINSPACE_ERRORS,
};

#[runtime_builtin(
    name = "linspace",
    category = "array/creation",
    summary = "Generate linearly spaced row vectors.",
    keywords = "linspace,range,vector,gpu",
    examples = "x = linspace(0, 1, 5)  % [0 0.25 0.5 0.75 1]",
    accel = "array_construct",
    type_resolver(linspace_type),
    descriptor(crate::builtins::array::creation::linspace::LINSPACE_DESCRIPTOR),
    integer_capabilities(crate::builtins::array::creation::linspace::INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::array::creation::linspace"
)]
async fn linspace_builtin(
    start: Value,
    stop: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(linspace_error(&LINSPACE_ERROR_ARG_COUNT));
    }

    let (start_scalar, start_source) = parse_scalar("linspace", start).await?;
    let (stop_scalar, stop_source) = parse_scalar("linspace", stop).await?;
    if start_source
        .as_ref()
        .zip(stop_source.as_ref())
        .is_some_and(|(left, right)| left.device_id != right.device_id)
    {
        return Err(builtin_error(
            "linspace: GPU endpoints must belong to the same provider",
        ));
    }
    let source = start_source.as_ref().or(stop_source.as_ref());

    let count = if rest.is_empty() {
        Count::Length(100)
    } else {
        parse_count(&rest[0]).await?
    };
    if matches!(count, Count::Nan) {
        let single = start_scalar.single || stop_scalar.single;
        if let Some(source) = source {
            if let Some(provider) =
                gpu_helpers::exact_provider_for_handle(source).filter(|provider| {
                    !single || provider.precision() == runmat_accelerate_api::ProviderPrecision::F32
                })
            {
                let data = [f64::NAN];
                let shape = [1usize, 1usize];
                if let Ok(handle) = provider.upload(&HostTensorView {
                    data: &data,
                    shape: &shape,
                }) {
                    return validated_sequence_output(
                        source,
                        provider,
                        handle,
                        vec![1, 1],
                        if single {
                            NumericDType::F32
                        } else {
                            NumericDType::F64
                        },
                        "linspace",
                    );
                }
            }
        }
        return if single {
            Tensor::from_f32(vec![f32::NAN], vec![1, 1])
        } else {
            Tensor::new(vec![f64::NAN], vec![1, 1])
        }
        .map(Value::Tensor)
        .map_err(|error| builtin_error(format!("linspace: {error}")));
    }
    let Count::Length(count) = count else {
        unreachable!("NaN count returned above")
    };

    let residency = sequence_gpu_preference(count, SequenceIntent::Linspace, source.is_some());
    if log::log_enabled!(log::Level::Trace) {
        trace!(
            "linspace: len={} prefer_gpu={} reason={:?}",
            count,
            residency.prefer_gpu,
            residency.reason
        );
    }
    let prefer_gpu = residency.prefer_gpu;
    build_sequence(start_scalar, stop_scalar, count, prefer_gpu, source)
}

#[derive(Clone, Copy)]
enum Scalar {
    Real(f64),
    Complex { re: f64, im: f64 },
}

#[derive(Clone, Copy)]
struct Endpoint {
    scalar: Scalar,
    single: bool,
}

impl Endpoint {
    fn parts(&self) -> (f64, f64) {
        match self.scalar {
            Scalar::Real(r) => (r, 0.0),
            Scalar::Complex { re, im } => (re, im),
        }
    }

    fn is_complex(&self) -> bool {
        matches!(self.scalar, Scalar::Complex { .. })
    }
}

async fn parse_scalar(
    name: &str,
    value: Value,
) -> crate::BuiltinResult<(Endpoint, Option<runmat_accelerate_api::GpuTensorHandle>)> {
    match value {
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_integer_type(&handle).is_some()
                || runmat_accelerate_api::handle_is_logical(&handle)
            {
                return Err(builtin_error(format!(
                    "{name}: endpoints must be single or double scalars"
                )));
            }
            let provider = gpu_helpers::exact_provider_for_handle(&handle)
                .ok_or_else(|| builtin_error(format!("{name}: no provider owns the endpoint")))?;
            match gpu_helpers::download_value_preserving_residency_async(provider, &handle).await? {
                Value::Tensor(tensor) => {
                    tensor_scalar(name, &tensor).map(|scalar| (scalar, Some(handle)))
                }
                Value::ComplexTensor(tensor) => {
                    complex_tensor_scalar(name, &tensor).map(|scalar| (scalar, Some(handle)))
                }
                _ => Err(builtin_error(format!(
                    "{name}: endpoints must be single or double scalars"
                ))),
            }
        }
        other => parse_scalar_host(name, other),
    }
}

fn parse_scalar_host(
    name: &str,
    value: Value,
) -> crate::BuiltinResult<(Endpoint, Option<runmat_accelerate_api::GpuTensorHandle>)> {
    match value {
        Value::Num(n) => Ok((
            Endpoint {
                scalar: Scalar::Real(n),
                single: false,
            },
            None,
        )),
        Value::Complex(re, im) => Ok((
            Endpoint {
                scalar: Scalar::Complex { re, im },
                single: false,
            },
            None,
        )),
        Value::Int(_) | Value::Bool(_) | Value::LogicalArray(_) => Err(builtin_error(format!(
            "{name}: endpoints must be single or double scalars"
        ))),
        Value::Tensor(t) => tensor_scalar(name, &t).map(|scalar| (scalar, None)),
        Value::ComplexTensor(t) => complex_tensor_scalar(name, &t).map(|scalar| (scalar, None)),
        Value::GpuTensor(_) => unreachable!("GpuTensor handled by parse_scalar"),
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => Err(builtin_error(
            format!("{name}: endpoints must be numeric scalars; received a string-like value"),
        )),
        other => Err(builtin_error(format!(
            "{name}: endpoints must be numeric scalars; received {other:?}"
        ))),
    }
}

fn tensor_scalar(name: &str, tensor: &Tensor) -> crate::BuiltinResult<Endpoint> {
    if !tensor::is_scalar_tensor(tensor) {
        return Err(builtin_error(format!("{name}: expected scalar input")));
    }
    if !matches!(
        tensor.numeric_dtype(),
        runmat_builtins::NumericDType::F32 | runmat_builtins::NumericDType::F64
    ) {
        return Err(builtin_error(format!(
            "{name}: endpoints must be single or double scalars"
        )));
    }
    Ok(Endpoint {
        scalar: Scalar::Real(tensor::tensor_value_f64(tensor, 0)),
        single: tensor.numeric_dtype() == NumericDType::F32,
    })
}

fn complex_tensor_scalar(name: &str, tensor: &ComplexTensor) -> crate::BuiltinResult<Endpoint> {
    if tensor.integer_storage().is_some() {
        return Err(builtin_error(format!(
            "{name}: endpoints must be single or double scalars"
        )));
    }
    let values = tensor.materialize_f64();
    if values.len() != 1 {
        return Err(builtin_error(format!("{name}: expected scalar input")));
    }
    let (re, im) = values[0];
    Ok(Endpoint {
        scalar: Scalar::Complex { re, im },
        single: tensor.numeric_dtype() == NumericDType::F32,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Count {
    Length(usize),
    Nan,
}

async fn parse_count(value: &Value) -> crate::BuiltinResult<Count> {
    match value {
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_is_logical(handle) {
                return Err(linspace_error(&LINSPACE_ERROR_COUNT_NOT_SCALAR));
            }
            let tensor = gpu_helpers::gather_tensor_async(handle).await?;
            if !tensor::is_scalar_tensor(&tensor) {
                return Err(linspace_error(&LINSPACE_ERROR_COUNT_NOT_SCALAR));
            }
            parse_tensor_count(&tensor)
        }
        other => parse_count_host(other),
    }
}

fn parse_count_host(value: &Value) -> crate::BuiltinResult<Count> {
    match value {
        Value::Int(i) => parse_integer_count(i),
        Value::Num(n) => parse_numeric_count(*n),
        Value::Tensor(t) => {
            if !tensor::is_scalar_tensor(t) {
                return Err(linspace_error(&LINSPACE_ERROR_COUNT_NOT_SCALAR));
            }
            parse_tensor_count(t)
        }
        Value::GpuTensor(_) => unreachable!("GpuTensor handled by parse_count"),
        other => Err(linspace_error_with_detail(
            &LINSPACE_ERROR_COUNT_NOT_SCALAR,
            format!("got {other:?}"),
        )),
    }
}

fn parse_tensor_count(tensor: &Tensor) -> crate::BuiltinResult<Count> {
    if let Some(storage) = tensor.integer_storage() {
        let value = storage
            .value_at(0)
            .expect("scalar integer tensor has one storage value");
        return parse_integer_count(&value);
    }
    parse_numeric_count(tensor::tensor_value_f64(tensor, 0))
}

fn parse_integer_count(value: &IntValue) -> crate::BuiltinResult<Count> {
    if value.try_to_i64().is_some_and(|value| value < 0) {
        return Ok(Count::Length(0));
    }
    value
        .try_to_usize()
        .map(Count::Length)
        .ok_or_else(|| linspace_error(&LINSPACE_ERROR_COUNT_TOO_LARGE))
}

fn parse_numeric_count(raw: f64) -> crate::BuiltinResult<Count> {
    if raw.is_nan() {
        return Ok(Count::Nan);
    }
    if raw.is_infinite() {
        return Err(linspace_error(&LINSPACE_ERROR_COUNT_NOT_FINITE));
    }
    let floored = raw.floor();
    if floored <= 0.0 {
        return Ok(Count::Length(0));
    }
    if floored > usize::MAX as f64 || (usize::BITS == 64 && floored == usize::MAX as f64) {
        return Err(linspace_error(&LINSPACE_ERROR_COUNT_TOO_LARGE));
    }
    Ok(Count::Length(floored as usize))
}

fn build_sequence(
    start: Endpoint,
    stop: Endpoint,
    count: usize,
    prefer_gpu: bool,
    source: Option<&runmat_accelerate_api::GpuTensorHandle>,
) -> crate::BuiltinResult<Value> {
    let (start_re, start_im) = start.parts();
    let (stop_re, stop_im) = stop.parts();
    let complex = start.is_complex() || stop.is_complex();
    let single = start.single || stop.single;

    if complex {
        let data = generate_complex_sequence(start_re, start_im, stop_re, stop_im, count);
        let tensor = if single {
            ComplexTensor::from_f32(
                data.into_iter()
                    .map(|(real, imag)| (real as f32, imag as f32))
                    .collect(),
                vec![1, count],
            )
        } else {
            ComplexTensor::new(data, vec![1, count])
        }
        .map_err(|e| builtin_error(format!("linspace: {e}")))?;
        let value = Value::ComplexTensor(tensor);
        return match source {
            Some(source) => gpu_helpers::restore_class_preserving_value(source, value, "linspace"),
            None => Ok(value),
        };
    }

    if prefer_gpu {
        #[cfg(all(test, feature = "wgpu"))]
        {
            if runmat_accelerate_api::provider().is_none() {
                let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                    runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
                );
            }
        }
        let provider = source
            .and_then(gpu_helpers::exact_provider_for_handle)
            .or_else(runmat_accelerate_api::provider);
        if let Some(provider) = provider.filter(|provider| {
            !single || provider.precision() == runmat_accelerate_api::ProviderPrecision::F32
        }) {
            if count > 0 {
                if log::log_enabled!(log::Level::Trace) {
                    trace!(
                        "linspace: attempting provider.linspace start={} stop={} count={}",
                        start_re,
                        stop_re,
                        count
                    );
                }
                match provider.linspace(start_re, stop_re, count) {
                    Ok(handle) => {
                        trace!("linspace: provider.linspace succeeded");
                        if let Some(source) = source {
                            return validated_sequence_output(
                                source,
                                provider,
                                handle,
                                vec![1, count],
                                if single {
                                    NumericDType::F32
                                } else {
                                    NumericDType::F64
                                },
                                "linspace",
                            );
                        }
                        return validated_unowned_sequence_output(
                            provider,
                            handle,
                            vec![1, count],
                            if single {
                                NumericDType::F32
                            } else {
                                NumericDType::F64
                            },
                            "linspace",
                        );
                    }
                    Err(err) => {
                        trace!("linspace: provider.linspace failed: {err}");
                    }
                }
            }
        }
    }

    let data = generate_real_sequence(start_re, stop_re, count);
    if prefer_gpu {
        #[cfg(all(test, feature = "wgpu"))]
        {
            if runmat_accelerate_api::provider().is_none() {
                let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                    runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
                );
            }
        }
        let provider = source
            .and_then(gpu_helpers::exact_provider_for_handle)
            .or_else(runmat_accelerate_api::provider);
        if let Some(provider) = provider.filter(|provider| {
            !single || provider.precision() == runmat_accelerate_api::ProviderPrecision::F32
        }) {
            let shape = [1usize, count];
            let view = HostTensorView {
                data: &data,
                shape: &shape,
            };
            if let Ok(handle) = provider.upload(&view) {
                if let Some(source) = source {
                    return validated_sequence_output(
                        source,
                        provider,
                        handle,
                        vec![1, count],
                        if single {
                            NumericDType::F32
                        } else {
                            NumericDType::F64
                        },
                        "linspace",
                    );
                }
                return validated_unowned_sequence_output(
                    provider,
                    handle,
                    vec![1, count],
                    if single {
                        NumericDType::F32
                    } else {
                        NumericDType::F64
                    },
                    "linspace",
                );
            }
        }
    }

    let tensor = if single {
        Tensor::from_f32(
            data.into_iter().map(|value| value as f32).collect(),
            vec![1, count],
        )
    } else {
        Tensor::new(data, vec![1, count])
    }
    .map_err(|e| builtin_error(format!("linspace: {e}")))?;
    Ok(Value::Tensor(tensor))
}

fn validated_sequence_output(
    source: &runmat_accelerate_api::GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    output: runmat_accelerate_api::GpuTensorHandle,
    shape: Vec<usize>,
    dtype: NumericDType,
    builtin: &str,
) -> crate::BuiltinResult<Value> {
    let expected_precision = match dtype {
        NumericDType::F32 => runmat_accelerate_api::ProviderPrecision::F32,
        _ => runmat_accelerate_api::ProviderPrecision::F64,
    };
    let valid = !gpu_helpers::same_gpu_handle(source, &output)
        && output.shape == shape
        && output.device_id == source.device_id
        && gpu_helpers::exact_provider_for_handle(&output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
        && runmat_accelerate_api::handle_storage(&output)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(&output) == Some(expected_precision)
        && runmat_accelerate_api::handle_integer_type(&output).is_none()
        && !runmat_accelerate_api::handle_is_logical(&output);
    if !valid {
        gpu_helpers::free_unprotected_exact_owner(&output, &[source]);
        return Err(builtin_error(format!(
            "{builtin}: provider returned malformed sequence output"
        )));
    }
    runmat_accelerate_api::set_handle_provenance(
        &output,
        runmat_accelerate_api::handle_provenance(source)
            .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic),
    );
    Ok(gpu_helpers::resident_gpu_value(output))
}

fn validated_unowned_sequence_output(
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    output: runmat_accelerate_api::GpuTensorHandle,
    shape: Vec<usize>,
    dtype: NumericDType,
    builtin: &str,
) -> crate::BuiltinResult<Value> {
    let expected_precision = match dtype {
        NumericDType::F32 => runmat_accelerate_api::ProviderPrecision::F32,
        _ => runmat_accelerate_api::ProviderPrecision::F64,
    };
    let valid = output.shape == shape
        && output.device_id == provider.device_id()
        && gpu_helpers::exact_provider_for_handle(&output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
        && runmat_accelerate_api::handle_storage(&output)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(&output) == Some(expected_precision)
        && runmat_accelerate_api::handle_integer_type(&output).is_none()
        && !runmat_accelerate_api::handle_is_logical(&output);
    if !valid {
        gpu_helpers::free_unprotected_exact_owner(&output, &[]);
        return Err(builtin_error(format!(
            "{builtin}: provider returned malformed sequence output"
        )));
    }
    Ok(gpu_helpers::resident_gpu_value(output))
}

fn generate_real_sequence(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![stop];
    }
    let mut data = Vec::with_capacity(count);
    let step = (stop - start) / ((count - 1) as f64);
    for idx in 0..count {
        data.push(start + (idx as f64) * step);
    }
    if let Some(last) = data.last_mut() {
        *last = stop;
    }
    data
}

fn generate_complex_sequence(
    start_re: f64,
    start_im: f64,
    stop_re: f64,
    stop_im: f64,
    count: usize,
) -> Vec<(f64, f64)> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![(stop_re, stop_im)];
    }
    let mut data = Vec::with_capacity(count);
    let step_re = (stop_re - start_re) / ((count - 1) as f64);
    let step_im = (stop_im - start_im) / ((count - 1) as f64);
    for idx in 0..count {
        let re = start_re + (idx as f64) * step_re;
        let im = start_im + (idx as f64) * step_im;
        data.push((re, im));
    }
    if let Some(last) = data.last_mut() {
        *last = (stop_re, stop_im);
    }
    data
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn count_parser_preserves_representable_uint64() {
        assert_eq!(
            parse_count_host(&Value::Int(IntValue::U64(u64::MAX))).ok(),
            usize::try_from(u64::MAX).ok().map(Count::Length)
        );
        assert_eq!(
            parse_count_host(&Value::Int(IntValue::I64(-1))).unwrap(),
            Count::Length(0)
        );
        assert!(parse_count_host(&Value::Num(usize::MAX as f64)).is_err());
        assert!(parse_count_host(&Value::Num((usize::MAX as f64) + 1.0)).is_err());
    }
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};
    use runmat_builtins::{IntValue, IntegerComplexStorage, IntegerStorage, Tensor};

    fn linspace_builtin(
        start: Value,
        stop: Value,
        rest: Vec<Value>,
    ) -> crate::BuiltinResult<Value> {
        block_on(super::linspace_builtin(start, stop, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_basic() {
        let result = linspace_builtin(
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::Int(IntValue::I32(5))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 5]);
                let expected = [0.0, 0.25, 0.5, 0.75, 1.0];
                for (idx, expected_val) in expected.iter().enumerate() {
                    assert!((t.materialize_f64()[idx] - expected_val).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn linspace_count_parser_preserves_typed_integer_tensors_exactly() {
        let large = 9_007_199_254_740_993_u64;
        let count = Tensor::new_integer(IntegerStorage::U64(vec![large]), vec![1, 1]).unwrap();

        assert_eq!(
            parse_count_host(&Value::Tensor(count)).unwrap(),
            Count::Length(large as usize)
        );
    }

    #[test]
    fn linspace_count_parser_ignores_poisoned_f64_mirrors_for_all_integer_classes() {
        let storages = [
            IntegerStorage::I8(vec![2]),
            IntegerStorage::I16(vec![2]),
            IntegerStorage::I32(vec![2]),
            IntegerStorage::I64(vec![2]),
            IntegerStorage::U8(vec![2]),
            IntegerStorage::U16(vec![2]),
            IntegerStorage::U32(vec![2]),
            IntegerStorage::U64(vec![2]),
        ];

        for storage in storages {
            let count = Tensor::new_integer(storage, vec![1, 1]).expect("integer count");
            assert_eq!(
                parse_count_host(&Value::Tensor(count)).unwrap(),
                Count::Length(2)
            );
        }
    }

    #[test]
    fn linspace_count_parser_maps_negative_typed_integer_tensors_to_empty() {
        let count = Tensor::new_integer(IntegerStorage::I64(vec![-1]), vec![1, 1]).unwrap();

        assert_eq!(
            parse_count_host(&Value::Tensor(count)).unwrap(),
            Count::Length(0)
        );
    }

    #[test]
    fn linspace_rejects_real_integer_tensor_endpoints() {
        let start =
            Tensor::new_integer(IntegerStorage::I64(vec![-3]), vec![1, 1]).expect("start tensor");
        let stop =
            Tensor::new_integer(IntegerStorage::U64(vec![9]), vec![1, 1]).expect("stop tensor");

        let error = linspace_builtin(
            Value::Tensor(start),
            Value::Tensor(stop),
            vec![Value::Int(IntValue::I32(3))],
        )
        .expect_err("integer endpoints must be rejected");
        assert!(error.message().contains("single or double"));
    }

    #[test]
    fn linspace_type_is_row_vector() {
        assert_eq!(
            linspace_type(&[Type::Num, Type::Num], &ResolveContext::new(Vec::new())),
            Type::Tensor {
                shape: Some(vec![Some(1), None])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_default_count() {
        let result =
            linspace_builtin(Value::Num(-1.0), Value::Num(1.0), Vec::new()).expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 100]);
                assert!((t.materialize_f64().first().copied().unwrap() + 1.0).abs() < 1e-12);
                assert!((t.materialize_f64().last().copied().unwrap() - 1.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_zero_count() {
        let result = linspace_builtin(
            Value::Num(0.0),
            Value::Num(10.0),
            vec![Value::Int(IntValue::I32(0))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 0]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected empty tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_single_point() {
        let result = linspace_builtin(
            Value::Num(5.0),
            Value::Num(9.0),
            vec![Value::Int(IntValue::I32(1))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                assert!((t.materialize_f64()[0] - 9.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_fractional_count_rounds_down() {
        let result = linspace_builtin(Value::Num(0.0), Value::Num(1.0), vec![Value::Num(3.5)])
            .expect("fractional counts floor");
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor")
        };
        assert_eq!(tensor.shape, vec![1, 3]);
        assert_eq!(tensor.materialize_f64(), vec![0.0, 0.5, 1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_negative_count_returns_empty() {
        let result = linspace_builtin(
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::Int(IntValue::I32(-2))],
        )
        .expect("negative count");
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor")
        };
        assert_eq!(tensor.shape, vec![1, 0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_infinite_count_errors() {
        let err = linspace_builtin(
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::Num(f64::INFINITY)],
        )
        .expect_err("expected error");
        assert!(err.message().contains("finite"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_nan_count_returns_scalar_nan() {
        let result = linspace_builtin(Value::Num(0.0), Value::Num(1.0), vec![Value::Num(f64::NAN)])
            .expect("NaN count");
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor")
        };
        assert_eq!(tensor.shape, vec![1, 1]);
        assert!(tensor.materialize_f64()[0].is_nan());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_non_scalar_count_errors() {
        let sz = Tensor::new(vec![2.0, 3.0], vec![2, 1]).unwrap();
        let err = linspace_builtin(Value::Num(0.0), Value::Num(1.0), vec![Value::Tensor(sz)])
            .expect_err("expected error");
        assert!(err.message().contains("scalar"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_complex_sequence() {
        let result = linspace_builtin(
            Value::Complex(1.0, 1.0),
            Value::Complex(-3.0, 2.0),
            vec![Value::Int(IntValue::I32(4))],
        )
        .expect("linspace");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                let expected = [
                    (1.0, 1.0),
                    (-0.3333333333333333, 1.3333333333333333),
                    (-1.6666666666666667, 1.6666666666666667),
                    (-3.0, 2.0),
                ];
                for (idx, &(re, im)) in expected.iter().enumerate() {
                    let (r, i) = t.materialize_f64()[idx];
                    assert!((r - re).abs() < 1e-9);
                    assert!((i - im).abs() < 1e-9);
                }
            }
            other => panic!("expected complex tensor, got {other:?}"),
        }
    }

    #[test]
    fn linspace_rejects_complex_integer_tensor_endpoints() {
        let start_storage =
            IntegerComplexStorage::new(IntegerStorage::I16(vec![1]), IntegerStorage::I16(vec![1]))
                .expect("start storage");
        let start = ComplexTensor::new_integer(start_storage, vec![1, 1]).expect("start tensor");
        let stop_storage =
            IntegerComplexStorage::new(IntegerStorage::I16(vec![-3]), IntegerStorage::I16(vec![2]))
                .expect("stop storage");
        let stop = ComplexTensor::new_integer(stop_storage, vec![1, 1]).expect("stop tensor");

        let error = linspace_builtin(
            Value::ComplexTensor(start),
            Value::ComplexTensor(stop),
            vec![Value::Int(IntValue::I32(4))],
        )
        .expect_err("integer endpoints must be rejected");
        assert!(error.message().contains("single or double"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_rejects_logical_endpoints() {
        let error = linspace_builtin(
            Value::Bool(true),
            Value::Bool(false),
            vec![Value::Int(IntValue::I32(3))],
        )
        .expect_err("logical endpoints must be rejected");
        assert!(error.message().contains("single or double"));
    }

    #[test]
    fn linspace_rejects_scalar_integer_endpoints() {
        let error = linspace_builtin(
            Value::Int(IntValue::I8(0)),
            Value::Num(1.0),
            vec![Value::Int(IntValue::I32(3))],
        )
        .expect_err("integer endpoints must be rejected");
        assert!(error.message().contains("single or double"));
    }

    #[test]
    fn linspace_rejects_resident_integer_endpoints_from_metadata() {
        test_support::with_test_provider(|provider| {
            let endpoint = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&[9_007_199_254_740_993]),
                    shape: &[1, 1],
                })
                .expect("integer endpoint upload");
            let error = linspace_builtin(
                Value::GpuTensor(endpoint.clone()),
                Value::Num(1.0),
                vec![Value::Int(IntValue::I32(3))],
            )
            .expect_err("resident integer endpoint must be rejected");
            assert!(error.message().contains("single or double"));
            provider.free(&endpoint).ok();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_rejects_logical_count() {
        let error = linspace_builtin(Value::Num(3.0), Value::Num(7.0), vec![Value::Bool(true)])
            .expect_err("logical count must be rejected");
        assert!(error.message().contains("scalar"));
    }

    #[test]
    fn linspace_rejects_resident_logical_count_before_gathering() {
        test_support::with_test_provider(|provider| {
            let count = provider
                .upload(&HostTensorView {
                    data: &[3.0],
                    shape: &[1, 1],
                })
                .expect("count upload");
            runmat_accelerate_api::set_handle_logical(&count, true);
            let error = linspace_builtin(
                Value::Num(3.0),
                Value::Num(7.0),
                vec![Value::GpuTensor(count.clone())],
            )
            .expect_err("resident logical count must be rejected");
            assert!(error.message().contains("scalar"));
            provider.free(&count).ok();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_tensor_scalar_arguments() {
        let start = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let stop = Tensor::new(vec![4.0], vec![1, 1]).unwrap();
        let result = linspace_builtin(
            Value::Tensor(start),
            Value::Tensor(stop),
            vec![Value::Int(IntValue::I32(3))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                let expected = [2.0, 3.0, 4.0];
                for (idx, expected_val) in expected.iter().enumerate() {
                    assert!((t.materialize_f64()[idx] - expected_val).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn linspace_preserves_single_real_and_complex_endpoint_class() {
        let start = Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap();
        let stop = Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap();
        let result = linspace_builtin(
            Value::Tensor(start),
            Value::Tensor(stop),
            vec![Value::Num(3.9)],
        )
        .unwrap();
        let Value::Tensor(result) = result else {
            panic!("expected real single tensor")
        };
        assert_eq!(result.numeric_dtype(), NumericDType::F32);
        assert_eq!(result.as_f32_slice(), Some(&[0.0, 0.5, 1.0][..]));

        let start = ComplexTensor::from_f32(vec![(0.0, 0.0)], vec![1, 1]).unwrap();
        let stop = ComplexTensor::from_f32(vec![(1.0, 0.0)], vec![1, 1]).unwrap();
        let result = linspace_builtin(
            Value::ComplexTensor(start),
            Value::ComplexTensor(stop),
            vec![Value::Int(IntValue::I32(2))],
        )
        .unwrap();
        let Value::ComplexTensor(result) = result else {
            panic!("expected complex single tensor")
        };
        assert_eq!(result.numeric_dtype(), NumericDType::F32);
        assert_eq!(result.as_f32_slice(), Some(&[(0.0, 0.0), (1.0, 0.0)][..]));
    }

    #[test]
    fn linspace_nan_count_preserves_single_class() {
        let start = Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap();
        let stop = Tensor::from_f32(vec![1.0], vec![1, 1]).unwrap();
        let result = linspace_builtin(
            Value::Tensor(start),
            Value::Tensor(stop),
            vec![Value::Num(f64::NAN)],
        )
        .unwrap();
        let Value::Tensor(result) = result else {
            panic!("expected single NaN tensor")
        };
        assert_eq!(result.numeric_dtype(), NumericDType::F32);
        assert!(result.as_f32_slice().unwrap()[0].is_nan());
    }

    #[test]
    fn linspace_rejects_typed_integer_tensor_endpoints() {
        let start = Tensor::new_integer(IntegerStorage::I16(vec![-2]), vec![1, 1]).expect("start");
        let stop = Tensor::new_integer(IntegerStorage::U16(vec![4]), vec![1, 1]).expect("stop");

        let error = linspace_builtin(
            Value::Tensor(start),
            Value::Tensor(stop),
            vec![Value::Int(IntValue::I32(4))],
        )
        .expect_err("integer endpoints must be rejected");
        assert!(error.message().contains("single or double"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_equal_endpoints_fill_with_endpoint() {
        let result = linspace_builtin(
            Value::Num(5.0),
            Value::Num(5.0),
            vec![Value::Int(IntValue::I32(4))],
        )
        .expect("linspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                assert!(t.materialize_f64().iter().all(|v| (*v - 5.0).abs() < 1e-12));
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let start = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let stop = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let start_view = runmat_accelerate_api::HostTensorView {
                data: &start.materialize_f64(),
                shape: &start.shape,
            };
            let stop_view = runmat_accelerate_api::HostTensorView {
                data: &stop.materialize_f64(),
                shape: &stop.shape,
            };
            let start_handle = provider.upload(&start_view).expect("upload start");
            let stop_handle = provider.upload(&stop_view).expect("upload stop");
            let result = linspace_builtin(
                Value::GpuTensor(start_handle),
                Value::GpuTensor(stop_handle),
                vec![Value::Int(IntValue::I32(5))],
            )
            .expect("linspace");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    let expected = [0.0, 0.25, 0.5, 0.75, 1.0];
                    assert_eq!(gathered.shape, vec![1, 5]);
                    for (idx, expected_val) in expected.iter().enumerate() {
                        assert!((gathered.materialize_f64()[idx] - expected_val).abs() < 1e-12);
                    }
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn linspace_gpu_nan_count_preserves_residency() {
        test_support::with_test_provider(|provider| {
            let start = provider
                .upload(&HostTensorView {
                    data: &[0.0],
                    shape: &[1, 1],
                })
                .unwrap();
            let result = linspace_builtin(
                Value::GpuTensor(start.clone()),
                Value::Num(1.0),
                vec![Value::Num(f64::NAN)],
            )
            .unwrap();
            assert!(matches!(result, Value::GpuTensor(_)));
            let gathered = test_support::gather(result).unwrap();
            assert_eq!(gathered.shape, vec![1, 1]);
            assert!(gathered.materialize_f64()[0].is_nan());
            provider.free(&start).ok();
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn linspace_gpu_zero_count_produces_gpu_empty_vector() {
        test_support::with_test_provider(|provider| {
            let start = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let stop = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
            let start_view = runmat_accelerate_api::HostTensorView {
                data: &start.materialize_f64(),
                shape: &start.shape,
            };
            let stop_view = runmat_accelerate_api::HostTensorView {
                data: &stop.materialize_f64(),
                shape: &stop.shape,
            };
            let start_handle = provider.upload(&start_view).expect("upload start");
            let stop_handle = provider.upload(&stop_view).expect("upload stop");
            let result = linspace_builtin(
                Value::GpuTensor(start_handle),
                Value::GpuTensor(stop_handle),
                vec![Value::Int(IntValue::I32(0))],
            )
            .expect("linspace");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape, vec![1, 0]);
                    assert!(gathered.materialize_f64().is_empty());
                }
                other => panic!("expected gpu tensor, got {other:?}"),
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn linspace_wgpu_matches_cpu() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };
        use runmat_accelerate_api::{AccelProvider, HostTensorView, ProviderPrecision};

        let Ok(provider) = register_wgpu_provider(WgpuProviderOptions::default()) else {
            return;
        };

        let start = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
        let stop = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let start_view = HostTensorView {
            data: &start.materialize_f64(),
            shape: &start.shape,
        };
        let stop_view = HostTensorView {
            data: &stop.materialize_f64(),
            shape: &stop.shape,
        };
        let start_handle = provider.upload(&start_view).expect("upload start");
        let stop_handle = provider.upload(&stop_view).expect("upload stop");

        let result = linspace_builtin(
            Value::GpuTensor(start_handle),
            Value::GpuTensor(stop_handle),
            vec![Value::Int(IntValue::I32(9))],
        )
        .expect("linspace");
        let gathered = test_support::gather(result).expect("gather");
        let expected = generate_real_sequence(0.0, 1.0, 9);

        let precision = runmat_accelerate_api::provider()
            .expect("provider")
            .precision();
        let tol = match precision {
            ProviderPrecision::F64 => 1e-12,
            ProviderPrecision::F32 => 1e-5,
        };

        assert_eq!(gathered.shape, vec![1, 9]);
        for (idx, expected_value) in expected.iter().enumerate() {
            let actual = gathered.materialize_f64()[idx];
            assert!(
                (actual - expected_value).abs() <= tol,
                "mismatch at {idx}: gpu={} expected={}",
                actual,
                expected_value
            );
        }
    }
}
