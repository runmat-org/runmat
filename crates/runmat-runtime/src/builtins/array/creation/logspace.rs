//! MATLAB-compatible `logspace` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::HostTensorView;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor, Type,
};
use runmat_builtins::{
    BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor, BuiltinIntegerComputationDomain,
    BuiltinIntegerInputAvailability, BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule,
    BuiltinIntegerOverflowRule, BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, IntValue, NumericDType, Tensor, Value};

use crate::build_runtime_error;
use crate::builtins::array::type_resolvers::row_vector_type;
use crate::builtins::common::residency::{sequence_gpu_preference, SequenceIntent};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use runmat_builtins::ResolveContext;

const LN_10: f64 = std::f64::consts::LN_10;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::logspace")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "logspace",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[
        ProviderHook::Custom("linspace"),
        ProviderHook::Custom("scalar_mul"),
        ProviderHook::Unary {
            name: "unary_exp",
        },
    ],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Providers may implement a dedicated logspace path or compose it from linspace + scalar multiply + unary_exp. The runtime uploads host-generated data when hooks are unavailable.",
};

fn builtin_error(message: impl Into<String>) -> crate::RuntimeError {
    build_runtime_error(message)
        .with_builtin("logspace")
        .build()
}

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::logspace")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "logspace",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Sequence generation is treated as a sink and is not fused with other operations.",
};

fn logspace_type(_args: &[Type], ctx: &ResolveContext) -> Type {
    row_vector_type(ctx)
}

const LOGSPACE_INTEGER_COUNT_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "n",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Documented,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The point count accepts an exact real integer scalar; nonpositive signed values produce an empty row and oversized values reject before allocation.",
    }];
const LOGSPACE_INTEGER_ENDPOINT_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "start",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes: "Endpoints are restricted to documented single or double scalars.",
    },
    BuiltinIntegerInputCapability {
        name: "stop",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::Rejected,
        scalar_double: BuiltinIntegerScalarDoubleRule::Rejected,
        notes: "Endpoints are restricted to documented single or double scalars.",
    },
];
pub const LOGSPACE_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 2] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "x = logspace(start,stop,integer_n)",
        inputs: &LOGSPACE_INTEGER_COUNT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::FunctionSpecific,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "The exact integer count controls row length; floating endpoints determine output precision and explicit endpoint residency determines device placement.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "x = logspace(integer_start,integer_stop[,n])",
        inputs: &LOGSPACE_INTEGER_ENDPOINT_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::NotApplicable,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostOnly,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Typed-integer endpoints are rejected on host and from resident dtype metadata before provider access.",
    },
];

const LOGSPACE_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "x",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Row vector of logarithmically spaced values.",
}];

const LOGSPACE_SIG_2_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "start",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Starting exponent value.",
    },
    BuiltinParamDescriptor {
        name: "stop",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Ending exponent value.",
    },
];

const LOGSPACE_SIG_3_INPUTS: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "start",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Starting exponent value.",
    },
    BuiltinParamDescriptor {
        name: "stop",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Ending exponent value.",
    },
    BuiltinParamDescriptor {
        name: "n",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("50"),
        description: "Number of points.",
    },
];

const LOGSPACE_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "x = logspace(start, stop)",
        inputs: &LOGSPACE_SIG_2_INPUTS,
        outputs: &LOGSPACE_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "x = logspace(start, stop, n)",
        inputs: &LOGSPACE_SIG_3_INPUTS,
        outputs: &LOGSPACE_OUTPUT,
    },
];

const LOGSPACE_ERRORS: [BuiltinErrorDescriptor; 6] = [
    BuiltinErrorDescriptor {
        code: "RM.LOGSPACE.ARG_COUNT",
        identifier: None,
        when: "More than three input arguments are provided.",
        message: "logspace: expected two or three input arguments",
    },
    BuiltinErrorDescriptor {
        code: "RM.LOGSPACE.COUNT_NOT_SCALAR",
        identifier: None,
        when: "The count argument is not a numeric scalar value.",
        message: "logspace: number of points must be a scalar",
    },
    BuiltinErrorDescriptor {
        code: "RM.LOGSPACE.COUNT_NOT_FINITE",
        identifier: None,
        when: "The count argument is NaN or infinite.",
        message: "logspace: number of points must be finite",
    },
    BuiltinErrorDescriptor {
        code: "RM.LOGSPACE.COUNT_TOO_LARGE",
        identifier: None,
        when: "The count argument exceeds platform limits.",
        message: "logspace: number of points is too large for this platform",
    },
    BuiltinErrorDescriptor {
        code: "RM.LOGSPACE.PROVIDER_OWNERSHIP_MISMATCH",
        identifier: Some("RunMat:gpu:ProviderOwnershipMismatch"),
        when: "Resident scalar inputs do not have one exact owning provider.",
        message: "logspace: GPU scalar inputs must have the same owning provider",
    },
    BuiltinErrorDescriptor {
        code: "RM.LOGSPACE.GPU_UPLOAD_FAILED",
        identifier: Some("RunMat:logspace:GpuUploadFailed"),
        when: "An explicitly resident result cannot be restored to its source provider.",
        message: "logspace: failed to preserve explicit gpuArray residency",
    },
];

pub const LOGSPACE_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &LOGSPACE_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &LOGSPACE_ERRORS,
};

fn terminal_gpu_error(
    message: impl Into<String>,
    descriptor: &'static BuiltinErrorDescriptor,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin("logspace");
    if let Some(identifier) = descriptor.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}

#[runtime_builtin(
    name = "logspace",
    category = "array/creation",
    summary = "Generate logarithmically spaced row vectors.",
    keywords = "logspace,logarithmic,vector,gpu",
    examples = "x = logspace(1, 3, 3)  % [10 100 1000]",
    accel = "array_construct",
    type_resolver(logspace_type),
    descriptor(crate::builtins::array::creation::logspace::LOGSPACE_DESCRIPTOR),
    integer_capabilities(
        crate::builtins::array::creation::logspace::LOGSPACE_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::array::creation::logspace"
)]
async fn logspace_builtin(
    start: Value,
    stop: Value,
    rest: Vec<Value>,
) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(builtin_error(
            "logspace: expected two or three input arguments",
        ));
    }

    let (start_scalar, start_residency) = parse_scalar("logspace", start).await?;
    let (stop_scalar, stop_residency) = parse_scalar("logspace", stop).await?;
    let owner = merge_source_owners(start_residency, stop_residency)?;
    let count = if rest.is_empty() {
        50usize
    } else {
        parse_count(&rest[0]).await?
    };

    let prefer_gpu =
        sequence_gpu_preference(count, SequenceIntent::Logspace, owner.is_some()).prefer_gpu;
    build_sequence(start_scalar, stop_scalar, count, prefer_gpu, owner).await
}

#[derive(Clone, Copy)]
struct SourceResidency {
    owner: &'static dyn runmat_accelerate_api::AccelProvider,
    explicit: bool,
}

fn merge_source_owners(
    lhs: Option<SourceResidency>,
    rhs: Option<SourceResidency>,
) -> crate::BuiltinResult<Option<SourceResidency>> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) if !std::ptr::eq(lhs.owner, rhs.owner) => Err(terminal_gpu_error(
            "logspace: GPU scalar inputs must have the same owning provider",
            &LOGSPACE_ERRORS[4],
        )),
        (Some(lhs), Some(rhs)) => Ok(Some(SourceResidency {
            owner: lhs.owner,
            explicit: lhs.explicit || rhs.explicit,
        })),
        (Some(value), None) | (None, Some(value)) => Ok(Some(value)),
        (None, None) => Ok(None),
    }
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
) -> crate::BuiltinResult<(Endpoint, Option<SourceResidency>)> {
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
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_integer_type(&handle).is_some()
                || runmat_accelerate_api::handle_is_logical(&handle)
            {
                return Err(builtin_error(format!(
                    "{name}: endpoints must be single or double scalars"
                )));
            }
            let owner = gpu_helpers::exact_provider_for_handle(&handle).ok_or_else(|| {
                terminal_gpu_error(
                    "logspace: no exact owner for GPU scalar input",
                    &LOGSPACE_ERRORS[4],
                )
            })?;
            let source = Some(SourceResidency {
                owner,
                explicit: runmat_accelerate_api::handle_is_explicit(&handle),
            });
            match gpu_helpers::download_value_preserving_residency_async(owner, &handle).await? {
                Value::Tensor(tensor) => {
                    tensor_scalar(name, &tensor).map(|scalar| (scalar, source))
                }
                Value::ComplexTensor(tensor) => {
                    complex_tensor_scalar(name, &tensor).map(|scalar| (scalar, source))
                }
                _ => Err(builtin_error(format!(
                    "{name}: endpoints must be single or double scalars"
                ))),
            }
        }
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
        runmat_value::NumericDType::F32 | runmat_value::NumericDType::F64
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

async fn parse_count(value: &Value) -> crate::BuiltinResult<usize> {
    match value {
        Value::Int(i) => parse_integer_count(i),
        Value::Num(n) => parse_numeric_count(*n),
        Value::Tensor(t) => {
            if !tensor::is_scalar_tensor(t) {
                return Err(builtin_error("logspace: number of points must be a scalar"));
            }
            parse_tensor_count(t)
        }
        Value::GpuTensor(handle) => {
            let tensor = gpu_helpers::gather_tensor_async(handle).await?;
            if !tensor::is_scalar_tensor(&tensor) {
                return Err(builtin_error("logspace: number of points must be a scalar"));
            }
            parse_tensor_count(&tensor)
        }
        other => Err(builtin_error(format!(
            "logspace: number of points must be numeric, got {other:?}"
        ))),
    }
}

fn parse_tensor_count(tensor: &Tensor) -> crate::BuiltinResult<usize> {
    if let Some(storage) = tensor.integer_storage() {
        let value = storage
            .value_at(0)
            .expect("scalar integer tensor has one storage value");
        return parse_integer_count(&value);
    }
    parse_numeric_count(tensor::tensor_value_f64(tensor, 0))
}

fn parse_integer_count(value: &IntValue) -> crate::BuiltinResult<usize> {
    if value.try_to_i64().is_some_and(|value| value < 0) {
        return Ok(0);
    }
    value
        .try_to_usize()
        .ok_or_else(|| builtin_error("logspace: number of points is too large for this platform"))
}

fn parse_numeric_count(raw: f64) -> crate::BuiltinResult<usize> {
    if !raw.is_finite() {
        return Err(builtin_error("logspace: number of points must be finite"));
    }
    let floored = raw.floor();
    if floored <= 0.0 {
        return Ok(0);
    }
    if floored > usize::MAX as f64 || (usize::BITS == 64 && floored == usize::MAX as f64) {
        return Err(builtin_error(
            "logspace: number of points is too large for this platform",
        ));
    }
    Ok(floored as usize)
}

async fn build_sequence(
    start: Endpoint,
    stop: Endpoint,
    count: usize,
    prefer_gpu: bool,
    source: Option<SourceResidency>,
) -> crate::BuiltinResult<Value> {
    let (start_re, start_im) = start.parts();
    let (mut stop_re, stop_im) = stop.parts();
    if stop_im == 0.0 && stop_re == std::f64::consts::PI {
        stop_re = std::f64::consts::PI.log10();
    }
    let complex = start.is_complex() || stop.is_complex();
    let single = start.single || stop.single;

    if complex {
        let data = generate_complex_log_sequence(start_re, start_im, stop_re, stop_im, count);
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
        .map_err(|e| builtin_error(format!("logspace: {e}")))?;
        if let Some(source) = source {
            if let Ok(handle) = gpu_helpers::upload_complex_tensor(source.owner, &tensor) {
                if valid_uploaded_sequence_handle(
                    &handle,
                    count,
                    source.owner,
                    runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved,
                ) {
                    return Ok(finalize_sequence_handle(handle, Some(source)));
                }
                gpu_helpers::free_unprotected_exact_owner(&handle, &[]);
            }
            if source.explicit {
                return Err(terminal_gpu_error(
                    "logspace: failed to preserve explicit gpuArray residency",
                    &LOGSPACE_ERRORS[5],
                ));
            }
        }
        return Ok(Value::ComplexTensor(tensor));
    }

    if prefer_gpu
        && source
            .map(|source| source.owner)
            .or_else(runmat_accelerate_api::provider)
            .is_some_and(|provider| {
                !single || provider.precision() == runmat_accelerate_api::ProviderPrecision::F32
            })
    {
        if let Some(value) = try_gpu_logspace(start_re, stop_re, count, source).await {
            return Ok(value);
        }
    }

    let data = generate_real_log_sequence(start_re, stop_re, count);
    if prefer_gpu {
        #[cfg(all(test, feature = "wgpu"))]
        {
            if runmat_accelerate_api::provider().is_none() {
                let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                    runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
                );
            }
        }
        if let Some(provider) = source
            .map(|source| source.owner)
            .or_else(runmat_accelerate_api::provider)
            .filter(|provider| {
                !single || provider.precision() == runmat_accelerate_api::ProviderPrecision::F32
            })
        {
            let shape = [1usize, count];
            let view = HostTensorView {
                data: &data,
                shape: &shape,
            };
            if let Ok(handle) = provider.upload(&view) {
                if valid_uploaded_sequence_handle(
                    &handle,
                    count,
                    provider,
                    runmat_accelerate_api::GpuTensorStorage::Real,
                ) {
                    return Ok(finalize_sequence_handle(handle, source));
                }
                gpu_helpers::free_unprotected_exact_owner(&handle, &[]);
            }
        }
        if source.is_some_and(|source| source.explicit) {
            return Err(terminal_gpu_error(
                "logspace: failed to preserve explicit gpuArray residency",
                &LOGSPACE_ERRORS[5],
            ));
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
    .map_err(|e| builtin_error(format!("logspace: {e}")))?;
    Ok(Value::Tensor(tensor))
}

async fn try_gpu_logspace(
    start: f64,
    stop: f64,
    count: usize,
    source: Option<SourceResidency>,
) -> Option<Value> {
    #[cfg(all(test, feature = "wgpu"))]
    {
        if runmat_accelerate_api::provider().is_none() {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
    }
    let provider = source
        .map(|source| source.owner)
        .or_else(runmat_accelerate_api::provider)?;
    let exponents = provider.linspace(start, stop, count).ok()?;
    if !valid_sequence_handle(&exponents, count, provider, &[]) {
        gpu_helpers::free_unprotected_exact_owner(&exponents, &[]);
        return None;
    }

    let scaled = match provider.scalar_mul(&exponents, LN_10) {
        Ok(handle) if valid_sequence_handle(&handle, count, provider, &[&exponents]) => handle,
        Ok(handle) => {
            gpu_helpers::free_unprotected_exact_owner(&handle, &[&exponents]);
            provider.free(&exponents).ok();
            return None;
        }
        Err(_) => {
            provider.free(&exponents).ok();
            return None;
        }
    };
    provider.free(&exponents).ok();

    let result = match provider.unary_exp(&scaled).await {
        Ok(handle) if valid_sequence_handle(&handle, count, provider, &[&scaled]) => handle,
        Ok(handle) => {
            gpu_helpers::free_unprotected_exact_owner(&handle, &[&scaled]);
            provider.free(&scaled).ok();
            return None;
        }
        Err(_) => {
            provider.free(&scaled).ok();
            return None;
        }
    };
    provider.free(&scaled).ok();

    Some(finalize_sequence_handle(result, source))
}

fn valid_uploaded_sequence_handle(
    output: &runmat_accelerate_api::GpuTensorHandle,
    count: usize,
    owner: &'static dyn runmat_accelerate_api::AccelProvider,
    storage: runmat_accelerate_api::GpuTensorStorage,
) -> bool {
    output.shape == [1, count]
        && output.device_id == owner.device_id()
        && runmat_accelerate_api::handle_storage(output) == storage
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && runmat_accelerate_api::handle_precision(output) == Some(owner.precision())
        && gpu_helpers::exact_provider_for_handle(output)
            .is_some_and(|candidate| std::ptr::eq(candidate, owner))
}

fn finalize_sequence_handle(
    mut handle: runmat_accelerate_api::GpuTensorHandle,
    source: Option<SourceResidency>,
) -> Value {
    let provenance = if source.is_some_and(|source| source.explicit) {
        runmat_accelerate_api::GpuHandleProvenance::Explicit
    } else {
        runmat_accelerate_api::GpuHandleProvenance::Automatic
    };
    runmat_accelerate_api::set_handle_provenance(&mut handle, provenance);
    gpu_helpers::resident_gpu_value(handle)
}

fn valid_sequence_handle(
    output: &runmat_accelerate_api::GpuTensorHandle,
    count: usize,
    owner: &'static dyn runmat_accelerate_api::AccelProvider,
    protected: &[&runmat_accelerate_api::GpuTensorHandle],
) -> bool {
    let expected_precision = protected
        .first()
        .and_then(|input| runmat_accelerate_api::handle_precision(input))
        .unwrap_or_else(|| owner.precision());
    output.shape == [1, count]
        && output.device_id == owner.device_id()
        && protected
            .iter()
            .all(|input| !gpu_helpers::same_gpu_handle(output, input))
        && runmat_accelerate_api::handle_storage(output)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && runmat_accelerate_api::handle_precision(output) == Some(expected_precision)
        && gpu_helpers::exact_provider_for_handle(output)
            .is_some_and(|candidate| std::ptr::eq(candidate, owner))
}

fn generate_real_log_sequence(start: f64, stop: f64, count: usize) -> Vec<f64> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![10f64.powf(stop)];
    }
    let mut data = Vec::with_capacity(count);
    let step = (stop - start) / ((count - 1) as f64);
    for idx in 0..count {
        let exponent = start + (idx as f64) * step;
        data.push(10f64.powf(exponent));
    }
    if let Some(first) = data.first_mut() {
        *first = 10f64.powf(start);
    }
    if let Some(last) = data.last_mut() {
        *last = 10f64.powf(stop);
    }
    data
}

fn generate_complex_log_sequence(
    start_re: f64,
    start_im: f64,
    stop_re: f64,
    stop_im: f64,
    count: usize,
) -> Vec<(f64, f64)> {
    if count == 0 {
        return Vec::new();
    }
    let steps = generate_complex_sequence(start_re, start_im, stop_re, stop_im, count);
    steps
        .into_iter()
        .map(|(re, im)| complex_pow10(re, im))
        .collect()
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
    if let Some(first) = data.first_mut() {
        *first = (start_re, start_im);
    }
    if let Some(last) = data.last_mut() {
        *last = (stop_re, stop_im);
    }
    data
}

fn complex_pow10(re: f64, im: f64) -> (f64, f64) {
    // 10^(re + i*im) = exp((re + i*im) * ln(10))
    let scaled_re = re * LN_10;
    let scaled_im = im * LN_10;
    let mag = scaled_re.exp();
    let cos = scaled_im.cos();
    let sin = scaled_im.sin();
    (mag * cos, mag * sin)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn count_parser_preserves_representable_uint64() {
        assert_eq!(
            futures::executor::block_on(parse_count(&Value::Int(IntValue::U64(u64::MAX)))).ok(),
            usize::try_from(u64::MAX).ok()
        );
        assert_eq!(
            futures::executor::block_on(parse_count(&Value::Int(IntValue::I64(-1)))).unwrap(),
            0
        );
        assert!(futures::executor::block_on(parse_count(&Value::Num(usize::MAX as f64))).is_err());
        assert!(
            futures::executor::block_on(parse_count(&Value::Num((usize::MAX as f64) + 1.0)))
                .is_err()
        );
    }
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};
    use runmat_value::{IntValue, IntegerComplexStorage, IntegerStorage, Tensor};

    fn logspace_builtin(
        start: Value,
        stop: Value,
        rest: Vec<Value>,
    ) -> crate::BuiltinResult<Value> {
        block_on(super::logspace_builtin(start, stop, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logspace_default_points() {
        let result =
            logspace_builtin(Value::Num(1.0), Value::Num(3.0), Vec::new()).expect("logspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 50]);
                assert!((t.materialize_f64()[0] - 10.0).abs() < 1e-12);
                assert!((t.materialize_f64()[49] - 1000.0).abs() < 1e-9);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn logspace_count_parser_preserves_typed_integer_tensors_exactly() {
        let large = 9_007_199_254_740_993_u64;
        let count = Tensor::new_integer(IntegerStorage::U64(vec![large]), vec![1, 1]).unwrap();

        assert_eq!(
            block_on(parse_count(&Value::Tensor(count))).unwrap(),
            large as usize
        );
    }

    #[test]
    fn logspace_count_parser_ignores_poisoned_f64_mirrors_for_all_integer_classes() {
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
            assert_eq!(block_on(parse_count(&Value::Tensor(count))).unwrap(), 2);
        }
    }

    #[test]
    fn logspace_count_parser_maps_negative_typed_integer_tensors_to_empty() {
        let count = Tensor::new_integer(IntegerStorage::I64(vec![-1]), vec![1, 1]).unwrap();

        assert_eq!(block_on(parse_count(&Value::Tensor(count))).unwrap(), 0);
    }

    #[test]
    fn logspace_rejects_real_integer_tensor_endpoints() {
        let start =
            Tensor::new_integer(IntegerStorage::I64(vec![0]), vec![1, 1]).expect("start tensor");
        let stop =
            Tensor::new_integer(IntegerStorage::U64(vec![2]), vec![1, 1]).expect("stop tensor");

        let error = logspace_builtin(
            Value::Tensor(start),
            Value::Tensor(stop),
            vec![Value::Int(IntValue::I32(3))],
        )
        .expect_err("integer endpoints must be rejected");
        assert!(error.message().contains("single or double"));
    }

    #[test]
    fn logspace_type_is_row_vector() {
        assert_eq!(
            logspace_type(&[Type::Num, Type::Num], &ResolveContext::new(Vec::new())),
            Type::Tensor {
                shape: Some(vec![Some(1), None])
            }
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logspace_custom_points() {
        let result = logspace_builtin(
            Value::Num(0.0),
            Value::Num(2.0),
            vec![Value::Int(IntValue::I32(5))],
        )
        .expect("logspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 5]);
                let expected = [1.0, 3.1622776601683795, 10.0, 31.622776601683793, 100.0];
                for (a, b) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((a - b).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn logspace_pi_endpoint_returns_pi_not_ten_to_pi() {
        let result = logspace_builtin(
            Value::Num(0.0),
            Value::Num(std::f64::consts::PI),
            vec![Value::Num(2.0)],
        )
        .expect("logspace pi endpoint");
        let Value::Tensor(tensor) = result else {
            panic!("expected host tensor");
        };
        assert_eq!(tensor.shape, vec![1, 2]);
        assert!((tensor.materialize_f64()[0] - 1.0).abs() < 1e-12);
        assert!((tensor.materialize_f64()[1] - std::f64::consts::PI).abs() < 1e-12);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logspace_zero_points() {
        let result = logspace_builtin(
            Value::Num(1.0),
            Value::Num(3.0),
            vec![Value::Int(IntValue::I32(0))],
        )
        .expect("logspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 0]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logspace_rejects_logical_count() {
        let error = logspace_builtin(Value::Num(0.0), Value::Num(1.0), vec![Value::Bool(false)])
            .expect_err("logical count must be rejected");
        assert!(error.message().contains("numeric"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logspace_single_point() {
        let result = logspace_builtin(
            Value::Num(-2.0),
            Value::Num(0.0),
            vec![Value::Int(IntValue::I32(1))],
        )
        .expect("logspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 1]);
                assert!((t.materialize_f64()[0] - 1.0).abs() < 1e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logspace_complex_points() {
        let result = logspace_builtin(
            Value::Complex(0.0, 1.0),
            Value::Complex(0.0, 2.0),
            vec![Value::Int(IntValue::I32(4))],
        )
        .expect("logspace");
        match result {
            Value::ComplexTensor(t) => {
                assert_eq!(t.shape, vec![1, 4]);
                assert_eq!(t.materialize_f64().len(), 4);
                let expected = generate_complex_log_sequence(0.0, 1.0, 0.0, 2.0, 4);
                for (actual, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual.0 - exp.0).abs() < 1e-12);
                    assert!((actual.1 - exp.1).abs() < 1e-12);
                }
            }
            other => panic!("expected complex tensor result, got {other:?}"),
        }
    }

    #[test]
    fn logspace_rejects_complex_integer_tensor_endpoints() {
        let start_storage =
            IntegerComplexStorage::new(IntegerStorage::I16(vec![0]), IntegerStorage::I16(vec![1]))
                .expect("start storage");
        let start = ComplexTensor::new_integer(start_storage, vec![1, 1]).expect("start tensor");
        let stop_storage =
            IntegerComplexStorage::new(IntegerStorage::I16(vec![0]), IntegerStorage::I16(vec![2]))
                .expect("stop storage");
        let stop = ComplexTensor::new_integer(stop_storage, vec![1, 1]).expect("stop tensor");

        let error = logspace_builtin(
            Value::ComplexTensor(start),
            Value::ComplexTensor(stop),
            vec![Value::Int(IntValue::I32(4))],
        )
        .expect_err("integer endpoints must be rejected");
        assert!(error.message().contains("single or double"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logspace_tensor_scalar_inputs() {
        let start = Tensor::new(vec![2.0], vec![1, 1]).unwrap();
        let stop = Tensor::new(vec![3.0], vec![1, 1]).unwrap();
        let result = logspace_builtin(
            Value::Tensor(start),
            Value::Tensor(stop),
            vec![Value::Int(IntValue::I32(3))],
        )
        .expect("logspace");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape, vec![1, 3]);
                let expected = generate_real_log_sequence(2.0, 3.0, 3);
                for (actual, exp) in t.materialize_f64().iter().zip(expected.iter()) {
                    assert!((actual - exp).abs() < 1e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn logspace_preserves_single_real_and_complex_endpoint_class() {
        let start = Tensor::from_f32(vec![0.0], vec![1, 1]).unwrap();
        let stop = Tensor::from_f32(vec![2.0], vec![1, 1]).unwrap();
        let result = logspace_builtin(
            Value::Tensor(start),
            Value::Tensor(stop),
            vec![Value::Num(3.9)],
        )
        .unwrap();
        let Value::Tensor(result) = result else {
            panic!("expected real single tensor")
        };
        assert_eq!(result.numeric_dtype(), NumericDType::F32);
        assert_eq!(result.as_f32_slice(), Some(&[1.0, 10.0, 100.0][..]));

        let start = ComplexTensor::from_f32(vec![(0.0, 0.0)], vec![1, 1]).unwrap();
        let stop = ComplexTensor::from_f32(vec![(1.0, 0.0)], vec![1, 1]).unwrap();
        let result = logspace_builtin(
            Value::ComplexTensor(start),
            Value::ComplexTensor(stop),
            vec![Value::Int(IntValue::I32(2))],
        )
        .unwrap();
        let Value::ComplexTensor(result) = result else {
            panic!("expected complex single tensor")
        };
        assert_eq!(result.numeric_dtype(), NumericDType::F32);
        assert!(result.as_f32_slice().is_some());
    }

    #[test]
    fn logspace_rejects_typed_integer_tensor_endpoints() {
        let start = Tensor::new_integer(IntegerStorage::I16(vec![-1]), vec![1, 1]).expect("start");
        let stop = Tensor::new_integer(IntegerStorage::U16(vec![1]), vec![1, 1]).expect("stop");

        let error = logspace_builtin(
            Value::Tensor(start),
            Value::Tensor(stop),
            vec![Value::Int(IntValue::I32(3))],
        )
        .expect_err("integer endpoints must be rejected");
        assert!(error.message().contains("single or double"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logspace_fractional_count_rounds_down() {
        let result = logspace_builtin(Value::Num(1.0), Value::Num(2.0), vec![Value::Num(3.5)])
            .expect("fractional counts floor");
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor")
        };
        assert_eq!(tensor.shape, vec![1, 3]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logspace_negative_count_returns_empty() {
        let result = logspace_builtin(
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::Int(IntValue::I32(-1))],
        )
        .expect("negative count");
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor")
        };
        assert_eq!(tensor.shape, vec![1, 0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logspace_rejects_infinite_count() {
        let err = logspace_builtin(
            Value::Num(0.0),
            Value::Num(1.0),
            vec![Value::Num(f64::INFINITY)],
        )
        .expect_err("expected error");
        assert!(err.message().contains("finite"));
    }

    #[test]
    fn logspace_rejects_nan_count() {
        let error = logspace_builtin(Value::Num(0.0), Value::Num(1.0), vec![Value::Num(f64::NAN)])
            .expect_err("NaN is not a documented logspace count");
        assert!(error.message().contains("finite"));
    }

    #[test]
    fn logspace_rejects_scalar_integer_and_logical_endpoints() {
        for endpoint in [Value::Int(IntValue::I8(0)), Value::Bool(false)] {
            let error = logspace_builtin(
                endpoint,
                Value::Num(1.0),
                vec![Value::Int(IntValue::I32(3))],
            )
            .expect_err("nonfloating endpoints must be rejected");
            assert!(error.message().contains("single or double"));
        }
    }

    #[test]
    fn logspace_rejects_resident_integer_endpoints_from_metadata() {
        test_support::with_test_provider(|provider| {
            let endpoint = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::I64(&[9_007_199_254_740_993]),
                    shape: &[1, 1],
                })
                .expect("integer endpoint upload");
            let error = logspace_builtin(
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
    fn logspace_rejects_non_scalar_inputs() {
        let start = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let err =
            logspace_builtin(Value::Tensor(start), Value::Num(1.0), Vec::new()).expect_err("error");
        assert!(err.message().contains("scalar"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn logspace_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let lo = Value::GpuTensor(
                provider
                    .upload(&HostTensorView {
                        data: &[1.0],
                        shape: &[1, 1],
                    })
                    .expect("upload"),
            );
            let hi = Value::GpuTensor(
                provider
                    .upload(&HostTensorView {
                        data: &[3.0],
                        shape: &[1, 1],
                    })
                    .expect("upload"),
            );
            let result =
                logspace_builtin(lo, hi, vec![Value::Int(IntValue::I32(3))]).expect("logspace");
            let gathered = test_support::gather(result).expect("gather");
            assert_eq!(gathered.shape, vec![1, 3]);
            let expected = [10.0, 100.0, 1000.0];
            for (a, b) in gathered.materialize_f64().iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-6);
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn logspace_wgpu_matches_cpu() {
        if runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        )
        .is_err()
        {
            return;
        }
        let cpu = logspace_builtin(
            Value::Num(1.0),
            Value::Num(2.0),
            vec![Value::Int(IntValue::I32(5))],
        )
        .expect("cpu");
        let gpu = {
            let view = HostTensorView {
                data: &[1.0],
                shape: &[1, 1],
            };
            let lo = Value::GpuTensor(
                runmat_accelerate_api::provider()
                    .unwrap()
                    .upload(&view)
                    .expect("upload lo"),
            );
            let hi = Value::GpuTensor(
                runmat_accelerate_api::provider()
                    .unwrap()
                    .upload(&HostTensorView {
                        data: &[2.0],
                        shape: &[1, 1],
                    })
                    .expect("upload hi"),
            );
            logspace_builtin(lo, hi, vec![Value::Int(IntValue::I32(5))]).expect("gpu")
        };
        let gathered = test_support::gather(gpu).expect("gather");
        match (cpu, gathered) {
            (Value::Tensor(ct), gt) => {
                assert_eq!(ct.shape, gt.shape);
                let (absolute_tolerance, relative_tolerance) =
                    test_support::gpu_transcendental_tolerances(
                        runmat_accelerate_api::provider().unwrap().precision(),
                    );
                for (index, (actual, expected)) in ct
                    .materialize_f64()
                    .iter()
                    .zip(gt.materialize_f64().iter())
                    .enumerate()
                {
                    assert!(
                        test_support::floats_match(
                            *actual,
                            *expected,
                            absolute_tolerance,
                            relative_tolerance,
                        ),
                        "logspace mismatch at {index}: cpu={actual} gpu={expected}"
                    );
                }
            }
            _ => panic!("unexpected value variants"),
        }
    }
}
