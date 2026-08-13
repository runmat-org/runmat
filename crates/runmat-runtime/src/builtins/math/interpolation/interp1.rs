//! MATLAB-compatible `interp1` builtin for dense real numeric data.

use runmat_accelerate_api::{
    GpuTensorHandle, GpuTensorStorage, HostTensorView, ProviderInterp1Extrapolation,
    ProviderInterp1Method, ProviderInterp1Request,
};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ResolveContext, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};

use super::pp::{
    build_pchip_pp, build_spline_pp, evaluate_linear_or_nearest, evaluate_pp,
    implicit_series_from_values, is_vector_shape, parse_extrapolation, parse_method, query_points,
    series_from_values, validate_breaks, vector_from_value, Extrapolation, InterpMethod,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, RuntimeError};

const NAME: &str = "interp1";

const INTERP1_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Vq",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Interpolated values at query points.",
}];

const INTERP1_INPUTS_Y_XQ: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample values at implicit X = 1:numel(Y).",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query points.",
    },
];

const INTERP1_INPUTS_X_Y_XQ: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample locations.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample values.",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query points.",
    },
];

const INTERP1_INPUTS_Y_XQ_METHOD: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample values at implicit X = 1:numel(Y).",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query points.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"linear\""),
        description: "Interpolation method: \"linear\", \"nearest\", \"spline\", or \"pchip\".",
    },
];

const INTERP1_INPUTS_Y_XQ_METHOD_EXTRAP: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample values at implicit X = 1:numel(Y).",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query points.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"linear\""),
        description: "Interpolation method: \"linear\", \"nearest\", \"spline\", or \"pchip\".",
    },
    BuiltinParamDescriptor {
        name: "extrap",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("NaN"),
        description: "Extrapolation mode: \"extrap\" or scalar fill value.",
    },
];

const INTERP1_INPUTS_X_Y_XQ_METHOD: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample locations.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample values.",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query points.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"linear\""),
        description: "Interpolation method: \"linear\", \"nearest\", \"spline\", or \"pchip\".",
    },
];

const INTERP1_INPUTS_X_Y_XQ_METHOD_EXTRAP: [BuiltinParamDescriptor; 5] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample locations.",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Sample values.",
    },
    BuiltinParamDescriptor {
        name: "Xq",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Query points.",
    },
    BuiltinParamDescriptor {
        name: "method",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Optional,
        default: Some("\"linear\""),
        description: "Interpolation method: \"linear\", \"nearest\", \"spline\", or \"pchip\".",
    },
    BuiltinParamDescriptor {
        name: "extrap",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Optional,
        default: Some("NaN"),
        description: "Extrapolation mode: \"extrap\" or scalar fill value.",
    },
];

const INTERP1_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "Vq = interp1(Y, Xq)",
        inputs: &INTERP1_INPUTS_Y_XQ,
        outputs: &INTERP1_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Vq = interp1(X, Y, Xq)",
        inputs: &INTERP1_INPUTS_X_Y_XQ,
        outputs: &INTERP1_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Vq = interp1(Y, Xq, method)",
        inputs: &INTERP1_INPUTS_Y_XQ_METHOD,
        outputs: &INTERP1_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Vq = interp1(X, Y, Xq, method)",
        inputs: &INTERP1_INPUTS_X_Y_XQ_METHOD,
        outputs: &INTERP1_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Vq = interp1(Y, Xq, extrap)",
        inputs: &INTERP1_INPUTS_Y_XQ_METHOD,
        outputs: &INTERP1_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Vq = interp1(X, Y, Xq, extrap)",
        inputs: &INTERP1_INPUTS_X_Y_XQ_METHOD,
        outputs: &INTERP1_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Vq = interp1(Y, Xq, method, extrap)",
        inputs: &INTERP1_INPUTS_Y_XQ_METHOD_EXTRAP,
        outputs: &INTERP1_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "Vq = interp1(X, Y, Xq, method, extrap)",
        inputs: &INTERP1_INPUTS_X_Y_XQ_METHOD_EXTRAP,
        outputs: &INTERP1_OUTPUT,
    },
];

const INTERP1_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERP1.INVALID_ARGUMENT",
    identifier: Some("RunMat:interp1:InvalidArgument"),
    when: "Argument count, method/extrapolation options, or shape constraints are invalid.",
    message: "interp1: invalid argument",
};

const INTERP1_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERP1.INVALID_INPUT",
    identifier: Some("RunMat:interp1:InvalidInput"),
    when: "Sample or query values cannot be converted to numeric interpolation domains.",
    message: "interp1: invalid input",
};

const INTERP1_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.INTERP1.INTERNAL",
    identifier: Some("RunMat:interp1:Internal"),
    when: "Interpolation evaluation fails due to internal tensor construction or solver paths.",
    message: "interp1: internal interpolation failure",
};

const INTERP1_ERRORS: [BuiltinErrorDescriptor; 3] = [
    INTERP1_ERROR_INVALID_ARGUMENT,
    INTERP1_ERROR_INVALID_INPUT,
    INTERP1_ERROR_INTERNAL,
];

pub const INTERP1_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &INTERP1_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &INTERP1_ERRORS,
};

const INTERP1_INTEGER_SAMPLE_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "interp1-integer-sample",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "interp1 with typed-integer sample locations or values is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Interp1IntegerSampleExtension"),
};
const INTERP1_INTEGER_QUERY_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "interp1-integer-query",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "interp1 with typed-integer query coordinates is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Interp1IntegerQueryExtension"),
};
const INTERP1_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "interp1-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "interp1 with logical numeric input is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Interp1LogicalInputExtension"),
};
const INTERP1_INTEGER_EXTRAPOLATION_EXTENSION: BuiltinExtensionDescriptor =
    BuiltinExtensionDescriptor {
        id: "interp1-integer-extrapolation",
        mode: BuiltinExtensionMode::RunMatOnly,
        description: "interp1 with a typed-integer extrapolation value is a RunMat extension",
        error_identifier: Some("RunMat:compatibility:Interp1IntegerExtrapolationExtension"),
    };
pub const INTERP1_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    INTERP1_INTEGER_SAMPLE_EXTENSION,
    INTERP1_INTEGER_QUERY_EXTENSION,
    INTERP1_LOGICAL_INPUT_EXTENSION,
    INTERP1_INTEGER_EXTRAPOLATION_EXTENSION,
];
const INTERP1_INTEGER_SAMPLE_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "X or Y",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes:
            "All values must be exactly representable before the binary64 interpolation boundary.",
    }];
const INTERP1_INTEGER_QUERY_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "Xq",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "All coordinates must be exactly representable before the binary64 interpolation boundary.",
    }];
const INTERP1_INTEGER_EXTRAPOLATION_INPUTS: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "extrapolation value",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The scalar fill value must be exactly representable before interpolation.",
    }];
pub const INTERP1_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "Vq = interp1(integer_X_or_Y, ..., Xq)",
        inputs: &INTERP1_INTEGER_SAMPLE_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat retains typed-integer samples as a checked extension; MATLAB-compatible modes retain the documented floating and temporal surface.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Vq = interp1(X, Y, integer_Xq)",
        inputs: &INTERP1_INTEGER_QUERY_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "Integer queries are classified before provider dispatch or gather and then cross one checked binary64 boundary.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "Vq = interp1(..., method, integer_extrapolation_value)",
        inputs: &INTERP1_INTEGER_EXTRAPOLATION_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::Error,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "RunMat accepts this broader scalar fill form only after compatibility and exactness checks that precede provider access or gather.",
    },
];

fn interp1_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn interp1_invalid_argument(detail: impl AsRef<str>) -> RuntimeError {
    interp1_error_with_message(
        format!(
            "{}: {}",
            INTERP1_ERROR_INVALID_ARGUMENT.message,
            detail.as_ref()
        ),
        &INTERP1_ERROR_INVALID_ARGUMENT,
    )
}

fn interp1_map_error(err: RuntimeError, fallback: &'static BuiltinErrorDescriptor) -> RuntimeError {
    if err.identifier().is_some() {
        err
    } else {
        interp1_error_with_message(err.message().to_string(), fallback)
    }
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::interpolation::interp1")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: NAME,
    op_kind: GpuOpKind::Custom("interpolation-1d"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::Matlab,
    provider_hooks: &[ProviderHook::Custom("interp1")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Linear and nearest interpolation keep resident real Y/query inputs on the provider when X is implicit or host-validated; spline/pchip and resident-X validation fall back to host.",
};

#[runmat_macros::register_fusion_spec(
    builtin_path = "crate::builtins::math::interpolation::interp1"
)]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: NAME,
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Interpolation is currently a runtime sink.",
};

fn interp1_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    let query = match args.len() {
        0 | 1 => return Type::tensor(),
        2 => args.get(1),
        _ => args.get(2),
    };
    match query {
        Some(Type::Num | Type::Int | Type::Bool) => Type::Num,
        Some(Type::Tensor { shape }) | Some(Type::Logical { shape }) => Type::Tensor {
            shape: shape.clone(),
        },
        _ => Type::tensor(),
    }
}

#[runtime_builtin(
    name = "interp1",
    category = "math/interpolation",
    summary = "Interpolate one-dimensional sampled data.",
    keywords = "interp1,interpolation,linear,nearest,spline,pchip",
    accel = "sink",
    sink = true,
    type_resolver(interp1_type),
    descriptor(crate::builtins::math::interpolation::interp1::INTERP1_DESCRIPTOR),
    extensions(crate::builtins::math::interpolation::interp1::INTERP1_EXTENSIONS),
    integer_capabilities(
        crate::builtins::math::interpolation::interp1::INTERP1_INTEGER_CAPABILITIES
    ),
    builtin_path = "crate::builtins::math::interpolation::interp1"
)]
async fn interp1_builtin(args: Vec<Value>) -> crate::BuiltinResult<Value> {
    preflight_interp1_inputs(&args).await?;
    if let Some(output) = try_interp1_gpu(&args).await? {
        return Ok(output);
    }
    let parsed = ParsedInterp1::parse(args)
        .await
        .map_err(|err| interp1_map_error(err, &INTERP1_ERROR_INVALID_INPUT))?;
    match parsed.method {
        InterpMethod::Linear | InterpMethod::Nearest => evaluate_linear_or_nearest(
            &parsed.series,
            &parsed.query,
            parsed.method,
            &parsed.extrap,
            NAME,
        )
        .map_err(|err| interp1_map_error(err, &INTERP1_ERROR_INTERNAL)),
        InterpMethod::Spline => {
            let pp = build_spline_pp(&parsed.series, NAME)
                .map_err(|err| interp1_map_error(err, &INTERP1_ERROR_INTERNAL))?;
            evaluate_pp(&pp, &parsed.query, &parsed.extrap_for_cubic(), NAME)
                .map_err(|err| interp1_map_error(err, &INTERP1_ERROR_INTERNAL))
        }
        InterpMethod::Pchip => {
            let pp = build_pchip_pp(&parsed.series, NAME)
                .map_err(|err| interp1_map_error(err, &INTERP1_ERROR_INTERNAL))?;
            evaluate_pp(&pp, &parsed.query, &parsed.extrap_for_cubic(), NAME)
                .map_err(|err| interp1_map_error(err, &INTERP1_ERROR_INTERNAL))
        }
    }
}

async fn preflight_interp1_inputs(args: &[Value]) -> crate::BuiltinResult<()> {
    use crate::builtins::common::validation::{
        native_integer_value_is_exact_f64_async, value_has_logical_class,
        value_has_native_integer_class,
    };
    if args.len() < 2 {
        return Ok(());
    }
    let explicit = args.len() >= 3 && !third_arg_is_option(args);
    let (samples, query) = if explicit {
        (&args[..2], &args[2])
    } else {
        (&args[..1], &args[1])
    };
    for value in samples {
        if value_has_native_integer_class(value) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &INTERP1_INTEGER_SAMPLE_EXTENSION,
                NAME,
            )?;
            if !native_integer_value_is_exact_f64_async(value).await? {
                return Err(interp1_error_with_message(
                    "interp1: integer samples must be exactly representable as double",
                    &INTERP1_ERROR_INVALID_INPUT,
                ));
            }
        }
    }
    if value_has_native_integer_class(query) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTERP1_INTEGER_QUERY_EXTENSION,
            NAME,
        )?;
        if !native_integer_value_is_exact_f64_async(query).await? {
            return Err(interp1_error_with_message(
                "interp1: integer query coordinates must be exactly representable as double",
                &INTERP1_ERROR_INVALID_INPUT,
            ));
        }
    }
    if samples
        .iter()
        .chain(std::iter::once(query))
        .any(value_has_logical_class)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &INTERP1_LOGICAL_INPUT_EXTENSION,
            NAME,
        )?;
    }
    let option_start = if explicit { 3 } else { 2 };
    for option in args.iter().skip(option_start) {
        if value_has_native_integer_class(option) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &INTERP1_INTEGER_EXTRAPOLATION_EXTENSION,
                NAME,
            )?;
            if !native_integer_value_is_exact_f64_async(option).await? {
                return Err(interp1_error_with_message(
                    "interp1: integer extrapolation value must be exactly representable as double",
                    &INTERP1_ERROR_INVALID_INPUT,
                ));
            }
        }
        if value_has_logical_class(option) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &INTERP1_LOGICAL_INPUT_EXTENSION,
                NAME,
            )?;
        }
    }
    Ok(())
}

struct GpuInterp1Series<'a> {
    x: Vec<f64>,
    y: &'a GpuTensorHandle,
    sample_len: usize,
    series_count: usize,
    trailing_shape: Vec<usize>,
}

struct GpuInterp1Query {
    handle: GpuTensorHandle,
    shape: Vec<usize>,
    len: usize,
    owned: bool,
}

async fn try_interp1_gpu(args: &[Value]) -> crate::BuiltinResult<Option<Value>> {
    if args.len() < 2 {
        return Ok(None);
    }

    let mut method = InterpMethod::Linear;
    let mut extrap = Extrapolation::Nan;
    let (series, query_value, options) = if args.len() == 2 || third_arg_is_option(args) {
        let Some(series) = implicit_gpu_series(&args[0])? else {
            return Ok(None);
        };
        (series, &args[1], &args[2..])
    } else {
        let Some(series) = explicit_gpu_series(&args[0], &args[1]).await? else {
            return Ok(None);
        };
        (series, &args[2], &args[3..])
    };

    for option in options {
        if let Some(parsed) = parse_extrapolation(option, NAME).await? {
            extrap = parsed;
            continue;
        }
        if let Some(parsed) = parse_method(option, NAME)? {
            method = parsed;
            continue;
        }
        return Ok(None);
    }
    if !matches!(method, InterpMethod::Linear | InterpMethod::Nearest) {
        return Ok(None);
    }

    let provider = gpu_helpers::exact_provider_for_handle(series.y).ok_or_else(|| {
        interp1_error_with_message(
            "interp1: no acceleration provider owns the resident Y input",
            &INTERP1_ERROR_INTERNAL,
        )
    })?;
    let series_shape = series.y.shape.clone();
    let series_metadata = gpu_helpers::snapshot_handle_metadata(series.y);
    let resident_query_metadata = match query_value {
        Value::GpuTensor(handle) => Some((handle, gpu_helpers::snapshot_handle_metadata(handle))),
        _ => None,
    };
    let series_precision = runmat_accelerate_api::handle_precision(series.y);
    if series_precision != Some(provider.precision())
        || !valid_interp1_gpu_input(series.y, &series_shape, series_precision, provider, &[])
    {
        return Err(interp1_error_with_message(
            "interp1: resident Y has contradictory floating metadata",
            &INTERP1_ERROR_INTERNAL,
        ));
    }
    let mut provenance = runmat_accelerate_api::handle_provenance(series.y)
        .unwrap_or(runmat_accelerate_api::GpuHandleProvenance::Automatic);
    let x_shape = [1, series.sample_len];
    let x_upload = provider.upload(&HostTensorView {
        data: &series.x,
        shape: &x_shape,
    });
    gpu_helpers::restore_handle_metadata(series.y, &series_metadata);
    if let Some((handle, metadata)) = resident_query_metadata.as_ref() {
        gpu_helpers::restore_handle_metadata(handle, metadata);
    }
    let x_handle = match x_upload {
        Ok(handle) => handle,
        Err(error) => {
            return Err(interp1_error_with_message(
                format!("interp1: X upload failed: {error}"),
                &INTERP1_ERROR_INTERNAL,
            ));
        }
    };
    let mut x_protected = vec![series.y];
    if let Some((handle, _)) = resident_query_metadata.as_ref() {
        x_protected.push(handle);
    }
    if !valid_interp1_gpu_input(series.y, &series_shape, series_precision, provider, &[])
        || !valid_interp1_gpu_input(
            &x_handle,
            &x_shape,
            series_precision,
            provider,
            &x_protected,
        )
    {
        gpu_helpers::free_unprotected_exact_owner(&x_handle, &x_protected);
        return Err(interp1_error_with_message(
            "interp1: provider returned an invalid X upload",
            &INTERP1_ERROR_INTERNAL,
        ));
    }
    let x_metadata = gpu_helpers::snapshot_handle_metadata(&x_handle);

    let query_result = gpu_query_points(query_value, provider).await;
    gpu_helpers::restore_handle_metadata(series.y, &series_metadata);
    gpu_helpers::restore_handle_metadata(&x_handle, &x_metadata);
    if let Some((handle, metadata)) = resident_query_metadata.as_ref() {
        gpu_helpers::restore_handle_metadata(handle, metadata);
    }
    let query = match query_result {
        Ok(Some(query)) => query,
        Ok(None) => {
            gpu_helpers::free_unprotected_exact_owner(&x_handle, &x_protected);
            return Ok(None);
        }
        Err(err) => {
            gpu_helpers::free_unprotected_exact_owner(&x_handle, &x_protected);
            return Err(err);
        }
    };
    let owned_query_protected = [series.y, &x_handle];
    let borrowed_query_protected = [&x_handle];
    let query_protected: &[&GpuTensorHandle] = if query.owned {
        &owned_query_protected
    } else {
        &borrowed_query_protected
    };
    if !valid_interp1_gpu_input(series.y, &series_shape, series_precision, provider, &[])
        || !valid_interp1_gpu_input(&x_handle, &x_shape, series_precision, provider, &[series.y])
        || !valid_interp1_gpu_input(
            &query.handle,
            &query.shape,
            series_precision,
            provider,
            query_protected,
        )
    {
        if query.owned && gpu_helpers::same_gpu_handle(&query.handle, &x_handle) {
            gpu_helpers::free_unprotected_exact_owner(&x_handle, &[series.y]);
        } else {
            if query.owned {
                gpu_helpers::free_unprotected_exact_owner(&query.handle, &[series.y, &x_handle]);
            }
            gpu_helpers::free_unprotected_exact_owner(&x_handle, &[series.y, &query.handle]);
        }
        return Err(interp1_error_with_message(
            "interp1: invalid resident Xq input",
            &INTERP1_ERROR_INTERNAL,
        ));
    }
    let query_metadata = gpu_helpers::snapshot_handle_metadata(&query.handle);
    if !query.owned
        && runmat_accelerate_api::handle_provenance(&query.handle)
            == Some(runmat_accelerate_api::GpuHandleProvenance::Explicit)
    {
        provenance = runmat_accelerate_api::GpuHandleProvenance::Explicit;
    }

    let output_shape =
        interp1_gpu_output_shape(&query.shape, series.series_count, &series.trailing_shape);
    let request = ProviderInterp1Request {
        x: &x_handle,
        y: series.y,
        xq: &query.handle,
        sample_len: series.sample_len,
        series_count: series.series_count,
        query_len: query.len,
        output_shape: &output_shape,
        method: provider_interp1_method(method),
        extrapolation: provider_interp1_extrapolation(&extrap),
        extrapolation_value: match extrap {
            Extrapolation::Value(value) => value,
            _ => f64::NAN,
        },
    };
    let result = provider.interp1(&request).await;
    gpu_helpers::restore_handle_metadata(series.y, &series_metadata);
    gpu_helpers::restore_handle_metadata(&x_handle, &x_metadata);
    gpu_helpers::restore_handle_metadata(&query.handle, &query_metadata);
    let valid = result.as_ref().is_ok_and(|output| {
        valid_interp1_gpu_output(
            output,
            &output_shape,
            series.y,
            &x_handle,
            &query.handle,
            provider,
            series_precision,
        )
    });
    gpu_helpers::free_unprotected_exact_owner(&x_handle, &[series.y, &query.handle]);
    if query.owned {
        gpu_helpers::free_unprotected_exact_owner(&query.handle, &[series.y, &x_handle]);
    }

    match result {
        Ok(output) if valid => {
            runmat_accelerate_api::set_handle_provenance(&output, provenance);
            Ok(Some(gpu_helpers::resident_gpu_value(output)))
        }
        Ok(output) => {
            gpu_helpers::free_unprotected_exact_owner(
                &output,
                &[series.y, &x_handle, &query.handle],
            );
            Err(interp1_error_with_message(
                "interp1: provider returned an invalid interpolation result",
                &INTERP1_ERROR_INTERNAL,
            ))
        }
        Err(error) if error.to_string() == "interp1 not supported by provider" => Ok(None),
        Err(error) => Err(build_runtime_error(format!(
            "interp1: provider execution failed: {error}"
        ))
        .with_builtin(NAME)
        .with_identifier("RunMat:interp1:Internal")
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()),
    }
}

fn valid_interp1_gpu_output(
    output: &GpuTensorHandle,
    output_shape: &[usize],
    y: &GpuTensorHandle,
    x: &GpuTensorHandle,
    xq: &GpuTensorHandle,
    provider: &dyn runmat_accelerate_api::AccelProvider,
    expected_precision: Option<runmat_accelerate_api::ProviderPrecision>,
) -> bool {
    output.shape == output_shape
        && output.device_id == y.device_id
        && ![y, x, xq]
            .iter()
            .any(|input| gpu_helpers::same_gpu_handle(input, output))
        && gpu_helpers::exact_provider_for_handle(output)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
        && runmat_accelerate_api::handle_storage(output) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(output) == expected_precision
        && runmat_accelerate_api::handle_integer_type(output).is_none()
        && !runmat_accelerate_api::handle_is_logical(output)
        && gpu_helpers::gpu_class_metadata_matches(output, expected_precision, None, false)
}

fn valid_interp1_gpu_input(
    handle: &GpuTensorHandle,
    expected_shape: &[usize],
    expected_precision: Option<runmat_accelerate_api::ProviderPrecision>,
    provider: &dyn runmat_accelerate_api::AccelProvider,
    protected: &[&GpuTensorHandle],
) -> bool {
    handle.shape == expected_shape
        && handle.device_id == provider.device_id()
        && !protected
            .iter()
            .any(|input| gpu_helpers::same_gpu_handle(input, handle))
        && gpu_helpers::exact_provider_for_handle(handle)
            .is_some_and(|owner| std::ptr::eq(owner, provider))
        && runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(handle) == expected_precision
        && runmat_accelerate_api::handle_integer_type(handle).is_none()
        && !runmat_accelerate_api::handle_is_logical(handle)
        && gpu_helpers::gpu_class_metadata_matches(handle, expected_precision, None, false)
}

fn implicit_gpu_series(value: &Value) -> crate::BuiltinResult<Option<GpuInterp1Series<'_>>> {
    let Value::GpuTensor(handle) = value else {
        return Ok(None);
    };
    if runmat_accelerate_api::handle_storage(handle) != GpuTensorStorage::Real
        || runmat_accelerate_api::handle_integer_type(handle).is_some()
        || runmat_accelerate_api::handle_is_logical(handle)
    {
        return Ok(None);
    }
    let Some((sample_len, series_count, trailing_shape)) = gpu_y_layout(handle, None)? else {
        return Ok(None);
    };
    let x = (1..=sample_len).map(|value| value as f64).collect();
    Ok(Some(GpuInterp1Series {
        x,
        y: handle,
        sample_len,
        series_count,
        trailing_shape,
    }))
}

async fn explicit_gpu_series<'a>(
    x_value: &Value,
    y_value: &'a Value,
) -> crate::BuiltinResult<Option<GpuInterp1Series<'a>>> {
    let Value::GpuTensor(handle) = y_value else {
        return Ok(None);
    };
    if matches!(x_value, Value::GpuTensor(_))
        || runmat_accelerate_api::handle_storage(handle) != GpuTensorStorage::Real
        || runmat_accelerate_api::handle_integer_type(handle).is_some()
        || runmat_accelerate_api::handle_is_logical(handle)
    {
        return Ok(None);
    }
    let x = vector_from_value(x_value.clone(), "X", NAME).await?;
    validate_breaks(&x, NAME)?;
    let Some((sample_len, series_count, trailing_shape)) = gpu_y_layout(handle, Some(x.len()))?
    else {
        return Ok(None);
    };
    Ok(Some(GpuInterp1Series {
        x,
        y: handle,
        sample_len,
        series_count,
        trailing_shape,
    }))
}

fn gpu_y_layout(
    handle: &GpuTensorHandle,
    explicit_sample_len: Option<usize>,
) -> crate::BuiltinResult<Option<(usize, usize, Vec<usize>)>> {
    let len = handle
        .shape
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
        .ok_or_else(|| {
            interp1_error_with_message("interp1: Y size overflow", &INTERP1_ERROR_INVALID_INPUT)
        })?;
    let shape = tensor::default_shape_for(&handle.shape, len);
    let sample_len = explicit_sample_len
        .unwrap_or_else(|| shape.iter().copied().find(|dim| *dim > 1).unwrap_or(len));
    if sample_len < 2 {
        return Ok(None);
    }
    if len != sample_len && !len.is_multiple_of(sample_len) {
        return Ok(None);
    }
    let (series_count, trailing_shape) = if len == sample_len && is_vector_shape(&shape) {
        (1, Vec::new())
    } else if shape.first().copied() == Some(sample_len) {
        let series = len / sample_len;
        let trailing = if shape.len() > 1 {
            shape[1..].to_vec()
        } else {
            vec![series]
        };
        (series, trailing)
    } else if len == sample_len {
        (1, Vec::new())
    } else {
        return Ok(None);
    };
    Ok(Some((sample_len, series_count, trailing_shape)))
}

async fn gpu_query_points(
    value: &Value,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
) -> crate::BuiltinResult<Option<GpuInterp1Query>> {
    match value {
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_storage(handle) != GpuTensorStorage::Real
                || runmat_accelerate_api::handle_integer_type(handle).is_some()
                || runmat_accelerate_api::handle_is_logical(handle)
            {
                return Ok(None);
            }
            let len = handle
                .shape
                .iter()
                .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
                .ok_or_else(|| {
                    interp1_error_with_message(
                        "interp1: Xq size overflow",
                        &INTERP1_ERROR_INVALID_INPUT,
                    )
                })?;
            if !gpu_helpers::exact_provider_for_handle(handle)
                .is_some_and(|owner| std::ptr::eq(owner, provider))
            {
                return Err(interp1_error_with_message(
                    "interp1: resident Xq is not owned by the Y provider",
                    &INTERP1_ERROR_INTERNAL,
                ));
            }
            Ok(Some(GpuInterp1Query {
                handle: handle.clone(),
                shape: tensor::default_shape_for(&handle.shape, len),
                len,
                owned: false,
            }))
        }
        other => {
            let query = query_points(other.clone(), NAME).await?;
            let uploaded = provider
                .upload(&HostTensorView {
                    data: &query.values,
                    shape: &query.shape,
                })
                .map_err(|err| {
                    interp1_error_with_message(
                        format!("interp1: Xq upload failed: {err}"),
                        &INTERP1_ERROR_INTERNAL,
                    )
                })?;
            Ok(Some(GpuInterp1Query {
                handle: uploaded,
                shape: query.shape,
                len: query.values.len(),
                owned: true,
            }))
        }
    }
}

fn interp1_gpu_output_shape(
    query_shape: &[usize],
    series_count: usize,
    trailing_shape: &[usize],
) -> Vec<usize> {
    if series_count == 1 {
        return query_shape.to_vec();
    }
    let mut shape = query_shape.to_vec();
    if trailing_shape.is_empty() {
        shape.push(series_count);
    } else {
        shape.extend(trailing_shape.iter().copied());
    }
    shape
}

fn provider_interp1_method(method: InterpMethod) -> ProviderInterp1Method {
    match method {
        InterpMethod::Linear => ProviderInterp1Method::Linear,
        InterpMethod::Nearest => ProviderInterp1Method::Nearest,
        _ => unreachable!("provider fast path is limited to direct methods"),
    }
}

fn provider_interp1_extrapolation(extrap: &Extrapolation) -> ProviderInterp1Extrapolation {
    match extrap {
        Extrapolation::Nan => ProviderInterp1Extrapolation::Nan,
        Extrapolation::Extrapolate => ProviderInterp1Extrapolation::Extrapolate,
        Extrapolation::Value(_) => ProviderInterp1Extrapolation::Value,
    }
}

struct ParsedInterp1 {
    series: super::pp::NumericSeries,
    query: super::pp::QueryPoints,
    method: InterpMethod,
    extrap: Extrapolation,
}

impl ParsedInterp1 {
    async fn parse(args: Vec<Value>) -> crate::BuiltinResult<Self> {
        if args.len() < 2 {
            return Err(interp1_invalid_argument(
                "expected at least Y and Xq arguments",
            ));
        }

        let mut method = InterpMethod::Linear;
        let mut extrap = Extrapolation::Nan;
        let (series, query, options) = if args.len() == 2 || third_arg_is_option(&args) {
            let mut iter = args.into_iter();
            let y = iter.next().expect("Y argument");
            let xq = iter.next().expect("Xq argument");
            let series = implicit_series_from_values(y, NAME).await?;
            let query = query_points(xq, NAME).await?;
            (series, query, iter.collect::<Vec<_>>())
        } else {
            let mut iter = args.into_iter();
            let x = iter.next().expect("X argument");
            let y = iter.next().expect("Y argument");
            let xq = iter.next().expect("Xq argument");
            let series = series_from_values(x, y, NAME).await?;
            let query = query_points(xq, NAME).await?;
            (series, query, iter.collect::<Vec<_>>())
        };

        for option in &options {
            if let Some(parsed) = parse_extrapolation(option, NAME).await? {
                extrap = parsed;
                continue;
            }
            if let Some(parsed) = parse_method(option, NAME)? {
                method = parsed;
                continue;
            }
            return Err(interp1_error_with_message(
                "interp1: unsupported interpolation option",
                &INTERP1_ERROR_INVALID_ARGUMENT,
            ));
        }

        Ok(Self {
            series,
            query,
            method,
            extrap,
        })
    }

    fn extrap_for_cubic(&self) -> Extrapolation {
        self.extrap.clone()
    }
}

fn third_arg_is_option(args: &[Value]) -> bool {
    args.get(2)
        .and_then(|value| crate::builtins::common::random_args::keyword_of(value))
        .is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntegerStorage, Tensor};

    fn row(values: &[f64]) -> Value {
        Value::Tensor(Tensor::new(values.to_vec(), vec![1, values.len()]).expect("tensor"))
    }

    fn int_row(storage: IntegerStorage) -> Value {
        let len = storage.len();
        let tensor = Tensor::new_integer(storage, vec![1, len]).expect("integer tensor");
        Value::Tensor(tensor)
    }

    fn run(args: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(interp1_builtin(args))
    }

    #[test]
    fn interp1_linear_midpoints() {
        let result = run(vec![
            row(&[1.0, 2.0, 3.0]),
            row(&[10.0, 20.0, 40.0]),
            row(&[1.5, 2.5]),
        ])
        .expect("interp1");
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.materialize_f64(), vec![15.0, 30.0]);
    }

    #[test]
    fn interp1_nearest() {
        let result = run(vec![
            row(&[1.0, 2.0, 3.0]),
            row(&[10.0, 20.0, 40.0]),
            row(&[1.2, 2.8]),
            Value::String("nearest".to_string()),
        ])
        .expect("interp1");
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.materialize_f64(), vec![10.0, 40.0]);
    }

    #[test]
    fn interp1_reads_typed_integer_x_y_and_query_exactly() {
        let args = || {
            vec![
                int_row(IntegerStorage::I16(vec![1, 2, 3])),
                int_row(IntegerStorage::U16(vec![10, 20, 40])),
                int_row(IntegerStorage::I16(vec![1, 2])),
            ]
        };
        let error = run(args()).expect_err("compatible mode rejects integer samples");
        assert_eq!(
            error.identifier(),
            INTERP1_INTEGER_SAMPLE_EXTENSION.error_identifier
        );
        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        let result = run(args()).expect("RunMat integer interpolation");
        let Value::Tensor(tensor) = result else {
            panic!("expected tensor");
        };
        assert_eq!(tensor.materialize_f64(), vec![10.0, 20.0]);
    }

    #[test]
    fn interp1_gpu_implicit_x_linear_keeps_output_resident() {
        test_support::with_test_provider(|provider| {
            let y = provider
                .upload(&HostTensorView {
                    data: &[10.0, 20.0, 40.0],
                    shape: &[1, 3],
                })
                .expect("upload y");
            let result = run(vec![Value::GpuTensor(y.clone()), row(&[1.5, 2.5])]).expect("interp1");
            let Value::GpuTensor(handle) = result else {
                panic!("expected resident gpu output");
            };
            assert_eq!(handle.shape, vec![1, 2]);
            let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
            assert_eq!(gathered.materialize_f64(), vec![15.0, 30.0]);
            let _ = provider.free(&y);
        });
    }

    #[test]
    fn interp1_gpu_explicit_x_resident_query_preserves_series_shape() {
        test_support::with_test_provider(|provider| {
            let y = provider
                .upload(&HostTensorView {
                    data: &[10.0, 20.0, 40.0, 100.0, 200.0, 400.0],
                    shape: &[3, 2],
                })
                .expect("upload y");
            let xq = provider
                .upload(&HostTensorView {
                    data: &[1.5, 2.5],
                    shape: &[1, 2],
                })
                .expect("upload xq");
            let result = run(vec![
                row(&[1.0, 2.0, 3.0]),
                Value::GpuTensor(y.clone()),
                Value::GpuTensor(xq.clone()),
            ])
            .expect("interp1");
            let Value::GpuTensor(handle) = result else {
                panic!("expected resident gpu output");
            };
            assert_eq!(handle.shape, vec![1, 2, 2]);
            let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
            assert_eq!(gathered.materialize_f64(), vec![15.0, 30.0, 150.0, 300.0]);
            let _ = provider.free(&y);
            let _ = provider.free(&xq);
        });
    }

    #[test]
    fn interp1_gpu_explicit_query_makes_output_explicit() {
        test_support::with_test_provider(|provider| {
            let y = provider
                .upload(&HostTensorView {
                    data: &[10.0, 20.0, 40.0],
                    shape: &[1, 3],
                })
                .expect("upload y");
            let xq = provider
                .upload(&HostTensorView {
                    data: &[1.5],
                    shape: &[1, 1],
                })
                .expect("upload xq");
            runmat_accelerate_api::set_handle_provenance(
                &xq,
                runmat_accelerate_api::GpuHandleProvenance::Explicit,
            );
            let result = run(vec![
                Value::GpuTensor(y.clone()),
                Value::GpuTensor(xq.clone()),
            ])
            .expect("interp1");
            let Value::GpuTensor(output) = result else {
                panic!("expected resident output");
            };
            assert_eq!(
                runmat_accelerate_api::handle_provenance(&output),
                Some(runmat_accelerate_api::GpuHandleProvenance::Explicit)
            );
            provider.free(&y).ok();
            provider.free(&xq).ok();
            provider.free(&output).ok();
        });
    }

    #[test]
    fn interp1_gpu_allows_same_resident_handle_for_y_and_query() {
        test_support::with_test_provider(|provider| {
            let y = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0, 3.0],
                    shape: &[1, 3],
                })
                .expect("upload y/query");
            let metadata = gpu_helpers::snapshot_handle_metadata(&y);
            let result = run(vec![
                Value::GpuTensor(y.clone()),
                Value::GpuTensor(y.clone()),
            ])
            .expect("same-handle Y/Xq interpolation");
            let Value::GpuTensor(output) = result else {
                panic!("expected resident output");
            };
            assert_eq!(output.shape, vec![1, 3]);
            assert_eq!(gpu_helpers::snapshot_handle_metadata(&y), metadata);
            let gathered = test_support::gather(Value::GpuTensor(output)).expect("gather");
            assert_eq!(gathered.materialize_f64(), vec![1.0, 2.0, 3.0]);
            provider.free(&y).ok();
        });
    }

    #[test]
    fn interp1_gpu_input_validation_rejects_alias_and_bad_class_metadata() {
        test_support::with_test_provider(|provider| {
            let y = provider
                .upload(&HostTensorView {
                    data: &[10.0, 20.0],
                    shape: &[1, 2],
                })
                .expect("upload y");
            let precision = runmat_accelerate_api::handle_precision(&y);
            assert!(!valid_interp1_gpu_input(
                &y,
                &[1, 2],
                precision,
                provider,
                &[&y],
            ));
            let other = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0],
                    shape: &[1, 2],
                })
                .expect("upload other");
            runmat_accelerate_api::set_handle_class_name(&other, "uint64");
            assert!(!valid_interp1_gpu_input(
                &other,
                &[1, 2],
                precision,
                provider,
                &[&y],
            ));
            provider.free(&y).ok();
            provider.free(&other).ok();
        });
    }

    #[test]
    fn interp1_gpu_scalar_fill_value_uses_provider_path() {
        test_support::with_test_provider(|provider| {
            let y = provider
                .upload(&HostTensorView {
                    data: &[10.0, 20.0],
                    shape: &[1, 2],
                })
                .expect("upload y");
            let result = run(vec![
                Value::GpuTensor(y.clone()),
                Value::Num(0.0),
                Value::String("linear".to_string()),
                Value::Num(99.0),
            ])
            .expect("interp1");
            let Value::GpuTensor(handle) = result else {
                panic!("expected resident gpu output");
            };
            let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
            assert_eq!(gathered.materialize_f64(), vec![99.0]);
            let _ = provider.free(&y);
        });
    }

    #[test]
    fn interp1_integer_extrapolation_is_independently_gated() {
        let args = || {
            vec![
                row(&[10.0, 20.0]),
                Value::Num(0.0),
                Value::String("linear".to_string()),
                Value::Int(runmat_builtins::IntValue::I16(99)),
            ]
        };
        let error = run(args()).expect_err("compatible mode rejects integer extrapolation");
        assert_eq!(
            error.identifier(),
            INTERP1_INTEGER_EXTRAPOLATION_EXTENSION.error_identifier
        );
        let _guard = crate::compatibility::push_runmat_extensions_enabled(true);
        assert_eq!(run(args()).expect("RunMat extrapolation"), Value::Num(99.0));
    }

    #[test]
    fn interp1_default_out_of_range_is_nan() {
        let result =
            run(vec![row(&[1.0, 2.0]), row(&[10.0, 20.0]), Value::Num(0.0)]).expect("interp1");
        let Value::Num(value) = result else {
            panic!("expected scalar");
        };
        assert!(value.is_nan());
    }

    #[test]
    fn interp1_extrapolates_when_requested() {
        let result = run(vec![
            row(&[1.0, 2.0]),
            row(&[10.0, 20.0]),
            Value::Num(0.0),
            Value::String("extrap".to_string()),
        ])
        .expect("interp1");
        assert_eq!(result, Value::Num(0.0));
    }

    #[test]
    fn interp1_descriptor_signatures_cover_surface() {
        let labels: Vec<&str> = INTERP1_DESCRIPTOR
            .signatures
            .iter()
            .map(|signature| signature.label)
            .collect();
        assert!(labels.contains(&"Vq = interp1(Y, Xq)"));
        assert!(labels.contains(&"Vq = interp1(X, Y, Xq)"));
        assert!(labels.contains(&"Vq = interp1(X, Y, Xq, method, extrap)"));
    }

    #[test]
    fn interp1_descriptor_errors_have_stable_codes() {
        let codes: Vec<&str> = INTERP1_DESCRIPTOR
            .errors
            .iter()
            .map(|error| error.code)
            .collect();
        assert!(codes.contains(&"RM.INTERP1.INVALID_ARGUMENT"));
        assert!(codes.contains(&"RM.INTERP1.INVALID_INPUT"));
        assert!(codes.contains(&"RM.INTERP1.INTERNAL"));
    }

    #[test]
    fn interp1_too_few_args_uses_stable_identifier() {
        let err = run(vec![row(&[1.0, 2.0])]).expect_err("expected interp1 argument error");
        assert_eq!(err.identifier(), INTERP1_ERROR_INVALID_ARGUMENT.identifier);
    }
}
