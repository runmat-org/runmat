//! MATLAB-compatible `cov` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{
    AccelProvider, CovNormalization, CovRows, CovarianceOptions, GpuTensorHandle, GpuTensorStorage,
    HostTensorView,
};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    IntValue, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor;
use crate::builtins::stats::type_resolvers::cov_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "cov";
const COV_OUTPUT: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Covariance matrix.",
}];

const COV_INPUTS_X: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "X",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input observations (rows are observations, columns are variables).",
}];

const COV_INPUTS_X_Y_OR_W: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input observations (rows are observations, columns are variables).",
    },
    BuiltinParamDescriptor {
        name: "Y_or_w",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second dataset (Y) or weight vector (w), depending on shape/position.",
    },
];

const COV_INPUTS_X_NORMALIZATION: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input observations (rows are observations, columns are variables).",
    },
    BuiltinParamDescriptor {
        name: "normalization",
        ty: BuiltinParamType::NumericScalar,
        arity: BuiltinParamArity::Required,
        default: Some("0"),
        description: "Normalization flag: 0 (unbiased) or 1 (biased).",
    },
];

const COV_INPUTS_X_ROWS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input observations (rows are observations, columns are variables).",
    },
    BuiltinParamDescriptor {
        name: "rows_option",
        ty: BuiltinParamType::StringScalar,
        arity: BuiltinParamArity::Required,
        default: Some("\"all\""),
        description: "Rows handling mode: 'all', 'omitrows', or 'partialrows'.",
    },
];

const COV_INPUTS_X_Y_OPT: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input observations (rows are observations, columns are variables).",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second dataset with matching size (or equal vector length).",
    },
    BuiltinParamDescriptor {
        name: "opt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Normalization flag or rows option.",
    },
];

const COV_INPUTS_X_Y_W: [BuiltinParamDescriptor; 3] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input observations (rows are observations, columns are variables).",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second dataset with matching size (or equal vector length).",
    },
    BuiltinParamDescriptor {
        name: "w",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Weight vector with one weight per observation row.",
    },
];

const COV_INPUTS_X_Y_W_OPT: [BuiltinParamDescriptor; 4] = [
    BuiltinParamDescriptor {
        name: "X",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Input observations (rows are observations, columns are variables).",
    },
    BuiltinParamDescriptor {
        name: "Y",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Second dataset with matching size (or equal vector length).",
    },
    BuiltinParamDescriptor {
        name: "w",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Weight vector with one weight per observation row.",
    },
    BuiltinParamDescriptor {
        name: "opt",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Normalization flag or rows option.",
    },
];

const COV_SIGNATURES: [BuiltinSignatureDescriptor; 7] = [
    BuiltinSignatureDescriptor {
        label: "C = cov(X)",
        inputs: &COV_INPUTS_X,
        outputs: &COV_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = cov(X, Y_or_w)",
        inputs: &COV_INPUTS_X_Y_OR_W,
        outputs: &COV_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = cov(X, normalization)",
        inputs: &COV_INPUTS_X_NORMALIZATION,
        outputs: &COV_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = cov(X, rows_option)",
        inputs: &COV_INPUTS_X_ROWS,
        outputs: &COV_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = cov(X, Y, opt)",
        inputs: &COV_INPUTS_X_Y_OPT,
        outputs: &COV_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = cov(X, Y, w)",
        inputs: &COV_INPUTS_X_Y_W,
        outputs: &COV_OUTPUT,
    },
    BuiltinSignatureDescriptor {
        label: "C = cov(X, Y, w, opt)",
        inputs: &COV_INPUTS_X_Y_W_OPT,
        outputs: &COV_OUTPUT,
    },
];

const COV_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COV.INVALID_ARGUMENT",
    identifier: Some("RunMat:cov:InvalidArgument"),
    when: "Arguments are malformed or unsupported for cov.",
    message: "cov: invalid argument",
};

const COV_ERROR_COMPLEX_UNSUPPORTED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COV.COMPLEX_UNSUPPORTED",
    identifier: Some("RunMat:cov:ComplexUnsupported"),
    when: "Any argument is complex-valued.",
    message: "cov: complex inputs are not supported yet",
};

const COV_ERROR_ROWS_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COV.ROWS_MISMATCH",
    identifier: Some("RunMat:cov:RowsMismatch"),
    when: "Two input datasets do not have the same size or equal vector lengths.",
    message: "cov: paired inputs must have the same size",
};

const COV_ERROR_NORMALIZATION_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COV.NORMALIZATION_INVALID",
    identifier: Some("RunMat:cov:NormalizationInvalid"),
    when: "Normalization flag is non-finite, non-integer, or not 0/1.",
    message: "cov: normalization flag is invalid",
};

const COV_ERROR_WEIGHT_VECTOR_LENGTH_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COV.WEIGHT_VECTOR_LENGTH_MISMATCH",
    identifier: Some("RunMat:cov:WeightVectorLengthMismatch"),
    when: "Weight vector length does not match observation row count.",
    message: "cov: weight vector length mismatch",
};

const COV_ERROR_ROWS_OPTION_UNKNOWN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COV.ROWS_OPTION_UNKNOWN",
    identifier: Some("RunMat:cov:RowsOptionUnknown"),
    when: "Rows option is not one of all/omitrows/partialrows.",
    message: "cov: unknown rows option",
};

const COV_ERROR_NORMALIZATION_DUPLICATE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COV.NORMALIZATION_DUPLICATE",
    identifier: Some("RunMat:cov:NormalizationDuplicate"),
    when: "Normalization flag is provided more than once.",
    message: "cov: normalization flag specified more than once",
};

const COV_ERROR_TOO_MANY_ARRAY_ARGUMENTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COV.TOO_MANY_ARRAY_ARGUMENTS",
    identifier: Some("RunMat:cov:TooManyArrayArguments"),
    when: "More than two data arrays (or Y plus weight) are provided.",
    message: "cov: too many array arguments",
};

const COV_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COV.INTERNAL",
    identifier: Some("RunMat:cov:Internal"),
    when: "Internal tensor conversion/allocation or covariance computation fails.",
    message: "cov: internal operation failed",
};

const COV_ERRORS: [BuiltinErrorDescriptor; 9] = [
    COV_ERROR_INVALID_ARGUMENT,
    COV_ERROR_COMPLEX_UNSUPPORTED,
    COV_ERROR_ROWS_MISMATCH,
    COV_ERROR_NORMALIZATION_INVALID,
    COV_ERROR_WEIGHT_VECTOR_LENGTH_MISMATCH,
    COV_ERROR_ROWS_OPTION_UNKNOWN,
    COV_ERROR_NORMALIZATION_DUPLICATE,
    COV_ERROR_TOO_MANY_ARRAY_ARGUMENTS,
    COV_ERROR_INTERNAL,
];

const COV_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cov-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cov with typed-integer observation data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CovIntegerDataExtension"),
};

const COV_LOGICAL_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cov-logical-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cov with logical observation data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CovLogicalDataExtension"),
};

const COV_TYPED_NORMALIZATION_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cov-typed-normalization",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cov with typed-integer or logical normalization is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CovTypedNormalizationExtension"),
};

const COV_VECTOR_WEIGHTS_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cov-vector-weights",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cov with a vector of observation weights is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CovVectorWeightsExtension"),
};

const COV_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    COV_INTEGER_DATA_EXTENSION,
    COV_LOGICAL_DATA_EXTENSION,
    COV_TYPED_NORMALIZATION_EXTENSION,
    COV_VECTOR_WEIGHTS_EXTENSION,
];

const COV_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A_or_B",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented observation domain is single or double; RunMat mode additionally accepts all eight real integer classes.",
    }];

const COV_INTEGER_NORMALIZATION_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "w",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "The documented scalar normalization weight is single or double; RunMat mode additionally accepts typed integer 0 or 1.",
    }];

const COV_INTEGER_VECTOR_WEIGHTS_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "vector_weights",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::Allowed,
        notes: "Observation-weight vectors are a RunMat extension for every numeric class; integer vectors are validated exactly before floating weighted covariance.",
    }];

const COV_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 3] = [
    BuiltinIntegerCapabilityDescriptor {
        form: "C = cov(integer_A_or_B, w, nanflag)",
        inputs: &COV_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat preserves exact integer differences while centering each variable before producing double covariance.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "C = cov(A, integer_w, nanflag)",
        inputs: &COV_INTEGER_NORMALIZATION_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::Structural,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::HostAndGpu,
        overload: BuiltinIntegerOverloadKind::StructuralParameter,
        notes: "Typed integer normalization is a RunMat-only control and must equal zero or one.",
    },
    BuiltinIntegerCapabilityDescriptor {
        form: "C = cov(A, integer_vector_weights, nanflag)",
        inputs: &COV_INTEGER_VECTOR_WEIGHTS_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::OptionDependent,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "RunMat-only observation-weight vectors are nonnegative and finite; integer weights enter the floating weighted covariance domain and the data input determines output precision.",
    },
];

pub const COV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &COV_SIGNATURES,
    output_mode: BuiltinOutputMode::Fixed,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &COV_ERRORS,
};

fn cov_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn cov_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    cov_error_with(error, error.message)
}

fn cov_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    cov_error_with(error, format!("{}: {detail}", error.message))
}

fn cov_internal_error(message: impl Into<String>) -> RuntimeError {
    cov_error_with(&COV_ERROR_INTERNAL, message)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::stats::summary::cov")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "cov",
    op_kind: GpuOpKind::Custom("summary-stats"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("covariance")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "GPU execution is available when rows='all' and no weight vector is supplied; other cases fall back to the CPU path.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::stats::summary::cov")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "cov",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "The covariance builtin is treated as a fusion boundary and executes via dedicated kernels or the host reference.",
};

#[runtime_builtin(
    name = "cov",
    category = "stats/summary",
    summary = "Compute covariance matrices.",
    keywords = "cov,covariance,statistics,weights,gpu",
    accel = "reduction",
    type_resolver(cov_type),
    descriptor(crate::builtins::stats::summary::cov::COV_DESCRIPTOR),
    extensions(COV_EXTENSIONS),
    integer_capabilities(COV_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::summary::cov"
)]
async fn cov_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let args = CovArgs::parse(value, rest)?;
    ensure_cov_extensions(&args)?;
    if let Some(result) = cov_try_gpu(&args).await? {
        return Ok(result);
    }
    cov_host(args).await
}

/// Public entry point for providers that need the reference implementation.
pub fn cov_from_tensors(
    left: Tensor,
    right: Option<Tensor>,
    rows: CovRows,
    weight: CovWeightSpec,
) -> BuiltinResult<Tensor> {
    let matrix = combine_tensors(left, right)?;
    if let CovWeightSpec::Vector(ref vec) = weight {
        if matrix.rows != vec.len() {
            return Err(cov_error_with_detail(
                &COV_ERROR_WEIGHT_VECTOR_LENGTH_MISMATCH,
                format!("expected {} elements", matrix.rows),
            ));
        }
    }
    match rows {
        CovRows::All => covariance_dense(&matrix, &weight),
        CovRows::OmitRows => {
            let (filtered, filtered_weight) = filter_complete_rows(&matrix, weight);
            covariance_dense(&filtered, &filtered_weight)
        }
        CovRows::PartialRows => covariance_pairwise(&matrix, &weight),
    }
}

#[derive(Debug)]
struct CovArgs {
    first: Value,
    second: Option<Value>,
    normalization: CovNormalization,
    rows: CovRows,
    weight_vector: Option<Value>,
    typed_normalization: bool,
}

impl CovArgs {
    fn parse(first: Value, rest: Vec<Value>) -> BuiltinResult<Self> {
        let mut second_candidate: Option<Value> = None;
        let mut weight_candidate: Option<Value> = None;
        let mut normalization = CovNormalization::Unbiased;
        let mut normalization_explicit = false;
        let mut typed_normalization = false;
        let mut rows = CovRows::All;

        let iter = rest.into_iter();
        for arg in iter {
            match arg {
                Value::String(_) | Value::StringArray(_) | Value::CharArray(_) => {
                    let key = tensor::value_to_string(&arg)
                        .ok_or_else(|| cov_error(&COV_ERROR_INVALID_ARGUMENT))?;
                    let lowered = key.trim().to_ascii_lowercase();
                    rows = parse_rows_option(&lowered)?;
                }
                Value::Tensor(_) | Value::LogicalArray(_) | Value::GpuTensor(_) => {
                    if second_candidate.is_none() {
                        second_candidate = Some(arg);
                    } else if weight_candidate.is_none() {
                        weight_candidate = Some(arg);
                    } else {
                        return Err(cov_error(&COV_ERROR_TOO_MANY_ARRAY_ARGUMENTS));
                    }
                }
                Value::Num(_) | Value::Int(_) | Value::Bool(_) => {
                    if normalization_explicit || weight_candidate.is_some() {
                        return Err(cov_error(&COV_ERROR_NORMALIZATION_DUPLICATE));
                    }
                    typed_normalization = matches!(arg, Value::Int(_) | Value::Bool(_));
                    normalization = parse_normalization(arg)?;
                    normalization_explicit = true;
                }
                Value::ComplexTensor(_) => {
                    return Err(cov_error(&COV_ERROR_COMPLEX_UNSUPPORTED));
                }
                other => {
                    return Err(cov_error_with_detail(
                        &COV_ERROR_INVALID_ARGUMENT,
                        format!("{other:?}"),
                    ))
                }
            }
        }

        if let Some(weight_array) = weight_candidate {
            // Explicit weight vector always takes precedence over dataset detection.
            return Ok(Self {
                first,
                second: second_candidate,
                normalization,
                rows,
                weight_vector: Some(weight_array),
                typed_normalization,
            });
        }

        let mut second = second_candidate;
        let mut weight_vector: Option<Value> = None;

        if let Some(candidate) = second.take() {
            if should_treat_as_weight(&first, &candidate, normalization_explicit, rows)? {
                weight_vector = Some(candidate);
            } else {
                second = Some(candidate);
            }
        }

        Ok(Self {
            first,
            second,
            normalization,
            rows,
            weight_vector,
            typed_normalization,
        })
    }
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(
            value,
            Value::GpuTensor(handle)
                if runmat_accelerate_api::handle_integer_type(handle).is_some()
        )
}

fn is_logical_value(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(
            value,
            Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle)
        )
}

fn ensure_cov_extensions(args: &CovArgs) -> BuiltinResult<()> {
    if is_typed_integer_value(&args.first)
        || args.second.as_ref().is_some_and(is_typed_integer_value)
    {
        crate::compatibility::ensure_builtin_extension_enabled(&COV_INTEGER_DATA_EXTENSION, NAME)?;
    }
    if is_logical_value(&args.first) || args.second.as_ref().is_some_and(is_logical_value) {
        crate::compatibility::ensure_builtin_extension_enabled(&COV_LOGICAL_DATA_EXTENSION, NAME)?;
    }
    if args.typed_normalization {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COV_TYPED_NORMALIZATION_EXTENSION,
            NAME,
        )?;
    }
    if args.weight_vector.is_some() {
        crate::compatibility::ensure_builtin_extension_enabled(
            &COV_VECTOR_WEIGHTS_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub enum CovWeightSpec {
    Scalar(CovNormalization),
    Vector(Vec<f64>),
}

async fn cov_try_gpu(args: &CovArgs) -> BuiltinResult<Option<Value>> {
    if args.rows != CovRows::All {
        return Ok(None);
    }

    let first_handle = match &args.first {
        Value::GpuTensor(handle) => handle,
        _ => return Ok(None),
    };

    let provider = match runmat_accelerate_api::provider_for_handle(first_handle)
        .or_else(runmat_accelerate_api::provider)
    {
        Some(p) => p,
        None => return Ok(None),
    };

    let maybe_second_handle = match &args.second {
        Some(Value::GpuTensor(handle)) => {
            let Some(second_provider) = runmat_accelerate_api::provider_for_handle(handle) else {
                return Ok(None);
            };
            if !std::ptr::eq(provider, second_provider) {
                return Ok(None);
            }
            Some(handle)
        }
        Some(_) => return Ok(None),
        None => None,
    };

    let rows = gpu_observation_count(first_handle, maybe_second_handle.is_some())?;
    let mut temporary_inputs = Vec::new();
    let weight_handle = match materialize_gpu_weight_vector(
        provider,
        args.weight_vector.as_ref(),
        rows,
        &mut temporary_inputs,
    )
    .await
    {
        Ok(weight) => weight,
        Err(err) => {
            free_temporary_gpu_inputs(provider, temporary_inputs);
            return Err(err);
        }
    };
    if args.weight_vector.is_some() && weight_handle.is_none() {
        free_temporary_gpu_inputs(provider, temporary_inputs);
        return Ok(None);
    }

    let options = CovarianceOptions {
        normalization: args.normalization,
        rows: args.rows,
        has_weight_vector: weight_handle.is_some(),
    };

    match provider
        .covariance(
            first_handle,
            maybe_second_handle,
            weight_handle.as_ref(),
            &options,
        )
        .await
    {
        Ok(result) => {
            free_temporary_gpu_inputs(provider, temporary_inputs);
            Ok(Some(Value::GpuTensor(result)))
        }
        Err(_) => {
            free_temporary_gpu_inputs(provider, temporary_inputs);
            Ok(None)
        }
    }
}

fn gpu_observation_count(handle: &GpuTensorHandle, paired_input: bool) -> BuiltinResult<usize> {
    if handle.shape.len() > 2 {
        return Err(cov_error_with_detail(
            &COV_ERROR_INVALID_ARGUMENT,
            "inputs must be 2-D matrices or vectors",
        ));
    }
    let len = handle.shape.iter().copied().product::<usize>();
    Ok(match handle.shape.as_slice() {
        [] => 1,
        _ if paired_input => len,
        [length] => *length,
        [rows, cols] if *rows == 1 || *cols == 1 => len,
        [rows, _] => *rows,
        _ => unreachable!("shape rank was checked above"),
    })
}

async fn materialize_gpu_weight_vector(
    provider: &dyn AccelProvider,
    value: Option<&Value>,
    expected_rows: usize,
    temporary_inputs: &mut Vec<GpuTensorHandle>,
) -> BuiltinResult<Option<GpuTensorHandle>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if expected_rows == 0 {
        return Err(cov_error_with_detail(
            &COV_ERROR_INVALID_ARGUMENT,
            "weight vector cannot be empty",
        ));
    }

    match value {
        Value::GpuTensor(handle) => {
            if runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::ComplexInterleaved
            {
                return Ok(None);
            }
            validate_gpu_weight_shape(handle, expected_rows)?;
            let Some(weight_provider) = runmat_accelerate_api::provider_for_handle(handle) else {
                return Ok(None);
            };
            if !std::ptr::eq(provider, weight_provider) {
                return Ok(None);
            }
            Ok(Some(handle.clone()))
        }
        other => {
            let weights = value_to_weight_vector(other.clone(), expected_rows).await?;
            let shape = [expected_rows, 1];
            let handle = provider
                .upload(&HostTensorView {
                    data: &weights,
                    shape: &shape,
                })
                .map_err(|err| cov_internal_error(err.to_string()))?;
            temporary_inputs.push(handle.clone());
            Ok(Some(handle))
        }
    }
}

fn validate_gpu_weight_shape(handle: &GpuTensorHandle, expected_rows: usize) -> BuiltinResult<()> {
    if handle.shape.len() > 2 {
        return Err(cov_error_with_detail(
            &COV_ERROR_INVALID_ARGUMENT,
            "weight vector must be one-dimensional",
        ));
    }
    let rows = if handle.shape.is_empty() {
        1
    } else {
        handle.shape[0]
    };
    let cols = if handle.shape.len() >= 2 {
        handle.shape[1]
    } else {
        1
    };
    if rows != 1 && cols != 1 {
        return Err(cov_error_with_detail(
            &COV_ERROR_INVALID_ARGUMENT,
            "weight vector must be one-dimensional",
        ));
    }
    if rows != expected_rows && cols != expected_rows {
        return Err(cov_error_with_detail(
            &COV_ERROR_WEIGHT_VECTOR_LENGTH_MISMATCH,
            format!("expected {expected_rows} elements"),
        ));
    }
    Ok(())
}

fn free_temporary_gpu_inputs(provider: &dyn AccelProvider, handles: Vec<GpuTensorHandle>) {
    for handle in handles {
        let _ = provider.free(&handle);
    }
}

async fn cov_host(args: CovArgs) -> BuiltinResult<Value> {
    let CovArgs {
        first,
        second,
        normalization,
        rows,
        weight_vector,
        ..
    } = args;

    let left = value_to_tensor_gather(first).await?;
    let right = match second {
        Some(value) => Some(value_to_tensor_gather(value).await?),
        None => None,
    };
    let expected_weight_rows = if right.is_some() || left.rows() == 1 || left.cols() == 1 {
        left.len()
    } else {
        left.rows()
    };

    let weight_spec = if let Some(weight_value) = weight_vector {
        let vector = value_to_weight_vector(weight_value, expected_weight_rows).await?;
        CovWeightSpec::Vector(vector)
    } else {
        CovWeightSpec::Scalar(normalization)
    };

    let tensor = cov_from_tensors(left, right, rows, weight_spec)?;
    Ok(Value::Tensor(tensor))
}

async fn value_to_tensor_gather(value: Value) -> BuiltinResult<Tensor> {
    match value {
        Value::GpuTensor(handle) => gpu_helpers::gather_tensor_async(&handle).await,
        Value::LogicalArray(logical) => {
            tensor::logical_to_tensor(&logical).map_err(cov_internal_error)
        }
        other => tensor::value_into_tensor_for("cov", other).map_err(cov_internal_error),
    }
}

async fn value_to_weight_vector(value: Value, expected_rows: usize) -> BuiltinResult<Vec<f64>> {
    let tensor = match value {
        Value::GpuTensor(handle) => gpu_helpers::gather_tensor_async(&handle).await?,
        Value::LogicalArray(logical) => {
            tensor::logical_to_tensor(&logical).map_err(cov_internal_error)?
        }
        other => tensor::value_into_tensor_for("cov", other).map_err(cov_internal_error)?,
    };
    if tensor.shape.len() > 2 {
        return Err(cov_error_with_detail(
            &COV_ERROR_INVALID_ARGUMENT,
            "weight vector must be one-dimensional",
        ));
    }
    if tensor.rows() != expected_rows && tensor.cols() != expected_rows {
        return Err(cov_error_with_detail(
            &COV_ERROR_WEIGHT_VECTOR_LENGTH_MISMATCH,
            format!("expected {expected_rows} elements"),
        ));
    }
    let values = tensor::tensor_into_values_f64(tensor);
    for (idx, weight) in values.iter().enumerate() {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(cov_error_with_detail(
                &COV_ERROR_INVALID_ARGUMENT,
                format!("weights must be non-negative finite values (index {idx})"),
            ));
        }
    }
    if values.is_empty() {
        return Err(cov_error_with_detail(
            &COV_ERROR_INVALID_ARGUMENT,
            "weight vector cannot be empty",
        ));
    }
    Ok(values)
}

fn parse_rows_option(value: &str) -> BuiltinResult<CovRows> {
    match value {
        "all" => Ok(CovRows::All),
        "omitrows" | "omit" => Ok(CovRows::OmitRows),
        "partialrows" | "partial" | "pairwise" => Ok(CovRows::PartialRows),
        other => Err(cov_error_with_detail(
            &COV_ERROR_ROWS_OPTION_UNKNOWN,
            format!("'{other}'"),
        )),
    }
}

fn parse_normalization(value: Value) -> BuiltinResult<CovNormalization> {
    match value {
        Value::Int(i) => match i.to_i64() {
            0 => Ok(CovNormalization::Unbiased),
            1 => Ok(CovNormalization::Biased),
            other => Err(cov_error_with_detail(
                &COV_ERROR_NORMALIZATION_INVALID,
                format!("expected 0 or 1, received {other}"),
            )),
        },
        Value::Num(n) => {
            if !n.is_finite() {
                return Err(cov_error_with_detail(
                    &COV_ERROR_NORMALIZATION_INVALID,
                    "value must be finite",
                ));
            }
            let rounded = n.round();
            if (rounded - n).abs() > 1.0e-12 {
                return Err(cov_error_with_detail(
                    &COV_ERROR_NORMALIZATION_INVALID,
                    "value must be an integer",
                ));
            }
            match rounded as i64 {
                0 => Ok(CovNormalization::Unbiased),
                1 => Ok(CovNormalization::Biased),
                other => Err(cov_error_with_detail(
                    &COV_ERROR_NORMALIZATION_INVALID,
                    format!("expected 0 or 1, received {other}"),
                )),
            }
        }
        Value::Bool(flag) => Ok(if flag {
            CovNormalization::Biased
        } else {
            CovNormalization::Unbiased
        }),
        other => Err(cov_error_with_detail(
            &COV_ERROR_NORMALIZATION_INVALID,
            format!("value must be numeric, received {other:?}"),
        )),
    }
}

fn should_treat_as_weight(
    first: &Value,
    candidate: &Value,
    normalization_explicit: bool,
    rows_option: CovRows,
) -> BuiltinResult<bool> {
    let (rows_first, cols_first) = value_rows_cols(first)?;
    let (rows_candidate, cols_candidate) = value_rows_cols(candidate)?;

    let is_vector = rows_candidate == 1
        || cols_candidate == 1
        || rows_candidate * cols_candidate == rows_candidate
            && (rows_candidate == rows_first || cols_candidate == rows_first);

    if !is_vector {
        return Ok(false);
    }

    if rows_candidate != rows_first && cols_candidate != rows_first {
        // Length mismatch, treat as dataset so the later validation emits the proper error.
        return Ok(false);
    }

    if (rows_first == 1 || cols_first == 1)
        && !normalization_explicit
        && matches!(rows_option, CovRows::All)
    {
        // Ambiguous `cov(x, y)` case – prefer dataset semantics for compatibility.
        return Ok(false);
    }

    Ok(true)
}

fn value_rows_cols(value: &Value) -> BuiltinResult<(usize, usize)> {
    match value {
        Value::Tensor(tensor) => Ok((tensor.rows(), tensor.cols())),
        Value::LogicalArray(array) => {
            if array.shape.len() > 2 {
                return Err(cov_error_with_detail(
                    &COV_ERROR_INVALID_ARGUMENT,
                    "inputs must be 2-D matrices or vectors",
                ));
            }
            let rows = if array.shape.is_empty() {
                1
            } else {
                array.shape[0]
            };
            let cols = if array.shape.len() >= 2 {
                array.shape[1]
            } else {
                1
            };
            Ok((rows, cols))
        }
        Value::GpuTensor(handle) => {
            if handle.shape.len() > 2 {
                return Err(cov_error_with_detail(
                    &COV_ERROR_INVALID_ARGUMENT,
                    "inputs must be 2-D matrices or vectors",
                ));
            }
            let rows = if handle.shape.is_empty() {
                1
            } else {
                handle.shape[0]
            };
            let cols = if handle.shape.len() >= 2 {
                handle.shape[1]
            } else {
                1
            };
            Ok((rows, cols))
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) => Ok((1, 1)),
        other => Err(cov_error_with_detail(
            &COV_ERROR_INVALID_ARGUMENT,
            format!("unsupported input type for shape inspection: {other:?}"),
        )),
    }
}

#[derive(Debug, Clone)]
struct Matrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    fn from_single_tensor(tensor: Tensor) -> BuiltinResult<Self> {
        if tensor.shape.len() > 2 {
            return Err(cov_error_with_detail(
                &COV_ERROR_INVALID_ARGUMENT,
                "inputs must be 2-D matrices or vectors",
            ));
        }
        let original_rows = tensor.rows();
        let original_cols = tensor.cols();
        let is_vector = original_rows == 1 || original_cols == 1;
        let (rows, cols) = if is_vector {
            (tensor.len(), 1)
        } else if tensor.is_empty() && original_cols == 0 {
            (0, 1)
        } else {
            (original_rows, original_cols)
        };
        Ok(Self {
            rows,
            cols,
            data: centered_tensor_values(&tensor, rows, cols),
        })
    }

    fn from_tensor_pair(left: Tensor, right: Tensor) -> BuiltinResult<Self> {
        if left.shape.len() > 2 || right.shape.len() > 2 {
            return Err(cov_error_with_detail(
                &COV_ERROR_INVALID_ARGUMENT,
                "inputs must be 2-D matrices or vectors",
            ));
        }
        let left_is_vector = left.rows() == 1 || left.cols() == 1;
        let right_is_vector = right.rows() == 1 || right.cols() == 1;
        let compatible = if left_is_vector && right_is_vector {
            left.len() == right.len()
        } else {
            left.shape == right.shape
        };
        if !compatible {
            return Err(cov_error(&COV_ERROR_ROWS_MISMATCH));
        }
        let rows = left.len();
        let mut data = centered_tensor_values(&left, rows, 1);
        data.extend(centered_tensor_values(&right, rows, 1));
        Ok(Self {
            data,
            rows,
            cols: 2,
        })
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row + col * self.rows]
    }

    #[inline]
    fn column(&self, col: usize) -> &[f64] {
        let start = col * self.rows;
        let end = start + self.rows;
        &self.data[start..end]
    }
}

fn combine_tensors(left: Tensor, right: Option<Tensor>) -> BuiltinResult<Matrix> {
    if let Some(right) = right {
        Matrix::from_tensor_pair(left, right)
    } else {
        Matrix::from_single_tensor(left)
    }
}

fn centered_tensor_values(tensor: &Tensor, rows: usize, cols: usize) -> Vec<f64> {
    let Some(storage) = tensor.integer_storage() else {
        return tensor.materialize_f64();
    };
    let mut values = Vec::with_capacity(rows.saturating_mul(cols));
    for col in 0..cols {
        let start = col * rows;
        let anchor = storage
            .value_at(start)
            .map(|value| int_value_to_i128(&value))
            .unwrap_or(0);
        for index in start..start + rows {
            let value = storage
                .value_at(index)
                .map(|value| int_value_to_i128(&value))
                .unwrap_or(anchor);
            values.push((value - anchor) as f64);
        }
    }
    values
}

fn int_value_to_i128(value: &IntValue) -> i128 {
    match value {
        IntValue::I8(value) => i128::from(*value),
        IntValue::I16(value) => i128::from(*value),
        IntValue::I32(value) => i128::from(*value),
        IntValue::I64(value) => i128::from(*value),
        IntValue::U8(value) => i128::from(*value),
        IntValue::U16(value) => i128::from(*value),
        IntValue::U32(value) => i128::from(*value),
        IntValue::U64(value) => i128::from(*value),
    }
}

fn covariance_dense(matrix: &Matrix, weight: &CovWeightSpec) -> BuiltinResult<Tensor> {
    let cols = matrix.cols;
    let rows = matrix.rows;

    if cols == 0 {
        return Tensor::new(Vec::new(), vec![0, 0]).map_err(cov_internal_error);
    }

    let mut result = vec![f64::NAN; cols * cols];

    match weight {
        CovWeightSpec::Scalar(normalization) => {
            let denom = match normalization {
                CovNormalization::Unbiased => ((rows as f64) - 1.0).max(1.0),
                CovNormalization::Biased => rows as f64,
            };
            if denom <= 0.0 {
                return Tensor::new(result, vec![cols, cols]).map_err(cov_internal_error);
            }

            let mut means = vec![0.0; cols];
            for (col, mean_slot) in means.iter_mut().enumerate() {
                let column = matrix.column(col);
                let mut sum = 0.0;
                let mut valid = true;
                for &value in column {
                    if !value.is_finite() {
                        valid = false;
                        break;
                    }
                    sum += value;
                }
                *mean_slot = if valid { sum / (rows as f64) } else { f64::NAN };
            }

            for i in 0..cols {
                for j in i..cols {
                    let value = covariance_unweighted_pair(matrix, i, j, &means, denom);
                    set_entry(&mut result, cols, i, j, sanitize_covariance(i == j, value));
                }
            }
        }
        CovWeightSpec::Vector(weights) => {
            if weights.len() != rows {
                return Err(cov_error_with_detail(
                    &COV_ERROR_WEIGHT_VECTOR_LENGTH_MISMATCH,
                    format!("expected {rows} elements"),
                ));
            }
            let sum_w: f64 = weights.iter().sum();
            if sum_w <= 0.0 {
                return Tensor::new(result, vec![cols, cols]).map_err(cov_internal_error);
            }
            let denom = sum_w - 1.0;
            if denom <= 0.0 {
                return Tensor::new(result, vec![cols, cols]).map_err(cov_internal_error);
            }

            let mut means = vec![0.0; cols];
            for (col, mean_slot) in means.iter_mut().enumerate() {
                let column = matrix.column(col);
                let mut weighted_sum = 0.0;
                let mut valid = true;
                for (row, &value) in column.iter().enumerate() {
                    if !value.is_finite() {
                        valid = false;
                        break;
                    }
                    weighted_sum += weights[row] * value;
                }
                *mean_slot = if valid {
                    weighted_sum / sum_w
                } else {
                    f64::NAN
                };
            }

            for i in 0..cols {
                for j in i..cols {
                    let value = covariance_weighted_pair(matrix, i, j, weights, &means, denom);
                    set_entry(&mut result, cols, i, j, sanitize_covariance(i == j, value));
                }
            }
        }
    }

    Tensor::new(result, vec![cols, cols]).map_err(cov_internal_error)
}

fn filter_complete_rows(matrix: &Matrix, weight: CovWeightSpec) -> (Matrix, CovWeightSpec) {
    if matrix.rows == 0 {
        return (
            Matrix {
                data: Vec::new(),
                rows: 0,
                cols: matrix.cols,
            },
            weight,
        );
    }

    let mut valid_rows = Vec::new();
    for row in 0..matrix.rows {
        let mut is_valid = true;
        for col in 0..matrix.cols {
            if !matrix.get(row, col).is_finite() {
                is_valid = false;
                break;
            }
        }
        if is_valid {
            valid_rows.push(row);
        }
    }

    if valid_rows.len() == matrix.rows {
        // No filtering required.
        return (matrix.clone(), weight);
    }

    let mut data = Vec::with_capacity(valid_rows.len() * matrix.cols);
    for col in 0..matrix.cols {
        for &row in &valid_rows {
            data.push(matrix.get(row, col));
        }
    }

    let filtered_matrix = Matrix {
        data,
        rows: valid_rows.len(),
        cols: matrix.cols,
    };

    let filtered_weight = match weight {
        CovWeightSpec::Scalar(norm) => CovWeightSpec::Scalar(norm),
        CovWeightSpec::Vector(vec) => {
            let mut filtered = Vec::with_capacity(valid_rows.len());
            for &row in &valid_rows {
                filtered.push(vec[row]);
            }
            CovWeightSpec::Vector(filtered)
        }
    };

    (filtered_matrix, filtered_weight)
}

fn covariance_pairwise(matrix: &Matrix, weight: &CovWeightSpec) -> BuiltinResult<Tensor> {
    let cols = matrix.cols;
    if cols == 0 {
        return Tensor::new(Vec::new(), vec![0, 0]).map_err(cov_internal_error);
    }
    let mut result = vec![f64::NAN; cols * cols];
    for i in 0..cols {
        let variance = covariance_pair(matrix, i, i, weight);
        set_entry(&mut result, cols, i, i, sanitize_covariance(true, variance));
        for j in (i + 1)..cols {
            let value = covariance_pair(matrix, i, j, weight);
            set_entry(&mut result, cols, i, j, sanitize_covariance(false, value));
        }
    }
    Tensor::new(result, vec![cols, cols]).map_err(cov_internal_error)
}

fn covariance_unweighted_pair(
    matrix: &Matrix,
    lhs: usize,
    rhs: usize,
    means: &[f64],
    denom: f64,
) -> f64 {
    if !means[lhs].is_finite() || !means[rhs].is_finite() {
        return f64::NAN;
    }
    let mut accumulator = 0.0;
    for row in 0..matrix.rows {
        let x = matrix.get(row, lhs);
        let y = matrix.get(row, rhs);
        if !x.is_finite() || !y.is_finite() {
            return f64::NAN;
        }
        accumulator += (x - means[lhs]) * (y - means[rhs]);
    }
    accumulator / denom
}

fn covariance_weighted_pair(
    matrix: &Matrix,
    lhs: usize,
    rhs: usize,
    weights: &[f64],
    means: &[f64],
    denom: f64,
) -> f64 {
    if !means[lhs].is_finite() || !means[rhs].is_finite() {
        return f64::NAN;
    }
    let mut accumulator = 0.0;
    for (row, &weight) in weights.iter().enumerate().take(matrix.rows) {
        if weight == 0.0 {
            continue;
        }
        let x = matrix.get(row, lhs);
        let y = matrix.get(row, rhs);
        if !x.is_finite() || !y.is_finite() {
            return f64::NAN;
        }
        accumulator += weight * (x - means[lhs]) * (y - means[rhs]);
    }
    accumulator / denom
}

fn covariance_pair(matrix: &Matrix, lhs: usize, rhs: usize, weight: &CovWeightSpec) -> f64 {
    match weight {
        CovWeightSpec::Scalar(normalization) => {
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            for row in 0..matrix.rows {
                let x = matrix.get(row, lhs);
                let y = matrix.get(row, rhs);
                if x.is_finite() && y.is_finite() {
                    xs.push(x);
                    ys.push(y);
                }
            }
            covariance_unweighted_slice(&xs, &ys, *normalization)
        }
        CovWeightSpec::Vector(weights) => {
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            let mut ws = Vec::new();
            for (row, &weight) in weights.iter().enumerate().take(matrix.rows) {
                let x = matrix.get(row, lhs);
                let y = matrix.get(row, rhs);
                if x.is_finite() && y.is_finite() {
                    xs.push(x);
                    ys.push(y);
                    ws.push(weight);
                }
            }
            covariance_weighted_slice(&xs, &ys, &ws)
        }
    }
}

fn covariance_unweighted_slice(xs: &[f64], ys: &[f64], normalization: CovNormalization) -> f64 {
    if xs.is_empty() || ys.is_empty() {
        return f64::NAN;
    }
    let n = xs.len().min(ys.len());
    if n == 0 {
        return f64::NAN;
    }
    let denom = match normalization {
        CovNormalization::Unbiased => (n as f64) - 1.0,
        CovNormalization::Biased => n as f64,
    };
    if denom <= 0.0 {
        return f64::NAN;
    }
    let sum_x: f64 = xs.iter().take(n).sum();
    let sum_y: f64 = ys.iter().take(n).sum();
    let mean_x = sum_x / (n as f64);
    let mean_y = sum_y / (n as f64);
    let mut accumulator = 0.0;
    for idx in 0..n {
        accumulator += (xs[idx] - mean_x) * (ys[idx] - mean_y);
    }
    accumulator / denom
}

fn covariance_weighted_slice(xs: &[f64], ys: &[f64], weights: &[f64]) -> f64 {
    if xs.is_empty() || ys.is_empty() || weights.is_empty() {
        return f64::NAN;
    }
    let n = xs.len().min(ys.len()).min(weights.len());
    if n == 0 {
        return f64::NAN;
    }
    let sum_w: f64 = weights.iter().take(n).sum();
    if sum_w <= 0.0 {
        return f64::NAN;
    }
    let denom = sum_w - 1.0;
    if denom <= 0.0 {
        return f64::NAN;
    }
    let mut mean_x = 0.0;
    let mut mean_y = 0.0;
    for idx in 0..n {
        mean_x += weights[idx] * xs[idx];
        mean_y += weights[idx] * ys[idx];
    }
    mean_x /= sum_w;
    mean_y /= sum_w;
    let mut accumulator = 0.0;
    for idx in 0..n {
        accumulator += weights[idx] * (xs[idx] - mean_x) * (ys[idx] - mean_y);
    }
    accumulator / denom
}

fn sanitize_covariance(is_diag: bool, value: f64) -> f64 {
    if !value.is_finite() {
        return value;
    }
    if is_diag && value < 0.0 && value > -1.0e-12 {
        0.0
    } else {
        value
    }
}

fn set_entry(buffer: &mut [f64], dim: usize, row: usize, col: usize, value: f64) {
    let idx = row + col * dim;
    buffer[idx] = value;
    if row != col {
        let symmetrical = col + row * dim;
        buffer[symmetrical] = value;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::{IntegerStorage, ResolveContext, Tensor, Type};

    fn assert_tensor_close(actual: &Tensor, expected: &[f64], tol: f64) {
        let dim = (expected.len() as f64).sqrt() as usize;
        assert_eq!(actual.shape, vec![dim, dim], "unexpected tensor shape");
        for (idx, (&got, &want)) in actual
            .materialize_f64()
            .iter()
            .zip(expected.iter())
            .enumerate()
        {
            if want.is_nan() {
                assert!(
                    got.is_nan(),
                    "expected NaN at linear index {idx}, found {got}"
                );
            } else {
                assert!(
                    (got - want).abs() <= tol,
                    "mismatch at linear index {idx}: got {got}, expected {want}"
                );
            }
        }
    }

    fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>, _poison: f64) -> Tensor {
        Tensor::new_integer(storage, shape).unwrap()
    }

    #[test]
    fn cov_type_preserves_column_count() {
        let out = cov_type(
            &[Type::Tensor {
                shape: Some(vec![Some(5), Some(3)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(3), Some(3)])
            }
        );
    }

    #[test]
    fn cov_type_vector_returns_scalar() {
        let out = cov_type(
            &[Type::Tensor {
                shape: Some(vec![Some(1), Some(4)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::Num);
    }

    #[test]
    fn cov_type_paired_matrices_returns_two_by_two() {
        let out = cov_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(5), Some(3)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(5), Some(3)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(2), Some(2)])
            }
        );
    }

    #[test]
    fn cov_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = COV_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"C = cov(X)"));
        assert!(labels.contains(&"C = cov(X, normalization)"));
        assert!(labels.contains(&"C = cov(X, Y, w, opt)"));
    }

    #[cfg(feature = "wgpu")]
    fn cov_builtin_sync(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        block_on(super::cov_builtin(value, rest))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_matrix_basic() {
        let tensor = Tensor::new(
            vec![
                4.0, 4.2, 3.9, 4.3, 4.1, //
                2.0, 2.1, 2.0, 2.1, 2.2, //
                0.60, 0.59, 0.58, 0.62, 0.63,
            ],
            vec![5, 3],
        )
        .unwrap();
        let result = block_on(cov_builtin(Value::Tensor(tensor), Vec::new())).expect("cov");
        let tensor = match result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };
        let expected = [
            0.0250, 0.0075, 0.00175, //
            0.0075, 0.0070, 0.00135, //
            0.00175, 0.00135, 0.00043,
        ];
        assert_tensor_close(&tensor, &expected, 1.0e-6);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_two_vectors() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let y = Tensor::new(vec![10.0, 11.0, 9.0, 12.0], vec![4, 1]).unwrap();
        let result = block_on(cov_builtin(Value::Tensor(x), vec![Value::Tensor(y)])).expect("cov");
        let tensor = match result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };
        let expected = [
            1.6666666666666667,
            0.6666666666666666, //
            0.6666666666666666,
            1.6666666666666667,
        ];
        assert_tensor_close(&tensor, &expected, 1.0e-6);
    }

    #[test]
    fn cov_row_vector_scalar_empty_and_paired_matrix_shapes_match_contract() {
        let row = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let Value::Tensor(row_cov) =
            block_on(cov_builtin(Value::Tensor(row), Vec::new())).expect("row covariance")
        else {
            panic!("expected tensor");
        };
        assert_tensor_close(&row_cov, &[1.0], 1.0e-12);

        let scalar = Tensor::new(vec![7.0], vec![1, 1]).unwrap();
        let Value::Tensor(scalar_cov) =
            block_on(cov_builtin(Value::Tensor(scalar), Vec::new())).expect("scalar covariance")
        else {
            panic!("expected tensor");
        };
        assert_tensor_close(&scalar_cov, &[0.0], 1.0e-12);

        let empty = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let Value::Tensor(empty_cov) =
            block_on(cov_builtin(Value::Tensor(empty), Vec::new())).expect("empty covariance")
        else {
            panic!("expected tensor");
        };
        assert_tensor_close(&empty_cov, &[f64::NAN], 0.0);

        let left = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let right = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let Value::Tensor(pair_cov) =
            block_on(cov_builtin(Value::Tensor(left), vec![Value::Tensor(right)]))
                .expect("paired matrix covariance")
        else {
            panic!("expected tensor");
        };
        let expected = [5.0 / 3.0; 4];
        assert_tensor_close(&pair_cov, &expected, 1.0e-12);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_weighted_vector() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = Tensor::new(
            vec![
                4.0, 4.2, 3.9, 4.3, 4.1, //
                2.0, 2.1, 2.0, 2.1, 2.2,
            ],
            vec![5, 2],
        )
        .unwrap();
        let weights = Tensor::new(vec![1.0, 1.0, 1.0, 2.0, 2.0], vec![5, 1]).unwrap();
        let result = block_on(cov_builtin(
            Value::Tensor(tensor),
            vec![Value::Tensor(weights)],
        ))
        .expect("cov");
        let tensor = match result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };
        let expected = [
            0.022380952380952376,
            0.004999999999999994, //
            0.004999999999999994,
            0.006666666666666678,
        ];
        assert_tensor_close(&tensor, &expected, 1.0e-6);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_accepts_typed_integer_matrix_and_weights() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = poisoned_int_tensor(
            IntegerStorage::I16(vec![
                4, 4, 3, 4, 4, //
                2, 2, 2, 2, 2,
            ]),
            vec![5, 2],
            f64::NAN,
        );
        let weights = poisoned_int_tensor(
            IntegerStorage::U16(vec![1, 1, 1, 2, 2]),
            vec![5, 1],
            f64::NAN,
        );
        let result = block_on(cov_builtin(
            Value::Tensor(tensor),
            vec![Value::Tensor(weights)],
        ))
        .expect("cov");
        let tensor = match result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };
        let expected = [
            1.0 / 7.0,
            0.0, //
            0.0,
            0.0,
        ];
        assert_tensor_close(&tensor, &expected, 1.0e-6);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_rejects_negative_typed_integer_weights() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 3, 4]), vec![2, 2], 0.0);
        let weights = poisoned_int_tensor(IntegerStorage::I16(vec![1, -1]), vec![2, 1], 1.0);
        let err = block_on(cov_builtin(
            Value::Tensor(tensor),
            vec![Value::Tensor(weights)],
        ))
        .expect_err("negative weights should fail");
        assert_eq!(err.identifier(), COV_ERROR_INVALID_ARGUMENT.identifier);
    }

    #[test]
    fn cov_extensions_are_independently_gated_before_computation() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer_error = block_on(cov_builtin(
            Value::Tensor(
                Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 3]), vec![3, 1]).unwrap(),
            ),
            Vec::new(),
        ))
        .unwrap_err();
        assert_eq!(
            integer_error.identifier(),
            Some("RunMat:compatibility:CovIntegerDataExtension")
        );

        let logical_error = block_on(cov_builtin(Value::Bool(true), Vec::new())).unwrap_err();
        assert_eq!(
            logical_error.identifier(),
            Some("RunMat:compatibility:CovLogicalDataExtension")
        );

        let data = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let normalization_error = block_on(cov_builtin(
            Value::Tensor(data.clone()),
            vec![Value::Int(IntValue::U8(1))],
        ))
        .unwrap_err();
        assert_eq!(
            normalization_error.identifier(),
            Some("RunMat:compatibility:CovTypedNormalizationExtension")
        );

        let weights = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let weight_error = block_on(cov_builtin(
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0], vec![3, 2]).unwrap()),
            vec![Value::Tensor(weights)],
        ))
        .unwrap_err();
        assert_eq!(
            weight_error.identifier(),
            Some("RunMat:compatibility:CovVectorWeightsExtension")
        );
    }

    #[test]
    fn cov_supports_all_eight_integer_classes_and_wide_centering() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let storages = vec![
            IntegerStorage::I8(vec![1, 2, 3]),
            IntegerStorage::I16(vec![1, 2, 3]),
            IntegerStorage::I32(vec![1, 2, 3]),
            IntegerStorage::I64(vec![1, 2, 3]),
            IntegerStorage::U8(vec![1, 2, 3]),
            IntegerStorage::U16(vec![1, 2, 3]),
            IntegerStorage::U32(vec![1, 2, 3]),
            IntegerStorage::U64(vec![1, 2, 3]),
        ];
        for storage in storages {
            let class = storage.class_name();
            let data = Tensor::new_integer(storage.clone(), vec![3, 1]).unwrap();
            let Value::Tensor(covariance) = block_on(cov_builtin(Value::Tensor(data), Vec::new()))
                .unwrap_or_else(|error| panic!("{class}: {error}"))
            else {
                panic!("{class}: expected tensor");
            };
            assert_tensor_close(&covariance, &[1.0], 1.0e-12);

            let left = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
            let right = Tensor::new_integer(storage.clone(), vec![3, 1]).unwrap();
            let Value::Tensor(pair) =
                block_on(cov_builtin(Value::Tensor(left), vec![Value::Tensor(right)]))
                    .unwrap_or_else(|error| panic!("{class} paired data: {error}"))
            else {
                panic!("{class}: expected paired tensor");
            };
            assert_tensor_close(&pair, &[1.0; 4], 1.0e-12);

            let matrix = Tensor::new(vec![1.0, 2.0, 3.0, 3.0, 2.0, 1.0], vec![3, 2]).unwrap();
            let weights = Tensor::new_integer(storage, vec![3, 1]).unwrap();
            let weighted = block_on(cov_builtin(
                Value::Tensor(matrix),
                vec![Value::Tensor(weights)],
            ))
            .unwrap_or_else(|error| panic!("{class} weights: {error}"));
            assert!(matches!(weighted, Value::Tensor(_)));
        }

        let base = 1_u64 << 63;
        let wide = Tensor::new_integer(
            IntegerStorage::U64(vec![base, base + 1, base + 2]),
            vec![3, 1],
        )
        .unwrap();
        let Value::Tensor(covariance) =
            block_on(cov_builtin(Value::Tensor(wide), Vec::new())).expect("wide covariance")
        else {
            panic!("expected tensor");
        };
        assert_tensor_close(&covariance, &[1.0], 1.0e-12);
    }

    #[test]
    fn cov_resident_integer_data_rejects_before_provider_dispatch() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new_integer(IntegerStorage::U16(vec![1, 2, 3]), vec![3, 1]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload integer");
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error =
                block_on(cov_builtin(Value::GpuTensor(handle.clone()), Vec::new())).unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:CovIntegerDataExtension")
            );
            let _ = provider.free(&handle);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_omitrows() {
        let tensor = Tensor::new(
            vec![
                1.0,
                3.0,
                f64::NAN,
                8.0, //
                f64::NAN,
                4.0,
                6.0,
                9.0, //
                2.0,
                5.0,
                7.0,
                10.0,
            ],
            vec![4, 3],
        )
        .unwrap();
        let result = block_on(cov_builtin(
            Value::Tensor(tensor),
            vec![Value::from("omitrows")],
        ))
        .expect("cov");
        let tensor = match result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };
        let expected = [
            12.5, 12.5, 12.5, //
            12.5, 12.5, 12.5, //
            12.5, 12.5, 12.5,
        ];
        assert_tensor_close(&tensor, &expected, 1.0e-6);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_partialrows() {
        let tensor = Tensor::new(
            vec![
                1.0,
                4.0,
                7.0, //
                2.0,
                f64::NAN,
                8.0, //
                f64::NAN,
                6.0,
                9.0,
            ],
            vec![3, 3],
        )
        .unwrap();
        let result = block_on(cov_builtin(
            Value::Tensor(tensor),
            vec![Value::from("partialrows")],
        ))
        .expect("cov");
        let tensor = match result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };
        let expected = [
            9.0,
            18.0,
            4.5, //
            18.0,
            18.0,
            f64::NAN, //
            4.5,
            f64::NAN,
            4.5,
        ];
        assert_tensor_close(&tensor, &expected, 1.0e-6);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_mismatched_rows_errors() {
        let left = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let right = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = block_on(cov_builtin(Value::Tensor(left), vec![Value::Tensor(right)]))
            .expect_err("expected mismatch error");
        assert_eq!(err.identifier(), COV_ERROR_ROWS_MISMATCH.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_invalid_flag_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = block_on(cov_builtin(Value::Tensor(tensor), vec![Value::Num(2.5)]))
            .expect_err("expected invalid flag error");
        assert_eq!(err.identifier(), COV_ERROR_NORMALIZATION_INVALID.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_weight_vector_length_mismatch_errors() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
        let y = Tensor::new(vec![10.0, 11.0, 12.0], vec![3, 1]).unwrap();
        let w = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = block_on(cov_builtin(
            Value::Tensor(x),
            vec![Value::Tensor(y), Value::Tensor(w)],
        ))
        .expect_err("expected weight length mismatch");
        assert_eq!(
            err.identifier(),
            COV_ERROR_WEIGHT_VECTOR_LENGTH_MISMATCH.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_unknown_rows_option_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = block_on(cov_builtin(
            Value::Tensor(tensor),
            vec![Value::from("rows"), Value::from("bogus")],
        ))
        .expect_err("expected unknown rows option error");
        assert_eq!(err.identifier(), COV_ERROR_ROWS_OPTION_UNKNOWN.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_duplicate_normalization_flag_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = block_on(cov_builtin(
            Value::Tensor(tensor),
            vec![Value::Num(0.0), Value::Num(1.0)],
        ))
        .expect_err("expected duplicate normalization flag error");
        assert_eq!(
            err.identifier(),
            COV_ERROR_NORMALIZATION_DUPLICATE.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_too_many_array_arguments_errors() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let y = Tensor::new(vec![4.0, 5.0, 6.0], vec![3, 1]).unwrap();
        let w = Tensor::new(vec![1.0, 1.0, 1.0], vec![3, 1]).unwrap();
        let z = Tensor::new(vec![7.0, 8.0, 9.0], vec![3, 1]).unwrap();
        let err = block_on(cov_builtin(
            Value::Tensor(x),
            vec![Value::Tensor(y), Value::Tensor(w), Value::Tensor(z)],
        ))
        .expect_err("expected too many array arguments error");
        assert_eq!(
            err.identifier(),
            COV_ERROR_TOO_MANY_ARRAY_ARGUMENTS.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(
                vec![
                    4.0, 4.2, 3.9, 4.3, 4.1, //
                    2.0, 2.1, 2.0, 2.1, 2.2,
                ],
                vec![5, 2],
            )
            .unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result = block_on(cov_builtin(Value::GpuTensor(handle), Vec::new())).expect("cov");
            let gathered = test_support::gather(result).expect("gather");
            let expected = [
                0.0250, 0.0075, //
                0.0075, 0.0070,
            ];
            assert_tensor_close(&gathered, &expected, 1.0e-6);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_gpu_host_weights_return_resident_result() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(
                vec![
                    4.0, 4.2, 3.9, 4.3, 4.1, //
                    2.0, 2.1, 2.0, 2.1, 2.2,
                ],
                vec![5, 2],
            )
            .unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let weights = Tensor::new(vec![1.0, 1.0, 1.0, 2.0, 2.0], vec![5, 1]).unwrap();

            let result = block_on(cov_builtin(
                Value::GpuTensor(handle),
                vec![Value::Tensor(weights)],
            ))
            .expect("weighted cov");
            assert!(matches!(result, Value::GpuTensor(_)));
            let gathered = test_support::gather(result).expect("gather");
            let expected = [
                0.022380952380952376,
                0.004999999999999994, //
                0.004999999999999994,
                0.006666666666666678,
            ];
            assert_tensor_close(&gathered, &expected, 1.0e-6);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_gpu_resident_weights_return_resident_result() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(
                vec![
                    4.0, 4.2, 3.9, 4.3, 4.1, //
                    2.0, 2.1, 2.0, 2.1, 2.2,
                ],
                vec![5, 2],
            )
            .unwrap();
            let weights = Tensor::new(vec![1.0, 1.0, 1.0, 2.0, 2.0], vec![1, 5]).unwrap();
            let data = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &tensor.materialize_f64(),
                    shape: &tensor.shape,
                })
                .expect("upload data");
            let weight_handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &weights.materialize_f64(),
                    shape: &weights.shape,
                })
                .expect("upload weights");

            let result = block_on(cov_builtin(
                Value::GpuTensor(data),
                vec![Value::GpuTensor(weight_handle)],
            ))
            .expect("weighted cov");
            assert!(matches!(result, Value::GpuTensor(_)));
            let gathered = test_support::gather(result).expect("gather");
            let expected = [
                0.022380952380952376,
                0.004999999999999994, //
                0.004999999999999994,
                0.006666666666666678,
            ];
            assert_tensor_close(&gathered, &expected, 1.0e-6);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_gpu_rejects_negative_resident_weights() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
            let weights = Tensor::new(vec![1.0, -1.0], vec![2, 1]).unwrap();
            let data = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &tensor.materialize_f64(),
                    shape: &tensor.shape,
                })
                .expect("upload data");
            let weight_handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &weights.materialize_f64(),
                    shape: &weights.shape,
                })
                .expect("upload weights");

            let err = block_on(cov_builtin(
                Value::GpuTensor(data),
                vec![Value::GpuTensor(weight_handle)],
            ))
            .expect_err("negative weights should fail");
            assert_eq!(err.identifier(), COV_ERROR_INVALID_ARGUMENT.identifier);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cov_wgpu_matches_cpu() {
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };

        let tensor = Tensor::new(
            vec![
                4.0, 4.2, 3.9, 4.3, 4.1, //
                2.0, 2.1, 2.0, 2.1, 2.2,
            ],
            vec![5, 2],
        )
        .unwrap();

        let cpu_result =
            block_on(cov_builtin(Value::Tensor(tensor.clone()), Vec::new())).expect("cov");
        let cpu_tensor = match cpu_result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };

        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");

        let gpu_value = cov_builtin_sync(Value::GpuTensor(handle), Vec::new()).expect("cov");
        let gathered = test_support::gather(gpu_value).expect("gather");

        assert_tensor_close(&gathered, &cpu_tensor.materialize_f64(), 1.0e-6);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cov_wgpu_paired_matrices_vectorize_to_two_variables() {
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let left = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let right = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let left_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &left.materialize_f64(),
                shape: &left.shape,
            })
            .expect("upload left");
        let right_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &right.materialize_f64(),
                shape: &right.shape,
            })
            .expect("upload right");
        let result = cov_builtin_sync(
            Value::GpuTensor(left_handle),
            vec![Value::GpuTensor(right_handle)],
        )
        .expect("paired covariance");
        assert!(matches!(result, Value::GpuTensor(_)));
        let gathered = test_support::gather(result).expect("gather");
        assert_tensor_close(&gathered, &[5.0 / 3.0; 4], 1.0e-5);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cov_wgpu_row_scalar_and_empty_match_host_shapes() {
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };

        for (data, shape, expected) in [
            (vec![1.0, 2.0, 3.0], vec![1, 3], 1.0),
            (vec![7.0], Vec::new(), 0.0),
            (Vec::new(), vec![0, 0], f64::NAN),
        ] {
            let handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &data,
                    shape: &shape,
                })
                .expect("upload");
            let result =
                cov_builtin_sync(Value::GpuTensor(handle), Vec::new()).expect("covariance");
            assert!(matches!(result, Value::GpuTensor(_)));
            let gathered = test_support::gather(result).expect("gather");
            assert_tensor_close(&gathered, &[expected], 1.0e-6);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn cov_wgpu_weighted_matches_cpu_and_stays_resident() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };

        let tensor = Tensor::new(
            vec![
                4.0, 4.2, 3.9, 4.3, 4.1, //
                2.0, 2.1, 2.0, 2.1, 2.2,
            ],
            vec![5, 2],
        )
        .unwrap();
        let weights = Tensor::new(vec![1.0, 1.0, 1.0, 2.0, 2.0], vec![1, 5]).unwrap();

        let cpu_result = block_on(cov_builtin(
            Value::Tensor(tensor.clone()),
            vec![Value::Tensor(weights.clone())],
        ))
        .expect("cov");
        let cpu_tensor = match cpu_result {
            Value::Tensor(t) => t,
            other => panic!("expected tensor result, got {other:?}"),
        };

        let data_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            })
            .expect("upload data");
        let weight_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &weights.materialize_f64(),
                shape: &weights.shape,
            })
            .expect("upload weights");

        let gpu_value = cov_builtin_sync(
            Value::GpuTensor(data_handle),
            vec![Value::GpuTensor(weight_handle)],
        )
        .expect("weighted cov");
        assert!(matches!(gpu_value, Value::GpuTensor(_)));
        let gathered = test_support::gather(gpu_value).expect("gather");

        assert_tensor_close(&gathered, &cpu_tensor.materialize_f64(), 1.0e-5);
    }
}
