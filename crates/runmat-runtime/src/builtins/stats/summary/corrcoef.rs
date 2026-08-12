//! MATLAB-compatible `corrcoef` builtin with GPU-aware semantics for RunMat.

use num_complex::Complex64;
use runmat_accelerate_api::{
    CorrcoefNormalization, CorrcoefOptions, CorrcoefRows, GpuTensorHandle, GpuTensorStorage,
};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexTensor, IntValue, Tensor, Value};

use crate::builtins::common::gpu_helpers;
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::tensor::{self, value_to_string};
use crate::builtins::stats::summary::distribution_math::{
    standard_normal_inv, student_t_cdf_upper,
};
use crate::builtins::stats::type_resolvers::corrcoef_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const NAME: &str = "corrcoef";
const OUTPUT_R: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Correlation coefficient matrix.",
};
const OUTPUT_P: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "P",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "P-values for testing zero correlation.",
};
const OUTPUT_RL: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "RL",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Lower confidence bounds for the correlation coefficients.",
};
const OUTPUT_RU: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "RU",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Upper confidence bounds for the correlation coefficients.",
};
const OUTPUTS_R: [BuiltinParamDescriptor; 1] = [OUTPUT_R];
const OUTPUTS_R_P: [BuiltinParamDescriptor; 2] = [OUTPUT_R, OUTPUT_P];
const OUTPUTS_FULL: [BuiltinParamDescriptor; 4] = [OUTPUT_R, OUTPUT_P, OUTPUT_RL, OUTPUT_RU];

const PARAM_A: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "A",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Input observations (rows are observations, columns are variables).",
};
const PARAM_B: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "B",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Second array with the same size as A; both arrays are vectorized.",
};
const PARAM_OPTIONS: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "nameValuePairs",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Variadic,
    default: None,
    description: "Alpha and Rows name-value arguments.",
};
const INPUTS_A: [BuiltinParamDescriptor; 1] = [PARAM_A];
const INPUTS_A_B: [BuiltinParamDescriptor; 2] = [PARAM_A, PARAM_B];
const INPUTS_A_OPTIONS: [BuiltinParamDescriptor; 2] = [PARAM_A, PARAM_OPTIONS];
const INPUTS_A_B_OPTIONS: [BuiltinParamDescriptor; 3] = [PARAM_A, PARAM_B, PARAM_OPTIONS];

const CORRCOEF_SIGNATURES: [BuiltinSignatureDescriptor; 8] = [
    BuiltinSignatureDescriptor {
        label: "R = corrcoef(A)",
        inputs: &INPUTS_A,
        outputs: &OUTPUTS_R,
    },
    BuiltinSignatureDescriptor {
        label: "R = corrcoef(A, B)",
        inputs: &INPUTS_A_B,
        outputs: &OUTPUTS_R,
    },
    BuiltinSignatureDescriptor {
        label: "[R, P] = corrcoef(A)",
        inputs: &INPUTS_A,
        outputs: &OUTPUTS_R_P,
    },
    BuiltinSignatureDescriptor {
        label: "[R, P] = corrcoef(A, B)",
        inputs: &INPUTS_A_B,
        outputs: &OUTPUTS_R_P,
    },
    BuiltinSignatureDescriptor {
        label: "[R, P, RL, RU] = corrcoef(A)",
        inputs: &INPUTS_A,
        outputs: &OUTPUTS_FULL,
    },
    BuiltinSignatureDescriptor {
        label: "[R, P, RL, RU] = corrcoef(A, B)",
        inputs: &INPUTS_A_B,
        outputs: &OUTPUTS_FULL,
    },
    BuiltinSignatureDescriptor {
        label: "___ = corrcoef(A, Name, Value)",
        inputs: &INPUTS_A_OPTIONS,
        outputs: &OUTPUTS_FULL,
    },
    BuiltinSignatureDescriptor {
        label: "___ = corrcoef(A, B, Name, Value)",
        inputs: &INPUTS_A_B_OPTIONS,
        outputs: &OUTPUTS_FULL,
    },
];

const CORRCOEF_ERROR_INVALID_ARGUMENT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CORRCOEF.INVALID_ARGUMENT",
    identifier: Some("RunMat:corrcoef:InvalidArgument"),
    when: "Arguments are malformed or unsupported for corrcoef.",
    message: "corrcoef: invalid argument",
};

const CORRCOEF_ERROR_COMPLEX_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CORRCOEF.COMPLEX_OUTPUTS",
    identifier: Some("RunMat:corrcoef:ComplexOutputs"),
    when: "P-values or confidence bounds are requested for complex correlation coefficients.",
    message: "corrcoef: P-values and confidence bounds are invalid for complex coefficients",
};

const CORRCOEF_ERROR_SIZE_MISMATCH: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CORRCOEF.SIZE_MISMATCH",
    identifier: Some("RunMat:corrcoef:SizeMismatch"),
    when: "A and B do not have the same size.",
    message: "corrcoef: A and B must have the same size",
};

const CORRCOEF_ERROR_ALPHA_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CORRCOEF.ALPHA_INVALID",
    identifier: Some("RunMat:corrcoef:AlphaInvalid"),
    when: "Alpha is not a finite single/double scalar strictly between zero and one.",
    message: "corrcoef: Alpha must be a floating scalar in the open interval (0,1)",
};

const CORRCOEF_ERROR_ROWS_OPTION_UNKNOWN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CORRCOEF.ROWS_OPTION_UNKNOWN",
    identifier: Some("RunMat:corrcoef:RowsOptionUnknown"),
    when: "Rows option value is not supported.",
    message: "corrcoef: unknown rows option",
};

const CORRCOEF_ERROR_OPTION_DUPLICATE: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CORRCOEF.OPTION_DUPLICATE",
    identifier: Some("RunMat:corrcoef:OptionDuplicate"),
    when: "Alpha or Rows is specified more than once.",
    message: "corrcoef: option specified more than once",
};

const CORRCOEF_ERROR_ROWS_OPTION_MALFORMED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CORRCOEF.ROWS_OPTION_MALFORMED",
    identifier: Some("RunMat:corrcoef:RowsOptionMalformed"),
    when: "Rows keyword is not followed by a valid string option.",
    message: "corrcoef: rows option is malformed",
};

const CORRCOEF_ERROR_OPTION_UNKNOWN: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CORRCOEF.OPTION_UNKNOWN",
    identifier: Some("RunMat:corrcoef:OptionUnknown"),
    when: "An unknown option keyword is provided.",
    message: "corrcoef: unknown option",
};

const CORRCOEF_ERROR_TOO_MANY_INPUT_ARRAYS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CORRCOEF.TOO_MANY_INPUT_ARRAYS",
    identifier: Some("RunMat:corrcoef:TooManyInputArrays"),
    when: "More than two data arrays are provided.",
    message: "corrcoef: too many input arrays",
};

const CORRCOEF_ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.CORRCOEF.INTERNAL",
    identifier: Some("RunMat:corrcoef:Internal"),
    when: "Internal tensor conversion/allocation or corrcoef computation fails.",
    message: "corrcoef: internal operation failed",
};

const CORRCOEF_ERRORS: [BuiltinErrorDescriptor; 10] = [
    CORRCOEF_ERROR_INVALID_ARGUMENT,
    CORRCOEF_ERROR_COMPLEX_OUTPUTS,
    CORRCOEF_ERROR_SIZE_MISMATCH,
    CORRCOEF_ERROR_ALPHA_INVALID,
    CORRCOEF_ERROR_ROWS_OPTION_UNKNOWN,
    CORRCOEF_ERROR_OPTION_DUPLICATE,
    CORRCOEF_ERROR_ROWS_OPTION_MALFORMED,
    CORRCOEF_ERROR_OPTION_UNKNOWN,
    CORRCOEF_ERROR_TOO_MANY_INPUT_ARRAYS,
    CORRCOEF_ERROR_INTERNAL,
];

const CORRCOEF_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "corrcoef-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "corrcoef with typed-integer observation data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CorrcoefIntegerDataExtension"),
};

const CORRCOEF_LOGICAL_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "corrcoef-logical-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "corrcoef with logical observation data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CorrcoefLogicalDataExtension"),
};

const CORRCOEF_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    CORRCOEF_INTEGER_DATA_EXTENSION,
    CORRCOEF_LOGICAL_DATA_EXTENSION,
];

const CORRCOEF_INTEGER_DATA_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "A_or_B",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented real or complex observation domain is single or double; RunMat mode additionally accepts all eight real and componentwise-complex integer classes.",
    }];

const CORRCOEF_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "___ = corrcoef(integer_A_or_B, Name, Value)",
        inputs: &CORRCOEF_INTEGER_DATA_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat preserves exact integer component differences while centering each variable before producing real or complex double correlation coefficients.",
    }];

pub const CORRCOEF_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &CORRCOEF_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &CORRCOEF_ERRORS,
};

fn corrcoef_error_with(
    error: &'static BuiltinErrorDescriptor,
    message: impl Into<String>,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn corrcoef_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    corrcoef_error_with(error, error.message)
}

fn corrcoef_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl std::fmt::Display,
) -> RuntimeError {
    corrcoef_error_with(error, format!("{}: {detail}", error.message))
}

fn corrcoef_internal_error(message: impl Into<String>) -> RuntimeError {
    corrcoef_error_with(&CORRCOEF_ERROR_INTERNAL, message)
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::stats::summary::corrcoef")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "corrcoef",
    op_kind: GpuOpKind::Custom("summary-stats"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("corrcoef")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Uses provider-side corrcoef kernels when rows='all'; other cases fall back to host execution.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::stats::summary::corrcoef")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "corrcoef",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: true,
    notes: "Fusion planner treats corrcoef as a non-fusible boundary; GPU execution is provided via a custom provider hook.",
};

#[runtime_builtin(
    name = "corrcoef",
    category = "stats/summary",
    summary = "Compute Pearson correlation coefficients.",
    keywords = "corrcoef,correlation,statistics,rows,alpha,p-value,confidence,gpu",
    accel = "reduction",
    type_resolver(corrcoef_type),
    descriptor(crate::builtins::stats::summary::corrcoef::CORRCOEF_DESCRIPTOR),
    extensions(CORRCOEF_EXTENSIONS),
    integer_capabilities(CORRCOEF_INTEGER_CAPABILITIES),
    builtin_path = "crate::builtins::stats::summary::corrcoef"
)]
async fn corrcoef_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let args = CorrcoefArgs::parse(value, rest)?;
    ensure_corrcoef_extensions(&args)?;
    let requested_outputs = crate::output_count::current_output_count();
    if requested_outputs == Some(0) {
        return Ok(Value::OutputList(Vec::new()));
    }
    if requested_outputs.is_none() || requested_outputs == Some(1) {
        if let Some(result) = corrcoef_try_gpu(&args).await? {
            return Ok(match requested_outputs {
                Some(1) => Value::OutputList(vec![result]),
                _ => result,
            });
        }
    }
    corrcoef_host(args, requested_outputs).await
}

/// Exposed for acceleration providers that need the host reference implementation.
pub fn corrcoef_from_tensors(
    left: Tensor,
    right: Option<Tensor>,
    _normalization: CorrcoefNormalization,
    rows: CorrcoefRows,
) -> BuiltinResult<Tensor> {
    match evaluate_inputs(
        NumericInput::Real(left),
        right.map(NumericInput::Real),
        rows,
    )? {
        CorrcoefEvaluation::Real(evaluation) => Ok(evaluation.r),
        CorrcoefEvaluation::Complex(_) => {
            unreachable!("real tensor inputs cannot produce a complex result")
        }
    }
}

#[derive(Debug)]
struct CorrcoefArgs {
    first: Value,
    second: Option<Value>,
    rows: CorrcoefRows,
    alpha: f64,
}

impl CorrcoefArgs {
    fn parse(first: Value, rest: Vec<Value>) -> BuiltinResult<Self> {
        let mut values = rest.into_iter();
        let mut second = None;
        let mut rows = CorrcoefRows::All;
        let mut alpha = 0.05;
        let mut rows_seen = false;
        let mut alpha_seen = false;
        let mut pending = values.next();

        if pending.as_ref().is_some_and(|value| !is_text_value(value)) {
            second = pending.take();
            pending = values.next();
        }

        while let Some(name) = pending {
            let key = value_to_string(&name)
                .ok_or_else(|| {
                    if second.is_some() {
                        corrcoef_error(&CORRCOEF_ERROR_TOO_MANY_INPUT_ARRAYS)
                    } else {
                        corrcoef_error(&CORRCOEF_ERROR_INVALID_ARGUMENT)
                    }
                })?
                .trim()
                .to_ascii_lowercase();
            let option = values.next().ok_or_else(|| {
                corrcoef_error_with_detail(
                    &CORRCOEF_ERROR_ROWS_OPTION_MALFORMED,
                    format!("option '{key}' requires a value"),
                )
            })?;
            match key.as_str() {
                "rows" => {
                    if rows_seen {
                        return Err(corrcoef_error_with_detail(
                            &CORRCOEF_ERROR_OPTION_DUPLICATE,
                            "'Rows'",
                        ));
                    }
                    let choice = value_to_string(&option).ok_or_else(|| {
                        corrcoef_error_with_detail(
                            &CORRCOEF_ERROR_ROWS_OPTION_MALFORMED,
                            "Rows must be a string value",
                        )
                    })?;
                    rows = parse_rows_option(choice.trim())?;
                    rows_seen = true;
                }
                "alpha" => {
                    if alpha_seen {
                        return Err(corrcoef_error_with_detail(
                            &CORRCOEF_ERROR_OPTION_DUPLICATE,
                            "'Alpha'",
                        ));
                    }
                    alpha = parse_alpha(&option)?;
                    alpha_seen = true;
                }
                _ => {
                    return Err(corrcoef_error_with_detail(
                        &CORRCOEF_ERROR_OPTION_UNKNOWN,
                        format!("'{key}'"),
                    ))
                }
            }
            pending = values.next();
        }

        Ok(Self {
            first,
            second,
            rows,
            alpha,
        })
    }
}

fn is_text_value(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    )
}

fn is_typed_integer_value(value: &Value) -> bool {
    matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(
            value,
            Value::ComplexTensor(tensor) if tensor.integer_storage().is_some()
        )
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

fn ensure_corrcoef_extensions(args: &CorrcoefArgs) -> BuiltinResult<()> {
    if is_typed_integer_value(&args.first)
        || args.second.as_ref().is_some_and(is_typed_integer_value)
    {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CORRCOEF_INTEGER_DATA_EXTENSION,
            NAME,
        )?;
    }
    if is_logical_value(&args.first) || args.second.as_ref().is_some_and(is_logical_value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &CORRCOEF_LOGICAL_DATA_EXTENSION,
            NAME,
        )?;
    }
    Ok(())
}

async fn corrcoef_try_gpu(args: &CorrcoefArgs) -> BuiltinResult<Option<Value>> {
    if args.rows != CorrcoefRows::All {
        return Ok(None);
    }
    let first_handle = match &args.first {
        Value::GpuTensor(handle) => handle,
        _ => return Ok(None),
    };
    if first_handle.shape.len() > 2 {
        return Err(corrcoef_error_with_detail(
            &CORRCOEF_ERROR_INVALID_ARGUMENT,
            "A must be a matrix",
        ));
    }
    if runmat_accelerate_api::handle_integer_type(first_handle).is_some()
        || runmat_accelerate_api::handle_is_logical(first_handle)
        || runmat_accelerate_api::handle_storage(first_handle)
            == GpuTensorStorage::ComplexInterleaved
    {
        return Ok(None);
    }
    let provider = match runmat_accelerate_api::provider_for_handle(first_handle)
        .or_else(runmat_accelerate_api::provider)
    {
        Some(provider) => provider,
        None => return Ok(None),
    };
    let maybe_second_handle = match &args.second {
        Some(Value::GpuTensor(handle)) => {
            if !same_size(&first_handle.shape, &handle.shape) {
                return Err(corrcoef_error(&CORRCOEF_ERROR_SIZE_MISMATCH));
            }
            if runmat_accelerate_api::handle_integer_type(handle).is_some()
                || runmat_accelerate_api::handle_is_logical(handle)
                || runmat_accelerate_api::handle_storage(handle)
                    == GpuTensorStorage::ComplexInterleaved
            {
                return Ok(None);
            }
            let Some(second_provider) = runmat_accelerate_api::provider_for_handle(handle)
                .or_else(runmat_accelerate_api::provider)
            else {
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

    let mut owned_concat: Option<GpuTensorHandle> = None;
    let matrix_handle = if let Some(second) = maybe_second_handle {
        let rows = first_handle.shape.iter().copied().product::<usize>();
        let left = match provider.reshape(first_handle, &[rows, 1]) {
            Ok(handle) => handle,
            Err(_) => return Ok(None),
        };
        let right = match provider.reshape(second, &[rows, 1]) {
            Ok(handle) => handle,
            Err(_) => {
                release_gpu_reshape(provider, first_handle, &left);
                return Ok(None);
            }
        };
        let result = provider.cat(2, &[left.clone(), right.clone()]);
        release_gpu_reshape(provider, first_handle, &left);
        release_gpu_reshape(provider, second, &right);
        match result {
            Ok(concat) => {
                owned_concat = Some(concat.clone());
                concat
            }
            Err(_) => return Ok(None),
        }
    } else {
        first_handle.clone()
    };

    let options = CorrcoefOptions {
        normalization: CorrcoefNormalization::Unbiased,
        rows: args.rows,
    };

    match provider.corrcoef(&matrix_handle, &options).await {
        Ok(result) => {
            if let Some(temp) = owned_concat {
                let _ = provider.free(&temp);
            }
            Ok(Some(Value::GpuTensor(result)))
        }
        Err(_) => {
            if let Some(temp) = owned_concat {
                let _ = provider.free(&temp);
            }
            Ok(None)
        }
    }
}

fn release_gpu_reshape(
    provider: &dyn runmat_accelerate_api::AccelProvider,
    original: &GpuTensorHandle,
    reshaped: &GpuTensorHandle,
) {
    if reshaped.buffer_id == original.buffer_id {
        let _ = provider.reshape(reshaped, &original.shape);
    } else {
        let _ = provider.free(reshaped);
    }
}

async fn corrcoef_host(
    args: CorrcoefArgs,
    requested_outputs: Option<usize>,
) -> BuiltinResult<Value> {
    let CorrcoefArgs {
        first,
        second,
        rows,
        alpha,
    } = args;
    let left = value_to_numeric_input_gather(first).await?;
    let right = match second {
        Some(value) => Some(value_to_numeric_input_gather(value).await?),
        None => None,
    };
    let evaluation = evaluate_inputs(left, right, rows)?;
    evaluation.outputs(requested_outputs, alpha)
}

async fn value_to_numeric_input_gather(value: Value) -> BuiltinResult<NumericInput> {
    let gathered = match value {
        Value::GpuTensor(handle) => Value::Tensor(gpu_helpers::gather_tensor_async(&handle).await?),
        other => other,
    };
    match gathered {
        Value::Complex(real, imag) => ComplexTensor::new(vec![(real, imag)], vec![1, 1])
            .map(NumericInput::Complex)
            .map_err(corrcoef_internal_error),
        Value::ComplexTensor(tensor) => Ok(NumericInput::Complex(tensor)),
        other => tensor::value_into_tensor_for("corrcoef", other)
            .map(NumericInput::Real)
            .map_err(corrcoef_internal_error),
    }
}

fn parse_rows_option(value: &str) -> BuiltinResult<CorrcoefRows> {
    match value {
        "all" => Ok(CorrcoefRows::All),
        "complete" | "completecase" | "completecases" => Ok(CorrcoefRows::Complete),
        "pairwise" | "pairwisecomplete" | "pairwisecompletecase" | "pairwisecompletecases" => {
            Ok(CorrcoefRows::Pairwise)
        }
        other => Err(corrcoef_error_with_detail(
            &CORRCOEF_ERROR_ROWS_OPTION_UNKNOWN,
            format!("'{other}'"),
        )),
    }
}

fn parse_alpha(value: &Value) -> BuiltinResult<f64> {
    match value {
        Value::Num(value) if value.is_finite() && *value > 0.0 && *value < 1.0 => Ok(*value),
        Value::Tensor(tensor)
            if tensor.len() == 1
                && tensor.integer_storage().is_none()
                && matches!(
                    tensor.numeric_dtype(),
                    runmat_value::NumericDType::F32 | runmat_value::NumericDType::F64
                ) =>
        {
            let value = tensor.materialize_f64()[0];
            if value.is_finite() && value > 0.0 && value < 1.0 {
                Ok(value)
            } else {
                Err(corrcoef_error(&CORRCOEF_ERROR_ALPHA_INVALID))
            }
        }
        _ => Err(corrcoef_error(&CORRCOEF_ERROR_ALPHA_INVALID)),
    }
}

#[derive(Debug)]
enum NumericInput {
    Real(Tensor),
    Complex(ComplexTensor),
}

impl NumericInput {
    fn shape(&self) -> &[usize] {
        match self {
            Self::Real(tensor) => &tensor.shape,
            Self::Complex(tensor) => &tensor.shape,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Real(tensor) => tensor.len(),
            Self::Complex(tensor) => tensor.len(),
        }
    }

    fn is_complex(&self) -> bool {
        matches!(self, Self::Complex(_))
    }
}

#[derive(Debug)]
enum CorrcoefEvaluation {
    Real(RealEvaluation),
    Complex(ComplexTensor),
}

#[derive(Debug)]
struct RealEvaluation {
    r: Tensor,
    counts: Vec<usize>,
}

impl CorrcoefEvaluation {
    fn outputs(self, requested_outputs: Option<usize>, alpha: f64) -> BuiltinResult<Value> {
        match self {
            Self::Complex(r) => match requested_outputs {
                None => Ok(Value::ComplexTensor(r)),
                Some(1) => Ok(Value::OutputList(vec![Value::ComplexTensor(r)])),
                Some(_) => Err(corrcoef_error(&CORRCOEF_ERROR_COMPLEX_OUTPUTS)),
            },
            Self::Real(evaluation) => evaluation.outputs(requested_outputs, alpha),
        }
    }
}

impl RealEvaluation {
    fn outputs(self, requested_outputs: Option<usize>, alpha: f64) -> BuiltinResult<Value> {
        let RealEvaluation { r, counts } = self;
        let Some(count) = requested_outputs else {
            return Ok(Value::Tensor(r));
        };
        if count > 4 {
            return Err(corrcoef_error_with_detail(
                &CORRCOEF_ERROR_INVALID_ARGUMENT,
                "too many output arguments; maximum is 4",
            ));
        }
        let mut outputs = Vec::with_capacity(count);
        if count >= 1 {
            outputs.push(Value::Tensor(r.clone()));
        }
        if count >= 2 {
            outputs.push(Value::Tensor(p_values(&r, &counts)?));
        }
        if count >= 3 {
            let (lower, upper) = confidence_bounds(&r, &counts, alpha)?;
            outputs.push(Value::Tensor(lower));
            if count >= 4 {
                outputs.push(Value::Tensor(upper));
            }
        }
        Ok(Value::OutputList(outputs))
    }
}

fn evaluate_inputs(
    left: NumericInput,
    right: Option<NumericInput>,
    rows: CorrcoefRows,
) -> BuiltinResult<CorrcoefEvaluation> {
    if left.shape().len() > 2 {
        return Err(corrcoef_error_with_detail(
            &CORRCOEF_ERROR_INVALID_ARGUMENT,
            "A must be a matrix",
        ));
    }
    if let Some(right) = right {
        if !same_size(left.shape(), right.shape()) {
            return Err(corrcoef_error(&CORRCOEF_ERROR_SIZE_MISMATCH));
        }
        if left.is_complex() || right.is_complex() {
            let matrix = ComplexMatrix::from_pair(left, right)?;
            return evaluate_complex_matrix(matrix, rows).map(CorrcoefEvaluation::Complex);
        }
        let (NumericInput::Real(left), NumericInput::Real(right)) = (left, right) else {
            unreachable!("complex inputs handled above")
        };
        evaluate_real_matrix(Matrix::from_pair(left, right), rows).map(CorrcoefEvaluation::Real)
    } else {
        match left {
            NumericInput::Real(tensor) => evaluate_real_matrix(Matrix::from_single(tensor), rows)
                .map(CorrcoefEvaluation::Real),
            NumericInput::Complex(tensor) => {
                evaluate_complex_matrix(ComplexMatrix::from_single(tensor), rows)
                    .map(CorrcoefEvaluation::Complex)
            }
        }
    }
}

fn same_size(left: &[usize], right: &[usize]) -> bool {
    canonical_shape(left) == canonical_shape(right)
}

fn canonical_shape(shape: &[usize]) -> Vec<usize> {
    let mut canonical = shape.to_vec();
    while canonical.len() > 2 && canonical.last() == Some(&1) {
        canonical.pop();
    }
    while canonical.len() < 2 {
        canonical.push(1);
    }
    canonical
}

#[derive(Debug, Clone)]
struct Matrix {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    fn from_single(tensor: Tensor) -> Self {
        let (rows, cols) = single_geometry(&tensor.shape, tensor.len());
        Self {
            data: centered_tensor_values(&tensor, rows, cols),
            rows,
            cols,
        }
    }

    fn from_pair(left: Tensor, right: Tensor) -> Self {
        let rows = left.len();
        if rows == 1 {
            return Self {
                data: centered_scalar_pair(&left, &right),
                rows: 2,
                cols: 1,
            };
        }
        let mut data = centered_tensor_values(&left, rows, 1);
        data.extend(centered_tensor_values(&right, rows, 1));
        Self {
            data,
            rows,
            cols: 2,
        }
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row + col * self.rows]
    }
}

fn centered_scalar_pair(left: &Tensor, right: &Tensor) -> Vec<f64> {
    match (left.integer_storage(), right.integer_storage()) {
        (Some(left), Some(right)) => {
            let left = left
                .value_at(0)
                .map(|value| int_value_to_i128(&value))
                .unwrap_or(0);
            let right = right
                .value_at(0)
                .map(|value| int_value_to_i128(&value))
                .unwrap_or(left);
            vec![0.0, (right - left) as f64]
        }
        _ => vec![left.materialize_f64()[0], right.materialize_f64()[0]],
    }
}

fn single_geometry(shape: &[usize], len: usize) -> (usize, usize) {
    let (rows, cols) = match shape {
        [] => (1, 1),
        [count] => (1, *count),
        [rows, cols, ..] => (*rows, *cols),
    };
    if rows == 1 || cols == 1 {
        (len, 1)
    } else {
        (rows, cols)
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

fn evaluate_real_matrix(matrix: Matrix, rows: CorrcoefRows) -> BuiltinResult<RealEvaluation> {
    let matrix = match rows {
        CorrcoefRows::Complete => filter_complete_rows(&matrix),
        CorrcoefRows::All | CorrcoefRows::Pairwise => matrix,
    };
    let cols = matrix.cols;
    let mut coefficients = vec![f64::NAN; cols * cols];
    let mut counts = vec![0usize; cols * cols];
    for lhs in 0..cols {
        for rhs in lhs..cols {
            let (coefficient, count) = real_pair(&matrix, lhs, rhs, rows == CorrcoefRows::Pairwise);
            set_symmetric(&mut coefficients, cols, lhs, rhs, coefficient);
            set_symmetric(&mut counts, cols, lhs, rhs, count);
        }
    }
    Ok(RealEvaluation {
        r: Tensor::new(coefficients, vec![cols, cols]).map_err(corrcoef_internal_error)?,
        counts,
    })
}

fn filter_complete_rows(matrix: &Matrix) -> Matrix {
    let valid = (0..matrix.rows)
        .filter(|&row| (0..matrix.cols).all(|col| !matrix.get(row, col).is_nan()))
        .collect::<Vec<_>>();
    let mut data = Vec::with_capacity(valid.len() * matrix.cols);
    for col in 0..matrix.cols {
        data.extend(valid.iter().map(|&row| matrix.get(row, col)));
    }
    Matrix {
        data,
        rows: valid.len(),
        cols: matrix.cols,
    }
}

fn real_pair(matrix: &Matrix, lhs: usize, rhs: usize, pairwise: bool) -> (f64, usize) {
    let mut xs = Vec::with_capacity(matrix.rows);
    let mut ys = Vec::with_capacity(matrix.rows);
    for row in 0..matrix.rows {
        let x = matrix.get(row, lhs);
        let y = matrix.get(row, rhs);
        if !x.is_nan() && !y.is_nan() {
            xs.push(x);
            ys.push(y);
        } else if !pairwise {
            return (f64::NAN, matrix.rows);
        }
    }
    (real_correlation(&xs, &ys), xs.len())
}

fn real_correlation(xs: &[f64], ys: &[f64]) -> f64 {
    let count = xs.len().min(ys.len());
    if count < 2 {
        return f64::NAN;
    }
    let mean_x = xs.iter().take(count).sum::<f64>() / count as f64;
    let mean_y = ys.iter().take(count).sum::<f64>() / count as f64;
    let mut sum_xx = 0.0;
    let mut sum_yy = 0.0;
    let mut sum_xy = 0.0;
    for index in 0..count {
        let dx = xs[index] - mean_x;
        let dy = ys[index] - mean_y;
        sum_xx += dx * dx;
        sum_yy += dy * dy;
        sum_xy += dx * dy;
    }
    if sum_xx <= 0.0 || sum_yy <= 0.0 {
        f64::NAN
    } else {
        clamp_correlation(sum_xy / (sum_xx.sqrt() * sum_yy.sqrt()))
    }
}

#[derive(Debug, Clone)]
struct ComplexMatrix {
    data: Vec<Complex64>,
    rows: usize,
    cols: usize,
}

impl ComplexMatrix {
    fn from_single(tensor: ComplexTensor) -> Self {
        let (rows, cols) = single_geometry(&tensor.shape, tensor.len());
        Self {
            data: centered_complex_values(&tensor, rows, cols),
            rows,
            cols,
        }
    }

    fn from_pair(left: NumericInput, right: NumericInput) -> BuiltinResult<Self> {
        let rows = left.len();
        if rows == 1 {
            return Ok(Self {
                data: centered_complex_scalar_pair(&left, &right),
                rows: 2,
                cols: 1,
            });
        }
        let mut data = centered_numeric_values(left, rows)?;
        data.extend(centered_numeric_values(right, rows)?);
        Ok(Self {
            data,
            rows,
            cols: 2,
        })
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> Complex64 {
        self.data[row + col * self.rows]
    }
}

fn centered_complex_scalar_pair(left: &NumericInput, right: &NumericInput) -> Vec<Complex64> {
    let left = exact_complex_scalar(left);
    let right = exact_complex_scalar(right);
    match (left, right) {
        (
            ExactComplexScalar::Integer(left_real, left_imag),
            ExactComplexScalar::Integer(right_real, right_imag),
        ) => vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(
                (right_real - left_real) as f64,
                (right_imag - left_imag) as f64,
            ),
        ],
        (left, right) => vec![left.materialize(), right.materialize()],
    }
}

enum ExactComplexScalar {
    Integer(i128, i128),
    Floating(Complex64),
}

impl ExactComplexScalar {
    fn materialize(self) -> Complex64 {
        match self {
            Self::Integer(real, imag) => Complex64::new(real as f64, imag as f64),
            Self::Floating(value) => value,
        }
    }
}

fn exact_complex_scalar(input: &NumericInput) -> ExactComplexScalar {
    match input {
        NumericInput::Real(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                let value = storage
                    .value_at(0)
                    .map(|value| int_value_to_i128(&value))
                    .unwrap_or(0);
                ExactComplexScalar::Integer(value, 0)
            } else {
                ExactComplexScalar::Floating(Complex64::new(tensor.materialize_f64()[0], 0.0))
            }
        }
        NumericInput::Complex(tensor) => {
            if let Some(storage) = tensor.integer_storage() {
                let real = storage
                    .real
                    .value_at(0)
                    .map(|value| int_value_to_i128(&value))
                    .unwrap_or(0);
                let imag = storage
                    .imag
                    .value_at(0)
                    .map(|value| int_value_to_i128(&value))
                    .unwrap_or(0);
                ExactComplexScalar::Integer(real, imag)
            } else {
                let (real, imag) = tensor.materialize_f64()[0];
                ExactComplexScalar::Floating(Complex64::new(real, imag))
            }
        }
    }
}

fn centered_numeric_values(input: NumericInput, rows: usize) -> BuiltinResult<Vec<Complex64>> {
    Ok(match input {
        NumericInput::Real(tensor) => centered_tensor_values(&tensor, rows, 1)
            .into_iter()
            .map(|value| Complex64::new(value, 0.0))
            .collect(),
        NumericInput::Complex(tensor) => centered_complex_values(&tensor, rows, 1),
    })
}

fn centered_complex_values(tensor: &ComplexTensor, rows: usize, cols: usize) -> Vec<Complex64> {
    let Some(storage) = tensor.integer_storage() else {
        return tensor
            .materialize_f64()
            .into_iter()
            .map(|(real, imag)| Complex64::new(real, imag))
            .collect();
    };
    let mut values = Vec::with_capacity(rows.saturating_mul(cols));
    for col in 0..cols {
        let start = col * rows;
        let real_anchor = storage
            .real
            .value_at(start)
            .map(|value| int_value_to_i128(&value))
            .unwrap_or(0);
        let imag_anchor = storage
            .imag
            .value_at(start)
            .map(|value| int_value_to_i128(&value))
            .unwrap_or(0);
        for index in start..start + rows {
            let real = storage
                .real
                .value_at(index)
                .map(|value| int_value_to_i128(&value))
                .unwrap_or(real_anchor);
            let imag = storage
                .imag
                .value_at(index)
                .map(|value| int_value_to_i128(&value))
                .unwrap_or(imag_anchor);
            values.push(Complex64::new(
                (real - real_anchor) as f64,
                (imag - imag_anchor) as f64,
            ));
        }
    }
    values
}

fn evaluate_complex_matrix(
    matrix: ComplexMatrix,
    rows: CorrcoefRows,
) -> BuiltinResult<ComplexTensor> {
    let matrix = match rows {
        CorrcoefRows::Complete => filter_complete_complex_rows(&matrix),
        CorrcoefRows::All | CorrcoefRows::Pairwise => matrix,
    };
    let cols = matrix.cols;
    let mut coefficients = vec![Complex64::new(f64::NAN, f64::NAN); cols * cols];
    for lhs in 0..cols {
        for rhs in lhs..cols {
            let mut coefficient = complex_pair(&matrix, lhs, rhs, rows == CorrcoefRows::Pairwise);
            if lhs == rhs && complex_is_finite(coefficient) {
                coefficient = Complex64::new(1.0, 0.0);
            }
            set_hermitian(&mut coefficients, cols, lhs, rhs, coefficient);
        }
    }
    ComplexTensor::new(
        coefficients
            .into_iter()
            .map(|value| (value.re, value.im))
            .collect(),
        vec![cols, cols],
    )
    .map_err(corrcoef_internal_error)
}

fn filter_complete_complex_rows(matrix: &ComplexMatrix) -> ComplexMatrix {
    let valid = (0..matrix.rows)
        .filter(|&row| (0..matrix.cols).all(|col| !complex_has_nan(matrix.get(row, col))))
        .collect::<Vec<_>>();
    let mut data = Vec::with_capacity(valid.len() * matrix.cols);
    for col in 0..matrix.cols {
        data.extend(valid.iter().map(|&row| matrix.get(row, col)));
    }
    ComplexMatrix {
        data,
        rows: valid.len(),
        cols: matrix.cols,
    }
}

fn complex_pair(matrix: &ComplexMatrix, lhs: usize, rhs: usize, pairwise: bool) -> Complex64 {
    let mut xs = Vec::with_capacity(matrix.rows);
    let mut ys = Vec::with_capacity(matrix.rows);
    for row in 0..matrix.rows {
        let x = matrix.get(row, lhs);
        let y = matrix.get(row, rhs);
        if !complex_has_nan(x) && !complex_has_nan(y) {
            xs.push(x);
            ys.push(y);
        } else if !pairwise {
            return Complex64::new(f64::NAN, f64::NAN);
        }
    }
    complex_correlation(&xs, &ys)
}

fn complex_correlation(xs: &[Complex64], ys: &[Complex64]) -> Complex64 {
    let count = xs.len().min(ys.len());
    if count < 2 {
        return Complex64::new(f64::NAN, f64::NAN);
    }
    let mean_x = xs.iter().take(count).copied().sum::<Complex64>() / count as f64;
    let mean_y = ys.iter().take(count).copied().sum::<Complex64>() / count as f64;
    let mut sum_xx = 0.0;
    let mut sum_yy = 0.0;
    let mut sum_xy = Complex64::new(0.0, 0.0);
    for index in 0..count {
        let dx = xs[index] - mean_x;
        let dy = ys[index] - mean_y;
        sum_xx += dx.norm_sqr();
        sum_yy += dy.norm_sqr();
        sum_xy += dx.conj() * dy;
    }
    if sum_xx <= 0.0 || sum_yy <= 0.0 {
        Complex64::new(f64::NAN, f64::NAN)
    } else {
        sum_xy / (sum_xx.sqrt() * sum_yy.sqrt())
    }
}

fn complex_is_finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}

fn complex_has_nan(value: Complex64) -> bool {
    value.re.is_nan() || value.im.is_nan()
}

fn p_values(r: &Tensor, counts: &[usize]) -> BuiltinResult<Tensor> {
    let dim = r.rows();
    let coefficients = r.materialize_f64();
    let mut values = vec![f64::NAN; coefficients.len()];
    for col in 0..dim {
        for row in 0..dim {
            let index = row + col * dim;
            let coefficient = coefficients[index];
            values[index] = if row == col && coefficient.is_finite() {
                1.0
            } else {
                correlation_p_value(coefficient, counts[index])
            };
        }
    }
    Tensor::new(values, r.shape.clone()).map_err(corrcoef_internal_error)
}

fn correlation_p_value(coefficient: f64, count: usize) -> f64 {
    if !coefficient.is_finite() || count <= 2 {
        return f64::NAN;
    }
    let magnitude = coefficient.abs();
    if magnitude >= 1.0 {
        return 0.0;
    }
    let degrees = (count - 2) as f64;
    let statistic = magnitude * (degrees / (1.0 - magnitude * magnitude)).sqrt();
    (2.0 * student_t_cdf_upper(statistic, degrees)).clamp(0.0, 1.0)
}

fn confidence_bounds(r: &Tensor, counts: &[usize], alpha: f64) -> BuiltinResult<(Tensor, Tensor)> {
    let dim = r.rows();
    let coefficients = r.materialize_f64();
    let mut lower = vec![f64::NAN; coefficients.len()];
    let mut upper = vec![f64::NAN; coefficients.len()];
    let critical = standard_normal_inv(1.0 - alpha / 2.0);
    for col in 0..dim {
        for row in 0..dim {
            let index = row + col * dim;
            let coefficient = coefficients[index];
            if row == col && coefficient.is_finite() {
                lower[index] = 1.0;
                upper[index] = 1.0;
                continue;
            }
            if !coefficient.is_finite() || counts[index] <= 3 {
                continue;
            }
            if coefficient.abs() >= 1.0 {
                lower[index] = coefficient.signum();
                upper[index] = coefficient.signum();
                continue;
            }
            let center = coefficient.atanh();
            let radius = critical / ((counts[index] - 3) as f64).sqrt();
            lower[index] = (center - radius).tanh();
            upper[index] = (center + radius).tanh();
        }
    }
    Ok((
        Tensor::new(lower, r.shape.clone()).map_err(corrcoef_internal_error)?,
        Tensor::new(upper, r.shape.clone()).map_err(corrcoef_internal_error)?,
    ))
}

fn clamp_correlation(value: f64) -> f64 {
    if value.is_finite() && (value.abs() - 1.0).abs() <= 1.0e-12 {
        value.signum()
    } else {
        value
    }
}

fn set_symmetric<T: Copy>(buffer: &mut [T], dim: usize, row: usize, col: usize, value: T) {
    buffer[row + col * dim] = value;
    buffer[col + row * dim] = value;
}

fn set_hermitian(buffer: &mut [Complex64], dim: usize, row: usize, col: usize, value: Complex64) {
    buffer[row + col * dim] = value;
    buffer[col + row * dim] = value.conj();
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    #[cfg(feature = "wgpu")]
    use crate::dispatcher::download_handle_async;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate_api::AccelProvider;
    use runmat_builtins::{ResolveContext, Type};
    use runmat_value::{
        ComplexTensor, IntegerComplexStorage, IntegerStorage, LogicalArray, Tensor, Value,
    };

    fn poisoned_int_tensor(storage: IntegerStorage, shape: Vec<usize>, _poison: f64) -> Tensor {
        Tensor::new_integer(storage, shape).unwrap()
    }

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

    #[test]
    fn corrcoef_type_preserves_column_count() {
        let out = corrcoef_type(
            &[Type::Tensor {
                shape: Some(vec![Some(6), Some(2)]),
            }],
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
    fn corrcoef_type_vector_returns_scalar() {
        let out = corrcoef_type(
            &[Type::Tensor {
                shape: Some(vec![Some(4), Some(1)]),
            }],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(out, Type::Num);
    }

    #[test]
    fn corrcoef_descriptor_signatures_cover_core_forms() {
        let labels: Vec<&str> = CORRCOEF_DESCRIPTOR
            .signatures
            .iter()
            .map(|sig| sig.label)
            .collect();
        assert!(labels.contains(&"R = corrcoef(A)"));
        assert!(labels.contains(&"R = corrcoef(A, B)"));
        assert!(labels.contains(&"[R, P, RL, RU] = corrcoef(A)"));
        assert!(labels.contains(&"___ = corrcoef(A, Name, Value)"));
        assert_eq!(
            CORRCOEF_DESCRIPTOR.output_mode,
            BuiltinOutputMode::ByRequestedOutputCount
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn corrcoef_matrix_basic() {
        let tensor = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, //
                2.0, 4.0, 6.0, 8.0, //
                4.0, 1.0, -1.0, 0.0,
            ],
            vec![4, 3],
        )
        .unwrap();
        let result =
            block_on(corrcoef_builtin(Value::Tensor(tensor), Vec::new())).expect("corrcoef");
        match result {
            Value::Tensor(out) => {
                let expected = [
                    1.0,
                    1.0,
                    -0.836_660_026_534,
                    1.0,
                    1.0,
                    -0.836_660_026_534,
                    -0.836_660_026_534,
                    -0.836_660_026_534,
                    1.0,
                ];
                assert_tensor_close(&out, &expected, 1.0e-10);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn corrcoef_accepts_typed_integer_matrix_inputs() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let tensor = poisoned_int_tensor(IntegerStorage::I16(vec![1, 2, 1, 4]), vec![2, 2], 0.0);
        let result =
            block_on(corrcoef_builtin(Value::Tensor(tensor), Vec::new())).expect("corrcoef");
        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                assert!((out.materialize_f64()[0] - 1.0).abs() < 1.0e-12);
                assert!((out.materialize_f64()[3] - 1.0).abs() < 1.0e-12);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[test]
    fn corrcoef_complete_rows_reads_typed_integer_storage() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let x = poisoned_int_tensor(
            IntegerStorage::I16(vec![1, 2, 3, 2, 4, 6]),
            vec![3, 2],
            f64::NAN,
        );
        let y = poisoned_int_tensor(
            IntegerStorage::U16(vec![5, 10, 15, 10, 20, 30]),
            vec![3, 2],
            f64::NAN,
        );

        let result = block_on(corrcoef_builtin(
            Value::Tensor(x),
            vec![
                Value::Tensor(y),
                Value::from("Rows"),
                Value::from("complete"),
            ],
        ))
        .expect("corrcoef complete rows");

        match result {
            Value::Tensor(out) => {
                assert_eq!(out.shape, vec![2, 2]);
                for value in out.materialize_f64() {
                    assert!((value - 1.0).abs() < 1.0e-12);
                }
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn corrcoef_two_inputs_vectorize_equal_size_matrices() {
        let left = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, //
                4.0, 5.0, 6.0, 7.0,
            ],
            vec![4, 2],
        )
        .unwrap();
        let right = Tensor::new(
            vec![
                8.0, 6.0, 7.0, 5.0, //
                2.0, 9.0, 1.0, 3.0,
            ],
            vec![4, 2],
        )
        .unwrap();
        let combined = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, //
                4.0, 5.0, 6.0, 7.0, //
                8.0, 6.0, 7.0, 5.0, //
                2.0, 9.0, 1.0, 3.0,
            ],
            vec![8, 2],
        )
        .unwrap();

        let via_two = block_on(corrcoef_builtin(
            Value::Tensor(left.clone()),
            vec![Value::Tensor(right.clone())],
        ))
        .expect("corrcoef");
        let via_combined = block_on(corrcoef_builtin(Value::Tensor(combined), Vec::new()))
            .expect("corrcoef combined");

        let expected_tensor = match via_combined {
            Value::Tensor(t) => t,
            _ => panic!("expected tensor output"),
        };
        let actual_tensor = match via_two {
            Value::Tensor(t) => t,
            _ => panic!("expected tensor output"),
        };
        assert_tensor_close(&actual_tensor, &expected_tensor.materialize_f64(), 1.0e-10);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn corrcoef_rows_complete_ignores_missing() {
        let tensor = Tensor::new(
            vec![
                1.0,
                f64::NAN,
                3.0,
                4.0, //
                2.0,
                5.0,
                f64::NAN,
                8.0,
            ],
            vec![4, 2],
        )
        .unwrap();
        let result = block_on(corrcoef_builtin(
            Value::Tensor(tensor),
            vec![Value::from("rows"), Value::from("complete")],
        ))
        .expect("corrcoef");
        match result {
            Value::Tensor(out) => {
                let expected = [
                    1.0, 1.0, //
                    1.0, 1.0,
                ];
                assert_tensor_close(&out, &expected, 1.0e-10);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn corrcoef_rows_options_do_not_treat_infinity_as_missing() {
        for rows in ["complete", "pairwise"] {
            let tensor =
                Tensor::new(vec![1.0, f64::INFINITY, 3.0, 1.0, 2.0, 3.0], vec![3, 2]).unwrap();
            let result = block_on(corrcoef_builtin(
                Value::Tensor(tensor),
                vec![Value::from("rows"), Value::from(rows)],
            ))
            .expect("corrcoef");
            let Value::Tensor(out) = result else {
                panic!("expected tensor result");
            };
            let values = out.materialize_f64();
            assert!(values[0].is_nan());
            assert!(values[1].is_nan());
            assert!(values[2].is_nan());
            assert_eq!(values[3], 1.0);
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn corrcoef_rows_pairwise_staggered_missing() {
        let tensor = Tensor::new(
            vec![
                1.0,
                f64::NAN,
                4.0,
                5.0, //
                2.0,
                5.0,
                f64::NAN,
                8.0, //
                3.0,
                1.0,
                6.0,
                f64::NAN,
            ],
            vec![4, 3],
        )
        .unwrap();
        let result = block_on(corrcoef_builtin(
            Value::Tensor(tensor),
            vec![Value::from("rows"), Value::from("pairwise")],
        ))
        .expect("corrcoef");
        match result {
            Value::Tensor(out) => {
                let expected = [
                    1.0, 1.0, 1.0, //
                    1.0, 1.0, -1.0, //
                    1.0, -1.0, 1.0,
                ];
                assert_tensor_close(&out, &expected, 1.0e-10);
            }
            other => panic!("expected tensor result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn corrcoef_numeric_scalar_second_argument_is_data() {
        let result = block_on(corrcoef_builtin(Value::Num(1.0), vec![Value::Num(2.0)]))
            .expect("scalar pair");
        let Value::Tensor(result) = result else {
            panic!("expected tensor")
        };
        assert_tensor_close(&result, &[1.0], 0.0);

        let equal = block_on(corrcoef_builtin(Value::Num(2.0), vec![Value::Num(2.0)]))
            .expect("equal scalar pair");
        let Value::Tensor(equal) = equal else {
            panic!("expected tensor")
        };
        assert_tensor_close(&equal, &[f64::NAN], 0.0);
    }

    #[test]
    fn corrcoef_public_scalar_vector_and_paired_empty_geometry() {
        let scalar = Tensor::new(vec![7.0], vec![1, 1]).unwrap();
        let Value::Tensor(scalar_result) =
            block_on(corrcoef_builtin(Value::Tensor(scalar), Vec::new())).expect("scalar")
        else {
            panic!("expected tensor")
        };
        assert_tensor_close(&scalar_result, &[f64::NAN], 0.0);

        for shape in [vec![1, 4], vec![4, 1]] {
            let vector = Tensor::new(vec![1.0, 2.0, 4.0, 8.0], shape).unwrap();
            let Value::Tensor(vector_result) =
                block_on(corrcoef_builtin(Value::Tensor(vector), Vec::new())).expect("vector")
            else {
                panic!("expected tensor")
            };
            assert_tensor_close(&vector_result, &[1.0], 0.0);
        }

        let left = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let right = Tensor::new(Vec::new(), vec![0, 0]).unwrap();
        let Value::Tensor(empty) = block_on(corrcoef_builtin(
            Value::Tensor(left),
            vec![Value::Tensor(right)],
        ))
        .expect("paired empty") else {
            panic!("expected tensor")
        };
        assert_tensor_close(&empty, &[f64::NAN; 4], 0.0);
    }

    #[test]
    fn corrcoef_supports_all_integer_classes_and_exact_wide_centering() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let storages = vec![
            IntegerStorage::I8(vec![1, 2, 3, 3, 2, 1]),
            IntegerStorage::I16(vec![1, 2, 3, 3, 2, 1]),
            IntegerStorage::I32(vec![1, 2, 3, 3, 2, 1]),
            IntegerStorage::I64(vec![1, 2, 3, 3, 2, 1]),
            IntegerStorage::U8(vec![1, 2, 3, 3, 2, 1]),
            IntegerStorage::U16(vec![1, 2, 3, 3, 2, 1]),
            IntegerStorage::U32(vec![1, 2, 3, 3, 2, 1]),
            IntegerStorage::U64(vec![1, 2, 3, 3, 2, 1]),
        ];
        for storage in storages {
            let class = storage.class_name();
            let tensor = Tensor::new_integer(storage, vec![3, 2]).unwrap();
            let Value::Tensor(result) =
                block_on(corrcoef_builtin(Value::Tensor(tensor), Vec::new()))
                    .unwrap_or_else(|error| panic!("{class}: {error}"))
            else {
                panic!("{class}: expected tensor")
            };
            assert_tensor_close(&result, &[1.0, -1.0, -1.0, 1.0], 1.0e-12);
        }

        let base = u64::MAX - 2;
        let wide = Tensor::new_integer(
            IntegerStorage::U64(vec![base, base + 1, base + 2, base + 2, base + 1, base]),
            vec![3, 2],
        )
        .unwrap();
        let Value::Tensor(result) =
            block_on(corrcoef_builtin(Value::Tensor(wide), Vec::new())).expect("wide corrcoef")
        else {
            panic!("expected tensor")
        };
        assert_tensor_close(&result, &[1.0, -1.0, -1.0, 1.0], 1.0e-12);
    }

    #[test]
    fn corrcoef_integer_and_logical_extensions_gate_before_dispatch() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer = Tensor::new_integer(IntegerStorage::I16(vec![1, 2, 3]), vec![3, 1]).unwrap();
        let integer_error =
            block_on(corrcoef_builtin(Value::Tensor(integer), Vec::new())).unwrap_err();
        assert_eq!(
            integer_error.identifier(),
            Some("RunMat:compatibility:CorrcoefIntegerDataExtension")
        );

        let logical = LogicalArray::new(vec![0, 1, 1], vec![3, 1]).unwrap();
        let logical_error =
            block_on(corrcoef_builtin(Value::LogicalArray(logical), Vec::new())).unwrap_err();
        assert_eq!(
            logical_error.identifier(),
            Some("RunMat:compatibility:CorrcoefLogicalDataExtension")
        );
    }

    #[test]
    fn corrcoef_resident_integer_gate_precedes_provider_dispatch() {
        test_support::with_test_provider(|provider| {
            let tensor =
                Tensor::new_integer(IntegerStorage::U16(vec![1, 2, 3]), vec![3, 1]).unwrap();
            let handle = gpu_helpers::upload_tensor(provider, &tensor).expect("upload integer");
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = block_on(corrcoef_builtin(
                Value::GpuTensor(handle.clone()),
                Vec::new(),
            ))
            .unwrap_err();
            assert_eq!(
                error.identifier(),
                Some("RunMat:compatibility:CorrcoefIntegerDataExtension")
            );
            let _ = provider.free(&handle);
        });
    }

    #[test]
    fn corrcoef_complex_coefficients_are_hermitian() {
        let tensor = ComplexTensor::new(
            vec![
                (1.0, 0.0),
                (2.0, 0.0),
                (3.0, 0.0),
                (0.0, 1.0),
                (0.0, 2.0),
                (0.0, 3.0),
            ],
            vec![3, 2],
        )
        .unwrap();
        let Value::ComplexTensor(result) =
            block_on(corrcoef_builtin(Value::ComplexTensor(tensor), Vec::new()))
                .expect("complex corrcoef")
        else {
            panic!("expected complex tensor")
        };
        assert_eq!(result.shape, vec![2, 2]);
        let values = result.materialize_f64();
        assert_eq!(values[0], (1.0, 0.0));
        assert!((values[1].0).abs() < 1.0e-12);
        assert!((values[1].1 + 1.0).abs() < 1.0e-12);
        assert!((values[2].0).abs() < 1.0e-12);
        assert!((values[2].1 - 1.0).abs() < 1.0e-12);
        assert_eq!(values[3], (1.0, 0.0));
    }

    #[test]
    fn corrcoef_complex_integer_components_center_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let base = u64::MAX - 2;
        let storage = IntegerComplexStorage::new(
            IntegerStorage::U64(vec![base, base + 1, base + 2]),
            IntegerStorage::U64(vec![base + 2, base + 1, base]),
        )
        .unwrap();
        let tensor = ComplexTensor::new_integer(storage, vec![3, 1]).unwrap();
        let Value::ComplexTensor(result) =
            block_on(corrcoef_builtin(Value::ComplexTensor(tensor), Vec::new()))
                .expect("complex integer corrcoef")
        else {
            panic!("expected complex tensor")
        };
        let values = result.materialize_f64();
        assert_eq!(result.shape, vec![1, 1]);
        assert!((values[0].0 - 1.0).abs() < 1.0e-12);
        assert!(values[0].1.abs() < 1.0e-12);
    }

    #[test]
    fn corrcoef_requested_outputs_include_p_values_and_alpha_bounds() {
        let tensor = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, //
                1.0, 2.0, 1.0, 3.0, 5.0, 4.0,
            ],
            vec![6, 2],
        )
        .unwrap();
        let _outputs = crate::output_count::push_output_count(Some(4));
        let Value::OutputList(outputs) = block_on(corrcoef_builtin(
            Value::Tensor(tensor),
            vec![Value::from("Alpha"), Value::Num(0.1)],
        ))
        .expect("four outputs") else {
            panic!("expected output list")
        };
        assert_eq!(outputs.len(), 4);
        let tensors = outputs
            .into_iter()
            .map(|value| match value {
                Value::Tensor(tensor) => tensor,
                other => panic!("expected tensor, got {other:?}"),
            })
            .collect::<Vec<_>>();
        for tensor in &tensors {
            assert_eq!(tensor.shape, vec![2, 2]);
        }
        let p = tensors[1].materialize_f64();
        assert_eq!(p[0], 1.0);
        assert_eq!(p[3], 1.0);
        assert!(p[1] > 0.0 && p[1] < 1.0);
        let r = tensors[0].materialize_f64()[1];
        let lower = tensors[2].materialize_f64()[1];
        let upper = tensors[3].materialize_f64()[1];
        assert!(lower < r && r < upper);
    }

    #[test]
    fn corrcoef_complex_multi_output_is_rejected() {
        let tensor =
            ComplexTensor::new(vec![(1.0, 1.0), (2.0, 0.0), (3.0, -1.0)], vec![3, 1]).unwrap();
        let _outputs = crate::output_count::push_output_count(Some(2));
        let error =
            block_on(corrcoef_builtin(Value::ComplexTensor(tensor), Vec::new())).unwrap_err();
        assert_eq!(
            error.identifier(),
            CORRCOEF_ERROR_COMPLEX_OUTPUTS.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn corrcoef_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let tensor = Tensor::new(
                vec![
                    1.0, 2.0, 3.0, 4.0, //
                    2.0, 4.0, 6.0, 8.0, //
                    4.0, 1.0, -1.0, 0.0,
                ],
                vec![4, 3],
            )
            .unwrap();
            let view = runmat_accelerate_api::HostTensorView {
                data: &tensor.materialize_f64(),
                shape: &tensor.shape,
            };
            let handle = provider.upload(&view).expect("upload");
            let result =
                block_on(corrcoef_builtin(Value::GpuTensor(handle), Vec::new())).expect("corrcoef");
            let gathered = test_support::gather(result).expect("gather");
            let expected = [
                1.0,
                1.0,
                -0.836_660_026_534,
                1.0,
                1.0,
                -0.836_660_026_534,
                -0.836_660_026_534,
                -0.836_660_026_534,
                1.0,
            ];
            assert_tensor_close(&gathered, &expected, 1.0e-10);
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn corrcoef_mismatched_sizes_error() {
        let left = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1]).unwrap();
        let right = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = block_on(corrcoef_builtin(
            Value::Tensor(left),
            vec![Value::Tensor(right)],
        ))
        .expect_err("expected mismatch error");
        assert_eq!(err.identifier(), CORRCOEF_ERROR_SIZE_MISMATCH.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn corrcoef_invalid_alpha_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = block_on(corrcoef_builtin(
            Value::Tensor(tensor),
            vec![Value::from("Alpha"), Value::Num(1.0)],
        ))
        .expect_err("expected invalid Alpha error");
        assert_eq!(err.identifier(), CORRCOEF_ERROR_ALPHA_INVALID.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn corrcoef_unknown_rows_option_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = block_on(corrcoef_builtin(
            Value::Tensor(tensor),
            vec![Value::from("rows"), Value::from("bogus")],
        ))
        .expect_err("expected unknown rows option error");
        assert_eq!(
            err.identifier(),
            CORRCOEF_ERROR_ROWS_OPTION_UNKNOWN.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn corrcoef_duplicate_alpha_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = block_on(corrcoef_builtin(
            Value::Tensor(tensor),
            vec![
                Value::from("Alpha"),
                Value::Num(0.05),
                Value::from("Alpha"),
                Value::Num(0.1),
            ],
        ))
        .expect_err("expected duplicate Alpha error");
        assert_eq!(err.identifier(), CORRCOEF_ERROR_OPTION_DUPLICATE.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn corrcoef_unknown_option_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let err = block_on(corrcoef_builtin(
            Value::Tensor(tensor),
            vec![Value::from("bogus"), Value::from("value")],
        ))
        .expect_err("expected unknown option error");
        assert_eq!(err.identifier(), CORRCOEF_ERROR_OPTION_UNKNOWN.identifier);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn corrcoef_rows_option_malformed_errors() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let non_string_option = Tensor::new(vec![1.0, 2.0], vec![2, 1]).unwrap();
        let err = block_on(corrcoef_builtin(
            Value::Tensor(tensor),
            vec![Value::from("rows"), Value::Tensor(non_string_option)],
        ))
        .expect_err("expected malformed rows option error");
        assert_eq!(
            err.identifier(),
            CORRCOEF_ERROR_ROWS_OPTION_MALFORMED.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn corrcoef_too_many_input_arrays_errors() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3, 1]).unwrap();
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3, 1]).unwrap();
        let c = Tensor::new(vec![7.0, 8.0, 9.0], vec![3, 1]).unwrap();
        let err = block_on(corrcoef_builtin(
            Value::Tensor(a),
            vec![Value::Tensor(b), Value::Tensor(c)],
        ))
        .expect_err("expected too many input arrays error");
        assert_eq!(
            err.identifier(),
            CORRCOEF_ERROR_TOO_MANY_INPUT_ARRAYS.identifier
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn corrcoef_wgpu_matches_cpu() {
        let _guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };
        let tensor = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, //
                2.0, 5.0, 6.0, 8.0, //
                4.0, 1.0, 7.0, 0.0,
            ],
            vec![4, 3],
        )
        .unwrap();
        let cpu = corrcoef_from_tensors(
            tensor.clone(),
            None,
            CorrcoefNormalization::Unbiased,
            CorrcoefRows::All,
        )
        .expect("cpu corrcoef");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.materialize_f64(),
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");
        let options = CorrcoefOptions {
            normalization: CorrcoefNormalization::Unbiased,
            rows: CorrcoefRows::All,
        };
        let gpu = block_on(provider.corrcoef(&handle, &options)).expect("corrcoef");
        let host = block_on(download_handle_async(provider, &gpu)).expect("download");
        let gathered =
            Tensor::new(host.data.clone(), host.shape.clone()).expect("tensor reconstruction");
        assert_tensor_close(&gathered, &cpu.materialize_f64(), 1.0e-6);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn corrcoef_wgpu_public_vector_and_paired_geometry() {
        let _guard = test_support::accel_test_lock();
        let Ok(provider) = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        ) else {
            return;
        };

        let row = Tensor::new(vec![1.0, 2.0, 4.0, 8.0], vec![1, 4]).unwrap();
        let row_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &row.materialize_f64(),
                shape: &row.shape,
            })
            .expect("row upload");
        let row_result = block_on(corrcoef_builtin(
            Value::GpuTensor(row_handle.clone()),
            Vec::new(),
        ))
        .expect("row corrcoef");
        let row_gathered = test_support::gather(row_result).expect("row gather");
        assert_tensor_close(&row_gathered, &[1.0], 1.0e-6);
        let _ = provider.free(&row_handle);

        let left = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let right = Tensor::new(vec![4.0, 1.0, 2.0, 8.0], vec![2, 2]).unwrap();
        let expected = corrcoef_from_tensors(
            left.clone(),
            Some(right.clone()),
            CorrcoefNormalization::Unbiased,
            CorrcoefRows::All,
        )
        .expect("host pair");
        let left_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &left.materialize_f64(),
                shape: &left.shape,
            })
            .expect("left upload");
        let right_handle = provider
            .upload(&runmat_accelerate_api::HostTensorView {
                data: &right.materialize_f64(),
                shape: &right.shape,
            })
            .expect("right upload");
        let paired = block_on(corrcoef_builtin(
            Value::GpuTensor(left_handle.clone()),
            vec![Value::GpuTensor(right_handle.clone())],
        ))
        .expect("paired corrcoef");
        let paired_gathered = test_support::gather(paired).expect("paired gather");
        assert_tensor_close(&paired_gathered, &expected.materialize_f64(), 1.0e-6);
        let left_after =
            block_on(download_handle_async(provider, &left_handle)).expect("left after corrcoef");
        let right_after =
            block_on(download_handle_async(provider, &right_handle)).expect("right after corrcoef");
        assert_eq!(left_after.shape, vec![2, 2]);
        assert_eq!(right_after.shape, vec![2, 2]);
        let _ = provider.free(&left_handle);
        let _ = provider.free(&right_handle);
    }
}
