//! MATLAB-compatible `cov` builtin with GPU-aware semantics for RunMat.

use runmat_accelerate_api::{CovNormalization, CovRows, CovarianceOptions};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    Tensor, Value,
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
        description: "Second dataset with matching row count.",
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
        description: "Second dataset with matching row count.",
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
        description: "Second dataset with matching row count.",
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
    when: "Two input datasets do not have the same number of rows.",
    message: "cov: inputs must have the same number of rows",
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
    builtin_path = "crate::builtins::stats::summary::cov"
)]
async fn cov_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    let args = CovArgs::parse(value, rest)?;
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
}

impl CovArgs {
    fn parse(first: Value, rest: Vec<Value>) -> BuiltinResult<Self> {
        let mut second_candidate: Option<Value> = None;
        let mut weight_candidate: Option<Value> = None;
        let mut normalization = CovNormalization::Unbiased;
        let mut normalization_explicit = false;
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
        })
    }
}

#[derive(Debug, Clone)]
pub enum CovWeightSpec {
    Scalar(CovNormalization),
    Vector(Vec<f64>),
}

async fn cov_try_gpu(args: &CovArgs) -> BuiltinResult<Option<Value>> {
    if args.rows != CovRows::All || args.weight_vector.is_some() {
        return Ok(None);
    }

    let provider = match runmat_accelerate_api::provider() {
        Some(p) => p,
        None => return Ok(None),
    };

    let first_handle = match &args.first {
        Value::GpuTensor(handle) => handle,
        _ => return Ok(None),
    };

    let maybe_second_handle = match &args.second {
        Some(Value::GpuTensor(handle)) => Some(handle),
        Some(_) => return Ok(None),
        None => None,
    };

    let options = CovarianceOptions {
        normalization: args.normalization,
        rows: args.rows,
        has_weight_vector: false,
    };

    match provider
        .covariance(first_handle, maybe_second_handle, None, &options)
        .await
    {
        Ok(result) => Ok(Some(Value::GpuTensor(result))),
        Err(_) => Ok(None),
    }
}

async fn cov_host(args: CovArgs) -> BuiltinResult<Value> {
    let CovArgs {
        first,
        second,
        normalization,
        rows,
        weight_vector,
    } = args;

    let left = value_to_tensor_gather(first).await?;
    let right = match second {
        Some(value) => Some(value_to_tensor_gather(value).await?),
        None => None,
    };

    let weight_spec = if let Some(weight_value) = weight_vector {
        let vector = value_to_weight_vector(weight_value, left.rows()).await?;
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
    for (idx, weight) in tensor.data.iter().enumerate() {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(cov_error_with_detail(
                &COV_ERROR_INVALID_ARGUMENT,
                format!("weights must be non-negative finite values (index {idx})"),
            ));
        }
    }
    if tensor.data.is_empty() {
        return Err(cov_error_with_detail(
            &COV_ERROR_INVALID_ARGUMENT,
            "weight vector cannot be empty",
        ));
    }
    Ok(tensor.data)
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

    if cols_first == 1 && !normalization_explicit && matches!(rows_option, CovRows::All) {
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
    fn from_tensor(name: &str, tensor: Tensor) -> BuiltinResult<Self> {
        if tensor.shape.len() > 2 {
            return Err(cov_error_with_detail(
                &COV_ERROR_INVALID_ARGUMENT,
                format!("{name}: inputs must be 2-D matrices or vectors"),
            ));
        }
        Ok(Self {
            rows: tensor.rows(),
            cols: tensor.cols(),
            data: tensor.data,
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
    let mut matrix = Matrix::from_tensor("cov", left)?;
    if let Some(second) = right {
        let right_matrix = Matrix::from_tensor("cov", second)?;
        if matrix.rows != right_matrix.rows {
            return Err(cov_error(&COV_ERROR_ROWS_MISMATCH));
        }
        matrix.cols += right_matrix.cols;
        matrix
            .data
            .extend_from_slice(&right_matrix.data[..right_matrix.rows * right_matrix.cols]);
    }
    Ok(matrix)
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
                CovNormalization::Unbiased => (rows as f64) - 1.0,
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
    use runmat_builtins::{ResolveContext, Tensor, Type};

    fn assert_tensor_close(actual: &Tensor, expected: &[f64], tol: f64) {
        let dim = (expected.len() as f64).sqrt() as usize;
        assert_eq!(actual.shape, vec![dim, dim], "unexpected tensor shape");
        for (idx, (&got, &want)) in actual.data.iter().zip(expected.iter()).enumerate() {
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

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn cov_weighted_vector() {
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
                data: &tensor.data,
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
    #[cfg(feature = "wgpu")]
    fn cov_wgpu_matches_cpu() {
        let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
            runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
        );

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

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let view = runmat_accelerate_api::HostTensorView {
            data: &tensor.data,
            shape: &tensor.shape,
        };
        let handle = provider.upload(&view).expect("upload");

        let gpu_value = cov_builtin_sync(Value::GpuTensor(handle), Vec::new()).expect("cov");
        let gathered = test_support::gather(gpu_value).expect("gather");

        assert_tensor_close(&gathered, &cpu_tensor.data, 1.0e-6);
    }
}
