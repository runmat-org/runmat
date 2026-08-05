//! Convert covariance matrices to correlation matrices.

use runmat_accelerate_api::{
    AccelProvider, GpuTensorHandle, GpuTensorStorage, ProviderCovarianceToCorrelationResult,
};
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    NumericDType, ResolveContext, Tensor, Type, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, gather_if_needed_async, BuiltinResult, RuntimeError};

const OUTPUT_R: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "R",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Correlation matrix corresponding to the covariance matrix.",
};

const OUTPUT_SIGMA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "sigma",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Optional,
    default: None,
    description: "Column vector of standard deviations computed from diag(C).",
};

const OUTPUT_EXP_SIGMA: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ExpSigma",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Row vector of standard deviations computed from diag(ExpCovariance).",
};

const OUTPUT_EXP_CORR: BuiltinParamDescriptor = BuiltinParamDescriptor {
    name: "ExpCorrC",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Correlation matrix corresponding to ExpCovariance.",
};

const OUTPUTS_R: [BuiltinParamDescriptor; 1] = [OUTPUT_R];
const OUTPUTS_R_SIGMA: [BuiltinParamDescriptor; 2] = [OUTPUT_R, OUTPUT_SIGMA];
const OUTPUTS_EXP_SIGMA_CORR: [BuiltinParamDescriptor; 2] = [OUTPUT_EXP_SIGMA, OUTPUT_EXP_CORR];

const INPUT_C: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "C",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square covariance matrix.",
}];

const CORRCOV_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "R = corrcov(C)",
        inputs: &INPUT_C,
        outputs: &OUTPUTS_R,
    },
    BuiltinSignatureDescriptor {
        label: "[R, sigma] = corrcov(C)",
        inputs: &INPUT_C,
        outputs: &OUTPUTS_R_SIGMA,
    },
];

const INPUT_EXP_COVARIANCE: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "ExpCovariance",
    ty: BuiltinParamType::Any,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Square covariance matrix.",
}];

const COV2CORR_SIGNATURES: [BuiltinSignatureDescriptor; 1] = [BuiltinSignatureDescriptor {
    label: "[ExpSigma, ExpCorrC] = cov2corr(ExpCovariance)",
    inputs: &INPUT_EXP_COVARIANCE,
    outputs: &OUTPUTS_EXP_SIGMA_CORR,
}];

const CORRCOV_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "corrcov-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "corrcov with typed-integer covariance data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CorrcovIntegerDataExtension"),
};

const CORRCOV_LOGICAL_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "corrcov-logical-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "corrcov with logical covariance data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:CorrcovLogicalDataExtension"),
};

const CORRCOV_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    CORRCOV_INTEGER_DATA_EXTENSION,
    CORRCOV_LOGICAL_DATA_EXTENSION,
];

const COV2CORR_INTEGER_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cov2corr-integer-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cov2corr with typed-integer covariance data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Cov2corrIntegerDataExtension"),
};

const COV2CORR_LOGICAL_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cov2corr-logical-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cov2corr with logical covariance data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Cov2corrLogicalDataExtension"),
};

const COV2CORR_SINGLE_DATA_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cov2corr-single-data",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cov2corr with single covariance data is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Cov2corrSingleDataExtension"),
};

const COV2CORR_GPU_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "cov2corr-gpu-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "cov2corr with a resident GPU covariance matrix is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:Cov2corrGpuInputExtension"),
};

const COV2CORR_EXTENSIONS: [BuiltinExtensionDescriptor; 4] = [
    COV2CORR_INTEGER_DATA_EXTENSION,
    COV2CORR_LOGICAL_DATA_EXTENSION,
    COV2CORR_SINGLE_DATA_EXTENSION,
    COV2CORR_GPU_INPUT_EXTENSION,
];

const CORRCOV_INTEGER_C_INPUT: [BuiltinIntegerInputCapability; 1] = [BuiltinIntegerInputCapability {
    name: "C",
    classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
    availability: BuiltinIntegerInputAvailability::RunMatOnly,
    scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
    notes: "The documented covariance-matrix domain is floating; RunMat mode additionally accepts all eight real integer classes.",
}];

const COV2CORR_INTEGER_C_INPUT: [BuiltinIntegerInputCapability; 1] =
    [BuiltinIntegerInputCapability {
        name: "ExpCovariance",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The documented Finance Toolbox covariance-matrix domain is double; RunMat mode additionally accepts all eight real integer classes.",
    }];

const CORRCOV_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[R, sigma] = corrcov(integer_C)",
        inputs: &CORRCOV_INTEGER_C_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat validates exact integer symmetry before converting covariance values into double square-root and normalization arithmetic.",
    }];

const COV2CORR_INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[ExpSigma, ExpCorrC] = cov2corr(integer_ExpCovariance)",
        inputs: &COV2CORR_INTEGER_C_INPUT,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::Multiple,
        notes: "RunMat validates exact integer symmetry before returning double standard deviations and correlation coefficients in the documented Finance Toolbox order and shapes.",
    }];

const ERROR_INTERNAL: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.COVARIANCE_CONVERSION.INTERNAL",
    identifier: None,
    when: "Internal tensor allocation fails.",
    message: "covariance conversion: internal error",
};

macro_rules! conversion_descriptor {
    ($name:literal, $signatures:expr, $invalid_when:literal) => {
        const ERRORS: [BuiltinErrorDescriptor; 2] = [
            BuiltinErrorDescriptor {
                code: concat!("RM.", $name, ".INVALID_ARGUMENT"),
                identifier: Some(concat!("RunMat:", $name, ":InvalidArgument")),
                when: $invalid_when,
                message: "covariance conversion: invalid argument",
            },
            BuiltinErrorDescriptor {
                code: concat!("RM.", $name, ".INTERNAL"),
                identifier: Some(concat!("RunMat:", $name, ":Internal")),
                when: ERROR_INTERNAL.when,
                message: ERROR_INTERNAL.message,
            },
        ];

        pub const DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
            signatures: &$signatures,
            output_mode: BuiltinOutputMode::ByRequestedOutputCount,
            completion_policy: BuiltinCompletionPolicy::Public,
            errors: &ERRORS,
        };
    };
}

fn corrcov_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    match args.first() {
        Some(Type::Tensor { shape }) | Some(Type::Logical { shape }) => Type::Tensor {
            shape: shape.clone(),
        },
        Some(Type::Num | Type::Int | Type::Bool) => Type::Num,
        Some(Type::Unknown) | None => Type::Unknown,
        _ => Type::Unknown,
    }
}

fn cov2corr_type(args: &[Type], _ctx: &ResolveContext) -> Type {
    match args.first() {
        Some(Type::Tensor { shape: Some(shape) }) | Some(Type::Logical { shape: Some(shape) }) => {
            let width = shape.first().copied().flatten();
            Type::Tensor {
                shape: Some(vec![Some(1), width]),
            }
        }
        Some(Type::Tensor { shape: None }) | Some(Type::Logical { shape: None }) => Type::tensor(),
        Some(Type::Num | Type::Int | Type::Bool) => Type::Num,
        Some(Type::Unknown) | None => Type::Unknown,
        _ => Type::Unknown,
    }
}

#[derive(Clone, Copy)]
enum ConversionKind {
    Corrcov,
    Cov2corr,
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

fn is_single_value(value: &Value) -> bool {
    matches!(
        value,
        Value::Tensor(tensor) if tensor.numeric_dtype() == NumericDType::F32
    ) || matches!(
        value,
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_precision(handle)
                == Some(runmat_accelerate_api::ProviderPrecision::F32)
                && runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
    )
}

fn ensure_conversion_extensions(kind: ConversionKind, value: &Value) -> BuiltinResult<()> {
    match kind {
        ConversionKind::Corrcov => {
            if is_typed_integer_value(value) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &CORRCOV_INTEGER_DATA_EXTENSION,
                    "corrcov",
                )?;
            }
            if is_logical_value(value) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &CORRCOV_LOGICAL_DATA_EXTENSION,
                    "corrcov",
                )?;
            }
        }
        ConversionKind::Cov2corr => {
            if is_typed_integer_value(value) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &COV2CORR_INTEGER_DATA_EXTENSION,
                    "cov2corr",
                )?;
            }
            if is_logical_value(value) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &COV2CORR_LOGICAL_DATA_EXTENSION,
                    "cov2corr",
                )?;
            }
            if is_single_value(value) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &COV2CORR_SINGLE_DATA_EXTENSION,
                    "cov2corr",
                )?;
            }
            if matches!(value, Value::GpuTensor(_)) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &COV2CORR_GPU_INPUT_EXTENSION,
                    "cov2corr",
                )?;
            }
        }
    }
    Ok(())
}

fn conversion_error(name: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(name)
        .with_identifier(format!("RunMat:{name}:InvalidArgument"))
        .build()
}

fn internal_error(name: &'static str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin(name)
        .with_identifier(format!("RunMat:{name}:Internal"))
        .build()
}

async fn covariance_tensor(
    name: &'static str,
    value: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Tensor> {
    if !rest.is_empty() {
        return Err(conversion_error(
            name,
            format!("{name}: expected exactly one input argument"),
        ));
    }
    let gathered = gather_if_needed_async(&value)
        .await
        .map_err(|err| conversion_error(name, format!("{name}: unable to gather input: {err}")))?;
    match gathered {
        Value::Tensor(tensor)
            if matches!(
                tensor.numeric_dtype(),
                NumericDType::F64 | NumericDType::F32
            ) =>
        {
            Ok(tensor)
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            validate_integer_covariance_structure(name, &tensor)?;
            tensor::integer_tensor_to_f64(tensor)
                .map_err(|err| conversion_error(name, format!("{name}: {err}")))
        }
        Value::Tensor(tensor) => Err(conversion_error(
            name,
            format!(
                "{name}: covariance matrix must be numeric real, got {}",
                tensor.numeric_dtype().class_name()
            ),
        )),
        Value::LogicalArray(logical) => {
            let data = logical.data.into_iter().map(f64::from).collect::<Vec<_>>();
            Tensor::new(data, logical.shape)
                .map_err(|err| internal_error(name, format!("{name}: {err}")))
        }
        Value::Num(value) => Tensor::new(vec![value], vec![1, 1])
            .map_err(|err| internal_error(name, format!("{name}: {err}"))),
        Value::Int(value) => Tensor::new(vec![value.to_f64()], vec![1, 1])
            .map_err(|err| internal_error(name, format!("{name}: {err}"))),
        Value::Bool(value) => Tensor::new(vec![if value { 1.0 } else { 0.0 }], vec![1, 1])
            .map_err(|err| internal_error(name, format!("{name}: {err}"))),
        Value::Complex(..) | Value::ComplexTensor(_) => Err(conversion_error(
            name,
            format!("{name}: complex covariance matrices are not supported"),
        )),
        other => Err(conversion_error(
            name,
            format!("{name}: expected a numeric covariance matrix, got {other:?}"),
        )),
    }
}

fn validate_integer_covariance_structure(name: &'static str, tensor: &Tensor) -> BuiltinResult<()> {
    if tensor.shape.len() > 2 || tensor.rows != tensor.cols {
        return Ok(());
    }
    let Some(storage) = tensor.integer_storage() else {
        return Ok(());
    };
    for col in 0..tensor.cols {
        for row in 0..col {
            if storage.value_at(row + col * tensor.rows)
                != storage.value_at(col + row * tensor.rows)
            {
                return Err(conversion_error(
                    name,
                    format!("{name}: covariance matrix must be symmetric"),
                ));
            }
        }
    }
    Ok(())
}

fn covariance_to_correlation(
    name: &'static str,
    covariance: Tensor,
) -> BuiltinResult<(Value, Value)> {
    validate_covariance_matrix(name, &covariance)?;
    let n = covariance.rows;
    let covariance_values = tensor::tensor_values_f64_cow(&covariance);
    let mut sigma = Vec::with_capacity(n);
    for idx in 0..n {
        sigma.push(covariance_values[idx + idx * n].sqrt());
    }

    let mut r = vec![0.0; n * n];
    for col in 0..n {
        for row in 0..n {
            let denom = sigma[row] * sigma[col];
            let idx = row + col * n;
            r[idx] = if denom == 0.0 {
                f64::NAN
            } else {
                covariance_values[idx] / denom
            };
        }
    }

    let dtype = covariance.numeric_dtype();
    let r = Tensor::new_with_dtype(r, vec![n, n], dtype)
        .map(tensor::tensor_into_value)
        .map_err(|err| internal_error(name, format!("{name}: {err}")))?;
    let sigma = Tensor::new_with_dtype(sigma, vec![n, 1], dtype)
        .map(tensor::tensor_into_value)
        .map_err(|err| internal_error(name, format!("{name}: {err}")))?;
    Ok((r, sigma))
}

fn validate_covariance_matrix(name: &'static str, tensor: &Tensor) -> BuiltinResult<()> {
    if tensor.shape.len() > 2 {
        return Err(conversion_error(
            name,
            format!("{name}: covariance matrix must be two-dimensional"),
        ));
    }
    if tensor.rows != tensor.cols {
        return Err(conversion_error(
            name,
            format!("{name}: covariance matrix must be square"),
        ));
    }
    let values = tensor::tensor_values_f64_cow(tensor);
    for &value in values.iter() {
        if !value.is_finite() {
            return Err(conversion_error(
                name,
                format!("{name}: covariance matrix must contain finite values"),
            ));
        }
    }
    for idx in 0..tensor.rows {
        let variance = values[idx + idx * tensor.rows];
        if variance < 0.0 {
            return Err(conversion_error(
                name,
                format!("{name}: covariance matrix diagonal entries must be nonnegative"),
            ));
        }
    }
    for col in 0..tensor.cols {
        for row in 0..col {
            let a = values[row + col * tensor.rows];
            let b = values[col + row * tensor.rows];
            let tol = 1.0e-10 * a.abs().max(b.abs()).max(1.0);
            if (a - b).abs() > tol {
                return Err(conversion_error(
                    name,
                    format!("{name}: covariance matrix must be symmetric"),
                ));
            }
            let variance_row = values[row + row * tensor.rows];
            let variance_col = values[col + col * tensor.rows];
            let max_covariance = (variance_row * variance_col).sqrt();
            let bound_tol = 1.0e-10 * max_covariance.max(a.abs()).max(1.0);
            if a.abs() > max_covariance + bound_tol {
                return Err(conversion_error(
                    name,
                    format!("{name}: covariance magnitude exceeds variance bounds"),
                ));
            }
        }
    }
    validate_positive_semidefinite(name, tensor.rows, &values)?;
    Ok(())
}

fn validate_positive_semidefinite(
    name: &'static str,
    n: usize,
    values: &[f64],
) -> BuiltinResult<()> {
    let scale = (0..n)
        .map(|index| values[index + index * n].abs())
        .fold(1.0_f64, f64::max);
    let tolerance = 1.0e-10 * scale;
    let mut lower = vec![0.0; n * n];
    for row in 0..n {
        for col in 0..=row {
            let mut residual = values[row + col * n];
            for k in 0..col {
                residual -= lower[row * n + k] * lower[col * n + k];
            }
            if row == col {
                if residual < -tolerance {
                    return Err(conversion_error(
                        name,
                        format!("{name}: covariance matrix must be positive semidefinite"),
                    ));
                }
                lower[row * n + col] = residual.max(0.0).sqrt();
            } else {
                let pivot = lower[col * n + col];
                if pivot > tolerance.sqrt() {
                    lower[row * n + col] = residual / pivot;
                } else if residual.abs() > tolerance {
                    return Err(conversion_error(
                        name,
                        format!("{name}: covariance matrix must be positive semidefinite"),
                    ));
                }
            }
        }
    }
    Ok(())
}

fn sigma_as_row(name: &'static str, sigma: Value) -> BuiltinResult<Value> {
    match sigma {
        Value::Tensor(tensor) => {
            let len = tensor.len();
            tensor
                .reshape(vec![1, len])
                .map(tensor::tensor_into_value)
                .map_err(|err| internal_error(name, format!("{name}: {err}")))
        }
        scalar => Ok(scalar),
    }
}

fn output_values(
    kind: ConversionKind,
    name: &'static str,
    r: Value,
    sigma: Value,
) -> BuiltinResult<Value> {
    let (first, second) = match kind {
        ConversionKind::Corrcov => (r, sigma),
        ConversionKind::Cov2corr => (sigma_as_row(name, sigma)?, r),
    };
    match crate::output_count::current_output_count() {
        Some(0) => Ok(Value::OutputList(Vec::new())),
        Some(1) => Ok(Value::OutputList(vec![first])),
        Some(2) => Ok(Value::OutputList(vec![first, second])),
        Some(_) => Err(conversion_error(
            name,
            format!("{name}: too many output arguments; maximum is 2"),
        )),
        None => Ok(first),
    }
}

struct CovarianceGpuEval {
    provider: &'static dyn AccelProvider,
    correlation: GpuTensorHandle,
    sigma: GpuTensorHandle,
}

impl CovarianceGpuEval {
    fn from_provider(
        provider: &'static dyn AccelProvider,
        result: ProviderCovarianceToCorrelationResult,
    ) -> Self {
        Self {
            provider,
            correlation: result.correlation,
            sigma: result.sigma,
        }
    }

    fn output(self, kind: ConversionKind, name: &'static str) -> BuiltinResult<Value> {
        let (first, second) = match kind {
            ConversionKind::Corrcov => (self.correlation, self.sigma),
            ConversionKind::Cov2corr => {
                let len = self.sigma.shape.iter().copied().product::<usize>();
                let sigma = match self.provider.reshape(&self.sigma, &[1, len]) {
                    Ok(handle) => handle,
                    Err(err) => {
                        let _ = self.provider.free(&self.correlation);
                        let _ = self.provider.free(&self.sigma);
                        return Err(conversion_error(name, format!("{name}: {err}")));
                    }
                };
                if sigma.buffer_id != self.sigma.buffer_id {
                    let _ = self.provider.free(&self.sigma);
                }
                (sigma, self.correlation)
            }
        };
        match crate::output_count::current_output_count() {
            Some(0) => {
                let _ = self.provider.free(&first);
                let _ = self.provider.free(&second);
                Ok(Value::OutputList(Vec::new()))
            }
            Some(1) => {
                let _ = self.provider.free(&second);
                Ok(Value::OutputList(vec![gpu_helpers::resident_gpu_value(
                    first,
                )]))
            }
            Some(2) => Ok(Value::OutputList(vec![
                gpu_helpers::resident_gpu_value(first),
                gpu_helpers::resident_gpu_value(second),
            ])),
            Some(_) => {
                let _ = self.provider.free(&first);
                let _ = self.provider.free(&second);
                Err(conversion_error(
                    name,
                    format!("{name}: too many output arguments; maximum is 2"),
                ))
            }
            None => {
                let _ = self.provider.free(&second);
                Ok(gpu_helpers::resident_gpu_value(first))
            }
        }
    }
}

fn provider_is_unsupported(err: &anyhow::Error) -> bool {
    let message = err.to_string();
    message.contains("not supported") || message.contains("unsupported")
}

fn try_covariance_gpu(
    name: &'static str,
    value: &Value,
) -> BuiltinResult<Option<CovarianceGpuEval>> {
    let Value::GpuTensor(handle) = value else {
        return Ok(None);
    };
    if runmat_accelerate_api::handle_storage(handle) == GpuTensorStorage::ComplexInterleaved {
        return Err(conversion_error(
            name,
            format!("{name}: complex covariance matrices are not supported"),
        ));
    }
    if runmat_accelerate_api::handle_integer_type(handle).is_some()
        || runmat_accelerate_api::handle_is_logical(handle)
    {
        return Ok(None);
    }
    let Some(provider) = runmat_accelerate_api::provider_for_handle(handle) else {
        return Ok(None);
    };
    match provider.covariance_to_correlation(handle) {
        Ok(result) => Ok(Some(CovarianceGpuEval::from_provider(provider, result))),
        Err(err) if provider_is_unsupported(&err) => Ok(None),
        Err(err) => Err(conversion_error(name, format!("{name}: {err}"))),
    }
}

async fn covariance_conversion_builtin(
    kind: ConversionKind,
    name: &'static str,
    value: Value,
    rest: Vec<Value>,
) -> BuiltinResult<Value> {
    if !rest.is_empty() {
        return Err(conversion_error(
            name,
            format!("{name}: expected exactly one input argument"),
        ));
    }
    ensure_conversion_extensions(kind, &value)?;
    if matches!(crate::output_count::current_output_count(), Some(0)) {
        return Ok(Value::OutputList(Vec::new()));
    }
    if let Some(eval) = try_covariance_gpu(name, &value)? {
        return eval.output(kind, name);
    }
    let covariance = covariance_tensor(name, value, Vec::new()).await?;
    let (r, sigma) = covariance_to_correlation(name, covariance)?;
    output_values(kind, name, r, sigma)
}

pub mod corrcov {
    use super::*;

    conversion_descriptor!(
        "corrcov",
        CORRCOV_SIGNATURES,
        "Input is nonnumeric, complex, outside the documented single/double domain or declared RunMat extensions, non-square, non-finite, asymmetric, not positive semidefinite, or too many arguments/outputs are supplied."
    );

    #[runmat_macros::register_gpu_spec(
        builtin_path = "crate::builtins::stats::summary::covariance_conversion::corrcov"
    )]
    pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
        name: "corrcov",
        op_kind: GpuOpKind::Custom("summary-stats"),
        supported_precisions: &[ScalarType::F32, ScalarType::F64],
        broadcast: BroadcastSemantics::None,
        provider_hooks: &[ProviderHook::Custom("covariance_to_correlation")],
        constant_strategy: ConstantStrategy::InlineLiteral,
        residency: ResidencyPolicy::NewHandle,
        nan_mode: ReductionNaN::Include,
        two_pass_threshold: None,
        workgroup_size: None,
        accepts_nan_mode: false,
        notes: "Resident real gpuArray covariance matrices use the provider covariance_to_correlation hook and return resident correlation/sigma outputs; unsupported providers fall back to the host reference path.",
    };

    #[runmat_macros::register_fusion_spec(
        builtin_path = "crate::builtins::stats::summary::covariance_conversion::corrcov"
    )]
    pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
        name: "corrcov",
        shape: ShapeRequirements::Any,
        constant_strategy: ConstantStrategy::InlineLiteral,
        elementwise: None,
        reduction: None,
        emits_nan: true,
        notes: "corrcov is a matrix-normalization operation and remains a fusion boundary.",
    };

    #[runtime_builtin(
        name = "corrcov",
        category = "stats/summary",
        summary = "Convert a covariance matrix to a correlation matrix.",
        keywords = "corrcov,covariance,correlation,standard deviation,statistics",
        accel = "sink",
        type_resolver(super::corrcov_type),
        descriptor(self::DESCRIPTOR),
        extensions(CORRCOV_EXTENSIONS),
        integer_capabilities(CORRCOV_INTEGER_CAPABILITIES),
        builtin_path = "crate::builtins::stats::summary::covariance_conversion::corrcov"
    )]
    pub(crate) async fn corrcov_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        covariance_conversion_builtin(ConversionKind::Corrcov, "corrcov", value, rest).await
    }
}

pub mod cov2corr {
    use super::*;

    conversion_descriptor!(
        "cov2corr",
        COV2CORR_SIGNATURES,
        "Input is nonnumeric, complex, outside the documented double domain or declared RunMat extensions, non-square, non-finite, asymmetric, not positive semidefinite, or too many arguments/outputs are supplied."
    );

    #[runmat_macros::register_gpu_spec(
        builtin_path = "crate::builtins::stats::summary::covariance_conversion::cov2corr"
    )]
    pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
        name: "cov2corr",
        op_kind: GpuOpKind::Custom("summary-stats"),
        supported_precisions: &[ScalarType::F32, ScalarType::F64],
        broadcast: BroadcastSemantics::None,
        provider_hooks: &[ProviderHook::Custom("covariance_to_correlation")],
        constant_strategy: ConstantStrategy::InlineLiteral,
        residency: ResidencyPolicy::NewHandle,
        nan_mode: ReductionNaN::Include,
        two_pass_threshold: None,
        workgroup_size: None,
        accepts_nan_mode: false,
        notes: "RunMat-only resident covariance inputs use the provider covariance_to_correlation hook; Finance Toolbox compatibility documents host double input.",
    };

    #[runmat_macros::register_fusion_spec(
        builtin_path = "crate::builtins::stats::summary::covariance_conversion::cov2corr"
    )]
    pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
        name: "cov2corr",
        shape: ShapeRequirements::Any,
        constant_strategy: ConstantStrategy::InlineLiteral,
        elementwise: None,
        reduction: None,
        emits_nan: true,
        notes: "cov2corr is a matrix-normalization operation and remains a fusion boundary.",
    };

    #[runtime_builtin(
        name = "cov2corr",
        category = "stats/summary",
        summary = "Convert a covariance matrix to a correlation matrix.",
        keywords = "cov2corr,covariance,correlation,standard deviation,statistics",
        accel = "sink",
        type_resolver(super::cov2corr_type),
        descriptor(self::DESCRIPTOR),
        extensions(COV2CORR_EXTENSIONS),
        integer_capabilities(COV2CORR_INTEGER_CAPABILITIES),
        builtin_path = "crate::builtins::stats::summary::covariance_conversion::cov2corr"
    )]
    pub(crate) async fn cov2corr_builtin(value: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
        covariance_conversion_builtin(ConversionKind::Cov2corr, "cov2corr", value, rest).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_builtins::IntegerStorage;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "expected {expected}, got {actual}"
        );
    }

    fn int_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Value {
        Value::Tensor(Tensor::new_integer(storage, shape).expect("integer tensor"))
    }

    fn assert_tensor_values(tensor: &Tensor, expected: &[f64]) {
        let values = tensor.materialize_f64();
        assert_eq!(values.len(), expected.len());
        for (actual, expected) in values.into_iter().zip(expected.iter().copied()) {
            assert_close(actual, expected);
        }
    }

    #[test]
    fn corrcov_converts_covariance_and_returns_sigma() {
        let c = Value::Tensor(Tensor::new(vec![4.0, 2.0, 2.0, 9.0], vec![2, 2]).expect("tensor"));
        let _guard = crate::output_count::push_output_count(Some(2));
        let out = block_on(corrcov::corrcov_builtin(c, Vec::new())).expect("corrcov");
        match out {
            Value::OutputList(values) => {
                assert_eq!(values.len(), 2);
                match &values[0] {
                    Value::Tensor(tensor) => {
                        assert_eq!(tensor.shape, vec![2, 2]);
                        assert_tensor_values(tensor, &[1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0]);
                    }
                    other => panic!("expected correlation tensor, got {other:?}"),
                }
                match &values[1] {
                    Value::Tensor(tensor) => {
                        assert_eq!(tensor.shape, vec![2, 1]);
                        assert_tensor_values(tensor, &[2.0, 3.0]);
                    }
                    other => panic!("expected sigma tensor, got {other:?}"),
                }
            }
            other => panic!("expected output list, got {other:?}"),
        }
    }

    #[test]
    fn corrcov_preserves_native_single_outputs() {
        let covariance = Value::Tensor(
            Tensor::from_f32(vec![4.0, 2.0, 2.0, 9.0], vec![2, 2]).expect("single covariance"),
        );
        let _guard = crate::output_count::push_output_count(Some(2));
        let Value::OutputList(outputs) =
            block_on(corrcov::corrcov_builtin(covariance, Vec::new())).expect("corrcov")
        else {
            panic!("expected output list");
        };
        let [Value::Tensor(correlation), Value::Tensor(sigma)] = outputs.as_slice() else {
            panic!("expected tensor outputs");
        };
        assert_eq!(correlation.numeric_dtype(), NumericDType::F32);
        assert_eq!(sigma.numeric_dtype(), NumericDType::F32);
        let third = f64::from(1.0_f32 / 3.0);
        assert_tensor_values(correlation, &[1.0, third, third, 1.0]);
        assert_tensor_values(sigma, &[2.0, 3.0]);
    }

    #[test]
    fn corrcov_gpu_input_returns_resident_outputs() {
        test_support::with_test_provider(|provider| {
            let covariance = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &[4.0, 2.0, 2.0, 9.0],
                    shape: &[2, 2],
                })
                .expect("upload covariance");
            provider.reset_telemetry();

            let _guard = crate::output_count::push_output_count(Some(2));
            let out = block_on(corrcov::corrcov_builtin(
                Value::GpuTensor(covariance),
                Vec::new(),
            ))
            .expect("corrcov gpu");
            let Value::OutputList(values) = out else {
                panic!("expected output list");
            };

            let telemetry = provider.telemetry_snapshot();
            assert_eq!(telemetry.upload_bytes, 0);
            assert_eq!(telemetry.download_bytes, 0);
            assert!(matches!(values[0], Value::GpuTensor(_)));
            assert!(matches!(values[1], Value::GpuTensor(_)));

            let correlation = test_support::gather(values[0].clone()).expect("correlation");
            let sigma = test_support::gather(values[1].clone()).expect("sigma");
            assert_eq!(correlation.shape, vec![2, 2]);
            assert_eq!(sigma.shape, vec![2, 1]);
            assert_tensor_values(&correlation, &[1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0]);
            assert_tensor_values(&sigma, &[2.0, 3.0]);
        });
    }

    #[test]
    fn cov2corr_gpu_input_preserves_invalid_covariance_error() {
        test_support::with_test_provider(|provider| {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let covariance = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &[1.0, 0.1, 0.2, 1.0],
                    shape: &[2, 2],
                })
                .expect("upload covariance");

            let err = block_on(cov2corr::cov2corr_builtin(
                Value::GpuTensor(covariance),
                Vec::new(),
            ))
            .expect_err("invalid covariance");

            assert_eq!(err.identifier(), Some("RunMat:cov2corr:InvalidArgument"));
            assert!(err.message().contains("symmetric"));
        });
    }

    #[test]
    fn cov2corr_gpu_extension_returns_finance_output_order() {
        test_support::with_test_provider(|provider| {
            let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
            let covariance = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &[4.0, 2.0, 2.0, 9.0],
                    shape: &[2, 2],
                })
                .expect("upload covariance");
            let _outputs = crate::output_count::push_output_count(Some(2));
            let Value::OutputList(outputs) = block_on(cov2corr::cov2corr_builtin(
                Value::GpuTensor(covariance),
                Vec::new(),
            ))
            .expect("cov2corr gpu") else {
                panic!("expected output list");
            };
            let sigma = test_support::gather(outputs[0].clone()).expect("sigma");
            let correlation = test_support::gather(outputs[1].clone()).expect("correlation");
            assert_eq!(sigma.shape, vec![1, 2]);
            assert_tensor_values(&sigma, &[2.0, 3.0]);
            assert_eq!(correlation.shape, vec![2, 2]);
            assert_tensor_values(&correlation, &[1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0]);
        });
    }

    #[test]
    fn cov2corr_returns_finance_output_order_and_shapes() {
        let c = Value::Tensor(Tensor::new(vec![16.0, 4.0, 4.0, 25.0], vec![2, 2]).expect("tensor"));
        let _outputs = crate::output_count::push_output_count(Some(2));
        let Value::OutputList(outputs) =
            block_on(cov2corr::cov2corr_builtin(c, Vec::new())).expect("cov2corr")
        else {
            panic!("expected output list");
        };
        let [Value::Tensor(sigma), Value::Tensor(correlation)] = outputs.as_slice() else {
            panic!("expected tensor outputs");
        };
        assert_eq!(sigma.shape, vec![1, 2]);
        assert_tensor_values(sigma, &[4.0, 5.0]);
        assert_eq!(correlation.shape, vec![2, 2]);
        assert_tensor_values(correlation, &[1.0, 0.2, 0.2, 1.0]);
    }

    #[test]
    fn covariance_conversion_preserves_empty_shapes_and_cov2corr_default_output() {
        let empty = || Value::Tensor(Tensor::new(Vec::new(), vec![0, 0]).expect("empty tensor"));
        {
            let _outputs = crate::output_count::push_output_count(Some(2));
            let Value::OutputList(outputs) =
                block_on(corrcov::corrcov_builtin(empty(), Vec::new())).expect("corrcov empty")
            else {
                panic!("expected output list");
            };
            let [Value::Tensor(correlation), Value::Tensor(sigma)] = outputs.as_slice() else {
                panic!("expected tensor outputs");
            };
            assert_eq!(correlation.shape, vec![0, 0]);
            assert_eq!(sigma.shape, vec![0, 1]);
        }
        {
            let _outputs = crate::output_count::push_output_count(Some(2));
            let Value::OutputList(outputs) =
                block_on(cov2corr::cov2corr_builtin(empty(), Vec::new())).expect("cov2corr empty")
            else {
                panic!("expected output list");
            };
            let [Value::Tensor(sigma), Value::Tensor(correlation)] = outputs.as_slice() else {
                panic!("expected tensor outputs");
            };
            assert_eq!(sigma.shape, vec![1, 0]);
            assert_eq!(correlation.shape, vec![0, 0]);
        }

        let covariance =
            Value::Tensor(Tensor::new(vec![16.0, 4.0, 4.0, 25.0], vec![2, 2]).expect("tensor"));
        let Value::Tensor(sigma) =
            block_on(cov2corr::cov2corr_builtin(covariance, Vec::new())).expect("cov2corr")
        else {
            panic!("expected default sigma output");
        };
        assert_eq!(sigma.shape, vec![1, 2]);
        assert_tensor_values(&sigma, &[4.0, 5.0]);
    }

    #[test]
    fn corrcov_zero_variance_produces_nan_correlations() {
        let c = Value::Tensor(Tensor::new(vec![0.0, 0.0, 0.0, 9.0], vec![2, 2]).expect("tensor"));
        let out = block_on(corrcov::corrcov_builtin(c, Vec::new())).expect("corrcov");
        match out {
            Value::Tensor(tensor) => {
                let values = tensor.materialize_f64();
                assert!(values[0].is_nan());
                assert!(values[1].is_nan());
                assert!(values[2].is_nan());
                assert_close(values[3], 1.0);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn corrcov_rejects_invalid_covariance_matrix() {
        let nonsquare = Value::Tensor(
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).expect("tensor"),
        );
        let err = block_on(corrcov::corrcov_builtin(nonsquare, Vec::new())).unwrap_err();
        assert_eq!(err.identifier(), Some("RunMat:corrcov:InvalidArgument"));
        assert!(err.message().contains("square"));

        let negative =
            Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, -1.0], vec![2, 2]).expect("tensor"));
        let err = block_on(corrcov::corrcov_builtin(negative, Vec::new())).unwrap_err();
        assert!(err.message().contains("nonnegative"));

        let asymmetric =
            Value::Tensor(Tensor::new(vec![1.0, 0.1, 0.2, 1.0], vec![2, 2]).expect("tensor"));
        let err = block_on(corrcov::corrcov_builtin(asymmetric, Vec::new())).unwrap_err();
        assert!(err.message().contains("symmetric"));

        let one_sided_nan =
            Value::Tensor(Tensor::new(vec![1.0, 2.0, f64::NAN, 1.0], vec![2, 2]).expect("tensor"));
        let err = block_on(corrcov::corrcov_builtin(one_sided_nan, Vec::new())).unwrap_err();
        assert!(err.message().contains("finite"));

        let impossible =
            Value::Tensor(Tensor::new(vec![1.0, 2.0, 2.0, 1.0], vec![2, 2]).expect("tensor"));
        let err = block_on(corrcov::corrcov_builtin(impossible, Vec::new())).unwrap_err();
        assert!(err.message().contains("variance bounds"));

        let indefinite = Value::Tensor(
            Tensor::new(
                vec![
                    1.0, 0.9, 0.9, //
                    0.9, 1.0, -0.9, //
                    0.9, -0.9, 1.0,
                ],
                vec![3, 3],
            )
            .unwrap(),
        );
        let err = block_on(corrcov::corrcov_builtin(indefinite, Vec::new())).unwrap_err();
        assert!(err.message().contains("positive semidefinite"));
    }

    #[test]
    fn corrcov_accepts_typed_integer_tensors_and_scalar_logical_extensions() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        {
            let integer = int_tensor(IntegerStorage::U16(vec![4, 2, 2, 9]), vec![2, 2]);
            let _guard = crate::output_count::push_output_count(Some(2));
            let out = block_on(corrcov::corrcov_builtin(integer, Vec::new())).unwrap();
            match out {
                Value::OutputList(values) => {
                    assert_eq!(values.len(), 2);
                    match &values[0] {
                        Value::Tensor(tensor) => {
                            assert_eq!(tensor.numeric_dtype(), NumericDType::F64);
                            assert_eq!(tensor.shape, vec![2, 2]);
                            assert_tensor_values(tensor, &[1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0]);
                        }
                        other => panic!("expected correlation tensor, got {other:?}"),
                    }
                    match &values[1] {
                        Value::Tensor(tensor) => {
                            assert_eq!(tensor.numeric_dtype(), NumericDType::F64);
                            assert_eq!(tensor.shape, vec![2, 1]);
                            assert_tensor_values(tensor, &[2.0, 3.0]);
                        }
                        other => panic!("expected sigma tensor, got {other:?}"),
                    }
                }
                other => panic!("expected output list, got {other:?}"),
            }
        }

        let scalar = block_on(corrcov::corrcov_builtin(Value::Num(4.0), Vec::new())).unwrap();
        match scalar {
            Value::Num(value) => assert_close(value, 1.0),
            other => panic!("expected scalar result, got {other:?}"),
        }

        let logical = Value::LogicalArray(
            runmat_builtins::LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).expect("logical"),
        );
        let out = block_on(corrcov::corrcov_builtin(logical, Vec::new())).unwrap();
        match out {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.numeric_dtype(), NumericDType::F64);
                assert_eq!(tensor.shape, vec![2, 2]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn covariance_conversion_extensions_reject_in_matlab_mode() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
        let integer = int_tensor(IntegerStorage::U16(vec![4, 2, 2, 9]), vec![2, 2]);
        let err = block_on(corrcov::corrcov_builtin(integer.clone(), Vec::new())).unwrap_err();
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:CorrcovIntegerDataExtension")
        );
        let err = block_on(cov2corr::cov2corr_builtin(integer, Vec::new())).unwrap_err();
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:Cov2corrIntegerDataExtension")
        );

        let logical = Value::LogicalArray(
            runmat_builtins::LogicalArray::new(vec![1, 0, 0, 1], vec![2, 2]).unwrap(),
        );
        let err = block_on(corrcov::corrcov_builtin(logical.clone(), Vec::new())).unwrap_err();
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:CorrcovLogicalDataExtension")
        );
        let err = block_on(cov2corr::cov2corr_builtin(logical, Vec::new())).unwrap_err();
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:Cov2corrLogicalDataExtension")
        );

        let single = Value::Tensor(Tensor::from_f32(vec![4.0], vec![1, 1]).unwrap());
        let err = block_on(cov2corr::cov2corr_builtin(single, Vec::new())).unwrap_err();
        assert_eq!(
            err.identifier(),
            Some("RunMat:compatibility:Cov2corrSingleDataExtension")
        );
    }

    #[test]
    fn covariance_conversion_supports_all_integer_classes_in_runmat_mode() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let storages = vec![
            IntegerStorage::I8(vec![4, 2, 2, 9]),
            IntegerStorage::I16(vec![4, 2, 2, 9]),
            IntegerStorage::I32(vec![4, 2, 2, 9]),
            IntegerStorage::I64(vec![4, 2, 2, 9]),
            IntegerStorage::U8(vec![4, 2, 2, 9]),
            IntegerStorage::U16(vec![4, 2, 2, 9]),
            IntegerStorage::U32(vec![4, 2, 2, 9]),
            IntegerStorage::U64(vec![4, 2, 2, 9]),
        ];
        for storage in storages {
            let covariance = int_tensor(storage, vec![2, 2]);
            let Value::Tensor(correlation) =
                block_on(corrcov::corrcov_builtin(covariance.clone(), Vec::new()))
                    .expect("integer corrcov")
            else {
                panic!("expected correlation tensor");
            };
            assert_eq!(correlation.numeric_dtype(), NumericDType::F64);
            assert_tensor_values(&correlation, &[1.0, 1.0 / 3.0, 1.0 / 3.0, 1.0]);

            let Value::Tensor(sigma) = block_on(cov2corr::cov2corr_builtin(covariance, Vec::new()))
                .expect("integer cov2corr")
            else {
                panic!("expected sigma tensor");
            };
            assert_eq!(sigma.numeric_dtype(), NumericDType::F64);
            assert_eq!(sigma.shape, vec![1, 2]);
            assert_tensor_values(&sigma, &[2.0, 3.0]);
        }
    }

    #[test]
    fn covariance_conversion_checks_wide_integer_symmetry_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let base = u64::MAX;
        let symmetric = int_tensor(
            IntegerStorage::U64(vec![base, base - 1, base - 1, base]),
            vec![2, 2],
        );
        let Value::Tensor(result) =
            block_on(corrcov::corrcov_builtin(symmetric, Vec::new())).expect("wide symmetric")
        else {
            panic!("expected tensor");
        };
        assert_eq!(result.shape, vec![2, 2]);

        let asymmetric = int_tensor(
            IntegerStorage::U64(vec![base, base - 1, base - 2, base]),
            vec![2, 2],
        );
        let err = block_on(corrcov::corrcov_builtin(asymmetric, Vec::new())).unwrap_err();
        assert!(err.message().contains("symmetric"));
    }

    #[test]
    fn covariance_conversion_resident_extension_gates_precede_dispatch() {
        test_support::with_test_provider(|provider| {
            let integer =
                Tensor::new_integer(IntegerStorage::U16(vec![4, 2, 2, 9]), vec![2, 2]).unwrap();
            let integer_handle =
                gpu_helpers::upload_tensor(provider, &integer).expect("integer upload");
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
                let err = block_on(corrcov::corrcov_builtin(
                    Value::GpuTensor(integer_handle.clone()),
                    Vec::new(),
                ))
                .unwrap_err();
                assert_eq!(
                    err.identifier(),
                    Some("RunMat:compatibility:CorrcovIntegerDataExtension")
                );
                let err = block_on(cov2corr::cov2corr_builtin(
                    Value::GpuTensor(integer_handle.clone()),
                    Vec::new(),
                ))
                .unwrap_err();
                assert_eq!(
                    err.identifier(),
                    Some("RunMat:compatibility:Cov2corrIntegerDataExtension")
                );
            }
            {
                let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
                let result = block_on(corrcov::corrcov_builtin(
                    Value::GpuTensor(integer_handle.clone()),
                    Vec::new(),
                ))
                .expect("resident integer gather fallback");
                assert!(matches!(result, Value::Tensor(_)));
            }
            let _ = provider.free(&integer_handle);

            let double_handle = provider
                .upload(&runmat_accelerate_api::HostTensorView {
                    data: &[4.0, 2.0, 2.0, 9.0],
                    shape: &[2, 2],
                })
                .unwrap();
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let err = block_on(cov2corr::cov2corr_builtin(
                Value::GpuTensor(double_handle.clone()),
                Vec::new(),
            ))
            .unwrap_err();
            assert_eq!(
                err.identifier(),
                Some("RunMat:compatibility:Cov2corrGpuInputExtension")
            );
            let _ = provider.free(&double_handle);
        });
    }

    #[test]
    fn corrcov_rejects_too_many_outputs() {
        let c = Value::Tensor(Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).expect("tensor"));
        let _guard = crate::output_count::push_output_count(Some(3));
        let err = block_on(corrcov::corrcov_builtin(c, Vec::new())).unwrap_err();
        assert!(err.message().contains("too many output"));
    }
}
