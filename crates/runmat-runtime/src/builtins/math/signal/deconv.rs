//! MATLAB-compatible `deconv` builtin with GPU-aware semantics for RunMat.

use num_complex::Complex;
use runmat_builtins::{
    BuiltinCompletionPolicy, BuiltinDescriptor, BuiltinErrorDescriptor, BuiltinExtensionDescriptor,
    BuiltinExtensionMode, BuiltinIntegerBackendRule, BuiltinIntegerCapabilityDescriptor,
    BuiltinIntegerComputationDomain, BuiltinIntegerInputAvailability,
    BuiltinIntegerInputCapability, BuiltinIntegerOutputClassRule, BuiltinIntegerOverflowRule,
    BuiltinIntegerOverloadKind, BuiltinIntegerScalarDoubleRule, BuiltinOutputMode,
    BuiltinParamArity, BuiltinParamDescriptor, BuiltinParamType, BuiltinSignatureDescriptor,
    ComplexStorage, ComplexTensor, LogicalArray, NumericDType, Tensor, Value,
};
use runmat_macros::runtime_builtin;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::signal::type_resolvers::deconv_type;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::math::signal::deconv")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "deconv",
    op_kind: GpuOpKind::Custom("deconv1d"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "RunMat gathers coefficients through their owning provider for host polynomial division, then validates and restores real or complex outputs to that same owner/device.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::math::signal::deconv")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "deconv",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Polynomial division is not part of current fusion pipelines; metadata is present for completeness.",
};

const BUILTIN_NAME: &str = "deconv";

const DECONV_INTEGER_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "deconv-integer-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "deconv with typed-integer coefficients is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DeconvIntegerInputExtension"),
};
const DECONV_LOGICAL_INPUT_EXTENSION: BuiltinExtensionDescriptor = BuiltinExtensionDescriptor {
    id: "deconv-logical-input",
    mode: BuiltinExtensionMode::RunMatOnly,
    description: "deconv with logical coefficients is a RunMat extension",
    error_identifier: Some("RunMat:compatibility:DeconvLogicalInputExtension"),
};
const DECONV_EXTENSIONS: [BuiltinExtensionDescriptor; 2] = [
    DECONV_INTEGER_INPUT_EXTENSION,
    DECONV_LOGICAL_INPUT_EXTENSION,
];
const DECONV_INTEGER_INPUTS: [BuiltinIntegerInputCapability; 2] = [
    BuiltinIntegerInputCapability {
        name: "numerator",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double coefficients; RunMat mode admits exactly representable real integer coefficients.",
    },
    BuiltinIntegerInputCapability {
        name: "denominator",
        classes: &crate::builtins::common::integer_capability::ALL_INTEGER_CLASSES,
        availability: BuiltinIntegerInputAvailability::RunMatOnly,
        scalar_double: BuiltinIntegerScalarDoubleRule::NotApplicable,
        notes: "The compatibility target documents single and double coefficients; RunMat mode admits exactly representable real integer coefficients.",
    },
];
pub const INTEGER_CAPABILITIES: [BuiltinIntegerCapabilityDescriptor; 1] =
    [BuiltinIntegerCapabilityDescriptor {
        form: "[Q,R] = deconv(integer_numerator, integer_denominator)",
        inputs: &DECONV_INTEGER_INPUTS,
        computation_domain: BuiltinIntegerComputationDomain::FloatingPoint,
        output_class: BuiltinIntegerOutputClassRule::Double,
        overflow: BuiltinIntegerOverflowRule::NotApplicable,
        backend: BuiltinIntegerBackendRule::GatherFallback,
        overload: BuiltinIntegerOverloadKind::FunctionSpecific,
        notes: "Each authoritative coefficient is checked for exact binary64 representation before polynomial division; resident inputs gather through their owners and results return to the first resident owner.",
    }];

const DECONV_OUTPUT_Q: [BuiltinParamDescriptor; 1] = [BuiltinParamDescriptor {
    name: "Q",
    ty: BuiltinParamType::NumericArray,
    arity: BuiltinParamArity::Required,
    default: None,
    description: "Quotient polynomial coefficients.",
}];

const DECONV_OUTPUT_QR: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "Q",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Quotient polynomial coefficients.",
    },
    BuiltinParamDescriptor {
        name: "R",
        ty: BuiltinParamType::NumericArray,
        arity: BuiltinParamArity::Optional,
        default: None,
        description: "Remainder polynomial coefficients.",
    },
];

const DECONV_INPUTS: [BuiltinParamDescriptor; 2] = [
    BuiltinParamDescriptor {
        name: "numerator",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Numerator coefficients.",
    },
    BuiltinParamDescriptor {
        name: "denominator",
        ty: BuiltinParamType::Any,
        arity: BuiltinParamArity::Required,
        default: None,
        description: "Denominator coefficients.",
    },
];

const DECONV_SIGNATURES: [BuiltinSignatureDescriptor; 2] = [
    BuiltinSignatureDescriptor {
        label: "Q = deconv(numerator, denominator)",
        inputs: &DECONV_INPUTS,
        outputs: &DECONV_OUTPUT_Q,
    },
    BuiltinSignatureDescriptor {
        label: "[Q, R] = deconv(numerator, denominator)",
        inputs: &DECONV_INPUTS,
        outputs: &DECONV_OUTPUT_QR,
    },
];

const DECONV_ERROR_INVALID_INPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DECONV.INVALID_INPUT",
    identifier: Some("RunMat:deconv:InvalidInput"),
    when: "Numerator/denominator input value is not supported for polynomial conversion.",
    message: "deconv: unsupported input type",
};

const DECONV_ERROR_VECTOR_REQUIRED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DECONV.VECTOR_REQUIRED",
    identifier: Some("RunMat:deconv:VectorRequired"),
    when: "Input is not scalar/row/column vector.",
    message: "deconv: inputs must be scalars, row vectors, or column vectors",
};

const DECONV_ERROR_DENOMINATOR_INVALID: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DECONV.DENOMINATOR_INVALID",
    identifier: Some("RunMat:deconv:DenominatorInvalid"),
    when: "Denominator is empty or contains only exact zero coefficients.",
    message: "deconv: denominator is invalid",
};

const DECONV_ERROR_GATHER_FAILED: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DECONV.GATHER_FAILED",
    identifier: Some("RunMat:deconv:GatherFailed"),
    when: "GPU input cannot be gathered for host fallback normalization.",
    message: "deconv: failed to gather GPU input",
};

const DECONV_ERROR_BUILD_COMPLEX_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DECONV.BUILD_COMPLEX_OUTPUT",
    identifier: Some("RunMat:deconv:BuildComplexOutput"),
    when: "Complex output tensor allocation fails.",
    message: "deconv: failed to build complex tensor",
};

const DECONV_ERROR_BUILD_OUTPUT: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DECONV.BUILD_OUTPUT",
    identifier: Some("RunMat:deconv:BuildOutput"),
    when: "Real output tensor allocation fails.",
    message: "deconv: failed to build tensor",
};
const DECONV_ERROR_TOO_MANY_OUTPUTS: BuiltinErrorDescriptor = BuiltinErrorDescriptor {
    code: "RM.DECONV.TOO_MANY_OUTPUTS",
    identifier: Some("RunMat:deconv:TooManyOutputs"),
    when: "More than two outputs are requested.",
    message: "deconv: too many output arguments",
};

const DECONV_ERRORS: [BuiltinErrorDescriptor; 7] = [
    DECONV_ERROR_INVALID_INPUT,
    DECONV_ERROR_VECTOR_REQUIRED,
    DECONV_ERROR_DENOMINATOR_INVALID,
    DECONV_ERROR_GATHER_FAILED,
    DECONV_ERROR_BUILD_COMPLEX_OUTPUT,
    DECONV_ERROR_BUILD_OUTPUT,
    DECONV_ERROR_TOO_MANY_OUTPUTS,
];

pub const DECONV_DESCRIPTOR: BuiltinDescriptor = BuiltinDescriptor {
    signatures: &DECONV_SIGNATURES,
    output_mode: BuiltinOutputMode::ByRequestedOutputCount,
    completion_policy: BuiltinCompletionPolicy::Public,
    errors: &DECONV_ERRORS,
};

fn deconv_error(error: &'static BuiltinErrorDescriptor) -> RuntimeError {
    deconv_error_with_message(error.message, error)
}

fn deconv_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    deconv_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn deconv_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn deconv_error_with_source(
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

#[runtime_builtin(
    name = "deconv",
    category = "math/signal",
    summary = "Compute one-dimensional deconvolution.",
    keywords = "deconv,deconvolution,polynomial division,signal,gpu",
    accel = "custom",
    type_resolver(deconv_type),
    extensions(DECONV_EXTENSIONS),
    integer_capabilities(INTEGER_CAPABILITIES),
    descriptor(crate::builtins::math::signal::deconv::DECONV_DESCRIPTOR),
    builtin_path = "crate::builtins::math::signal::deconv"
)]
async fn deconv_builtin(numerator: Value, denominator: Value) -> crate::BuiltinResult<Value> {
    if matches!(crate::output_count::current_output_count(), Some(count) if count > 2) {
        return Err(build_runtime_error("deconv: too many output arguments")
            .with_builtin(BUILTIN_NAME)
            .with_identifier("RunMat:deconv:TooManyOutputs")
            .build());
    }
    let eval = evaluate(numerator, denominator).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            free_deconv_output(&eval.quotient);
            free_deconv_output(&eval.remainder);
            return Ok(Value::OutputList(Vec::new()));
        }
        let mut outputs = vec![eval.quotient()];
        if out_count >= 2 {
            outputs.push(eval.remainder());
        } else {
            free_deconv_output(&eval.remainder);
        }
        return Ok(Value::OutputList(outputs));
    }
    let quotient = eval.quotient();
    free_deconv_output(&eval.remainder);
    Ok(quotient)
}

fn free_deconv_output(value: &Value) {
    if let Value::GpuTensor(handle) = value {
        if let Some(owner) = runmat_accelerate_api::provider_for_handle(handle) {
            let _ = owner.free(handle);
        }
    }
}

/// Evaluate `deconv` and retain both outputs for multi-value contexts.
pub async fn evaluate(numerator: Value, denominator: Value) -> BuiltinResult<DeconvEval> {
    ensure_extensions(&numerator)?;
    ensure_extensions(&denominator)?;
    crate::builtins::math::trigonometry::inverse_helpers::ensure_integer_exact_f64(
        &numerator,
        BUILTIN_NAME,
    )?;
    crate::builtins::math::trigonometry::inverse_helpers::ensure_integer_exact_f64(
        &denominator,
        BUILTIN_NAME,
    )?;
    crate::builtins::common::validation::reject_typed_complex_integer(&numerator, BUILTIN_NAME)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&denominator, BUILTIN_NAME)?;
    let integer_or_logical = [&numerator, &denominator]
        .iter()
        .any(|value| value_is_integer_or_logical(value));
    let single_output =
        !integer_or_logical && (value_is_single(&numerator) || value_is_single(&denominator));
    let complex_output = value_is_complex(&numerator) || value_is_complex(&denominator);
    let resident_prototype = [&numerator, &denominator]
        .iter()
        .find_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        });
    let provider = resident_prototype
        .as_ref()
        .and_then(runmat_accelerate_api::provider_for_handle);
    if [&numerator, &denominator]
        .iter()
        .any(|value| matches!(value, Value::GpuTensor(_)))
        && provider.is_none()
    {
        return Err(deconv_error_with_detail(
            &DECONV_ERROR_GATHER_FAILED,
            "resident input has no owning provider",
        ));
    }
    if let (Some(provider), Some(prototype)) = (provider, resident_prototype.as_ref()) {
        for value in [&numerator, &denominator] {
            if let Value::GpuTensor(handle) = value {
                let owner =
                    runmat_accelerate_api::provider_for_handle(handle).ok_or_else(|| {
                        deconv_error_with_detail(
                            &DECONV_ERROR_GATHER_FAILED,
                            "resident input has no owning provider",
                        )
                    })?;
                if !std::ptr::eq(owner, provider) || handle.device_id != prototype.device_id {
                    return Err(deconv_error_with_detail(
                        &DECONV_ERROR_INVALID_INPUT,
                        "resident inputs must share one owning provider and device",
                    ));
                }
            }
        }
    }
    let num_input = convert_value(numerator).await?;
    let den_input = convert_value(denominator).await?;

    let (quotient_raw, remainder_raw) = polynomial_division(&num_input.data, &den_input.data)?;

    let orientation = orientation_from_hint(num_input.hint);
    let quotient = convert_output(quotient_raw, orientation, single_output, complex_output)?;
    let remainder = convert_output(remainder_raw, orientation, single_output, complex_output)?;
    let (quotient, remainder) = if let (Some(provider), Some(prototype)) =
        (provider, resident_prototype.as_ref())
    {
        let quotient = if integer_or_logical {
            quotient
        } else {
            crate::builtins::math::trigonometry::inverse_helpers::align_floating_value_precision(
                quotient,
                prototype,
                BUILTIN_NAME,
            )?
        };
        let remainder = if integer_or_logical {
            remainder
        } else {
            crate::builtins::math::trigonometry::inverse_helpers::align_floating_value_precision(
                remainder,
                prototype,
                BUILTIN_NAME,
            )?
        };
        let quotient = crate::builtins::math::trigonometry::inverse_helpers::upload_value_like(
            provider,
            quotient,
            BUILTIN_NAME,
            prototype,
        )?;
        let remainder =
            match crate::builtins::math::trigonometry::inverse_helpers::upload_value_like(
                provider,
                remainder,
                BUILTIN_NAME,
                prototype,
            ) {
                Ok(value) => value,
                Err(error) => {
                    if let Value::GpuTensor(handle) = &quotient {
                        let owner =
                            runmat_accelerate_api::provider_for_handle(handle).unwrap_or(provider);
                        let _ = owner.free(handle);
                    }
                    return Err(error);
                }
            };
        (quotient, remainder)
    } else {
        (quotient, remainder)
    };

    Ok(DeconvEval {
        quotient,
        remainder,
    })
}

/// Evaluation envelope used by both builtin and bytecode multi-output paths.
#[derive(Clone, Debug)]
pub struct DeconvEval {
    quotient: Value,
    remainder: Value,
}

impl DeconvEval {
    /// Quotient polynomial (`q`).
    pub fn quotient(&self) -> Value {
        self.quotient.clone()
    }

    /// Remainder polynomial (`r`).
    pub fn remainder(&self) -> Value {
        self.remainder.clone()
    }
}

#[derive(Clone)]
struct PolyInput {
    data: Vec<Complex<f64>>,
    hint: OrientationHint,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OrientationHint {
    Row,
    Column,
    Scalar,
    General,
    Empty,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Orientation {
    Row,
    Column,
}

#[async_recursion::async_recursion(?Send)]
async fn convert_value(value: Value) -> BuiltinResult<PolyInput> {
    match value {
        Value::GpuTensor(handle) => {
            let gathered = gpu_helpers::gather_value_async(&Value::GpuTensor(handle.clone()))
                .await
                .map_err(|flow| {
                    let message = flow.message().to_owned();
                    deconv_error_with_source(
                        &DECONV_ERROR_GATHER_FAILED,
                        message,
                        map_control_flow_with_builtin(flow, BUILTIN_NAME),
                    )
                })?;
            crate::builtins::math::trigonometry::inverse_helpers::ensure_integer_exact_f64(
                &gathered,
                BUILTIN_NAME,
            )?;
            convert_value(gathered).await
        }
        Value::Tensor(tensor) => convert_tensor(tensor),
        Value::ComplexTensor(tensor) => convert_complex_tensor(tensor),
        Value::LogicalArray(logical) => convert_logical_array(logical),
        Value::Num(n) => Ok(PolyInput {
            data: vec![Complex::new(n, 0.0)],
            hint: OrientationHint::Scalar,
        }),
        Value::Int(int_val) => {
            let num = int_val.to_f64();
            Ok(PolyInput {
                data: vec![Complex::new(num, 0.0)],
                hint: OrientationHint::Scalar,
            })
        }
        Value::Bool(flag) => Ok(PolyInput {
            data: vec![Complex::new(if flag { 1.0 } else { 0.0 }, 0.0)],
            hint: OrientationHint::Scalar,
        }),
        Value::Complex(re, im) => Ok(PolyInput {
            data: vec![Complex::new(re, im)],
            hint: OrientationHint::Scalar,
        }),
        other => Err(deconv_error_with_detail(
            &DECONV_ERROR_INVALID_INPUT,
            format!("{other:?}"),
        )),
    }
}

fn convert_tensor(tensor: Tensor) -> BuiltinResult<PolyInput> {
    let rows = tensor.rows;
    let cols = tensor.cols;
    let data = tensor::tensor_into_values_f64(tensor);
    let len = data.len();
    let hint = classify_orientation(rows, cols, len);
    ensure_vector(hint)?;
    let data = data.into_iter().map(|re| Complex::new(re, 0.0)).collect();
    Ok(PolyInput { data, hint })
}

fn convert_complex_tensor(tensor: ComplexTensor) -> BuiltinResult<PolyInput> {
    let rows = tensor.rows;
    let cols = tensor.cols;
    let data = tensor::complex_tensor_into_values_complex64(tensor);
    let len = data.len();
    let hint = classify_orientation(rows, cols, len);
    ensure_vector(hint)?;
    Ok(PolyInput { data, hint })
}

fn convert_logical_array(array: LogicalArray) -> BuiltinResult<PolyInput> {
    let hint = classify_orientation(
        array.shape.first().copied().unwrap_or(0),
        array.shape.get(1).copied().unwrap_or(0),
        array.data.len(),
    );
    ensure_vector(hint)?;
    let data = array
        .data
        .into_iter()
        .map(|bit| Complex::new(if bit != 0 { 1.0 } else { 0.0 }, 0.0))
        .collect();
    Ok(PolyInput { data, hint })
}

fn ensure_vector(hint: OrientationHint) -> BuiltinResult<()> {
    if matches!(hint, OrientationHint::General) {
        Err(deconv_error(&DECONV_ERROR_VECTOR_REQUIRED))
    } else {
        Ok(())
    }
}

fn ensure_extensions(value: &Value) -> BuiltinResult<()> {
    let integer = matches!(value, Value::Int(_))
        || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some());
    if integer {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DECONV_INTEGER_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    let logical = matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_is_logical(handle));
    if logical {
        crate::compatibility::ensure_builtin_extension_enabled(
            &DECONV_LOGICAL_INPUT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

fn value_is_single(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor.numeric_dtype() == NumericDType::F32,
        Value::ComplexTensor(tensor) => tensor.numeric_dtype() == NumericDType::F32,
        Value::GpuTensor(handle)
            if runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle) =>
        {
            runmat_accelerate_api::handle_precision(handle)
                == Some(runmat_accelerate_api::ProviderPrecision::F32)
        }
        _ => false,
    }
}

fn value_is_integer_or_logical(value: &Value) -> bool {
    matches!(
        value,
        Value::Int(_) | Value::Bool(_) | Value::LogicalArray(_)
    ) || matches!(value, Value::Tensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some() || runmat_accelerate_api::handle_is_logical(handle))
}

fn value_is_complex(value: &Value) -> bool {
    matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_))
        || matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_storage(handle) == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved)
}

fn classify_orientation(rows: usize, cols: usize, len: usize) -> OrientationHint {
    if len == 0 {
        OrientationHint::Empty
    } else if rows == 1 && cols == 1 {
        OrientationHint::Scalar
    } else if rows == 1 {
        OrientationHint::Row
    } else if cols == 1 {
        OrientationHint::Column
    } else {
        OrientationHint::General
    }
}

fn orientation_from_hint(hint: OrientationHint) -> Orientation {
    match hint {
        OrientationHint::Column => Orientation::Column,
        OrientationHint::Row | OrientationHint::Scalar | OrientationHint::Empty => Orientation::Row,
        OrientationHint::General => Orientation::Column,
    }
}

type PolyDivision = (Vec<Complex<f64>>, Vec<Complex<f64>>);

fn polynomial_division(
    numerator: &[Complex<f64>],
    denominator: &[Complex<f64>],
) -> BuiltinResult<PolyDivision> {
    if denominator.is_empty() {
        return Err(deconv_error_with_detail(
            &DECONV_ERROR_DENOMINATOR_INVALID,
            "must not be empty",
        ));
    }

    let (den_trim, _) = trim_leading_zeros(denominator);
    if den_trim.is_empty() {
        return Err(deconv_error_with_detail(
            &DECONV_ERROR_DENOMINATOR_INVALID,
            "must contain at least one non-zero coefficient",
        ));
    }

    let (num_trim, num_all_zero) = trim_leading_zeros(numerator);
    if num_all_zero {
        return Ok((vec![Complex::new(0.0, 0.0)], numerator.to_vec()));
    }

    if num_trim.len() < den_trim.len() {
        return Ok((vec![Complex::new(0.0, 0.0)], numerator.to_vec()));
    }

    let divisor_lead = den_trim[0];

    let q_len = num_trim.len() - den_trim.len() + 1;
    let mut quotient = vec![Complex::new(0.0, 0.0); q_len];
    let mut working = num_trim.clone();

    for k in 0..q_len {
        let coeff = working[k] / divisor_lead;
        quotient[k] = coeff;
        for j in 0..den_trim.len() {
            working[k + j] -= coeff * den_trim[j];
        }
    }

    let product = convolve_complex(den_trim.as_slice(), quotient.as_slice());
    let mut remainder = numerator.to_vec();
    let offset = remainder.len().saturating_sub(product.len());
    for (index, value) in product.into_iter().enumerate() {
        remainder[offset + index] -= value;
    }

    Ok((quotient, remainder))
}

fn trim_leading_zeros(data: &[Complex<f64>]) -> (Vec<Complex<f64>>, bool) {
    if data.is_empty() {
        return (Vec::new(), true);
    }
    let first_non_zero = data.iter().position(|c| !is_exact_zero(c));
    match first_non_zero {
        Some(idx) => {
            let trimmed = data[idx..].to_vec();
            (trimmed, false)
        }
        None => (Vec::new(), true),
    }
}

fn convolve_complex(lhs: &[Complex<f64>], rhs: &[Complex<f64>]) -> Vec<Complex<f64>> {
    if lhs.is_empty() || rhs.is_empty() {
        return Vec::new();
    }
    let mut output = vec![Complex::new(0.0, 0.0); lhs.len() + rhs.len() - 1];
    for (i, left) in lhs.iter().enumerate() {
        for (j, right) in rhs.iter().enumerate() {
            output[i + j] += *left * *right;
        }
    }
    output
}

fn is_exact_zero(value: &Complex<f64>) -> bool {
    value.re == 0.0 && value.im == 0.0
}

fn convert_output(
    data: Vec<Complex<f64>>,
    orientation: Orientation,
    single_output: bool,
    complex_output: bool,
) -> BuiltinResult<Value> {
    let len = data.len();
    let shape = match (orientation, len) {
        (Orientation::Row, 0) => vec![1, 0],
        (Orientation::Column, 0) => vec![0, 1],
        (Orientation::Row, 1) | (Orientation::Column, 1) => vec![1, 1],
        (Orientation::Row, _) => vec![1, len],
        (Orientation::Column, _) => vec![len, 1],
    };

    if !complex_output {
        let real_data: Vec<f64> = data.into_iter().map(|c| c.re).collect();
        finalize_real(real_data, shape, single_output)
    } else {
        let complex_data: Vec<(f64, f64)> = data.into_iter().map(|c| (c.re, c.im)).collect();
        let storage = if single_output {
            ComplexStorage::F32(
                complex_data
                    .into_iter()
                    .map(|(re, im)| (re as f32, im as f32))
                    .collect(),
            )
        } else {
            ComplexStorage::F64(complex_data)
        };
        let tensor = ComplexTensor::from_complex_storage(storage, shape)
            .map_err(|e| deconv_error_with_detail(&DECONV_ERROR_BUILD_COMPLEX_OUTPUT, &e))?;
        Ok(crate::builtins::common::random_args::complex_tensor_into_value(tensor))
    }
}

fn finalize_real(data: Vec<f64>, shape: Vec<usize>, single_output: bool) -> BuiltinResult<Value> {
    let tensor = if single_output {
        Tensor::from_f32(data.into_iter().map(|value| value as f32).collect(), shape)
    } else {
        Tensor::new(data, shape)
    }
    .map_err(|e| deconv_error_with_detail(&DECONV_ERROR_BUILD_OUTPUT, &e))?;
    Ok(tensor::tensor_into_value(tensor))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
    use runmat_accelerate_api::HostTensorView;
    use runmat_builtins::{
        builtin_function_by_name, IntValue, IntegerStorage, ResolveContext, Type,
    };

    fn error_message(error: RuntimeError) -> String {
        error.message().to_string()
    }

    fn evaluate(numerator: Value, denominator: Value) -> BuiltinResult<DeconvEval> {
        block_on(super::evaluate(numerator, denominator))
    }

    fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Tensor {
        let tensor = Tensor::new_integer(storage, shape).expect("typed integer tensor");
        tensor
    }

    #[test]
    fn deconv_type_uses_numerator_orientation() {
        let out = deconv_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(5)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(3)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            out,
            Type::Tensor {
                shape: Some(vec![Some(1), None])
            }
        );
    }

    #[test]
    fn deconv_type_denominator_longer_returns_scalar_zero_shape() {
        let output = deconv_type(
            &[
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(2)]),
                },
                Type::Tensor {
                    shape: Some(vec![Some(1), Some(3)]),
                },
            ],
            &ResolveContext::new(Vec::new()),
        );
        assert_eq!(
            output,
            Type::Tensor {
                shape: Some(vec![Some(1), None])
            }
        );
    }

    #[test]
    fn deconv_descriptor_signatures_and_errors() {
        let builtin = builtin_function_by_name(BUILTIN_NAME).expect("deconv builtin");
        let descriptor = builtin.descriptor.expect("deconv descriptor");
        let labels: Vec<&str> = descriptor.signatures.iter().map(|sig| sig.label).collect();
        assert!(labels.contains(&"Q = deconv(numerator, denominator)"));
        assert!(labels.contains(&"[Q, R] = deconv(numerator, denominator)"));
        assert_eq!(
            descriptor.output_mode,
            BuiltinOutputMode::ByRequestedOutputCount
        );
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.DECONV.DENOMINATOR_INVALID"));
        assert!(descriptor
            .errors
            .iter()
            .any(|err| err.code == "RM.DECONV.TOO_MANY_OUTPUTS"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deconv_exact_division() {
        let numerator = Tensor::new(vec![1.0, 3.0, 3.0, 1.0], vec![1, 4]).unwrap();
        let denominator = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let value =
            deconv_builtin(Value::Tensor(numerator), Value::Tensor(denominator)).expect("deconv");
        match value {
            Value::Tensor(q) => {
                assert_eq!(q.shape, vec![1, 3]);
                assert_eq!(q.materialize_f64(), vec![1.0, 2.0, 1.0]);
            }
            other => panic!("expected tensor quotient, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deconv_reads_typed_integer_coefficients_exactly() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let numerator = integer_tensor(IntegerStorage::I16(vec![1, 3, 3, 1]), vec![1, 4]);
        let denominator = integer_tensor(IntegerStorage::U16(vec![1, 1]), vec![1, 2]);

        let eval =
            evaluate(Value::Tensor(numerator), Value::Tensor(denominator)).expect("evaluate");

        assert_eq!(real_vector(eval.quotient()), vec![1.0, 2.0, 1.0]);
        assert_eq!(real_vector(eval.remainder()), vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn deconv_integer_extension_is_gated_and_wide_values_reject() {
        let strict = crate::compatibility::push_runmat_extensions_enabled(false);
        let error = evaluate(Value::Int(IntValue::I32(1)), Value::Num(1.0))
            .expect_err("strict mode rejects integer extension");
        assert_eq!(
            error.identifier(),
            DECONV_INTEGER_INPUT_EXTENSION.error_identifier
        );
        drop(strict);

        let compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let error = evaluate(Value::Int(IntValue::U64((1u64 << 53) + 1)), Value::Num(1.0))
            .expect_err("inexact binary64 boundary rejects");
        assert!(error.message().contains("exactly representable as double"));
        drop(compat);
    }

    #[test]
    fn deconv_preserves_single_and_full_length_remainder() {
        let numerator = Tensor::from_f32(vec![1.0, 4.0, 7.0], vec![1, 3]).expect("single");
        let denominator = Tensor::from_f32(vec![1.0, 2.0], vec![1, 2]).expect("single");
        let eval = evaluate(Value::Tensor(numerator), Value::Tensor(denominator)).expect("deconv");
        match eval.quotient() {
            Value::Tensor(output) => assert_eq!(output.numeric_dtype(), NumericDType::F32),
            other => panic!("expected single quotient, got {other:?}"),
        }
        match eval.remainder() {
            Value::Tensor(output) => {
                assert_eq!(output.numeric_dtype(), NumericDType::F32);
                assert_eq!(output.shape, vec![1, 3]);
            }
            other => panic!("expected single remainder, got {other:?}"),
        }
    }

    #[test]
    fn deconv_integer_or_logical_inputs_force_double_independent_of_residency() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let single = Tensor::from_f32(vec![1.0, 1.0], vec![1, 2]).expect("single");
        let integer = integer_tensor(IntegerStorage::U8(vec![1, 3, 3, 1]), vec![1, 4]);
        let integer_eval = evaluate(Value::Tensor(integer), Value::Tensor(single.clone()))
            .expect("integer and single");
        for output in [integer_eval.quotient(), integer_eval.remainder()] {
            match output {
                Value::Tensor(tensor) => assert_eq!(tensor.numeric_dtype(), NumericDType::F64),
                Value::Num(_) => {}
                other => panic!("expected double real output, got {other:?}"),
            }
        }

        let logical = LogicalArray::new(vec![1, 1, 1, 1], vec![1, 4]).expect("logical");
        let logical_eval = evaluate(Value::LogicalArray(logical), Value::Tensor(single))
            .expect("logical and single");
        for output in [logical_eval.quotient(), logical_eval.remainder()] {
            match output {
                Value::Tensor(tensor) => assert_eq!(tensor.numeric_dtype(), NumericDType::F64),
                Value::Num(_) => {}
                other => panic!("expected double real output, got {other:?}"),
            }
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deconv_with_remainder() {
        let numerator = Tensor::new(vec![1.0, 4.0, 7.0], vec![1, 3]).unwrap();
        let denominator = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let eval =
            evaluate(Value::Tensor(numerator), Value::Tensor(denominator)).expect("evaluate");
        let quotient = real_vector(eval.quotient());
        assert_eq!(quotient, vec![1.0, 2.0]);
        let remainder = real_vector(eval.remainder());
        assert_eq!(remainder, vec![0.0, 0.0, 3.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deconv_denominator_longer() {
        let numerator = Tensor::new(vec![3.0, 5.0], vec![1, 2]).unwrap();
        let denominator = Tensor::new(vec![1.0, 0.0, 2.0], vec![1, 3]).unwrap();
        let eval =
            evaluate(Value::Tensor(numerator), Value::Tensor(denominator)).expect("evaluate");
        let quotient = real_vector(eval.quotient());
        assert_eq!(quotient, vec![0.0]);
        let remainder = real_vector(eval.remainder());
        assert_eq!(remainder, vec![3.0, 5.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deconv_leading_zeros() {
        let numerator = Tensor::new(vec![0.0, 0.0, 1.0, 2.0], vec![1, 4]).unwrap();
        let denominator = Tensor::new(vec![0.0, 1.0, 1.0], vec![1, 3]).unwrap();
        let eval =
            evaluate(Value::Tensor(numerator), Value::Tensor(denominator)).expect("evaluate");
        let quotient = real_vector(eval.quotient());
        assert_eq!(quotient, vec![1.0]);
        let remainder = real_vector(eval.remainder());
        assert_eq!(remainder, vec![0.0, 0.0, 0.0, 1.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deconv_complex_coefficients() {
        let numerator = Value::ComplexTensor(
            ComplexTensor::new(vec![(1.0, 2.0), (3.0, -4.0), (2.0, 0.0)], vec![1, 3]).unwrap(),
        );
        let denominator = Value::ComplexTensor(
            ComplexTensor::new(vec![(1.0, -1.0), (2.0, 1.0)], vec![1, 2]).unwrap(),
        );

        let eval = evaluate(numerator, denominator).expect("evaluate");
        match eval.quotient() {
            Value::ComplexTensor(q) => {
                assert_eq!(q.materialize_f64().len(), 2);
            }
            other => panic!("unexpected quotient {other:?}"),
        }
        match eval.remainder() {
            Value::ComplexTensor(_) | Value::Complex(_, _) => {
                // Accept either scalar complex or tensor form depending on trimming.
            }
            Value::Tensor(r) => {
                assert!(r.materialize_f64().iter().all(|v| v.abs() <= 1e-9));
            }
            other => panic!("unexpected remainder {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deconv_reconstructs_original() {
        let numerator = vec![1.0, -3.0, 3.0, -1.0];
        let denominator = vec![1.0, -1.0];
        let eval = evaluate(
            Value::Tensor(Tensor::new(numerator.clone(), vec![1, 4]).unwrap()),
            Value::Tensor(Tensor::new(denominator.clone(), vec![1, 2]).unwrap()),
        )
        .expect("evaluate");

        let quotient = match eval.quotient() {
            Value::Tensor(t) => t.materialize_f64(),
            other => panic!("unexpected quotient {other:?}"),
        };
        let remainder = real_vector(eval.remainder());

        let reconstructed = add_polynomials(&convolve(&denominator, &quotient), &remainder);

        assert!(
            reconstructed
                .iter()
                .zip(numerator.iter())
                .all(|(a, b)| (a - b).abs() <= 1e-8),
            "reconstructed {:?} != {:?}",
            reconstructed,
            numerator
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deconv_denominator_zero_error() {
        let numerator = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let denominator = Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap();
        let err = error_message(
            deconv_builtin(Value::Tensor(numerator), Value::Tensor(denominator)).unwrap_err(),
        );
        assert!(err.contains("denominator"));
    }

    #[test]
    fn deconv_treats_only_exact_zero_as_zero() {
        let numerator = Tensor::new(vec![1.0e-20, 2.0e-20], vec![1, 2]).expect("numerator");
        let denominator = Tensor::new(vec![1.0e-20, 1.0], vec![1, 2]).expect("denominator");
        let eval = evaluate(Value::Tensor(numerator), Value::Tensor(denominator))
            .expect("tiny nonzero leading coefficient remains significant");
        assert_eq!(real_vector(eval.quotient()), vec![1.0]);
        assert_eq!(real_vector(eval.remainder()).len(), 2);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deconv_rejects_matrix_inputs() {
        let numerator = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let denominator = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
        let err = error_message(
            deconv_builtin(Value::Tensor(numerator), Value::Tensor(denominator)).unwrap_err(),
        );
        assert!(err.contains("vectors"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deconv_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let numerator = Tensor::new(vec![1.0, 3.0, 3.0, 1.0], vec![1, 4]).unwrap();
            let denominator = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
            let n_view = HostTensorView {
                data: &numerator.materialize_f64(),
                shape: &numerator.shape,
            };
            let d_view = HostTensorView {
                data: &denominator.materialize_f64(),
                shape: &denominator.shape,
            };
            let n_handle = provider.upload(&n_view).expect("upload numerator");
            let d_handle = provider.upload(&d_view).expect("upload denominator");

            let eval =
                evaluate(Value::GpuTensor(n_handle), Value::GpuTensor(d_handle)).expect("evaluate");

            match eval.quotient() {
                Value::GpuTensor(handle) => {
                    let gathered =
                        test_support::gather(Value::GpuTensor(handle)).expect("gather quotient");
                    assert_eq!(gathered.materialize_f64(), vec![1.0, 2.0, 1.0]);
                }
                other => panic!("expected GPU quotient, got {other:?}"),
            }
        });
    }

    #[test]
    fn deconv_resident_integer_outputs_are_untyped_double() {
        use runmat_accelerate_api::{
            HostIntegerDataView, HostIntegerTensorView, ProviderPrecision,
        };

        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let numerator_data = [1u16, 3, 3, 1];
            let denominator_data = [1u16, 1];
            let numerator = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U16(&numerator_data),
                    shape: &[1, 4],
                })
                .expect("upload integer numerator");
            let denominator = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U16(&denominator_data),
                    shape: &[1, 2],
                })
                .expect("upload integer denominator");

            let eval = evaluate(Value::GpuTensor(numerator), Value::GpuTensor(denominator))
                .expect("evaluate");
            for output in [eval.quotient(), eval.remainder()] {
                let Value::GpuTensor(handle) = output else {
                    panic!("expected resident output")
                };
                assert_eq!(
                    runmat_accelerate_api::handle_precision(&handle),
                    Some(ProviderPrecision::F64)
                );
                assert!(runmat_accelerate_api::handle_integer_type(&handle).is_none());
                assert!(!runmat_accelerate_api::handle_is_logical(&handle));
            }
        });
    }

    #[test]
    fn deconv_builtin_returns_only_requested_resident_outputs() {
        test_support::with_test_provider(|provider| {
            let numerator_data = [1.0, 3.0, 3.0, 1.0];
            let denominator_data = [1.0, 1.0];
            let numerator = provider
                .upload(&HostTensorView {
                    data: &numerator_data,
                    shape: &[1, 4],
                })
                .expect("upload numerator");
            let denominator = provider
                .upload(&HostTensorView {
                    data: &denominator_data,
                    shape: &[1, 2],
                })
                .expect("upload denominator");
            let guard = crate::output_count::push_output_count(Some(1));
            let output = deconv_builtin(Value::GpuTensor(numerator), Value::GpuTensor(denominator))
                .expect("deconv");
            drop(guard);

            let Value::OutputList(outputs) = output else {
                panic!("expected requested output list")
            };
            assert_eq!(outputs.len(), 1);
            let Value::GpuTensor(quotient) = &outputs[0] else {
                panic!("expected resident quotient")
            };
            assert!(block_on(provider.download(quotient)).is_ok());
        });
    }

    #[test]
    fn deconv_resident_output_cleanup_releases_the_handle() {
        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0],
                    shape: &[1, 2],
                })
                .expect("upload");
            free_deconv_output(&Value::GpuTensor(handle.clone()));
            assert!(block_on(provider.download(&handle)).is_err());
        });
    }

    #[test]
    fn deconv_resident_logical_outputs_are_untyped_double() {
        use runmat_accelerate_api::ProviderPrecision;

        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let numerator = provider
                .upload(&HostTensorView {
                    data: &[1.0, 1.0, 1.0],
                    shape: &[1, 3],
                })
                .expect("upload logical numerator");
            runmat_accelerate_api::set_handle_logical(&numerator, true);
            let denominator = provider
                .upload(&HostTensorView {
                    data: &[1.0, 1.0],
                    shape: &[1, 2],
                })
                .expect("upload denominator");

            let eval = evaluate(Value::GpuTensor(numerator), Value::GpuTensor(denominator))
                .expect("evaluate");
            for output in [eval.quotient(), eval.remainder()] {
                let Value::GpuTensor(handle) = output else {
                    panic!("expected resident output")
                };
                assert_eq!(
                    runmat_accelerate_api::handle_precision(&handle),
                    Some(ProviderPrecision::F64)
                );
                assert!(runmat_accelerate_api::handle_integer_type(&handle).is_none());
                assert!(!runmat_accelerate_api::handle_is_logical(&handle));
            }
        });
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn deconv_wgpu_matches_cpu() {
        register_wgpu_provider(WgpuProviderOptions::default()).expect("wgpu provider");

        let numerator = Tensor::new(vec![1.0, -2.0, 3.0, -4.0, 5.0], vec![1, 5]).unwrap();
        let denominator = Tensor::new(vec![1.0, -1.0, 2.0], vec![1, 3]).unwrap();

        let cpu_eval = evaluate(
            Value::Tensor(numerator.clone()),
            Value::Tensor(denominator.clone()),
        )
        .expect("cpu evaluate");
        let cpu_q = real_vector(cpu_eval.quotient());
        let cpu_r = real_vector(cpu_eval.remainder());

        let provider = runmat_accelerate_api::provider().expect("wgpu provider registered");
        let num_handle = provider
            .upload(&HostTensorView {
                data: &numerator.materialize_f64(),
                shape: &numerator.shape,
            })
            .expect("upload numerator");
        let den_handle = provider
            .upload(&HostTensorView {
                data: &denominator.materialize_f64(),
                shape: &denominator.shape,
            })
            .expect("upload denominator");

        let gpu_eval = evaluate(Value::GpuTensor(num_handle), Value::GpuTensor(den_handle))
            .expect("gpu evaluate");
        let gpu_q = real_vector(gpu_eval.quotient());
        let gpu_r = real_vector(gpu_eval.remainder());

        assert_eq!(gpu_q.len(), cpu_q.len());
        assert_eq!(gpu_r.len(), cpu_r.len());
        for (a, b) in gpu_q.iter().zip(cpu_q.iter()) {
            assert!((a - b).abs() <= 1e-10, "gpu quotient {a} != cpu {b}");
        }
        for (a, b) in gpu_r.iter().zip(cpu_r.iter()) {
            assert!((a - b).abs() <= 1e-10, "gpu remainder {a} != cpu {b}");
        }
    }

    fn real_vector(value: Value) -> Vec<f64> {
        match value {
            Value::Tensor(t) => t.materialize_f64(),
            Value::Num(n) => vec![n],
            Value::GpuTensor(handle) => {
                let gathered =
                    test_support::gather(Value::GpuTensor(handle)).expect("gather gpu output");
                gathered.materialize_f64()
            }
            Value::Complex(re, im) => {
                assert!(im.abs() <= 1e-9);
                vec![re]
            }
            Value::ComplexTensor(t) => {
                assert!(t.materialize_f64().iter().all(|(_, im)| im.abs() <= 1e-9));
                t.materialize_f64().into_iter().map(|(re, _)| re).collect()
            }
            other => panic!("expected real-valued tensor, got {other:?}"),
        }
    }

    fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
        if a.is_empty() || b.is_empty() {
            return Vec::new();
        }
        let mut out = vec![0.0; a.len() + b.len() - 1];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                out[i + j] += ai * bj;
            }
        }
        out
    }

    fn add_polynomials(a: &[f64], b: &[f64]) -> Vec<f64> {
        let len = a.len().max(b.len());
        let mut out = vec![0.0; len];
        for (i, &v) in a.iter().rev().enumerate() {
            let idx = len - 1 - i;
            out[idx] += v;
        }
        for (i, &v) in b.iter().rev().enumerate() {
            let idx = len - 1 - i;
            out[idx] += v;
        }
        out
    }

    fn deconv_builtin(numerator: Value, denominator: Value) -> BuiltinResult<Value> {
        block_on(super::deconv_builtin(numerator, denominator))
    }
}
