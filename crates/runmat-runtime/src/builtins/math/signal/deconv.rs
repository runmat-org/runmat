//! MATLAB-compatible `deconv` builtin with GPU-aware semantics for RunMat.

#[cfg(test)]
#[path = "deconv/placement_tests.rs"]
mod placement_tests;

use num_complex::Complex;
use runmat_builtins::{
    BuiltinErrorDescriptor, DECONV_ERROR_BUILD_COMPLEX_OUTPUT, DECONV_ERROR_BUILD_OUTPUT,
    DECONV_ERROR_DENOMINATOR_INVALID, DECONV_ERROR_GATHER_FAILED, DECONV_ERROR_INVALID_INPUT,
    DECONV_ERROR_LATE_UPLOAD_ADMISSION, DECONV_ERROR_TOO_MANY_OUTPUTS,
    DECONV_ERROR_VECTOR_REQUIRED, DECONV_INTEGER_INPUT_EXTENSION, DECONV_LOGICAL_INPUT_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{ComplexStorage, ComplexTensor, NumericDType, Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::validation::PreparedIntegerPayload;
use crate::builtins::common::{gpu_helpers, map_control_flow_with_builtin, tensor};
use crate::builtins::math::signal::common::SignalCooperation;
use crate::{build_runtime_error, BuiltinResult, GpuGatherRetry, RuntimeError};

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
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::signal::deconv"
)]
async fn deconv_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    numerator: Value,
    denominator: Value,
) -> crate::BuiltinResult<Value> {
    let requested = crate::output_count::current_output_count();
    if matches!(requested, Some(count) if count > 2) {
        return Err(build_runtime_error(DECONV_ERROR_TOO_MANY_OUTPUTS.message)
            .with_builtin(BUILTIN_NAME)
            .with_identifier(
                DECONV_ERROR_TOO_MANY_OUTPUTS
                    .identifier
                    .expect("deconv too-many-outputs descriptor identifier"),
            )
            .build());
    }
    let mut outputs = evaluate_requested(
        numerator,
        denominator,
        requested.unwrap_or(1),
        runtime.as_ref(),
    )
    .await?;
    match requested {
        Some(0) => Ok(Value::output_list(Vec::new())),
        Some(count) => {
            let mut published = vec![outputs.quotient.take().ok_or_else(deconv_missing_output)?];
            if count == 2 {
                published.push(outputs.remainder.take().ok_or_else(deconv_missing_output)?);
            }
            Ok(Value::output_list(published))
        }
        None => outputs.quotient.take().ok_or_else(deconv_missing_output),
    }
}

/// Evaluate `deconv` and retain both outputs for multi-value contexts.
pub async fn evaluate(numerator: Value, denominator: Value) -> BuiltinResult<DeconvEval> {
    let mut outputs = evaluate_requested(numerator, denominator, 2, None).await?;
    Ok(DeconvEval {
        quotient: outputs.quotient.take().ok_or_else(deconv_missing_output)?,
        remainder: outputs.remainder.take().ok_or_else(deconv_missing_output)?,
    })
}

fn deconv_missing_output() -> RuntimeError {
    deconv_error_with_detail(
        &DECONV_ERROR_BUILD_OUTPUT,
        "requested result was not constructed",
    )
}

struct RequestedDeconvOutputs {
    quotient: Option<Value>,
    remainder: Option<Value>,
}

async fn evaluate_requested(
    numerator: Value,
    denominator: Value,
    requested: usize,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<RequestedDeconvOutputs> {
    ensure_extensions(&numerator)?;
    ensure_extensions(&denominator)?;
    admit_deconv_operand(&numerator)?;
    admit_deconv_operand(&denominator)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&numerator, BUILTIN_NAME)?;
    crate::builtins::common::validation::reject_typed_complex_integer(&denominator, BUILTIN_NAME)?;
    let mut cooperation = SignalCooperation::new(runtime, BUILTIN_NAME);
    cooperation.completed(0).await?;
    let integer_or_logical = [&numerator, &denominator]
        .iter()
        .any(|value| value_is_integer_or_logical(value));
    let single_output =
        !integer_or_logical && (value_is_single(&numerator) || value_is_single(&denominator));
    let complex_output = value_is_complex(&numerator) || value_is_complex(&denominator);
    let resident_inputs = [&numerator, &denominator]
        .iter()
        .filter_map(|value| match value {
            Value::GpuTensor(handle) => Some(handle.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    let placement = if requested > 0 && !complex_output {
        if let Some(numerator_len) = deconv_operand_len(&numerator) {
            let element_type = if single_output {
                runmat_accelerate_api::ProviderElementType::F32
            } else {
                runmat_accelerate_api::ProviderElementType::F64
            };
            let bounds = (0..requested)
                .map(|index| runmat_accelerate_api::ProviderTransferOutputBound {
                    element_type,
                    storage: runmat_accelerate_api::ProviderStorage::DenseReal,
                    layout: runmat_accelerate_api::ProviderLayout::ColumnMajorContiguous,
                    residency: runmat_accelerate_api::ProviderResidency::Device,
                    max_elements: u64::try_from(if index == 0 {
                        numerator_len.max(1)
                    } else {
                        numerator_len
                    })
                    .unwrap_or(u64::MAX),
                })
                .collect();
            super::common::convolution::placement::select_bounded_host_variant(
                runtime,
                BUILTIN_NAME,
                "division.host",
                &[&numerator, &denominator],
                &resident_inputs,
                bounds,
                deconv_host_scratch_bound(&numerator, &denominator),
            )?
        } else {
            None
        }
    } else {
        None
    };
    // Complete host-visible exactness admission before either resident readback.
    for value in [&numerator, &denominator] {
        if !matches!(value, Value::GpuTensor(_)) {
            prepare_deconv_integer(value, &mut cooperation).await?;
        }
    }
    preflight_denominator(&denominator, &mut cooperation).await?;
    let resident_prototype = resident_inputs.first();
    let provider = resident_prototype.and_then(runmat_accelerate_api::provider_for_handle);
    if !resident_inputs.is_empty() && provider.is_none() {
        return Err(deconv_error_with_detail(
            &DECONV_ERROR_GATHER_FAILED,
            "resident input has no owning provider",
        ));
    }
    if let (Some(provider), Some(prototype)) = (provider, resident_prototype) {
        for value in [&numerator, &denominator] {
            if let Value::GpuTensor(handle) = value {
                let owner =
                    runmat_accelerate_api::provider_for_handle(handle).ok_or_else(|| {
                        deconv_error_with_detail(
                            &DECONV_ERROR_GATHER_FAILED,
                            "resident input has no owning provider",
                        )
                    })?;
                if !runmat_accelerate_api::same_provider_instance(owner, provider)
                    || handle.device_id() != prototype.device_id()
                {
                    return Err(deconv_error_with_detail(
                        &DECONV_ERROR_INVALID_INPUT,
                        "resident inputs must share one owning provider and device",
                    ));
                }
            }
        }
    }
    let numerator_payload = if matches!(&numerator, Value::GpuTensor(_)) {
        prepare_deconv_integer(&numerator, &mut cooperation).await?
    } else {
        None
    };
    let num_input = convert_value(numerator, numerator_payload, &mut cooperation).await?;
    let denominator_payload = if matches!(&denominator, Value::GpuTensor(_)) {
        prepare_deconv_integer(&denominator, &mut cooperation).await?
    } else {
        None
    };
    let den_input = convert_value(denominator, denominator_payload, &mut cooperation).await?;

    let (quotient_raw, remainder_raw) =
        polynomial_division(&num_input.data, &den_input.data, &mut cooperation).await?;

    let orientation = orientation_from_hint(num_input.hint);
    let mut quotient = (requested >= 1)
        .then(|| convert_output(quotient_raw, orientation, single_output, complex_output))
        .transpose()?;
    let mut remainder = (requested >= 2)
        .then(|| convert_output(remainder_raw, orientation, single_output, complex_output))
        .transpose()?;
    if let (Some(placement), Some(provider)) = (placement.as_ref(), provider) {
        let produced = quotient.iter().chain(remainder.iter()).collect::<Vec<_>>();
        if let Err(cause) = placement.validate_actual_host_outputs(provider, &produced) {
            // The selected whole-call route promised a bounded, exact-upload
            // presentation. A late rejection cannot be repaired by running a
            // different effectful restore beneath the same failed receipt.
            return Err(
                build_runtime_error(DECONV_ERROR_LATE_UPLOAD_ADMISSION.message)
                    .with_builtin(BUILTIN_NAME)
                    .with_identifier(
                        DECONV_ERROR_LATE_UPLOAD_ADMISSION
                            .identifier
                            .expect("late upload admission descriptor identifier"),
                    )
                    .with_source(cause)
                    .build(),
            );
        }
    }
    if let Some(prototype) = resident_prototype {
        let caller_inputs = resident_inputs.iter().collect::<Vec<_>>();
        if let Some(value) = quotient.take() {
            quotient = Some(
                runmat_accelerate_api::observation::with_provider_observation(
                    placement.as_ref().and_then(|selected| selected.recorder()),
                    || {
                        gpu_helpers::restore_class_preserving_value_with_operands(
                            prototype,
                            value,
                            BUILTIN_NAME,
                            gpu_helpers::RestoreOperandContext::new(&caller_inputs, &caller_inputs),
                        )
                    },
                )?,
            );
        }
        if let Some(value) = remainder.take() {
            let mut protected = caller_inputs.clone();
            if let Some(Value::GpuTensor(handle)) = quotient.as_ref() {
                protected.push(handle);
            }
            remainder = Some(
                runmat_accelerate_api::observation::with_provider_observation(
                    placement.as_ref().and_then(|selected| selected.recorder()),
                    || {
                        gpu_helpers::restore_class_preserving_value_with_operands(
                            prototype,
                            value,
                            BUILTIN_NAME,
                            gpu_helpers::RestoreOperandContext::new(&protected, &caller_inputs),
                        )
                    },
                )?,
            );
        }
    }
    if let Some(placement) = placement {
        let published = quotient.iter().chain(remainder.iter()).collect::<Vec<_>>();
        placement.finish_outputs(&published, requested);
    }
    Ok(RequestedDeconvOutputs {
        quotient,
        remainder,
    })
}

fn deconv_operand_len(value: &Value) -> Option<usize> {
    match value {
        Value::Tensor(tensor) => {
            super::common::convolution::checked_vector_elements(tensor.shape())
        }
        Value::ComplexTensor(tensor) => {
            super::common::convolution::checked_vector_elements(tensor.shape())
        }
        Value::LogicalArray(array) => {
            super::common::convolution::checked_vector_elements(&array.shape)
        }
        Value::GpuTensor(handle) => {
            super::common::convolution::checked_vector_elements(handle.shape())
        }
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => Some(1),
        _ => None,
    }
}

fn deconv_host_scratch_bound(a: &Value, b: &Value) -> runmat_execution::ResourceByteBound {
    let bound = (|| {
        let n = u64::try_from(deconv_operand_len(a)?).ok()?;
        let m = u64::try_from(deconv_operand_len(b)?).ok()?;
        let product = if n == 0 || m == 0 {
            0
        } else {
            n.checked_add(m)?.checked_sub(1)?
        };
        // Materialized inputs, trimmed working copy, quotient, product and
        // full-length remainder may coexist. Both requested host outputs are
        // conservatively counted even for a one-output call.
        let complex_elements = n
            .checked_add(m)?
            .checked_add(n)?
            .checked_add(n.max(1))?
            .checked_add(product)?
            .checked_add(n)?;
        let numerical = complex_elements
            .checked_mul(16)?
            .checked_add(n.max(1).checked_add(n)?.checked_mul(16)?)?;
        // Resident denominator preflight can read it before ordinary input
        // conversion; count both readback phases with raw/decoded/value copies.
        let gather = [a, b].into_iter().try_fold(0_u64, |total, value| {
            let bytes = match value {
                Value::GpuTensor(handle) => handle.physical().requested_bytes().checked_mul(6)?,
                _ => 0,
            };
            total.checked_add(bytes)
        })?;
        numerical.checked_add(gather)
    })();
    bound
        .map(runmat_execution::ResourceByteBound::Known)
        .unwrap_or(runmat_execution::ResourceByteBound::Unavailable(
            runmat_execution::ResourceBoundUnavailable::NoScratchBound,
        ))
}

async fn preflight_denominator(
    denominator: &Value,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<()> {
    let invalid = match denominator {
        Value::Num(number) => *number == 0.0,
        Value::Int(integer) => integer.is_zero(),
        Value::Bool(flag) => !flag,
        Value::Complex(real, imaginary) => *real == 0.0 && *imaginary == 0.0,
        Value::Tensor(tensor) => {
            let mut nonzero = false;
            for index in 0..tensor.len() {
                nonzero |= tensor
                    .numeric_value_at(index)
                    .is_some_and(|value| !value.is_zero());
                if (index + 1).is_multiple_of(super::common::SIGNAL_WORK_CHUNK as usize) {
                    cooperation
                        .completed(super::common::SIGNAL_WORK_CHUNK)
                        .await?;
                }
                if nonzero {
                    break;
                }
            }
            !nonzero
        }
        Value::ComplexTensor(tensor) => {
            let mut nonzero = false;
            for index in 0..tensor.len() {
                nonzero |= tensor
                    .numeric_value_at(index)
                    .is_some_and(|(real, imaginary)| !real.is_zero() || !imaginary.is_zero());
                if (index + 1).is_multiple_of(super::common::SIGNAL_WORK_CHUNK as usize) {
                    cooperation
                        .completed(super::common::SIGNAL_WORK_CHUNK)
                        .await?;
                }
                if nonzero {
                    break;
                }
            }
            !nonzero
        }
        Value::LogicalArray(array) => {
            let mut nonzero = false;
            for chunk in array.data.chunks(super::common::SIGNAL_WORK_CHUNK as usize) {
                nonzero |= chunk.iter().any(|value| *value != 0);
                cooperation.completed(chunk.len() as u64).await?;
                if nonzero {
                    break;
                }
            }
            !nonzero
        }
        Value::GpuTensor(handle) => handle.shape().contains(&0),
        _ => false,
    };
    if invalid {
        return Err(deconv_error_with_detail(
            &DECONV_ERROR_DENOMINATOR_INVALID,
            "must contain at least one non-zero coefficient",
        ));
    }
    Ok(())
}

fn admit_deconv_operand(value: &Value) -> BuiltinResult<()> {
    if matches!(value, Value::GpuTensor(_)) {
        super::common::convolution::admit_resident_operand(value, || {
            deconv_error(&DECONV_ERROR_INVALID_INPUT)
        })?;
    }
    let shape = match value {
        Value::Tensor(tensor) => Some(tensor.shape()),
        Value::ComplexTensor(tensor) => Some(tensor.shape()),
        Value::LogicalArray(array) => Some(array.shape.as_slice()),
        Value::GpuTensor(handle) => Some(handle.shape()),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => None,
        _ => return Err(deconv_error(&DECONV_ERROR_INVALID_INPUT)),
    };
    if shape
        .is_some_and(|shape| super::common::convolution::checked_vector_elements(shape).is_none())
    {
        return Err(deconv_error(&DECONV_ERROR_VECTOR_REQUIRED));
    }
    Ok(())
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

async fn prepare_deconv_integer(
    value: &Value,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Option<PreparedIntegerPayload>> {
    let prepared = cooperation
        .prepare_integer_receipt(value, &DECONV_INTEGER_INPUT_EXTENSION)
        .await?;
    if !prepared.is_exact() {
        let mut error = deconv_error_with_detail(
            &DECONV_ERROR_INVALID_INPUT,
            "integer input must be exactly representable as double",
        );
        error.gpu_gather_retry = GpuGatherRetry::Never;
        return Err(error);
    }
    Ok(prepared.into_resident_payload())
}

async fn convert_value(
    value: Value,
    integer_payload: Option<PreparedIntegerPayload>,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<PolyInput> {
    match value {
        Value::GpuTensor(handle) => {
            let gathered = if let Some(payload) = integer_payload {
                match payload {
                    PreparedIntegerPayload::Real(gathered) => {
                        let tensor = Tensor::new_integer(
                            gpu_helpers::integer_storage_from_owned(gathered.data),
                            gathered.shape,
                        )
                        .map_err(|detail| {
                            deconv_error_with_detail(&DECONV_ERROR_GATHER_FAILED, detail)
                        })?;
                        Value::Tensor(tensor)
                    }
                    PreparedIntegerPayload::Complex(_) => {
                        return Err(deconv_error(&DECONV_ERROR_INVALID_INPUT));
                    }
                }
            } else {
                gpu_helpers::gather_value_async(&Value::GpuTensor(handle))
                    .await
                    .map_err(|flow| {
                        let message = flow.message().to_owned();
                        deconv_error_with_source(
                            &DECONV_ERROR_GATHER_FAILED,
                            message,
                            map_control_flow_with_builtin(flow, BUILTIN_NAME),
                        )
                    })?
            };
            convert_host_value(gathered, cooperation).await
        }
        host => convert_host_value(host, cooperation).await,
    }
}

async fn convert_host_value(
    value: Value,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<PolyInput> {
    let hint = match &value {
        Value::Tensor(tensor) => {
            let (rows, cols) = dense_deconv_dimensions(tensor.shape());
            classify_orientation(rows, cols, tensor.len())
        }
        Value::ComplexTensor(tensor) => {
            classify_orientation(tensor.rows(), tensor.cols(), tensor.len())
        }
        Value::LogicalArray(array) => classify_orientation(
            array.shape.first().copied().unwrap_or(0),
            array.shape.get(1).copied().unwrap_or(0),
            array.data.len(),
        ),
        Value::Num(_) | Value::Int(_) | Value::Bool(_) | Value::Complex(_, _) => {
            OrientationHint::Scalar
        }
        Value::GpuTensor(_) => Err(deconv_error_with_detail(
            &DECONV_ERROR_GATHER_FAILED,
            "provider returned a resident handle instead of host coefficients",
        ))?,
        other => Err(deconv_error_with_detail(
            &DECONV_ERROR_INVALID_INPUT,
            format!("{other:?}"),
        ))?,
    };
    ensure_vector(hint)?;
    let data = super::common::convolution::materialize_host_coefficients(
        &value,
        cooperation,
        || deconv_error(&DECONV_ERROR_INVALID_INPUT),
        deconv_allocation_error,
    )
    .await?;
    Ok(PolyInput { data, hint })
}

/// Preserve deconv's coefficient-orientation policy for dense tensors: an
/// explicit rank-one shape is a row vector, while rank zero has no matrix
/// orientation and is rejected by the vector admission below.
fn dense_deconv_dimensions(shape: &[usize]) -> (usize, usize) {
    match shape {
        [] => (0, 0),
        [cols] => (1, *cols),
        [rows, cols, ..] => (*rows, *cols),
    }
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

async fn polynomial_division(
    numerator: &[Complex<f64>],
    denominator: &[Complex<f64>],
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<PolyDivision> {
    if denominator.is_empty() {
        return Err(deconv_error_with_detail(
            &DECONV_ERROR_DENOMINATOR_INVALID,
            "must not be empty",
        ));
    }

    let den_trim = trim_leading_zeros(denominator, cooperation).await?;
    if den_trim.is_empty() {
        return Err(deconv_error_with_detail(
            &DECONV_ERROR_DENOMINATOR_INVALID,
            "must contain at least one non-zero coefficient",
        ));
    }

    let num_trim = trim_leading_zeros(numerator, cooperation).await?;
    if num_trim.is_empty() {
        return Ok((
            vec![Complex::new(0.0, 0.0)],
            copy_coefficients(numerator, cooperation).await?,
        ));
    }

    if num_trim.len() < den_trim.len() {
        return Ok((
            vec![Complex::new(0.0, 0.0)],
            copy_coefficients(numerator, cooperation).await?,
        ));
    }

    let divisor_lead = den_trim[0];

    let q_len = num_trim.len() - den_trim.len() + 1;
    let mut quotient = zero_coefficients(q_len)?;
    let mut working = copy_coefficients(num_trim, cooperation).await?;
    let mut work = 0_u64;

    for k in 0..q_len {
        let coeff = working[k] / divisor_lead;
        quotient[k] = coeff;
        for j in 0..den_trim.len() {
            working[k + j] -= coeff * den_trim[j];
            work += 1;
            if work == super::common::SIGNAL_WORK_CHUNK {
                cooperation.completed(work).await?;
                work = 0;
            }
        }
    }
    cooperation.completed(work).await?;

    let product = convolve_complex(den_trim, &quotient, cooperation).await?;
    let mut remainder = copy_coefficients(numerator, cooperation).await?;
    let offset = remainder.len().saturating_sub(product.len());
    let mut work = 0_u64;
    for (index, value) in product.into_iter().enumerate() {
        remainder[offset + index] -= value;
        work += 1;
        if work == super::common::SIGNAL_WORK_CHUNK {
            cooperation.completed(work).await?;
            work = 0;
        }
    }
    cooperation.completed(work).await?;

    Ok((quotient, remainder))
}

async fn trim_leading_zeros<'a>(
    data: &'a [Complex<f64>],
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<&'a [Complex<f64>]> {
    let mut work = 0_u64;
    for (index, value) in data.iter().enumerate() {
        work += 1;
        if work == super::common::SIGNAL_WORK_CHUNK {
            cooperation.completed(work).await?;
            work = 0;
        }
        if !is_exact_zero(value) {
            cooperation.completed(work).await?;
            return Ok(&data[index..]);
        }
    }
    cooperation.completed(work).await?;
    Ok(&[])
}

async fn convolve_complex(
    lhs: &[Complex<f64>],
    rhs: &[Complex<f64>],
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Vec<Complex<f64>>> {
    if lhs.is_empty() || rhs.is_empty() {
        return Ok(Vec::new());
    }
    let len = lhs
        .len()
        .checked_add(rhs.len())
        .and_then(|n| n.checked_sub(1))
        .ok_or_else(|| deconv_error(&DECONV_ERROR_BUILD_OUTPUT))?;
    let mut output = zero_coefficients(len)?;
    let mut work = 0_u64;
    for (i, left) in lhs.iter().enumerate() {
        for (j, right) in rhs.iter().enumerate() {
            output[i + j] += *left * *right;
            work += 1;
            if work == super::common::SIGNAL_WORK_CHUNK {
                cooperation.completed(work).await?;
                work = 0;
            }
        }
    }
    cooperation.completed(work).await?;
    Ok(output)
}

fn zero_coefficients(len: usize) -> BuiltinResult<Vec<Complex<f64>>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(deconv_allocation_error)?;
    values.resize(len, Complex::new(0.0, 0.0));
    Ok(values)
}

async fn copy_coefficients(
    source: &[Complex<f64>],
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<Vec<Complex<f64>>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(source.len())
        .map_err(deconv_allocation_error)?;
    for chunk in source.chunks(super::common::SIGNAL_WORK_CHUNK as usize) {
        values.extend_from_slice(chunk);
        cooperation.completed(chunk.len() as u64).await?;
    }
    Ok(values)
}

fn deconv_allocation_error(cause: std::collections::TryReserveError) -> RuntimeError {
    let mut builder = build_runtime_error(DECONV_ERROR_BUILD_OUTPUT.message)
        .with_builtin(BUILTIN_NAME)
        .with_source(cause)
        .with_gpu_gather_retry(GpuGatherRetry::Never);
    if let Some(identifier) = DECONV_ERROR_BUILD_OUTPUT.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
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
            ComplexStorage::F64(complex_data.into())
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
#[path = "deconv/provider_tests.rs"]
mod provider_tests;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    #[cfg(feature = "wgpu")]
    use runmat_accelerate::backend::wgpu::provider::{register_wgpu_provider, WgpuProviderOptions};
    use runmat_accelerate_api::HostTensorView;
    use runmat_value::{IntValue, IntegerStorage, LogicalArray};

    fn error_message(error: RuntimeError) -> String {
        error.message().to_string()
    }

    fn evaluate(numerator: Value, denominator: Value) -> BuiltinResult<DeconvEval> {
        block_on(super::evaluate(numerator, denominator))
    }

    fn integer_tensor(storage: IntegerStorage, shape: Vec<usize>) -> Tensor {
        Tensor::new_integer(storage, shape).expect("typed integer tensor")
    }

    #[test]
    fn later_invalid_denominator_shape_rejects_before_resident_download() {
        let _serial = test_support::accel_test_lock();
        let provider =
            super::super::common::convolution::test_support::RejectTransferProvider::new();
        let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let numerator = provider.input(vec![1.0, 2.0], vec![1, 2]);
        let denominator = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let error = deconv_builtin(
            Value::GpuTensor(numerator.handle.clone()),
            Value::Tensor(denominator),
        )
        .expect_err("denominator matrix is invalid");
        assert_eq!(error.identifier(), DECONV_ERROR_VECTOR_REQUIRED.identifier);
        assert_eq!(provider.downloads(), 0);
        numerator.handle.allocation_lease().validate_live().unwrap();
    }

    #[test]
    fn known_empty_or_zero_denominator_rejects_before_resident_numerator_download() {
        let _serial = test_support::accel_test_lock();
        let provider =
            super::super::common::convolution::test_support::RejectTransferProvider::new();
        let _selected = runmat_accelerate_api::ThreadProviderGuard::set(Some(provider));
        let numerator = provider.input(vec![1.0, 2.0], vec![1, 2]);
        for denominator in [
            Tensor::new(Vec::new(), vec![1, 0]).expect("empty denominator"),
            Tensor::new(vec![0.0, 0.0], vec![1, 2]).expect("zero denominator"),
        ] {
            let error = deconv_builtin(
                Value::GpuTensor(numerator.handle.clone()),
                Value::Tensor(denominator),
            )
            .expect_err("known invalid denominator");
            assert_eq!(
                error.identifier(),
                DECONV_ERROR_DENOMINATOR_INVALID.identifier
            );
            assert_eq!(provider.downloads(), 0);
            numerator.handle.allocation_lease().validate_live().unwrap();
        }
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
                assert_eq!(q.shape(), [1, 3]);
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
    fn deconv_rejects_typed_complex_integer_coefficients() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let storage = runmat_value::IntegerComplexStorage::new(
            IntegerStorage::I16(vec![1, 2]),
            IntegerStorage::I16(vec![0, 1]),
        )
        .expect("integer components");
        let coefficients =
            ComplexTensor::from_complex_storage(ComplexStorage::Integer(storage), vec![1, 2])
                .expect("complex integer coefficients");
        let error = evaluate(Value::ComplexTensor(coefficients), Value::Num(1.0))
            .expect_err("complex integer arithmetic is unsupported");
        assert!(error
            .message()
            .contains("complex numbers with integer types"));
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
                assert_eq!(output.shape(), [1, 3]);
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
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    fn deconv_preserves_dense_rank_zero_and_rank_one_orientation_policy() {
        let scalar = Tensor::new(vec![2.0], Vec::new()).expect("rank-zero tensor");
        let error = evaluate(Value::Tensor(scalar), Value::Num(1.0))
            .expect_err("rank-zero dense coefficients have no vector orientation");
        assert_eq!(error.identifier(), DECONV_ERROR_VECTOR_REQUIRED.identifier);

        let row = Tensor::new(vec![2.0, 4.0], vec![2]).expect("rank-one tensor");
        let quotient = evaluate(Value::Tensor(row), Value::Num(1.0))
            .expect("rank-one dense coefficients remain a row vector")
            .quotient();
        let Value::Tensor(quotient) = quotient else {
            panic!("expected dense quotient")
        };
        assert_eq!(quotient.shape(), [1, 2]);
        assert_eq!(quotient.materialize_f64(), vec![2.0, 4.0]);
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn deconv_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let numerator = Tensor::new(vec![1.0, 3.0, 3.0, 1.0], vec![1, 4]).unwrap();
            let denominator = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();
            let n_view = HostTensorView {
                data: &numerator.materialize_f64(),
                shape: numerator.shape(),
            };
            let d_view = HostTensorView {
                data: &denominator.materialize_f64(),
                shape: denominator.shape(),
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
    fn deconv_resident_wide_integer_exactness_uses_native_payload() {
        use runmat_accelerate_api::{HostIntegerDataView, HostIntegerTensorView};

        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let numerator = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&[1u64 << 54, 0]),
                    shape: &[1, 2],
                })
                .expect("upload exact wide numerator");
            let quotient = evaluate(Value::GpuTensor(numerator), Value::Num(1.0))
                .expect("exact wide coefficients")
                .quotient();
            let gathered = test_support::gather(quotient).expect("gather quotient");
            assert_eq!(gathered.materialize_f64(), vec![(1u64 << 54) as f64, 0.0]);

            let inexact = provider
                .upload_integer(&HostIntegerTensorView {
                    data: HostIntegerDataView::U64(&[(1u64 << 53) + 1]),
                    shape: &[1, 1],
                })
                .expect("upload inexact wide numerator");
            let failure = evaluate(Value::GpuTensor(inexact), Value::Num(1.0))
                .expect_err("inexact wide coefficients");
            assert_eq!(failure.identifier(), Some("RunMat:deconv:InvalidInput"));
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
            let guard =
                crate::output_count::push_output_count(Some(1)).expect("output-count scope");
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
    fn deconv_resident_output_drop_preserves_caller_alias_until_final_lease() {
        test_support::with_test_provider(|provider| {
            let handle = provider
                .upload(&HostTensorView {
                    data: &[1.0, 2.0],
                    shape: &[1, 2],
                })
                .expect("upload");
            let lease = handle.allocation_lease().downgrade();
            let original = handle.wire_descriptor();
            let output = Value::GpuTensor(handle.clone());
            drop(output);
            let downloaded =
                block_on(provider.download(&handle)).expect("caller alias remains live");
            assert_eq!(downloaded.data, vec![1.0, 2.0]);
            assert_eq!(downloaded.shape, vec![1, 2]);
            assert_eq!(handle.wire_descriptor(), original);
            assert!(lease.acquire().is_some());
            handle.release().expect("release final caller lease");
            assert!(lease.acquire().is_none());
        });
    }

    #[test]
    fn deconv_resident_logical_outputs_are_untyped_double() {
        use runmat_accelerate_api::ProviderPrecision;

        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        test_support::with_test_provider(|provider| {
            let numerator = provider
                .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                    data: &[1, 1, 1],
                    shape: &[1, 3],
                })
                .expect("upload logical numerator");
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
                shape: numerator.shape(),
            })
            .expect("upload numerator");
        let den_handle = provider
            .upload(&HostTensorView {
                data: &denominator.materialize_f64(),
                shape: denominator.shape(),
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
        block_on(super::deconv_builtin(None, numerator, denominator))
    }

    #[test]
    fn registered_deconv_binding_honors_invocation_cancellation() {
        let source = runmat_execution::cancellation::CancellationSource::new();
        let runtime = crate::context::RuntimeContext::with_cancellation(
            std::rc::Rc::new(crate::execution::RuntimeExecutionService::new()),
            source.token(),
        );
        source.cancel(runmat_execution::CancellationReason::User);
        let error = block_on(super::deconv_builtin(
            Some(runtime),
            Value::Num(1.0),
            Value::Num(2.0),
        ))
        .expect_err("cancelled invocation");
        assert_eq!(error.identifier(), Some("RunMat:Cancelled"));
        assert_eq!(error.gpu_gather_retry(), GpuGatherRetry::Never);
        assert!(error.source.is_some(), "preserve execution-service cause");
    }
}
