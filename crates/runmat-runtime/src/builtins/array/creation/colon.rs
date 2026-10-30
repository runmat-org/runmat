//! MATLAB-compatible `colon` builtin with GPU-aware semantics for RunMat.
use runmat_builtins::{
    COLON_ERROR_ARG_COUNT, COLON_ERROR_CHAR_CODEPOINT_RANGE,
    COLON_ERROR_CHAR_NON_INTEGER_CODEPOINT, COLON_ERROR_COMPLEX_IMAGINARY_NONZERO,
    COLON_ERROR_INTERNAL, COLON_ERROR_NON_FINITE_INPUT, COLON_ERROR_NON_SCALAR_INPUT,
    COLON_ERROR_SEQUENCE_LIMIT, COLON_ERROR_SEQUENCE_RANGE, COLON_ERROR_UNSUPPORTED_STRING_INPUT,
    COLON_GPU_64_BIT_INTEGER_EXTENSION, COLON_LOGICAL_INPUT_EXTENSION,
    COLON_ZERO_IMAGINARY_COMPLEX_EXTENSION,
};

use runmat_builtins::BuiltinErrorDescriptor;
use runmat_macros::runtime_builtin;
use runmat_value::{CharArray, ComplexTensor, IntValue, LogicalArray, NumericDType, Tensor, Value};

use crate::build_runtime_error;
use crate::builtins::common::integer_conversion::{IntegerClass, IntegerClassExt};
use crate::builtins::common::residency::{sequence_gpu_preference, SequenceIntent};
use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ProviderHook, ReductionNaN, ResidencyPolicy, ScalarType, ShapeRequirements,
};
use crate::builtins::common::{gpu_helpers, tensor};

const MIN_RATIO_TOL: f64 = f64::EPSILON * 8.0;
const MAX_RATIO_TOL: f64 = 1e-9;
const ZERO_IM_TOL: f64 = f64::EPSILON * 32.0;
const CHAR_TOL: f64 = 1e-6;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScalarOrigin {
    Numeric,
    Char,
}

#[derive(Clone)]
struct ParsedScalar {
    value: f64,
    prefer_gpu: bool,
    origin: ScalarOrigin,
    source: Option<runmat_accelerate_api::GpuTensorHandle>,
}

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::array::creation::colon")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "colon",
    op_kind: GpuOpKind::Custom("generator"),
    supported_precisions: &[ScalarType::F32, ScalarType::F64],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[ProviderHook::Custom("sequence")],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::NewHandle,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "The typed double sequence hook preserves exact owner and endpoints; pre-attempt absence permits class-preserving host upload.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::array::creation::colon")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "colon",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Sequence generation is treated as a sink; it does not participate in fusion.",
};

const BUILTIN_NAME: &str = "colon";

fn colon_error(error: &'static BuiltinErrorDescriptor) -> crate::RuntimeError {
    colon_error_with_message(error.message, error)
}

fn colon_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> crate::RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

fn sequence_provider_error(
    stage: &str,
    error: runmat_accelerate_api::ProviderExecutionError,
) -> crate::RuntimeError {
    let mut result = colon_error_with_message(
        format!("colon: {stage} failed: {error}"),
        &COLON_ERROR_INTERNAL,
    );
    result.gpu_gather_retry = crate::GpuGatherRetry::Never;
    crate::runtime_error::with_provider_execution_source(result, error)
}

#[runtime_builtin(
    name = "colon",
    binding_variant = "default",
    builtin_path = "crate::builtins::array::creation::colon"
)]
async fn colon_builtin(
    start: Value,
    step_or_end: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    if rest.len() > 1 {
        return Err(colon_error(&COLON_ERROR_ARG_COUNT));
    }

    ensure_colon_extensions(&start, &step_or_end, rest.view())?;

    if [&start, &step_or_end].into_iter().chain(rest.iter()).any(|value| matches!(value, Value::GpuTensor(handle) if runmat_accelerate_api::handle_integer_type(handle).is_some())) {
        return resident_integer_colon(start, step_or_end, rest).await;
    }

    let (integer_step, integer_stop) = match rest.first() {
        Some(stop) => (Some(&step_or_end), stop),
        None => (None, &step_or_end),
    };
    if let Some(result) = try_integer_sequence(&start, integer_step, integer_stop)? {
        return Ok(result);
    }

    let start_scalar = parse_real_scalar("colon", start).await?;

    if rest.is_empty() {
        let stop_scalar = parse_real_scalar("colon", step_or_end).await?;
        let step = default_step(start_scalar.value, stop_scalar.value);
        let char_mode =
            start_scalar.origin == ScalarOrigin::Char && stop_scalar.origin == ScalarOrigin::Char;
        let explicit_gpu = if char_mode {
            false
        } else {
            start_scalar.prefer_gpu || stop_scalar.prefer_gpu
        };
        let source = sequence_source(&[&start_scalar, &stop_scalar])?;
        let protected = [&start_scalar, &stop_scalar]
            .into_iter()
            .filter_map(|scalar| scalar.source.as_ref())
            .collect::<Vec<_>>();
        build_sequence(
            start_scalar.value,
            step,
            stop_scalar.value,
            explicit_gpu,
            char_mode,
            source.as_ref(),
            &protected,
        )
    } else {
        let step_scalar = parse_real_scalar("colon", step_or_end).await?;
        let stop_scalar = parse_real_scalar("colon", rest[0].clone()).await?;
        let char_mode =
            start_scalar.origin == ScalarOrigin::Char && stop_scalar.origin == ScalarOrigin::Char;
        let explicit_gpu = if char_mode {
            false
        } else {
            start_scalar.prefer_gpu || step_scalar.prefer_gpu || stop_scalar.prefer_gpu
        };
        let source = sequence_source(&[&start_scalar, &step_scalar, &stop_scalar])?;
        let protected = [&start_scalar, &step_scalar, &stop_scalar]
            .into_iter()
            .filter_map(|scalar| scalar.source.as_ref())
            .collect::<Vec<_>>();
        if step_scalar.value == 0.0 {
            return if char_mode {
                build_char_sequence(Vec::new())
            } else {
                finalize_numeric_sequence(Vec::new(), explicit_gpu, source.as_ref(), &protected)
            };
        }
        build_sequence(
            start_scalar.value,
            step_scalar.value,
            stop_scalar.value,
            explicit_gpu,
            char_mode,
            source.as_ref(),
            &protected,
        )
    }
}

async fn resident_integer_colon(
    start: Value,
    step_or_end: Value,
    rest: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<Value> {
    let mut owner_device = None;
    for value in [&start, &step_or_end].into_iter().chain(rest.iter()) {
        if let Value::GpuTensor(handle) = value {
            if runmat_accelerate_api::handle_is_logical(handle) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &COLON_LOGICAL_INPUT_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            if runmat_accelerate_api::handle_storage(handle)
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &COLON_ZERO_IMAGINARY_COMPLEX_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            let device = runmat_accelerate_api::provider_for_handle(handle)
                .ok_or_else(|| {
                    colon_error_with_message(
                        "colon: resident input provider is unavailable",
                        &COLON_ERROR_INTERNAL,
                    )
                })?
                .device_id();
            if owner_device.is_some_and(|owner| owner != device) {
                return Err(colon_error_with_message(
                    "colon: resident inputs must belong to the same provider",
                    &COLON_ERROR_INTERNAL,
                ));
            }
            owner_device = Some(device);
        }
    }
    let provider = [&start, &step_or_end]
        .into_iter()
        .chain(rest.iter())
        .find_map(|value| match value {
            Value::GpuTensor(handle) => runmat_accelerate_api::provider_for_handle(handle),
            _ => None,
        });
    let start = gather_colon_operand(start).await?;
    let step_or_end = gather_colon_operand(step_or_end).await?;
    let rest = match rest.into_iter().next() {
        Some(value) => vec![gather_colon_operand(value).await?],
        None => Vec::new(),
    };
    let (step, stop) = match rest.first() {
        Some(stop) => (Some(&step_or_end), stop),
        None => (None, &step_or_end),
    };
    let result = try_integer_sequence(&start, step, stop)?.ok_or_else(|| {
        colon_error_with_message(
            "colon: resident integer operands require a typed-integer sequence",
            &COLON_ERROR_INTERNAL,
        )
    })?;
    let Some(provider) = provider else {
        return Ok(result);
    };
    let tensor = tensor::value_into_tensor_for(BUILTIN_NAME, result).map_err(|error| {
        colon_error_with_message(format!("colon: {error}"), &COLON_ERROR_INTERNAL)
    })?;
    let handle = gpu_helpers::upload_tensor(provider, &tensor).map_err(|error| {
        colon_error_with_message(format!("colon: {error}"), &COLON_ERROR_INTERNAL)
    })?;
    gpu_helpers::resident_gpu_value(handle)
}

async fn gather_colon_operand(value: Value) -> crate::BuiltinResult<Value> {
    match value {
        Value::GpuTensor(_) => gpu_helpers::gather_value_async(&value).await,
        other => Ok(other),
    }
}

fn ensure_colon_extensions(
    start: &Value,
    step_or_end: &Value,
    rest: crate::call::FunctionArgumentView<'_>,
) -> crate::BuiltinResult<()> {
    for value in std::iter::once(start)
        .chain(std::iter::once(step_or_end))
        .chain(rest.iter())
    {
        if matches!(value, Value::Bool(_) | Value::LogicalArray(_)) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &COLON_LOGICAL_INPUT_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        if matches!(value, Value::Complex(_, _) | Value::ComplexTensor(_)) {
            crate::compatibility::ensure_builtin_extension_enabled(
                &COLON_ZERO_IMAGINARY_COMPLEX_EXTENSION,
                BUILTIN_NAME,
            )?;
        }
        if let Value::GpuTensor(handle) = value {
            if runmat_accelerate_api::handle_is_logical(handle) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &COLON_LOGICAL_INPUT_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            if runmat_accelerate_api::handle_storage(handle)
                == runmat_accelerate_api::GpuTensorStorage::ComplexInterleaved
            {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &COLON_ZERO_IMAGINARY_COMPLEX_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
            if matches!(
                runmat_accelerate_api::handle_integer_type(handle),
                Some(
                    runmat_accelerate_api::IntegerElementType::I64
                        | runmat_accelerate_api::IntegerElementType::U64
                )
            ) {
                crate::compatibility::ensure_builtin_extension_enabled(
                    &COLON_GPU_64_BIT_INTEGER_EXTENSION,
                    BUILTIN_NAME,
                )?;
            }
        }
    }
    Ok(())
}

/// Integer colon expressions must retain their exact class and cannot use the
/// floating compatibility view of a typed tensor. In particular, range checks
/// happen before materialization so `int8(1):256` fails rather than producing
/// a double vector or a saturated final element.
fn try_integer_sequence(
    start: &Value,
    step: Option<&Value>,
    stop: &Value,
) -> crate::BuiltinResult<Option<Value>> {
    let target = integer_target_from_values([start, stop].into_iter().chain(step))?;
    let Some(target) = target else {
        return Ok(None);
    };

    let start = integer_colon_value(start, target, true)?;
    let stop = integer_colon_value(stop, target, true)?;
    let step = match step {
        Some(step) => integer_colon_value(step, target, false)?,
        None => 1,
    };
    if step == 0 {
        let tensor =
            Tensor::new_integer(target.storage(Vec::new()), vec![1, 0]).map_err(|error| {
                colon_error_with_message(format!("colon: {error}"), &COLON_ERROR_INTERNAL)
            })?;
        return Ok(Some(tensor::tensor_into_value(tensor)));
    }

    let count = integer_progression_count(start, step, stop)?;
    let count = usize::try_from(count).map_err(|_| colon_error(&COLON_ERROR_SEQUENCE_LIMIT))?;
    let mut values = Vec::with_capacity(count);
    let mut value = start;
    for index in 0..count {
        values.push(integer_value_from_i128(target, value));
        if index + 1 < count {
            value = value
                .checked_add(step)
                .ok_or_else(|| colon_error(&COLON_ERROR_SEQUENCE_RANGE))?;
        }
    }
    let tensor = Tensor::new_integer(target.storage(values), vec![1, count]).map_err(|error| {
        colon_error_with_message(format!("colon: {error}"), &COLON_ERROR_INTERNAL)
    })?;
    Ok(Some(tensor::tensor_into_value(tensor)))
}

fn integer_target_from_values<'a>(
    values: impl IntoIterator<Item = &'a Value>,
) -> crate::BuiltinResult<Option<IntegerClass>> {
    let mut target = None;
    for value in values {
        let Some(candidate) = typed_integer_target(value)? else {
            continue;
        };
        if let Some(target) = target {
            if target != candidate {
                return Err(colon_error_with_message(
                    "colon: integer operands must have the same integer class",
                    &COLON_ERROR_UNSUPPORTED_STRING_INPUT,
                ));
            }
        } else {
            target = Some(candidate);
        }
    }
    Ok(target)
}

fn typed_integer_target(value: &Value) -> crate::BuiltinResult<Option<IntegerClass>> {
    match value {
        Value::Int(value) => Ok(Some(IntegerClass::from_int_value(value))),
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            if !tensor::is_scalar_tensor(tensor) {
                return Err(colon_error_with_message(
                    COLON_ERROR_NON_SCALAR_INPUT.message,
                    &COLON_ERROR_NON_SCALAR_INPUT,
                ));
            }
            Ok(tensor.integer_storage().map(IntegerClass::from_storage))
        }
        Value::ComplexTensor(tensor) if tensor.integer_storage().is_some() => {
            if complex_tensor_element_len(tensor) != 1 {
                return Err(colon_error_with_message(
                    COLON_ERROR_NON_SCALAR_INPUT.message,
                    &COLON_ERROR_NON_SCALAR_INPUT,
                ));
            }
            let storage = tensor
                .integer_storage()
                .expect("typed complex integer storage");
            if !storage
                .imag()
                .value_at(0)
                .expect("scalar imaginary value")
                .is_zero()
            {
                return Err(colon_error(&COLON_ERROR_COMPLEX_IMAGINARY_NONZERO));
            }
            Ok(Some(IntegerClass::from_storage(storage.real())))
        }
        _ => Ok(None),
    }
}

fn integer_colon_value(
    value: &Value,
    target: IntegerClass,
    require_target_range: bool,
) -> crate::BuiltinResult<i128> {
    let integer = match value {
        Value::Int(value) => {
            if IntegerClass::from_int_value(value) != target {
                return Err(colon_error_with_message(
                    "colon: integer operands must have the same integer class",
                    &COLON_ERROR_UNSUPPORTED_STRING_INPUT,
                ));
            }
            int_value_to_i128(value)
        }
        Value::Tensor(tensor) if tensor.integer_storage().is_some() => {
            if !tensor::is_scalar_tensor(tensor) {
                return Err(colon_error_with_message(
                    COLON_ERROR_NON_SCALAR_INPUT.message,
                    &COLON_ERROR_NON_SCALAR_INPUT,
                ));
            }
            let storage = tensor
                .integer_storage()
                .expect("typed tensor storage is present");
            if IntegerClass::from_storage(storage) != target {
                return Err(colon_error_with_message(
                    "colon: integer operands must have the same integer class",
                    &COLON_ERROR_UNSUPPORTED_STRING_INPUT,
                ));
            }
            int_value_to_i128(&storage.value_at(0).expect("scalar typed tensor value"))
        }
        Value::ComplexTensor(tensor) if tensor.integer_storage().is_some() => {
            if complex_tensor_element_len(tensor) != 1 {
                return Err(colon_error_with_message(
                    COLON_ERROR_NON_SCALAR_INPUT.message,
                    &COLON_ERROR_NON_SCALAR_INPUT,
                ));
            }
            let storage = tensor
                .integer_storage()
                .expect("typed complex integer storage");
            if !storage
                .imag()
                .value_at(0)
                .expect("scalar imaginary value")
                .is_zero()
            {
                return Err(colon_error(&COLON_ERROR_COMPLEX_IMAGINARY_NONZERO));
            }
            if IntegerClass::from_storage(storage.real()) != target {
                return Err(colon_error_with_message(
                    "colon: integer operands must have the same integer class",
                    &COLON_ERROR_UNSUPPORTED_STRING_INPUT,
                ));
            }
            int_value_to_i128(&storage.real().value_at(0).expect("scalar real value"))
        }
        Value::Num(value) => float_to_integral_i128(*value)?,
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            if tensor.numeric_dtype() != NumericDType::F64 {
                return Err(colon_error_with_message(
                    "colon: integer sequences permit only a full scalar double as a noninteger operand",
                    &COLON_ERROR_SEQUENCE_RANGE,
                ));
            }
            float_to_integral_i128(tensor::tensor_value_f64(tensor, 0))?
        }
        Value::Bool(value) => i128::from(u8::from(*value)),
        Value::LogicalArray(array) if array.len() == 1 => i128::from(u8::from(array.data[0] != 0)),
        _ => {
            return Err(colon_error_with_message(
                "colon: integer sequences require real scalar operands",
                &COLON_ERROR_UNSUPPORTED_STRING_INPUT,
            ))
        }
    };
    if require_target_range && !integer_target_contains(target, integer) {
        return Err(colon_error_with_message(
            "colon: integer range values must be representable in the output class",
            &COLON_ERROR_SEQUENCE_RANGE,
        ));
    }
    Ok(integer)
}

fn float_to_integral_i128(value: f64) -> crate::BuiltinResult<i128> {
    if !value.is_finite() {
        return Err(colon_error(&COLON_ERROR_NON_FINITE_INPUT));
    }
    if value.fract() != 0.0 || value < i64::MIN as f64 || value >= 18_446_744_073_709_551_616.0 {
        return Err(colon_error_with_message(
            "colon: integer sequences require integer-valued scalar operands",
            &COLON_ERROR_SEQUENCE_RANGE,
        ));
    }
    Ok(value as i128)
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

fn integer_target_contains(target: IntegerClass, value: i128) -> bool {
    match target {
        IntegerClass::Int8 => value >= i128::from(i8::MIN) && value <= i128::from(i8::MAX),
        IntegerClass::Int16 => value >= i128::from(i16::MIN) && value <= i128::from(i16::MAX),
        IntegerClass::Int32 => value >= i128::from(i32::MIN) && value <= i128::from(i32::MAX),
        IntegerClass::Int64 => value >= i128::from(i64::MIN) && value <= i128::from(i64::MAX),
        IntegerClass::UInt8 => value >= 0 && value <= i128::from(u8::MAX),
        IntegerClass::UInt16 => value >= 0 && value <= i128::from(u16::MAX),
        IntegerClass::UInt32 => value >= 0 && value <= i128::from(u32::MAX),
        IntegerClass::UInt64 => value >= 0 && value <= i128::from(u64::MAX),
    }
}

fn integer_value_from_i128(target: IntegerClass, value: i128) -> IntValue {
    debug_assert!(integer_target_contains(target, value));
    match target {
        IntegerClass::Int8 => IntValue::I8(value as i8),
        IntegerClass::Int16 => IntValue::I16(value as i16),
        IntegerClass::Int32 => IntValue::I32(value as i32),
        IntegerClass::Int64 => IntValue::I64(value as i64),
        IntegerClass::UInt8 => IntValue::U8(value as u8),
        IntegerClass::UInt16 => IntValue::U16(value as u16),
        IntegerClass::UInt32 => IntValue::U32(value as u32),
        IntegerClass::UInt64 => IntValue::U64(value as u64),
    }
}

fn integer_progression_count(start: i128, step: i128, stop: i128) -> crate::BuiltinResult<u128> {
    if (step > 0 && start > stop) || (step < 0 && start < stop) {
        return Ok(0);
    }
    let distance = if step > 0 { stop - start } else { start - stop };
    let step = step.unsigned_abs();
    let count = (distance as u128)
        .checked_div(step)
        .and_then(|count| count.checked_add(1))
        .ok_or_else(|| colon_error(&COLON_ERROR_SEQUENCE_LIMIT))?;
    if count > usize::MAX as u128 {
        return Err(colon_error(&COLON_ERROR_SEQUENCE_LIMIT));
    }
    Ok(count)
}

fn build_sequence(
    start: f64,
    step: f64,
    stop: f64,
    explicit_gpu: bool,
    char_mode: bool,
    source: Option<&runmat_accelerate_api::GpuTensorHandle>,
    protected: &[&runmat_accelerate_api::GpuTensorHandle],
) -> crate::BuiltinResult<Value> {
    if !start.is_finite() || !step.is_finite() || !stop.is_finite() {
        return Err(colon_error(&COLON_ERROR_NON_FINITE_INPUT));
    }
    if step == 0.0 {
        return finalize_numeric_sequence(Vec::new(), explicit_gpu, source, protected);
    }

    let plan = plan_progression(start, step, stop)?;

    if char_mode {
        let data = materialize_progression(&plan, start, step)?;
        return build_char_sequence(data);
    }

    if plan.count == 0 {
        return finalize_numeric_sequence(Vec::new(), explicit_gpu, source, protected);
    }

    let prefer_gpu =
        sequence_gpu_preference(plan.count, SequenceIntent::Colon, explicit_gpu).prefer_gpu;

    if prefer_gpu {
        #[cfg(all(test, feature = "wgpu"))]
        {
            let _ = runmat_accelerate::backend::wgpu::provider::register_wgpu_provider(
                runmat_accelerate::backend::wgpu::provider::WgpuProviderOptions::default(),
            );
        }
        if let Some(provider) = source
            .and_then(gpu_helpers::exact_provider_for_handle)
            .or_else(runmat_accelerate_api::provider)
        {
            let request = runmat_accelerate_api::ProviderSequenceRequest::linear(
                start,
                plan.final_end,
                plan.count,
                runmat_accelerate_api::ProviderPrecision::F64,
            );
            match provider.sequence(&request) {
                Ok(Some(handle)) => {
                    return validated_sequence_output(
                        handle, provider, source, protected, plan.count,
                    )
                }
                Ok(None) => {}
                Err(error) => return Err(sequence_provider_error("provider sequence", error)),
            }
        }
    }

    let data = materialize_progression(&plan, start, step)?;
    finalize_numeric_sequence(data, prefer_gpu, source, protected)
}

fn finalize_numeric_sequence(
    data: Vec<f64>,
    prefer_gpu: bool,
    source: Option<&runmat_accelerate_api::GpuTensorHandle>,
    protected: &[&runmat_accelerate_api::GpuTensorHandle],
) -> crate::BuiltinResult<Value> {
    let len = data.len();
    let shape = vec![1usize, len];
    let tensor = Tensor::new(data, shape)
        .map_err(|e| colon_error_with_message(format!("colon: {e}"), &COLON_ERROR_INTERNAL))?;

    if let Some(source) = source {
        let provider = gpu_helpers::exact_provider_for_handle(source).ok_or_else(|| {
            colon_error_with_message(
                "colon: no exact owner for resident scalar",
                &COLON_ERROR_INTERNAL,
            )
        })?;
        if !gpu_helpers::real_sequence_upload_feasible(
            provider,
            runmat_accelerate_api::ProviderPrecision::F64,
            tensor.shape(),
        ) {
            if runmat_accelerate_api::handle_is_explicit(source) {
                let mut error = colon_error_with_message(
                    "colon: owner cannot preserve double GPU class",
                    &COLON_ERROR_INTERNAL,
                );
                error.gpu_gather_retry = crate::GpuGatherRetry::Never;
                return Err(error);
            }
            return Ok(tensor::tensor_into_value(tensor));
        }
        return gpu_helpers::restore_class_preserving_value_with_operands(
            source,
            Value::Tensor(tensor),
            "colon",
            gpu_helpers::RestoreOperandContext::new(protected, protected),
        );
    }

    if prefer_gpu {
        if let Some(provider) = runmat_accelerate_api::provider() {
            if !gpu_helpers::real_sequence_upload_feasible(
                provider,
                runmat_accelerate_api::ProviderPrecision::F64,
                tensor.shape(),
            ) {
                return Ok(tensor::tensor_into_value(tensor));
            }
            match gpu_helpers::upload_tensor_protecting_typed(provider, &tensor, &[]) {
                Ok(handle) => return validated_sequence_output(handle, provider, None, &[], len),
                Err(runmat_accelerate_api::NumericUploadError::Provider(error))
                    if error.is_unsupported() => {}
                Err(error) => {
                    return Err(sequence_provider_error(
                        "typed sequence upload",
                        error.into_execution_error(),
                    ))
                }
            }
        }
    }

    Ok(tensor::tensor_into_value(tensor))
}

fn validated_sequence_output(
    output: runmat_accelerate_api::GpuTensorHandle,
    provider: &'static dyn runmat_accelerate_api::AccelProvider,
    source: Option<&runmat_accelerate_api::GpuTensorHandle>,
    protected: &[&runmat_accelerate_api::GpuTensorHandle],
    count: usize,
) -> crate::BuiltinResult<Value> {
    let valid = output.shape() == [1, count]
        && output.device_id() == provider.device_id()
        && gpu_helpers::exact_provider_for_handle(&output)
            .is_some_and(|owner| runmat_accelerate_api::same_provider_instance(owner, provider))
        && runmat_accelerate_api::handle_storage(&output)
            == runmat_accelerate_api::GpuTensorStorage::Real
        && runmat_accelerate_api::handle_precision(&output)
            == Some(runmat_accelerate_api::ProviderPrecision::F64)
        && runmat_accelerate_api::handle_integer_type(&output).is_none()
        && !runmat_accelerate_api::handle_is_logical(&output)
        && protected
            .iter()
            .all(|input| !input.shares_allocation_with(&output));
    if !valid {
        return Err(gpu_helpers::reject_gpu_output(
            output,
            colon_error_with_message(
                "colon: malformed provider sequence output",
                &COLON_ERROR_INTERNAL,
            ),
        ));
    }
    let output = if source.is_some() {
        gpu_helpers::propagate_owned_output_provenance(output, protected.iter().copied())?
    } else {
        output
    };
    gpu_helpers::resident_gpu_value(output)
}

struct ProgressionPlan {
    count: usize,
    final_end: f64,
}

fn plan_progression(start: f64, step: f64, stop: f64) -> crate::BuiltinResult<ProgressionPlan> {
    let tol = tolerance(start, step, stop);
    let step_abs = step.abs();

    if step > 0.0 && start > stop + tol {
        return Ok(ProgressionPlan {
            count: 0,
            final_end: start,
        });
    }
    if step < 0.0 && start < stop - tol {
        return Ok(ProgressionPlan {
            count: 0,
            final_end: start,
        });
    }

    let diff = (stop - start) / step;
    if !diff.is_finite() {
        return Err(colon_error(&COLON_ERROR_SEQUENCE_RANGE));
    }

    let ratio_raw = (tol / step_abs).abs();
    let ratio_tol = ratio_raw
        .max(MIN_RATIO_TOL)
        .clamp(f64::EPSILON, MAX_RATIO_TOL);
    let mut approx = diff + ratio_tol;

    if approx < 0.0 {
        if approx.abs() <= ratio_tol {
            approx = 0.0;
        } else {
            return Ok(ProgressionPlan {
                count: 0,
                final_end: start,
            });
        }
    }

    if approx.is_infinite() || approx > usize::MAX as f64 {
        return Err(colon_error(&COLON_ERROR_SEQUENCE_LIMIT));
    }

    let floor = approx.floor();
    let count = floor as usize;
    let count = count
        .checked_add(1)
        .ok_or_else(|| colon_error(&COLON_ERROR_SEQUENCE_LIMIT))?;

    if count == 0 {
        return Ok(ProgressionPlan {
            count: 0,
            final_end: start,
        });
    }

    let computed_end = start + step * ((count - 1) as f64);
    let final_end = if (computed_end - stop).abs() <= tol {
        stop
    } else {
        computed_end
    };

    Ok(ProgressionPlan { count, final_end })
}

fn materialize_progression(
    plan: &ProgressionPlan,
    start: f64,
    step: f64,
) -> crate::BuiltinResult<Vec<f64>> {
    if plan.count == 0 {
        return Ok(Vec::new());
    }
    let mut data = Vec::new();
    data.try_reserve_exact(plan.count).map_err(|error| {
        colon_error_with_message(
            format!("colon: cannot reserve sequence: {error}"),
            &COLON_ERROR_SEQUENCE_LIMIT,
        )
    })?;
    for idx in 0..plan.count {
        data.push(start + step * (idx as f64));
    }
    if let Some(first) = data.first_mut() {
        *first = start;
    }
    if let Some(last) = data.last_mut() {
        *last = plan.final_end;
    }
    Ok(data)
}

fn default_step(_start: f64, _stop: f64) -> f64 {
    // MATLAB's implicit step is always +1. Descending sequences require an explicit
    // negative increment (three-argument form); otherwise the result is empty.
    1.0
}

fn tolerance(start: f64, step: f64, stop: f64) -> f64 {
    let span = (stop - start).abs();
    let base = start.abs().max(stop.abs()).max(span).max(1.0);
    let step_term = step.abs().max(1.0);
    let tol = base * f64::EPSILON * 32.0 + step_term * f64::EPSILON * 16.0;
    tol.max(f64::EPSILON)
}

async fn parse_real_scalar(name: &str, value: Value) -> crate::BuiltinResult<ParsedScalar> {
    match value {
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::numeric_readback_for_handle(&handle).map_err(|_| {
                colon_error_with_message(
                    "colon: no exact owner for resident scalar",
                    &COLON_ERROR_INTERNAL,
                )
            })?;
            let tensor = gpu_helpers::gather_tensor_async(&handle).await?;
            let scalar = tensor_scalar(name, &tensor)?;
            Ok(ParsedScalar {
                value: scalar,
                prefer_gpu: true,
                origin: ScalarOrigin::Numeric,
                source: Some(handle),
            })
        }
        other => parse_real_scalar_host(name, other),
    }
}

fn sequence_source(
    scalars: &[&ParsedScalar],
) -> crate::BuiltinResult<Option<runmat_accelerate_api::GpuTensorHandle>> {
    gpu_helpers::select_resident_output_source(
        scalars.iter().filter_map(|scalar| scalar.source.clone()),
        "colon",
    )
}

fn parse_real_scalar_host(name: &str, value: Value) -> crate::BuiltinResult<ParsedScalar> {
    match value {
        Value::Num(n) => ensure_finite(name, n).map(|v| ParsedScalar {
            value: v,
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
            source: None,
        }),
        Value::Int(i) => Ok(ParsedScalar {
            value: i.to_f64(),
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
            source: None,
        }),
        Value::Bool(b) => Ok(ParsedScalar {
            value: if b { 1.0 } else { 0.0 },
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
            source: None,
        }),
        Value::Tensor(t) => tensor_scalar(name, &t).map(|v| ParsedScalar {
            value: v,
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
            source: None,
        }),
        Value::LogicalArray(logical) => logical_scalar(name, &logical).map(|v| ParsedScalar {
            value: v,
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
            source: None,
        }),
        Value::Complex(re, im) => complex_to_real(name, re, im).map(|v| ParsedScalar {
            value: v,
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
            source: None,
        }),
        Value::ComplexTensor(t) => complex_tensor_scalar(name, &t).map(|v| ParsedScalar {
            value: v,
            prefer_gpu: false,
            origin: ScalarOrigin::Numeric,
            source: None,
        }),
        Value::CharArray(chars) => char_scalar(name, &chars).map(|v| ParsedScalar {
            value: v,
            prefer_gpu: false,
            origin: ScalarOrigin::Char,
            source: None,
        }),
        Value::String(_) | Value::StringArray(_) => Err(colon_error_with_message(
            format!("{name}: inputs must be real scalar values; received a string-like argument"),
            &COLON_ERROR_UNSUPPORTED_STRING_INPUT,
        )),
        Value::GpuTensor(_) => unreachable!("GpuTensor handled by parse_real_scalar"),
        other => Err(colon_error_with_message(
            format!("{name}: inputs must be real scalar values; received {other:?}"),
            &COLON_ERROR_UNSUPPORTED_STRING_INPUT,
        )),
    }
}

fn ensure_finite(name: &str, value: f64) -> crate::BuiltinResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(colon_error_with_message(
            format!("{name}: inputs must be finite numeric scalars"),
            &COLON_ERROR_NON_FINITE_INPUT,
        ))
    }
}

fn tensor_scalar(name: &str, tensor: &Tensor) -> crate::BuiltinResult<f64> {
    if !tensor::is_scalar_tensor(tensor) {
        return Err(colon_error_with_message(
            format!("{name}: expected scalar input"),
            &COLON_ERROR_NON_SCALAR_INPUT,
        ));
    }
    ensure_finite(name, tensor::tensor_value_f64(tensor, 0))
}

fn logical_scalar(name: &str, logical: &LogicalArray) -> crate::BuiltinResult<f64> {
    if logical.len() != 1 {
        return Err(colon_error_with_message(
            format!("{name}: expected scalar input"),
            &COLON_ERROR_NON_SCALAR_INPUT,
        ));
    }
    Ok(if logical.data[0] != 0 { 1.0 } else { 0.0 })
}

fn complex_to_real(name: &str, re: f64, im: f64) -> crate::BuiltinResult<f64> {
    if im.abs() > ZERO_IM_TOL * re.abs().max(1.0) {
        return Err(colon_error_with_message(
            format!("{name}: complex inputs must have zero imaginary part"),
            &COLON_ERROR_COMPLEX_IMAGINARY_NONZERO,
        ));
    }
    ensure_finite(name, re)
}

fn complex_tensor_scalar(name: &str, tensor: &ComplexTensor) -> crate::BuiltinResult<f64> {
    if complex_tensor_element_len(tensor) != 1 {
        return Err(colon_error_with_message(
            format!("{name}: expected scalar input"),
            &COLON_ERROR_NON_SCALAR_INPUT,
        ));
    }
    if let Some(storage) = tensor.integer_storage() {
        let re = storage
            .real()
            .value_at(0)
            .expect("scalar complex integer tensor has one real value")
            .to_f64();
        let im = storage
            .imag()
            .value_at(0)
            .expect("scalar complex integer tensor has one imaginary value")
            .to_f64();
        return complex_to_real(name, re, im);
    }
    let (re, im) = tensor.materialize_f64()[0];
    complex_to_real(name, re, im)
}

fn complex_tensor_element_len(tensor: &ComplexTensor) -> usize {
    tensor.len()
}

fn char_scalar(name: &str, array: &CharArray) -> crate::BuiltinResult<f64> {
    if array.data().len() != 1 {
        return Err(colon_error_with_message(
            format!("{name}: expected scalar input"),
            &COLON_ERROR_NON_SCALAR_INPUT,
        ));
    }
    let ch = array.data()[0];
    Ok(f64::from(ch.raw()))
}

fn build_char_sequence(data: Vec<f64>) -> crate::BuiltinResult<Value> {
    let len = data.len();
    let mut chars = Vec::with_capacity(len);
    for value in data {
        let rounded = value.round();
        if (value - rounded).abs() > CHAR_TOL {
            return Err(colon_error(&COLON_ERROR_CHAR_NON_INTEGER_CODEPOINT));
        }
        if !(0.0..=f64::from(u16::MAX)).contains(&rounded) {
            return Err(colon_error(&COLON_ERROR_CHAR_CODEPOINT_RANGE));
        }
        chars.push(runmat_value::Utf16CodeUnit::from_raw(rounded as u16));
    }

    let array = CharArray::from_code_units(chars, 1, len)
        .map_err(|e| colon_error_with_message(format!("colon: {e}"), &COLON_ERROR_INTERNAL))?;
    Ok(Value::CharArray(array))
}

#[cfg(test)]
mod f32_preferred_owner;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::builtins::common::test_support;
    use futures::executor::block_on;
    use runmat_accelerate_api::HostTensorView;
    use runmat_value::{
        CharArray, ComplexTensor, IntValue, IntegerComplexStorage, IntegerStorage, Tensor,
    };

    fn colon_builtin(start: Value, stop: Value, rest: Vec<Value>) -> crate::BuiltinResult<Value> {
        block_on(super::colon_builtin(
            start,
            stop,
            crate::builtins::common::test_support::builtin_arguments(rest),
        ))
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_basic_increasing() {
        let result = colon_builtin(Value::Num(1.0), Value::Num(5.0), Vec::new()).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), vec![1, 5]);
                assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn colon_preserves_exact_integer_storage_for_all_classes() {
        let cases = [
            (
                IntValue::I8(-2),
                IntValue::I8(2),
                IntegerStorage::I8(vec![-2, -1, 0, 1, 2]),
            ),
            (
                IntValue::I16(-2),
                IntValue::I16(2),
                IntegerStorage::I16(vec![-2, -1, 0, 1, 2]),
            ),
            (
                IntValue::I32(-2),
                IntValue::I32(2),
                IntegerStorage::I32(vec![-2, -1, 0, 1, 2]),
            ),
            (
                IntValue::I64(-2),
                IntValue::I64(2),
                IntegerStorage::I64(vec![-2, -1, 0, 1, 2]),
            ),
            (
                IntValue::U8(0),
                IntValue::U8(4),
                IntegerStorage::U8(vec![0, 1, 2, 3, 4]),
            ),
            (
                IntValue::U16(0),
                IntValue::U16(4),
                IntegerStorage::U16(vec![0, 1, 2, 3, 4]),
            ),
            (
                IntValue::U32(0),
                IntValue::U32(4),
                IntegerStorage::U32(vec![0, 1, 2, 3, 4]),
            ),
            (
                IntValue::U64(9_007_199_254_740_992),
                IntValue::U64(9_007_199_254_740_994),
                IntegerStorage::U64(vec![
                    9_007_199_254_740_992,
                    9_007_199_254_740_993,
                    9_007_199_254_740_994,
                ]),
            ),
        ];

        for (start, stop, expected) in cases {
            let value = colon_builtin(Value::Int(start), Value::Int(stop), Vec::new())
                .expect("integer colon");
            let Value::Tensor(tensor) = value else {
                panic!("integer sequence should be an array");
            };
            assert_eq!(tensor.integer_storage(), Some(&expected));
        }
    }

    #[test]
    fn colon_integer_sequences_support_signed_steps_and_reject_unrepresentable_ranges() {
        let value = colon_builtin(
            Value::Int(IntValue::U32(5)),
            Value::Num(-2.0),
            vec![Value::Int(IntValue::U32(1))],
        )
        .expect("unsigned endpoint with signed step");
        let Value::Tensor(tensor) = value else {
            panic!("integer sequence should be an array");
        };
        assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::U32(vec![5, 3, 1]))
        );

        let err = colon_builtin(Value::Int(IntValue::I8(1)), Value::Num(256.0), Vec::new())
            .expect_err("out-of-range integer endpoint must fail");
        assert!(err.message().contains("representable"));

        let err = colon_builtin(
            Value::Int(IntValue::I16(1)),
            Value::Int(IntValue::I8(2)),
            Vec::new(),
        )
        .expect_err("mixed integer classes must fail");
        assert!(err.message().contains("same integer class"));

        assert_eq!(
            colon_builtin(
                Value::Int(IntValue::U64(u64::MAX)),
                Value::Int(IntValue::U64(u64::MAX)),
                Vec::new(),
            )
            .expect("singleton uint64 range"),
            Value::Int(IntValue::U64(u64::MAX))
        );
    }

    #[test]
    fn colon_accepts_each_integer_class_in_every_operand_position_with_scalar_doubles() {
        let values = [
            IntValue::I8(1),
            IntValue::I16(1),
            IntValue::I32(1),
            IntValue::I64(1),
            IntValue::U8(1),
            IntValue::U16(1),
            IntValue::U32(1),
            IntValue::U64(1),
        ];
        for value in values {
            let target = IntegerClass::from_int_value(&value);
            let typed_one = integer_value_from_i128(target, 1);
            let typed_three = integer_value_from_i128(target, 3);
            for result in [
                colon_builtin(Value::Int(typed_one.clone()), Value::Num(3.0), Vec::new()),
                colon_builtin(
                    Value::Num(1.0),
                    Value::Int(typed_one),
                    vec![Value::Num(3.0)],
                ),
                colon_builtin(
                    Value::Num(1.0),
                    Value::Num(1.0),
                    vec![Value::Int(typed_three)],
                ),
            ] {
                let Value::Tensor(output) = result.expect("mixed scalar-double colon") else {
                    panic!("expected typed tensor");
                };
                assert_eq!(
                    output.integer_storage(),
                    Some(&target.storage(vec![
                        integer_value_from_i128(target, 1),
                        integer_value_from_i128(target, 2),
                        integer_value_from_i128(target, 3),
                    ]))
                );
            }
        }
    }

    #[test]
    fn colon_rejects_single_and_unlike_integer_companions() {
        let single = Tensor::from_f32(vec![3.0], vec![1, 1]).expect("single scalar");
        let error = colon_builtin(
            Value::Int(IntValue::I16(1)),
            Value::Tensor(single),
            Vec::new(),
        )
        .expect_err("single companion must reject");
        assert!(error.message().contains("scalar double"));

        let error = colon_builtin(
            Value::Int(IntValue::U8(1)),
            Value::Num(1.0),
            vec![Value::Int(IntValue::I8(3))],
        )
        .expect_err("unlike integer classes must reject");
        assert!(error.message().contains("same integer class"));
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_two_arg_descending_returns_empty() {
        let result = colon_builtin(Value::Num(5.0), Value::Num(1.0), Vec::new()).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), vec![1, 0]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_three_arg_descending() {
        let result =
            colon_builtin(Value::Num(5.0), Value::Num(-1.0), vec![Value::Num(1.0)]).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), vec![1, 5]);
                assert_eq!(t.materialize_f64(), vec![5.0, 4.0, 3.0, 2.0, 1.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_custom_step_reaches_stop() {
        let result =
            colon_builtin(Value::Num(0.0), Value::Num(0.5), vec![Value::Num(2.0)]).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), vec![1, 5]);
                assert_eq!(t.materialize_f64(), vec![0.0, 0.5, 1.0, 1.5, 2.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_custom_step_stops_before_bound() {
        let result =
            colon_builtin(Value::Num(0.0), Value::Num(2.0), vec![Value::Num(5.0)]).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), vec![1, 3]);
                assert_eq!(t.materialize_f64(), vec![0.0, 2.0, 4.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_sign_mismatch_returns_empty() {
        let result =
            colon_builtin(Value::Num(0.0), Value::Num(1.0), vec![Value::Num(-1.0)]).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.shape(), vec![1, 0]);
                assert!(t.materialize_f64().is_empty());
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_zero_increment_returns_empty_in_selected_class() {
        let Value::Tensor(result) =
            colon_builtin(Value::Num(0.0), Value::Num(0.0), vec![Value::Num(1.0)]).expect("empty")
        else {
            panic!("expected empty tensor");
        };
        assert_eq!(result.shape(), vec![1, 0]);
        let Value::Tensor(typed) = colon_builtin(
            Value::Int(IntValue::U16(0)),
            Value::Num(0.0),
            vec![Value::Int(IntValue::U16(1))],
        )
        .expect("typed empty") else {
            panic!("expected typed empty");
        };
        assert_eq!(
            typed.integer_storage(),
            Some(&IntegerStorage::U16(Vec::new()))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_accepts_scalar_tensors() {
        let start = Tensor::new(vec![1.0], vec![1, 1]).unwrap();
        let stop = Tensor::new(vec![3.0], vec![1, 1]).unwrap();
        let result =
            colon_builtin(Value::Tensor(start), Value::Tensor(stop), Vec::new()).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.materialize_f64(), vec![1.0, 2.0, 3.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn colon_scalar_tensors_read_typed_integer_storage_exactly() {
        let start = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_993]),
            vec![1, 1],
        )
        .unwrap();
        let stop = Tensor::new_integer(
            runmat_value::IntegerStorage::U64(vec![9_007_199_254_740_995]),
            vec![1, 1],
        )
        .unwrap();

        let result =
            colon_builtin(Value::Tensor(start), Value::Tensor(stop), Vec::new()).expect("colon");
        match result {
            Value::Tensor(tensor) => assert_eq!(
                tensor.integer_storage(),
                Some(&runmat_value::IntegerStorage::U64(vec![
                    9_007_199_254_740_993,
                    9_007_199_254_740_994,
                    9_007_199_254_740_995,
                ]))
            ),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn colon_double_path_scalar_tensor_reads_typed_integer_storage_exactly() {
        let start =
            Tensor::new_integer(runmat_value::IntegerStorage::U16(vec![4]), vec![1, 1]).unwrap();
        let stop = Tensor::new(vec![6.0], vec![1, 1]).unwrap();

        let result =
            colon_builtin(Value::Tensor(start), Value::Tensor(stop), Vec::new()).expect("colon");
        match result {
            Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![4.0, 5.0, 6.0]),
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[test]
    fn colon_int64_accepts_integral_scalar_double_values_and_tensors_exactly() {
        let start = Value::Int(IntValue::I64(1));
        let Value::Tensor(scalar) =
            colon_builtin(start.clone(), Value::Num(3.0), Vec::new()).expect("scalar double")
        else {
            panic!("expected typed range");
        };
        assert_eq!(
            scalar.integer_storage(),
            Some(&IntegerStorage::I64(vec![1, 2, 3]))
        );

        let double_tensor = Tensor::new(vec![3.0], vec![1, 1]).expect("double scalar");
        let Value::Tensor(tensor) = colon_builtin(start, Value::Tensor(double_tensor), Vec::new())
            .expect("scalar double tensor")
        else {
            panic!("expected typed range");
        };
        assert_eq!(
            tensor.integer_storage(),
            Some(&IntegerStorage::I64(vec![1, 2, 3]))
        );
    }

    #[test]
    fn colon_resident_uint32_preserves_exact_class_and_residency() {
        test_support::with_test_provider(|provider| {
            let start = Tensor::new_integer(IntegerStorage::U32(vec![u32::MAX - 2]), vec![1, 1])
                .expect("start");
            let stop =
                Tensor::new_integer(IntegerStorage::U32(vec![u32::MAX]), vec![1, 1]).expect("stop");
            let start = gpu_helpers::upload_tensor(provider, &start).expect("upload start");
            let stop = gpu_helpers::upload_tensor(provider, &stop).expect("upload stop");
            let result = colon_builtin(Value::GpuTensor(start), Value::GpuTensor(stop), Vec::new())
                .expect("resident colon");
            let Value::GpuTensor(handle) = &result else {
                panic!("expected resident output");
            };
            assert_eq!(
                runmat_accelerate_api::handle_integer_type(handle),
                Some(runmat_accelerate_api::IntegerElementType::U32)
            );
            let gathered = block_on(gpu_helpers::gather_value_async(&result)).expect("gather");
            let Value::Tensor(tensor) = gathered else {
                panic!("expected typed tensor");
            };
            assert_eq!(
                tensor.integer_storage(),
                Some(&IntegerStorage::U32(vec![
                    u32::MAX - 2,
                    u32::MAX - 1,
                    u32::MAX
                ]))
            );
        });
    }

    #[test]
    fn colon_resident_uint64_is_independently_mode_gated() {
        test_support::with_test_provider(|provider| {
            let value = Tensor::new_integer(IntegerStorage::U64(vec![u64::MAX]), vec![1, 1])
                .expect("value");
            let handle = gpu_helpers::upload_tensor(provider, &value).expect("upload");
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = colon_builtin(
                Value::GpuTensor(handle.clone()),
                Value::GpuTensor(handle),
                Vec::new(),
            )
            .expect_err("uint64 GPU colon is extension");
            assert_eq!(
                error.identifier(),
                COLON_GPU_64_BIT_INTEGER_EXTENSION.error_identifier
            );
        });
    }

    #[test]
    fn colon_resident_logical_and_complex_metadata_are_gated_before_dispatch() {
        test_support::with_test_provider(|provider| {
            let logical = provider
                .upload_logical(&runmat_accelerate_api::HostLogicalTensorView {
                    data: &[1],
                    shape: &[1, 1],
                })
                .expect("logical upload");
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = colon_builtin(
                Value::GpuTensor(logical.clone()),
                Value::GpuTensor(logical),
                Vec::new(),
            )
            .expect_err("logical gate");
            assert_eq!(
                error.identifier(),
                COLON_LOGICAL_INPUT_EXTENSION.error_identifier
            );
            drop(_compat);
            let complex = gpu_helpers::upload_complex_tensor(
                provider,
                &runmat_value::ComplexTensor::new(vec![(1.0, 0.0)], vec![1, 1]).unwrap(),
            )
            .expect("complex upload");
            let _compat = crate::compatibility::push_runmat_extensions_enabled(false);
            let error = colon_builtin(
                Value::GpuTensor(complex.clone()),
                Value::GpuTensor(complex),
                Vec::new(),
            )
            .expect_err("complex gate");
            assert_eq!(
                error.identifier(),
                COLON_ZERO_IMAGINARY_COMPLEX_EXTENSION.error_identifier
            );
        });
    }

    #[test]
    fn colon_complex_integer_scalar_tensors_read_exact_storage() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let start = ComplexTensor::new_integer(
            IntegerComplexStorage::new(IntegerStorage::I16(vec![1]), IntegerStorage::I16(vec![0]))
                .expect("complex storage"),
            vec![1, 1],
        )
        .expect("start");
        let stop = ComplexTensor::new_integer(
            IntegerComplexStorage::new(IntegerStorage::I16(vec![3]), IntegerStorage::I16(vec![0]))
                .expect("complex storage"),
            vec![1, 1],
        )
        .expect("stop");

        let result = colon_builtin(
            Value::ComplexTensor(start),
            Value::ComplexTensor(stop),
            Vec::new(),
        )
        .expect("colon");
        match result {
            Value::Tensor(tensor) => assert_eq!(tensor.materialize_f64(), vec![1.0, 2.0, 3.0]),
            other => panic!("expected tensor, got {other:?}"),
        }

        let nonzero_imag = ComplexTensor::new_integer(
            IntegerComplexStorage::new(IntegerStorage::I16(vec![1]), IntegerStorage::I16(vec![1]))
                .expect("complex storage"),
            vec![1, 1],
        )
        .expect("nonzero imag");
        let err = colon_builtin(
            Value::ComplexTensor(nonzero_imag),
            Value::Num(3.0),
            Vec::new(),
        )
        .expect_err("nonzero imaginary part must reject");
        assert_eq!(
            err.identifier(),
            COLON_ERROR_COMPLEX_IMAGINARY_NONZERO.identifier
        );

        let wide_start = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![9_007_199_254_740_993]),
                IntegerStorage::U64(vec![0]),
            )
            .expect("wide start"),
            vec![1, 1],
        )
        .expect("wide start tensor");
        let wide_stop = ComplexTensor::new_integer(
            IntegerComplexStorage::new(
                IntegerStorage::U64(vec![9_007_199_254_740_995]),
                IntegerStorage::U64(vec![0]),
            )
            .expect("wide stop"),
            vec![1, 1],
        )
        .expect("wide stop tensor");
        let Value::Tensor(wide) = colon_builtin(
            Value::ComplexTensor(wide_start),
            Value::ComplexTensor(wide_stop),
            Vec::new(),
        )
        .expect("wide exact complex colon") else {
            panic!("expected typed range");
        };
        assert_eq!(
            wide.integer_storage(),
            Some(&IntegerStorage::U64(vec![
                9_007_199_254_740_993,
                9_007_199_254_740_994,
                9_007_199_254_740_995
            ]))
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_gpu_roundtrip() {
        test_support::with_test_provider(|provider| {
            let start = Tensor::new(vec![0.0], vec![1, 1]).unwrap();
            let start_view = HostTensorView {
                data: &start.materialize_f64(),
                shape: start.shape(),
            };
            let start_handle = provider.upload(&start_view).expect("upload start");

            let result = colon_builtin(
                Value::GpuTensor(start_handle),
                Value::Num(0.5),
                vec![Value::Num(2.0)],
            )
            .expect("colon");

            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.shape(), vec![1, 5]);
                    assert_eq!(gathered.materialize_f64(), vec![0.0, 0.5, 1.0, 1.5, 2.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }

    #[test]
    fn double_colon_uses_f64_capability_of_f32_preferred_owner() {
        super::f32_preferred_owner::with_provider(|provider| {
            assert_eq!(
                provider.precision(),
                runmat_accelerate_api::ProviderPrecision::F32
            );
            assert!(gpu_helpers::real_sequence_upload_feasible(
                provider,
                runmat_accelerate_api::ProviderPrecision::F64,
                &[1, 3],
            ));
            let output = finalize_numeric_sequence(vec![0.0, 0.5, 1.0], true, None, &[])
                .expect("class-preserving sequence");
            let Value::GpuTensor(handle) = output else {
                panic!("expected resident double")
            };
            assert_eq!(
                runmat_accelerate_api::handle_precision(&handle),
                Some(runmat_accelerate_api::ProviderPrecision::F64)
            );
            assert_eq!(handle.device_id(), provider.device_id());
            assert_eq!(handle.shape(), [1, 3]);
            let tensor = test_support::gather(Value::GpuTensor(handle)).expect("gather");
            assert_eq!(tensor.numeric_dtype(), NumericDType::F64);
            assert_eq!(tensor.materialize_f64(), vec![0.0, 0.5, 1.0]);
            // This finite value would round to 1 in F32: a preferred-class
            // conversion plus an F64 tag cannot satisfy the native oracle.
            let exact = 1.0_f64 + 2.0_f64.powi(-40);
            let output = finalize_numeric_sequence(vec![exact], true, None, &[])
                .expect("native F64 sequence despite F32 preference");
            let Value::GpuTensor(handle) = output else {
                panic!("expected native resident F64 probe");
            };
            assert_eq!(
                runmat_accelerate_api::handle_precision(&handle),
                Some(runmat_accelerate_api::ProviderPrecision::F64)
            );
            let tensor = test_support::gather(Value::GpuTensor(handle)).expect("native readback");
            assert_eq!(tensor.as_f64_slice().unwrap()[0].to_bits(), exact.to_bits());
        });
    }

    #[test]
    fn colon_host_progression_retains_negative_zero_start() {
        let plan = plan_progression(-0.0, 1.0, 1.0).unwrap();
        assert_eq!(
            materialize_progression(&plan, -0.0, 1.0).unwrap()[0].to_bits(),
            (-0.0f64).to_bits()
        );
    }

    #[test]
    fn colon_host_progression_rejects_unreservable_count() {
        let plan = ProgressionPlan {
            count: usize::MAX,
            final_end: 1.0,
        };
        assert!(materialize_progression(&plan, 0.0, 1.0).is_err());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    #[cfg(feature = "wgpu")]
    fn colon_wgpu_matches_cpu() {
        use runmat_accelerate::backend::wgpu::provider::{
            register_wgpu_provider, WgpuProviderOptions,
        };

        let _ = register_wgpu_provider(WgpuProviderOptions::default());

        let cpu = colon_builtin(Value::Num(-2.0), Value::Num(0.5), vec![Value::Num(1.0)])
            .expect("colon host");

        let provider = runmat_accelerate_api::provider().expect("wgpu provider");
        let start = Tensor::new(vec![-2.0], vec![1, 1]).unwrap();
        let start_view = HostTensorView {
            data: &start.materialize_f64(),
            shape: start.shape(),
        };
        let start_handle = provider.upload(&start_view).expect("upload start");
        let gpu = colon_builtin(
            Value::GpuTensor(start_handle),
            Value::Num(0.5),
            vec![Value::Num(1.0)],
        )
        .expect("colon gpu");

        let gathered = match gpu {
            Value::GpuTensor(handle) => {
                test_support::gather(Value::GpuTensor(handle)).expect("gather gpu")
            }
            Value::Tensor(tensor)
                if provider.precision() == runmat_accelerate_api::ProviderPrecision::F32 =>
            {
                assert_eq!(tensor.numeric_dtype(), NumericDType::F64);
                tensor
            }
            other => panic!("expected GPU tensor, got {other:?}"),
        };

        let expected = match cpu {
            Value::Tensor(t) => t,
            Value::Num(n) => Tensor::new(vec![n], vec![1, 1]).unwrap(),
            other => panic!("unexpected CPU result {other:?}"),
        };

        assert_eq!(gathered.shape(), expected.shape());
        assert_eq!(gathered.materialize_f64(), expected.materialize_f64());
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_bool_inputs_promote() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let result =
            colon_builtin(Value::Bool(false), Value::Bool(true), Vec::new()).expect("colon");
        match result {
            Value::Tensor(t) => {
                assert_eq!(t.materialize_f64(), vec![0.0, 1.0]);
            }
            other => panic!("expected tensor, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_char_increasing() {
        let raw = |unit| {
            Value::CharArray(
                CharArray::from_code_units(vec![runmat_value::Utf16CodeUnit::from_raw(unit)], 1, 1)
                    .expect("raw endpoint"),
            )
        };
        let Value::CharArray(units) =
            colon_builtin(raw(0xD7FF), raw(0xD801), Vec::new()).expect("raw UTF-16 range")
        else {
            panic!("expected character range");
        };
        assert_eq!(
            units
                .data()
                .iter()
                .map(|unit| unit.raw())
                .collect::<Vec<_>>(),
            [0xD7FF, 0xD800, 0xD801]
        );
        let Value::CharArray(last) =
            colon_builtin(raw(0xFFFF), raw(0xFFFF), Vec::new()).expect("last code unit")
        else {
            panic!("expected character range");
        };
        assert_eq!(last.data()[0].raw(), 0xFFFF);
        assert!(build_char_sequence(vec![65536.0]).is_err());
        assert!(char_scalar("colon", &CharArray::new_row("\u{1F642}")).is_err());
        let empty = CharArray::from_code_units_with_shape(Vec::new(), vec![1, 1, 0]).unwrap();
        assert!(char_scalar("colon", &empty).is_err());
        let start = Value::CharArray(CharArray::new_row("a"));
        let stop = Value::CharArray(CharArray::new_row("e"));
        let result = colon_builtin(start, stop, Vec::new()).expect("colon");
        match result {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 1);
                assert_eq!(arr.cols, 5);
                let expected: Vec<_> = "abcde"
                    .encode_utf16()
                    .map(runmat_value::Utf16CodeUnit::from_raw)
                    .collect();
                assert_eq!(arr.data(), expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_char_with_step() {
        let start = Value::CharArray(CharArray::new_row("a"));
        let step = Value::Num(2.0);
        let stop = Value::CharArray(CharArray::new_row("g"));
        let result = colon_builtin(start, step, vec![stop]).expect("colon");
        match result {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 1);
                assert_eq!(arr.cols, 4);
                let expected: Vec<_> = "aceg"
                    .encode_utf16()
                    .map(runmat_value::Utf16CodeUnit::from_raw)
                    .collect();
                assert_eq!(arr.data(), expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_equal_endpoints_singleton() {
        let result = colon_builtin(Value::Num(3.0), Value::Num(3.0), Vec::new()).expect("colon");
        match result {
            Value::Num(v) => assert!((v - 3.0).abs() < 1e-12),
            Value::Tensor(t) => {
                assert_eq!(t.materialize_f64(), vec![3.0]);
            }
            other => panic!("expected scalar-compatible result, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_complex_imaginary_errors() {
        let _compat = crate::compatibility::push_runmat_extensions_enabled(true);
        let err = colon_builtin(Value::Complex(1.0, 1e-2), Value::Num(2.0), Vec::new())
            .expect_err("colon should reject complex inputs");
        assert!(
            err.message().contains("zero imaginary part"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_string_input_errors() {
        let err = colon_builtin(Value::from("hello"), Value::Num(2.0), Vec::new())
            .expect_err("colon should reject string inputs");
        assert!(
            err.message().contains("string-like"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_char_two_arg_descending_returns_empty() {
        let start = Value::CharArray(CharArray::new_row("f"));
        let stop = Value::CharArray(CharArray::new_row("b"));
        let result = colon_builtin(start, stop, Vec::new()).expect("colon");
        match result {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 1);
                assert_eq!(arr.cols, 0);
                assert!(arr.data().is_empty());
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_char_three_arg_descending() {
        let start = Value::CharArray(CharArray::new_row("f"));
        let step = Value::Num(-1.0);
        let stop = Value::CharArray(CharArray::new_row("b"));
        let result = colon_builtin(start, step, vec![stop]).expect("colon");
        match result {
            Value::CharArray(arr) => {
                assert_eq!(arr.rows, 1);
                assert_eq!(arr.cols, 5);
                let expected: Vec<_> = "fedcb"
                    .encode_utf16()
                    .map(runmat_value::Utf16CodeUnit::from_raw)
                    .collect();
                assert_eq!(arr.data(), expected);
            }
            other => panic!("expected char array, got {other:?}"),
        }
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_char_fractional_step_errors() {
        let start = Value::CharArray(CharArray::new_row("a"));
        let stop = Value::CharArray(CharArray::new_row("d"));
        let err = colon_builtin(start, Value::Num(1.5), vec![stop])
            .expect_err("colon should reject fractional char steps");
        assert!(
            err.message()
                .contains("character sequence requires integer"),
            "unexpected error message: {err}"
        );
    }

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn colon_gpu_step_scalar_residency() {
        test_support::with_test_provider(|provider| {
            let step = Tensor::new(vec![0.5], vec![1, 1]).unwrap();
            let view = HostTensorView {
                data: &step.materialize_f64(),
                shape: step.shape(),
            };
            let step_handle = provider.upload(&view).expect("upload step");
            let result = colon_builtin(
                Value::Num(0.0),
                Value::GpuTensor(step_handle),
                vec![Value::Num(2.0)],
            )
            .expect("colon");
            match result {
                Value::GpuTensor(handle) => {
                    let gathered = test_support::gather(Value::GpuTensor(handle)).expect("gather");
                    assert_eq!(gathered.materialize_f64(), vec![0.0, 0.5, 1.0, 1.5, 2.0]);
                }
                other => panic!("expected GPU tensor, got {other:?}"),
            }
        });
    }
}
