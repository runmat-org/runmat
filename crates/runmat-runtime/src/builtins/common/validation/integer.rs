//! Shared integer classification and exact conversion boundaries.

use super::{
    build_runtime_error, handle_integer_type, handle_is_logical, handle_storage, BuiltinResult,
    ComplexTensor, GpuTensorStorage, IntValue, RuntimeError, Value,
};
use runmat_builtins::BuiltinExtensionDescriptor;

/// MATLAB stores complex integer arrays, but does not support arithmetic on
/// them. Arithmetic builtins use this before selecting floating or provider
/// execution paths so exact integer components are never coerced to `f64`.
pub fn is_typed_complex_integer(value: &Value) -> bool {
    matches!(value, Value::ComplexTensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle)
            if handle_integer_type(handle).is_some()
                && handle_storage(handle) == GpuTensorStorage::ComplexInterleaved)
}

/// Reject a value that would otherwise enter a floating complex operation.
pub fn reject_typed_complex_integer(value: &Value, builtin: &str) -> BuiltinResult<()> {
    if is_typed_complex_integer(value) {
        return Err(build_runtime_error(format!(
            "{builtin}: operations involving complex numbers with integer types are not supported"
        ))
        .build());
    }
    Ok(())
}

/// Reject operations that would consume the lossy `f64` compatibility view of
/// a typed complex integer tensor. MATLAB permits storage/inspection of these
/// values, but not operations on them.
pub fn reject_typed_complex_integer_tensor(
    tensor: &ComplexTensor,
    builtin: &str,
) -> BuiltinResult<()> {
    if tensor.integer_storage().is_some() {
        return Err(build_runtime_error(format!(
            "{builtin}: operations involving complex numbers with integer types are not supported"
        ))
        .build());
    }
    Ok(())
}

pub fn value_has_native_integer_class(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_some(),
        Value::SparseTensor(tensor) => tensor.integer_storage().is_some(),
        Value::ComplexTensor(tensor) => tensor.integer_storage().is_some(),
        Value::GpuTensor(handle) => handle_integer_type(handle).is_some(),
        _ => false,
    }
}

/// Returns whether `value`, including aggregate payloads, contains a native
/// integer class. Use this at compatibility boundaries before a recursive
/// gather can erase resident class metadata.
pub fn value_contains_native_integer_class(value: &Value) -> bool {
    match value {
        Value::Cell(cell) => cell.data.iter().any(value_contains_native_integer_class),
        Value::Struct(value) => value
            .fields
            .values()
            .any(value_contains_native_integer_class),
        Value::Object(value) => value
            .properties
            .values()
            .any(value_contains_native_integer_class),
        Value::Closure(value) => value
            .captures
            .iter()
            .any(value_contains_native_integer_class),
        Value::OutputList(values) => values.iter().any(value_contains_native_integer_class),
        _ => value_has_native_integer_class(value),
    }
}

/// Returns whether `value`, including aggregate payloads, contains a handle
/// created through explicit `gpuArray` intent rather than automatic residency.
pub fn value_contains_explicit_gpu(value: &Value) -> bool {
    match value {
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_is_explicit(handle),
        Value::Cell(cell) => cell.data.iter().any(value_contains_explicit_gpu),
        Value::Struct(value) => value.fields.values().any(value_contains_explicit_gpu),
        Value::Object(value) => value.properties.values().any(value_contains_explicit_gpu),
        Value::Closure(value) => value.captures.iter().any(value_contains_explicit_gpu),
        Value::OutputList(values) => values.iter().any(value_contains_explicit_gpu),
        _ => false,
    }
}

pub fn value_has_logical_class(value: &Value) -> bool {
    matches!(value, Value::Bool(_) | Value::LogicalArray(_))
        || matches!(value, Value::SparseTensor(tensor) if tensor.is_logical())
        || matches!(value, Value::GpuTensor(handle) if handle_is_logical(handle))
}

/// Returns whether a fixed-width integer can cross a binary64 boundary
/// without changing its value. Exactness depends on significant bits, so an
/// aligned wide integer can be exact even when its magnitude exceeds
/// `flintmax`.
pub fn integer_is_exact_f64(value: &IntValue) -> bool {
    let magnitude = match value {
        IntValue::I8(value) => u64::from(value.unsigned_abs()),
        IntValue::I16(value) => u64::from(value.unsigned_abs()),
        IntValue::I32(value) => u64::from(value.unsigned_abs()),
        IntValue::I64(value) => value.unsigned_abs(),
        IntValue::U8(value) => u64::from(*value),
        IntValue::U16(value) => u64::from(*value),
        IntValue::U32(value) => u64::from(*value),
        IntValue::U64(value) => *value,
    };
    if magnitude == 0 {
        return true;
    }
    let significant_bits = u64::BITS - magnitude.leading_zeros();
    significant_bits <= f64::MANTISSA_DIGITS
        || magnitude.trailing_zeros() >= significant_bits - f64::MANTISSA_DIGITS
}

pub fn native_integer_value_is_exact_f64(value: &Value) -> bool {
    let exact = integer_is_exact_f64;
    match value {
        Value::Int(value) => exact(value),
        Value::Tensor(tensor) => tensor
            .integer_storage()
            .is_none_or(|storage| storage.exact_values().iter().all(exact)),
        Value::SparseTensor(tensor) => tensor
            .integer_storage()
            .is_none_or(|storage| storage.exact_values().iter().all(exact)),
        Value::ComplexTensor(tensor) => tensor.integer_storage().is_none_or(|storage| {
            storage.real.exact_values().iter().all(exact)
                && storage.imag.exact_values().iter().all(exact)
        }),
        Value::Cell(cell) => cell.data.iter().all(native_integer_value_is_exact_f64),
        Value::Struct(value) => value.fields.values().all(native_integer_value_is_exact_f64),
        Value::OutputList(values) => values.iter().all(native_integer_value_is_exact_f64),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_type(handle)
            .is_none_or(|element_type| element_type.element_size() <= 4),
        _ => true,
    }
}

/// Check a native integer value against a binary64 boundary without rejecting a
/// resident 64-bit value solely because its contents are not visible in handle
/// metadata. Compatibility admission must run before calling this helper.
pub async fn native_integer_value_is_exact_f64_async(value: &Value) -> Result<bool, RuntimeError> {
    if native_integer_value_is_exact_f64(value) {
        return Ok(true);
    }
    if matches!(value, Value::GpuTensor(handle) if handle_integer_type(handle).is_some()) {
        let Value::GpuTensor(handle) = value else {
            unreachable!();
        };
        let provider = crate::builtins::common::gpu_helpers::exact_provider_for_handle(handle)
            .ok_or_else(|| {
                build_runtime_error(
                    "integer exactness check: no acceleration provider owns the input handle",
                )
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
            })?;
        let expected_type = handle_integer_type(handle).expect("integer type checked above");
        if runmat_accelerate_api::handle_storage(handle)
            != runmat_accelerate_api::GpuTensorStorage::Real
            || handle_is_logical(handle)
            || runmat_accelerate_api::handle_precision(handle).is_some()
            || !crate::builtins::common::gpu_helpers::gpu_class_metadata_matches(
                handle,
                None,
                Some(expected_type),
                false,
            )
        {
            return Err(build_runtime_error(
                "integer exactness check: input handle has contradictory integer metadata",
            )
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build());
        }
        let snapshot = crate::builtins::common::gpu_helpers::snapshot_handle_metadata(handle);
        let gathered = provider.download_integer(handle).await.map_err(|error| {
            build_runtime_error(format!(
                "integer exactness check: owner-preserving download failed: {error}"
            ))
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build()
        });
        crate::builtins::common::gpu_helpers::restore_handle_metadata(handle, &snapshot);
        let gathered = gathered?;
        if gathered.shape != handle.shape || gathered.data.element_type() != expected_type {
            return Err(build_runtime_error(
                "integer exactness check: provider returned contradictory integer payload metadata",
            )
            .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
            .build());
        }
        let exact = integer_is_exact_f64;
        let values_exact = match &gathered.data {
            runmat_accelerate_api::HostIntegerDataOwned::I8(values) => values
                .iter()
                .all(|value| exact(&IntValue::I64(i64::from(*value)))),
            runmat_accelerate_api::HostIntegerDataOwned::I16(values) => values
                .iter()
                .all(|value| exact(&IntValue::I64(i64::from(*value)))),
            runmat_accelerate_api::HostIntegerDataOwned::I32(values) => values
                .iter()
                .all(|value| exact(&IntValue::I64(i64::from(*value)))),
            runmat_accelerate_api::HostIntegerDataOwned::I64(values) => {
                values.iter().all(|value| exact(&IntValue::I64(*value)))
            }
            runmat_accelerate_api::HostIntegerDataOwned::U8(values) => values
                .iter()
                .all(|value| exact(&IntValue::U64(u64::from(*value)))),
            runmat_accelerate_api::HostIntegerDataOwned::U16(values) => values
                .iter()
                .all(|value| exact(&IntValue::U64(u64::from(*value)))),
            runmat_accelerate_api::HostIntegerDataOwned::U32(values) => values
                .iter()
                .all(|value| exact(&IntValue::U64(u64::from(*value)))),
            runmat_accelerate_api::HostIntegerDataOwned::U64(values) => {
                values.iter().all(|value| exact(&IntValue::U64(*value)))
            }
        };
        return Ok(values_exact);
    }
    Ok(false)
}

/// Gate a native-integer RunMat extension before provider lookup or gathering,
/// then prove that its authoritative values can cross a binary64 computation
/// boundary without rounding.
pub async fn ensure_runmat_integer_f64_boundary(
    value: &Value,
    extension: &'static BuiltinExtensionDescriptor,
    builtin: &'static str,
    role: &str,
) -> BuiltinResult<()> {
    if !value_has_native_integer_class(value) {
        return Ok(());
    }
    crate::compatibility::ensure_builtin_extension_enabled(extension, builtin)?;
    if !native_integer_value_is_exact_f64_async(value).await? {
        return Err(build_runtime_error(format!(
            "{builtin}: integer {role} values must be exactly representable as double"
        ))
        .with_builtin(builtin)
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build());
    }
    Ok(())
}
