//! Shared integer classification and exact conversion boundaries.

use super::{
    build_runtime_error, handle_integer_type, handle_is_logical, handle_storage, BuiltinResult,
    ComplexTensor, GpuTensorStorage, IntValue, RuntimeError, Value,
};
use runmat_builtins::{BuiltinCatalogIdentity, BuiltinErrorDescriptor, BuiltinExtensionDescriptor};

mod prepared;
pub use prepared::{
    prepare_direct_integer_f64_boundary, IntegerBoundaryFailure, PreparedIntegerBoundary,
    PreparedIntegerPayload,
};

/// MATLAB stores complex integer arrays, but does not support arithmetic on
/// them. Arithmetic builtins use this before selecting floating or provider
/// execution paths so exact integer components are never coerced to `f64`.
pub fn is_typed_complex_integer(value: &Value) -> bool {
    matches!(value, Value::ComplexTensor(tensor) if tensor.integer_storage().is_some())
        || matches!(value, Value::GpuTensor(handle)
            if handle_integer_type(handle).is_some()
                && handle_storage(handle) == GpuTensorStorage::ComplexInterleaved)
}

/// Keep a class rejection attached to its selected catalog error owner.
pub fn reject_typed_complex_integer_for(
    value: &Value,
    builtin: BuiltinCatalogIdentity,
    error: &'static BuiltinErrorDescriptor,
) -> BuiltinResult<()> {
    if is_typed_complex_integer(value) {
        return Err(crate::runtime_descriptor_error_with_detail(
            builtin.name,
            error,
            "typed complex integer input is not supported",
        ));
    }
    Ok(())
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
        Value::Cell(cell) => cell.data().iter().any(value_contains_native_integer_class),
        Value::Struct(value) => value
            .fields
            .values()
            .any(value_contains_native_integer_class),
        Value::StructArray(value) => value.any_value(value_contains_native_integer_class),
        Value::ObjectArray(value) => value.data().iter().any(value_contains_native_integer_class),
        Value::Object(value) => value
            .properties()
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
        Value::Cell(cell) => cell.data().iter().any(value_contains_explicit_gpu),
        Value::Struct(value) => value.fields.values().any(value_contains_explicit_gpu),
        Value::StructArray(value) => value.any_value(value_contains_explicit_gpu),
        Value::ObjectArray(value) => value.data().iter().any(value_contains_explicit_gpu),
        Value::Object(value) => value.properties().values().any(value_contains_explicit_gpu),
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
    value.to_exact_f64().is_some()
}

pub fn native_integer_value_is_exact_f64(value: &Value) -> bool {
    fn storage_exact(storage: &runmat_value::IntegerStorage) -> bool {
        (0..storage.len()).all(|index| {
            storage
                .value_at(index)
                .is_some_and(|value| integer_is_exact_f64(&value))
        })
    }

    let exact = integer_is_exact_f64;
    match value {
        Value::Int(value) => exact(value),
        Value::Tensor(tensor) => tensor.integer_storage().is_none_or(storage_exact),
        Value::SparseTensor(tensor) => tensor.integer_storage().is_none_or(storage_exact),
        Value::ComplexTensor(tensor) => tensor
            .integer_storage()
            .is_none_or(|storage| storage_exact(storage.real()) && storage_exact(storage.imag())),
        Value::Cell(cell) => cell.data().iter().all(native_integer_value_is_exact_f64),
        Value::Struct(value) => value.fields.values().all(native_integer_value_is_exact_f64),
        Value::StructArray(value) => {
            !value.any_value(|child| !native_integer_value_is_exact_f64(child))
        }
        Value::Object(value) => value
            .properties()
            .values()
            .all(native_integer_value_is_exact_f64),
        Value::Closure(value) => value.captures.iter().all(native_integer_value_is_exact_f64),
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
    if matches!(
        value,
        Value::Int(_)
            | Value::Tensor(_)
            | Value::SparseTensor(_)
            | Value::ComplexTensor(_)
            | Value::GpuTensor(_)
    ) {
        return prepare_direct_integer_f64_boundary(value, None)
            .await
            .map(|prepared| prepared.is_exact())
            .map_err(IntegerBoundaryFailure::into_runtime_error);
    }
    // Aggregate traversal remains the legacy synchronous contract. The direct
    // prepared API deliberately does not pretend that nested resident values
    // or GC-backed payloads can be held safely across an await.
    if native_integer_value_is_exact_f64(value) {
        return Ok(true);
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
    ensure_integer_f64_exactness(value, builtin, role).await
}

async fn ensure_integer_f64_exactness(
    value: &Value,
    builtin: &'static str,
    role: &str,
) -> BuiltinResult<()> {
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

/// Admit a real-integer extension and reject unsupported complex storage before
/// exactness preparation can download a resident payload.
pub async fn ensure_runmat_real_integer_f64_boundary(
    value: &Value,
    extension: &'static BuiltinExtensionDescriptor,
    builtin: runmat_builtins::BuiltinCatalogIdentity,
    invalid_input: &'static runmat_builtins::BuiltinErrorDescriptor,
    role: &str,
) -> BuiltinResult<()> {
    if value_has_native_integer_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(extension, builtin.name)?;
    }
    reject_typed_complex_integer_for(value, builtin, invalid_input)?;
    if value_has_native_integer_class(value) {
        ensure_integer_f64_exactness(value, builtin.name, role).await?;
    }
    Ok(())
}
