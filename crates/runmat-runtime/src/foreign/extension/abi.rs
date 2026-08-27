use std::ffi::c_void;

use runmat_extension_abi::{
    RunMatBufferLease, RunMatBufferLeaseHandle, RunMatStatusCode, RunMatValueHandle,
    RunMatValueKind,
};

use super::registry::{BufferBorrowError, ExtensionHostContext, ValueAccessError};

unsafe fn host<'a>(context: *mut c_void) -> Option<&'a ExtensionHostContext> {
    // SAFETY: `ExtensionHost::host_vtable` supplies the stable address of its
    // C-facing context. Extensions may use it only while that host is alive.
    unsafe { context.cast::<ExtensionHostContext>().as_ref() }
}

fn value_status(result: Result<(), ValueAccessError>) -> RunMatStatusCode {
    match result {
        Ok(()) => RunMatStatusCode::Ok,
        Err(ValueAccessError::AffinityViolation) => RunMatStatusCode::AffinityViolation,
        Err(ValueAccessError::StaleHandle) => RunMatStatusCode::StaleHandle,
    }
}

pub(super) unsafe extern "C" fn retain_value(
    context: *mut c_void,
    value: RunMatValueHandle,
) -> RunMatStatusCode {
    let Some(host) = (unsafe { host(context) }) else {
        return RunMatStatusCode::StaleHandle;
    };
    value_status(host.retain(value))
}

pub(super) unsafe extern "C" fn release_value(
    context: *mut c_void,
    value: RunMatValueHandle,
) -> RunMatStatusCode {
    let Some(host) = (unsafe { host(context) }) else {
        return RunMatStatusCode::StaleHandle;
    };
    value_status(host.release_value(value))
}

pub(super) unsafe extern "C" fn value_kind(
    context: *mut c_void,
    value: RunMatValueHandle,
    kind: *mut RunMatValueKind,
) -> RunMatStatusCode {
    let Some(host) = (unsafe { host(context) }) else {
        return RunMatStatusCode::InvalidArgument;
    };
    let value_kind = match host.with_value(value, classify) {
        Ok(value_kind) => value_kind,
        Err(ValueAccessError::AffinityViolation) => {
            return RunMatStatusCode::AffinityViolation;
        }
        Err(ValueAccessError::StaleHandle) => return RunMatStatusCode::StaleHandle,
    };
    let Some(kind) = (unsafe { kind.as_mut() }) else {
        return RunMatStatusCode::InvalidArgument;
    };
    *kind = value_kind;
    RunMatStatusCode::Ok
}

pub(super) unsafe extern "C" fn borrow_buffer_lease(
    context: *mut c_void,
    value: RunMatValueHandle,
    lease: *mut RunMatBufferLease,
) -> RunMatStatusCode {
    let Some(host) = (unsafe { host(context) }) else {
        return RunMatStatusCode::InvalidArgument;
    };
    let Some(lease) = (unsafe { lease.as_mut() }) else {
        return RunMatStatusCode::InvalidArgument;
    };
    match host.borrow_buffer(value) {
        Ok(value) => {
            *lease = value;
            RunMatStatusCode::Ok
        }
        Err(BufferBorrowError::AffinityViolation) => RunMatStatusCode::AffinityViolation,
        Err(BufferBorrowError::StaleHandle) => RunMatStatusCode::StaleHandle,
        Err(BufferBorrowError::Unsupported) => RunMatStatusCode::Unsupported,
    }
}

pub(super) unsafe extern "C" fn release_buffer(
    context: *mut c_void,
    lease: RunMatBufferLeaseHandle,
) -> RunMatStatusCode {
    let Some(host) = (unsafe { host(context) }) else {
        return RunMatStatusCode::StaleHandle;
    };
    if host.release_buffer(lease) {
        RunMatStatusCode::Ok
    } else {
        RunMatStatusCode::StaleHandle
    }
}

fn classify(value: &runmat_value::Value) -> RunMatValueKind {
    use runmat_value::Value;
    match value {
        Value::Int(_) | Value::Num(_) | Value::Complex(_, _) | Value::Bool(_) => {
            RunMatValueKind::Scalar
        }
        Value::Tensor(_) | Value::ComplexTensor(_) => RunMatValueKind::Dense,
        Value::SparseTensor(_) => RunMatValueKind::Sparse,
        Value::LogicalArray(_) => RunMatValueKind::Logical,
        Value::CharArray(_) => RunMatValueKind::Character,
        Value::String(_) | Value::StringArray(_) => RunMatValueKind::String,
        Value::Cell(_) => RunMatValueKind::Cell,
        Value::Struct(_) => RunMatValueKind::Structure,
        Value::Object(_) | Value::ObjectArray(_) | Value::HandleObject(_) => {
            RunMatValueKind::Object
        }
        Value::FunctionHandle(_)
        | Value::ExternalFunctionHandle(_)
        | Value::MethodFunctionHandle(_)
        | Value::BoundFunctionHandle { .. }
        | Value::Closure(_) => RunMatValueKind::Callable,
        Value::Foreign(_) => RunMatValueKind::Foreign,
        _ => RunMatValueKind::Unknown,
    }
}
