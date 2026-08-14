use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};

use runmat_runtime::native::*;

use crate::JitError;

use super::state::HostState;

pub(super) fn table(state: &mut HostState) -> NativeHostVTable {
    NativeHostVTable {
        abi_version: NATIVE_ABI_VERSION.encoded(),
        struct_size: std::mem::size_of::<NativeHostVTable>() as u32,
        context: (state as *mut HostState).cast(),
        retain_value: Some(retain),
        release_value: Some(release),
        slow_call: Some(slow_call),
        poll_safepoint: Some(poll_safepoint),
        source_lookup: Some(source_lookup),
        execute_site: Some(execute_site),
    }
}

unsafe extern "C" fn retain(context: *mut c_void, value: NativeValueRef) -> NativeHostStatus {
    boundary(context, |state| state.arena.retain(value))
}

unsafe extern "C" fn release(context: *mut c_void, value: NativeValueRef) -> NativeHostStatus {
    boundary(context, |state| state.arena.release(value))
}

unsafe extern "C" fn slow_call(
    context: *mut c_void,
    _: *mut NativeCall,
    _: *mut NativeExit,
) -> NativeHostStatus {
    boundary(context, |_| {
        Err(JitError::UnsupportedSite(
            "standalone slow-call callback is not yet wired".into(),
        ))
    })
}

unsafe extern "C" fn poll_safepoint(
    context: *mut c_void,
    safepoint: *const NativeSafepoint,
    exit: *mut NativeExit,
) -> NativeHostStatus {
    if safepoint.is_null() || exit.is_null() {
        return NativeHostStatus::INVALID_ARGUMENT;
    }
    boundary(context, |state| {
        // SAFETY: the pointers were checked and are borrowed for this callback.
        unsafe { (*safepoint).validate() }.map_err(|error| JitError::Host(error.to_string()))?;
        if state
            .runtime
            .cancellation()
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            // SAFETY: exit is a checked writable callback output.
            unsafe {
                *exit = NativeExit::cancelled(NativeCancellation {
                    reason: NativeCancellationReason::REQUESTED,
                    reserved: 0,
                    generation: 1,
                })
            };
        }
        Ok(())
    })
}

unsafe extern "C" fn source_lookup(
    context: *mut c_void,
    source: u32,
    output: *mut NativeSourceMapEntry,
) -> NativeHostStatus {
    if output.is_null() {
        return NativeHostStatus::INVALID_ARGUMENT;
    }
    boundary(context, |_| {
        // Source strings remain owned by Core's portable source map. The first
        // host cohort provides identity; Core integration adds borrowed text.
        // SAFETY: output is a checked writable callback output.
        unsafe {
            *output = NativeSourceMapEntry {
                source,
                ..NativeSourceMapEntry::default()
            }
        };
        Ok(())
    })
}

unsafe extern "C" fn execute_site(
    context: *mut c_void,
    call: *mut NativeCall,
    request: *const NativeSiteRequest,
    outcome: *mut NativeSiteOutcome,
    exit: *mut NativeExit,
) -> NativeHostStatus {
    if call.is_null() || request.is_null() || outcome.is_null() || exit.is_null() {
        return NativeHostStatus::INVALID_ARGUMENT;
    }
    boundary(context, |state| {
        // SAFETY: callback inputs were checked and remain borrowed for this call.
        let request = unsafe { *request };
        request
            .validate()
            .map_err(|error| JitError::Host(error.to_string()))?;
        let result = super::site::execute(
            state,
            // SAFETY: checked writable call transaction.
            unsafe { &mut *call },
            request,
            // SAFETY: checked writable exit output.
            unsafe { &mut *exit },
        );
        match result {
            Ok(decision) => {
                // SAFETY: checked writable semantic outcome output.
                unsafe { *outcome = decision };
                Ok(())
            }
            Err(JitError::Runtime(error)) => {
                let exception = state
                    .arena
                    .insert(runmat_value::Value::String(error.message().to_string()));
                state.last_error = Some(*error);
                // SAFETY: callback outputs are checked and writable.
                unsafe {
                    *exit = NativeExit::exception(NativeException {
                        handle: exception.handle,
                        generation: exception.generation,
                        source: state.current_source,
                    });
                    *outcome = NativeSiteOutcome::exit();
                }
                Ok(())
            }
            Err(error) => Err(error),
        }
    })
}

fn boundary(
    context: *mut c_void,
    operation: impl FnOnce(&mut HostState) -> Result<(), JitError>,
) -> NativeHostStatus {
    if context.is_null() {
        return NativeHostStatus::INVALID_ARGUMENT;
    }
    // SAFETY: every vtable is built with its invocation-owned HostState and is
    // retained until the generated entrypoint returns synchronously.
    let state = unsafe { &mut *context.cast::<HostState>() };
    match catch_unwind(AssertUnwindSafe(|| operation(state))) {
        Ok(Ok(())) => NativeHostStatus::OK,
        Ok(Err(JitError::StaleValue)) => NativeHostStatus::STALE_VALUE,
        Ok(Err(error)) => {
            state.host_failure = Some(error);
            NativeHostStatus::HOST_FAILURE
        }
        Err(_) => {
            state.host_failure = Some(JitError::Host(
                "panic contained at native host boundary".into(),
            ));
            NativeHostStatus::HOST_FAILURE
        }
    }
}
