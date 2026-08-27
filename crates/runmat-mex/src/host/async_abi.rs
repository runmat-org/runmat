use std::ffi::{c_char, c_void, CStr};
use std::sync::Arc;
use std::time::Duration;

use super::{MexAsyncHostServices, MexAsyncOperation, MexAsyncResult, MexCallState, MexDiagnostic};
use crate::MxArray;

unsafe fn host<'a>(host: *mut c_void) -> Option<&'a MexCallState> {
    // SAFETY: the loader binds only pointer-stable `MexCallState` instances.
    unsafe { host.cast::<MexCallState>().as_ref() }
}

fn text(pointer: *const c_char) -> Option<String> {
    if pointer.is_null() {
        return None;
    }
    // SAFETY: private ABI callers provide a readable NUL-terminated string.
    Some(
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned(),
    )
}

fn register(host: &MexCallState, request: Arc<dyn MexAsyncResult>) -> Result<u64, MexDiagnostic> {
    let mut state = host.lock().map_err(|()| poisoned())?;
    let start = state.next_async_request.max(1);
    let mut id = start;
    loop {
        if let std::collections::btree_map::Entry::Vacant(entry) = state.async_requests.entry(id) {
            entry.insert(request);
            state.next_async_request = id.wrapping_add(1).max(1);
            return Ok(id);
        }
        id = id.wrapping_add(1).max(1);
        if id == start {
            return Err(MexDiagnostic {
                identifier: Some("RunMat:MEX:AsyncCapacity".into()),
                message: "the MEX asynchronous request table is full".into(),
            });
        }
    }
}

fn submit(
    host: &MexCallState,
    engine_context: u64,
    operation: MexAsyncOperation,
) -> Result<u64, MexDiagnostic> {
    let services = host
        .lock()
        .map_err(|()| poisoned())?
        .engine_contexts
        .get(&engine_context)
        .cloned()
        .ok_or_else(|| MexDiagnostic {
            identifier: Some("RunMat:MEX:InvalidEngineContext".into()),
            message: "the C++ RunMat engine context is no longer active".into(),
        })?;
    let request = services.submit(operation)?;
    register(host, request)
}

pub(crate) unsafe extern "C" fn create_engine_context(host_pointer: *mut c_void) -> u64 {
    let Some(host) = (unsafe { host(host_pointer) }) else {
        return 0;
    };
    let mut state = match host.lock() {
        Ok(state) => state,
        Err(()) => return fail(host, poisoned()),
    };
    if !state.services.is_available() {
        state.error = Some(MexDiagnostic {
            identifier: Some("RunMat:MEX:HostServiceUnavailable".into()),
            message: "C++ RunMat engine contexts require an active MEX invocation".into(),
        });
        return 0;
    }
    let start = state.next_engine_context.max(1);
    let mut id = start;
    while state.engine_contexts.contains_key(&id) {
        id = id.wrapping_add(1).max(1);
        if id == start {
            state.error = Some(MexDiagnostic {
                identifier: Some("RunMat:MEX:EngineContextCapacity".into()),
                message: "the C++ RunMat engine context table is full".into(),
            });
            return 0;
        }
    }
    let services = state.services.clone();
    state.engine_contexts.insert(id, services);
    state.next_engine_context = id.wrapping_add(1).max(1);
    id
}

pub(crate) unsafe extern "C" fn release_engine_context(host_pointer: *mut c_void, id: u64) {
    if let Some(host) = unsafe { host(host_pointer) } {
        if let Ok(mut state) = host.lock() {
            state.engine_contexts.remove(&id);
        }
    }
}

fn clone_array(host: &MexCallState, value: *const MxArray) -> Result<MxArray, MexDiagnostic> {
    host.lock()
        .map_err(|()| poisoned())?
        .mx
        .arena()
        .get(value)
        .cloned()
        .map_err(|error| invalid(error.to_string()))
}

fn request(host: &MexCallState, id: u64) -> Result<Arc<dyn MexAsyncResult>, MexDiagnostic> {
    host.lock()
        .map_err(|()| poisoned())?
        .async_requests
        .get(&id)
        .cloned()
        .ok_or_else(|| MexDiagnostic {
            identifier: Some("RunMat:MEX:InvalidAsyncRequest".into()),
            message: format!("MEX asynchronous request {id} is not active"),
        })
}

fn fail(host: &MexCallState, diagnostic: MexDiagnostic) -> u64 {
    if let Ok(mut state) = host.lock() {
        state.error = Some(diagnostic);
    }
    0
}

fn poisoned() -> MexDiagnostic {
    MexDiagnostic {
        identifier: Some("RunMat:MEX:HostState".into()),
        message: "the MEX host state is unavailable".into(),
    }
}

pub(crate) unsafe extern "C" fn submit_eval(
    host_pointer: *mut c_void,
    engine_context: u64,
    command: *const c_char,
    capture_stdout: i32,
    capture_stderr: i32,
) -> u64 {
    let Some(host) = (unsafe { host(host_pointer) }) else {
        return 0;
    };
    let Some(command) = text(command) else {
        return fail(host, invalid("evaluation command"));
    };
    match submit(
        host,
        engine_context,
        MexAsyncOperation::Eval {
            command,
            capture_stdout: capture_stdout != 0,
            capture_stderr: capture_stderr != 0,
        },
    ) {
        Ok(id) => id,
        Err(error) => fail(host, error),
    }
}

pub(crate) unsafe extern "C" fn submit_call(
    host_pointer: *mut c_void,
    engine_context: u64,
    function: *const c_char,
    requested_outputs: usize,
    argument_count: usize,
    arguments: *const *const MxArray,
    capture_stdout: i32,
    capture_stderr: i32,
) -> u64 {
    let Some(host) = (unsafe { host(host_pointer) }) else {
        return 0;
    };
    let Some(function) = text(function) else {
        return fail(host, invalid("function name"));
    };
    if argument_count != 0 && arguments.is_null() {
        return fail(host, invalid("argument array"));
    }
    let pointers = if argument_count == 0 {
        &[][..]
    } else {
        // SAFETY: the caller supplies exactly `argument_count` readable pointers.
        unsafe { std::slice::from_raw_parts(arguments, argument_count) }
    };
    let values = {
        let state = match host.lock() {
            Ok(state) => state,
            Err(()) => return fail(host, poisoned()),
        };
        match pointers
            .iter()
            .map(|pointer| state.mx.arena().get(*pointer).cloned())
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(values) => values,
            Err(error) => return fail(host, invalid(error.to_string())),
        }
    };
    match submit(
        host,
        engine_context,
        MexAsyncOperation::Call {
            function,
            arguments: values,
            requested_outputs,
            capture_stdout: capture_stdout != 0,
            capture_stderr: capture_stderr != 0,
        },
    ) {
        Ok(id) => id,
        Err(error) => fail(host, error),
    }
}

pub(crate) unsafe extern "C" fn submit_get_variable(
    host_pointer: *mut c_void,
    engine_context: u64,
    workspace: *const c_char,
    name: *const c_char,
) -> u64 {
    let Some(host) = (unsafe { host(host_pointer) }) else {
        return 0;
    };
    let (Some(workspace), Some(name)) = (text(workspace), text(name)) else {
        return fail(host, invalid("workspace variable name"));
    };
    match submit(
        host,
        engine_context,
        MexAsyncOperation::GetVariable { workspace, name },
    ) {
        Ok(id) => id,
        Err(error) => fail(host, error),
    }
}

pub(crate) unsafe extern "C" fn submit_put_variable(
    host_pointer: *mut c_void,
    engine_context: u64,
    workspace: *const c_char,
    name: *const c_char,
    value: *const MxArray,
) -> u64 {
    let Some(host) = (unsafe { host(host_pointer) }) else {
        return 0;
    };
    let (Some(workspace), Some(name)) = (text(workspace), text(name)) else {
        return fail(host, invalid("workspace variable name"));
    };
    let value = match clone_array(host, value) {
        Ok(value) => value,
        Err(error) => return fail(host, error),
    };
    match submit(
        host,
        engine_context,
        MexAsyncOperation::PutVariable {
            workspace,
            name,
            value,
        },
    ) {
        Ok(id) => id,
        Err(error) => fail(host, error),
    }
}

pub(crate) unsafe extern "C" fn submit_get_property(
    host_pointer: *mut c_void,
    engine_context: u64,
    object: *const MxArray,
    index: usize,
    name: *const c_char,
) -> u64 {
    let Some(host) = (unsafe { host(host_pointer) }) else {
        return 0;
    };
    let Some(name) = text(name) else {
        return fail(host, invalid("property name"));
    };
    let object = match clone_array(host, object) {
        Ok(object) => object,
        Err(error) => return fail(host, error),
    };
    match submit(
        host,
        engine_context,
        MexAsyncOperation::GetObjectProperty {
            object,
            index,
            name,
        },
    ) {
        Ok(id) => id,
        Err(error) => fail(host, error),
    }
}

pub(crate) unsafe extern "C" fn submit_set_property(
    host_pointer: *mut c_void,
    engine_context: u64,
    object: *mut MxArray,
    index: usize,
    name: *const c_char,
    value: *const MxArray,
) -> u64 {
    let Some(host) = (unsafe { host(host_pointer) }) else {
        return 0;
    };
    let Some(name) = text(name) else {
        return fail(host, invalid("property name"));
    };
    let object_pointer = object as usize;
    let object = match clone_array(host, object) {
        Ok(object) => object,
        Err(error) => return fail(host, error),
    };
    let value = match clone_array(host, value) {
        Ok(value) => value,
        Err(error) => return fail(host, error),
    };
    match submit(
        host,
        engine_context,
        MexAsyncOperation::SetObjectProperty {
            object,
            index,
            name,
            value,
        },
    ) {
        Ok(id) => {
            if let Ok(mut state) = host.lock() {
                state.async_property_targets.insert(id, object_pointer);
            }
            id
        }
        Err(error) => fail(host, error),
    }
}

pub(crate) unsafe extern "C" fn is_ready(host_pointer: *mut c_void, id: u64) -> i32 {
    let Some(host) = (unsafe { host(host_pointer) }) else {
        return 0;
    };
    request(host, id).is_ok_and(|request| request.is_ready()) as i32
}

pub(crate) unsafe extern "C" fn wait(
    host_pointer: *mut c_void,
    id: u64,
    timeout_millis: i64,
) -> i32 {
    let Some(host) = (unsafe { host(host_pointer) }) else {
        return 0;
    };
    let timeout = (timeout_millis >= 0).then(|| Duration::from_millis(timeout_millis as u64));
    request(host, id).is_ok_and(|request| request.wait(timeout)) as i32
}

pub(crate) unsafe extern "C" fn cancel(
    host_pointer: *mut c_void,
    id: u64,
    allow_interrupt: i32,
) -> i32 {
    let Some(host) = (unsafe { host(host_pointer) }) else {
        return 0;
    };
    request(host, id).is_ok_and(|request| request.cancel(allow_interrupt != 0)) as i32
}

pub(crate) unsafe extern "C" fn copy_result(
    host_pointer: *mut c_void,
    id: u64,
    output_capacity: usize,
    outputs: *mut *mut MxArray,
) -> i32 {
    let Some(host) = (unsafe { host(host_pointer) }) else {
        return 1;
    };
    let completion = match request(host, id) {
        Ok(request) => request.result(),
        Err(error) => {
            fail(host, error);
            return 1;
        }
    };
    let mut result = match completion.result {
        Ok(result) => result,
        Err(error) if error.identifier.as_deref() == Some("RunMat:MEX:Cancelled") => return 2,
        Err(_) => return 1,
    };
    let mut state = match host.lock() {
        Ok(state) => state,
        Err(()) => {
            fail(host, poisoned());
            return 1;
        }
    };
    if let Some(target) = state.async_property_targets.get(&id).copied() {
        if result.len() != 1 {
            state.error = Some(invalid("asynchronous property result"));
            return 1;
        }
        let updated = result.remove(0);
        match state.mx.arena_mut().get_mut(target as *mut MxArray) {
            Ok(object) => *object = updated,
            Err(error) => {
                state.error = Some(invalid(error.to_string()));
                return 1;
            }
        }
    }
    if result.len() > output_capacity || (!result.is_empty() && outputs.is_null()) {
        state.error = Some(invalid("asynchronous result output array"));
        return 1;
    }
    for (index, value) in result.into_iter().enumerate() {
        // SAFETY: capacity and nullability were validated above.
        unsafe { outputs.add(index).write(state.mx.allocate(value)) };
    }
    0
}

pub(crate) unsafe extern "C" fn copy_text(
    host_pointer: *mut c_void,
    id: u64,
    field: u32,
    output: *mut c_char,
    output_capacity: usize,
) -> usize {
    let Some(host) = (unsafe { host(host_pointer) }) else {
        return 0;
    };
    let completion = match request(host, id) {
        Ok(request) => request.result(),
        Err(error) => {
            fail(host, error);
            return 0;
        }
    };
    let value = match field {
        0 => completion.stdout.as_str(),
        1 => completion.stderr.as_str(),
        2 => completion
            .result
            .as_ref()
            .err()
            .and_then(|error| error.identifier.as_deref())
            .unwrap_or_default(),
        3 => completion
            .result
            .as_ref()
            .err()
            .map(|error| error.message.as_str())
            .unwrap_or_default(),
        _ => {
            fail(host, invalid("asynchronous text field"));
            return 0;
        }
    };
    let bytes = value.as_bytes();
    if output.is_null() || output_capacity == 0 {
        return bytes.len();
    }
    let copied = bytes.len().min(output_capacity);
    // SAFETY: the caller provides a writable buffer of `output_capacity` bytes.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), output.cast::<u8>(), copied) };
    bytes.len()
}

pub(crate) unsafe extern "C" fn release(host_pointer: *mut c_void, id: u64) {
    if let Some(host) = unsafe { host(host_pointer) } {
        if let Ok(mut state) = host.lock() {
            state.async_requests.remove(&id);
            state.async_property_targets.remove(&id);
        }
    }
}

fn invalid(name: impl Into<String>) -> MexDiagnostic {
    MexDiagnostic {
        identifier: Some("RunMat:MEX:AsyncArgument".into()),
        message: format!("invalid {}", name.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MexEngineCompletion, MxApiMode};

    struct CompletedRequest(MexEngineCompletion<Vec<MxArray>>);

    struct PendingRequest;

    impl MexAsyncResult for PendingRequest {
        fn cancel(&self, _: bool) -> bool {
            false
        }

        fn is_ready(&self) -> bool {
            false
        }

        fn wait(&self, _: Option<Duration>) -> bool {
            false
        }

        fn result(&self) -> MexEngineCompletion<Vec<MxArray>> {
            panic!("a pending request has no result")
        }
    }

    impl MexAsyncResult for CompletedRequest {
        fn cancel(&self, _: bool) -> bool {
            false
        }

        fn is_ready(&self) -> bool {
            true
        }

        fn wait(&self, _: Option<Duration>) -> bool {
            true
        }

        fn result(&self) -> MexEngineCompletion<Vec<MxArray>> {
            self.0.clone()
        }
    }

    fn insert(state: &MexCallState, completion: MexEngineCompletion<Vec<MxArray>>) -> u64 {
        register(state, Arc::new(CompletedRequest(completion))).expect("register completed request")
    }

    #[test]
    fn pending_requests_retain_module_lifecycle_until_completion_or_release() {
        let state = MexCallState::new(MxApiMode::InterleavedComplex);
        let request = register(&state, Arc::new(PendingRequest)).expect("register pending request");
        assert!(state.has_pending_async_requests().unwrap());

        // SAFETY: the request belongs to `state` and is released exactly once.
        unsafe {
            release(
                std::ptr::from_ref(&state).cast_mut().cast::<c_void>(),
                request,
            );
        }
        assert!(!state.has_pending_async_requests().unwrap());
    }

    unsafe fn copy_field(state: &MexCallState, request: u64, field: u32) -> String {
        let host = std::ptr::from_ref(state).cast_mut().cast::<c_void>();
        // SAFETY: `host` points to `state`, and the query does not write an output buffer.
        let length = unsafe { copy_text(host, request, field, std::ptr::null_mut(), 0) };
        let mut output = vec![0u8; length];
        // SAFETY: `output` has exactly the queried writable capacity.
        let required = unsafe {
            copy_text(
                host,
                request,
                field,
                output.as_mut_ptr().cast(),
                output.len(),
            )
        };
        assert_eq!(required, length);
        String::from_utf8(output).expect("completion text is UTF-8")
    }

    #[test]
    fn completion_text_preserves_streams_and_structured_diagnostics() {
        let state = MexCallState::new(MxApiMode::InterleavedComplex);
        let success = insert(
            &state,
            MexEngineCompletion {
                result: Ok(Vec::new()),
                stdout: "output: π".into(),
                stderr: "error: λ".into(),
            },
        );
        // SAFETY: the request belongs to `state` for the duration of these reads.
        unsafe {
            assert_eq!(copy_field(&state, success, 0), "output: π");
            assert_eq!(copy_field(&state, success, 1), "error: λ");
        }

        let failure = insert(
            &state,
            MexEngineCompletion::from_result(Err(MexDiagnostic {
                identifier: Some("Fixture:Failure".into()),
                message: "structured failure".into(),
            })),
        );
        let host = std::ptr::from_ref(&state).cast_mut().cast::<c_void>();
        // SAFETY: the request belongs to `state`; no output array is expected on failure.
        assert_eq!(
            unsafe { copy_result(host, failure, 0, std::ptr::null_mut()) },
            1
        );
        // SAFETY: the request belongs to `state` for the duration of these reads.
        unsafe {
            assert_eq!(copy_field(&state, failure, 2), "Fixture:Failure");
            assert_eq!(copy_field(&state, failure, 3), "structured failure");
        }
    }

    #[test]
    fn cancelled_completion_has_a_distinct_result_status() {
        let state = MexCallState::new(MxApiMode::InterleavedComplex);
        let request = insert(
            &state,
            MexEngineCompletion::from_result(Err(MexDiagnostic {
                identifier: Some("RunMat:MEX:Cancelled".into()),
                message: "cancelled".into(),
            })),
        );
        let host = std::ptr::from_ref(&state).cast_mut().cast::<c_void>();
        // SAFETY: the request belongs to `state`; no output array is expected.
        assert_eq!(
            unsafe { copy_result(host, request, 0, std::ptr::null_mut()) },
            2
        );
    }
}
