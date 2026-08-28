use std::ffi::c_void;

use runmat_accelerate_api::{NativeDeviceApi, NativeDeviceComponent, NativeDeviceContextGuard};

use super::abi::{MexCallState, MexCallStateInner};
use crate::{MxArray, MxClassId};

fn state<'a>(host: *mut c_void) -> Option<std::sync::MutexGuard<'a, MexCallStateInner>> {
    // SAFETY: host tables are built from a live MexCallState and the loader
    // pins that state for the module lifetime.
    unsafe { host.cast::<MexCallState>().as_ref() }?.lock().ok()
}

fn fail<T>(state: &mut MexCallStateInner, result: Result<T, String>, fallback: T) -> T {
    match result {
        Ok(value) => value,
        Err(error) => {
            state.fail(error);
            fallback
        }
    }
}

fn shape(ndim: usize, dims: *const usize) -> Result<Vec<usize>, String> {
    if ndim == 0 {
        return Ok(Vec::new());
    }
    if dims.is_null() {
        return Err("dimension pointer is null".into());
    }
    // SAFETY: the native caller provides ndim readable values; they are
    // copied before the callback returns.
    Ok(unsafe { std::slice::from_raw_parts(dims, ndim) }.to_vec())
}

fn class_id_from_abi(value: i32) -> Result<MxClassId, String> {
    match value {
        3 => Ok(MxClassId::Logical),
        6 => Ok(MxClassId::Double),
        7 => Ok(MxClassId::Single),
        8 => Ok(MxClassId::Int8),
        9 => Ok(MxClassId::Uint8),
        10 => Ok(MxClassId::Int16),
        11 => Ok(MxClassId::Uint16),
        12 => Ok(MxClassId::Int32),
        13 => Ok(MxClassId::Uint32),
        14 => Ok(MxClassId::Int64),
        15 => Ok(MxClassId::Uint64),
        _ => Err(format!("mxClassID {value} cannot be stored in a GPU array")),
    }
}

pub(super) unsafe extern "C" fn context_enter(host: *mut c_void) -> *mut c_void {
    let result = runmat_accelerate_api::provider_for_native_device(NativeDeviceApi::Cuda)
        .ok_or_else(|| {
            "GPU MEX requires a registered CUDA provider; WGPU storage is not CUDA memory"
                .to_string()
        })
        .and_then(|provider| {
            provider
                .enter_native_device_context()
                .map_err(|error| error.to_string())
        });
    match result {
        Ok(guard) => Box::into_raw(Box::new(guard)).cast(),
        Err(error) => {
            if let Some(mut state) = state(host) {
                state.fail(error);
            }
            std::ptr::null_mut()
        }
    }
}

pub(super) unsafe extern "C" fn context_leave(host: *mut c_void, guard: *mut c_void) -> i32 {
    if guard.is_null() {
        return 0;
    }
    let synchronize = runmat_accelerate_api::provider_for_native_device(NativeDeviceApi::Cuda)
        .ok_or_else(|| "CUDA provider became unavailable during MEX invocation".to_string())
        .and_then(|provider| {
            provider
                .synchronize_native_device()
                .map_err(|error| error.to_string())
        });
    // SAFETY: context_enter returns exactly one boxed guard to the C support
    // layer, which calls context_leave once on every invocation exit path.
    let guard = unsafe { Box::from_raw(guard.cast::<NativeDeviceContextGuard>()) };
    let leave = guard.leave().map_err(|error| error.to_string());
    match (synchronize, leave) {
        (Ok(()), Ok(())) => 0,
        (synchronize, leave) => {
            let error = match (synchronize.err(), leave.err()) {
                (Some(synchronize), Some(leave)) => {
                    format!(
                        "CUDA synchronization failed: {synchronize}; context exit failed: {leave}"
                    )
                }
                (Some(synchronize), None) => {
                    format!("CUDA synchronization failed: {synchronize}")
                }
                (None, Some(leave)) => format!("CUDA context exit failed: {leave}"),
                (None, None) => unreachable!("matched at least one CUDA teardown failure"),
            };
            if let Some(mut state) = state(host) {
                state.fail(error);
            }
            1
        }
    }
}

pub(super) unsafe extern "C" fn create_from_array(
    host: *mut c_void,
    source: *const MxArray,
    independent_copy: i32,
) -> *mut MxArray {
    let Some(mut state) = state(host) else {
        return std::ptr::null_mut();
    };
    let result = state
        .mx
        .gpu_create_from_array(source, independent_copy != 0);
    fail(&mut state, result, std::ptr::null_mut())
}

pub(super) unsafe extern "C" fn create(
    host: *mut c_void,
    ndim: usize,
    dims: *const usize,
    class_id: i32,
    complex: i32,
    initialize: i32,
) -> *mut MxArray {
    let Some(mut state) = state(host) else {
        return std::ptr::null_mut();
    };
    let result = shape(ndim, dims).and_then(|shape| {
        let class_id = class_id_from_abi(class_id)?;
        state
            .mx
            .gpu_create(class_id, shape, complex != 0, initialize != 0)
    });
    fail(&mut state, result, std::ptr::null_mut())
}

pub(super) unsafe extern "C" fn to_host(host: *mut c_void, source: *const MxArray) -> *mut MxArray {
    let Some(mut state) = state(host) else {
        return std::ptr::null_mut();
    };
    let result = state.mx.gpu_to_host(source);
    fail(&mut state, result, std::ptr::null_mut())
}

pub(super) unsafe extern "C" fn data(
    host: *mut c_void,
    source: *mut MxArray,
    writable: i32,
) -> *mut c_void {
    let Some(mut state) = state(host) else {
        return std::ptr::null_mut();
    };
    let result = state
        .mx
        .gpu_data(source, writable != 0)
        .and_then(|address| {
            usize::try_from(address)
                .map(|address| address as *mut c_void)
                .map_err(|_| "native device address exceeds the host pointer width".to_string())
        });
    fail(&mut state, result, std::ptr::null_mut())
}

pub(super) unsafe extern "C" fn class_id(host: *mut c_void, source: *const MxArray) -> i32 {
    let Some(mut state) = state(host) else {
        return 0;
    };
    let result = state.mx.gpu_class_id(source).map(|class| class as i32);
    fail(&mut state, result, 0)
}

pub(super) unsafe extern "C" fn is_array(host: *mut c_void, source: *const MxArray) -> i32 {
    state(host)
        .map(|state| i32::from(state.mx.is_gpu_array(source)))
        .unwrap_or(0)
}

pub(super) unsafe extern "C" fn is_same(
    host: *mut c_void,
    left: *const MxArray,
    right: *const MxArray,
) -> i32 {
    let Some(mut state) = state(host) else {
        return 0;
    };
    let result = state.mx.gpu_is_same(left, right).map(i32::from);
    fail(&mut state, result, 0)
}

pub(super) unsafe extern "C" fn copy_component(
    host: *mut c_void,
    source: *const MxArray,
    component: i32,
) -> *mut MxArray {
    let Some(mut state) = state(host) else {
        return std::ptr::null_mut();
    };
    let component = match component {
        0 => Ok(NativeDeviceComponent::Real),
        1 => Ok(NativeDeviceComponent::Imaginary),
        _ => Err("invalid GPU component selector".to_string()),
    };
    let result = component.and_then(|component| state.mx.gpu_copy_component(source, component));
    fail(&mut state, result, std::ptr::null_mut())
}

pub(super) unsafe extern "C" fn create_complex(
    host: *mut c_void,
    real: *const MxArray,
    imaginary: *const MxArray,
) -> *mut MxArray {
    let Some(mut state) = state(host) else {
        return std::ptr::null_mut();
    };
    let result = state.mx.gpu_create_complex(real, imaginary);
    fail(&mut state, result, std::ptr::null_mut())
}
