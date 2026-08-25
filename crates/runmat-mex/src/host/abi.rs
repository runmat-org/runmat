use std::ffi::{c_char, c_void, CStr};

use crate::{MxApi, MxApiMode, MxArray, MxClassId};

pub const MEX_HOST_ABI_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MexDiagnostic {
    pub identifier: Option<String>,
    pub message: String,
}

#[derive(Debug)]
pub struct MexCallState {
    pub mx: MxApi,
    pub error: Option<MexDiagnostic>,
    pub warnings: Vec<MexDiagnostic>,
    pub console: String,
}

impl MexCallState {
    pub fn new(mode: MxApiMode) -> Self {
        Self {
            mx: MxApi::new(mode),
            error: None,
            warnings: Vec::new(),
            console: String::new(),
        }
    }

    pub fn host_api(&mut self) -> MexHostApiV1 {
        MexHostApiV1 {
            abi_version: MEX_HOST_ABI_VERSION,
            host: std::ptr::from_mut(self).cast(),
            create_numeric,
            create_double_scalar,
            create_logical,
            duplicate_array,
            destroy_array,
            class_id,
            number_of_dimensions,
            dimensions,
            number_of_elements,
            set_dimensions,
            data,
            imaginary_data,
            is_complex,
            set_error,
            emit_warning,
            write_console,
            has_error,
        }
    }

    fn fail(&mut self, message: impl Into<String>) {
        if self.error.is_none() {
            self.error = Some(MexDiagnostic {
                identifier: Some("RunMat:MEX:MatrixApi".into()),
                message: message.into(),
            });
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct MexHostApiV1 {
    pub abi_version: u32,
    pub host: *mut c_void,
    pub create_numeric:
        unsafe extern "C" fn(*mut c_void, usize, *const usize, i32, i32) -> *mut MxArray,
    pub create_double_scalar: unsafe extern "C" fn(*mut c_void, f64) -> *mut MxArray,
    pub create_logical: unsafe extern "C" fn(*mut c_void, usize, *const usize) -> *mut MxArray,
    pub duplicate_array: unsafe extern "C" fn(*mut c_void, *const MxArray) -> *mut MxArray,
    pub destroy_array: unsafe extern "C" fn(*mut c_void, *mut MxArray) -> i32,
    pub class_id: unsafe extern "C" fn(*mut c_void, *const MxArray) -> i32,
    pub number_of_dimensions: unsafe extern "C" fn(*mut c_void, *const MxArray) -> usize,
    pub dimensions: unsafe extern "C" fn(*mut c_void, *const MxArray) -> *const usize,
    pub number_of_elements: unsafe extern "C" fn(*mut c_void, *const MxArray) -> usize,
    pub set_dimensions: unsafe extern "C" fn(*mut c_void, *mut MxArray, usize, *const usize) -> i32,
    pub data: unsafe extern "C" fn(*mut c_void, *mut MxArray, i32) -> *mut c_void,
    pub imaginary_data: unsafe extern "C" fn(*mut c_void, *mut MxArray) -> *mut c_void,
    pub is_complex: unsafe extern "C" fn(*mut c_void, *const MxArray) -> i32,
    pub set_error: unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char),
    pub emit_warning: unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char),
    pub write_console: unsafe extern "C" fn(*mut c_void, *const c_char),
    pub has_error: unsafe extern "C" fn(*mut c_void) -> i32,
}

unsafe fn state<'a>(host: *mut c_void) -> Option<&'a mut MexCallState> {
    // SAFETY: every vtable is created by `MexCallState::host_api` and remains
    // scoped to the synchronous invocation that owns this state.
    unsafe { host.cast::<MexCallState>().as_mut() }
}

unsafe fn shape(ndim: usize, dims: *const usize) -> Result<Vec<usize>, String> {
    if ndim == 0 {
        return Ok(Vec::new());
    }
    if dims.is_null() {
        return Err("dimension pointer is null".into());
    }
    // SAFETY: the C caller promises `ndim` readable dimension elements for
    // the duration of this callback. We immediately copy them.
    Ok(unsafe { std::slice::from_raw_parts(dims, ndim) }.to_vec())
}

fn class_from_abi(value: i32) -> Option<MxClassId> {
    Some(match value {
        0 => MxClassId::Unknown,
        1 => MxClassId::Cell,
        2 => MxClassId::Struct,
        3 => MxClassId::Logical,
        4 => MxClassId::Char,
        5 => MxClassId::Void,
        6 => MxClassId::Double,
        7 => MxClassId::Single,
        8 => MxClassId::Int8,
        9 => MxClassId::Uint8,
        10 => MxClassId::Int16,
        11 => MxClassId::Uint16,
        12 => MxClassId::Int32,
        13 => MxClassId::Uint32,
        14 => MxClassId::Int64,
        15 => MxClassId::Uint64,
        16 => MxClassId::Function,
        17 => MxClassId::Opaque,
        18 => MxClassId::Object,
        _ => return None,
    })
}

unsafe extern "C" fn create_numeric(
    host: *mut c_void,
    ndim: usize,
    dims: *const usize,
    class_id: i32,
    complexity: i32,
) -> *mut MxArray {
    let Some(state) = (unsafe { state(host) }) else {
        return std::ptr::null_mut();
    };
    let result = unsafe { shape(ndim, dims) }.and_then(|shape| {
        let class_id = class_from_abi(class_id).ok_or_else(|| "invalid mxClassID".to_string())?;
        if !matches!(complexity, 0 | 1) {
            return Err("invalid mxComplexity".into());
        }
        state.mx.create_numeric(class_id, shape, complexity == 1)
    });
    match result {
        Ok(value) => value,
        Err(error) => {
            state.fail(error);
            std::ptr::null_mut()
        }
    }
}

unsafe extern "C" fn create_double_scalar(host: *mut c_void, value: f64) -> *mut MxArray {
    unsafe { state(host) }
        .map(|state| state.mx.create_double_scalar(value))
        .unwrap_or(std::ptr::null_mut())
}

unsafe extern "C" fn create_logical(
    host: *mut c_void,
    ndim: usize,
    dims: *const usize,
) -> *mut MxArray {
    let Some(state) = (unsafe { state(host) }) else {
        return std::ptr::null_mut();
    };
    match unsafe { shape(ndim, dims) }.and_then(|shape| state.mx.create_logical(shape)) {
        Ok(value) => value,
        Err(error) => {
            state.fail(error);
            std::ptr::null_mut()
        }
    }
}

unsafe extern "C" fn duplicate_array(host: *mut c_void, value: *const MxArray) -> *mut MxArray {
    let Some(state) = (unsafe { state(host) }) else {
        return std::ptr::null_mut();
    };
    match state.mx.duplicate(value) {
        Ok(value) => value,
        Err(error) => {
            state.fail(error.to_string());
            std::ptr::null_mut()
        }
    }
}

unsafe extern "C" fn destroy_array(host: *mut c_void, value: *mut MxArray) -> i32 {
    let Some(state) = (unsafe { state(host) }) else {
        return 1;
    };
    match state.mx.destroy(value) {
        Ok(()) => 0,
        Err(error) => {
            state.fail(error.to_string());
            1
        }
    }
}

unsafe extern "C" fn class_id(host: *mut c_void, value: *const MxArray) -> i32 {
    let Some(state) = (unsafe { state(host) }) else {
        return MxClassId::Unknown as i32;
    };
    state.mx.class_id(value).unwrap_or(MxClassId::Unknown) as i32
}

unsafe extern "C" fn number_of_dimensions(host: *mut c_void, value: *const MxArray) -> usize {
    unsafe { state(host) }
        .and_then(|state| state.mx.shape(value).ok().map(<[usize]>::len))
        .unwrap_or(0)
}

unsafe extern "C" fn dimensions(host: *mut c_void, value: *const MxArray) -> *const usize {
    unsafe { state(host) }
        .and_then(|state| state.mx.shape(value).ok().map(<[usize]>::as_ptr))
        .unwrap_or(std::ptr::null())
}

unsafe extern "C" fn number_of_elements(host: *mut c_void, value: *const MxArray) -> usize {
    unsafe { state(host) }
        .and_then(|state| state.mx.numel(value).ok())
        .unwrap_or(0)
}

unsafe extern "C" fn set_dimensions(
    host: *mut c_void,
    value: *mut MxArray,
    ndim: usize,
    dims: *const usize,
) -> i32 {
    let Some(state) = (unsafe { state(host) }) else {
        return 1;
    };
    match unsafe { shape(ndim, dims) }.and_then(|shape| state.mx.set_shape(value, shape)) {
        Ok(()) => 0,
        Err(error) => {
            state.fail(error);
            1
        }
    }
}

unsafe extern "C" fn data(
    host: *mut c_void,
    value: *mut MxArray,
    expected_class: i32,
) -> *mut c_void {
    let Some(state) = (unsafe { state(host) }) else {
        return std::ptr::null_mut();
    };
    let expected = if expected_class == MxClassId::Unknown as i32 {
        None
    } else {
        class_from_abi(expected_class).and_then(MxClassId::numeric_dtype)
    };
    match state.mx.data_pointer(value, expected) {
        Ok(value) => value,
        Err(error) => {
            state.fail(error);
            std::ptr::null_mut()
        }
    }
}

unsafe extern "C" fn imaginary_data(host: *mut c_void, value: *mut MxArray) -> *mut c_void {
    let Some(state) = (unsafe { state(host) }) else {
        return std::ptr::null_mut();
    };
    match state.mx.imaginary_pointer(value) {
        Ok(value) => value,
        Err(error) => {
            state.fail(error);
            std::ptr::null_mut()
        }
    }
}

unsafe extern "C" fn is_complex(host: *mut c_void, value: *const MxArray) -> i32 {
    unsafe { state(host) }
        .and_then(|state| state.mx.is_complex(value).ok())
        .map(i32::from)
        .unwrap_or(0)
}

fn c_string(value: *const c_char) -> Option<String> {
    if value.is_null() {
        return None;
    }
    // SAFETY: strings originate in the bound C shim and are valid through the
    // callback. Invalid UTF-8 is preserved lossily in diagnostics only.
    Some(
        unsafe { CStr::from_ptr(value) }
            .to_string_lossy()
            .into_owned(),
    )
}

unsafe extern "C" fn set_error(
    host: *mut c_void,
    identifier: *const c_char,
    message: *const c_char,
) {
    if let Some(state) = unsafe { state(host) } {
        state.error = Some(MexDiagnostic {
            identifier: c_string(identifier).filter(|value| !value.is_empty()),
            message: c_string(message).unwrap_or_else(|| "MEX function failed".into()),
        });
    }
}

unsafe extern "C" fn emit_warning(
    host: *mut c_void,
    identifier: *const c_char,
    message: *const c_char,
) {
    if let Some(state) = unsafe { state(host) } {
        state.warnings.push(MexDiagnostic {
            identifier: c_string(identifier).filter(|value| !value.is_empty()),
            message: c_string(message).unwrap_or_default(),
        });
    }
}

unsafe extern "C" fn write_console(host: *mut c_void, text: *const c_char) {
    if let (Some(state), Some(text)) = (unsafe { state(host) }, c_string(text)) {
        state.console.push_str(&text);
    }
}

unsafe extern "C" fn has_error(host: *mut c_void) -> i32 {
    unsafe { state(host) }
        .map(|state| i32::from(state.error.is_some()))
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vtable_uses_opaque_host_and_reports_bad_class_without_unwinding() {
        let mut state = MexCallState::new(MxApiMode::SeparateComplex);
        let api = state.host_api();
        let dims = [1usize, 1];
        let value = unsafe { (api.create_numeric)(api.host, dims.len(), dims.as_ptr(), 999, 0) };
        assert!(value.is_null());
        assert!(state.error.as_ref().unwrap().message.contains("mxClassID"));
    }
}
