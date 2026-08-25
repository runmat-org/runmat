use std::collections::BTreeSet;
use std::ffi::{c_char, c_void, CStr, CString};
use std::rc::Rc;

use crate::{value_from_mx, value_to_mx, MexHostServices, MxApi, MxApiMode, MxArray, MxClassId};

pub const MEX_HOST_ABI_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MexDiagnostic {
    pub identifier: Option<String>,
    pub message: String,
}

pub struct MexCallState {
    pub mx: MxApi,
    pub error: Option<MexDiagnostic>,
    pub warnings: Vec<MexDiagnostic>,
    pub console: String,
    field_name_cache: Vec<CString>,
    services: Rc<dyn MexHostServices>,
    global_arrays: BTreeSet<usize>,
}

impl MexCallState {
    pub fn new(mode: MxApiMode) -> Self {
        Self {
            mx: MxApi::new(mode),
            error: None,
            warnings: Vec::new(),
            console: String::new(),
            field_name_cache: Vec::new(),
            services: Rc::new(super::UnavailableMexHostServices),
            global_arrays: BTreeSet::new(),
        }
    }

    pub fn with_services(mode: MxApiMode, services: Rc<dyn MexHostServices>) -> Self {
        Self {
            services,
            ..Self::new(mode)
        }
    }

    pub fn set_services(&mut self, services: Rc<dyn MexHostServices>) {
        self.services = services;
    }

    pub fn host_api(&mut self) -> MexHostApiV1 {
        MexHostApiV1 {
            abi_version: MEX_HOST_ABI_VERSION,
            host: std::ptr::from_mut(self).cast(),
            create_numeric,
            create_double_scalar,
            create_logical,
            create_char,
            create_cell,
            create_struct,
            create_sparse,
            duplicate_array,
            destroy_array,
            class_id,
            number_of_dimensions,
            dimensions,
            number_of_elements,
            set_dimensions,
            data,
            imaginary_data,
            replace_data,
            replace_imaginary_data,
            is_complex,
            is_sparse,
            get_cell,
            set_cell,
            number_of_fields,
            field_name,
            field_number,
            add_field,
            remove_field,
            get_field,
            set_field,
            sparse_row_indices,
            sparse_column_pointers,
            sparse_nzmax,
            set_sparse_nzmax,
            replace_sparse_row_indices,
            replace_sparse_column_pointers,
            make_array_persistent,
            eval,
            call,
            get_variable,
            put_variable,
            is_global,
            take_error,
            set_error,
            emit_warning,
            write_console,
            has_error,
        }
    }

    pub fn begin_call(&mut self) {
        self.error = None;
        self.warnings.clear();
        self.console.clear();
        self.field_name_cache.clear();
    }

    pub fn finish_call(&mut self) {
        self.mx.finish_call();
        self.field_name_cache.clear();
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
    pub create_char: unsafe extern "C" fn(*mut c_void, usize, *const usize) -> *mut MxArray,
    pub create_cell: unsafe extern "C" fn(*mut c_void, usize, *const usize) -> *mut MxArray,
    pub create_struct: unsafe extern "C" fn(
        *mut c_void,
        usize,
        *const usize,
        i32,
        *const *const c_char,
    ) -> *mut MxArray,
    pub create_sparse: unsafe extern "C" fn(*mut c_void, usize, usize, usize, i32) -> *mut MxArray,
    pub duplicate_array: unsafe extern "C" fn(*mut c_void, *const MxArray) -> *mut MxArray,
    pub destroy_array: unsafe extern "C" fn(*mut c_void, *mut MxArray) -> i32,
    pub class_id: unsafe extern "C" fn(*mut c_void, *const MxArray) -> i32,
    pub number_of_dimensions: unsafe extern "C" fn(*mut c_void, *const MxArray) -> usize,
    pub dimensions: unsafe extern "C" fn(*mut c_void, *const MxArray) -> *const usize,
    pub number_of_elements: unsafe extern "C" fn(*mut c_void, *const MxArray) -> usize,
    pub set_dimensions: unsafe extern "C" fn(*mut c_void, *mut MxArray, usize, *const usize) -> i32,
    pub data: unsafe extern "C" fn(*mut c_void, *mut MxArray, i32) -> *mut c_void,
    pub imaginary_data: unsafe extern "C" fn(*mut c_void, *mut MxArray) -> *mut c_void,
    pub replace_data: unsafe extern "C" fn(*mut c_void, *mut MxArray, *const c_void) -> i32,
    pub replace_imaginary_data:
        unsafe extern "C" fn(*mut c_void, *mut MxArray, *const c_void) -> i32,
    pub is_complex: unsafe extern "C" fn(*mut c_void, *const MxArray) -> i32,
    pub is_sparse: unsafe extern "C" fn(*mut c_void, *const MxArray) -> i32,
    pub get_cell: unsafe extern "C" fn(*mut c_void, *const MxArray, usize) -> *mut MxArray,
    pub set_cell: unsafe extern "C" fn(*mut c_void, *mut MxArray, usize, *mut MxArray) -> i32,
    pub number_of_fields: unsafe extern "C" fn(*mut c_void, *const MxArray) -> i32,
    pub field_name: unsafe extern "C" fn(*mut c_void, *const MxArray, i32) -> *const c_char,
    pub field_number: unsafe extern "C" fn(*mut c_void, *const MxArray, *const c_char) -> i32,
    pub add_field: unsafe extern "C" fn(*mut c_void, *mut MxArray, *const c_char) -> i32,
    pub remove_field: unsafe extern "C" fn(*mut c_void, *mut MxArray, i32) -> i32,
    pub get_field: unsafe extern "C" fn(*mut c_void, *const MxArray, usize, i32) -> *mut MxArray,
    pub set_field: unsafe extern "C" fn(*mut c_void, *mut MxArray, usize, i32, *mut MxArray) -> i32,
    pub sparse_row_indices: unsafe extern "C" fn(*mut c_void, *mut MxArray) -> *mut usize,
    pub sparse_column_pointers: unsafe extern "C" fn(*mut c_void, *mut MxArray) -> *mut usize,
    pub sparse_nzmax: unsafe extern "C" fn(*mut c_void, *mut MxArray) -> usize,
    pub set_sparse_nzmax: unsafe extern "C" fn(*mut c_void, *mut MxArray, usize) -> i32,
    pub replace_sparse_row_indices:
        unsafe extern "C" fn(*mut c_void, *mut MxArray, *const usize) -> i32,
    pub replace_sparse_column_pointers:
        unsafe extern "C" fn(*mut c_void, *mut MxArray, *const usize) -> i32,
    pub make_array_persistent: unsafe extern "C" fn(*mut c_void, *mut MxArray) -> i32,
    pub eval: unsafe extern "C" fn(*mut c_void, *const c_char) -> i32,
    pub call: unsafe extern "C" fn(
        *mut c_void,
        *const c_char,
        i32,
        *mut *mut MxArray,
        i32,
        *const *const MxArray,
    ) -> i32,
    pub get_variable:
        unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char) -> *mut MxArray,
    pub put_variable:
        unsafe extern "C" fn(*mut c_void, *const c_char, *const c_char, *const MxArray) -> i32,
    pub is_global: unsafe extern "C" fn(*mut c_void, *const MxArray) -> i32,
    pub take_error: unsafe extern "C" fn(*mut c_void) -> *mut MxArray,
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

unsafe extern "C" fn create_char(
    host: *mut c_void,
    ndim: usize,
    dims: *const usize,
) -> *mut MxArray {
    let Some(state) = (unsafe { state(host) }) else {
        return std::ptr::null_mut();
    };
    match unsafe { shape(ndim, dims) }.and_then(|shape| state.mx.create_char(shape)) {
        Ok(value) => value,
        Err(error) => {
            state.fail(error);
            std::ptr::null_mut()
        }
    }
}

unsafe extern "C" fn create_cell(
    host: *mut c_void,
    ndim: usize,
    dims: *const usize,
) -> *mut MxArray {
    let Some(state) = (unsafe { state(host) }) else {
        return std::ptr::null_mut();
    };
    match unsafe { shape(ndim, dims) }.and_then(|shape| state.mx.create_cell(shape)) {
        Ok(value) => value,
        Err(error) => {
            state.fail(error);
            std::ptr::null_mut()
        }
    }
}

unsafe extern "C" fn create_struct(
    host: *mut c_void,
    ndim: usize,
    dims: *const usize,
    field_count: i32,
    field_names: *const *const c_char,
) -> *mut MxArray {
    let Some(state) = (unsafe { state(host) }) else {
        return std::ptr::null_mut();
    };
    let result = unsafe { shape(ndim, dims) }.and_then(|shape| {
        let field_count = usize::try_from(field_count)
            .map_err(|_| "struct field count must be non-negative".to_string())?;
        if field_count > 0 && field_names.is_null() {
            return Err("struct field-name pointer is null".into());
        }
        let pointers = if field_count == 0 {
            &[][..]
        } else {
            // SAFETY: the gateway promises one readable pointer per field.
            unsafe { std::slice::from_raw_parts(field_names, field_count) }
        };
        let fields = pointers
            .iter()
            .map(|pointer| {
                c_string(*pointer).ok_or_else(|| "struct field name is null".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        state.mx.create_struct(shape, fields)
    });
    match result {
        Ok(value) => value,
        Err(error) => {
            state.fail(error);
            std::ptr::null_mut()
        }
    }
}

unsafe extern "C" fn create_sparse(
    host: *mut c_void,
    rows: usize,
    cols: usize,
    nzmax: usize,
    logical: i32,
) -> *mut MxArray {
    let Some(state) = (unsafe { state(host) }) else {
        return std::ptr::null_mut();
    };
    match state.mx.create_sparse(rows, cols, nzmax, logical != 0) {
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
        class_from_abi(expected_class)
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

unsafe extern "C" fn replace_data(
    host: *mut c_void,
    value: *mut MxArray,
    source: *const c_void,
) -> i32 {
    replace_data_component(host, value, source, false)
}

unsafe extern "C" fn replace_imaginary_data(
    host: *mut c_void,
    value: *mut MxArray,
    source: *const c_void,
) -> i32 {
    replace_data_component(host, value, source, true)
}

fn replace_data_component(
    host: *mut c_void,
    value: *mut MxArray,
    source: *const c_void,
    imaginary: bool,
) -> i32 {
    let Some(state) = (unsafe { state(host) }) else {
        return 1;
    };
    // SAFETY: the C shim transfers a buffer matching the array's reported size.
    match unsafe { state.mx.replace_data(value, source, imaginary) } {
        Ok(()) => 0,
        Err(error) => {
            state.fail(error);
            1
        }
    }
}

unsafe extern "C" fn is_complex(host: *mut c_void, value: *const MxArray) -> i32 {
    unsafe { state(host) }
        .and_then(|state| state.mx.is_complex(value).ok())
        .map(i32::from)
        .unwrap_or(0)
}

unsafe extern "C" fn is_sparse(host: *mut c_void, value: *const MxArray) -> i32 {
    unsafe { state(host) }
        .and_then(|state| state.mx.is_sparse(value).ok())
        .map(i32::from)
        .unwrap_or(0)
}

unsafe extern "C" fn get_cell(
    host: *mut c_void,
    value: *const MxArray,
    index: usize,
) -> *mut MxArray {
    let Some(state) = (unsafe { state(host) }) else {
        return std::ptr::null_mut();
    };
    match state.mx.get_cell(value, index) {
        Ok(value) => value,
        Err(error) => {
            state.fail(error);
            std::ptr::null_mut()
        }
    }
}

unsafe extern "C" fn set_cell(
    host: *mut c_void,
    value: *mut MxArray,
    index: usize,
    child: *mut MxArray,
) -> i32 {
    let Some(state) = (unsafe { state(host) }) else {
        return 1;
    };
    match state.mx.set_cell(value, index, child) {
        Ok(()) => 0,
        Err(error) => {
            state.fail(error);
            1
        }
    }
}

unsafe extern "C" fn number_of_fields(host: *mut c_void, value: *const MxArray) -> i32 {
    unsafe { state(host) }
        .and_then(|state| state.mx.field_names(value).ok())
        .and_then(|fields| i32::try_from(fields.len()).ok())
        .unwrap_or(0)
}

unsafe extern "C" fn field_name(
    host: *mut c_void,
    value: *const MxArray,
    field: i32,
) -> *const c_char {
    let Some(state) = (unsafe { state(host) }) else {
        return std::ptr::null();
    };
    let Ok(field) = usize::try_from(field) else {
        return std::ptr::null();
    };
    let name = state
        .mx
        .field_names(value)
        .ok()
        .and_then(|fields| fields.get(field))
        .cloned();
    let Some(name) = name.and_then(|name| CString::new(name).ok()) else {
        return std::ptr::null();
    };
    state.field_name_cache.push(name);
    state
        .field_name_cache
        .last()
        .map(|name| name.as_ptr())
        .unwrap_or(std::ptr::null())
}

unsafe extern "C" fn field_number(
    host: *mut c_void,
    value: *const MxArray,
    name: *const c_char,
) -> i32 {
    let Some(name) = c_string(name) else {
        return -1;
    };
    unsafe { state(host) }
        .and_then(|state| state.mx.field_names(value).ok())
        .and_then(|fields| fields.iter().position(|field| field == &name))
        .and_then(|index| i32::try_from(index).ok())
        .unwrap_or(-1)
}

unsafe extern "C" fn add_field(host: *mut c_void, value: *mut MxArray, name: *const c_char) -> i32 {
    let Some(state) = (unsafe { state(host) }) else {
        return -1;
    };
    let Some(name) = c_string(name) else {
        state.fail("struct field name is null");
        return -1;
    };
    match state.mx.add_field(value, name) {
        Ok(index) => i32::try_from(index).unwrap_or(-1),
        Err(error) => {
            state.fail(error);
            -1
        }
    }
}

unsafe extern "C" fn remove_field(host: *mut c_void, value: *mut MxArray, field: i32) -> i32 {
    let Some(state) = (unsafe { state(host) }) else {
        return 1;
    };
    let Ok(field) = usize::try_from(field) else {
        state.fail("struct field number must be non-negative");
        return 1;
    };
    match state.mx.remove_field(value, field) {
        Ok(()) => 0,
        Err(error) => {
            state.fail(error);
            1
        }
    }
}

unsafe extern "C" fn get_field(
    host: *mut c_void,
    value: *const MxArray,
    element: usize,
    field: i32,
) -> *mut MxArray {
    let Some(state) = (unsafe { state(host) }) else {
        return std::ptr::null_mut();
    };
    let Ok(field) = usize::try_from(field) else {
        state.fail("struct field number must be non-negative");
        return std::ptr::null_mut();
    };
    match state.mx.get_field(value, element, field) {
        Ok(value) => value,
        Err(error) => {
            state.fail(error);
            std::ptr::null_mut()
        }
    }
}

unsafe extern "C" fn set_field(
    host: *mut c_void,
    value: *mut MxArray,
    element: usize,
    field: i32,
    child: *mut MxArray,
) -> i32 {
    let Some(state) = (unsafe { state(host) }) else {
        return 1;
    };
    let Ok(field) = usize::try_from(field) else {
        state.fail("struct field number must be non-negative");
        return 1;
    };
    match state.mx.set_field(value, element, field, child) {
        Ok(()) => 0,
        Err(error) => {
            state.fail(error);
            1
        }
    }
}

unsafe extern "C" fn sparse_row_indices(host: *mut c_void, value: *mut MxArray) -> *mut usize {
    let Some(state) = (unsafe { state(host) }) else {
        return std::ptr::null_mut();
    };
    match state.mx.sparse_indices(value) {
        Ok((rows, _, _)) => rows,
        Err(error) => {
            state.fail(error);
            std::ptr::null_mut()
        }
    }
}

unsafe extern "C" fn sparse_column_pointers(host: *mut c_void, value: *mut MxArray) -> *mut usize {
    let Some(state) = (unsafe { state(host) }) else {
        return std::ptr::null_mut();
    };
    match state.mx.sparse_indices(value) {
        Ok((_, columns, _)) => columns,
        Err(error) => {
            state.fail(error);
            std::ptr::null_mut()
        }
    }
}

unsafe extern "C" fn sparse_nzmax(host: *mut c_void, value: *mut MxArray) -> usize {
    unsafe { state(host) }
        .and_then(|state| state.mx.sparse_indices(value).ok())
        .map(|(_, _, nzmax)| nzmax)
        .unwrap_or(0)
}

unsafe extern "C" fn set_sparse_nzmax(host: *mut c_void, value: *mut MxArray, nzmax: usize) -> i32 {
    let Some(state) = (unsafe { state(host) }) else {
        return 1;
    };
    match state.mx.set_nzmax(value, nzmax) {
        Ok(()) => 0,
        Err(error) => {
            state.fail(error);
            1
        }
    }
}

unsafe extern "C" fn replace_sparse_row_indices(
    host: *mut c_void,
    value: *mut MxArray,
    source: *const usize,
) -> i32 {
    replace_sparse_indices(host, value, source, false)
}

unsafe extern "C" fn replace_sparse_column_pointers(
    host: *mut c_void,
    value: *mut MxArray,
    source: *const usize,
) -> i32 {
    replace_sparse_indices(host, value, source, true)
}

fn replace_sparse_indices(
    host: *mut c_void,
    value: *mut MxArray,
    source: *const usize,
    columns: bool,
) -> i32 {
    let Some(state) = (unsafe { state(host) }) else {
        return 1;
    };
    // SAFETY: the shim transfers the fixed row-capacity or column-pointer buffer.
    match unsafe { state.mx.replace_sparse_indices(value, source, columns) } {
        Ok(()) => 0,
        Err(error) => {
            state.fail(error);
            1
        }
    }
}

unsafe extern "C" fn make_array_persistent(host: *mut c_void, value: *mut MxArray) -> i32 {
    let Some(state) = (unsafe { state(host) }) else {
        return 1;
    };
    match state.mx.make_persistent(value) {
        Ok(()) => 0,
        Err(error) => {
            state.fail(error);
            1
        }
    }
}

unsafe extern "C" fn eval(host: *mut c_void, command: *const c_char) -> i32 {
    let Some(state) = (unsafe { state(host) }) else {
        return 1;
    };
    let Some(command) = c_string(command) else {
        state.fail("evaluation command is null");
        return 1;
    };
    match state.services.eval(&command) {
        Ok(()) => 0,
        Err(error) => {
            state.error = Some(error);
            1
        }
    }
}

unsafe extern "C" fn call(
    host: *mut c_void,
    function: *const c_char,
    nlhs: i32,
    plhs: *mut *mut MxArray,
    nrhs: i32,
    prhs: *const *const MxArray,
) -> i32 {
    let Some(state) = (unsafe { state(host) }) else {
        return 1;
    };
    let result = (|| {
        let function = c_string(function).ok_or_else(|| MexDiagnostic {
            identifier: Some("RunMat:MEX:InvalidFunction".into()),
            message: "callback function name is null".into(),
        })?;
        let nlhs = usize::try_from(nlhs).map_err(|_| invalid_count("output"))?;
        let nrhs = usize::try_from(nrhs).map_err(|_| invalid_count("input"))?;
        if nlhs > 0 && plhs.is_null() {
            return Err(invalid_pointer("output"));
        }
        if nrhs > 0 && prhs.is_null() {
            return Err(invalid_pointer("input"));
        }
        let inputs = if nrhs == 0 {
            &[][..]
        } else {
            // SAFETY: the shim supplies `nrhs` readable array pointers.
            unsafe { std::slice::from_raw_parts(prhs, nrhs) }
        };
        let arguments = inputs
            .iter()
            .map(|pointer| {
                state
                    .mx
                    .arena()
                    .get(*pointer)
                    .map_err(|error| MexDiagnostic {
                        identifier: Some("RunMat:MEX:InvalidArray".into()),
                        message: error.to_string(),
                    })
                    .and_then(|value| {
                        value_from_mx(value).map_err(|error| MexDiagnostic {
                            identifier: Some("RunMat:MEX:Conversion".into()),
                            message: error.message,
                        })
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = state.services.call(&function, arguments, nlhs)?;
        if outputs.len() != nlhs {
            return Err(MexDiagnostic {
                identifier: Some("RunMat:MEX:OutputCount".into()),
                message: format!(
                    "callback '{function}' returned {} outputs when {nlhs} were requested",
                    outputs.len()
                ),
            });
        }
        let output_pointers = if nlhs == 0 {
            &mut [][..]
        } else {
            // SAFETY: the shim supplies `nlhs` writable output slots.
            unsafe { std::slice::from_raw_parts_mut(plhs, nlhs) }
        };
        for (slot, output) in output_pointers.iter_mut().zip(outputs) {
            let value = value_to_mx(&output, state.mx.mode()).map_err(|error| MexDiagnostic {
                identifier: Some("RunMat:MEX:Conversion".into()),
                message: error.message,
            })?;
            *slot = state.mx.allocate(value);
        }
        Ok(())
    })();
    match result {
        Ok(()) => 0,
        Err(error) => {
            state.error = Some(error);
            1
        }
    }
}

unsafe extern "C" fn get_variable(
    host: *mut c_void,
    workspace: *const c_char,
    name: *const c_char,
) -> *mut MxArray {
    let Some(state) = (unsafe { state(host) }) else {
        return std::ptr::null_mut();
    };
    let Some(workspace) = c_string(workspace) else {
        state.fail("workspace name is null");
        return std::ptr::null_mut();
    };
    let Some(name) = c_string(name) else {
        state.fail("variable name is null");
        return std::ptr::null_mut();
    };
    match state.services.get_variable(&workspace, &name) {
        Ok(Some(value)) => match value_to_mx(&value, state.mx.mode()) {
            Ok(value) => {
                let pointer = state.mx.allocate(value);
                if workspace == "global" {
                    state.global_arrays.insert(pointer as usize);
                }
                pointer
            }
            Err(error) => {
                state.fail(error.message);
                std::ptr::null_mut()
            }
        },
        Ok(None) => std::ptr::null_mut(),
        Err(error) => {
            state.error = Some(error);
            std::ptr::null_mut()
        }
    }
}

unsafe extern "C" fn put_variable(
    host: *mut c_void,
    workspace: *const c_char,
    name: *const c_char,
    value: *const MxArray,
) -> i32 {
    let Some(state) = (unsafe { state(host) }) else {
        return 1;
    };
    let result = (|| {
        let workspace = c_string(workspace).ok_or_else(|| invalid_pointer("workspace name"))?;
        let name = c_string(name).ok_or_else(|| invalid_pointer("variable name"))?;
        let value = state.mx.arena().get(value).map_err(|error| MexDiagnostic {
            identifier: Some("RunMat:MEX:InvalidArray".into()),
            message: error.to_string(),
        })?;
        let value = value_from_mx(value).map_err(|error| MexDiagnostic {
            identifier: Some("RunMat:MEX:Conversion".into()),
            message: error.message,
        })?;
        state.services.put_variable(&workspace, &name, value)
    })();
    match result {
        Ok(()) => 0,
        Err(error) => {
            state.error = Some(error);
            1
        }
    }
}

unsafe extern "C" fn is_global(host: *mut c_void, value: *const MxArray) -> i32 {
    unsafe { state(host) }
        .map(|state| i32::from(state.global_arrays.contains(&(value as usize))))
        .unwrap_or(0)
}

unsafe extern "C" fn take_error(host: *mut c_void) -> *mut MxArray {
    let Some(state) = (unsafe { state(host) }) else {
        return std::ptr::null_mut();
    };
    let Some(error) = state.error.take() else {
        return std::ptr::null_mut();
    };
    let mode = state.mx.mode();
    let identifier = value_to_mx(
        &runmat_value::Value::String(error.identifier.unwrap_or_default()),
        mode,
    );
    let message = value_to_mx(&runmat_value::Value::String(error.message), mode);
    match (identifier, message) {
        (Ok(identifier), Ok(message)) => {
            let value = crate::MxArray::structure(
                vec!["identifier".into(), "message".into()],
                vec![Some(Box::new(identifier)), Some(Box::new(message))],
                vec![1, 1],
            );
            match value {
                Ok(value) => state.mx.allocate(value),
                Err(error) => {
                    state.fail(error);
                    std::ptr::null_mut()
                }
            }
        }
        (Err(error), _) | (_, Err(error)) => {
            state.fail(error.message);
            std::ptr::null_mut()
        }
    }
}

fn invalid_count(kind: &str) -> MexDiagnostic {
    MexDiagnostic {
        identifier: Some("RunMat:MEX:InvalidCount".into()),
        message: format!("{kind} count must be non-negative"),
    }
}

fn invalid_pointer(kind: &str) -> MexDiagnostic {
    MexDiagnostic {
        identifier: Some("RunMat:MEX:InvalidPointer".into()),
        message: format!("{kind} pointer is null"),
    }
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
