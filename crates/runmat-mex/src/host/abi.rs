use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{c_char, c_void, CStr, CString};
use std::rc::Rc;
use std::sync::{Arc, Mutex, MutexGuard};

#[cfg(not(target_family = "wasm"))]
use super::ConcurrentMexBoundaryHostServices;
use super::{
    async_abi, BoundaryServiceSlot, DirectMexBoundaryHostServices, LocalBoundaryServiceGuard,
    MexBoundaryHostServices,
};
use crate::mxarray::{MxArrayData, MxSparse, MxSparseValues};
use crate::{
    value_to_mx_for_interface_in_context, MexAsyncResult, MexHostServices, MxApi, MxApiMode,
    MxArray, MxBoundaryInterface, MxClassId,
};

pub const MEX_HOST_ABI_VERSION: u32 = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MexDiagnostic {
    pub identifier: Option<String>,
    pub message: String,
}

pub struct MexCallState {
    inner: Mutex<MexCallStateInner>,
}

pub(crate) struct MexCallStateInner {
    pub mx: MxApi,
    pub error: Option<MexDiagnostic>,
    pub warnings: Vec<MexDiagnostic>,
    pub console: String,
    field_name_cache: Vec<CString>,
    pub(crate) services: BoundaryServiceSlot,
    local_service_guard: Option<LocalBoundaryServiceGuard>,
    global_arrays: BTreeSet<usize>,
    interface: MxBoundaryInterface,
    pub(crate) async_requests: BTreeMap<u64, Arc<dyn MexAsyncResult>>,
    pub(crate) async_property_targets: BTreeMap<u64, usize>,
    pub(crate) next_async_request: u64,
    pub(crate) engine_contexts: BTreeMap<u64, BoundaryServiceSlot>,
    pub(crate) next_engine_context: u64,
}

impl MexCallState {
    pub fn new(mode: MxApiMode) -> Self {
        let interface = MxBoundaryInterface::CMatrix;
        Self::from_inner(MexCallStateInner {
            mx: MxApi::new(mode),
            error: None,
            warnings: Vec::new(),
            console: String::new(),
            field_name_cache: Vec::new(),
            services: BoundaryServiceSlot::Unavailable,
            local_service_guard: None,
            global_arrays: BTreeSet::new(),
            interface,
            async_requests: BTreeMap::new(),
            async_property_targets: BTreeMap::new(),
            next_async_request: 1,
            engine_contexts: BTreeMap::new(),
            next_engine_context: 1,
        })
    }

    pub fn with_services(mode: MxApiMode, services: Rc<dyn MexHostServices>) -> Self {
        Self::with_services_for_interface(mode, MxBoundaryInterface::CMatrix, services)
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn for_interface(mode: MxApiMode, interface: MxBoundaryInterface) -> Self {
        Self::from_inner(MexCallStateInner {
            mx: MxApi::new(mode),
            error: None,
            warnings: Vec::new(),
            console: String::new(),
            field_name_cache: Vec::new(),
            services: BoundaryServiceSlot::Unavailable,
            local_service_guard: None,
            global_arrays: BTreeSet::new(),
            interface,
            async_requests: BTreeMap::new(),
            async_property_targets: BTreeMap::new(),
            next_async_request: 1,
            engine_contexts: BTreeMap::new(),
            next_engine_context: 1,
        })
    }

    pub(crate) fn with_services_for_interface(
        mode: MxApiMode,
        interface: MxBoundaryInterface,
        services: Rc<dyn MexHostServices>,
    ) -> Self {
        let value_context = Rc::new(crate::MxValueContext::new());
        let services = Rc::new(DirectMexBoundaryHostServices::new(
            services,
            Rc::clone(&value_context),
            mode,
            interface,
        ));
        let (local_service_guard, services) = LocalBoundaryServiceGuard::register(services);
        Self::from_inner(MexCallStateInner {
            mx: MxApi::new(mode),
            error: None,
            warnings: Vec::new(),
            console: String::new(),
            field_name_cache: Vec::new(),
            services,
            local_service_guard: Some(local_service_guard),
            global_arrays: BTreeSet::new(),
            interface,
            async_requests: BTreeMap::new(),
            async_property_targets: BTreeMap::new(),
            next_async_request: 1,
            engine_contexts: BTreeMap::new(),
            next_engine_context: 1,
        })
    }

    pub fn set_services(&self, services: Rc<dyn MexHostServices>) {
        self.set_services_for_context(services, Rc::new(crate::MxValueContext::new()));
    }

    pub(crate) fn set_services_for_context(
        &self,
        services: Rc<dyn MexHostServices>,
        values: Rc<crate::MxValueContext>,
    ) {
        let mut state = self.inner.lock().expect("MEX state is not poisoned");
        let services = Rc::new(DirectMexBoundaryHostServices::new(
            services,
            values,
            state.mx.mode(),
            state.interface,
        ));
        let (guard, services) = LocalBoundaryServiceGuard::register(services);
        state.local_service_guard = Some(guard);
        state.services = services;
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn set_boundary_services(
        &self,
        services: std::sync::Arc<dyn ConcurrentMexBoundaryHostServices>,
    ) {
        let mut state = self.inner.lock().expect("MEX state is not poisoned");
        state.local_service_guard = None;
        state.services = BoundaryServiceSlot::concurrent(services);
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn with_boundary_services_for_interface(
        mode: MxApiMode,
        interface: MxBoundaryInterface,
        services: std::sync::Arc<dyn ConcurrentMexBoundaryHostServices>,
    ) -> Self {
        Self::from_inner(MexCallStateInner {
            mx: MxApi::new(mode),
            error: None,
            warnings: Vec::new(),
            console: String::new(),
            field_name_cache: Vec::new(),
            services: BoundaryServiceSlot::concurrent(services),
            local_service_guard: None,
            global_arrays: BTreeSet::new(),
            interface,
            async_requests: BTreeMap::new(),
            async_property_targets: BTreeMap::new(),
            next_async_request: 1,
            engine_contexts: BTreeMap::new(),
            next_engine_context: 1,
        })
    }

    fn from_inner(inner: MexCallStateInner) -> Self {
        Self {
            inner: Mutex::new(inner),
        }
    }

    pub(crate) fn lock(&self) -> Result<MutexGuard<'_, MexCallStateInner>, ()> {
        self.inner.lock().map_err(|_| ())
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn has_pending_async_requests(&self) -> Result<bool, ()> {
        let requests = self
            .lock()?
            .async_requests
            .values()
            .cloned()
            .collect::<Vec<_>>();
        Ok(requests.iter().any(|request| !request.is_ready()))
    }

    pub fn host_api(&self) -> MexHostApiV1 {
        MexHostApiV1 {
            abi_version: MEX_HOST_ABI_VERSION,
            host: std::ptr::from_ref(self).cast_mut().cast(),
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
            class_name,
            number_of_dimensions,
            dimensions,
            number_of_elements,
            set_dimensions,
            data,
            imaginary_data,
            replace_data,
            replace_imaginary_data,
            make_complex,
            make_real,
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
            set_class_name,
            get_property,
            set_property,
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
            allocate_memory,
            reallocate_memory,
            free_memory,
            make_memory_persistent,
            share_array,
            record_host_copy,
            data_array_type,
            create_string_array,
            string_length,
            copy_string,
            set_string,
            retain_data_array,
            release_data_array,
            engine_context_create: async_abi::create_engine_context,
            engine_context_release: async_abi::release_engine_context,
            async_submit_eval: async_abi::submit_eval,
            async_submit_call: async_abi::submit_call,
            async_submit_get_variable: async_abi::submit_get_variable,
            async_submit_put_variable: async_abi::submit_put_variable,
            async_submit_get_property: async_abi::submit_get_property,
            async_submit_set_property: async_abi::submit_set_property,
            async_is_ready: async_abi::is_ready,
            async_wait: async_abi::wait,
            async_cancel: async_abi::cancel,
            async_copy_result: async_abi::copy_result,
            async_copy_text: async_abi::copy_text,
            async_release: async_abi::release,
        }
    }

    pub fn begin_call(&self) {
        let mut state = self.inner.lock().expect("MEX state is not poisoned");
        state.error = None;
        state.warnings.clear();
        state.console.clear();
        state.field_name_cache.clear();
        state.mx.begin_call();
    }

    pub fn finish_call(&self) {
        let mut state = self.inner.lock().expect("MEX state is not poisoned");
        state.mx.finish_call();
        state.error = None;
        state.warnings.clear();
        state.console.clear();
        state.field_name_cache.clear();
        if !state.services.is_concurrent() {
            state.local_service_guard = None;
            state.services = BoundaryServiceSlot::Unavailable;
        }
    }
}

impl MexCallStateInner {
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
    pub class_name: unsafe extern "C" fn(*mut c_void, *const MxArray) -> *const c_char,
    pub number_of_dimensions: unsafe extern "C" fn(*mut c_void, *const MxArray) -> usize,
    pub dimensions: unsafe extern "C" fn(*mut c_void, *const MxArray) -> *const usize,
    pub number_of_elements: unsafe extern "C" fn(*mut c_void, *const MxArray) -> usize,
    pub set_dimensions: unsafe extern "C" fn(*mut c_void, *mut MxArray, usize, *const usize) -> i32,
    pub data: unsafe extern "C" fn(*mut c_void, *mut MxArray, i32) -> *mut c_void,
    pub imaginary_data: unsafe extern "C" fn(*mut c_void, *mut MxArray) -> *mut c_void,
    pub replace_data: unsafe extern "C" fn(*mut c_void, *mut MxArray, *const c_void) -> i32,
    pub replace_imaginary_data:
        unsafe extern "C" fn(*mut c_void, *mut MxArray, *const c_void) -> i32,
    pub make_complex: unsafe extern "C" fn(*mut c_void, *mut MxArray) -> i32,
    pub make_real: unsafe extern "C" fn(*mut c_void, *mut MxArray) -> i32,
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
    pub set_class_name: unsafe extern "C" fn(*mut c_void, *mut MxArray, *const c_char) -> i32,
    pub get_property:
        unsafe extern "C" fn(*mut c_void, *const MxArray, usize, *const c_char) -> *mut MxArray,
    pub set_property:
        unsafe extern "C" fn(*mut c_void, *mut MxArray, usize, *const c_char, *mut MxArray) -> i32,
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
    // ABI v3 fields are appended after the complete v2 prefix.
    pub allocate_memory: unsafe extern "C" fn(*mut c_void, usize, i32) -> *mut c_void,
    pub reallocate_memory: unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> *mut c_void,
    pub free_memory: unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32,
    pub make_memory_persistent: unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32,
    // ABI v4 field is appended after the complete v3 prefix.
    pub share_array: unsafe extern "C" fn(*mut c_void, *const MxArray) -> *mut MxArray,
    // ABI v5 field is appended after the complete v4 prefix.
    pub record_host_copy: unsafe extern "C" fn(*mut c_void, u32, usize),
    // ABI v6 fields are appended after the complete v5 prefix.
    pub data_array_type: unsafe extern "C" fn(*mut c_void, *const MxArray) -> i32,
    pub create_string_array: unsafe extern "C" fn(*mut c_void, usize, *const usize) -> *mut MxArray,
    pub string_length: unsafe extern "C" fn(*mut c_void, *const MxArray, usize) -> usize,
    pub copy_string:
        unsafe extern "C" fn(*mut c_void, *const MxArray, usize, *mut u16, usize) -> i32,
    pub set_string:
        unsafe extern "C" fn(*mut c_void, *mut MxArray, usize, *const u16, usize) -> i32,
    // ABI v7 fields are appended after the complete v6 prefix.
    pub retain_data_array: unsafe extern "C" fn(*mut c_void, *const MxArray) -> i32,
    pub release_data_array: unsafe extern "C" fn(*mut c_void, *mut MxArray, i32) -> i32,
    // ABI v8 fields are appended after the complete v7 prefix.
    pub engine_context_create: unsafe extern "C" fn(*mut c_void) -> u64,
    pub engine_context_release: unsafe extern "C" fn(*mut c_void, u64),
    pub async_submit_eval: unsafe extern "C" fn(*mut c_void, u64, *const c_char, i32, i32) -> u64,
    pub async_submit_call: unsafe extern "C" fn(
        *mut c_void,
        u64,
        *const c_char,
        usize,
        usize,
        *const *const MxArray,
        i32,
        i32,
    ) -> u64,
    pub async_submit_get_variable:
        unsafe extern "C" fn(*mut c_void, u64, *const c_char, *const c_char) -> u64,
    pub async_submit_put_variable:
        unsafe extern "C" fn(*mut c_void, u64, *const c_char, *const c_char, *const MxArray) -> u64,
    pub async_submit_get_property:
        unsafe extern "C" fn(*mut c_void, u64, *const MxArray, usize, *const c_char) -> u64,
    pub async_submit_set_property: unsafe extern "C" fn(
        *mut c_void,
        u64,
        *mut MxArray,
        usize,
        *const c_char,
        *const MxArray,
    ) -> u64,
    pub async_is_ready: unsafe extern "C" fn(*mut c_void, u64) -> i32,
    pub async_wait: unsafe extern "C" fn(*mut c_void, u64, i64) -> i32,
    pub async_cancel: unsafe extern "C" fn(*mut c_void, u64, i32) -> i32,
    pub async_copy_result: unsafe extern "C" fn(*mut c_void, u64, usize, *mut *mut MxArray) -> i32,
    pub async_copy_text: unsafe extern "C" fn(*mut c_void, u64, u32, *mut c_char, usize) -> usize,
    pub async_release: unsafe extern "C" fn(*mut c_void, u64),
}

unsafe fn state<'a>(host: *mut c_void) -> Option<MutexGuard<'a, MexCallStateInner>> {
    // SAFETY: every vtable is created by `MexCallState::host_api`; the loader
    // pins that state for the module lifetime and tears it down only after the
    // module's retained Data API controls and asynchronous work have ended.
    unsafe { host.cast::<MexCallState>().as_ref() }?.lock().ok()
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
    let Some(mut state) = (unsafe { state(host) }) else {
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
        .map(|mut state| state.mx.create_double_scalar(value))
        .unwrap_or(std::ptr::null_mut())
}

unsafe extern "C" fn create_logical(
    host: *mut c_void,
    ndim: usize,
    dims: *const usize,
) -> *mut MxArray {
    let Some(mut state) = (unsafe { state(host) }) else {
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
    let Some(mut state) = (unsafe { state(host) }) else {
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
    let Some(mut state) = (unsafe { state(host) }) else {
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
    let Some(mut state) = (unsafe { state(host) }) else {
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
    storage_kind: i32,
) -> *mut MxArray {
    let Some(mut state) = (unsafe { state(host) }) else {
        return std::ptr::null_mut();
    };
    match state.mx.create_sparse(rows, cols, nzmax, storage_kind) {
        Ok(value) => value,
        Err(error) => {
            state.fail(error);
            std::ptr::null_mut()
        }
    }
}

unsafe extern "C" fn duplicate_array(host: *mut c_void, value: *const MxArray) -> *mut MxArray {
    let Some(mut state) = (unsafe { state(host) }) else {
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

unsafe extern "C" fn share_array(host: *mut c_void, value: *const MxArray) -> *mut MxArray {
    let Some(mut state) = (unsafe { state(host) }) else {
        return std::ptr::null_mut();
    };
    match state.mx.share(value) {
        Ok(value) => value,
        Err(error) => {
            state.fail(error.to_string());
            std::ptr::null_mut()
        }
    }
}

unsafe extern "C" fn retain_data_array(host: *mut c_void, value: *const MxArray) -> i32 {
    let Some(mut state) = (unsafe { state(host) }) else {
        return 1;
    };
    match state.mx.retain_data_api(value) {
        Ok(()) => 0,
        Err(error) => {
            state.fail(error.to_string());
            1
        }
    }
}

unsafe extern "C" fn release_data_array(host: *mut c_void, value: *mut MxArray, owned: i32) -> i32 {
    let Some(mut state) = (unsafe { state(host) }) else {
        return 1;
    };
    match state.mx.release_data_api(value, owned != 0) {
        Ok(()) => 0,
        Err(error) => {
            state.fail(error.to_string());
            1
        }
    }
}

unsafe extern "C" fn record_host_copy(_host: *mut c_void, reason: u32, byte_length: usize) {
    let reason = match reason {
        value if value == runmat_value::HostCopyReason::MemoryLayoutConversion as u32 => {
            Some(runmat_value::HostCopyReason::MemoryLayoutConversion)
        }
        value if value == runmat_value::HostCopyReason::SparseLayoutConversion as u32 => {
            Some(runmat_value::HostCopyReason::SparseLayoutConversion)
        }
        _ => None,
    };
    if let Some(reason) = reason {
        runmat_value::record_host_copy(reason, byte_length);
    }
}

unsafe extern "C" fn data_array_type(host: *mut c_void, value: *const MxArray) -> i32 {
    let Some(mut state) = (unsafe { state(host) }) else {
        return 0;
    };
    let Ok(value) = state.mx.arena().get(value) else {
        state.fail("invalid Data API array");
        return 0;
    };
    match value.data() {
        MxArrayData::String(_) => 31,
        MxArrayData::Object { properties, .. }
            if properties
                .iter()
                .any(|property| property == "__enum_member__") =>
        {
            27
        }
        MxArrayData::Object { .. } => 25,
        MxArrayData::Handle(_) => 26,
        MxArrayData::Sparse(MxSparse {
            values: MxSparseValues::Logical(_),
            ..
        }) => 28,
        MxArrayData::Sparse(_) if value.is_complex() => 30,
        MxArrayData::Sparse(_) => 29,
        _ => 0,
    }
}

unsafe extern "C" fn create_string_array(
    host: *mut c_void,
    ndim: usize,
    dims: *const usize,
) -> *mut MxArray {
    let Some(mut state) = (unsafe { state(host) }) else {
        return std::ptr::null_mut();
    };
    let shape = match unsafe { shape(ndim, dims) } {
        Ok(shape) => shape,
        Err(error) => {
            state.fail(error);
            return std::ptr::null_mut();
        }
    };
    let Some(count) = shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
    else {
        state.fail("string array dimensions exceed platform limits");
        return std::ptr::null_mut();
    };
    match MxArray::string(vec!["<missing>".to_string(); count], shape) {
        Ok(value) => state.mx.allocate(value),
        Err(error) => {
            state.fail(error);
            std::ptr::null_mut()
        }
    }
}

unsafe extern "C" fn string_length(
    host: *mut c_void,
    value: *const MxArray,
    index: usize,
) -> usize {
    let Some(mut state) = (unsafe { state(host) }) else {
        return usize::MAX;
    };
    let Ok(value) = state.mx.arena().get(value) else {
        state.fail("invalid Data API string array");
        return usize::MAX;
    };
    let MxArrayData::String(values) = value.data() else {
        state.fail("array is not a Data API string array");
        return usize::MAX;
    };
    let Some(value) = values.get(index) else {
        state.fail("string array index is out of range");
        return usize::MAX;
    };
    value.encode_utf16().count()
}

unsafe extern "C" fn copy_string(
    host: *mut c_void,
    value: *const MxArray,
    index: usize,
    output: *mut u16,
    output_length: usize,
) -> i32 {
    let Some(mut state) = (unsafe { state(host) }) else {
        return 1;
    };
    let Ok(value) = state.mx.arena().get(value) else {
        state.fail("invalid Data API string array");
        return 1;
    };
    let MxArrayData::String(values) = value.data() else {
        state.fail("array is not a Data API string array");
        return 1;
    };
    let Some(value) = values.get(index) else {
        state.fail("string array index is out of range");
        return 1;
    };
    let encoded = value.encode_utf16().collect::<Vec<_>>();
    if encoded.len() != output_length || (output.is_null() && output_length != 0) {
        state.fail("Data API string output buffer has the wrong length");
        return 1;
    }
    if output_length != 0 {
        // SAFETY: the C++ caller supplies exactly `output_length` writable UTF-16 units.
        unsafe { std::ptr::copy_nonoverlapping(encoded.as_ptr(), output, output_length) };
    }
    runmat_value::record_host_copy(
        runmat_value::HostCopyReason::CharacterEncoding,
        output_length.saturating_mul(std::mem::size_of::<u16>()),
    );
    0
}

unsafe extern "C" fn set_string(
    host: *mut c_void,
    value: *mut MxArray,
    index: usize,
    input: *const u16,
    input_length: usize,
) -> i32 {
    let Some(mut state) = (unsafe { state(host) }) else {
        return 1;
    };
    let units = if input_length == 0 {
        &[][..]
    } else if input.is_null() {
        state.fail("Data API string input pointer is null");
        return 1;
    } else {
        // SAFETY: the C++ caller supplies `input_length` readable UTF-16 units.
        unsafe { std::slice::from_raw_parts(input, input_length) }
    };
    let text = match String::from_utf16(units) {
        Ok(text) => text,
        Err(_) => {
            state.fail("Data API string contains invalid UTF-16");
            return 1;
        }
    };
    runmat_value::record_host_copy(
        runmat_value::HostCopyReason::CharacterEncoding,
        input_length.saturating_mul(std::mem::size_of::<u16>()),
    );
    let Ok(value) = state.mx.arena_mut().get_mut(value) else {
        state.fail("invalid Data API string array");
        return 1;
    };
    let MxArrayData::String(values) = value.data_mut() else {
        state.fail("array is not a Data API string array");
        return 1;
    };
    let Some(slot) = values.get_mut(index) else {
        state.fail("string array index is out of range");
        return 1;
    };
    *slot = text;
    0
}

unsafe extern "C" fn destroy_array(host: *mut c_void, value: *mut MxArray) -> i32 {
    let Some(mut state) = (unsafe { state(host) }) else {
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

unsafe extern "C" fn class_name(host: *mut c_void, value: *const MxArray) -> *const c_char {
    let Some(mut state) = (unsafe { state(host) }) else {
        return std::ptr::null();
    };
    let name = match state.mx.arena().get(value) {
        Ok(value) => value.class_name(),
        Err(error) => {
            state.fail(error.to_string());
            return std::ptr::null();
        }
    };
    let Ok(name) = CString::new(name) else {
        state.fail("mxArray class name contains a null byte");
        return std::ptr::null();
    };
    state.field_name_cache.push(name);
    state
        .field_name_cache
        .last()
        .map(|name| name.as_ptr())
        .unwrap_or(std::ptr::null())
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
    let Some(mut state) = (unsafe { state(host) }) else {
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
    let Some(mut state) = (unsafe { state(host) }) else {
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
    let Some(mut state) = (unsafe { state(host) }) else {
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

unsafe extern "C" fn make_complex(host: *mut c_void, value: *mut MxArray) -> i32 {
    let Some(mut state) = (unsafe { state(host) }) else {
        return 1;
    };
    match state.mx.make_complex(value) {
        Ok(()) => 0,
        Err(error) => {
            state.fail(error);
            1
        }
    }
}

unsafe extern "C" fn make_real(host: *mut c_void, value: *mut MxArray) -> i32 {
    let Some(mut state) = (unsafe { state(host) }) else {
        return 1;
    };
    match state.mx.make_real(value) {
        Ok(()) => 0,
        Err(error) => {
            state.fail(error);
            1
        }
    }
}

fn replace_data_component(
    host: *mut c_void,
    value: *mut MxArray,
    source: *const c_void,
    imaginary: bool,
) -> i32 {
    let Some(mut state) = (unsafe { state(host) }) else {
        return 1;
    };
    // SAFETY: the native support source transfers a buffer matching the array's reported size.
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
    let Some(mut state) = (unsafe { state(host) }) else {
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
    let Some(mut state) = (unsafe { state(host) }) else {
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
        .and_then(|state| {
            state
                .mx
                .field_names(value)
                .ok()
                .and_then(|fields| i32::try_from(fields.len()).ok())
        })
        .unwrap_or(0)
}

unsafe extern "C" fn field_name(
    host: *mut c_void,
    value: *const MxArray,
    field: i32,
) -> *const c_char {
    let Some(mut state) = (unsafe { state(host) }) else {
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
        .and_then(|state| {
            state
                .mx
                .field_names(value)
                .ok()
                .and_then(|fields| fields.iter().position(|field| field == &name))
                .and_then(|index| i32::try_from(index).ok())
        })
        .unwrap_or(-1)
}

unsafe extern "C" fn add_field(host: *mut c_void, value: *mut MxArray, name: *const c_char) -> i32 {
    let Some(mut state) = (unsafe { state(host) }) else {
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
    let Some(mut state) = (unsafe { state(host) }) else {
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
    let Some(mut state) = (unsafe { state(host) }) else {
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
    let Some(mut state) = (unsafe { state(host) }) else {
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

unsafe extern "C" fn set_class_name(
    host: *mut c_void,
    value: *mut MxArray,
    class_name: *const c_char,
) -> i32 {
    let Some(mut state) = (unsafe { state(host) }) else {
        return 1;
    };
    let Some(class_name) = c_string(class_name) else {
        state.fail("object class name is null");
        return 1;
    };
    match state.mx.set_class_name(value, class_name) {
        Ok(()) => 0,
        Err(error) => {
            state.fail(error);
            1
        }
    }
}

unsafe extern "C" fn get_property(
    host: *mut c_void,
    value: *const MxArray,
    index: usize,
    property_name: *const c_char,
) -> *mut MxArray {
    let Some(mut state) = (unsafe { state(host) }) else {
        return std::ptr::null_mut();
    };
    let Some(property_name) = c_string(property_name) else {
        state.fail("object property name is null");
        return std::ptr::null_mut();
    };
    let handle = state
        .mx
        .arena()
        .get(value)
        .ok()
        .and_then(|array| match array.data() {
            MxArrayData::Handle(_) => Some(array.clone()),
            _ => None,
        });
    if let Some(handle) = handle {
        if index != 0 {
            state.fail("handle object index is out of range");
            return std::ptr::null_mut();
        }
        return match state
            .services
            .get_object_property(handle, index, &property_name)
        {
            Ok(value) => state.mx.allocate(value),
            Err(error) => {
                state.error = Some(error);
                std::ptr::null_mut()
            }
        };
    }
    match state.mx.get_property(value, index, &property_name) {
        Ok(value) => value,
        Err(error) => {
            state.fail(error);
            std::ptr::null_mut()
        }
    }
}

unsafe extern "C" fn set_property(
    host: *mut c_void,
    value: *mut MxArray,
    index: usize,
    property_name: *const c_char,
    child: *mut MxArray,
) -> i32 {
    let Some(mut state) = (unsafe { state(host) }) else {
        return 1;
    };
    let Some(property_name) = c_string(property_name) else {
        state.fail("object property name is null");
        return 1;
    };
    let handle = state
        .mx
        .arena()
        .get(value)
        .ok()
        .and_then(|array| match array.data() {
            MxArrayData::Handle(_) => Some(array.clone()),
            _ => None,
        });
    if let Some(handle) = handle {
        if index != 0 {
            state.fail("handle object index is out of range");
            return 1;
        }
        let property_value = match state.mx.arena().get(child).cloned() {
            Ok(value) => value,
            Err(error) => {
                state.fail(error.to_string());
                return 1;
            }
        };
        if state.mx.destroy(child).is_err() {
            state.fail("could not consume handle property value");
            return 1;
        }
        return match state.services.set_object_property(
            handle,
            index,
            &property_name,
            property_value,
        ) {
            Ok(updated) => match state.mx.arena_mut().get_mut(value) {
                Ok(array) => {
                    *array = updated;
                    0
                }
                Err(error) => {
                    state.fail(error.to_string());
                    1
                }
            },
            Err(error) => {
                state.error = Some(error);
                1
            }
        };
    }
    match state.mx.set_property(value, index, &property_name, child) {
        Ok(()) => 0,
        Err(error) => {
            state.fail(error);
            1
        }
    }
}

unsafe extern "C" fn sparse_row_indices(host: *mut c_void, value: *mut MxArray) -> *mut usize {
    let Some(mut state) = (unsafe { state(host) }) else {
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
    let Some(mut state) = (unsafe { state(host) }) else {
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
        .and_then(|mut state| state.mx.sparse_indices(value).ok())
        .map(|(_, _, nzmax)| nzmax)
        .unwrap_or(0)
}

unsafe extern "C" fn set_sparse_nzmax(host: *mut c_void, value: *mut MxArray, nzmax: usize) -> i32 {
    let Some(mut state) = (unsafe { state(host) }) else {
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
    let Some(mut state) = (unsafe { state(host) }) else {
        return 1;
    };
    // SAFETY: the support source transfers the fixed row-capacity or column-pointer buffer.
    match unsafe { state.mx.replace_sparse_indices(value, source, columns) } {
        Ok(()) => 0,
        Err(error) => {
            state.fail(error);
            1
        }
    }
}

unsafe extern "C" fn allocate_memory(
    host: *mut c_void,
    byte_length: usize,
    zeroed: i32,
) -> *mut c_void {
    let Some(mut state) = (unsafe { state(host) }) else {
        return std::ptr::null_mut();
    };
    state.mx.allocate_memory(byte_length, zeroed != 0)
}

unsafe extern "C" fn reallocate_memory(
    host: *mut c_void,
    pointer: *mut c_void,
    byte_length: usize,
) -> *mut c_void {
    let Some(mut state) = (unsafe { state(host) }) else {
        return std::ptr::null_mut();
    };
    state.mx.reallocate_memory(pointer, byte_length)
}

unsafe extern "C" fn free_memory(host: *mut c_void, pointer: *mut c_void) -> i32 {
    let Some(mut state) = (unsafe { state(host) }) else {
        return 1;
    };
    i32::from(!state.mx.free_memory(pointer))
}

unsafe extern "C" fn make_memory_persistent(host: *mut c_void, pointer: *mut c_void) -> i32 {
    let Some(mut state) = (unsafe { state(host) }) else {
        return 1;
    };
    i32::from(!state.mx.make_memory_persistent(pointer))
}

unsafe extern "C" fn make_array_persistent(host: *mut c_void, value: *mut MxArray) -> i32 {
    let Some(mut state) = (unsafe { state(host) }) else {
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
    let Some(mut state) = (unsafe { state(host) }) else {
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
    let Some(mut state) = (unsafe { state(host) }) else {
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
            // SAFETY: the support source supplies `nrhs` readable array pointers.
            unsafe { std::slice::from_raw_parts(prhs, nrhs) }
        };
        let arguments = inputs
            .iter()
            .map(|pointer| {
                state
                    .mx
                    .arena()
                    .get(*pointer)
                    .cloned()
                    .map_err(|error| MexDiagnostic {
                        identifier: Some("RunMat:MEX:InvalidArray".into()),
                        message: error.to_string(),
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
            // SAFETY: the support source supplies `nlhs` writable output slots.
            unsafe { std::slice::from_raw_parts_mut(plhs, nlhs) }
        };
        for (slot, output) in output_pointers.iter_mut().zip(outputs) {
            *slot = state.mx.allocate(output);
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
    let Some(mut state) = (unsafe { state(host) }) else {
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
        Ok(Some(value)) => {
            let pointer = state.mx.allocate(value);
            if workspace == "global" {
                state.global_arrays.insert(pointer as usize);
            }
            pointer
        }
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
    let Some(mut state) = (unsafe { state(host) }) else {
        return 1;
    };
    let result = (|| {
        let workspace = c_string(workspace).ok_or_else(|| invalid_pointer("workspace name"))?;
        let name = c_string(name).ok_or_else(|| invalid_pointer("variable name"))?;
        let value = state
            .mx
            .arena()
            .get(value)
            .cloned()
            .map_err(|error| MexDiagnostic {
                identifier: Some("RunMat:MEX:InvalidArray".into()),
                message: error.to_string(),
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
    let Some(mut state) = (unsafe { state(host) }) else {
        return std::ptr::null_mut();
    };
    let Some(error) = state.error.take() else {
        return std::ptr::null_mut();
    };
    let mode = state.mx.mode();
    let identifier = value_to_mx_for_interface_in_context(
        &runmat_value::Value::String(error.identifier.unwrap_or_default()),
        mode,
        state.interface,
        None,
    );
    let message = value_to_mx_for_interface_in_context(
        &runmat_value::Value::String(error.message),
        mode,
        state.interface,
        None,
    );
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
    // SAFETY: strings originate in the bound native support source and are valid through the
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
    if let Some(mut state) = unsafe { state(host) } {
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
    if let Some(mut state) = unsafe { state(host) } {
        state.warnings.push(MexDiagnostic {
            identifier: c_string(identifier).filter(|value| !value.is_empty()),
            message: c_string(message).unwrap_or_default(),
        });
    }
}

unsafe extern "C" fn write_console(host: *mut c_void, text: *const c_char) {
    if let (Some(mut state), Some(text)) = (unsafe { state(host) }, c_string(text)) {
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
    fn allocator_and_data_api_callbacks_extend_complete_previous_prefixes() {
        let pointer_size = std::mem::size_of::<usize>();
        assert_eq!(
            std::mem::offset_of!(MexHostApiV1, allocate_memory),
            std::mem::size_of::<MexHostApiV1>() - 27 * pointer_size
        );
        assert_eq!(
            std::mem::offset_of!(MexHostApiV1, make_memory_persistent),
            std::mem::size_of::<MexHostApiV1>() - 24 * pointer_size
        );
        assert_eq!(
            std::mem::offset_of!(MexHostApiV1, share_array),
            std::mem::size_of::<MexHostApiV1>() - 23 * pointer_size
        );
        assert_eq!(
            std::mem::offset_of!(MexHostApiV1, record_host_copy),
            std::mem::size_of::<MexHostApiV1>() - 22 * pointer_size
        );
        assert_eq!(
            std::mem::offset_of!(MexHostApiV1, data_array_type),
            std::mem::size_of::<MexHostApiV1>() - 21 * pointer_size
        );
        assert_eq!(
            std::mem::offset_of!(MexHostApiV1, set_string),
            std::mem::size_of::<MexHostApiV1>() - 17 * pointer_size
        );
        assert_eq!(
            std::mem::offset_of!(MexHostApiV1, engine_context_create),
            std::mem::size_of::<MexHostApiV1>() - 14 * pointer_size
        );
        assert_eq!(
            std::mem::offset_of!(MexHostApiV1, async_submit_eval),
            std::mem::size_of::<MexHostApiV1>() - 12 * pointer_size
        );
        assert_eq!(
            std::mem::offset_of!(MexHostApiV1, async_release),
            std::mem::size_of::<MexHostApiV1>() - pointer_size
        );
    }

    #[test]
    fn vtable_uses_opaque_host_and_reports_bad_class_without_unwinding() {
        let state = MexCallState::new(MxApiMode::SeparateComplex);
        let api = state.host_api();
        let dims = [1usize, 1];
        let value = unsafe { (api.create_numeric)(api.host, dims.len(), dims.as_ptr(), 999, 0) };
        assert!(value.is_null());
        assert!(state
            .lock()
            .unwrap()
            .error
            .as_ref()
            .unwrap()
            .message
            .contains("mxClassID"));
    }

    #[test]
    fn host_call_state_is_safe_for_data_api_worker_access() {
        fn require_send_sync<T: Send + Sync>() {}
        require_send_sync::<MexCallState>();
    }
}
