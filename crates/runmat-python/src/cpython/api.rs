use std::ffi::{c_char, c_double, c_int, c_longlong, c_ulong, c_ulonglong, c_void, CString};
use std::mem::ManuallyDrop;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use libloading::Library;

use crate::value::PythonBufferOwner;
use crate::PythonError;

pub(crate) type PyObject = c_void;
pub(crate) type PySsize = isize;
pub(crate) type PyGilState = c_int;
type PyCapsuleDestructor = unsafe extern "C" fn(*mut PyObject);
pub(crate) type PyCapsuleGetPointer =
    unsafe extern "C" fn(*mut PyObject, *const c_char) -> *mut c_void;
pub(crate) type PyMethod = unsafe extern "C" fn(*mut PyObject, *mut PyObject) -> *mut PyObject;

#[derive(Clone, Copy)]
pub(crate) struct PythonInterrupt {
    pub(super) ensure: unsafe extern "C" fn() -> PyGilState,
    pub(super) release: unsafe extern "C" fn(PyGilState),
    pub(super) set_async_exception: unsafe extern "C" fn(c_ulong, *mut PyObject) -> c_int,
    pub(super) thread: c_ulong,
    pub(super) exception: usize,
}

impl PythonInterrupt {
    pub(crate) fn request(self) -> bool {
        // SAFETY: attaching this caller is required by PyThreadState_SetAsyncExc.
        // The exception is an owned builtin reference retained by the interpreter,
        // and the target identity was captured on its execution lane.
        unsafe {
            let state = (self.ensure)();
            let changed = (self.set_async_exception)(self.thread, self.exception as *mut PyObject);
            (self.release)(state);
            changed == 1
        }
    }
}

#[repr(C)]
pub(crate) struct PyMethodDef {
    pub name: *const c_char,
    pub method: PyMethod,
    pub flags: c_int,
    pub documentation: *const c_char,
}

static CAPSULE_GET_POINTER: OnceLock<PyCapsuleGetPointer> = OnceLock::new();
static PROCESS_INTERPRETER_LIBRARY: Mutex<Option<std::path::PathBuf>> = Mutex::new(None);

pub(crate) unsafe fn capsule_pointer(capsule: *mut PyObject, name: *const c_char) -> *mut c_void {
    CAPSULE_GET_POINTER
        .get()
        .map_or(std::ptr::null_mut(), |get| unsafe { get(capsule, name) })
}

pub(crate) struct CpythonApi {
    pub py_initialize_ex: unsafe extern "C" fn(c_int),
    pub py_is_initialized: unsafe extern "C" fn() -> c_int,
    pub py_gil_state_ensure: unsafe extern "C" fn() -> PyGilState,
    pub py_gil_state_release: unsafe extern "C" fn(PyGilState),
    pub py_eval_save_thread: unsafe extern "C" fn() -> *mut c_void,
    pub py_import_import_module: unsafe extern "C" fn(*const c_char) -> *mut PyObject,
    pub py_object_get_attr_string:
        unsafe extern "C" fn(*mut PyObject, *const c_char) -> *mut PyObject,
    pub py_object_set_attr_string:
        unsafe extern "C" fn(*mut PyObject, *const c_char, *mut PyObject) -> c_int,
    pub py_object_call:
        unsafe extern "C" fn(*mut PyObject, *mut PyObject, *mut PyObject) -> *mut PyObject,
    pub py_object_is_instance: unsafe extern "C" fn(*mut PyObject, *mut PyObject) -> c_int,
    pub py_object_is_true: unsafe extern "C" fn(*mut PyObject) -> c_int,
    pub py_callable_check: unsafe extern "C" fn(*mut PyObject) -> c_int,
    pub py_object_str: unsafe extern "C" fn(*mut PyObject) -> *mut PyObject,
    pub py_object_get_item: unsafe extern "C" fn(*mut PyObject, *mut PyObject) -> *mut PyObject,
    pub py_object_set_item:
        unsafe extern "C" fn(*mut PyObject, *mut PyObject, *mut PyObject) -> c_int,
    pub py_object_get_iter: unsafe extern "C" fn(*mut PyObject) -> *mut PyObject,
    pub py_iter_next: unsafe extern "C" fn(*mut PyObject) -> *mut PyObject,
    pub py_sequence_check: unsafe extern "C" fn(*mut PyObject) -> c_int,
    pub py_mapping_check: unsafe extern "C" fn(*mut PyObject) -> c_int,
    pub py_tuple_new: unsafe extern "C" fn(PySsize) -> *mut PyObject,
    pub py_tuple_set_item: unsafe extern "C" fn(*mut PyObject, PySsize, *mut PyObject) -> c_int,
    pub py_tuple_get_item: unsafe extern "C" fn(*mut PyObject, PySsize) -> *mut PyObject,
    pub py_tuple_size: unsafe extern "C" fn(*mut PyObject) -> PySsize,
    pub py_list_new: unsafe extern "C" fn(PySsize) -> *mut PyObject,
    pub py_list_set_item: unsafe extern "C" fn(*mut PyObject, PySsize, *mut PyObject) -> c_int,
    pub py_list_get_item: unsafe extern "C" fn(*mut PyObject, PySsize) -> *mut PyObject,
    pub py_list_size: unsafe extern "C" fn(*mut PyObject) -> PySsize,
    pub py_dict_new: unsafe extern "C" fn() -> *mut PyObject,
    pub py_dict_set_item:
        unsafe extern "C" fn(*mut PyObject, *mut PyObject, *mut PyObject) -> c_int,
    pub py_dict_set_item_string:
        unsafe extern "C" fn(*mut PyObject, *const c_char, *mut PyObject) -> c_int,
    pub py_float_from_double: unsafe extern "C" fn(c_double) -> *mut PyObject,
    pub py_float_as_double: unsafe extern "C" fn(*mut PyObject) -> c_double,
    pub py_long_from_long_long: unsafe extern "C" fn(c_longlong) -> *mut PyObject,
    pub py_long_from_unsigned_long_long: unsafe extern "C" fn(c_ulonglong) -> *mut PyObject,
    pub py_long_as_long_long_and_overflow:
        unsafe extern "C" fn(*mut PyObject, *mut c_int) -> c_longlong,
    pub py_long_as_unsigned_long_long: unsafe extern "C" fn(*mut PyObject) -> c_ulonglong,
    pub py_bool_from_long: unsafe extern "C" fn(c_int) -> *mut PyObject,
    pub py_complex_from_doubles: unsafe extern "C" fn(c_double, c_double) -> *mut PyObject,
    pub py_complex_real_as_double: unsafe extern "C" fn(*mut PyObject) -> c_double,
    pub py_complex_imag_as_double: unsafe extern "C" fn(*mut PyObject) -> c_double,
    pub py_unicode_from_string_and_size:
        unsafe extern "C" fn(*const c_char, PySsize) -> *mut PyObject,
    pub py_unicode_as_utf8_and_size:
        unsafe extern "C" fn(*mut PyObject, *mut PySsize) -> *const c_char,
    pub py_bytes_from_string_and_size:
        unsafe extern "C" fn(*const c_char, PySsize) -> *mut PyObject,
    pub py_bytes_as_string_and_size:
        unsafe extern "C" fn(*mut PyObject, *mut *mut c_char, *mut PySsize) -> c_int,
    pub py_capsule_new: unsafe extern "C" fn(
        *mut c_void,
        *const c_char,
        Option<PyCapsuleDestructor>,
    ) -> *mut PyObject,
    pub py_cfunction_new_ex:
        unsafe extern "C" fn(*mut PyMethodDef, *mut PyObject, *mut PyObject) -> *mut PyObject,
    pub py_run_string_flags: unsafe extern "C" fn(
        *const c_char,
        c_int,
        *mut PyObject,
        *mut PyObject,
        *mut c_void,
    ) -> *mut PyObject,
    pub py_err_occurred: unsafe extern "C" fn() -> *mut PyObject,
    pub py_err_fetch:
        unsafe extern "C" fn(*mut *mut PyObject, *mut *mut PyObject, *mut *mut PyObject),
    pub py_err_normalize_exception:
        unsafe extern "C" fn(*mut *mut PyObject, *mut *mut PyObject, *mut *mut PyObject),
    pub py_err_clear: unsafe extern "C" fn(),
    pub py_err_set_string: unsafe extern "C" fn(*mut PyObject, *const c_char),
    pub py_thread_get_thread_ident: unsafe extern "C" fn() -> c_ulong,
    pub py_thread_state_set_async_exc: unsafe extern "C" fn(c_ulong, *mut PyObject) -> c_int,
    pub py_inc_ref: unsafe extern "C" fn(*mut PyObject),
    pub py_dec_ref: unsafe extern "C" fn(*mut PyObject),
    pub py_decode_locale: unsafe extern "C" fn(*const c_char, *mut usize) -> *mut c_void,
    pub py_set_program_name: unsafe extern "C" fn(*const c_void),
    pub py_set_python_home: unsafe extern "C" fn(*const c_void),
    initialized: bool,
    program_name: *mut c_void,
    python_home: *mut c_void,
    library_path: std::path::PathBuf,
    library: ManuallyDrop<Library>,
}

impl std::fmt::Debug for CpythonApi {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CpythonApi")
            .field("initialized", &self.initialized)
            .finish_non_exhaustive()
    }
}

impl CpythonApi {
    pub(crate) fn load(path: &Path) -> Result<Self, PythonError> {
        let library = open_library(path)?;
        // SAFETY: every symbol below is part of the CPython stable embedding
        // ABI. The loaded library remains resident for the lifetime of this
        // table and, after initialization, for the lifetime of the process.
        unsafe {
            let capsule_get_pointer = symbol(&library, b"PyCapsule_GetPointer\0")?;
            let _ = CAPSULE_GET_POINTER.set(capsule_get_pointer);
            Ok(Self {
                py_initialize_ex: symbol(&library, b"Py_InitializeEx\0")?,
                py_is_initialized: symbol(&library, b"Py_IsInitialized\0")?,
                py_gil_state_ensure: symbol(&library, b"PyGILState_Ensure\0")?,
                py_gil_state_release: symbol(&library, b"PyGILState_Release\0")?,
                py_eval_save_thread: symbol(&library, b"PyEval_SaveThread\0")?,
                py_import_import_module: symbol(&library, b"PyImport_ImportModule\0")?,
                py_object_get_attr_string: symbol(&library, b"PyObject_GetAttrString\0")?,
                py_object_set_attr_string: symbol(&library, b"PyObject_SetAttrString\0")?,
                py_object_call: symbol(&library, b"PyObject_Call\0")?,
                py_object_is_instance: symbol(&library, b"PyObject_IsInstance\0")?,
                py_object_is_true: symbol(&library, b"PyObject_IsTrue\0")?,
                py_callable_check: symbol(&library, b"PyCallable_Check\0")?,
                py_object_str: symbol(&library, b"PyObject_Str\0")?,
                py_object_get_item: symbol(&library, b"PyObject_GetItem\0")?,
                py_object_set_item: symbol(&library, b"PyObject_SetItem\0")?,
                py_object_get_iter: symbol(&library, b"PyObject_GetIter\0")?,
                py_iter_next: symbol(&library, b"PyIter_Next\0")?,
                py_sequence_check: symbol(&library, b"PySequence_Check\0")?,
                py_mapping_check: symbol(&library, b"PyMapping_Check\0")?,
                py_tuple_new: symbol(&library, b"PyTuple_New\0")?,
                py_tuple_set_item: symbol(&library, b"PyTuple_SetItem\0")?,
                py_tuple_get_item: symbol(&library, b"PyTuple_GetItem\0")?,
                py_tuple_size: symbol(&library, b"PyTuple_Size\0")?,
                py_list_new: symbol(&library, b"PyList_New\0")?,
                py_list_set_item: symbol(&library, b"PyList_SetItem\0")?,
                py_list_get_item: symbol(&library, b"PyList_GetItem\0")?,
                py_list_size: symbol(&library, b"PyList_Size\0")?,
                py_dict_new: symbol(&library, b"PyDict_New\0")?,
                py_dict_set_item: symbol(&library, b"PyDict_SetItem\0")?,
                py_dict_set_item_string: symbol(&library, b"PyDict_SetItemString\0")?,
                py_float_from_double: symbol(&library, b"PyFloat_FromDouble\0")?,
                py_float_as_double: symbol(&library, b"PyFloat_AsDouble\0")?,
                py_long_from_long_long: symbol(&library, b"PyLong_FromLongLong\0")?,
                py_long_from_unsigned_long_long: symbol(
                    &library,
                    b"PyLong_FromUnsignedLongLong\0",
                )?,
                py_long_as_long_long_and_overflow: symbol(
                    &library,
                    b"PyLong_AsLongLongAndOverflow\0",
                )?,
                py_long_as_unsigned_long_long: symbol(&library, b"PyLong_AsUnsignedLongLong\0")?,
                py_bool_from_long: symbol(&library, b"PyBool_FromLong\0")?,
                py_complex_from_doubles: symbol(&library, b"PyComplex_FromDoubles\0")?,
                py_complex_real_as_double: symbol(&library, b"PyComplex_RealAsDouble\0")?,
                py_complex_imag_as_double: symbol(&library, b"PyComplex_ImagAsDouble\0")?,
                py_unicode_from_string_and_size: symbol(
                    &library,
                    b"PyUnicode_FromStringAndSize\0",
                )?,
                py_unicode_as_utf8_and_size: symbol(&library, b"PyUnicode_AsUTF8AndSize\0")?,
                py_bytes_from_string_and_size: symbol(&library, b"PyBytes_FromStringAndSize\0")?,
                py_bytes_as_string_and_size: symbol(&library, b"PyBytes_AsStringAndSize\0")?,
                py_capsule_new: symbol(&library, b"PyCapsule_New\0")?,
                py_cfunction_new_ex: symbol(&library, b"PyCFunction_NewEx\0")?,
                py_run_string_flags: symbol(&library, b"PyRun_StringFlags\0")?,
                py_err_occurred: symbol(&library, b"PyErr_Occurred\0")?,
                py_err_fetch: symbol(&library, b"PyErr_Fetch\0")?,
                py_err_normalize_exception: symbol(&library, b"PyErr_NormalizeException\0")?,
                py_err_clear: symbol(&library, b"PyErr_Clear\0")?,
                py_err_set_string: symbol(&library, b"PyErr_SetString\0")?,
                py_thread_get_thread_ident: symbol(&library, b"PyThread_get_thread_ident\0")?,
                py_thread_state_set_async_exc: symbol(&library, b"PyThreadState_SetAsyncExc\0")?,
                py_inc_ref: symbol(&library, b"Py_IncRef\0")?,
                py_dec_ref: symbol(&library, b"Py_DecRef\0")?,
                py_decode_locale: symbol(&library, b"Py_DecodeLocale\0")?,
                py_set_program_name: symbol(&library, b"Py_SetProgramName\0")?,
                py_set_python_home: symbol(&library, b"Py_SetPythonHome\0")?,
                initialized: false,
                program_name: std::ptr::null_mut(),
                python_home: std::ptr::null_mut(),
                library_path: path.to_path_buf(),
                library: ManuallyDrop::new(library),
            })
        }
    }

    pub(crate) fn owner_capsule(
        &self,
        owner: std::sync::Arc<dyn PythonBufferOwner>,
    ) -> Result<*mut PyObject, PythonError> {
        let pointer = Box::into_raw(Box::new(owner)).cast::<c_void>();
        // SAFETY: the box remains owned by the capsule until its destructor
        // recovers the same pointer. A null name permits lookup with null.
        let capsule = unsafe {
            (self.py_capsule_new)(pointer, std::ptr::null(), Some(release_owner_capsule))
        };
        if capsule.is_null() {
            // SAFETY: capsule construction failed, so ownership never moved.
            unsafe {
                drop(Box::from_raw(
                    pointer.cast::<std::sync::Arc<dyn PythonBufferOwner>>(),
                ))
            };
            Err(unsafe { capsule_error(self) })
        } else {
            Ok(capsule)
        }
    }

    pub(crate) fn initialize(
        &mut self,
        executable: &Path,
        home: &Path,
    ) -> Result<bool, PythonError> {
        let requested_library = std::fs::canonicalize(&self.library_path).map_err(|error| {
            PythonError::host(
                "PythonConfigurationError",
                format!(
                    "could not resolve CPython library {}: {error}",
                    self.library_path.display()
                ),
            )
        })?;
        let mut process_library = PROCESS_INTERPRETER_LIBRARY.lock().map_err(|_| {
            PythonError::host(
                "PythonInitializationError",
                "the process CPython identity lock was poisoned",
            )
        })?;
        if let Some(active_library) = process_library.as_ref() {
            if active_library != &requested_library {
                return Err(PythonError::host(
                    "PythonConfigurationError",
                    format!(
                        "this RunMat process already hosts CPython from {}; use OutOfProcess mode to select {}",
                        active_library.display(),
                        requested_library.display()
                    ),
                ));
            }
        }
        // SAFETY: initialization occurs once on the interpreter lane before any
        // Python object exists. The decoded wide strings remain owned by this
        // process-resident table for the interpreter lifetime.
        unsafe {
            if (self.py_is_initialized)() != 0 {
                self.initialized = true;
                *process_library = Some(requested_library);
                return Ok(false);
            }
            self.program_name = self.decode_path(executable)?;
            self.python_home = self.decode_path(home)?;
            (self.py_set_program_name)(self.program_name);
            (self.py_set_python_home)(self.python_home);
            (self.py_initialize_ex)(0);
            if (self.py_is_initialized)() == 0 {
                return Err(PythonError::host(
                    "PythonInitializationError",
                    "CPython did not enter an initialized state",
                ));
            }
            self.initialized = true;
            *process_library = Some(requested_library);
        }
        Ok(true)
    }

    unsafe fn decode_path(&self, path: &Path) -> Result<*mut c_void, PythonError> {
        let bytes = path.to_string_lossy();
        let c_path = CString::new(bytes.as_bytes()).map_err(|_| {
            PythonError::host(
                "PythonConfigurationError",
                format!("Python path contains a null byte: {}", path.display()),
            )
        })?;
        let mut size = 0usize;
        // SAFETY: c_path is NUL-terminated and Py_DecodeLocale writes only the
        // reported size value before returning an owned wide string.
        let decoded = unsafe { (self.py_decode_locale)(c_path.as_ptr(), &mut size) };
        if decoded.is_null() {
            Err(PythonError::host(
                "PythonConfigurationError",
                format!("could not decode Python path {}", path.display()),
            ))
        } else {
            Ok(decoded)
        }
    }
}

unsafe extern "C" fn release_owner_capsule(capsule: *mut PyObject) {
    let Some(get_pointer) = CAPSULE_GET_POINTER.get().copied() else {
        return;
    };
    // SAFETY: this destructor is registered only for unnamed owner capsules.
    let pointer = unsafe { get_pointer(capsule, std::ptr::null()) };
    if !pointer.is_null() {
        // SAFETY: owner_capsule created exactly one box at this address.
        unsafe {
            drop(Box::from_raw(
                pointer.cast::<std::sync::Arc<dyn PythonBufferOwner>>(),
            ))
        };
    }
}

unsafe fn capsule_error(api: &CpythonApi) -> PythonError {
    if unsafe { (api.py_err_occurred)() }.is_null() {
        PythonError::host("PythonCapsuleError", "could not create Python buffer owner")
    } else {
        unsafe { (api.py_err_clear)() };
        PythonError::host(
            "PythonCapsuleError",
            "CPython rejected the RunMat buffer owner capsule",
        )
    }
}

impl Drop for CpythonApi {
    fn drop(&mut self) {
        if !self.initialized {
            // SAFETY: no interpreter or Python extension can retain symbols
            // from a library that was never initialized.
            unsafe { ManuallyDrop::drop(&mut self.library) };
        }
    }
}

unsafe fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T, PythonError> {
    // SAFETY: the caller specifies the stable-ABI signature corresponding to
    // this exact symbol name and keeps the library resident.
    unsafe { library.get::<T>(name) }
        .map(|symbol| *symbol)
        .map_err(|error| {
            PythonError::host(
                "PythonLibraryError",
                format!(
                    "CPython library does not export {}: {error}",
                    String::from_utf8_lossy(&name[..name.len().saturating_sub(1)])
                ),
            )
        })
}

fn open_library(path: &Path) -> Result<Library, PythonError> {
    #[cfg(unix)]
    {
        // SAFETY: the path was produced by the selected interpreter. Global
        // visibility is required by native extension modules that resolve the
        // CPython ABI from the embedding process.
        let library = unsafe {
            libloading::os::unix::Library::open(
                Some(path),
                libc_flags::RTLD_NOW | libc_flags::RTLD_GLOBAL,
            )
        }
        .map(Library::from);
        library.map_err(|error| library_error(path, error))
    }
    #[cfg(not(unix))]
    {
        // SAFETY: the selected interpreter reported this exact library path.
        unsafe { Library::new(path) }.map_err(|error| library_error(path, error))
    }
}

fn library_error(path: &Path, error: libloading::Error) -> PythonError {
    PythonError::host(
        "PythonLibraryError",
        format!("could not load {}: {error}", path.display()),
    )
}

#[cfg(unix)]
mod libc_flags {
    pub const RTLD_NOW: i32 = 0x2;
    #[cfg(target_os = "macos")]
    pub const RTLD_GLOBAL: i32 = 0x8;
    #[cfg(not(target_os = "macos"))]
    pub const RTLD_GLOBAL: i32 = 0x100;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{discover_python, PythonDiscoveryRequest};

    #[test]
    fn selected_interpreter_exports_the_required_stable_abi() {
        let Ok(installation) = discover_python(&PythonDiscoveryRequest::default()) else {
            return;
        };
        let api = CpythonApi::load(&installation.library).expect("load CPython stable ABI");
        // SAFETY: querying initialization state has no precondition.
        assert!(matches!(unsafe { (api.py_is_initialized)() }, 0 | 1));
    }
}
