use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::ffi::{c_int, CStr};
use std::path::PathBuf;
use std::rc::{Rc, Weak};

use runmat_process_host::private_bytes::PrivateByteLease;

use super::api::{CpythonApi, PyObject, PySsize};
use super::support::{attr, capture_error, checked, set_dict_string, unicode};
use crate::{
    PythonError, PythonInstallation, PythonObjectHandle, PythonObjectMetadata, PythonValue,
};

pub(super) const PY_FILE_INPUT: c_int = 257;
const PY_TPFLAGS_HEAPTYPE: std::ffi::c_ulong = 1 << 9;
const PY_SQ_ITEM: c_int = 44;
const PY_TP_ITER: c_int = 62;

pub(super) struct BuiltinTypes {
    pub(super) none: *mut PyObject,
    pub(super) bool_: *mut PyObject,
    pub(super) int: *mut PyObject,
    pub(super) float: *mut PyObject,
    pub(super) complex: *mut PyObject,
    pub(super) str_: *mut PyObject,
    pub(super) list: *mut PyObject,
    pub(super) bytes: *mut PyObject,
    pub(super) datetime: *mut PyObject,
    pub(super) timedelta: *mut PyObject,
    pub(super) type_: *mut PyObject,
    pub(super) runtime_error: *mut PyObject,
    pub(super) keyboard_interrupt: *mut PyObject,
    pub(super) module: *mut PyObject,
    pub(super) buffer_view: *mut PyObject,
}

pub(crate) struct Interpreter {
    pub(super) api: CpythonApi,
    pub(super) builtins: BuiltinTypes,
    pub(super) workspace: *mut PyObject,
    pub(super) objects: RefCell<BTreeMap<PythonObjectHandle, *mut PyObject>>,
    pub(super) next_handle: Cell<u64>,
    pub(super) active_callbacks: RefCell<Option<crate::session::protocol::PythonLaneEventSink>>,
    self_reference: RefCell<Weak<Interpreter>>,
    pub(super) owned_imports: RefCell<BTreeMap<String, *mut PyObject>>,
    pub(super) module_path_list: RefCell<Option<*mut PyObject>>,
    pub(super) inserted_module_paths: RefCell<Vec<*mut PyObject>>,
    managed_exposure_unverified: Cell<bool>,
    all_type_slots_inspectable: bool,
}

/// A nested callback command temporarily replaces the selected event sink.
/// Restore the suspended sink on ordinary return and unwinding alike.
struct CallbackEventScope<'a> {
    active: &'a RefCell<Option<crate::session::protocol::PythonLaneEventSink>>,
    previous: Option<crate::session::protocol::PythonLaneEventSink>,
}

impl<'a> CallbackEventScope<'a> {
    fn install(
        active: &'a RefCell<Option<crate::session::protocol::PythonLaneEventSink>>,
        selected: Option<crate::session::protocol::PythonLaneEventSink>,
    ) -> Self {
        let previous = active.replace(selected);
        Self { active, previous }
    }
}

impl Drop for CallbackEventScope<'_> {
    fn drop(&mut self) {
        self.active.replace(self.previous.take());
    }
}

impl Interpreter {
    pub(crate) fn start(installation: &PythonInstallation) -> Result<Rc<Self>, PythonError> {
        let mut api = CpythonApi::load(&installation.library)?;
        // Read the loaded library's version, not a caller-supplied discovery
        // record, before deciding whether static type slots are inspectable.
        let reported_version = unsafe { CStr::from_ptr((api.py_get_version)()) }
            .to_str()
            .map_err(|_| {
                PythonError::host("PythonConfigurationError", "CPython version is not UTF-8")
            })?;
        let mut version_parts = reported_version.split('.');
        let major = version_parts
            .next()
            .and_then(|part| part.parse::<u16>().ok());
        let minor = version_parts.next().and_then(|part| {
            let digits: String = part
                .chars()
                .take_while(|character| character.is_ascii_digit())
                .collect();
            digits.parse::<u16>().ok()
        });
        let (Some(major), Some(minor)) = (major, minor) else {
            return Err(PythonError::host(
                "PythonConfigurationError",
                "CPython version could not be identified",
            ));
        };
        let initialized_here = api.initialize(&installation.executable, &installation.home)?;
        // SAFETY: this is the interpreter lane. It owns the initialization
        // thread state and releases the GIL before entering its request loop.
        unsafe {
            let state = (api.py_gil_state_ensure)();
            let builtins_module = checked(
                &api,
                (api.py_import_import_module)(c"builtins".as_ptr()),
                "import builtins",
            )?;
            let builtins = BuiltinTypes {
                none: attr(&api, builtins_module, "None")?,
                bool_: attr(&api, builtins_module, "bool")?,
                int: attr(&api, builtins_module, "int")?,
                float: attr(&api, builtins_module, "float")?,
                complex: attr(&api, builtins_module, "complex")?,
                str_: attr(&api, builtins_module, "str")?,
                list: attr(&api, builtins_module, "list")?,
                bytes: attr(&api, builtins_module, "bytes")?,
                datetime: std::ptr::null_mut(),
                timedelta: std::ptr::null_mut(),
                type_: attr(&api, builtins_module, "type")?,
                runtime_error: attr(&api, builtins_module, "RuntimeError")?,
                keyboard_interrupt: attr(&api, builtins_module, "KeyboardInterrupt")?,
                module: builtins_module,
                buffer_view: std::ptr::null_mut(),
            };
            let datetime_module = checked(
                &api,
                (api.py_import_import_module)(c"datetime".as_ptr()),
                "import datetime",
            )?;
            let datetime = attr(&api, datetime_module, "datetime")?;
            let timedelta = attr(&api, datetime_module, "timedelta")?;
            (api.py_dec_ref)(datetime_module);
            let workspace = checked(&api, (api.py_dict_new)(), "create Python workspace")?;
            set_dict_string(&api, workspace, "__builtins__", builtins.module)?;
            let support = checked(&api, (api.py_dict_new)(), "create Python support workspace")?;
            set_dict_string(&api, support, "__builtins__", builtins.module)?;
            let bootstrap = c"class _RunMatBufferView:\n    def __init__(self, interface, owner):\n        self.__array_interface__ = interface\n        self.__runmat_owner = owner\n";
            let bootstrap_result = checked(
                &api,
                (api.py_run_string_flags)(
                    bootstrap.as_ptr(),
                    PY_FILE_INPUT,
                    support,
                    support,
                    std::ptr::null_mut(),
                ),
                "initialize Python buffer support",
            )?;
            (api.py_dec_ref)(bootstrap_result);
            let buffer_view_key = (api.py_unicode_from_string_and_size)(
                c"_RunMatBufferView".as_ptr(),
                "_RunMatBufferView".len() as PySsize,
            );
            let buffer_view_key = checked(&api, buffer_view_key, "create support name")?;
            let buffer_view = checked(
                &api,
                (api.py_object_get_item)(support, buffer_view_key),
                "load Python buffer support",
            )?;
            (api.py_dec_ref)(buffer_view_key);
            (api.py_dec_ref)(support);
            let mut builtins = builtins;
            builtins.datetime = datetime;
            builtins.timedelta = timedelta;
            builtins.buffer_view = buffer_view;
            (api.py_gil_state_release)(state);
            if initialized_here {
                let _thread_state = (api.py_eval_save_thread)();
            }
            let interpreter = Rc::new(Self {
                api,
                all_type_slots_inspectable: (major, minor) >= (3, 10),
                builtins,
                workspace,
                objects: RefCell::new(BTreeMap::new()),
                next_handle: Cell::new(1),
                active_callbacks: RefCell::new(None),
                self_reference: RefCell::new(Weak::new()),
                owned_imports: RefCell::new(BTreeMap::new()),
                module_path_list: RefCell::new(None),
                inserted_module_paths: RefCell::new(Vec::new()),
                managed_exposure_unverified: Cell::new(false),
            });
            *interpreter.self_reference.borrow_mut() = Rc::downgrade(&interpreter);
            Ok(interpreter)
        }
    }

    pub(crate) fn metadata(
        &self,
        handle: PythonObjectHandle,
    ) -> Result<PythonObjectMetadata, PythonError> {
        self.with_gil(|this| this.metadata_with_gil(handle))
    }

    pub(crate) fn prepend_module_paths(
        &self,
        paths: &[PathBuf],
        managed: Option<(&[PathBuf], &PrivateByteLease)>,
    ) -> Result<(), PythonError> {
        self.with_gil(|this| {
            let sys = unsafe {
                checked(
                    &this.api,
                    (this.api.py_import_import_module)(c"sys".as_ptr()),
                    "import sys",
                )?
            };
            let sys_dict = unsafe { (this.api.py_module_get_dict)(sys) };
            let path_list = if sys_dict.is_null() {
                unsafe { (this.api.py_err_clear)() };
                std::ptr::null_mut()
            } else {
                unsafe { (this.api.py_dict_get_item_string)(sys_dict, c"path".as_ptr()) }
            };
            if !path_list.is_null() {
                unsafe { (this.api.py_inc_ref)(path_list) };
            }
            this.decref(sys);
            if path_list.is_null() {
                return Err(PythonError::host(
                    "PythonImportOwnerError",
                    "process sys.path is unavailable",
                ));
            }
            let path_type = unsafe { (this.api.py_object_type)(path_list) };
            let path_type = match unsafe {
                checked(&this.api, path_type, "inspect Python module path list")
            } {
                Ok(path_type) => path_type,
                Err(error) => {
                    this.decref(path_list);
                    return Err(error);
                }
            };
            let exact_list = path_type == this.builtins.list;
            this.decref(path_type);
            if managed.is_some() && !exact_list {
                this.decref(path_list);
                return Err(PythonError::host(
                    "PythonImportOwnerError",
                    "managed wheel imports require the process sys.path to be an exact built-in list",
                ));
            }
            let insert = unsafe { attr(&this.api, path_list, "insert") };
            let insert = match insert {
                Ok(insert) => insert,
                Err(error) => {
                    this.decref(path_list);
                    return Err(error);
                }
            };
            if exact_list {
                *this.module_path_list.borrow_mut() = Some(path_list);
            }
            let result = (|| {
                for path in paths.iter().rev() {
                    let arguments = this.values_to_tuple(vec![
                        PythonValue::Signed(0),
                        PythonValue::String(path.to_string_lossy().into_owned()),
                    ])?;
                    let result = unsafe {
                        (this.api.py_object_call)(insert, arguments, std::ptr::null_mut())
                    };
                    let result =
                        unsafe { checked(&this.api, result, "prepend Python module path") };
                    if result.is_err() {
                        this.decref(arguments);
                        return result.map(|_| ());
                    }
                    let result = result?;
                    this.decref(result);
                    let inserted = unsafe { (this.api.py_tuple_get_item)(arguments, 1) };
                    if exact_list {
                        // The list and tuple each own the item; retain our own
                        // identity so cleanup never removes an equal path.
                        unsafe { (this.api.py_inc_ref)(inserted) };
                        this.inserted_module_paths.borrow_mut().push(inserted);
                    }
                    this.decref(arguments);
                    if let Some((roots, lease)) = managed {
                        if roots.contains(path) {
                            if let Err(error) = this.api.retain_exposed_wheel(lease) {
                                this.managed_exposure_unverified.set(true);
                                return Err(error);
                            }
                        }
                    }
                }
                Ok(())
            })();
            this.decref(insert);
            if !exact_list {
                // Preserve the pre-existing dynamic-list behavior for raw
                // paths; only a built-in list is eligible for C-API cleanup.
                this.decref(path_list);
            }
            result
        })
    }

    pub(crate) fn managed_exposure_unverified(&self) -> bool {
        self.managed_exposure_unverified.get()
    }

    pub(crate) fn clear_with_import_roots(
        &self,
        roots: &[PathBuf],
        names: &std::collections::BTreeSet<String>,
    ) -> bool {
        let state = unsafe { (self.api.py_gil_state_ensure)() };
        self.active_callbacks.borrow_mut().take();
        let objects = std::mem::take(&mut *self.objects.borrow_mut());
        for (_, object) in objects {
            // SAFETY: registry references are owned and released on the lane.
            unsafe { (self.api.py_dec_ref)(object) };
        }
        // SAFETY: these are all owned references created during startup.
        unsafe {
            (self.api.py_dec_ref)(self.workspace);
            (self.api.py_dec_ref)(self.builtins.none);
            (self.api.py_dec_ref)(self.builtins.bool_);
            (self.api.py_dec_ref)(self.builtins.int);
            (self.api.py_dec_ref)(self.builtins.float);
            (self.api.py_dec_ref)(self.builtins.complex);
            (self.api.py_dec_ref)(self.builtins.str_);
            (self.api.py_dec_ref)(self.builtins.list);
            (self.api.py_dec_ref)(self.builtins.bytes);
            (self.api.py_dec_ref)(self.builtins.datetime);
            (self.api.py_dec_ref)(self.builtins.timedelta);
            (self.api.py_dec_ref)(self.builtins.type_);
            (self.api.py_dec_ref)(self.builtins.runtime_error);
            (self.api.py_dec_ref)(self.builtins.keyboard_interrupt);
            (self.api.py_dec_ref)(self.builtins.buffer_view);
            (self.api.py_dec_ref)(self.builtins.module);
        }
        self.capture_owned_imports_with_gil(roots);
        let paths_clean = self.clear_module_paths_with_gil();
        let imports_clean = self.clear_owned_imports_with_gil();
        let remaining = self.remaining_managed_import_with_gil(roots, names);
        unsafe { (self.api.py_gil_state_release)(state) };
        imports_clean && paths_clean && remaining.is_none()
    }

    pub(super) fn with_gil<T>(
        &self,
        operation: impl FnOnce(&Self) -> Result<T, PythonError>,
    ) -> Result<T, PythonError> {
        // SAFETY: only the interpreter lane calls this method. GIL-state APIs
        // attach the lane for the duration of one complete request.
        let state = unsafe { (self.api.py_gil_state_ensure)() };
        let result = operation(self);
        // SAFETY: this exactly balances the ensure above after all temporary
        // Python references created by the operation have been released.
        unsafe { (self.api.py_gil_state_release)(state) };
        result
    }

    pub(crate) fn with_callback_events<T>(
        &self,
        events: crate::session::protocol::PythonLaneEventSink,
        operation: impl FnOnce() -> T,
    ) -> T {
        let _events = CallbackEventScope::install(&self.active_callbacks, Some(events));
        operation()
    }

    /// A synchronous reentry caller cannot drive another callback while it
    /// waits for this lane. Mask the inherited sink so a nested callback
    /// fails instead of waiting for an unserviced event.
    pub(crate) fn without_callback_events<T>(&self, operation: impl FnOnce() -> T) -> T {
        let _events = CallbackEventScope::install(&self.active_callbacks, None);
        operation()
    }

    pub(super) fn weak_reference(&self) -> Weak<Interpreter> {
        self.self_reference.borrow().clone()
    }

    pub(crate) fn interrupt_handle(&self) -> super::api::PythonInterrupt {
        super::api::PythonInterrupt {
            ensure: self.api.py_gil_state_ensure,
            release: self.api.py_gil_state_release,
            set_async_exception: self.api.py_thread_state_set_async_exc,
            thread: unsafe { (self.api.py_thread_get_thread_ident)() },
            exception: self.builtins.keyboard_interrupt as usize,
        }
    }

    pub(super) fn metadata_with_gil(
        &self,
        handle: PythonObjectHandle,
    ) -> Result<PythonObjectMetadata, PythonError> {
        let object = self.object(handle)?;
        let tuple = self.values_to_tuple(vec![PythonValue::Object(handle)])?;
        let type_object = unsafe {
            checked(
                &self.api,
                (self.api.py_object_call)(self.builtins.type_, tuple, std::ptr::null_mut()),
                "inspect Python object type",
            )?
        };
        self.decref(tuple);
        let metadata = (|| {
            let type_name = unsafe {
                let name = attr(&self.api, type_object, "__name__")?;
                let text = unicode(&self.api, name);
                self.decref(name);
                text?
            };
            let module = unsafe {
                let name = attr(&self.api, type_object, "__module__")?;
                let text = unicode(&self.api, name);
                self.decref(name);
                text?
            };
            Ok(PythonObjectMetadata {
                handle,
                type_name,
                module,
                callable: unsafe { (self.api.py_callable_check)(object) != 0 },
                iterable: self.iterator_slot_capability(type_object)?,
                mapping: unsafe { (self.api.py_mapping_check)(object) != 0 },
                sequence: unsafe { (self.api.py_sequence_check)(object) != 0 },
                none: object == self.builtins.none,
            })
        })();
        self.decref(type_object);
        metadata
    }

    fn iterator_slot_capability(
        &self,
        type_object: *mut PyObject,
    ) -> Result<Option<bool>, PythonError> {
        // PyType_GetSlot accepts heap types in 3.9 and all types in 3.10+.
        // Unknown is preferable to calling user __iter__ during metadata.
        let flags = unsafe { (self.api.py_type_get_flags)(type_object) };
        if !self.all_type_slots_inspectable && flags & PY_TPFLAGS_HEAPTYPE == 0 {
            return Ok(None);
        }
        let iter = unsafe { (self.api.py_type_get_slot)(type_object, PY_TP_ITER) };
        if iter.is_null() && !unsafe { (self.api.py_err_occurred)() }.is_null() {
            return Err(unsafe { capture_error(&self.api, "inspect Python iterator slot") });
        }
        let item = unsafe { (self.api.py_type_get_slot)(type_object, PY_SQ_ITEM) };
        if item.is_null() && !unsafe { (self.api.py_err_occurred)() }.is_null() {
            return Err(unsafe { capture_error(&self.api, "inspect Python sequence slot") });
        }
        Ok(Some(!iter.is_null() || !item.is_null()))
    }

    pub(super) fn object(&self, handle: PythonObjectHandle) -> Result<*mut PyObject, PythonError> {
        self.objects.borrow().get(&handle).copied().ok_or_else(|| {
            PythonError::host(
                "PythonStaleObject",
                format!("Python object handle {} is not live", handle.0),
            )
        })
    }

    pub(super) fn is_instance(
        &self,
        object: *mut PyObject,
        type_object: *mut PyObject,
    ) -> Result<bool, PythonError> {
        let result = unsafe { (self.api.py_object_is_instance)(object, type_object) };
        if result < 0 {
            Err(unsafe { capture_error(&self.api, "inspect Python value type") })
        } else {
            Ok(result != 0)
        }
    }

    pub(super) fn status(&self, status: c_int, operation: &str) -> Result<(), PythonError> {
        if status < 0 {
            Err(unsafe { capture_error(&self.api, operation) })
        } else {
            Ok(())
        }
    }

    pub(super) fn incref(&self, object: *mut PyObject) {
        unsafe { (self.api.py_inc_ref)(object) }
    }

    pub(super) fn decref(&self, object: *mut PyObject) {
        unsafe { (self.api.py_dec_ref)(object) }
    }
}
