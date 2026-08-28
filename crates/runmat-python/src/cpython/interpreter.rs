use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::ffi::c_int;
use std::rc::{Rc, Weak};
use std::sync::mpsc;

use super::api::{CpythonApi, PyObject, PySsize};
use super::support::{attr, capture_error, checked, set_dict_string, unicode};
use crate::{
    PythonError, PythonInstallation, PythonObjectHandle, PythonObjectMetadata, PythonValue,
};

pub(super) const PY_FILE_INPUT: c_int = 257;

pub(super) struct BuiltinTypes {
    pub(super) none: *mut PyObject,
    pub(super) bool_: *mut PyObject,
    pub(super) int: *mut PyObject,
    pub(super) float: *mut PyObject,
    pub(super) complex: *mut PyObject,
    pub(super) str_: *mut PyObject,
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
    pub(super) active_callbacks: RefCell<Option<mpsc::SyncSender<crate::object::PythonLaneEvent>>>,
    self_reference: RefCell<Weak<Interpreter>>,
}

impl Interpreter {
    pub(crate) fn start(installation: &PythonInstallation) -> Result<Rc<Self>, PythonError> {
        let mut api = CpythonApi::load(&installation.library)?;
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
                builtins,
                workspace,
                objects: RefCell::new(BTreeMap::new()),
                next_handle: Cell::new(1),
                active_callbacks: RefCell::new(None),
                self_reference: RefCell::new(Weak::new()),
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
        paths: &[std::path::PathBuf],
    ) -> Result<(), PythonError> {
        self.with_gil(|this| {
            let sys = unsafe {
                checked(
                    &this.api,
                    (this.api.py_import_import_module)(c"sys".as_ptr()),
                    "import sys",
                )?
            };
            let path_list = unsafe { attr(&this.api, sys, "path") };
            this.decref(sys);
            let path_list = path_list?;
            let insert = unsafe { attr(&this.api, path_list, "insert") };
            this.decref(path_list);
            let insert = insert?;
            for path in paths.iter().rev() {
                let arguments = this.values_to_tuple(vec![
                    PythonValue::Signed(0),
                    PythonValue::String(path.to_string_lossy().into_owned()),
                ])?;
                let result =
                    unsafe { (this.api.py_object_call)(insert, arguments, std::ptr::null_mut()) };
                this.decref(arguments);
                let result = unsafe { checked(&this.api, result, "prepend Python module path")? };
                this.decref(result);
            }
            this.decref(insert);
            Ok(())
        })
    }

    pub(crate) fn clear(&self) {
        let state = unsafe { (self.api.py_gil_state_ensure)() };
        for (_, object) in std::mem::take(&mut *self.objects.borrow_mut()) {
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
            (self.api.py_dec_ref)(self.builtins.bytes);
            (self.api.py_dec_ref)(self.builtins.datetime);
            (self.api.py_dec_ref)(self.builtins.timedelta);
            (self.api.py_dec_ref)(self.builtins.type_);
            (self.api.py_dec_ref)(self.builtins.runtime_error);
            (self.api.py_dec_ref)(self.builtins.keyboard_interrupt);
            (self.api.py_dec_ref)(self.builtins.buffer_view);
            (self.api.py_dec_ref)(self.builtins.module);
            (self.api.py_gil_state_release)(state);
        }
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
        events: mpsc::SyncSender<crate::object::PythonLaneEvent>,
        operation: impl FnOnce() -> T,
    ) -> T {
        let previous = self.active_callbacks.replace(Some(events));
        let result = operation();
        self.active_callbacks.replace(previous);
        result
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
        self.decref(type_object);
        Ok(PythonObjectMetadata {
            handle,
            type_name,
            module,
            callable: unsafe { (self.api.py_callable_check)(object) != 0 },
            iterable: unsafe {
                let iterator = (self.api.py_object_get_iter)(object);
                if iterator.is_null() {
                    (self.api.py_err_clear)();
                    false
                } else {
                    (self.api.py_dec_ref)(iterator);
                    true
                }
            },
            mapping: unsafe { (self.api.py_mapping_check)(object) != 0 },
            sequence: unsafe { (self.api.py_sequence_check)(object) != 0 },
            none: object == self.builtins.none,
        })
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
