use std::ffi::CString;
use std::path::{Path, PathBuf};

use super::api::{PyObject, PySsize};
use super::interpreter::Interpreter;
use crate::PythonError;

impl Interpreter {
    pub(crate) fn admit_managed_imports(
        &self,
        names: &std::collections::BTreeSet<String>,
    ) -> Result<(), PythonError> {
        self.with_gil(|this| {
            if let Some(present) = this.remaining_managed_import_with_gil(&[], names) {
                return Err(PythonError::host(
                    "PythonImportOwnerError",
                    format!(
                        "Python namespace `{present}` is already present in the shared interpreter"
                    ),
                ));
            }
            Ok(())
        })
    }

    pub(super) fn remaining_managed_import_with_gil(
        &self,
        roots: &[PathBuf],
        names: &std::collections::BTreeSet<String>,
    ) -> Option<String> {
        if roots.is_empty() && names.is_empty() {
            return None;
        }
        let modules = unsafe { (self.api.py_import_get_module_dict)() };
        let mut position: PySsize = 0;
        loop {
            let mut key: *mut PyObject = std::ptr::null_mut();
            let mut module: *mut PyObject = std::ptr::null_mut();
            if unsafe { (self.api.py_dict_next)(modules, &mut position, &mut key, &mut module) }
                == 0
            {
                return None;
            }
            let namespace_match = {
                let Some(present) = (unsafe { self.borrowed_static_unicode(key) }) else {
                    continue;
                };
                names.iter().any(|name| {
                    present == name.as_str()
                        || present
                            .strip_prefix(name.as_str())
                            .is_some_and(|suffix| suffix.starts_with('.'))
                })
            };
            if namespace_match {
                return unsafe { self.borrowed_static_unicode(key) }.map(str::to_owned);
            }
            if roots.is_empty() {
                continue;
            }
            let owned_name = if self.module_file_is_within_roots(module, roots) {
                unsafe { self.borrowed_static_unicode(key) }.map(str::to_owned)
            } else {
                None
            };
            if owned_name.is_some() {
                return owned_name;
            }
        }
    }

    /// Called on the lane while holding the GIL. Only static module-dictionary
    /// metadata is inspected; module attributes and namespace path iterators
    /// may execute arbitrary Python and are intentionally not consulted.
    pub(super) fn capture_owned_imports_with_gil(&self, roots: &[PathBuf]) {
        if roots.is_empty() {
            return;
        }
        let modules = unsafe { (self.api.py_import_get_module_dict)() };
        let mut position: PySsize = 0;
        let mut candidates = Vec::new();
        loop {
            let mut key: *mut PyObject = std::ptr::null_mut();
            let mut module: *mut PyObject = std::ptr::null_mut();
            let found =
                unsafe { (self.api.py_dict_next)(modules, &mut position, &mut key, &mut module) };
            if found == 0 {
                break;
            }
            if (unsafe { self.borrowed_static_unicode(key) }).is_none() {
                continue;
            }
            let candidate = if self.module_file_is_within_roots(module, roots) {
                unsafe { self.borrowed_static_unicode(key) }.and_then(|name| {
                    self.owned_imports
                        .borrow()
                        .get(name)
                        .is_none_or(|old| *old != module)
                        .then(|| name.to_owned())
                })
            } else {
                None
            };
            if let Some(name) = candidate {
                unsafe { (self.api.py_inc_ref)(module) };
                candidates.push((name, module));
            }
        }
        let mut owned = self.owned_imports.borrow_mut();
        let mut replaced = Vec::new();
        for (name, module) in candidates {
            if let Some(previous) = owned.insert(name, module) {
                replaced.push(previous);
            }
        }
        drop(owned);
        for previous in replaced {
            unsafe { (self.api.py_dec_ref)(previous) };
        }
    }

    pub(super) fn clear_owned_imports_with_gil(&self) -> bool {
        let modules = unsafe { (self.api.py_import_get_module_dict)() };
        let mut clean = true;
        let owned = std::mem::take(&mut *self.owned_imports.borrow_mut());
        for (name, original) in owned {
            if let Ok(key) = CString::new(name) {
                let current = unsafe { (self.api.py_dict_get_item_string)(modules, key.as_ptr()) };
                if current == original {
                    if unsafe { (self.api.py_dict_del_item_string)(modules, key.as_ptr()) } != 0 {
                        unsafe { (self.api.py_err_clear)() };
                        clean = false;
                    }
                } else if !current.is_null() {
                    // A different module now occupies the same name. We must
                    // retain this generation's claim instead of certifying
                    // cleanup from its original pointer alone.
                    clean = false;
                }
            } else {
                clean = false;
            }
            unsafe { (self.api.py_dec_ref)(original) };
        }
        clean
    }

    pub(super) fn clear_module_paths_with_gil(&self) -> bool {
        let list = self.module_path_list.borrow_mut().take();
        let Some(list) = list else {
            return true;
        };
        let inserted = std::mem::take(&mut *self.inserted_module_paths.borrow_mut());
        let sys = unsafe { (self.api.py_import_import_module)(c"sys".as_ptr()) };
        let live_list = if sys.is_null() {
            unsafe { (self.api.py_err_clear)() };
            std::ptr::null_mut()
        } else {
            let dictionary = unsafe { (self.api.py_module_get_dict)(sys) };
            let current = if dictionary.is_null() {
                unsafe { (self.api.py_err_clear)() };
                std::ptr::null_mut()
            } else {
                unsafe { (self.api.py_dict_get_item_string)(dictionary, c"path".as_ptr()) }
            };
            unsafe { (self.api.py_dec_ref)(sys) };
            current
        };
        if live_list != list {
            for original in inserted {
                unsafe { (self.api.py_dec_ref)(original) };
            }
            unsafe { (self.api.py_dec_ref)(list) };
            return false;
        }
        let mut clean = true;
        for original in inserted {
            let count = unsafe { (self.api.py_list_size)(list) };
            if count < 0 {
                unsafe { (self.api.py_err_clear)() };
                clean = false;
            } else {
                let mut found = false;
                for index in (0..count).rev() {
                    if unsafe { (self.api.py_list_get_item)(list, index) } == original {
                        found = true;
                        if unsafe {
                            (self.api.py_list_set_slice)(
                                list,
                                index,
                                index + 1,
                                std::ptr::null_mut(),
                            )
                        } != 0
                        {
                            unsafe { (self.api.py_err_clear)() };
                            clean = false;
                        }
                        break;
                    }
                }
                if !found {
                    clean = false;
                }
            }
            unsafe { (self.api.py_dec_ref)(original) };
        }
        unsafe { (self.api.py_dec_ref)(list) };
        clean
    }

    fn module_file_is_within_roots(&self, module: *mut PyObject, roots: &[PathBuf]) -> bool {
        let module_dict = unsafe { (self.api.py_module_get_dict)(module) };
        if module_dict.is_null() {
            unsafe { (self.api.py_err_clear)() };
            return false;
        }
        // A hashed lookup may compare against an arbitrary colliding key and
        // execute Python while the outer sys.modules PyDict_Next is in flight.
        // Walk only static dictionary entries here; Unicode inspection and
        // PyDict_Next do not invoke key equality or module attributes.
        let mut position: PySsize = 0;
        loop {
            let mut key: *mut PyObject = std::ptr::null_mut();
            let mut value: *mut PyObject = std::ptr::null_mut();
            if unsafe { (self.api.py_dict_next)(module_dict, &mut position, &mut key, &mut value) }
                == 0
            {
                return false;
            }
            let key_length = unsafe { (self.api.py_unicode_get_length)(key) };
            if key_length < 0 {
                unsafe { (self.api.py_err_clear)() };
                continue;
            }
            if key_length != 8
                || unsafe {
                    (self.api.py_unicode_compare_with_ascii_string)(key, c"__file__".as_ptr())
                } != 0
            {
                continue;
            }
            return (unsafe { self.borrowed_static_unicode(value) }).is_some_and(|filename| {
                roots
                    .iter()
                    .any(|root| Path::new(filename).starts_with(root))
            });
        }
    }

    /// # Safety
    /// The caller holds the GIL and keeps `object` alive until the returned
    /// view is consumed. It must not retain the view across code that can
    /// mutate the owning Python dictionary or invoke arbitrary Python code.
    /// `PyUnicode_AsUTF8AndSize` exposes the object's cached UTF-8 storage;
    /// this method never transfers ownership of that storage to Rust.
    unsafe fn borrowed_static_unicode<'gil>(&self, object: *mut PyObject) -> Option<&'gil str> {
        if object.is_null() {
            return None;
        }
        let mut length: PySsize = 0;
        let pointer = unsafe { (self.api.py_unicode_as_utf8_and_size)(object, &mut length) };
        if pointer.is_null() {
            unsafe { (self.api.py_err_clear)() };
            return None;
        }
        let bytes = unsafe { std::slice::from_raw_parts(pointer.cast::<u8>(), length as usize) };
        std::str::from_utf8(bytes).ok()
    }
}
