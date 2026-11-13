use std::fs;
use std::path::Path;

use super::api::PyObject;
use super::interpreter::{Interpreter, PY_FILE_INPUT};
use super::support::{attr, c_string, checked, set_dict_string};
use crate::{PythonError, PythonValue};

impl Interpreter {
    pub(super) fn execute(
        &self,
        code: &str,
        inputs: Vec<(String, PythonValue)>,
        outputs: Vec<String>,
        workspace: *mut PyObject,
    ) -> Result<Vec<PythonValue>, PythonError> {
        for (name, value) in inputs {
            let value = self.to_owned(value)?;
            let result = unsafe { set_dict_string(&self.api, workspace, &name, value) };
            self.decref(value);
            result?;
        }
        let code = c_string(code, "Python source")?;
        // SAFETY: source and workspace are valid and the workspace is used as
        // both globals and locals, matching Python's module execution model.
        let result = unsafe {
            (self.api.py_run_string_flags)(
                code.as_ptr(),
                PY_FILE_INPUT,
                workspace,
                workspace,
                std::ptr::null_mut(),
            )
        };
        let result = unsafe { checked(&self.api, result, "execute Python source")? };
        self.decref(result);
        self.read_outputs(workspace, outputs)
    }

    pub(super) fn execute_file(
        &self,
        path: &Path,
        arguments: Vec<String>,
        inputs: Vec<(String, PythonValue)>,
        outputs: Vec<String>,
    ) -> Result<Vec<PythonValue>, PythonError> {
        let code = fs::read_to_string(path).map_err(|error| {
            PythonError::host(
                "PythonFileError",
                format!("could not read {}: {error}", path.display()),
            )
        })?;
        // SAFETY: constructors return owned references or set an exception.
        let workspace = unsafe {
            checked(
                &self.api,
                (self.api.py_dict_new)(),
                "create script workspace",
            )?
        };
        let setup =
            unsafe { set_dict_string(&self.api, workspace, "__builtins__", self.builtins.module) };
        if let Err(error) = setup {
            self.decref(workspace);
            return Err(error);
        }
        let file_value = self.to_owned(PythonValue::String(path.to_string_lossy().into_owned()))?;
        let file_result = unsafe { set_dict_string(&self.api, workspace, "__file__", file_value) };
        self.decref(file_value);
        if let Err(error) = file_result {
            self.decref(workspace);
            return Err(error);
        }

        let sys = match unsafe {
            checked(
                &self.api,
                (self.api.py_import_import_module)(c"sys".as_ptr()),
                "import sys",
            )
        } {
            Ok(sys) => sys,
            Err(error) => {
                self.decref(workspace);
                return Err(error);
            }
        };
        let previous_argv = match unsafe { attr(&self.api, sys, "argv") } {
            Ok(previous_argv) => previous_argv,
            Err(error) => {
                self.decref(sys);
                self.decref(workspace);
                return Err(error);
            }
        };
        let mut argv = Vec::with_capacity(arguments.len() + 1);
        argv.push(PythonValue::String(path.to_string_lossy().into_owned()));
        argv.extend(arguments.into_iter().map(PythonValue::String));
        let argv = match self.to_owned(PythonValue::List(argv)) {
            Ok(argv) => argv,
            Err(error) => {
                self.decref(previous_argv);
                self.decref(sys);
                self.decref(workspace);
                return Err(error);
            }
        };
        let argv_name = c"argv";
        let argv_status =
            unsafe { (self.api.py_object_set_attr_string)(sys, argv_name.as_ptr(), argv) };
        self.decref(argv);
        if let Err(error) = self.status(argv_status, "set Python script arguments") {
            self.decref(previous_argv);
            self.decref(sys);
            self.decref(workspace);
            return Err(error);
        }

        let result = self.execute(&code, inputs, outputs, workspace);
        // SAFETY: restoring the saved owned argv reference cannot borrow from
        // the temporary script workspace.
        let restore_status =
            unsafe { (self.api.py_object_set_attr_string)(sys, argv_name.as_ptr(), previous_argv) };
        self.decref(previous_argv);
        self.decref(sys);
        self.decref(workspace);
        self.status(restore_status, "restore Python script arguments")?;
        result
    }

    pub(super) fn read_outputs(
        &self,
        workspace: *mut PyObject,
        outputs: Vec<String>,
    ) -> Result<Vec<PythonValue>, PythonError> {
        outputs
            .into_iter()
            .map(|name| {
                let key = self.to_owned(PythonValue::String(name.clone()))?;
                // SAFETY: workspace and key are live references.
                let value = unsafe { (self.api.py_object_get_item)(workspace, key) };
                self.decref(key);
                let value =
                    unsafe { checked(&self.api, value, &format!("read Python output {name}"))? };
                self.convert_owned(value)
            })
            .collect()
    }

    pub(super) fn resolve_qualified(&self, name: &str) -> Result<*mut PyObject, PythonError> {
        let name = name.strip_prefix("py.").unwrap_or(name);
        let segments = name.split('.').collect::<Vec<_>>();
        if segments.is_empty() || segments.iter().any(|segment| segment.is_empty()) {
            return Err(PythonError::host(
                "PythonNameError",
                format!("invalid Python qualified name {name:?}"),
            ));
        }
        if segments.len() == 1 {
            if let Ok(object) = unsafe { attr(&self.api, self.builtins.module, segments[0]) } {
                return Ok(object);
            }
            unsafe { (self.api.py_err_clear)() };
        }
        for module_len in (1..=segments.len()).rev() {
            let module_name = segments[..module_len].join(".");
            let module_name = c_string(&module_name, "Python module name")?;
            // SAFETY: module_name is a valid NUL-terminated string.
            let module = unsafe { (self.api.py_import_import_module)(module_name.as_ptr()) };
            if module.is_null() {
                // Import resolution intentionally probes progressively shorter
                // module prefixes. Only the final failure is user-visible.
                unsafe { (self.api.py_err_clear)() };
                continue;
            }
            let mut current = module;
            let mut failed = None;
            for segment in &segments[module_len..] {
                match unsafe { attr(&self.api, current, segment) } {
                    Ok(next) => {
                        self.decref(current);
                        current = next;
                    }
                    Err(error) => {
                        self.decref(current);
                        failed = Some(error);
                        break;
                    }
                }
            }
            return failed.map_or(Ok(current), Err);
        }
        Err(PythonError::host(
            "PythonImportError",
            format!("could not resolve Python name {name}"),
        ))
    }

    pub(super) fn call_object(
        &self,
        callable: *mut PyObject,
        mut arguments: Vec<PythonValue>,
    ) -> Result<PythonValue, PythonError> {
        // SAFETY: callable is a live reference on the interpreter lane.
        if unsafe { (self.api.py_callable_check)(callable) } == 0 {
            return Err(PythonError::host(
                "PythonTypeError",
                "resolved Python object is not callable",
            ));
        }
        let keywords = match arguments.last() {
            Some(PythonValue::Keywords(_)) => match arguments.pop() {
                Some(PythonValue::Keywords(values)) => values,
                _ => unreachable!(),
            },
            _ => Vec::new(),
        };
        let positional = self.values_to_tuple(arguments)?;
        let keyword_dict = self.keywords_to_dict(keywords)?;
        // SAFETY: call arguments are live owned containers. A null keyword
        // pointer is the canonical no-keywords representation.
        let result = unsafe {
            (self.api.py_object_call)(
                callable,
                positional,
                keyword_dict.unwrap_or(std::ptr::null_mut()),
            )
        };
        self.decref(positional);
        if let Some(keyword_dict) = keyword_dict {
            self.decref(keyword_dict);
        }
        let result = unsafe { checked(&self.api, result, "call Python object")? };
        self.convert_owned(result)
    }
}
