use std::path::PathBuf;
use std::rc::Rc;

use runmat_mex::{MexDiagnostic, MexHostServices, MexModule};
use runmat_runtime::{build_runtime_error, RuntimeError};
use runmat_value::Value;

use super::*;

pub(super) fn clear_modules(
    modules: &std::cell::RefCell<HashMap<PathBuf, Rc<MexModule>>>,
    request: &runmat_runtime::user_functions::DynamicFunctionClearRequest,
) -> Result<(), RuntimeError> {
    let mut first_error = None;
    modules.borrow_mut().retain(|path, module| {
        let selected = match request {
            runmat_runtime::user_functions::DynamicFunctionClearRequest::All
            | runmat_runtime::user_functions::DynamicFunctionClearRequest::NativeExtensions => true,
            runmat_runtime::user_functions::DynamicFunctionClearRequest::Named(name) => {
                path_matches_clear_name(path, name)
            }
        };
        if !selected {
            return true;
        }
        match module.clear() {
            Ok(cleared) => !cleared,
            Err(error) => {
                if first_error.is_none() {
                    first_error = Some(runtime_error(
                        "MexClear",
                        format!("Could not clear MEX function '{}': {error}", path.display()),
                    ));
                }
                true
            }
        }
    });
    match first_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

pub(super) fn path_matches_clear_name(path: &std::path::Path, name: &str) -> bool {
    let normalized = name.replace(['/', '\\'], std::path::MAIN_SEPARATOR_STR);
    let requested = std::path::Path::new(&normalized);
    let requested = requested.with_extension("");
    if requested.components().count() > 1 {
        return path.with_extension("").ends_with(requested);
    }
    let Some(candidate) = path.file_stem().and_then(|stem| stem.to_str()) else {
        return false;
    };
    let Some(requested) = requested.file_name().and_then(|stem| stem.to_str()) else {
        return false;
    };
    if cfg!(target_os = "windows") {
        candidate.eq_ignore_ascii_case(requested)
    } else {
        candidate == requested
    }
}

pub(super) async fn load_and_call(
    name: &str,
    arguments: Vec<Value>,
    requested_outputs: usize,
    modules: Rc<std::cell::RefCell<HashMap<PathBuf, Rc<MexModule>>>>,
) -> Option<Result<Value, RuntimeError>> {
    let extension = format!(".{}", runmat_mex::mex_suffix()?);
    let path = match runmat_runtime::builtins::common::path_search::find_file_with_extensions(
        name,
        &[extension.as_str()],
        "MEX function resolution",
    )
    .await
    {
        Ok(Some(path)) => path,
        Ok(None) => return None,
        Err(error) => return Some(Err(runtime_error("FunctionResolution", error))),
    };
    let canonical = runmat_filesystem::canonicalize_async(&path)
        .await
        .unwrap_or(path);
    let cached_module = {
        let modules = modules.borrow();
        modules.get(&canonical).cloned()
    };
    let module = if let Some(module) = cached_module {
        module
    } else {
        let module = match MexModule::load(&canonical) {
            Ok(module) => Rc::new(module),
            Err(error) => {
                return Some(Err(runtime_error(
                    "MexLoad",
                    format!(
                        "Could not load MEX function '{}': {error}",
                        canonical.display()
                    ),
                )))
            }
        };
        modules
            .borrow_mut()
            .insert(canonical.clone(), Rc::clone(&module));
        module
    };
    let result = module.invoke_with_services(
        &arguments,
        requested_outputs,
        module.api_mode(),
        Rc::new(RuntimeMexHostServices),
    );
    Some(match result {
        Ok(invocation) => {
            if !invocation.console.is_empty() {
                runmat_runtime::console::record_console_output(
                    runmat_runtime::console::ConsoleStream::Stdout,
                    invocation.console,
                );
            }
            for warning in invocation.warnings {
                runmat_runtime::warning_store::push(
                    warning
                        .identifier
                        .as_deref()
                        .unwrap_or("RunMat:MEX:Warning"),
                    &warning.message,
                );
            }
            Ok(if requested_outputs == 1 {
                invocation
                    .outputs
                    .into_iter()
                    .next()
                    .unwrap_or_else(|| Value::OutputList(Vec::new()))
            } else {
                Value::OutputList(invocation.outputs)
            })
        }
        Err(error) => Err(runtime_error("MexInvocation", error.to_string())),
    })
}

struct RuntimeMexHostServices;

impl MexHostServices for RuntimeMexHostServices {
    fn eval(&self, command: &str) -> Result<(), MexDiagnostic> {
        pollster::block_on(runmat_runtime::call_builtin_async_with_outputs(
            "eval",
            &[Value::String(command.to_string())],
            0,
        ))
        .map(|_| ())
        .map_err(runtime_diagnostic)
    }

    fn call(
        &self,
        function: &str,
        arguments: Vec<Value>,
        requested_outputs: usize,
    ) -> Result<Vec<Value>, MexDiagnostic> {
        let value = pollster::block_on(runmat_runtime::call_feval_async_with_outputs(
            Value::from(function),
            &arguments,
            requested_outputs,
        ))
        .map_err(runtime_diagnostic)?;
        match (requested_outputs, value) {
            (0, _) => Ok(Vec::new()),
            (_, Value::OutputList(values)) => Ok(values),
            (1, value) => Ok(vec![value]),
            (count, _) => Err(MexDiagnostic {
                identifier: Some("RunMat:MEX:CallbackOutputs".into()),
                message: format!("callback did not produce the requested {count} outputs"),
            }),
        }
    }

    fn get_variable(&self, workspace: &str, name: &str) -> Result<Option<Value>, MexDiagnostic> {
        validate_workspace(workspace)?;
        if workspace == "global" {
            Ok(runmat_runtime::workspace::session::global_value(name))
        } else {
            Ok(runmat_runtime::workspace::lookup(name))
        }
    }

    fn put_variable(&self, workspace: &str, name: &str, value: Value) -> Result<(), MexDiagnostic> {
        validate_workspace(workspace)?;
        if workspace == "global" {
            runmat_runtime::workspace::session::store_global_named(name, value);
            Ok(())
        } else {
            runmat_runtime::workspace::assign(name, value).map_err(|message| MexDiagnostic {
                identifier: Some("RunMat:MEX:Workspace".into()),
                message,
            })
        }
    }
}

fn validate_workspace(workspace: &str) -> Result<(), MexDiagnostic> {
    if matches!(workspace, "base" | "caller" | "global") {
        Ok(())
    } else {
        Err(MexDiagnostic {
            identifier: Some("RunMat:MEX:Workspace".into()),
            message: format!("unknown MEX workspace '{workspace}'"),
        })
    }
}

fn runtime_diagnostic(error: RuntimeError) -> MexDiagnostic {
    MexDiagnostic {
        identifier: error.identifier().map(str::to_string),
        message: error.message().to_string(),
    }
}

fn runtime_error(identifier: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier(format!("RunMat:{identifier}"))
        .build()
}
