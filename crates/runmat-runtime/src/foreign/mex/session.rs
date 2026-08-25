use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::Ordering;

use runmat_mex::{MexLoadError, MexModule};
use runmat_value::Value;

use super::RuntimeMexHostServices;
use crate::context::RuntimeContext;
use crate::user_functions::DynamicFunctionClearRequest;
use crate::{build_runtime_error, RuntimeError};

/// Session-scoped owner of dynamically loaded MEX modules.
///
/// Core, JIT callbacks, and standalone AOT execution all use this same path so
/// discovery, manifest admission, persistent state, callback authority, and
/// `mexAtExit` lifecycle behavior cannot drift between execution modes.
#[derive(Default)]
pub struct MexRuntimeSession {
    modules: RefCell<HashMap<PathBuf, Rc<MexModule>>>,
}

impl MexRuntimeSession {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn load_and_call(
        &self,
        name: &str,
        arguments: Vec<Value>,
        requested_outputs: usize,
        runtime: RuntimeContext,
    ) -> Option<Result<Value, RuntimeError>> {
        if runtime.cancellation().load(Ordering::Relaxed) {
            return Some(Err(runtime_error(
                "Cancelled",
                format!("MEX invocation '{name}' was cancelled before entering native code"),
            )));
        }

        let extension = format!(".{}", runmat_mex::mex_suffix()?);
        let path = match crate::builtins::common::path_search::find_file_with_extensions(
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
        let module = match self.module(&canonical) {
            Ok(module) => module,
            Err(error) => return Some(Err(error)),
        };
        let result = module.invoke_with_services(
            &arguments,
            requested_outputs,
            module.api_mode(),
            Rc::new(RuntimeMexHostServices::new(runtime)),
        );
        Some(match result {
            Ok(invocation) => {
                if !invocation.console.is_empty() {
                    crate::console::record_console_output(
                        crate::console::ConsoleStream::Stdout,
                        invocation.console,
                    );
                }
                for warning in invocation.warnings {
                    crate::warning_store::push(
                        warning
                            .identifier
                            .as_deref()
                            .unwrap_or("RunMat:MEX:Warning"),
                        &warning.message,
                    );
                }
                Ok(match requested_outputs {
                    0 => Value::OutputList(Vec::new()),
                    1 => invocation
                        .outputs
                        .into_iter()
                        .next()
                        .unwrap_or_else(|| Value::OutputList(Vec::new())),
                    _ => Value::OutputList(invocation.outputs),
                })
            }
            Err(error) => Err(runtime_error("MexInvocation", error.to_string())),
        })
    }

    /// Clear selected modules and execute their registered `mexAtExit` hooks.
    pub fn clear(
        &self,
        request: &DynamicFunctionClearRequest,
        runtime: RuntimeContext,
    ) -> Result<(), RuntimeError> {
        let mut first_error = None;
        self.modules.borrow_mut().retain(|path, module| {
            let selected = match request {
                DynamicFunctionClearRequest::All
                | DynamicFunctionClearRequest::NativeExtensions => true,
                DynamicFunctionClearRequest::Named(name) => path_matches_name(path, name),
            };
            if !selected {
                return true;
            }
            match module.clear_with_services(Rc::new(RuntimeMexHostServices::new(runtime.clone())))
            {
                Ok(cleared) => !cleared,
                Err(error) => {
                    let lifecycle_finished = matches!(&error, MexLoadError::Invocation { .. });
                    first_error.get_or_insert_with(|| {
                        runtime_error(
                            "MexClear",
                            format!("could not clear MEX function '{}': {error}", path.display()),
                        )
                    });
                    !lifecycle_finished
                }
            }
        });
        first_error.map_or(Ok(()), Err)
    }

    /// Force final teardown of every module owned by this runtime session.
    pub fn shutdown(&self, runtime: RuntimeContext) -> Result<(), RuntimeError> {
        let modules = self.modules.borrow_mut().drain().collect::<Vec<_>>();
        let mut first_error = None;
        for (path, module) in modules {
            if let Err(error) =
                module.shutdown_with_services(Rc::new(RuntimeMexHostServices::new(runtime.clone())))
            {
                first_error.get_or_insert_with(|| {
                    runtime_error(
                        "MexShutdown",
                        format!(
                            "could not shut down MEX function '{}': {error}",
                            path.display()
                        ),
                    )
                });
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    fn module(&self, path: &Path) -> Result<Rc<MexModule>, RuntimeError> {
        if let Some(module) = self.modules.borrow().get(path).cloned() {
            return Ok(module);
        }
        let module = Rc::new(MexModule::load(path).map_err(|error| {
            runtime_error(
                "MexLoad",
                format!("could not load MEX function '{}': {error}", path.display()),
            )
        })?);
        self.modules
            .borrow_mut()
            .insert(path.to_path_buf(), Rc::clone(&module));
        Ok(module)
    }
}

fn path_matches_name(path: &Path, name: &str) -> bool {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .is_some_and(|stem| stem.eq_ignore_ascii_case(name))
}

fn runtime_error(identifier: &str, message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier(format!("RunMat:{identifier}"))
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::RuntimeExecutionService;

    #[test]
    fn cancelled_invocation_never_reaches_discovery_or_native_code() {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        runtime.cancellation().store(true, Ordering::Relaxed);
        let result = futures::executor::block_on(MexRuntimeSession::new().load_and_call(
            "module_that_must_not_be_resolved",
            Vec::new(),
            0,
            runtime,
        ))
        .expect("pre-cancelled MEX call is handled")
        .expect_err("pre-cancelled MEX call fails");

        assert_eq!(result.identifier(), Some("RunMat:Cancelled"));
    }

    #[test]
    fn clear_name_matching_is_case_insensitive_and_exact() {
        assert!(path_matches_name(
            Path::new("/tmp/NativeFilter.mexa64"),
            "nativefilter"
        ));
        assert!(!path_matches_name(
            Path::new("/tmp/NativeFilterHelper.mexa64"),
            "nativefilter"
        ));
    }
}
