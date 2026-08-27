use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::atomic::Ordering;

use runmat_mex::{MexInvocation, MexLoadError, MexModule};
use runmat_value::Value;

use super::{
    IsolatedMexClient, MexBinaryTier, MexIsolationPolicy, MexLifecycleOutcome, MexWireError,
    RuntimeMexHostServices, UnmanifestedMexPolicy,
};
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
    isolated: RefCell<HashMap<PathBuf, IsolatedMexClient>>,
    active_isolated: RefCell<HashSet<PathBuf>>,
    isolation_policy: RefCell<MexIsolationPolicy>,
    shutdown_in_progress: Cell<bool>,
}

impl MexRuntimeSession {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_isolation_policy(&self, policy: MexIsolationPolicy) {
        *self.isolation_policy.borrow_mut() = policy;
    }

    pub async fn load_and_call(
        &self,
        name: &str,
        arguments: Vec<Value>,
        requested_outputs: usize,
        runtime: RuntimeContext,
    ) -> Option<Result<Value, RuntimeError>> {
        if self.shutdown_in_progress.get() {
            return Some(Err(runtime_error(
                "MEX:Lifecycle",
                "MEX modules cannot be invoked while the session is shutting down",
            )));
        }
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
        if !runmat_mex::MexArtifactManifest::path_for_module(&canonical).is_file() {
            let policy = *self.isolation_policy.borrow();
            if policy.unmanifested == UnmanifestedMexPolicy::Deny {
                return Some(Err(runtime_error(
                    "MEX:UnmanifestedBinaryDenied",
                    format!(
                        "unmanifested MEX module '{}' is disabled by runtime policy",
                        canonical.display()
                    ),
                )));
            }
            let mut client = match self.take_isolated_module(&canonical).await {
                Ok(client) => client,
                Err(error) => return Some(Err(error)),
            };
            let result = client
                .invoke(
                    &arguments,
                    requested_outputs,
                    None,
                    MexBinaryTier::RunMatCompatibleIsolated,
                    runtime.clone(),
                    policy.invocation_timeout,
                )
                .await;
            self.active_isolated.borrow_mut().remove(&canonical);
            if result.is_ok() {
                self.isolated.borrow_mut().insert(canonical.clone(), client);
            }
            return Some(match result {
                Ok(invocation) => finish_invocation(invocation, requested_outputs),
                Err(error) => Err(wire_runtime_error(error)),
            });
        }
        if runtime.execution_stack() != crate::context::RuntimeExecutionStack::Process {
            return Some(Err(runtime_error(
                "Foreign:ExecutionStackViolation",
                "in-process MEX invocation requires the process thread stack",
            )));
        }
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
            Ok(invocation) => finish_invocation(invocation, requested_outputs),
            Err(error) => Err(runtime_error("MexInvocation", error.to_string())),
        })
    }

    /// Clear selected modules and execute their registered `mexAtExit` hooks.
    pub fn clear(
        &self,
        request: &DynamicFunctionClearRequest,
        runtime: RuntimeContext,
    ) -> Result<(), RuntimeError> {
        self.isolated.borrow_mut().retain(|path, _| match request {
            DynamicFunctionClearRequest::All | DynamicFunctionClearRequest::NativeExtensions => {
                false
            }
            DynamicFunctionClearRequest::Named(name) => !path_matches_name(path, name),
        });
        self.clear_in_process(request, runtime)
    }

    /// Clear selected modules after completing isolated-host lifecycle RPC.
    pub async fn clear_gracefully(
        &self,
        request: &DynamicFunctionClearRequest,
        runtime: RuntimeContext,
    ) -> Result<(), RuntimeError> {
        let policy = *self.isolation_policy.borrow();
        let selected = self
            .isolated
            .borrow()
            .keys()
            .filter(|path| selected_by_request(path, request))
            .cloned()
            .collect::<Vec<_>>();
        let mut first_error = None;
        for path in selected {
            let Some(mut client) = self.isolated.borrow_mut().remove(&path) else {
                continue;
            };
            if !self.active_isolated.borrow_mut().insert(path.clone()) {
                first_error.get_or_insert_with(|| {
                    runtime_error(
                        "MEX:Lifecycle",
                        format!(
                            "cannot clear MEX function '{}' while it is active",
                            path.display()
                        ),
                    )
                });
                self.isolated.borrow_mut().insert(path, client);
                continue;
            }
            match client
                .clear(runtime.clone(), policy.invocation_timeout)
                .await
            {
                Ok(MexLifecycleOutcome::Retained) => {
                    self.isolated.borrow_mut().insert(path.clone(), client);
                }
                Ok(MexLifecycleOutcome::Cleared | MexLifecycleOutcome::Shutdown) => {}
                Err(error) => {
                    first_error.get_or_insert_with(|| wire_runtime_error(error));
                }
            }
            self.active_isolated.borrow_mut().remove(&path);
        }
        if let Err(error) = self.clear_in_process(request, runtime) {
            first_error.get_or_insert(error);
        }
        first_error.map_or(Ok(()), Err)
    }

    /// Force final teardown of every module owned by this runtime session.
    pub fn shutdown(&self, runtime: RuntimeContext) -> Result<(), RuntimeError> {
        self.isolated.borrow_mut().clear();
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

    /// Complete isolated and in-process final lifecycle before session exit.
    pub async fn shutdown_gracefully(&self, runtime: RuntimeContext) -> Result<(), RuntimeError> {
        if self.shutdown_in_progress.replace(true) {
            return Err(runtime_error(
                "MEX:Lifecycle",
                "foreign runtime shutdown is already in progress",
            ));
        }
        let _lifecycle = ShutdownGuard(&self.shutdown_in_progress);
        let policy = *self.isolation_policy.borrow();
        let clients = self.isolated.borrow_mut().drain().collect::<Vec<_>>();
        let mut first_error = None;
        for (_, mut client) in clients {
            if let Err(error) = client
                .shutdown(runtime.clone(), policy.invocation_timeout)
                .await
            {
                first_error.get_or_insert_with(|| wire_runtime_error(error));
            }
        }
        if let Err(error) = self.shutdown(runtime) {
            first_error.get_or_insert(error);
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

    async fn take_isolated_module(&self, path: &Path) -> Result<IsolatedMexClient, RuntimeError> {
        if self.active_isolated.borrow().contains(path) {
            return Err(runtime_error(
                "MEX:ReentrantInvocation",
                "recursive invocation of the same isolated MEX module is not supported",
            ));
        }
        self.active_isolated.borrow_mut().insert(path.to_path_buf());
        if let Some(client) = self.isolated.borrow_mut().remove(path) {
            return Ok(client);
        }
        match IsolatedMexClient::spawn(path.to_path_buf()).await {
            Ok(client) => Ok(client),
            Err(error) => {
                self.active_isolated.borrow_mut().remove(path);
                Err(wire_runtime_error(error))
            }
        }
    }

    fn clear_in_process(
        &self,
        request: &DynamicFunctionClearRequest,
        runtime: RuntimeContext,
    ) -> Result<(), RuntimeError> {
        let mut first_error = None;
        self.modules.borrow_mut().retain(|path, module| {
            if !selected_by_request(path, request) {
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
}

struct ShutdownGuard<'a>(&'a Cell<bool>);

impl Drop for ShutdownGuard<'_> {
    fn drop(&mut self) {
        self.0.set(false);
    }
}

fn selected_by_request(path: &Path, request: &DynamicFunctionClearRequest) -> bool {
    match request {
        DynamicFunctionClearRequest::All | DynamicFunctionClearRequest::NativeExtensions => true,
        DynamicFunctionClearRequest::Named(name) => path_matches_name(path, name),
    }
}

fn finish_invocation(
    invocation: MexInvocation,
    requested_outputs: usize,
) -> Result<Value, RuntimeError> {
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

fn wire_runtime_error(error: MexWireError) -> RuntimeError {
    let mut message = error.message;
    if let Some(dependency) = error.dependency {
        message.push_str(&format!(" (dependency: {dependency})"));
    }
    build_runtime_error(message)
        .with_identifier(error.identifier)
        .build()
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
