use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{atomic::Ordering, Arc};

use futures::FutureExt;
use runmat_mex::{
    DirectMexBoundaryHostServices, MexApi, MexInvocation, MexNativeInvocation, MxValueContext,
};
use runmat_value::Value;

use super::{
    native_lane::{NativeMexLane, NativeModuleLoadError},
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
    native_lane: RefCell<Option<Rc<NativeMexLane>>>,
    native_modules: RefCell<HashSet<PathBuf>>,
    native_values: RefCell<HashMap<PathBuf, Rc<MxValueContext>>>,
    isolated: RefCell<HashMap<PathBuf, IsolatedMexClient>>,
    active_isolated: RefCell<HashSet<PathBuf>>,
    isolated_ready: Arc<tokio::sync::Notify>,
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

    /// Service one callback submitted by native MEX work after its gateway
    /// returned. Session hosts poll this alongside their command/input loop.
    pub async fn service_background_once(&self) -> bool {
        let lane = self.native_lane.borrow().as_ref().cloned();
        let native = async move {
            match lane {
                Some(lane) => lane.service_background_once().await,
                None => futures::future::pending().await,
            }
        }
        .fuse();
        let isolated = self.service_isolated_background_once().fuse();
        futures::pin_mut!(native, isolated);
        futures::select! {
            serviced = native => serviced,
            serviced = isolated => serviced,
        }
    }

    async fn service_isolated_background_once(&self) -> bool {
        loop {
            let paths = self.isolated.borrow().keys().cloned().collect::<Vec<_>>();
            if paths.is_empty() {
                return futures::future::pending().await;
            }
            for path in paths {
                let Some(mut client) = self.isolated.borrow_mut().remove(&path) else {
                    continue;
                };
                if !self.active_isolated.borrow_mut().insert(path.clone()) {
                    self.isolated.borrow_mut().insert(path, client);
                    continue;
                }
                let result = client.service_ready_background().await;
                self.active_isolated.borrow_mut().remove(&path);
                match result {
                    Ok(serviced) => {
                        self.isolated.borrow_mut().insert(path, client);
                        if serviced {
                            return true;
                        }
                    }
                    Err(error) => {
                        tracing::warn!(
                            module = %path.display(),
                            identifier = %error.identifier,
                            message = %error.message,
                            "isolated MEX background service stopped"
                        );
                        return true;
                    }
                }
            }
            self.isolated_ready.notified().await;
        }
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
            return Some(
                self.invoke_isolated(
                    &canonical,
                    &arguments,
                    requested_outputs,
                    None,
                    MexBinaryTier::RunMatCompatibleIsolated,
                    runtime,
                    policy.invocation_timeout,
                )
                .await,
            );
        }
        if runtime.execution_stack() != crate::context::RuntimeExecutionStack::Process {
            return Some(Err(runtime_error(
                "Foreign:ExecutionStackViolation",
                "in-process MEX invocation requires the process thread stack",
            )));
        }
        let lane = match self.native_lane() {
            Ok(lane) => lane,
            Err(error) => return Some(Err(error)),
        };
        let metadata = match lane.load(&canonical).await {
            Ok(metadata) => metadata,
            Err(NativeModuleLoadError::IsolatedHostRequired) => {
                let policy = *self.isolation_policy.borrow();
                return Some(
                    self.invoke_isolated(
                        &canonical,
                        &arguments,
                        requested_outputs,
                        None,
                        MexBinaryTier::RunMatExact,
                        runtime,
                        policy.invocation_timeout,
                    )
                    .await,
                );
            }
            Err(error) => {
                return Some(Err(runtime_error(
                    "MexLoad",
                    format!(
                        "could not load MEX function '{}': {error}",
                        canonical.display()
                    ),
                )))
            }
        };
        self.native_modules.borrow_mut().insert(canonical.clone());
        let values = self
            .native_values
            .borrow_mut()
            .entry(canonical.clone())
            .or_insert_with(|| Rc::new(MxValueContext::new()))
            .clone();
        let mut inputs = Vec::with_capacity(arguments.len());
        for argument in &arguments {
            let argument = match crate::gather_if_needed_async(argument).await {
                Ok(argument) => argument,
                Err(error) => return Some(Err(error)),
            };
            match values.encode(&argument, metadata.mode, metadata.interface) {
                Ok(argument) => inputs.push(argument),
                Err(error) => return Some(Err(runtime_error("MEX:Conversion", error.to_string()))),
            }
        }
        let services = Rc::new(DirectMexBoundaryHostServices::new(
            Rc::new(RuntimeMexHostServices::new(runtime)),
            values.clone(),
            metadata.mode,
            metadata.interface,
        ));
        let result = lane
            .invoke(&canonical, inputs, requested_outputs, services)
            .await;
        Some(match result {
            Ok(invocation) => decode_native_invocation(invocation, &values)
                .and_then(|invocation| finish_invocation(invocation, requested_outputs)),
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
        self.shutdown_native(runtime)
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

    fn native_lane(&self) -> Result<Rc<NativeMexLane>, RuntimeError> {
        if let Some(lane) = self.native_lane.borrow().as_ref() {
            return Ok(lane.clone());
        }
        let lane = Rc::new(
            NativeMexLane::spawn().map_err(|error| runtime_error("MEX:NativeLane", error))?,
        );
        *self.native_lane.borrow_mut() = Some(lane.clone());
        Ok(lane)
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
        match IsolatedMexClient::spawn(path.to_path_buf(), Arc::clone(&self.isolated_ready)).await {
            Ok(client) => Ok(client),
            Err(error) => {
                self.active_isolated.borrow_mut().remove(path);
                Err(wire_runtime_error(error))
            }
        }
    }

    async fn invoke_isolated(
        &self,
        path: &Path,
        arguments: &[Value],
        requested_outputs: usize,
        api: Option<MexApi>,
        tier: MexBinaryTier,
        runtime: RuntimeContext,
        timeout: Option<std::time::Duration>,
    ) -> Result<Value, RuntimeError> {
        let mut client = self.take_isolated_module(path).await?;
        let result = client
            .invoke(arguments, requested_outputs, api, tier, runtime, timeout)
            .await;
        self.active_isolated.borrow_mut().remove(path);
        if result.is_ok() {
            self.isolated
                .borrow_mut()
                .insert(path.to_path_buf(), client);
        }
        match result {
            Ok(invocation) => finish_invocation(invocation, requested_outputs),
            Err(error) => Err(wire_runtime_error(error)),
        }
    }

    fn clear_in_process(
        &self,
        request: &DynamicFunctionClearRequest,
        runtime: RuntimeContext,
    ) -> Result<(), RuntimeError> {
        let Some(lane) = self.native_lane.borrow().as_ref().cloned() else {
            return Ok(());
        };
        let paths = self
            .native_modules
            .borrow()
            .iter()
            .filter(|path| selected_by_request(path, request))
            .cloned()
            .collect::<Vec<_>>();
        let mut first_error = None;
        for path in paths {
            let Some(values) = self.native_values.borrow().get(&path).cloned() else {
                continue;
            };
            let metadata = match futures::executor::block_on(lane.load(&path)) {
                Ok(metadata) => metadata,
                Err(error) => {
                    first_error.get_or_insert_with(|| runtime_error("MexClear", error.to_string()));
                    continue;
                }
            };
            let services = Rc::new(DirectMexBoundaryHostServices::new(
                Rc::new(RuntimeMexHostServices::new(runtime.clone())),
                values,
                metadata.mode,
                metadata.interface,
            ));
            match futures::executor::block_on(lane.clear(&path, services)) {
                Ok(true) => {
                    self.native_modules.borrow_mut().remove(&path);
                    self.native_values.borrow_mut().remove(&path);
                }
                Ok(false) => {}
                Err(error) => {
                    first_error.get_or_insert_with(|| {
                        runtime_error(
                            "MexClear",
                            format!("could not clear MEX function '{}': {error}", path.display()),
                        )
                    });
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    fn shutdown_native(&self, runtime: RuntimeContext) -> Result<(), RuntimeError> {
        let Some(lane) = self.native_lane.borrow().as_ref().cloned() else {
            return Ok(());
        };
        let paths = self
            .native_modules
            .borrow()
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let mut first_error = None;
        for path in paths {
            let Some(values) = self.native_values.borrow().get(&path).cloned() else {
                continue;
            };
            let metadata = match futures::executor::block_on(lane.load(&path)) {
                Ok(metadata) => metadata,
                Err(error) => {
                    first_error
                        .get_or_insert_with(|| runtime_error("MexShutdown", error.to_string()));
                    continue;
                }
            };
            let services = Rc::new(DirectMexBoundaryHostServices::new(
                Rc::new(RuntimeMexHostServices::new(runtime.clone())),
                values,
                metadata.mode,
                metadata.interface,
            ));
            if let Err(error) = futures::executor::block_on(lane.shutdown_module(&path, services)) {
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
        self.native_modules.borrow_mut().clear();
        self.native_values.borrow_mut().clear();
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

fn decode_native_invocation(
    invocation: MexNativeInvocation,
    values: &MxValueContext,
) -> Result<MexInvocation, RuntimeError> {
    let outputs = invocation
        .outputs
        .iter()
        .map(|output| {
            values
                .decode(output)
                .map_err(|error| runtime_error("MEX:Conversion", error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(MexInvocation {
        outputs,
        warnings: invocation.warnings,
        console: invocation.console,
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
