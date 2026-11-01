use std::cell::{Cell, RefCell};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use futures::FutureExt;
use runmat_mex::{MexApi, MexNativeInvocation, MxBoundaryCodec, MxValueContext};
use runmat_process_host::ProcessGroupPlacement;
use runmat_types::{CapabilityRequirement, ForeignCapability};
use runmat_value::Value;

use super::isolation::{in_isolated_callback, IsolatedMexRetirement};
use super::RootedMexInvocation;
use super::{
    native_lane::{NativeLaneOwner, NativeMexLane, NativeModuleLoadError},
    IsolatedMexClient, MexBinaryTier, MexIsolationPolicy, MexLifecycleOutcome, MexWireError,
    RuntimeMexHostServices, UnmanifestedMexPolicy,
};
use crate::context::ForeignCall;
use crate::context::{LocalTaskCommandOrigin, RuntimeContext};
use crate::user_functions::DynamicFunctionClearRequest;
use crate::{build_runtime_error, RuntimeError};
mod artifact_catalog;
mod isolated_client_slot;
mod background_output;
mod ordinary_admission;
mod retained;
mod token_root;

pub use artifact_catalog::{MexArtifactExpectation, PreparedMexArtifactCatalog};
use isolated_client_slot::IsolatedClientSlot;
use token_root::RegisteredMxValueContext;

#[derive(Clone)]
struct InstalledMexArtifact {
    path: PathBuf,
    identity: runmat_types::ForeignArtifactIdentity,
    interop: runmat_types::InteropManifest,
}

/// Read-only name selection performed by the same owner that invokes MEX.
/// An installed artifact precedes path discovery; no binary is loaded here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MexNameSelection {
    Installed {
        path: PathBuf,
        identity: runmat_types::ForeignArtifactIdentity,
        interop: runmat_types::InteropManifest,
    },
    SearchPath(crate::builtins::common::path_search::FileCandidate),
}

impl MexNameSelection {
    pub fn path(&self) -> &Path {
        match self {
            Self::Installed { path, .. } => path,
            Self::SearchPath(candidate) => candidate.path(),
        }
    }
}

#[derive(Clone, Copy)]
enum MexResolutionScope {
    SearchPath,
    InstalledOnly,
}

/// Session-scoped owner of dynamically loaded MEX modules.
///
/// Core, JIT callbacks, and standalone AOT execution all use this same path so
/// discovery, manifest admission, persistent state, callback authority, and
/// `mexAtExit` lifecycle behavior cannot drift between execution modes.
#[derive(Default)]
pub struct MexRuntimeSession {
    installed_modules: RefCell<HashMap<String, InstalledMexArtifact>>,
    native_lane: RefCell<Option<Rc<NativeMexLane>>>,
    native_lane_owner: NativeLaneOwner,
    native_lane_owner_retired: Cell<bool>,
    parent_native_lane: Option<MexNativeLaneChildFactory>,
    native_modules: RefCell<HashSet<PathBuf>>,
    native_values: RefCell<HashMap<PathBuf, Rc<RegisteredMxValueContext>>>,
    isolated: RefCell<HashMap<PathBuf, Rc<IsolatedClientSlot>>>,
    active_isolated: RefCell<HashSet<PathBuf>>,
    isolated_ready: Arc<tokio::sync::Notify>,
    isolation_policy: RefCell<MexIsolationPolicy>,
    installed_byte_lease: RefCell<Option<runmat_process_host::private_bytes::PrivateByteLease>>,
    ordinary_modules: RefCell<Vec<ordinary_admission::AdmittedOrdinaryMexArtifact>>,
    shutdown_in_progress: Rc<Cell<bool>>,
    abandoned: Cell<bool>,
    inflight_operations: RefCell<Vec<crate::context::RuntimeLocalTaskLease>>,
    process_group_placement: ProcessGroupPlacement,
}

/// A weak parent-lane authority for a nested test session. The child retains
/// its own module, token and lifecycle owners while using the process-gate
/// owner's native thread for callback reentry.
#[derive(Clone)]
pub struct MexNativeLaneChildFactory(std::rc::Weak<MexRuntimeSession>);

impl MexNativeLaneChildFactory {
    fn lane(&self) -> Result<Rc<NativeMexLane>, RuntimeError> {
        let parent = self
            .0
            .upgrade()
            .ok_or_else(|| runtime_error("MEX:NativeLane", "the parent MEX session has retired"))?;
        if parent.abandoned.get() || parent.shutdown_in_progress.get() {
            return Err(runtime_error(
                "MEX:NativeLane",
                "the parent MEX session is closing and cannot lend its native lane",
            ));
        }
        // A live parent may not have needed a lane yet. Its owner issues
        // that first lane here so both callback depths use one native thread.
        // A closed parent never manufactures a replacement lane.
        parent.native_lane()
    }
}

/// Holds the MEX policy cell through a synchronous selected-owner publication.
pub struct ReadyMexIsolationPolicy<'a> {
    policy: std::cell::RefMut<'a, MexIsolationPolicy>,
    next: MexIsolationPolicy,
}

pub struct ReadyMexArtifactCatalog<'a> {
    catalog: std::cell::RefMut<'a, HashMap<String, InstalledMexArtifact>>,
    lease: std::cell::RefMut<'a, Option<runmat_process_host::private_bytes::PrivateByteLease>>,
    next: PreparedMexArtifactCatalog,
}

impl ReadyMexArtifactCatalog<'_> {
    pub fn commit(mut self) {
        *self.catalog = self.next.modules;
        *self.lease = self.next.byte_lease;
    }
}

impl ReadyMexIsolationPolicy<'_> {
    pub fn commit(mut self) {
        *self.policy = self.next;
    }
}

pub enum MexSessionRetirement {
    Retained(RuntimeError),
    Retired(Result<(), RuntimeError>),
}

struct IsolatedActivity<'a> {
    active: &'a RefCell<HashSet<PathBuf>>,
    path: PathBuf,
}

impl Drop for IsolatedActivity<'_> {
    fn drop(&mut self) {
        self.active.borrow_mut().remove(&self.path);
    }
}

impl MexSessionRetirement {
    pub fn into_result(self) -> Result<(), RuntimeError> {
        match self {
            Self::Retained(error) => Err(error),
            Self::Retired(result) => result,
        }
    }
}

impl MexRuntimeSession {
    pub fn native_lane_child_factory(self: &Rc<Self>) -> MexNativeLaneChildFactory {
        MexNativeLaneChildFactory(Rc::downgrade(self))
    }

    pub fn new_on_parent_native_lane(parent: MexNativeLaneChildFactory) -> Self {
        Self {
            parent_native_lane: Some(parent),
            ..Self::default()
        }
    }

    pub fn has_unretired_work(&self) -> bool {
        self.native_modules
            .try_borrow()
            .map_or(true, |modules| !modules.is_empty())
            || self
                .native_values
                .try_borrow()
                .map_or(true, |values| !values.is_empty())
            || self
                .isolated
                .try_borrow()
                .map_or(true, |clients| !clients.is_empty())
            || self.native_lane.try_borrow().map_or(true, |lane| {
                lane.as_ref()
                    .is_some_and(|lane| lane.has_pending_services_for(&self.native_lane_owner))
            })
    }

    /// Forced host disposal revokes origin callback authority before token
    /// registrations retire. This is Abandoned, not acknowledged mexAtExit.
    pub fn force_abandon(&self) {
        if self.abandoned.replace(true) {
            return;
        }
        self.shutdown_in_progress.set(true);
        if let Some(lane) = self.native_lane.borrow().as_ref() {
            lane.force_abandon_for(&self.native_lane_owner);
        }
        self.retire_native_lane_owner();
        let clients = std::mem::take(&mut *self.isolated.borrow_mut());
        drop(clients);
        self.native_modules.borrow_mut().clear();
        let registrations = std::mem::take(&mut *self.native_values.borrow_mut());
        drop(registrations);
        for artifact in self.ordinary_modules.borrow().iter() {
            if let Err(error) = artifact.persist_unverified() {
                eprintln!("ordinary MEX private bytes retained without recovery marker: {error}");
            }
        }
    }
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_in_process_group(placement: ProcessGroupPlacement) -> Self {
        Self {
            process_group_placement: placement,
            ..Self::default()
        }
    }

    /// Adapt this executable owner to the runtime's read-only name port.
    pub fn name_observer(self: &Rc<Self>) -> Rc<crate::user_functions::NativeNameObserver> {
        self.name_observer_with_scope(MexResolutionScope::SearchPath)
    }

    /// Selected units observe only modules installed by their checked receipt.
    /// In particular, a missing installed name cannot consult the process path.
    pub fn installed_name_observer(
        self: &Rc<Self>,
    ) -> Rc<crate::user_functions::NativeNameObserver> {
        self.name_observer_with_scope(MexResolutionScope::InstalledOnly)
    }

    fn name_observer_with_scope(
        self: &Rc<Self>,
        resolution: MexResolutionScope,
    ) -> Rc<crate::user_functions::NativeNameObserver> {
        let owner = Rc::clone(self);
        Rc::new(move |runtime, name| {
            let owner = Rc::clone(&owner);
            crate::user_functions::admit_native_name_observation_with(
                &runtime,
                move || async move {
                    let selection = match resolution {
                        MexResolutionScope::SearchPath => {
                            owner.observe_name(&name).await.map_err(|error| {
                                error.into_identified_runtime_error(
                                    "MEX function resolution",
                                    "RunMat:FunctionResolution",
                                )
                            })?
                        }
                        MexResolutionScope::InstalledOnly => owner.observe_installed_name(&name),
                    };
                    Ok(selection.map(|selection| match selection {
                        MexNameSelection::Installed {
                            path,
                            identity,
                            interop,
                        } => crate::user_functions::NativeNameFact {
                            path,
                            origin: None,
                            installed: true,
                            installed_artifact: Some(
                                crate::user_functions::InstalledNativeArtifactFact {
                                    identity,
                                    interop,
                                },
                            ),
                        },
                        MexNameSelection::SearchPath(candidate) => {
                            let (path, origin) = candidate.into_parts();
                            crate::user_functions::NativeNameFact {
                                path,
                                origin: Some(origin),
                                installed: false,
                                installed_artifact: None,
                            }
                        }
                    }))
                },
            )
        })
    }

    fn observe_installed_name(&self, name: &str) -> Option<MexNameSelection> {
        self.installed_modules
            .borrow()
            .get(name)
            .cloned()
            .map(|artifact| MexNameSelection::Installed {
                path: artifact.path,
                identity: artifact.identity,
                interop: artifact.interop,
            })
    }

    /// Observe the exact installed-or-current-platform path selected by
    /// `load_and_call`, without loading, validating, or executing the module.
    pub async fn observe_name(
        &self,
        name: &str,
    ) -> Result<Option<MexNameSelection>, crate::builtins::common::path_search::PathSearchError>
    {
        if let Some(installed) = self.observe_installed_name(name) {
            return Ok(Some(installed));
        }
        self.observe_search_path_name(name).await
    }

    async fn observe_search_path_name(
        &self,
        name: &str,
    ) -> Result<Option<MexNameSelection>, crate::builtins::common::path_search::PathSearchError>
    {
        let Some(suffix) = runmat_mex::mex_suffix() else {
            return Ok(None);
        };
        let extension = format!(".{suffix}");
        crate::builtins::common::path_search::find_file_with_extensions_with_origin(
            name,
            &[extension.as_str()],
            "MEX function resolution",
        )
        .await
        .map(|candidate| candidate.map(MexNameSelection::SearchPath))
    }

    pub fn set_isolation_policy(&self, policy: MexIsolationPolicy) {
        *self.isolation_policy.borrow_mut() = policy;
    }

    pub fn admit_isolation_policy(
        &self,
        next: MexIsolationPolicy,
    ) -> Result<ReadyMexIsolationPolicy<'_>, RuntimeError> {
        let policy = self.isolation_policy.try_borrow_mut().map_err(|_| {
            runtime_error(
                "MEX:Policy",
                "MEX isolation policy is busy during publication",
            )
        })?;
        Ok(ReadyMexIsolationPolicy { policy, next })
    }

    pub fn clear_installed_artifacts(&self) {
        self.installed_modules.borrow_mut().clear();
    }

    /// Forced disposal cannot prove a retained MEX child has stopped using
    /// installed bytes; leave its exact private root for recovery instead.
    pub fn persist_installed_bytes_unverified(&self, reason: &str) -> Result<(), String> {
        if let Some(lease) = self.installed_byte_lease.borrow().as_ref() {
            lease.persist_unverified(None, reason)?;
        }
        Ok(())
    }

    /// Publish validated metadata after the host has retired old modules.
    /// Publication performs no filesystem access or fallible validation.
    pub fn replace_installed_artifacts(&self, catalog: PreparedMexArtifactCatalog) {
        let mut modules = self.installed_modules.borrow_mut();
        let mut lease = self.installed_byte_lease.borrow_mut();
        *modules = catalog.modules;
        *lease = catalog.byte_lease;
    }

    pub fn admit_artifact_catalog(
        &self,
        next: PreparedMexArtifactCatalog,
    ) -> Result<ReadyMexArtifactCatalog<'_>, RuntimeError> {
        let catalog = self
            .installed_modules
            .try_borrow_mut()
            .map_err(|_| runtime_error("MEX:Artifact", "MEX catalog is busy during publication"))?;
        let lease = self.installed_byte_lease.try_borrow_mut().map_err(|_| {
            runtime_error("MEX:Artifact", "MEX byte owner is busy during publication")
        })?;
        Ok(ReadyMexArtifactCatalog {
            catalog,
            lease,
            next,
        })
    }

    /// Register an exact, already-materialized MEX artifact with this session.
    ///
    /// The module remains owned by its caller. Its canonical sidecar manifest
    /// is verified before the name becomes visible to dynamic resolution.
    pub fn install_artifact(&self, module: &Path) -> Result<String, RuntimeError> {
        let (name, installed) = artifact_catalog::prepare_artifact(module, None)?;
        let mut modules = self.installed_modules.borrow_mut();
        if let Some(existing) = modules.get(&name) {
            if existing.path != installed.path || existing.identity != installed.identity {
                return Err(runtime_error(
                    "MEX:ArtifactConflict",
                    format!(
                        "MEX module '{}' is already installed with a different artifact",
                        name
                    ),
                ));
            }
            return Ok(existing.identity.to_string());
        }
        let identity = installed.identity.to_string();
        modules.insert(name, installed);
        Ok(identity)
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
            let paths = self
                .isolated
                .borrow()
                .keys()
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>();
            if paths.is_empty() {
                self.isolated_ready.notified().await;
                continue;
            }
            for path in &paths {
                let Some(client) = self.isolated.borrow().get(path).cloned() else {
                    continue;
                };
                let Ok(_activity) = self.enter_isolated(path) else {
                    continue;
                };
                let Ok(mut operation) = client.acquire() else {
                    tracing::warn!(module = %path.display(), "isolated MEX transport is busy after activity admission");
                    return true;
                };
                let result = operation.client().service_ready_background().await;
                match result {
                    Ok(serviced) => {
                        if serviced {
                            return true;
                        }
                    }
                    Err(error) => {
                        tracing::warn!(
                            module = %path.display(),
                            identifier = ?error.identifier(),
                            error = %error,
                            "isolated MEX background service stopped"
                        );
                        return true;
                    }
                }
            }
            match self.wait_for_isolated_activity(&paths).await {
                Ok(true) => return true,
                Ok(false) => {}
                Err(error) => {
                    tracing::warn!(error = %error, "isolated MEX application output stopped");
                    return true;
                }
            }
        }
    }

    pub fn load_and_call<'a>(
        self: &'a Rc<Self>,
        name: &'a str,
        arguments: crate::call::RootedFunctionArguments,
        requested_outputs: usize,
        runtime: RuntimeContext,
    ) -> impl std::future::Future<Output = crate::user_functions::DynamicFunctionLoadAttempt> + 'a
    {
        self.load_with_resolution(
            name,
            arguments,
            requested_outputs,
            runtime,
            MexResolutionScope::SearchPath,
        )
    }

    /// Selected units may invoke only a module already installed by their
    /// authenticated private receipt. They never rediscover a host path.
    pub fn load_installed_and_call<'a>(
        self: &'a Rc<Self>,
        name: &'a str,
        arguments: crate::call::RootedFunctionArguments,
        requested_outputs: usize,
        runtime: RuntimeContext,
    ) -> impl std::future::Future<Output = crate::user_functions::DynamicFunctionLoadAttempt> + 'a
    {
        self.load_with_resolution(
            name,
            arguments,
            requested_outputs,
            runtime,
            MexResolutionScope::InstalledOnly,
        )
    }

    async fn load_with_resolution(
        self: &Rc<Self>,
        name: &str,
        arguments: crate::call::RootedFunctionArguments,
        requested_outputs: usize,
        runtime: RuntimeContext,
        resolution: MexResolutionScope,
    ) -> crate::user_functions::DynamicFunctionLoadAttempt {
        if self.abandoned.get() || self.shutdown_in_progress.get() {
            return ControlFlow::Break(Err(runtime_error(
                "MEX:Lifecycle",
                "MEX modules cannot be invoked while the session is shutting down",
            )));
        }
        if runtime.is_cancelled() {
            return ControlFlow::Break(Err(runtime_error(
                "Cancelled",
                format!("MEX invocation '{name}' was cancelled before entering native code"),
            )));
        }

        let path = match resolution {
            MexResolutionScope::InstalledOnly => self
                .installed_modules
                .borrow()
                .get(name)
                .map(|artifact| artifact.path.clone()),
            MexResolutionScope::SearchPath => match self.observe_name(name).await {
                Ok(Some(selection)) => Some(selection.path().to_path_buf()),
                Ok(None) => None,
                Err(error) => {
                    return ControlFlow::Break(Err(error.into_identified_runtime_error(
                        "MEX function resolution",
                        "RunMat:FunctionResolution",
                    )))
                }
            },
        };
        let Some(path) = path else {
            return ControlFlow::Continue(arguments);
        };
        let canonical = runmat_filesystem::canonicalize_async(&path)
            .await
            .unwrap_or(path);
        ControlFlow::Break(
            self.dispatch_admitted_path(canonical, arguments, requested_outputs, runtime)
                .await
                .unwrap_or_else(|| {
                    Err(runtime_error(
                        "MEX:SelectionChanged",
                        "selected MEX target retired during invocation",
                    ))
                }),
        )
    }

    async fn dispatch_admitted_path(
        self: &Rc<Self>,
        canonical: PathBuf,
        arguments: crate::call::RootedFunctionArguments,
        requested_outputs: usize,
        runtime: RuntimeContext,
    ) -> Option<Result<crate::execution::RootedValueSequence, RuntimeError>> {
        let Some(tasks) = runtime.service_ports().local_tasks().cloned() else {
            return Some(Err(runtime_error(
                "MEX:LifecycleHostUnavailable",
                "MEX invocation requires a scoped native lifecycle host",
            )));
        };
        if tasks.owner_token().is_none() {
            return Some(Err(runtime_error(
                "MEX:LifecycleHostUnavailable",
                "MEX invocation requires a host-owned local task supervisor",
            )));
        }
        let owner = Rc::clone(self);
        let (sender, receiver) = futures::channel::oneshot::channel();
        let lease = match runtime.spawn_local_with_command_origin(
            "MEX invocation",
            LocalTaskCommandOrigin::CurrentSelected,
            move |task_runtime| {
                let scoped_runtime = task_runtime.clone();
                Box::pin(scoped_runtime.scope(async move {
                    let result = if task_runtime.is_cancelled() {
                        Some(Err(runtime_error(
                            "Cancelled",
                            "MEX invocation was cancelled before its host task started",
                        )))
                    } else {
                        owner
                            .load_selected_and_call(
                                canonical,
                                arguments,
                                requested_outputs,
                                task_runtime,
                            )
                            .await
                    };
                    let _ = sender.send(result);
                }))
            },
        ) {
            Ok(lease) => lease,
            Err(error) => return Some(Err(error)),
        };
        self.inflight_operations
            .borrow_mut()
            .retain(|pending| pending.outcome().is_none());
        self.inflight_operations.borrow_mut().push(lease.clone());
        let reply = receiver.await;
        // The reply is sent just before the scoped task retires. Wait for its
        // actual completion so a following clear cannot mistake the module
        // for still in flight and retain its private bytes after unloading it.
        let outcome = lease.join().await;
        if outcome == crate::context::RuntimeLocalTaskOutcome::Cancelled {
            return Some(Err(runtime_error(
                "MEX:Lifecycle",
                "host cancelled an admitted MEX invocation before retirement",
            )));
        }
        match reply {
            Ok(result) => result,
            Err(_) => Some(Err(runtime_error(
                "MEX:Lifecycle",
                "host stopped before MEX invocation completed",
            ))),
        }
    }

    async fn load_selected_and_call(
        &self,
        canonical: PathBuf,
        arguments: crate::call::RootedFunctionArguments,
        requested_outputs: usize,
        runtime: RuntimeContext,
    ) -> Option<Result<crate::execution::RootedValueSequence, RuntimeError>> {
        if runtime.is_cancelled() {
            return Some(Err(runtime_error(
                "Cancelled",
                "MEX invocation was cancelled before native admission",
            )));
        }
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
        // Arbitrary in-process native instructions have no safe preemption
        // hook. Reject a requested deadline before loading a library (which
        // can itself run native initializers), never by abandoning live native
        // references or the origin callback services after a timer fires.
        if let Err(error) =
            admit_manifested_native_timeout(self.isolation_policy.borrow().invocation_timeout)
        {
            return Some(Err(error));
        }
        let lane = match self.native_lane() {
            Ok(lane) => lane,
            Err(error) => return Some(Err(error)),
        };
        if runtime.is_cancelled() {
            return Some(Err(runtime_error(
                "Cancelled",
                "MEX invocation was cancelled before native load",
            )));
        }
        let metadata = match lane.load_for(&self.native_lane_owner, &canonical).await {
            Ok(metadata) => metadata,
            Err(NativeModuleLoadError::IsolatedHostRequired) => {
                if runtime.is_cancelled() {
                    return Some(Err(runtime_error(
                        "Cancelled",
                        "MEX invocation was cancelled before isolated call",
                    )));
                }
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
        // Native load itself takes ownership of this private path. Keep it
        // tracked even if token registration or input conversion later fails.
        self.native_modules.borrow_mut().insert(canonical.clone());
        let registration = match self.native_value_registration(&canonical) {
            Ok(registration) => registration,
            Err(error) => return Some(Err(error)),
        };
        let values = registration.context();
        let mut inputs = Vec::with_capacity(arguments.len());
        for argument in arguments.iter() {
            let argument = match crate::gather_if_needed_async(argument).await {
                Ok(argument) => argument,
                Err(error) => return Some(Err(error)),
            };
            match values.encode(&argument, metadata.mode, metadata.interface) {
                Ok(argument) => inputs.push(argument),
                Err(error) => return Some(Err(runtime_error("MEX:Conversion", error.to_string()))),
            }
        }
        if runtime.is_cancelled() {
            return Some(Err(runtime_error(
                "Cancelled",
                "MEX invocation was cancelled before native call",
            )));
        }
        let host = Rc::new(RuntimeMexHostServices::new(runtime));
        let codec = Rc::new(MxBoundaryCodec::new_with_origin_lease(
            values.clone(),
            metadata.mode,
            metadata.interface,
            registration,
        ));
        let services = Rc::new(super::services::OriginMexBoundaryAdapter::new(host, codec));
        let result = lane
            .invoke_for(
                &self.native_lane_owner,
                &canonical,
                inputs,
                requested_outputs,
                services,
            )
            .await;
        Some(match result {
            Ok(invocation) => decode_native_invocation(invocation, &values)
                .and_then(|invocation| finish_invocation(invocation, requested_outputs)),
            Err(error) => Err(native_runtime_error(
                "MexInvocation",
                error.to_string(),
                error,
            )),
        })
    }

    /// Await selected native modules and their registered `mexAtExit` hooks.
    pub async fn clear(
        &self,
        request: &DynamicFunctionClearRequest,
        runtime: RuntimeContext,
    ) -> Result<(), RuntimeError> {
        let ordinary_selected = self.ordinary_selected_private_paths(request);
        if self
            .isolated
            .borrow()
            .keys()
            .any(|path| selected_by_request(path, request) || ordinary_selected.contains(path))
        {
            return Err(runtime_error(
                "MEX:Lifecycle",
                "isolated MEX modules require clear_gracefully while their host is active",
            ));
        }
        let result = self.clear_in_process(request, runtime).await;
        self.prune_retired_ordinary(request);
        result
    }

    /// Clear selected modules after completing isolated-host lifecycle RPC.
    pub async fn clear_gracefully(
        self: &Rc<Self>,
        request: &DynamicFunctionClearRequest,
        runtime: RuntimeContext,
    ) -> Result<(), RuntimeError> {
        if self.abandoned.get() || self.shutdown_in_progress.get() {
            return Err(runtime_error(
                "MEX:Lifecycle",
                "MEX clear cannot start after session shutdown or abandonment",
            ));
        }
        let tasks = runtime
            .service_ports()
            .local_tasks()
            .cloned()
            .ok_or_else(|| {
                runtime_error(
                    "MEX:LifecycleHostUnavailable",
                    "MEX clear requires a scoped native lifecycle host",
                )
            })?;
        if tasks.owner_token().is_none() {
            return Err(runtime_error(
                "MEX:LifecycleHostUnavailable",
                "MEX clear requires a host-owned local task supervisor",
            ));
        }
        let owner = Rc::clone(self);
        let request = request.clone();
        let (sender, receiver) = futures::channel::oneshot::channel();
        let lease = runtime.spawn_local_with_command_origin(
            "MEX clear",
            LocalTaskCommandOrigin::CurrentSelected,
            move |task_runtime| {
                let scoped_runtime = task_runtime.clone();
                Box::pin(scoped_runtime.scope(async move {
                    let result = owner.clear_gracefully_inner(&request, task_runtime).await;
                    let _ = sender.send(result);
                }))
            },
        )?;
        let result = receiver.await;
        let outcome = lease.join().await;
        if outcome == crate::context::RuntimeLocalTaskOutcome::Cancelled {
            return Err(runtime_error(
                "MEX:Lifecycle",
                "host cancelled MEX clear before its lifecycle reply was settled",
            ));
        }
        result.map_err(|_| {
            runtime_error("MEX:Lifecycle", "host stopped before MEX clear completed")
        })?
    }

    async fn clear_gracefully_inner(
        &self,
        request: &DynamicFunctionClearRequest,
        runtime: RuntimeContext,
    ) -> Result<(), RuntimeError> {
        let policy = *self.isolation_policy.borrow();
        let ordinary_selected = self.ordinary_selected_private_paths(request);
        let selected = self
            .isolated
            .borrow()
            .keys()
            .filter(|path| selected_by_request(path, request) || ordinary_selected.contains(*path))
            .cloned()
            .collect::<Vec<_>>();
        let mut first_error = None;
        let mut isolated_retained = false;
        for path in selected {
            let Some(client) = self.isolated.borrow().get(&path).cloned() else {
                continue;
            };
            let Ok(_activity) = self.enter_isolated(&path) else {
                isolated_retained = true;
                first_error.get_or_insert_with(|| {
                    runtime_error(
                        "MEX:Lifecycle",
                        format!(
                            "cannot clear MEX function '{}' while it is active",
                            path.display()
                        ),
                    )
                });
                continue;
            };
            let Ok(mut operation) = client.acquire() else {
                isolated_retained = true;
                first_error.get_or_insert_with(|| {
                    runtime_error(
                        "MEX:Lifecycle",
                        format!(
                            "isolated MEX transport '{}' is busy during clear",
                            path.display()
                        ),
                    )
                });
                continue;
            };
            let clear = operation
                .client()
                .clear(runtime.clone(), policy.invocation_timeout)
                .await;
            match clear {
                Ok(MexLifecycleOutcome::Retained) => {}
                Ok(MexLifecycleOutcome::Cleared) => {
                    // Clear unloads the module but leaves the isolated host
                    // alive. Retire that host and await its exit before the
                    // session releases the client and its callback authority.
                    let retirement = operation
                        .client()
                        .shutdown(runtime.clone(), policy.invocation_timeout)
                        .await;
                    match retirement {
                        IsolatedMexRetirement::Retired(result) => {
                            self.isolated.borrow_mut().remove(&path);
                            if let Err(error) = result {
                                first_error.get_or_insert(error);
                            }
                        }
                        IsolatedMexRetirement::Retained(error) => {
                            isolated_retained = true;
                            first_error.get_or_insert(error);
                        }
                    }
                }
                Ok(MexLifecycleOutcome::Retired { .. }) => {
                    isolated_retained = true;
                    first_error.get_or_insert_with(|| {
                        runtime_error("MEX:Lifecycle", "host returned a shutdown result for clear")
                    });
                }
                Err(error) => {
                    isolated_retained = true;
                    first_error.get_or_insert(error);
                }
            }
        }
        if isolated_retained {
            self.prune_retired_ordinary(request);
            return Err(first_error.unwrap_or_else(|| {
                runtime_error("MEX:Lifecycle", "an isolated MEX host remains active")
            }));
        }
        if let Err(error) = self.clear_in_process(request, runtime).await {
            first_error.get_or_insert(error);
        }
        if let Err(error) = self.settle_native_commands().await {
            first_error.get_or_insert(error);
        }
        self.prune_retired_ordinary(request);
        first_error.map_or(Ok(()), Err)
    }

    /// Await final teardown of every module owned by this runtime session.
    pub async fn shutdown(&self, runtime: RuntimeContext) -> Result<(), RuntimeError> {
        if !self.isolated.borrow().is_empty() {
            return Err(runtime_error(
                "MEX:Lifecycle",
                "isolated MEX hosts require graceful shutdown before session disposal",
            ));
        }
        if self
            .native_lane
            .borrow()
            .as_ref()
            .is_some_and(|lane| lane.has_pending_services_for(&self.native_lane_owner))
        {
            return Err(runtime_error(
                "MEX:Lifecycle",
                "native MEX work requires graceful settlement before session disposal",
            ));
        }
        let retirement = self.shutdown_native(runtime).await;
        if matches!(retirement, MexSessionRetirement::Retired(_)) {
            self.retire_native_lane_owner();
        }
        retirement.into_result()
    }

    /// Complete isolated and in-process final lifecycle before session exit.
    pub async fn shutdown_gracefully(
        self: &Rc<Self>,
        runtime: RuntimeContext,
    ) -> Result<(), RuntimeError> {
        self.shutdown_gracefully_retirement(runtime)
            .await
            .into_result()
    }

    /// Return actual session retirement separately from a post-unload
    /// diagnostic. Hosts may close dependent callback services only after
    /// `Retired`, regardless of whether mexAtExit reported an error.
    pub async fn shutdown_gracefully_retirement(
        self: &Rc<Self>,
        runtime: RuntimeContext,
    ) -> MexSessionRetirement {
        if self.abandoned.get() {
            return MexSessionRetirement::Retained(runtime_error(
                "MEX:Lifecycle",
                "MEX session was already abandoned by its host",
            ));
        }
        if in_isolated_callback() {
            return MexSessionRetirement::Retained(runtime_error(
                "MEX:Lifecycle",
                "cannot shut down the MEX session from its active host callback",
            ));
        }
        if self
            .native_lane
            .borrow()
            .as_ref()
            .is_some_and(|lane| lane.in_callback_for(&self.native_lane_owner))
        {
            return MexSessionRetirement::Retained(runtime_error(
                "MEX:Lifecycle",
                "cannot shut down the MEX session from its active native callback",
            ));
        }
        if self.shutdown_in_progress.replace(true) {
            return MexSessionRetirement::Retained(runtime_error(
                "MEX:Lifecycle",
                "foreign runtime shutdown is already in progress",
            ));
        }
        let _admission = ShutdownGuard(Rc::clone(&self.shutdown_in_progress));
        let Some(tasks) = runtime.service_ports().local_tasks().cloned() else {
            return MexSessionRetirement::Retained(runtime_error(
                "MEX:LifecycleHostUnavailable",
                "MEX shutdown requires a scoped native lifecycle host",
            ));
        };
        if tasks.owner_token().is_none() {
            return MexSessionRetirement::Retained(runtime_error(
                "MEX:LifecycleHostUnavailable",
                "MEX shutdown requires a host-owned local task supervisor",
            ));
        }
        let owner = Rc::clone(self);
        let (sender, receiver) = futures::channel::oneshot::channel();
        let lease = match runtime.spawn_local_with_command_origin(
            "MEX shutdown",
            LocalTaskCommandOrigin::CurrentSelected,
            move |task_runtime| {
                let scoped_runtime = task_runtime.clone();
                Box::pin(scoped_runtime.scope(async move {
                    let _admission = _admission;
                    let result = owner.shutdown_gracefully_inner(task_runtime).await;
                    let _ = sender.send(result);
                }))
            },
        ) {
            Ok(lease) => lease,
            Err(error) => return MexSessionRetirement::Retained(error),
        };
        let response = receiver.await;
        let task_outcome = lease.join().await;
        match response {
            Ok(retirement) => {
                if task_outcome == crate::context::RuntimeLocalTaskOutcome::Cancelled {
                    MexSessionRetirement::Retained(runtime_error(
                        "MEX:Lifecycle",
                        "host cancelled MEX shutdown before command retirement",
                    ))
                } else {
                    retirement
                }
            }
            Err(_) => MexSessionRetirement::Retained(runtime_error(
                "MEX:Lifecycle",
                "host stopped before MEX shutdown completed",
            )),
        }
    }

    async fn shutdown_gracefully_inner(&self, runtime: RuntimeContext) -> MexSessionRetirement {
        let abandoned_operation = self.join_inflight_operations().await;
        if let Err(error) = self.settle_native_commands().await {
            return MexSessionRetirement::Retained(error);
        }
        let policy = *self.isolation_policy.borrow();
        let paths = self
            .isolated
            .borrow()
            .keys()
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        let mut first_error = abandoned_operation;
        let mut retained = false;
        for path in paths {
            let Some(client) = self.isolated.borrow().get(&path).cloned() else {
                continue;
            };
            let Ok(_activity) = self.enter_isolated(&path) else {
                retained = true;
                first_error.get_or_insert_with(|| {
                    runtime_error(
                        "MEX:Lifecycle",
                        format!(
                            "MEX function '{}' is active during shutdown",
                            path.display()
                        ),
                    )
                });
                continue;
            };
            let Ok(mut operation) = client.acquire() else {
                retained = true;
                first_error.get_or_insert_with(|| {
                    runtime_error(
                        "MEX:Lifecycle",
                        format!(
                            "isolated MEX transport '{}' is busy during shutdown",
                            path.display()
                        ),
                    )
                });
                continue;
            };
            let retirement = operation
                .client()
                .shutdown(runtime.clone(), policy.invocation_timeout)
                .await;
            match retirement {
                IsolatedMexRetirement::Retired(result) => {
                    self.isolated.borrow_mut().remove(&path);
                    if let Err(error) = result {
                        first_error.get_or_insert(error);
                    }
                }
                IsolatedMexRetirement::Retained(error) => {
                    retained = true;
                    first_error.get_or_insert(error);
                }
            }
        }
        if retained {
            // A live isolated host may still call back into native MEX. Keep
            // both callback dependencies and their token registrations until
            // the isolated command reaches an actual retirement boundary.
            return MexSessionRetirement::Retained(first_error.unwrap_or_else(|| {
                runtime_error("MEX:Lifecycle", "an isolated MEX host remains active")
            }));
        }
        match self.shutdown_native(runtime).await {
            MexSessionRetirement::Retained(error) => {
                retained = true;
                first_error.get_or_insert(error);
            }
            MexSessionRetirement::Retired(Err(error)) => {
                first_error.get_or_insert(error);
            }
            MexSessionRetirement::Retired(Ok(())) => {}
        }
        if retained {
            // A rejected native unload may still retain async-result handles.
            // Waiting for their Drop here would wait on the very unload that
            // was rejected. Keep the host authority for a later retry.
            return MexSessionRetirement::Retained(first_error.unwrap_or_else(|| {
                runtime_error("MEX:Lifecycle", "MEX shutdown has live native work")
            }));
        }
        if let Err(error) = self.settle_native_pending().await {
            retained = true;
            first_error.get_or_insert(error);
        }
        if self.abandoned.get() {
            retained = true;
            first_error.get_or_insert_with(|| {
                runtime_error("MEX:Lifecycle", "MEX host was abandoned during shutdown")
            });
        }
        if retained {
            MexSessionRetirement::Retained(first_error.unwrap_or_else(|| {
                runtime_error("MEX:Lifecycle", "MEX shutdown has live native work")
            }))
        } else {
            self.ordinary_modules.borrow_mut().clear();
            self.retire_native_lane_owner();
            MexSessionRetirement::Retired(first_error.map_or(Ok(()), Err))
        }
    }

    async fn settle_native_pending(&self) -> Result<(), RuntimeError> {
        let lane = self.native_lane.borrow().as_ref().cloned();
        if let Some(lane) = lane {
            lane.settle_pending_for(&self.native_lane_owner)
                .await
                .map_err(|error| {
                    native_runtime_error(
                        "MexShutdown",
                        format!("could not finish pending MEX native work: {error}"),
                        error,
                    )
                })?;
        }
        Ok(())
    }

    async fn settle_native_commands(&self) -> Result<(), RuntimeError> {
        let lane = self.native_lane.borrow().as_ref().cloned();
        if let Some(lane) = lane {
            lane.settle_commands_for(&self.native_lane_owner)
                .await
                .map_err(|error| {
                    native_runtime_error(
                        "MexShutdown",
                        format!("could not finish pending MEX native command: {error}"),
                        error,
                    )
                })?;
        }
        Ok(())
    }

    async fn join_inflight_operations(&self) -> Option<RuntimeError> {
        let leases = self.inflight_operations.borrow().clone();
        let mut first_error = None;
        for lease in leases {
            if lease.join().await == crate::context::RuntimeLocalTaskOutcome::Cancelled {
                first_error.get_or_insert_with(|| {
                    runtime_error(
                        "MEX:Lifecycle",
                        "MEX host cancelled an admitted invocation before its reply was settled",
                    )
                });
            }
        }
        self.inflight_operations.borrow_mut().clear();
        first_error
    }

    fn native_lane(&self) -> Result<Rc<NativeMexLane>, RuntimeError> {
        if self.native_lane_owner_retired.get() || self.abandoned.get() {
            return Err(runtime_error(
                "MEX:NativeLane",
                "the MEX session has retired",
            ));
        }
        if let Some(lane) = self.native_lane.borrow().as_ref() {
            return Ok(lane.clone());
        }
        let lane = if let Some(parent) = &self.parent_native_lane {
            parent.lane()?
        } else {
            Rc::new(NativeMexLane::spawn().map_err(|error| runtime_error("MEX:NativeLane", error))?)
        };
        lane.register_session_owner(&self.native_lane_owner)
            .map_err(|error| runtime_error("MEX:NativeLane", error.to_string()))?;
        *self.native_lane.borrow_mut() = Some(lane.clone());
        Ok(lane)
    }

    fn retire_native_lane_owner(&self) {
        if self.native_lane_owner_retired.replace(true) {
            return;
        }
        let lane = self.native_lane.borrow_mut().take();
        if let Some(lane) = lane {
            lane.retire_session_owner(&self.native_lane_owner);
        }
    }

    /// Publish the token-table root before any input can create a token.
    #[cfg(test)]
    fn native_value_context(&self, path: &Path) -> Result<Rc<MxValueContext>, RuntimeError> {
        self.native_value_registration(path)
            .map(|value| value.context())
    }

    fn native_value_registration(
        &self,
        path: &Path,
    ) -> Result<Rc<RegisteredMxValueContext>, RuntimeError> {
        if let Some(values) = self.native_values.borrow().get(path) {
            return Ok(Rc::clone(values));
        }
        let values = Rc::new(
            RegisteredMxValueContext::new(Rc::new(MxValueContext::new()))
                .map_err(|error| runtime_error("MEX:Conversion", error.to_string()))?,
        );
        self.native_values
            .borrow_mut()
            .insert(path.to_path_buf(), Rc::clone(&values));
        Ok(values)
    }

    fn enter_isolated(&self, path: &Path) -> Result<IsolatedActivity<'_>, RuntimeError> {
        if !self.active_isolated.borrow_mut().insert(path.to_path_buf()) {
            return Err(runtime_error(
                "MEX:ReentrantInvocation",
                "recursive invocation of the same isolated MEX module is not supported",
            ));
        }
        Ok(IsolatedActivity {
            active: &self.active_isolated,
            path: path.to_path_buf(),
        })
    }

    async fn isolated_module(
        &self,
        path: &Path,
        runtime: &RuntimeContext,
    ) -> Result<Rc<IsolatedClientSlot>, RuntimeError> {
        if let Some(client) = self.isolated.borrow().get(path) {
            return Ok(Rc::clone(client));
        }
        match IsolatedMexClient::spawn(
            Some(runtime),
            path.to_path_buf(),
            Arc::clone(&self.isolated_ready),
            self.process_group_placement,
        )
        .await
        {
            Ok(client) => {
                let owner = IsolatedClientSlot::new(client);
                self.isolated
                    .borrow_mut()
                    .insert(path.to_path_buf(), Rc::clone(&owner));
                self.isolated_ready.notify_one();
                Ok(owner)
            }
            Err(error) => Err(wire_runtime_error(error)),
        }
    }

    async fn invoke_isolated(
        &self,
        path: &Path,
        arguments: &crate::call::RootedFunctionArguments,
        requested_outputs: usize,
        api: Option<MexApi>,
        tier: MexBinaryTier,
        runtime: RuntimeContext,
        timeout: Option<std::time::Duration>,
    ) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
        let _activity = self.enter_isolated(path)?;
        let client = self.isolated_module(path, &runtime).await?;
        let mut operation = client.acquire().map_err(|_| {
            runtime_error(
                "MEX:Lifecycle",
                format!(
                    "isolated MEX transport '{}' is busy during invocation",
                    path.display()
                ),
            )
        })?;
        let result = operation
            .client()
            .invoke(arguments, requested_outputs, api, tier, runtime, timeout)
            .await;
        match result {
            Ok(invocation) => finish_invocation(invocation, requested_outputs),
            Err(error) => Err(error),
        }
    }

    async fn clear_in_process(
        &self,
        request: &DynamicFunctionClearRequest,
        runtime: RuntimeContext,
    ) -> Result<(), RuntimeError> {
        let Some(lane) = self.native_lane.borrow().as_ref().cloned() else {
            return Ok(());
        };
        let ordinary_selected = self.ordinary_selected_private_paths(request);
        let paths = self
            .native_modules
            .borrow()
            .iter()
            .filter(|path| selected_by_request(path, request) || ordinary_selected.contains(*path))
            .cloned()
            .collect::<Vec<_>>();
        let mut first_error = None;
        for path in paths {
            let Some(registration) = self.native_values.borrow().get(&path).cloned() else {
                first_error.get_or_insert_with(|| {
                    runtime_error(
                        "MEX:Lifecycle",
                        format!(
                            "loaded MEX module '{}' has no registered token context",
                            path.display()
                        ),
                    )
                });
                continue;
            };
            let metadata = match lane.load_for(&self.native_lane_owner, &path).await {
                Ok(metadata) => metadata,
                Err(error) => {
                    first_error.get_or_insert_with(|| runtime_error("MexClear", error.to_string()));
                    continue;
                }
            };
            let host = Rc::new(RuntimeMexHostServices::new(runtime.clone()));
            let codec = Rc::new(MxBoundaryCodec::new_with_origin_lease(
                registration.context(),
                metadata.mode,
                metadata.interface,
                registration,
            ));
            let services = Rc::new(super::services::OriginMexBoundaryAdapter::new(host, codec));
            match lane
                .clear_for(&self.native_lane_owner, &path, services)
                .await
            {
                Ok(true) => {
                    self.native_modules.borrow_mut().remove(&path);
                    self.release_native_value_context(&path);
                }
                Ok(false) => {}
                Err(error) => {
                    first_error.get_or_insert_with(|| {
                        native_runtime_error(
                            "MexClear",
                            format!("could not clear MEX function '{}': {error}", path.display()),
                            error,
                        )
                    });
                }
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    async fn shutdown_native(&self, runtime: RuntimeContext) -> MexSessionRetirement {
        let Some(lane) = self.native_lane.borrow().as_ref().cloned() else {
            self.release_all_native_value_contexts();
            return MexSessionRetirement::Retired(Ok(()));
        };
        let paths = self
            .native_modules
            .borrow()
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let mut first_error = None;
        let mut retained = false;
        for path in paths {
            let Some(registration) = self.native_values.borrow().get(&path).cloned() else {
                retained = true;
                first_error.get_or_insert_with(|| {
                    runtime_error(
                        "MexShutdown",
                        format!(
                            "MEX function '{}' has no registered token context",
                            path.display()
                        ),
                    )
                });
                continue;
            };
            let metadata = match lane.load_for(&self.native_lane_owner, &path).await {
                Ok(metadata) => metadata,
                Err(error) => {
                    retained = true;
                    first_error
                        .get_or_insert_with(|| runtime_error("MexShutdown", error.to_string()));
                    continue;
                }
            };
            let host = Rc::new(RuntimeMexHostServices::new(runtime.clone()));
            let codec = Rc::new(MxBoundaryCodec::new_with_origin_lease(
                registration.context(),
                metadata.mode,
                metadata.interface,
                registration,
            ));
            let services = Rc::new(super::services::OriginMexBoundaryAdapter::new(host, codec));
            match lane
                .shutdown_module_for(&self.native_lane_owner, &path, services)
                .await
            {
                Ok(runmat_mex::MexModuleRetirement::Retired(result)) => {
                    self.native_modules.borrow_mut().remove(&path);
                    self.release_native_value_context(&path);
                    if let Err(error) = result {
                        first_error.get_or_insert_with(|| {
                            native_runtime_error(
                                "MexShutdown",
                                format!(
                                    "could not shut down MEX function '{}': {error}",
                                    path.display()
                                ),
                                error,
                            )
                        });
                    }
                }
                Ok(runmat_mex::MexModuleRetirement::Retained(error)) | Err(error) => {
                    retained = true;
                    first_error.get_or_insert_with(|| {
                        native_runtime_error(
                            "MexShutdown",
                            format!(
                                "could not shut down MEX function '{}': {error}",
                                path.display()
                            ),
                            error,
                        )
                    });
                }
            }
        }
        if retained {
            MexSessionRetirement::Retained(first_error.unwrap_or_else(|| {
                runtime_error("MexShutdown", "MEX native module remains active")
            }))
        } else {
            MexSessionRetirement::Retired(first_error.map_or(Ok(()), Err))
        }
    }

    fn release_native_value_context(&self, path: &Path) {
        // The native lane has completed mexAtExit/clear before this registration
        // retires. Drop outside the map borrow: retirement observers can reenter.
        let retired = self.native_values.borrow_mut().remove(path);
        drop(retired);
    }

    fn release_all_native_value_contexts(&self) {
        // Likewise, all native shutdown callbacks have finished first.
        let retired = std::mem::take(&mut *self.native_values.borrow_mut());
        drop(retired);
    }
}

impl super::super::ForeignAdapter for MexRuntimeSession {
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        self.visit_session_retained_values(visitor)
    }

    fn descriptor(&self) -> super::super::ForeignAdapterDescriptor {
        let mut capabilities = BTreeSet::from([
            CapabilityRequirement::ForeignRuntime,
            CapabilityRequirement::NativeCode,
        ]);
        if runmat_accelerate_api::provider_for_native_device(
            runmat_accelerate_api::NativeDeviceApi::Cuda,
        )
        .is_some()
        {
            capabilities.insert(CapabilityRequirement::Accelerator);
        }
        super::super::ForeignAdapterDescriptor {
            adapter: runmat_types::ForeignAdapterId::new(runmat_mex::MEX_ADAPTER_ID)
                .expect("the built-in MEX adapter identity is valid"),
            version: runmat_mex::MEX_ADAPTER_VERSION,
            capabilities,
            foreign_capabilities: BTreeSet::from([
                ForeignCapability::Invoke,
                ForeignCapability::Callback,
                ForeignCapability::Transfer,
            ]),
            artifact_identities: self
                .installed_modules
                .borrow()
                .values()
                .map(|artifact| artifact.identity.clone())
                .collect(),
            supports_wasm: false,
            supports_host_bridge: false,
            execution_stack: runmat_types::ExecutionStackRequirement::Process,
        }
    }

    fn invoke(
        &self,
        _context: RuntimeContext,
        _call: ForeignCall,
    ) -> super::super::ForeignAdapterFuture {
        Box::pin(async {
            crate::context::ForeignCallCompletion::invoked(Err(runtime_error(
                "MEX:InvalidDispatch",
                "MEX modules are invoked through function resolution",
            )))
        })
    }
}

struct ShutdownGuard(Rc<Cell<bool>>);

impl Drop for ShutdownGuard {
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
    invocation: RootedMexInvocation,
    requested_outputs: usize,
) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
    let RootedMexInvocation {
        outputs,
        warnings,
        console,
    } = invocation;
    if !console.is_empty() {
        crate::console::record_console_output(crate::console::ConsoleStream::Stdout, console)?;
    }
    for warning in warnings {
        crate::warning_store::push(
            warning
                .identifier
                .as_deref()
                .unwrap_or("RunMat:MEX:Warning"),
            &warning.message,
        );
    }
    if outputs.values().len() < requested_outputs {
        return Err(runtime_error(
            "MEX:OutputCount",
            format!(
                "MEX invocation produced {} outputs, but {requested_outputs} were requested",
                outputs.values().len()
            ),
        ));
    }
    outputs.with_owned_output_storage(|mut selected| {
        selected.truncate(requested_outputs);
        let sequence = match requested_outputs {
            0 => Ok(runmat_value::ValueSequence::empty()),
            1 => super::super::single_output_sequence(
                selected
                    .try_into_first()
                    .map_err(|_| {
                        crate::sequence::sequence_error_to_runtime(
                            runmat_value::ValueSequenceError::SharedOutputStorage,
                        )
                    })?
                    .expect("validated one MEX output"),
            ),
            _ => runmat_value::ValueSequence::from_output_storage(selected)
                .map_err(crate::sequence::sequence_error_to_runtime),
        }?;
        crate::execution::RootedValueSequence::new(sequence)
            .map_err(|error| error.into_runtime_error())
    })
}

fn decode_native_invocation(
    invocation: MexNativeInvocation,
    values: &MxValueContext,
) -> Result<RootedMexInvocation, RuntimeError> {
    let mut outputs =
        crate::execution::RootedValuesBuilder::with_capacity(invocation.outputs.len());
    for output in &invocation.outputs {
        let decoded = values
            .decode(output)
            .map_err(|error| runtime_error("MEX:Conversion", error.to_string()))?;
        outputs
            .push(decoded)
            .map_err(|error| error.into_runtime_error())?;
    }
    Ok(RootedMexInvocation {
        outputs: outputs
            .finish()
            .map_err(|error| error.into_runtime_error())?,
        warnings: invocation.warnings,
        console: invocation.console,
    })
}

pub(in crate::foreign::mex) fn wire_runtime_error(error: MexWireError) -> RuntimeError {
    let mut message = error.message.clone();
    if let Some(dependency) = &error.dependency {
        message.push_str(&format!(" (dependency: {dependency})"));
    }
    build_runtime_error(message)
        .with_identifier(error.identifier.clone())
        .with_source(error)
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

fn admit_manifested_native_timeout(
    timeout: Option<std::time::Duration>,
) -> Result<(), RuntimeError> {
    if timeout.is_some() {
        let mut error = runtime_error(
            "MEX:TimeoutUnsupported",
            "foreign.mex.timeout_ms cannot be enforced for manifested in-process MEX code; bounded deadlines require a supported isolated execution route. Remove the timeout only to explicitly allow native execution without a hard deadline",
        );
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        return Err(error);
    }
    Ok(())
}

#[cfg(not(target_family = "wasm"))]
fn native_runtime_error(
    identifier: &str,
    message: impl Into<String>,
    source: runmat_mex::MexLoadError,
) -> RuntimeError {
    build_runtime_error(message)
        .with_identifier(format!("RunMat:{identifier}"))
        .with_source(source)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::RuntimeExecutionService;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn manifested_native_timeout_admission_preserves_unbounded_default() {
        assert!(
            admit_manifested_native_timeout(MexIsolationPolicy::default().invocation_timeout)
                .is_ok()
        );
        for timeout in [
            std::time::Duration::from_nanos(1),
            std::time::Duration::from_secs(60),
        ] {
            let error = admit_manifested_native_timeout(Some(timeout)).unwrap_err();
            assert_eq!(error.identifier(), Some("RunMat:MEX:TimeoutUnsupported"));
            assert_eq!(error.gpu_gather_retry, crate::GpuGatherRetry::Never);
            assert!(error.message().contains("isolated"));
        }
        let directory = tempfile::tempdir().unwrap();
        let module = directory.path().join("must_not_load.mex");
        // Presence chooses the manifested route; neither these invalid bytes
        // nor a nonexistent module may reach the native loader under a deadline.
        std::fs::write(
            runmat_mex::MexArtifactManifest::path_for_module(&module),
            b"must not be parsed or loaded",
        )
        .unwrap();
        let session = MexRuntimeSession::new();
        session.set_isolation_policy(MexIsolationPolicy {
            invocation_timeout: Some(std::time::Duration::from_millis(1)),
            ..MexIsolationPolicy::default()
        });
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let result = futures::executor::block_on(session.load_selected_and_call(
            module,
            crate::call::RootedFunctionArguments::ordinary(
                crate::execution::RootedValues::new(Vec::new()).unwrap(),
            ),
            0,
            runtime,
        ))
        .unwrap()
        .unwrap_err();
        assert_eq!(result.identifier(), Some("RunMat:MEX:TimeoutUnsupported"));
        assert!(session.native_lane.borrow().is_none());
        assert!(session.native_modules.borrow().is_empty());
        assert!(session.native_values.borrow().is_empty());
    }

    #[test]
    fn selected_mex_outputs_are_rooted_before_decoded_batch_retires() {
        runmat_gc::gc_test_context(|| {
            let first = runmat_gc::gc_allocate_rooted(Value::String("first".into()))
                .expect("allocate first output");
            let second = runmat_gc::gc_allocate_rooted(Value::String("second".into()))
                .expect("allocate second output");
            let handles = [first.handle(), second.handle()];
            let outputs = crate::execution::RootedValues::new(
                handles
                    .iter()
                    .map(|handle| {
                        Value::HandleObject(runmat_value::HandleRef {
                            class_name: "MexOutput".into(),
                            target: *handle,
                            valid: true,
                        })
                    })
                    .collect(),
            )
            .expect("root decoded outputs");
            first.unroot().expect("transfer first allocation root");
            second.unroot().expect("transfer second allocation root");

            let expected_addresses = handles.map(|handle| runmat_gc::gc_handle_addr(&handle));
            let observed = Arc::new(AtomicBool::new(false));
            let witness = Arc::clone(&observed);
            let _observer = runmat_gc::RootRetirementObserver::register(Arc::new(move |event| {
                if let runmat_gc::RootRetirement::Explicit(handle) = event {
                    if expected_addresses.contains(&runmat_gc::gc_handle_addr(&handle))
                        && !witness.swap(true, Ordering::SeqCst)
                    {
                        runmat_gc::gc_collect_major()
                            .expect("collect during decoded-root retirement");
                        assert!(runmat_gc::gc_clone_value(&handle).is_ok());
                    }
                }
            }))
            .expect("register retirement witness");

            let sequence = finish_invocation(
                RootedMexInvocation {
                    outputs,
                    warnings: Vec::new(),
                    console: String::new(),
                },
                2,
            )
            .expect("finish rooted MEX result");
            assert!(observed.load(Ordering::SeqCst));
            for handle in handles {
                assert!(runmat_gc::gc_clone_value(&handle).is_ok());
            }
            drop(sequence);
        });
    }

    #[test]
    fn native_name_observation_selects_path_without_loading_module() {
        let Some(suffix) = runmat_mex::mex_suffix() else {
            return;
        };
        let temporary = tempfile::tempdir().expect("temporary MEX search root");
        let module = temporary
            .path()
            .join(format!("__native_observe_fixture__.{suffix}"));
        std::fs::write(&module, b"not a loadable MEX").expect("candidate bytes");
        let search = Arc::new(crate::builtins::common::path_state::SearchPath::new(
            temporary.path().to_string_lossy().into_owned(),
        ));
        let runtime =
            RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_search_path(search);
        let _scope = crate::context::RuntimeContextGuard::enter(runtime);
        let owner = MexRuntimeSession::new();
        let observed =
            futures::executor::block_on(owner.observe_name("__native_observe_fixture__"))
                .expect("name search")
                .expect("candidate exists");
        assert!(matches!(observed, MexNameSelection::SearchPath(_)));
        assert_eq!(observed.path(), module.as_path());
        assert!(owner.native_modules.borrow().is_empty());
        assert!(owner.installed_modules.borrow().is_empty());
    }

    #[test]
    fn selected_closed_source_cannot_observe_an_ambient_mex_candidate() {
        let Some(suffix) = runmat_mex::mex_suffix() else {
            return;
        };
        let temporary = tempfile::tempdir().expect("temporary MEX search root");
        let module = temporary
            .path()
            .join(format!("closed_mex_fixture.{suffix}"));
        std::fs::write(&module, b"candidate only").expect("candidate bytes");
        let search = Arc::new(crate::builtins::common::path_state::SearchPath::new(
            temporary.path().to_string_lossy().into_owned(),
        ));
        let owner = Rc::new(MexRuntimeSession::new());
        let runtime =
            RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_search_path(search);
        runtime.set_native_name_observer(Some(owner.name_observer()));
        let closed = runtime.with_selected_native_route(None);
        let _scope = crate::context::RuntimeContextGuard::enter(closed);
        let observed = futures::executor::block_on(crate::user_functions::observe_runtime_name(
            "closed_mex_fixture",
            crate::user_functions::RuntimeNameQuery {
                route: crate::user_functions::RuntimeNameRoute::DynamicCall,
                filter: crate::user_functions::RuntimeNameFilter::File,
                all: false,
            },
        ))
        .expect("selected name observation");
        assert!(observed.selected().is_none());
    }

    #[test]
    fn cancelled_invocation_never_reaches_discovery_or_native_code() {
        let cancellation = runmat_execution::cancellation::CancellationSource::new();
        let runtime = RuntimeContext::with_cancellation(
            Rc::new(RuntimeExecutionService::new()),
            cancellation.token(),
        );
        cancellation.cancel(runmat_execution::CancellationReason::User);
        let result = futures::executor::block_on(Rc::new(MexRuntimeSession::new()).load_and_call(
            "module_that_must_not_be_resolved",
            crate::call::RootedFunctionArguments::ordinary(
                crate::execution::RootedValues::new(Vec::new()).expect("empty arguments"),
            ),
            0,
            runtime,
        ))
        .break_value()
        .expect("pre-cancelled MEX call is handled")
        .expect_err("pre-cancelled MEX call fails");

        assert_eq!(result.identifier(), Some("RunMat:Cancelled"));

        let available = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let text = String::from("selected MEX argument owner");
        let payload = text.as_ptr();
        let missing = futures::executor::block_on(
            Rc::new(MexRuntimeSession::new()).load_installed_and_call(
                "module_not_installed_in_this_owner",
                crate::call::RootedFunctionArguments::ordinary(
                    crate::execution::RootedValues::in_context(
                        vec![Value::String(text.into())],
                        &available,
                    )
                    .expect("root the original MEX argument"),
                ),
                0,
                available,
            ),
        );
        let std::ops::ControlFlow::Continue(arguments) = missing else {
            panic!("missing installed MEX returns original arguments");
        };
        assert_eq!(arguments.len(), 1);
        let Some(Value::String(text)) = arguments.first() else {
            panic!("missing MEX returned original string argument");
        };
        assert_eq!(text.as_str().expect("ordinary text").as_ptr(), payload);
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
