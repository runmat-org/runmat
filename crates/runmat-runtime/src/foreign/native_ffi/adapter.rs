use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use runmat_builtins::{NativeFfiOperation, NativeLoadOption};
use runmat_native_ffi::{
    artifact_identity, invoke_symbol_with_bindings, prepare_header_with_declarations_report,
    prepare_native_interface, CallbackBinding, HeaderPreparation, InvocationArguments,
    InvocationValue, NativeInterfaceArtifactManifest, NativeInterfacePreparation,
    NativeLibraryMetadata, NativePointerResource, NativeScalar, NativeType, ParameterDirection,
    PointerBinding, PointerOwnership, SymbolPrototype, NATIVE_FFI_METADATA_SCHEMA_VERSION,
};

impl InvocationArguments for crate::call::RootedFunctionArguments {
    fn len(&self) -> usize {
        crate::call::RootedFunctionArguments::len(self)
    }
    fn get(&self, index: usize) -> Option<&Value> {
        crate::call::RootedFunctionArguments::get(self, index)
    }
}

struct ArgumentWindow<'a, A: InvocationArguments + ?Sized> {
    source: &'a A,
    offset: usize,
}

impl<A: InvocationArguments + ?Sized> InvocationArguments for ArgumentWindow<'_, A> {
    fn len(&self) -> usize {
        self.source.len().saturating_sub(self.offset)
    }
    fn get(&self, index: usize) -> Option<&Value> {
        self.source.get(self.offset.checked_add(index)?)
    }
}
pub use runmat_native_ffi::{NATIVE_FFI_ADAPTER_ID, NATIVE_FFI_ADAPTER_VERSION};
use runmat_process_host::shared_memory::SharedSnapshotParent;
use runmat_process_host::{ContainedChildRetirement, ProcessGroupPlacement, ProcessSpawnEvidence};
use runmat_types::{
    CapabilityRequirement, ForeignAffinity, ForeignCapability, ForeignLifetime, ForeignOwnership,
    ForeignTypeIdentity,
};
use runmat_value::{CellArray, ForeignResourceKey, Value};

use super::super::{
    foreign_error, foreign_error_with_source, ForeignAdapter, ForeignAdapterDescriptor,
    ForeignAdapterFuture, ForeignErrorKind, ForeignExecutionPolicy, ForeignHandleRegistry,
    ForeignHostRegistration, ForeignHostRelease, ForeignResourceMetadata,
};
use super::request::{
    default_alias, invalid_call, legacy_pointer_type, optional_string_argument, string_argument,
};
use super::session::{LibraryEntry, NativeFfiSessionState, OpaquePointerView, PointerEntry};
use super::{
    NativeClientRetirement, NativeFfiHostClient, NativeFfiIsolationPolicy,
    NativeProjectLoadRequest, NativeWireError,
};
use crate::context::{ForeignCall, ForeignCallCompletion, RuntimeContext};
use crate::RuntimeError;

mod call;
mod child_inventory;
mod direct_snapshot;
mod library;
pub(super) use library::PreparedLibraryReadError;
mod pointer_roles;
pub(crate) use pointer_roles::declared_pointer_method;
mod pointer_storage;
mod pointer_view;
mod project_catalog;
mod project_replacement;
mod project_retirement;

pub use project_replacement::{
    NativeProjectArtifactExpectation, PreparedNativeProjectReplacement,
    ReadyNativeProjectReplacement,
};
pub use project_retirement::VerifiedNativeProjectRetirement;

#[cfg(test)]
use call::requested_output_sequence;
pub(crate) use call::{is_callable, isolated_callback_value};
use direct_snapshot::DirectNativeSnapshot;
use pointer_view::{
    declared_pointer_type, default_structure_value, foreign_ownership, pointer_metadata,
    pointer_type_identity,
};
use project_catalog::NativeProjectCatalog;

static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
struct LegacyLoadReport {
    alias: String,
    notfound: Vec<String>,
    warnings: String,
}

struct ResolvedPointerType {
    pointee: NativeType,
    metadata: Rc<NativeLibraryMetadata>,
    library: Option<Rc<runmat_native_ffi::LoadedLibrary>>,
}

#[derive(Debug, Default)]
struct ReleaseQueue(Mutex<Vec<u64>>);

impl ForeignHostRelease for ReleaseQueue {
    fn release(&self, key: &ForeignResourceKey) {
        if let Ok(mut queue) = self.0.lock() {
            queue.push(key.handle);
        }
    }
}

pub struct NativeFfiAdapter {
    handles: ForeignHandleRegistry,
    host_identity: String,
    state: RefCell<NativeFfiSessionState>,
    released: Arc<ReleaseQueue>,
    callback_router: Option<Rc<dyn NativeFfiCallbackRouter>>,
    isolation_policy: Cell<NativeFfiIsolationPolicy>,
    isolated_client: Rc<RefCell<Option<NativeFfiHostClient>>>,
    isolated_call_active: Rc<Cell<bool>>,
    bootstrap_unverified: ProcessSpawnEvidence,
    project_catalog: Rc<RefCell<NativeProjectCatalog>>,
    bootstrap_state: Rc<Cell<NativeBootstrapState>>,
    project_retirement: super::super::project_retirement::ProjectRetirementAuthority,
    prepared_artifact_identities: RefCell<BTreeSet<String>>,
    compiler_frontend: PathBuf,
    process_group_placement: Cell<ProcessGroupPlacement>,
    snapshot_parent: RefCell<Option<SharedSnapshotParent>>,
    private_byte_lease: RefCell<Option<runmat_process_host::private_bytes::PrivateByteLease>>,
    private_root_origin: Cell<NativePrivateRootOrigin>,
    direct_snapshot_sequence: Cell<u64>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum NativePrivateRootOrigin {
    None,
    AdmittedRoot,
    DirectSnapshot,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum NativeBootstrapState {
    Unstarted,
    InProgress,
    Ready,
    Failed,
}

/// Native isolation must retain the same child when shutdown did not prove
/// its owned process tree retired.
pub enum NativeFfiRetirement {
    Retained(RuntimeError),
    Retired {
        scope: Option<ContainedChildRetirement>,
        result: Result<(), RuntimeError>,
    },
}

#[derive(Debug)]
struct NativeRetirementSource(NativeWireError);

impl std::fmt::Display for NativeRetirementSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.0.identifier, self.0.message)
    }
}

impl std::error::Error for NativeRetirementSource {}

fn native_retirement_error(error: NativeWireError) -> RuntimeError {
    crate::build_runtime_error(error.message.clone())
        .with_builtin("native_ffi")
        .with_identifier(error.identifier.clone())
        .with_source(NativeRetirementSource(error))
        .build()
}

struct IsolatedClientRetirementGuard {
    slot: Rc<RefCell<Option<NativeFfiHostClient>>>,
    client: Option<NativeFfiHostClient>,
}

impl Drop for IsolatedClientRetirementGuard {
    fn drop(&mut self) {
        if let Some(client) = self.client.take() {
            // No RefCell borrow spans shutdown await. The active-call guard
            // excludes another invocation or retirement until restoration.
            *self.slot.borrow_mut() = Some(client);
        }
    }
}

struct IsolatedCallActiveGuard(Rc<Cell<bool>>);

impl Drop for IsolatedCallActiveGuard {
    fn drop(&mut self) {
        self.0.set(false);
    }
}

pub trait NativeFfiCallbackRouter {
    fn visit_retained_values(&self, _visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        None
    }
    fn dispatch(
        &self,
        callback_id: u64,
        arguments: &[Value],
    ) -> Result<crate::execution::RootedValue, RuntimeError>;
}

impl std::fmt::Debug for NativeFfiAdapter {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeFfiAdapter")
            .field("host_identity", &self.host_identity)
            .field("state", &self.state)
            .finish_non_exhaustive()
    }
}

impl NativeFfiAdapter {
    /// Only the native host's internal adapter executes synchronously. Public
    /// in-process calls use its dedicated OS thread; untrusted calls use a
    /// contained process. Both hosts share the same invocation protocol.
    fn uses_host(&self, policy: NativeFfiIsolationPolicy) -> bool {
        policy.isolate || self.callback_router.is_none()
    }

    pub fn new(handles: ForeignHandleRegistry) -> Result<Rc<Self>, RuntimeError> {
        Self::new_with_callback_router(handles, None)
    }

    pub fn new_in_process_group(
        handles: ForeignHandleRegistry,
        placement: ProcessGroupPlacement,
        snapshot_parent: Option<SharedSnapshotParent>,
    ) -> Result<Rc<Self>, RuntimeError> {
        let adapter = Self::new(handles)?;
        adapter.process_group_placement.set(placement);
        *adapter.snapshot_parent.borrow_mut() = snapshot_parent;
        Ok(adapter)
    }

    pub(crate) fn new_with_callback_router(
        handles: ForeignHandleRegistry,
        callback_router: Option<Rc<dyn NativeFfiCallbackRouter>>,
    ) -> Result<Rc<Self>, RuntimeError> {
        Self::new_with_callback_router_and_frontend(handles, callback_router, "clang".into())
    }

    pub(crate) fn new_with_callback_router_and_frontend(
        handles: ForeignHandleRegistry,
        callback_router: Option<Rc<dyn NativeFfiCallbackRouter>>,
        compiler_frontend: PathBuf,
    ) -> Result<Rc<Self>, RuntimeError> {
        let sequence = NEXT_SESSION.fetch_add(1, Ordering::Relaxed);
        let host_identity = format!("native-ffi-{sequence}");
        let released = Arc::new(ReleaseQueue::default());
        handles.register_host(ForeignHostRegistration {
            identity: host_identity.clone(),
            adapter: NATIVE_FFI_ADAPTER_ID.to_string(),
            session_identity: host_identity.clone(),
            capabilities: BTreeSet::from([
                ForeignCapability::Invoke,
                ForeignCapability::Read,
                ForeignCapability::Write,
                ForeignCapability::Callback,
                ForeignCapability::Transfer,
            ]),
            policy: ForeignExecutionPolicy::trusted_in_process(),
            release: released.clone(),
        })?;
        Ok(Rc::new(Self {
            handles,
            host_identity,
            state: RefCell::new(NativeFfiSessionState::default()),
            released,
            callback_router,
            isolation_policy: Cell::new(NativeFfiIsolationPolicy::in_process()),
            isolated_client: Rc::new(RefCell::new(None)),
            isolated_call_active: Rc::new(Cell::new(false)),
            bootstrap_unverified: ProcessSpawnEvidence::default(),
            project_catalog: Rc::new(RefCell::new(NativeProjectCatalog::default())),
            bootstrap_state: Rc::new(Cell::new(NativeBootstrapState::Unstarted)),
            project_retirement:
                super::super::project_retirement::ProjectRetirementAuthority::default(),
            prepared_artifact_identities: RefCell::new(BTreeSet::new()),
            compiler_frontend,
            process_group_placement: Cell::new(ProcessGroupPlacement::OwnGroup),
            snapshot_parent: RefCell::new(None),
            private_byte_lease: RefCell::new(None),
            private_root_origin: Cell::new(NativePrivateRootOrigin::None),
            direct_snapshot_sequence: Cell::new(0),
        }))
    }

    pub fn set_isolation_policy(
        &self,
        policy: NativeFfiIsolationPolicy,
    ) -> Result<(), RuntimeError> {
        let revision = self.idle_project_identity()?;
        if !self.state.borrow().libraries.is_empty()
            || !self.project_catalog.borrow().is_empty()
            || self.isolated_client.borrow().is_some()
            || self.isolated_call_active.get()
        {
            return Err(invalid_call(
                "native-library isolation must be configured before loading an interface",
            ));
        }
        self.handles.set_host_policy(
            &self.host_identity,
            if policy.isolate {
                ForeignExecutionPolicy {
                    trust: super::super::ForeignTrust::Untrusted,
                    isolation: super::super::ForeignIsolation::IsolatedProcess,
                }
            } else {
                ForeignExecutionPolicy::trusted_in_process()
            },
        )?;
        self.isolation_policy.set(policy);
        self.project_retirement.publish_fresh(&revision);
        Ok(())
    }

    fn idle_project_identity(&self) -> Result<Rc<()>, RuntimeError> {
        let identity = self.project_retirement.snapshot();
        if !self.project_retirement.matches_idle(&identity) {
            return Err(invalid_call(
                "native project replacement authority is active",
            ));
        }
        Ok(identity)
    }

    pub fn isolation_policy(&self) -> NativeFfiIsolationPolicy {
        self.isolation_policy.get()
    }

    pub async fn shutdown(&self) -> Result<(), RuntimeError> {
        match self.shutdown_retirement(None).await {
            NativeFfiRetirement::Retained(error) => Err(error),
            NativeFfiRetirement::Retired { result, .. } => result,
        }
    }

    pub async fn shutdown_retirement(
        &self,
        context: Option<&RuntimeContext>,
    ) -> NativeFfiRetirement {
        if self.project_retirement.is_pending() {
            return NativeFfiRetirement::Retained(invalid_call(
                "native project retirement authority is active during terminal shutdown",
            ));
        }
        self.retire_isolated_client(true, context).await
    }

    async fn retire_isolated_client(
        &self,
        terminal: bool,
        context: Option<&RuntimeContext>,
    ) -> NativeFfiRetirement {
        if self.isolated_call_active.replace(true) {
            return NativeFfiRetirement::Retained(invalid_call(
                "native-library shutdown cannot begin during an active isolated call",
            ));
        }
        let _active = IsolatedCallActiveGuard(Rc::clone(&self.isolated_call_active));
        let client = self.isolated_client.borrow_mut().take();
        let Some(client) = client else {
            if self.bootstrap_unverified.is_unverified() {
                return NativeFfiRetirement::Retained(invalid_call(
                    "isolated native-library bootstrap tree retirement is unverified",
                ));
            }
            self.bootstrap_state.set(NativeBootstrapState::Unstarted);
            if terminal {
                self.clear_retired_state();
            }
            return NativeFfiRetirement::Retired {
                scope: None,
                result: Ok(()),
            };
        };
        let mut guard = IsolatedClientRetirementGuard {
            slot: Rc::clone(&self.isolated_client),
            client: Some(client),
        };
        let released_handles = self
            .released
            .0
            .lock()
            .map(|mut queue| std::mem::take(&mut *queue))
            .unwrap_or_default();
        let retirement = guard
            .client
            .as_mut()
            .expect("retirement guard owns exact isolated client")
            .shutdown_retirement(context, released_handles)
            .await;
        match retirement {
            NativeClientRetirement::Retained {
                retirement,
                prior_protocol,
            } => {
                let retirement = native_retirement_error(retirement);
                let error = match prior_protocol {
                    Some(prior) => crate::runtime_error::complete_with_cleanup::<()>(
                        Err(native_retirement_error(prior)),
                        Err(retirement),
                    )
                    .expect_err("unverified retirement supplies an error"),
                    None => retirement,
                };
                NativeFfiRetirement::Retained(error)
            }
            NativeClientRetirement::Retired { scope, result } => {
                drop(guard.client.take());
                self.bootstrap_state.set(NativeBootstrapState::Unstarted);
                if terminal {
                    self.clear_retired_state();
                }
                NativeFfiRetirement::Retired {
                    scope,
                    result: result.map_err(native_retirement_error),
                }
            }
        }
    }

    fn clear_retired_state(&self) {
        self.state.borrow_mut().libraries.clear();
        self.state.borrow_mut().pointers.clear();
        *self.project_catalog.borrow_mut() = NativeProjectCatalog::default();
        self.bootstrap_state.set(NativeBootstrapState::Unstarted);
        self.prepared_artifact_identities.borrow_mut().clear();
        if let Ok(mut released) = self.released.0.lock() {
            released.clear();
        }
        self.private_byte_lease.borrow_mut().take();
        self.private_root_origin.set(NativePrivateRootOrigin::None);
        self.direct_snapshot_sequence.set(0);
        let identity = self.project_retirement.snapshot();
        self.project_retirement.publish_fresh(&identity);
    }

    pub fn attach_private_byte_lease(
        &self,
        lease: runmat_process_host::private_bytes::PrivateByteLease,
    ) -> Result<(), RuntimeError> {
        let revision = self.idle_project_identity()?;
        if self.isolated_call_active.get()
            || self.bootstrap_unverified.is_unverified()
            || self.isolated_client.borrow().is_some()
            || self.private_byte_lease.borrow().is_some()
        {
            return Err(invalid_call(
                "native-library private byte lease cannot change after host admission",
            ));
        }
        *self.private_byte_lease.borrow_mut() = Some(lease);
        self.private_root_origin
            .set(NativePrivateRootOrigin::AdmittedRoot);
        self.project_retirement.publish_fresh(&revision);
        Ok(())
    }

    pub fn install_prepared_artifact(
        &self,
        library_path: &std::path::Path,
        manifest_path: &std::path::Path,
    ) -> Result<String, RuntimeError> {
        let revision = self.idle_project_identity()?;
        if self.uses_host(self.isolation_policy.get())
            && (self.isolated_call_active.get()
                || self.isolated_client.borrow().is_some()
                || self.bootstrap_state.get() != NativeBootstrapState::Unstarted)
        {
            return Err(invalid_call(
                "isolated prepared native artifacts must be installed before child startup",
            ));
        }
        let isolated = self.uses_host(self.isolation_policy.get());
        let mut direct_snapshot = if isolated
            && self.private_root_origin.get() != NativePrivateRootOrigin::AdmittedRoot
        {
            let sequence = self.direct_snapshot_sequence.get();
            self.direct_snapshot_sequence.set(
                sequence
                    .checked_add(1)
                    .ok_or_else(|| invalid_call("native direct snapshot sequence exhausted"))?,
            );
            Some(DirectNativeSnapshot::capture(
                self.private_byte_lease.borrow().clone(),
                sequence,
                library_path,
                manifest_path,
            )?)
        } else {
            None
        };
        let prepared = self.prepare_project_artifact(
            direct_snapshot
                .as_ref()
                .map_or(library_path, |snapshot| snapshot.library.as_path()),
            direct_snapshot
                .as_ref()
                .map_or(manifest_path, |snapshot| snapshot.manifest.as_path()),
            None,
        )?;
        self.validate_available_alias(&prepared.alias)?;
        let alias = prepared.alias;
        let addition = if let Some(load) = prepared.project_load {
            let lease = direct_snapshot
                .as_ref()
                .map(|snapshot| snapshot.lease.clone())
                .or_else(|| self.private_byte_lease.borrow().clone())
                .ok_or_else(|| {
                    invalid_call("isolated prepared native artifact lacks its private byte owner")
                })?;
            let mut addition = NativeProjectCatalog {
                loads: vec![load],
                lease: None,
            };
            addition.bind_private_root(lease.clone())?;
            Some((addition, lease))
        } else {
            None
        };
        if let Some(entry) = prepared.library {
            self.state
                .borrow_mut()
                .libraries
                .insert(alias.clone(), entry);
        }
        if let Some((addition, lease)) = addition {
            let mut catalog = self.project_catalog.borrow_mut();
            self.prepared_artifact_identities
                .borrow_mut()
                .insert(prepared.identity.to_string());
            catalog.loads.extend(addition.loads);
            catalog.lease = Some(lease.clone());
            if direct_snapshot.is_some() {
                *self.private_byte_lease.borrow_mut() = Some(lease);
                self.private_root_origin
                    .set(NativePrivateRootOrigin::DirectSnapshot);
            }
        }
        if let Some(snapshot) = direct_snapshot.as_mut() {
            snapshot.commit();
        }
        self.project_retirement.publish_fresh(&revision);
        Ok(alias)
    }

    fn reap_released(&self) {
        let handles = self
            .released
            .0
            .lock()
            .map(|mut queue| std::mem::take(&mut *queue))
            .unwrap_or_default();
        if handles.is_empty() {
            return;
        }
        let mut state = self.state.borrow_mut();
        for handle in handles {
            state.pointers.remove(&handle);
        }
    }

    fn invoke_operation(
        &self,
        context: RuntimeContext,
        call: ForeignCall,
    ) -> Result<runmat_value::ValueSequence, RuntimeError> {
        self.reap_released();
        let operation = NativeFfiOperation::from_symbol(&call.symbol).ok_or_else(|| {
            invalid_call(format!("unknown native FFI operation `{}`", call.symbol))
        })?;
        if operation == NativeFfiOperation::LoadReport {
            return self.load_report(&call.arguments);
        }
        if operation == NativeFfiOperation::Call {
            return self.call(context, &call.arguments, call.requested_outputs);
        }
        let value = match operation {
            NativeFfiOperation::Load => self.load(&call.arguments),
            NativeFfiOperation::LoadPrepared => self.load_prepared(&call.arguments),
            NativeFfiOperation::Unload => self.unload(&call.arguments),
            NativeFfiOperation::IsLoaded => self.is_loaded(&call.arguments),
            NativeFfiOperation::Functions => self.functions(&call.arguments),
            NativeFfiOperation::Pointer => self.pointer(&call.arguments),
            NativeFfiOperation::Structure => self.structure(&call.arguments),
            NativeFfiOperation::PointerValue => self.pointer_value(&call.arguments),
            NativeFfiOperation::GetMember => self.get_member(&call.arguments),
            NativeFfiOperation::SetMember => self.set_member(&context, &call.arguments),
            NativeFfiOperation::InvokeMember => self.invoke_member(&call.arguments),
            NativeFfiOperation::BuildInterface => self.build_interface(&call.arguments),
            NativeFfiOperation::LoadReport | NativeFfiOperation::Call => {
                unreachable!()
            }
        }?;
        super::super::single_output_sequence(value)
    }
}

impl Drop for NativeFfiAdapter {
    fn drop(&mut self) {
        // The ProcessHost child owns both snapshot and project bytes through
        // its single retirement-bound resource lease.
        if let Ok(mut slot) = self.isolated_client.try_borrow_mut() {
            let client = slot.take();
            drop(slot);
            drop(client);
        }
        let _ = self.handles.unregister_host(&self.host_identity);
    }
}

impl ForeignAdapter for NativeFfiAdapter {
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        if let Some(router) = &self.callback_router {
            router.visit_retained_values(visitor)?;
        }
        Some(())
    }
    fn descriptor(&self) -> ForeignAdapterDescriptor {
        let mut artifact_identities = self
            .state
            .borrow()
            .libraries
            .values()
            .map(|entry| entry.artifact_identity.clone())
            .collect::<BTreeSet<_>>();
        artifact_identities.extend(self.prepared_artifact_identities.borrow().iter().cloned());
        ForeignAdapterDescriptor {
            adapter: runmat_types::ForeignAdapterId::new(NATIVE_FFI_ADAPTER_ID)
                .expect("the built-in native FFI adapter identity is valid"),
            version: NATIVE_FFI_ADAPTER_VERSION,
            capabilities: BTreeSet::from([
                CapabilityRequirement::ForeignRuntime,
                CapabilityRequirement::NativeCode,
            ]),
            foreign_capabilities: BTreeSet::from([
                ForeignCapability::Invoke,
                ForeignCapability::Read,
                ForeignCapability::Write,
                ForeignCapability::Callback,
                ForeignCapability::Transfer,
            ]),
            artifact_identities: artifact_identities
                .into_iter()
                .map(|identity| {
                    runmat_types::ForeignArtifactIdentity::new(identity)
                        .expect("installed native interface identities are canonical")
                })
                .collect(),
            supports_wasm: false,
            supports_host_bridge: false,
            execution_stack: runmat_types::ExecutionStackRequirement::Process,
        }
    }

    fn invoke(&self, context: RuntimeContext, call: ForeignCall) -> ForeignAdapterFuture {
        // Spawn evidence stays set while a verified, owned child is live. It
        // blocks re-entry only when no client owns that child (for example,
        // containment failed after OS spawn or retirement was unverified).
        if self.bootstrap_unverified.is_unverified() && self.isolated_client.borrow().is_none() {
            return Box::pin(async {
                ForeignCallCompletion::invoked(Err(invalid_call(
                    "isolated native-library bootstrap has an unverified child",
                )))
            });
        }
        if self.project_retirement.is_pending() {
            return Box::pin(async {
                ForeignCallCompletion::invoked(Err(invalid_call(
                    "native project retirement is in progress",
                )))
            });
        }
        let policy = self.isolation_policy.get();
        if !self.uses_host(policy) {
            let result =
                crate::foreign::runtime::root_adapter_result(self.invoke_operation(context, call));
            return Box::pin(async move { ForeignCallCompletion::invoked(result) });
        }
        if !self.project_catalog.borrow().is_empty()
            && self.project_catalog.borrow().lease.is_none()
        {
            return Box::pin(async {
                ForeignCallCompletion::invoked(Err(invalid_call(
                    "isolated native project catalog lacks its private byte owner",
                )))
            });
        }
        if self.isolated_call_active.replace(true) {
            return Box::pin(async {
                ForeignCallCompletion::invoked(Err(invalid_call(
                    "recursive entry into the active isolated native-library host is not supported",
                )))
            });
        }
        let client_slot = Rc::clone(&self.isolated_client);
        let active = IsolatedCallActiveGuard(Rc::clone(&self.isolated_call_active));
        let catalog = Rc::clone(&self.project_catalog);
        let bootstrap_state = Rc::clone(&self.bootstrap_state);
        let handles = self.handles.clone();
        let host_identity = self.host_identity.clone();
        let released = Arc::clone(&self.released);
        let process_group_placement = self.process_group_placement.get();
        let snapshot_parent = self.snapshot_parent.borrow().clone();
        let bootstrap_unverified = self.bootstrap_unverified.clone();
        let project_bytes = catalog.borrow().lease.clone();
        Box::pin(async move {
            let result = async {
                let _call_roots =
                    crate::execution::RootedValueLease::for_values(call.arguments.iter())
                        .map_err(|error| error.into_runtime_error())?;
                let mut owner = IsolatedClientRetirementGuard {
                    slot: Rc::clone(&client_slot),
                    client: client_slot.borrow_mut().take(),
                };
                if owner.client.is_none() {
                    owner.client = Some(
                        if policy.isolate {
                            NativeFfiHostClient::spawn(
                                Some(&context),
                                process_group_placement,
                                bootstrap_unverified.clone(),
                                snapshot_parent,
                                project_bytes,
                            )
                            .await
                        } else {
                            NativeFfiHostClient::spawn_thread(snapshot_parent, project_bytes).await
                        }
                        .map_err(|error| {
                            crate::build_runtime_error(error.message)
                                .with_builtin("native_ffi")
                                .with_identifier(error.identifier)
                                .build()
                        })?,
                    );
                }
                let client = owner.client.as_mut().expect("isolated client initialized");
                let released_handles = released
                    .0
                    .lock()
                    .map(|mut queue| std::mem::take(&mut *queue))
                    .unwrap_or_default();
                if bootstrap_state.get() == NativeBootstrapState::Unstarted {
                    bootstrap_state.set(NativeBootstrapState::InProgress);
                    let loads = catalog.borrow().loads.clone();
                    for load in loads {
                        client
                            .begin_project_load()
                            .map_err(native_retirement_error)?;
                        if let Err(error) = client
                            .load_project(
                                context.clone(),
                                load,
                                &handles,
                                &host_identity,
                                policy.invocation_timeout,
                            )
                            .await
                        {
                            bootstrap_state.set(NativeBootstrapState::Failed);
                            return Err(error);
                        }
                    }
                    bootstrap_state.set(NativeBootstrapState::Ready);
                } else if bootstrap_state.get() != NativeBootstrapState::Ready {
                    return Err(invalid_call(
                        "isolated native project bootstrap requires verified retirement",
                    ));
                }
                let requested_outputs = call.requested_outputs;
                client.begin_invocation().map_err(|error| {
                    crate::build_runtime_error(error.message)
                        .with_builtin("native_ffi")
                        .with_identifier(error.identifier)
                        .build()
                })?;
                let outcome = client
                    .invoke(
                        context,
                        call,
                        &handles,
                        &host_identity,
                        released_handles,
                        policy.invocation_timeout,
                    )
                    .await
                    .and_then(|outputs| {
                        outputs.with_owned_output_storage(|values| {
                            isolated_outputs(values, requested_outputs)
                        })
                    });
                outcome
            }
            .await;
            drop(active);
            ForeignCallCompletion::invoked(result)
        })
    }

    fn is_isolated(&self) -> bool {
        self.isolation_policy.get().isolate
    }
}

fn isolated_outputs(
    outputs: runmat_value::OutputListStorage,
    requested_outputs: usize,
) -> Result<crate::execution::RootedValueSequence, RuntimeError> {
    let sequence = match (requested_outputs, outputs.len()) {
        (1, 1) => runmat_value::ValueSequence::single(
            outputs
                .try_into_first()
                .map_err(|_| {
                    crate::sequence::sequence_error_to_runtime(
                        runmat_value::ValueSequenceError::SharedOutputStorage,
                    )
                })?
                .expect("one isolated native FFI output"),
        )
        .map_err(|error| foreign_error_with_source(ForeignErrorKind::InvalidCall, error)),
        _ => runmat_value::ValueSequence::from_output_storage(outputs)
            .map_err(|error| foreign_error_with_source(ForeignErrorKind::InvalidCall, error)),
    }?;
    crate::execution::RootedValueSequence::new(sequence).map_err(|error| error.into_runtime_error())
}

#[cfg(test)]
mod output_sequence_tests {
    use super::*;

    #[test]
    fn requested_native_outputs_preserve_zero_one_and_many_cardinalities() {
        let source = vec![Value::Num(1.0), Value::Num(2.0), Value::Num(3.0)];
        assert!(requested_output_sequence(source.clone(), 0)
            .expect("zero outputs")
            .is_empty());
        assert_eq!(
            requested_output_sequence(source.clone(), 1)
                .expect("one output")
                .into_output_storage()
                .as_slice(),
            &[Value::Num(1.0)]
        );
        assert_eq!(
            requested_output_sequence(source.clone(), 3)
                .expect("three outputs")
                .into_output_storage()
                .as_slice(),
            source.as_slice()
        );
        assert!(requested_output_sequence(vec![Value::Num(1.0)], 2).is_err());
    }
}
