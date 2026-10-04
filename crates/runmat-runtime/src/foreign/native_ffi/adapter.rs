use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use runmat_builtins::{NativeFfiOperation, NativeLoadOption};
use runmat_native_ffi::{
    artifact_identity, invoke_symbol_with_bindings, prepare_header_with_declarations_report,
    prepare_native_interface, CallbackBinding, HeaderPreparation, InvocationValue,
    NativeInterfaceArtifactManifest, NativeInterfacePreparation, NativeLibraryMetadata,
    NativePointerResource, NativeScalar, NativeType, ParameterDirection, PointerBinding,
    PointerOwnership, SymbolPrototype, NATIVE_FFI_METADATA_SCHEMA_VERSION,
};
pub use runmat_native_ffi::{NATIVE_FFI_ADAPTER_ID, NATIVE_FFI_ADAPTER_VERSION};
use runmat_types::{
    CapabilityRequirement, ForeignAffinity, ForeignCapability, ForeignLifetime, ForeignOwnership,
    ForeignTypeIdentity,
};
use runmat_value::{CellArray, ForeignResourceKey, StructValue, Value};

use super::super::{
    foreign_error, foreign_error_with_source, ForeignAdapter, ForeignAdapterDescriptor,
    ForeignAdapterFuture, ForeignErrorKind, ForeignExecutionPolicy, ForeignHandleRegistry,
    ForeignHostRegistration, ForeignHostRelease, ForeignResourceMetadata,
};
use super::request::{
    default_alias, invalid_call, legacy_pointer_type, optional_string_argument, string_argument,
};
use super::session::{LibraryEntry, NativeFfiSessionState, OpaquePointerView, PointerEntry};
use super::{IsolatedNativeFfiClient, NativeFfiIsolationPolicy};
use crate::context::{ForeignCall, RuntimeContext};
use crate::RuntimeError;

mod call;
mod library;
mod pointer_storage;
mod pointer_view;

#[cfg(test)]
use call::requested_output_sequence;
pub(crate) use call::{is_callable, isolated_callback_value};
use pointer_view::{
    declared_pointer_type, default_structure_value, foreign_ownership, pointer_metadata,
    pointer_type_identity,
};

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
    isolated_client: Rc<RefCell<Option<IsolatedNativeFfiClient>>>,
    isolated_call_active: Rc<Cell<bool>>,
    pending_isolated_calls: Rc<RefCell<Vec<ForeignCall>>>,
    prepared_artifact_identities: RefCell<BTreeSet<String>>,
    compiler_frontend: PathBuf,
}

pub trait NativeFfiCallbackRouter {
    fn visit_retained_values(&self, _visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        None
    }
    fn dispatch(&self, callback_id: u64, arguments: &[Value]) -> Result<Value, String>;
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
    pub fn new(handles: ForeignHandleRegistry) -> Result<Rc<Self>, RuntimeError> {
        Self::new_with_callback_router(handles, None)
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
            pending_isolated_calls: Rc::new(RefCell::new(Vec::new())),
            prepared_artifact_identities: RefCell::new(BTreeSet::new()),
            compiler_frontend,
        }))
    }

    pub fn set_isolation_policy(
        &self,
        policy: NativeFfiIsolationPolicy,
    ) -> Result<(), RuntimeError> {
        if !self.state.borrow().libraries.is_empty()
            || !self.pending_isolated_calls.borrow().is_empty()
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
        Ok(())
    }

    pub async fn shutdown(&self) -> Result<(), RuntimeError> {
        if self.isolated_call_active.replace(true) {
            return Err(invalid_call(
                "native-library shutdown cannot begin during an active isolated call",
            ));
        }
        let result = async {
            let client = self.isolated_client.borrow_mut().take();
            let Some(mut client) = client else {
                self.state.borrow_mut().libraries.clear();
                return Ok(());
            };
            let released_handles = self
                .released
                .0
                .lock()
                .map(|mut queue| std::mem::take(&mut *queue))
                .unwrap_or_default();
            client.shutdown(released_handles).await.map_err(|error| {
                crate::build_runtime_error(error.message)
                    .with_builtin("native_ffi")
                    .with_identifier(error.identifier)
                    .build()
            })
        }
        .await;
        self.state.borrow_mut().libraries.clear();
        self.state.borrow_mut().pointers.clear();
        self.pending_isolated_calls.borrow_mut().clear();
        self.prepared_artifact_identities.borrow_mut().clear();
        if let Ok(mut released) = self.released.0.lock() {
            released.clear();
        }
        self.isolated_call_active.set(false);
        result
    }

    pub fn install_prepared_artifact(
        &self,
        library_path: &std::path::Path,
        manifest_path: &std::path::Path,
    ) -> Result<String, RuntimeError> {
        let manifest = NativeInterfaceArtifactManifest::read(manifest_path)
            .map_err(|error| foreign_error_with_source(ForeignErrorKind::LoadFailed, error))?;
        let alias = manifest.interface_name.clone();
        if self.isolation_policy.get().isolate {
            let library_bytes = std::fs::read(library_path).map_err(|error| {
                foreign_error(
                    ForeignErrorKind::LoadFailed,
                    format!(
                        "could not read native library {}: {error}",
                        library_path.display()
                    ),
                )
            })?;
            manifest
                .validate_current_library(&library_bytes)
                .map_err(|error| foreign_error_with_source(ForeignErrorKind::LoadFailed, error))?;
            self.prepared_artifact_identities
                .borrow_mut()
                .insert(manifest.identity.to_string());
            self.pending_isolated_calls.borrow_mut().push(ForeignCall {
                adapter: NATIVE_FFI_ADAPTER_ID.into(),
                symbol: "load_prepared".into(),
                arguments: vec![
                    Value::String(library_path.display().to_string()),
                    Value::String(manifest_path.display().to_string()),
                    Value::String(alias.clone()),
                ],
                requested_outputs: 1,
            });
        } else {
            self.load_prepared_manifest(&alias, library_path, manifest)?;
        }
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
            NativeFfiOperation::SetMember => self.set_member(&call.arguments),
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
        let _ = self.handles.unregister_host(&self.host_identity);
    }
}

impl ForeignAdapter for NativeFfiAdapter {
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        for call in self.pending_isolated_calls.try_borrow().ok()?.iter() {
            for value in &call.arguments {
                visitor(value)?;
            }
        }
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
        let policy = self.isolation_policy.get();
        if !policy.isolate {
            let result = self.invoke_operation(context, call);
            return Box::pin(async move { result });
        }
        if self.isolated_call_active.replace(true) {
            return Box::pin(async {
                Err(invalid_call(
                    "recursive entry into the active isolated native-library host is not supported",
                ))
            });
        }
        let client_slot = Rc::clone(&self.isolated_client);
        let active = Rc::clone(&self.isolated_call_active);
        let pending = Rc::clone(&self.pending_isolated_calls);
        let handles = self.handles.clone();
        let host_identity = self.host_identity.clone();
        let released = Arc::clone(&self.released);
        Box::pin(async move {
            let result = async {
                let mut client = client_slot.borrow_mut().take();
                if client.is_none() {
                    client = Some(IsolatedNativeFfiClient::spawn().await.map_err(|error| {
                        crate::build_runtime_error(error.message)
                            .with_builtin("native_ffi")
                            .with_identifier(error.identifier)
                            .build()
                    })?);
                }
                let mut client = client.expect("isolated client initialized");
                let released_handles = released
                    .0
                    .lock()
                    .map(|mut queue| std::mem::take(&mut *queue))
                    .unwrap_or_default();
                let queued = std::mem::take(&mut *pending.borrow_mut());
                for queued_call in queued {
                    client
                        .invoke(
                            context.clone(),
                            queued_call,
                            &handles,
                            &host_identity,
                            Vec::new(),
                            policy.invocation_timeout,
                        )
                        .await?;
                }
                let requested_outputs = call.requested_outputs;
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
                    .and_then(|outputs| isolated_outputs(outputs, requested_outputs));
                let terminal = outcome.as_ref().err().is_some_and(|error| {
                    matches!(
                        error.identifier(),
                        Some(
                            "RunMat:NativeFFI:Timeout"
                                | "RunMat:NativeFFI:Cancelled"
                                | "RunMat:NativeFFI:HostCrashed"
                        )
                    )
                });
                if !terminal {
                    *client_slot.borrow_mut() = Some(client);
                }
                outcome
            }
            .await;
            active.set(false);
            result
        })
    }

    fn is_isolated(&self) -> bool {
        self.isolation_policy.get().isolate
    }
}

fn isolated_outputs(
    mut outputs: Vec<Value>,
    requested_outputs: usize,
) -> Result<runmat_value::ValueSequence, RuntimeError> {
    match (requested_outputs, outputs.len()) {
        (1, 1) => runmat_value::ValueSequence::single(
            outputs.pop().expect("one isolated native FFI output"),
        )
        .map_err(|error| foreign_error_with_source(ForeignErrorKind::InvalidCall, error)),
        _ => runmat_value::ValueSequence::comma_separated(outputs)
            .map_err(|error| foreign_error_with_source(ForeignErrorKind::InvalidCall, error)),
    }
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
                .into_values(),
            vec![Value::Num(1.0)]
        );
        assert_eq!(
            requested_output_sequence(source.clone(), 3)
                .expect("three outputs")
                .into_values(),
            source
        );
        assert!(requested_output_sequence(vec![Value::Num(1.0)], 2).is_err());
    }
}
