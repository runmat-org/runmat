use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use runmat_native_ffi::{
    artifact_identity, invoke_symbol_with_bindings, prepare_header,
    prepare_header_with_declarations_report, CallbackBinding, HeaderPreparation, InvocationValue,
    NativeInterfaceArtifactManifest, NativeLibraryMetadata, NativePointerResource, NativeScalar,
    NativeType, ParameterDirection, PointerBinding, PointerOwnership, SymbolPrototype,
    NATIVE_FFI_METADATA_SCHEMA_VERSION,
};
pub use runmat_native_ffi::{NATIVE_FFI_ADAPTER_ID, NATIVE_FFI_ADAPTER_VERSION};
use runmat_types::{
    CapabilityRequirement, ForeignAffinity, ForeignCapability, ForeignLifetime, ForeignOwnership,
    ForeignTypeIdentity,
};
use runmat_value::{CellArray, ForeignResourceKey, StructValue, Value};

use super::super::{
    foreign_error, ForeignAdapter, ForeignAdapterDescriptor, ForeignAdapterFuture,
    ForeignErrorKind, ForeignExecutionPolicy, ForeignHandleRegistry, ForeignHostRegistration,
    ForeignHostRelease, ForeignResourceMetadata,
};
use super::request::{
    default_alias, invalid_call, legacy_pointer_type, optional_string_argument, string_argument,
};
use super::session::{LibraryEntry, NativeFfiSessionState, OpaquePointerView, PointerEntry};
use super::{IsolatedNativeFfiClient, NativeFfiIsolationPolicy};
use crate::context::{ForeignCall, RuntimeContext};
use crate::RuntimeError;

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
            .map_err(|error| foreign_error(ForeignErrorKind::LoadFailed, error.to_string()))?;
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
                .map_err(|error| foreign_error(ForeignErrorKind::LoadFailed, error.to_string()))?;
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

    fn load(&self, arguments: &[Value]) -> Result<Value, RuntimeError> {
        self.load_legacy(arguments)
            .map(|report| Value::String(report.alias))
    }

    fn load_report(&self, arguments: &[Value]) -> Result<Value, RuntimeError> {
        let report = self.load_legacy(arguments)?;
        let count = report.notfound.len();
        let notfound = CellArray::new(
            report.notfound.into_iter().map(Value::String).collect(),
            1,
            count,
        )
        .map_err(invalid_call)?;
        Ok(Value::OutputList(vec![
            Value::Cell(notfound),
            Value::String(report.warnings),
        ]))
    }

    fn load_legacy(&self, arguments: &[Value]) -> Result<LegacyLoadReport, RuntimeError> {
        if arguments.len() < 2 {
            return Err(invalid_call(
                "native FFI load expects library path, header path, and an optional alias",
            ));
        }
        let library_path = string_argument(arguments, 0, "load")?;
        let header_path = string_argument(arguments, 1, "load")?;
        let alias = optional_string_argument(arguments, 2, "load")?
            .map(Ok)
            .unwrap_or_else(|| default_alias(&library_path))?;
        let mut include_directories = Vec::new();
        let mut additional_headers = Vec::new();
        let options = &arguments[3..];
        if !options.is_empty()
            && options.len().is_multiple_of(2)
            && options.iter().step_by(2).all(|value| {
                String::try_from(value).is_ok_and(|name| {
                    matches!(
                        name.to_ascii_lowercase().as_str(),
                        "includepath" | "addheader"
                    )
                })
            })
        {
            for pair in options.chunks_exact(2) {
                let name = String::try_from(&pair[0]).expect("validated option name");
                let value = PathBuf::from(String::try_from(&pair[1]).map_err(|_| {
                    invalid_call(format!("native FFI load option `{name}` must be text"))
                })?);
                match name.to_ascii_lowercase().as_str() {
                    "includepath" => include_directories.push(value),
                    "addheader" => additional_headers.push(value),
                    _ => unreachable!("validated native FFI load option"),
                }
            }
        } else {
            include_directories = options
                .iter()
                .enumerate()
                .map(|(index, _)| string_argument(arguments, index + 3, "load").map(PathBuf::from))
                .collect::<Result<Vec<_>, _>>()?;
        }
        self.validate_available_alias(&alias)?;
        let preparation = HeaderPreparation {
            header: PathBuf::from(header_path),
            library_name: alias.clone(),
            library_path: library_path.clone(),
            target_triple: target_lexicon::HOST.to_string(),
            clang: self.compiler_frontend.clone(),
            include_directories,
            definitions: Vec::new(),
        };
        let preparation =
            prepare_header_with_declarations_report(&preparation, &additional_headers)
                .map_err(|error| foreign_error(ForeignErrorKind::LoadFailed, error.to_string()))?;
        let identity = artifact_identity(&preparation.metadata)
            .map_err(|error| foreign_error(ForeignErrorKind::LoadFailed, error.to_string()))?;
        let notfound =
            self.insert_library(&alias, &library_path, preparation.metadata, identity, true)?;
        Ok(LegacyLoadReport {
            alias,
            notfound,
            warnings: preparation.warnings,
        })
    }

    fn load_prepared(&self, arguments: &[Value]) -> Result<Value, RuntimeError> {
        if !(2..=3).contains(&arguments.len()) {
            return Err(invalid_call(
                "prepared native FFI load expects a library path, manifest path, and optional alias",
            ));
        }
        let library_path = PathBuf::from(string_argument(arguments, 0, "load_prepared")?);
        let manifest_path = PathBuf::from(string_argument(arguments, 1, "load_prepared")?);
        let manifest = NativeInterfaceArtifactManifest::read(&manifest_path)
            .map_err(|error| foreign_error(ForeignErrorKind::LoadFailed, error.to_string()))?;
        let alias = optional_string_argument(arguments, 2, "load_prepared")?
            .unwrap_or_else(|| manifest.interface_name.clone());
        self.load_prepared_manifest(&alias, &library_path, manifest)
    }

    fn load_prepared_manifest(
        &self,
        alias: &str,
        library_path: &std::path::Path,
        manifest: NativeInterfaceArtifactManifest,
    ) -> Result<Value, RuntimeError> {
        self.validate_available_alias(alias)?;
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
            .map_err(|error| foreign_error(ForeignErrorKind::LoadFailed, error.to_string()))?;
        let metadata = manifest
            .materialized_metadata(library_path)
            .map_err(|error| foreign_error(ForeignErrorKind::LoadFailed, error.to_string()))?;
        let identity = manifest.identity.to_string();
        self.insert_library(alias, library_path, metadata, identity, false)?;
        Ok(Value::String(alias.into()))
    }

    fn validate_available_alias(&self, alias: &str) -> Result<(), RuntimeError> {
        if alias.trim().is_empty() {
            return Err(invalid_call("native library alias must not be empty"));
        }
        if self.state.borrow().libraries.contains_key(alias) {
            return Err(invalid_call(format!(
                "native library alias `{alias}` is already loaded"
            )));
        }
        Ok(())
    }

    fn insert_library(
        &self,
        alias: &str,
        library_path: impl AsRef<std::path::Path>,
        metadata: NativeLibraryMetadata,
        artifact_identity: String,
        allow_missing_symbols: bool,
    ) -> Result<Vec<String>, RuntimeError> {
        let library = runmat_native_ffi::LoadedLibrary::open(library_path.as_ref())
            .map_err(|error| foreign_error(ForeignErrorKind::LoadFailed, error.to_string()))?;
        let mut metadata = metadata;
        let mut missing = Vec::new();
        for declared_library in &mut metadata.libraries {
            let mut available = Vec::with_capacity(declared_library.symbols.len());
            for symbol in std::mem::take(&mut declared_library.symbols) {
                if library.symbol(&symbol.exported_name).is_ok() {
                    available.push(symbol);
                } else {
                    missing.push(symbol.name.clone());
                    if !allow_missing_symbols {
                        continue;
                    }
                }
            }
            declared_library.symbols = available;
        }
        if !allow_missing_symbols && !missing.is_empty() {
            return Err(foreign_error(
                ForeignErrorKind::LoadFailed,
                format!(
                    "prepared native interface is missing declared symbols: {}",
                    missing.join(", ")
                ),
            ));
        }
        self.prepared_artifact_identities
            .borrow_mut()
            .insert(artifact_identity.clone());
        self.state.borrow_mut().libraries.insert(
            alias.into(),
            LibraryEntry {
                library: Rc::new(library),
                metadata: Rc::new(metadata),
                artifact_identity,
            },
        );
        Ok(missing)
    }

    fn unload(&self, arguments: &[Value]) -> Result<Value, RuntimeError> {
        if arguments.len() != 1 {
            return Err(invalid_call("native FFI unload expects one library alias"));
        }
        let alias = string_argument(arguments, 0, "unload")?;
        let removed = self.state.borrow_mut().libraries.remove(&alias).is_some();
        if !removed {
            return Err(invalid_call(format!(
                "native library alias `{alias}` is not loaded"
            )));
        }
        Ok(Value::Bool(true))
    }

    fn is_loaded(&self, arguments: &[Value]) -> Result<Value, RuntimeError> {
        if arguments.len() != 1 {
            return Err(invalid_call(
                "native FFI is_loaded expects one library alias",
            ));
        }
        let alias = string_argument(arguments, 0, "is_loaded")?;
        Ok(Value::Bool(
            self.state.borrow().libraries.contains_key(&alias),
        ))
    }

    fn functions(&self, arguments: &[Value]) -> Result<Value, RuntimeError> {
        if !(1..=2).contains(&arguments.len()) {
            return Err(invalid_call(
                "native FFI functions expects a library alias and optional -full flag",
            ));
        }
        let alias = string_argument(arguments, 0, "functions")?;
        let full = match optional_string_argument(arguments, 1, "functions")? {
            None => false,
            Some(option) if option.eq_ignore_ascii_case("-full") => true,
            Some(option) => {
                return Err(invalid_call(format!(
                    "unsupported libfunctions option `{option}`; expected -full"
                )))
            }
        };
        let entry = self.library(&alias)?;
        let values = entry.metadata.libraries[0]
            .symbols
            .iter()
            .map(|symbol| {
                Value::String(if full {
                    legacy_signature(symbol)
                } else {
                    symbol.name.clone()
                })
            })
            .collect::<Vec<_>>();
        CellArray::new(values.clone(), 1, values.len())
            .map(Value::Cell)
            .map_err(invalid_call)
    }

    fn pointer(&self, arguments: &[Value]) -> Result<Value, RuntimeError> {
        if arguments.len() > 2 {
            return Err(invalid_call(
                "native FFI pointer expects an optional pointer type and initial value",
            ));
        }
        let type_name = if arguments.is_empty() {
            "voidPtr".to_string()
        } else {
            string_argument(arguments, 0, "pointer")?
        };
        let resolved = self.resolve_pointer_type(&type_name)?;
        let pointer = match arguments.get(1) {
            Some(value) => {
                NativePointerResource::new(resolved.pointee.clone(), value, &resolved.metadata)
                    .map_err(|error| {
                        foreign_error(ForeignErrorKind::InvalidCall, error.to_string())
                    })?
            }
            None => NativePointerResource::null(resolved.pointee),
        };
        self.register_caller_owned_pointer(
            pointer,
            resolved.metadata,
            type_name,
            "lib.pointer".into(),
            resolved.library,
        )
    }

    fn resolve_pointer_type(&self, type_name: &str) -> Result<ResolvedPointerType, RuntimeError> {
        if let Ok(pointee) = legacy_pointer_type(type_name) {
            return Ok(ResolvedPointerType {
                pointee,
                metadata: Rc::new(pointer_metadata()),
                library: None,
            });
        }
        let declared = type_name
            .strip_suffix("Ptr")
            .filter(|name| !name.is_empty())
            .unwrap_or(type_name);
        let mut matches = self
            .state
            .borrow()
            .libraries
            .values()
            .filter_map(|entry| {
                declared_pointer_type(declared, &entry.metadata).map(|pointee| {
                    (
                        pointee,
                        Rc::clone(&entry.metadata),
                        Rc::clone(&entry.library),
                    )
                })
            })
            .collect::<Vec<_>>();
        let Some((pointee, metadata, library)) = matches.pop() else {
            return Err(invalid_call(format!(
                "unsupported libpointer type `{type_name}`; use a numeric pointer type or a type declared by a loaded library"
            )));
        };
        if matches
            .iter()
            .any(|(candidate, _, _)| candidate != &pointee)
        {
            return Err(invalid_call(format!(
                "libpointer type `{type_name}` is ambiguous across loaded libraries"
            )));
        }
        Ok(ResolvedPointerType {
            pointee,
            metadata,
            library: Some(library),
        })
    }

    fn register_caller_owned_pointer(
        &self,
        pointer: NativePointerResource,
        metadata: Rc<NativeLibraryMetadata>,
        type_name: String,
        nominal_type: String,
        library: Option<Rc<runmat_native_ffi::LoadedLibrary>>,
    ) -> Result<Value, RuntimeError> {
        let reference = self.handles.register_resource(
            &self.host_identity,
            ForeignResourceMetadata {
                type_identity: pointer_type_identity(nominal_type),
                ownership: ForeignOwnership::Owned,
                affinity: ForeignAffinity::OriginThread,
                lifetime: ForeignLifetime::Session,
            },
        )?;
        self.state.borrow_mut().pointers.insert(
            reference.handle,
            PointerEntry::CallerOwned {
                pointer: Rc::new(pointer),
                metadata,
                type_name,
                library,
            },
        );
        Ok(Value::Foreign(reference))
    }

    fn structure(&self, arguments: &[Value]) -> Result<Value, RuntimeError> {
        if !(1..=2).contains(&arguments.len()) {
            return Err(invalid_call(
                "native FFI structure expects a declared structure type and optional value",
            ));
        }
        let requested = string_argument(arguments, 0, "structure")?;
        let (entry, structure_name) = self.structure_library(&requested)?;
        let initial_value = match arguments.get(1) {
            Some(Value::Struct(value)) => Value::Struct(value.clone()),
            Some(_) => {
                return Err(invalid_call(
                    "native structure initialization requires a scalar structure value",
                ))
            }
            None => Value::Struct(default_structure_value(
                &structure_name,
                entry.metadata.as_ref(),
            )?),
        };
        let pointer = NativePointerResource::new(
            NativeType::Structure {
                name: structure_name.clone(),
            },
            &initial_value,
            &entry.metadata,
        )
        .map_err(|error| foreign_error(ForeignErrorKind::InvalidCall, error.to_string()))?;
        self.register_caller_owned_pointer(
            pointer,
            Rc::clone(&entry.metadata),
            structure_name.clone(),
            format!("lib.{structure_name}"),
            Some(Rc::clone(&entry.library)),
        )
    }

    fn structure_library(&self, requested: &str) -> Result<(LibraryEntry, String), RuntimeError> {
        let (requested_alias, structure_name) = requested
            .split_once('.')
            .map(|(alias, name)| (Some(alias), name))
            .unwrap_or((None, requested));
        if structure_name.trim().is_empty() {
            return Err(invalid_call("native structure type must not be empty"));
        }
        let state = self.state.borrow();
        let mut matches = state
            .libraries
            .iter()
            .filter_map(|(alias, entry)| {
                if requested_alias.is_some_and(|requested| requested != alias.as_str()) {
                    return None;
                }
                entry
                    .metadata
                    .structures
                    .iter()
                    .find(|definition| definition.name == structure_name)
                    .cloned()
                    .map(|definition| (entry.clone(), definition))
            })
            .collect::<Vec<_>>();
        let Some((entry, definition)) = matches.pop() else {
            return Err(invalid_call(format!(
                "native structure type `{requested}` is not declared by a loaded library"
            )));
        };
        if matches
            .iter()
            .any(|(_, candidate)| candidate != &definition)
        {
            return Err(invalid_call(format!(
                "native structure type `{structure_name}` is ambiguous; qualify it with the library alias"
            )));
        }
        Ok((entry, structure_name.to_string()))
    }

    fn pointer_value(&self, arguments: &[Value]) -> Result<Value, RuntimeError> {
        if arguments.len() != 1 {
            return Err(invalid_call(
                "native FFI pointer_value expects one pointer resource",
            ));
        }
        let Value::Foreign(reference) = &arguments[0] else {
            return Err(invalid_call("pointer_value requires a foreign pointer"));
        };
        self.handles.resolve(reference, ForeignCapability::Read)?;
        let entry = self
            .state
            .borrow()
            .pointers
            .get(&reference.handle)
            .cloned()
            .ok_or_else(|| {
                foreign_error(
                    ForeignErrorKind::StaleHandle,
                    "native pointer resource has been released",
                )
            })?;
        match entry {
            PointerEntry::CallerOwned {
                pointer, metadata, ..
            } => pointer
                .value(&metadata)
                .map_err(|error| foreign_error(ForeignErrorKind::InvalidCall, error.to_string())),
            PointerEntry::Opaque { .. } => Err(invalid_call(
                "opaque library pointer values cannot be read without an explicit copy contract",
            )),
        }
    }

    fn get_member(&self, arguments: &[Value]) -> Result<Value, RuntimeError> {
        if arguments.len() != 2 {
            return Err(invalid_call(
                "native FFI get_member expects a foreign resource and member name",
            ));
        }
        let member = string_argument(arguments, 1, "get_member")?;
        let Value::Foreign(reference) = &arguments[0] else {
            return Err(invalid_call(
                "native FFI member access requires a foreign resource",
            ));
        };
        self.handles.resolve(reference, ForeignCapability::Read)?;
        let entry = self
            .state
            .borrow()
            .pointers
            .get(&reference.handle)
            .cloned()
            .ok_or_else(|| {
                foreign_error(
                    ForeignErrorKind::StaleHandle,
                    "native pointer resource has been released",
                )
            })?;
        match (member.to_ascii_lowercase().as_str(), entry) {
            (
                "value",
                PointerEntry::CallerOwned {
                    pointer, metadata, ..
                },
            ) => pointer
                .value(&metadata)
                .map_err(|error| foreign_error(ForeignErrorKind::InvalidCall, error.to_string())),
            ("datatype", PointerEntry::CallerOwned { type_name, .. }) => {
                Ok(Value::String(type_name))
            }
            (
                "value",
                PointerEntry::Opaque {
                    pointer,
                    view,
                    metadata,
                    ..
                },
            ) => {
                let view = view.borrow().clone().ok_or_else(|| {
                    invalid_call(
                        "opaque library pointer values require setdatatype before they can be copied",
                    )
                })?;
                runmat_native_ffi::copy_pointer_value(
                    &pointer,
                    &view.pointee,
                    &view.shape,
                    &metadata,
                )
                .map_err(|error| foreign_error(ForeignErrorKind::InvalidCall, error.to_string()))
            }
            ("datatype", PointerEntry::Opaque { pointer, view, .. }) => Ok(Value::String(
                view.borrow()
                    .as_ref()
                    .map(|view| view.type_name.clone())
                    .unwrap_or_else(|| format!("{:?}Ptr", pointer.pointee)),
            )),
            (
                _,
                PointerEntry::CallerOwned {
                    pointer, metadata, ..
                },
            ) => {
                let value = pointer.value(&metadata).map_err(|error| {
                    foreign_error(ForeignErrorKind::InvalidCall, error.to_string())
                })?;
                let Value::Struct(structure) = value else {
                    return Err(invalid_call(format!(
                        "native pointer has no member `{member}`"
                    )));
                };
                structure.fields.get(&member).cloned().ok_or_else(|| {
                    invalid_call(format!("native structure has no field `{member}`"))
                })
            }
            _ => Err(invalid_call(format!(
                "native pointer has no member `{member}`"
            ))),
        }
    }

    fn set_member(&self, arguments: &[Value]) -> Result<Value, RuntimeError> {
        if arguments.len() != 3 {
            return Err(invalid_call(
                "native FFI set_member expects a foreign resource, member name, and value",
            ));
        }
        let member = string_argument(arguments, 1, "set_member")?;
        let Value::Foreign(reference) = &arguments[0] else {
            return Err(invalid_call(
                "native FFI member assignment requires a foreign resource",
            ));
        };
        self.handles.resolve(reference, ForeignCapability::Write)?;
        let entry = self
            .state
            .borrow()
            .pointers
            .get(&reference.handle)
            .cloned()
            .ok_or_else(|| {
                foreign_error(
                    ForeignErrorKind::StaleHandle,
                    "native pointer resource has been released",
                )
            })?;
        match entry {
            PointerEntry::CallerOwned { pointer, metadata, .. } => {
                let replacement = if member.eq_ignore_ascii_case("Value") {
                    arguments[2].clone()
                } else if member.eq_ignore_ascii_case("DataType") {
                    return Err(invalid_call(
                        "native pointer member `DataType` is read-only",
                    ));
                } else {
                    let current = pointer.value(&metadata).map_err(|error| {
                        foreign_error(ForeignErrorKind::InvalidCall, error.to_string())
                    })?;
                    let Value::Struct(mut structure) = current else {
                        return Err(invalid_call(format!(
                            "native pointer member `{member}` is unavailable"
                        )));
                    };
                    let Some(field) = structure.fields.get_mut(&member) else {
                        return Err(invalid_call(format!(
                            "native structure has no field `{member}`"
                        )));
                    };
                    *field = arguments[2].clone();
                    Value::Struct(structure)
                };
                pointer.set_value(&replacement, &metadata).map_err(|error| {
                    foreign_error(ForeignErrorKind::InvalidCall, error.to_string())
                })?;
                Ok(arguments[0].clone())
            }
            PointerEntry::Opaque { .. } => Err(invalid_call(
                "opaque library pointer values cannot be assigned without an explicit copy contract",
            )),
        }
    }

    fn invoke_member(&self, arguments: &[Value]) -> Result<Value, RuntimeError> {
        if arguments.len() < 2 {
            return Err(invalid_call(
                "native FFI invoke_member expects a foreign resource and method name",
            ));
        }
        let Value::Foreign(reference) = &arguments[0] else {
            return Err(invalid_call(
                "native FFI method invocation requires a foreign resource",
            ));
        };
        let method = string_argument(arguments, 1, "invoke_member")?;
        self.handles.resolve(reference, ForeignCapability::Write)?;
        match method.to_ascii_lowercase().as_str() {
            "setdatatype" => self.set_pointer_datatype(reference, &arguments[2..]),
            "isnull" => self.pointer_is_null(reference, &arguments[2..]),
            "reshape" => self.reshape_pointer(reference, &arguments[2..]),
            _ => Err(invalid_call(format!(
                "native pointer has no method `{method}`"
            ))),
        }
    }

    fn set_pointer_datatype(
        &self,
        reference: &runmat_value::ForeignRef,
        arguments: &[Value],
    ) -> Result<Value, RuntimeError> {
        if arguments.len() < 2 {
            return Err(invalid_call(
                "setdatatype expects a pointer type and one or more dimensions",
            ));
        }
        let type_name = string_argument(arguments, 0, "setdatatype")?;
        let shape = arguments[1..]
            .iter()
            .map(pointer_dimension)
            .collect::<Result<Vec<_>, _>>()?;
        shape.iter().try_fold(1usize, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or_else(|| invalid_call("setdatatype dimensions overflow usize"))
        })?;

        let mut state = self.state.borrow_mut();
        let entry = state.pointers.get_mut(&reference.handle).ok_or_else(|| {
            foreign_error(
                ForeignErrorKind::StaleHandle,
                "native pointer resource has been released",
            )
        })?;
        let PointerEntry::Opaque { view, metadata, .. } = entry else {
            return Err(invalid_call(
                "setdatatype applies to opaque pointers returned by a native library",
            ));
        };
        let pointee = pointer_view_type(&type_name, metadata)?;
        *view.borrow_mut() = Some(OpaquePointerView {
            type_name,
            pointee,
            shape,
        });
        Ok(Value::Foreign(reference.clone()))
    }

    fn pointer_is_null(
        &self,
        reference: &runmat_value::ForeignRef,
        arguments: &[Value],
    ) -> Result<Value, RuntimeError> {
        if !arguments.is_empty() {
            return Err(invalid_call("isNull does not accept arguments"));
        }
        let state = self.state.borrow();
        let entry = state.pointers.get(&reference.handle).ok_or_else(|| {
            foreign_error(
                ForeignErrorKind::StaleHandle,
                "native pointer resource has been released",
            )
        })?;
        Ok(Value::Bool(match entry {
            PointerEntry::CallerOwned { pointer, .. } => pointer.is_null(),
            PointerEntry::Opaque { pointer, .. } => pointer.is_null(),
        }))
    }

    fn reshape_pointer(
        &self,
        reference: &runmat_value::ForeignRef,
        arguments: &[Value],
    ) -> Result<Value, RuntimeError> {
        if arguments.is_empty() {
            return Err(invalid_call("reshape expects one or more dimensions"));
        }
        let shape = arguments
            .iter()
            .map(pointer_dimension)
            .collect::<Result<Vec<_>, _>>()?;
        let requested = shape.iter().try_fold(1usize, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or_else(|| invalid_call("reshape dimensions overflow usize"))
        })?;
        let mut state = self.state.borrow_mut();
        let entry = state.pointers.get_mut(&reference.handle).ok_or_else(|| {
            foreign_error(
                ForeignErrorKind::StaleHandle,
                "native pointer resource has been released",
            )
        })?;
        let PointerEntry::Opaque { view, .. } = entry else {
            return Err(invalid_call(
                "reshape on caller-owned pointers is not supported because their allocation shape is fixed",
            ));
        };
        let mut view = view.borrow_mut();
        let current = view.as_mut().ok_or_else(|| {
            invalid_call("reshape requires setdatatype to establish the pointer extent first")
        })?;
        let existing = current.shape.iter().product::<usize>();
        if requested != existing {
            return Err(invalid_call(format!(
                "reshape cannot change a pointer view from {existing} to {requested} elements"
            )));
        }
        current.shape = shape;
        Ok(Value::Foreign(reference.clone()))
    }

    fn call(
        &self,
        context: RuntimeContext,
        arguments: &[Value],
        requested_outputs: usize,
    ) -> Result<Value, RuntimeError> {
        if arguments.len() < 2 {
            return Err(invalid_call(
                "native FFI call expects a library alias, symbol, and its arguments",
            ));
        }
        let alias = string_argument(arguments, 0, "call")?;
        let symbol_name = string_argument(arguments, 1, "call")?;
        let entry = self.library(&alias)?;
        let prototype = entry.metadata.libraries[0]
            .symbols
            .iter()
            .find(|prototype| {
                prototype.name == symbol_name || prototype.exported_name == symbol_name
            })
            .cloned()
            .ok_or_else(|| invalid_call(format!("unknown native symbol `{symbol_name}`")))?;
        let call_arguments = arguments[2..].to_vec();
        if call_arguments.len() != prototype.parameters.len() {
            return Err(invalid_call(format!(
                "native symbol `{symbol_name}` expects {} arguments, received {}",
                prototype.parameters.len(),
                call_arguments.len()
            )));
        }

        let mut pointer_entries = Vec::new();
        let mut callback_entries = Vec::new();
        for (index, (parameter, value)) in
            prototype.parameters.iter().zip(&call_arguments).enumerate()
        {
            match (&parameter.ty, value) {
                (NativeType::Pointer { .. }, Value::Foreign(reference)) => {
                    self.handles.resolve(reference, ForeignCapability::Invoke)?;
                    let pointer = self
                        .state
                        .borrow()
                        .pointers
                        .get(&reference.handle)
                        .cloned()
                        .ok_or_else(|| {
                            foreign_error(
                                ForeignErrorKind::StaleHandle,
                                "native pointer resource has been released",
                            )
                        })?;
                    pointer_entries.push((index, value.clone(), pointer));
                }
                (NativeType::Callback { .. }, value) if is_callable(value) => {
                    callback_entries.push((index, value.clone()));
                }
                _ => {}
            }
        }

        let callback_dispatches = callback_entries
            .iter()
            .map(|(_, callable)| {
                let context = context.clone();
                let callable = callable.clone();
                let callback_router = self.callback_router.clone();
                Box::new(move |callback_arguments: &[Value]| {
                    if let (Some(router), Some(callback_id)) =
                        (callback_router.as_ref(), isolated_callback_id(&callable))
                    {
                        return router.dispatch(callback_id, callback_arguments);
                    }
                    pollster::block_on(context.scope(crate::call_feval_async_with_outputs(
                        callable.clone(),
                        callback_arguments,
                        1,
                    )))
                    .map_err(|error| error.to_string())
                }) as Box<dyn runmat_native_ffi::CallbackDispatch>
            })
            .collect::<Vec<_>>();
        let callback_bindings = callback_entries
            .iter()
            .zip(&callback_dispatches)
            .map(|((index, _), dispatch)| CallbackBinding {
                argument_index: *index,
                dispatch: dispatch.as_ref(),
            })
            .collect::<Vec<_>>();
        let pointer_bindings = pointer_entries
            .iter()
            .map(|(index, _, pointer)| match pointer {
                PointerEntry::CallerOwned { pointer, .. } => {
                    PointerBinding::resource(*index, pointer)
                }
                PointerEntry::Opaque { pointer, .. } => PointerBinding::opaque(*index, pointer),
            })
            .collect::<Vec<_>>();

        let result = invoke_symbol_with_bindings(
            &entry.library,
            &prototype,
            &call_arguments,
            &entry.metadata,
            &callback_bindings,
            &pointer_bindings,
        )
        .map_err(|error| foreign_error(ForeignErrorKind::InvocationFailed, error.to_string()))?;
        let mut outputs = Vec::new();
        if let Some(return_value) = result.return_value {
            outputs.push(match return_value {
                InvocationValue::Value(value) => value,
                InvocationValue::Pointer(pointer) => {
                    self.register_opaque_pointer(pointer, &entry)?
                }
            });
        }
        let copied_outputs = result
            .output_parameters
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        for (index, parameter) in prototype.parameters.iter().enumerate() {
            if matches!(
                parameter.direction,
                runmat_native_ffi::ParameterDirection::Output
                    | runmat_native_ffi::ParameterDirection::InputOutput
            ) {
                if let Some(value) = copied_outputs.get(&index) {
                    outputs.push(value.clone());
                } else if let Some((_, reference, _)) =
                    pointer_entries.iter().find(|(bound, _, _)| *bound == index)
                {
                    outputs.push(reference.clone());
                }
            }
        }
        requested_output_value(outputs, requested_outputs)
    }

    fn register_opaque_pointer(
        &self,
        pointer: runmat_native_ffi::NativePointer,
        library: &LibraryEntry,
    ) -> Result<Value, RuntimeError> {
        let ownership = foreign_ownership(pointer.ownership);
        let reference = self.handles.register_resource(
            &self.host_identity,
            ForeignResourceMetadata {
                type_identity: pointer_type_identity("lib.pointer".into()),
                ownership,
                affinity: ForeignAffinity::OriginThread,
                lifetime: ForeignLifetime::Session,
            },
        )?;
        self.state.borrow_mut().pointers.insert(
            reference.handle,
            PointerEntry::Opaque {
                pointer: Rc::new(pointer),
                view: Rc::new(RefCell::new(None)),
                library: Rc::clone(&library.library),
                metadata: Rc::clone(&library.metadata),
            },
        );
        Ok(Value::Foreign(reference))
    }

    fn library(&self, alias: &str) -> Result<LibraryEntry, RuntimeError> {
        self.state
            .borrow()
            .libraries
            .get(alias)
            .cloned()
            .ok_or_else(|| invalid_call(format!("native library alias `{alias}` is not loaded")))
    }

    fn build_interface(&self, arguments: &[Value]) -> Result<Value, RuntimeError> {
        if arguments.is_empty() || arguments.len().is_multiple_of(2) {
            return Err(invalid_call(
                "clibgen.buildInterface expects a header followed by name-value pairs",
            ));
        }
        let header = string_argument(arguments, 0, "build_interface")?;
        let mut library = None;
        let mut alias = None;
        let mut includes = Vec::new();
        for pair in arguments[1..].chunks_exact(2) {
            let name = String::try_from(&pair[0])
                .map_err(|_| invalid_call("buildInterface option names must be text"))?;
            match name.to_ascii_lowercase().as_str() {
                "libraries" => {
                    library = Some(String::try_from(&pair[1]).map_err(|_| {
                        invalid_call("buildInterface Libraries must name one shared library")
                    })?);
                }
                "interfacename" => {
                    alias = Some(String::try_from(&pair[1]).map_err(|_| {
                        invalid_call("buildInterface InterfaceName must be text")
                    })?);
                }
                "includepath" | "includedirectories" => {
                    includes.push(PathBuf::from(String::try_from(&pair[1]).map_err(|_| {
                        invalid_call("buildInterface include path must be text")
                    })?));
                }
                unsupported => {
                    return Err(invalid_call(format!(
                        "buildInterface option `{unsupported}` is not supported by the C ABI interface builder"
                    )))
                }
            }
        }
        let library = library.ok_or_else(|| {
            invalid_call("buildInterface requires the Libraries option for a native interface")
        })?;
        let alias = alias.map(Ok).unwrap_or_else(|| default_alias(&header))?;
        self.validate_available_alias(&alias)?;
        let library_path = PathBuf::from(&library);
        let metadata = prepare_header(&HeaderPreparation {
            header: PathBuf::from(header),
            library_name: alias.clone(),
            library_path: library.clone(),
            target_triple: target_lexicon::HOST.to_string(),
            clang: self.compiler_frontend.clone(),
            include_directories: includes,
            definitions: Vec::new(),
        })
        .map_err(|error| foreign_error(ForeignErrorKind::LoadFailed, error.to_string()))?;
        let manifest = NativeInterfaceArtifactManifest::from_library_path(
            alias.clone(),
            metadata,
            &library_path,
        )
        .map_err(|error| foreign_error(ForeignErrorKind::LoadFailed, error.to_string()))?;
        let manifest_path = NativeInterfaceArtifactManifest::path_for_library(&library_path);
        manifest
            .publish(&manifest_path)
            .map_err(|error| foreign_error(ForeignErrorKind::LoadFailed, error.to_string()))?;
        self.load_prepared_manifest(&alias, &library_path, manifest)
    }

    fn invoke_operation(
        &self,
        context: RuntimeContext,
        call: ForeignCall,
    ) -> Result<Value, RuntimeError> {
        self.reap_released();
        match call.symbol.as_str() {
            "load" => self.load(&call.arguments),
            "load_report" => self.load_report(&call.arguments),
            "load_prepared" => self.load_prepared(&call.arguments),
            "unload" => self.unload(&call.arguments),
            "is_loaded" => self.is_loaded(&call.arguments),
            "functions" => self.functions(&call.arguments),
            "pointer" => self.pointer(&call.arguments),
            "structure" => self.structure(&call.arguments),
            "pointer_value" => self.pointer_value(&call.arguments),
            "get_member" => self.get_member(&call.arguments),
            "set_member" => self.set_member(&call.arguments),
            "invoke_member" => self.invoke_member(&call.arguments),
            "call" => self.call(context, &call.arguments, call.requested_outputs),
            "build_interface" => self.build_interface(&call.arguments),
            operation => Err(invalid_call(format!(
                "unknown native FFI operation `{operation}`"
            ))),
        }
    }
}

impl Drop for NativeFfiAdapter {
    fn drop(&mut self) {
        let _ = self.handles.unregister_host(&self.host_identity);
    }
}

impl ForeignAdapter for NativeFfiAdapter {
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
                let outcome = client
                    .invoke(
                        context,
                        call,
                        &handles,
                        &host_identity,
                        released_handles,
                        policy.invocation_timeout,
                    )
                    .await;
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

fn pointer_metadata() -> NativeLibraryMetadata {
    NativeLibraryMetadata {
        schema_version: NATIVE_FFI_METADATA_SCHEMA_VERSION,
        target_triple: target_lexicon::HOST.to_string(),
        source_digest: "session-owned-pointer".into(),
        libraries: Vec::new(),
        structures: Vec::new(),
        enumerations: Vec::new(),
        aliases: Vec::new(),
    }
}

fn declared_pointer_type(name: &str, metadata: &NativeLibraryMetadata) -> Option<NativeType> {
    metadata
        .aliases
        .iter()
        .find(|alias| alias.name == name)
        .map(|alias| alias.target.clone())
        .or_else(|| {
            metadata
                .structures
                .iter()
                .any(|definition| definition.name == name)
                .then(|| NativeType::Structure { name: name.into() })
        })
        .or_else(|| {
            metadata
                .enumerations
                .iter()
                .find(|definition| definition.name == name)
                .map(|definition| NativeType::Enumeration {
                    name: name.into(),
                    storage: definition.storage,
                })
        })
}

fn pointer_view_type(
    type_name: &str,
    metadata: &NativeLibraryMetadata,
) -> Result<NativeType, RuntimeError> {
    if let Ok(pointee) = legacy_pointer_type(type_name) {
        return Ok(pointee);
    }
    let declared = type_name
        .strip_suffix("Ptr")
        .filter(|name| !name.is_empty())
        .unwrap_or(type_name);
    declared_pointer_type(declared, metadata).ok_or_else(|| {
        invalid_call(format!(
            "setdatatype type `{type_name}` is not a numeric type or a declaration from the pointer's library"
        ))
    })
}

fn pointer_dimension(value: &Value) -> Result<usize, RuntimeError> {
    let dimension = match value {
        Value::Int(value) => value.try_to_usize(),
        Value::Num(value) if value.is_finite() && *value >= 1.0 && value.fract() == 0.0 => {
            usize::try_from(*value as u128).ok()
        }
        Value::Tensor(tensor) if tensor.len() == 1 => {
            tensor.numeric_value_at(0).and_then(|value| {
                value.into_int_value().map_or_else(
                    || {
                        let value = value.materialize_f64();
                        (value.is_finite() && value >= 1.0 && value.fract() == 0.0)
                            .then(|| usize::try_from(value as u128).ok())
                            .flatten()
                    },
                    |value| value.try_to_usize(),
                )
            })
        }
        _ => None,
    };
    dimension
        .filter(|dimension| *dimension > 0)
        .ok_or_else(|| invalid_call("pointer dimensions must be positive integer scalars"))
}

fn default_structure_value(
    name: &str,
    metadata: &NativeLibraryMetadata,
) -> Result<StructValue, RuntimeError> {
    let definition = metadata
        .structures
        .iter()
        .find(|definition| definition.name == name)
        .ok_or_else(|| invalid_call(format!("missing native structure definition `{name}`")))?;
    let mut value = StructValue::new();
    for field in &definition.fields {
        let field_value = match &field.ty {
            NativeType::Scalar { .. } | NativeType::Enumeration { .. } => Value::Num(0.0),
            NativeType::Structure { name } => {
                Value::Struct(default_structure_value(name, metadata)?)
            }
            unsupported => {
                return Err(invalid_call(format!(
                    "native structure field `{}` has unsupported default type {unsupported:?}",
                    field.name
                )))
            }
        };
        value.fields.insert(field.name.clone(), field_value);
    }
    Ok(value)
}

fn pointer_type_identity(nominal_type: String) -> ForeignTypeIdentity {
    ForeignTypeIdentity {
        family: NATIVE_FFI_ADAPTER_ID.into(),
        name: nominal_type,
        version: 1,
    }
}

fn foreign_ownership(ownership: PointerOwnership) -> ForeignOwnership {
    match ownership {
        PointerOwnership::Borrowed => ForeignOwnership::Borrowed,
        PointerOwnership::CallerOwned | PointerOwnership::LibraryOwned => ForeignOwnership::Owned,
        PointerOwnership::Shared => ForeignOwnership::Shared,
    }
}

fn legacy_signature(symbol: &SymbolPrototype) -> String {
    let outputs = std::iter::once((&symbol.return_type, None))
        .filter(|(ty, _)| !matches!(ty, NativeType::Void))
        .chain(symbol.parameters.iter().filter_map(|parameter| {
            matches!(
                parameter.direction,
                ParameterDirection::Output | ParameterDirection::InputOutput
            )
            .then_some((&parameter.ty, Some(parameter.name.as_str())))
        }))
        .map(|(ty, name)| match name {
            Some(name) => format!("{} {name}", legacy_type_name(ty)),
            None => legacy_type_name(ty),
        })
        .collect::<Vec<_>>();
    let inputs = symbol
        .parameters
        .iter()
        .map(|parameter| legacy_type_name(&parameter.ty))
        .collect::<Vec<_>>()
        .join(", ");
    match outputs.as_slice() {
        [] => format!("{}({inputs})", symbol.name),
        [output] => format!("{output} {}({inputs})", symbol.name),
        outputs => format!("[{}] {}({inputs})", outputs.join(", "), symbol.name),
    }
}

fn legacy_type_name(ty: &NativeType) -> String {
    match ty {
        NativeType::Void => "void".into(),
        NativeType::Scalar { scalar } => match scalar {
            NativeScalar::Bool => "logical",
            NativeScalar::Char | NativeScalar::SignedChar => "int8",
            NativeScalar::UnsignedChar => "uint8",
            NativeScalar::Short => "int16",
            NativeScalar::UnsignedShort => "uint16",
            NativeScalar::Int => "int32",
            NativeScalar::UnsignedInt => "uint32",
            NativeScalar::Long | NativeScalar::LongLong => "int64",
            NativeScalar::UnsignedLong | NativeScalar::UnsignedLongLong => "uint64",
            NativeScalar::I8 => "int8",
            NativeScalar::U8 => "uint8",
            NativeScalar::I16 => "int16",
            NativeScalar::U16 => "uint16",
            NativeScalar::I32 => "int32",
            NativeScalar::U32 => "uint32",
            NativeScalar::I64 | NativeScalar::Isize => "int64",
            NativeScalar::U64 | NativeScalar::Usize => "uint64",
            NativeScalar::F32 => "single",
            NativeScalar::F64 => "double",
        }
        .into(),
        NativeType::Pointer { pointee, .. } => format!("{}Ptr", legacy_type_name(pointee)),
        NativeType::Array { element, .. } => format!("{}Ptr", legacy_type_name(element)),
        NativeType::Structure { name } | NativeType::Enumeration { name, .. } => name.clone(),
        NativeType::Callback { .. } => "FcnPtr".into(),
    }
}

pub(crate) fn is_callable(value: &Value) -> bool {
    matches!(
        value,
        Value::FunctionHandle(_)
            | Value::ExternalFunctionHandle(_)
            | Value::MethodFunctionHandle(_)
            | Value::BoundFunctionHandle { .. }
            | Value::Closure(_)
    )
}

pub(crate) fn isolated_callback_value(callback_id: u64) -> Value {
    Value::FunctionHandle(format!(
        "__runmat_native_ffi_isolated_callback_{callback_id}"
    ))
}

fn isolated_callback_id(value: &Value) -> Option<u64> {
    let Value::FunctionHandle(name) = value else {
        return None;
    };
    name.strip_prefix("__runmat_native_ffi_isolated_callback_")?
        .parse()
        .ok()
}

fn requested_output_value(
    outputs: Vec<Value>,
    requested_outputs: usize,
) -> Result<Value, RuntimeError> {
    if requested_outputs > outputs.len() {
        return Err(invalid_call(format!(
            "native call provides {} outputs, but {requested_outputs} were requested",
            outputs.len()
        )));
    }
    match requested_outputs {
        0 => Ok(Value::OutputList(Vec::new())),
        1 => Ok(outputs.into_iter().next().expect("validated output count")),
        count => Ok(Value::OutputList(outputs.into_iter().take(count).collect())),
    }
}
