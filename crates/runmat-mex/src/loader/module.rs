use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

use libloading::Library;
use parking_lot::ReentrantMutex;
use runmat_value::Value;
use thiserror::Error;

use crate::{
    value_from_mx_in_context, value_to_mx_for_interface_in_context,
    ConcurrentMexBoundaryHostServices, MexArtifactManifest, MexCallState, MexHostApiV1,
    MexHostServices, MexSourceLanguage, MxApiMode, MxArray, MxBoundaryInterface, MxValueContext,
    UnavailableMexHostServices,
};

type BindHost = unsafe extern "C" fn(*const MexHostApiV1) -> i32;
type HostAbiVersion = unsafe extern "C" fn() -> u32;
type InvokeMex = unsafe extern "C" fn(i32, *mut *mut MxArray, i32, *const *const MxArray) -> i32;
type IsLocked = unsafe extern "C" fn() -> i32;
type ApiMode = unsafe extern "C" fn() -> i32;
type Unload = unsafe extern "C" fn();
type CppUnload = unsafe extern "C" fn();

/// Legacy C MEX code can contain process-global C runtime state. Serialize
/// module loading, invocation, and unloading across sessions while allowing
/// same-thread callback reentry.
static MEX_PROCESS_GATE: LazyLock<ReentrantMutex<()>> = LazyLock::new(|| ReentrantMutex::new(()));

/// A native library image owns process-global state even when the platform
/// loader is asked to open the same path more than once. Keep exactly one
/// in-process owner for each image. Other RunMat sessions can execute the
/// module in an isolated extension host, which gives them an independent
/// library image and host authority.
static IN_PROCESS_MODULES: LazyLock<Mutex<HashSet<PathBuf>>> =
    LazyLock::new(|| Mutex::new(HashSet::new()));

struct InProcessModuleLease {
    path: PathBuf,
}

impl InProcessModuleLease {
    fn acquire(path: &Path) -> Result<Self, MexLoadError> {
        let path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        let mut loaded = IN_PROCESS_MODULES
            .lock()
            .map_err(|_| MexLoadError::Poisoned)?;
        if !loaded.insert(path.clone()) {
            return Err(MexLoadError::InProcessOwnership {
                path: path.display().to_string(),
            });
        }
        Ok(Self { path })
    }
}

impl Drop for InProcessModuleLease {
    fn drop(&mut self) {
        if let Ok(mut loaded) = IN_PROCESS_MODULES.lock() {
            loaded.remove(&self.path);
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MexInvocation {
    pub outputs: Vec<Value>,
    pub warnings: Vec<crate::MexDiagnostic>,
    pub console: String,
}

/// Result of an invocation at the native array boundary.
///
/// This workspace-facing form preserves shared array storage and opaque handle
/// tokens. Runtime `Value` conversion remains the responsibility of the
/// originating runtime task.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct MexNativeInvocation {
    pub outputs: Vec<MxArray>,
    pub warnings: Vec<crate::MexDiagnostic>,
    pub console: String,
}

#[derive(Debug, Error)]
pub enum MexLoadError {
    #[error("failed to read MEX artifact manifest {path}: {source}")]
    ArtifactManifestRead {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("MEX artifact manifest {path} is invalid or does not match its module: {message}")]
    ArtifactManifest { path: String, message: String },
    #[error("failed to read MEX module {path} for artifact admission: {source}")]
    ModuleRead {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to load MEX module {path}: {source}")]
    Load {
        path: String,
        #[source]
        source: libloading::Error,
    },
    #[error("MEX module is missing required symbol {symbol}: {source}")]
    Symbol {
        symbol: &'static str,
        #[source]
        source: libloading::Error,
    },
    #[error("MEX binary compatibility check failed for {path}: {message}")]
    BinaryCompatibility { path: String, message: String },
    #[error("MEX module {path} already has an in-process owner; use an isolated extension host")]
    InProcessOwnership { path: String },
    #[error("MEX input conversion failed: {0}")]
    Input(String),
    #[error("MEX function failed{identifier}: {message}")]
    Invocation { identifier: String, message: String },
    #[error("MEX function did not assign output {0}")]
    MissingOutput(usize),
    #[error("MEX output conversion failed: {0}")]
    Output(String),
    #[error("MEX module invocation lock is poisoned")]
    Poisoned,
    #[error("recursive invocation of the same C MEX module is not supported")]
    ReentrantModule,
}

impl MexLoadError {
    pub fn requires_isolated_host(&self) -> bool {
        matches!(self, Self::InProcessOwnership { .. })
    }

    /// Return the native dependency named by the platform loader, when the
    /// loader provided one separately from the module being opened.
    pub fn missing_dependency(&self) -> Option<String> {
        let Self::Load { path, source } = self else {
            return None;
        };
        missing_dependency_from_loader_message(path, &source.to_string())
    }
}

fn missing_dependency_from_loader_message(module: &str, message: &str) -> Option<String> {
    if let Some((_, remainder)) = message.split_once("Library not loaded:") {
        return remainder
            .lines()
            .next()
            .map(str::trim)
            .filter(|dependency| !dependency.is_empty() && *dependency != module)
            .map(str::to_owned);
    }
    let marker = ": cannot open shared object file";
    if let Some((dependency, _)) = message.split_once(marker) {
        let dependency = dependency
            .rsplit_once(": ")
            .map_or(dependency, |(_, dependency)| dependency)
            .trim();
        if !dependency.is_empty() && dependency != module {
            return Some(dependency.to_owned());
        }
    }
    None
}

pub struct MexModule {
    _in_process_lease: InProcessModuleLease,
    _library: Library,
    bind: BindHost,
    invoke: InvokeMex,
    is_locked: IsLocked,
    mode: MxApiMode,
    interface: MxBoundaryInterface,
    unload: Unload,
    cpp_unload: Option<CppUnload>,
    state: Mutex<Option<MexModuleState>>,
}

struct MexModuleState {
    call: Box<MexCallState>,
    values: std::rc::Rc<MxValueContext>,
    api: Box<MexHostApiV1>,
}

impl MexModuleState {
    fn direct(mode: MxApiMode, interface: MxBoundaryInterface) -> Self {
        let call = Box::new(MexCallState::for_interface(mode, interface));
        let api = Box::new(call.host_api());
        Self {
            call,
            values: std::rc::Rc::new(MxValueContext::new()),
            api,
        }
    }

    fn native(
        mode: MxApiMode,
        interface: MxBoundaryInterface,
        services: std::sync::Arc<dyn ConcurrentMexBoundaryHostServices>,
    ) -> Self {
        let call = Box::new(MexCallState::with_boundary_services_for_interface(
            mode, interface, services,
        ));
        let api = Box::new(call.host_api());
        Self {
            call,
            values: std::rc::Rc::new(MxValueContext::new()),
            api,
        }
    }
}

impl MexModule {
    unsafe fn unload_bound(&self, state: Option<&MexModuleState>) {
        if let Some(state) = state {
            // SAFETY: the module state and its pointer-stable vtable outlive
            // C++ gateway destruction and the shim's synchronous exit hook.
            unsafe {
                let _ = (self.bind)(state.api.as_ref());
                if let Some(cpp_unload) = self.cpp_unload {
                    cpp_unload();
                }
            }
        }
        // SAFETY: the loader owns the module and serializes all lifecycle work.
        unsafe { (self.unload)() };
    }

    pub fn load(path: &Path) -> Result<Self, MexLoadError> {
        let lease = InProcessModuleLease::acquire(path)?;
        Self::load_inner(path, true, lease)
    }

    /// Load an unmanifested module built for RunMat's compatibility interface.
    ///
    /// Callers must keep this module inside an isolated extension host.
    pub fn load_compatible_isolated(path: &Path) -> Result<Self, MexLoadError> {
        let expected = crate::mex_suffix().ok_or_else(|| MexLoadError::BinaryCompatibility {
            path: path.display().to_string(),
            message: "the current platform does not define a MEX suffix".into(),
        })?;
        let actual = path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or_default();
        if actual != expected {
            return Err(MexLoadError::BinaryCompatibility {
                path: path.display().to_string(),
                message: format!("expected platform suffix '.{expected}', found '.{actual}'"),
            });
        }
        let lease = InProcessModuleLease::acquire(path)?;
        Self::load_inner(path, false, lease)
    }

    fn load_inner(
        path: &Path,
        require_manifest: bool,
        in_process_lease: InProcessModuleLease,
    ) -> Result<Self, MexLoadError> {
        let _process_guard = MEX_PROCESS_GATE.lock();
        let interface = if require_manifest {
            match admit_artifact(path)?.source_language {
                MexSourceLanguage::C => MxBoundaryInterface::CMatrix,
                MexSourceLanguage::Cxx => MxBoundaryInterface::CxxData,
            }
        } else {
            MxBoundaryInterface::CMatrix
        };
        // SAFETY: the library remains owned by `Self`; all resolved function
        // pointers are copied only after their exact C signatures are checked.
        let library = unsafe { Library::new(path) }.map_err(|source| MexLoadError::Load {
            path: path.display().to_string(),
            source,
        })?;
        // SAFETY: this no-argument query is part of RunMat's private module
        // interface and returns a fixed-width integer.
        let host_abi_version =
            unsafe { library.get::<HostAbiVersion>(b"runmatMexHostAbiVersion\0") }.map_err(
                |source| {
                    if require_manifest {
                        MexLoadError::Symbol {
                            symbol: "runmatMexHostAbiVersion",
                            source,
                        }
                    } else {
                        MexLoadError::BinaryCompatibility {
                            path: path.display().to_string(),
                            message: format!(
                                "module does not expose RunMat's compatibility interface: {source}"
                            ),
                        }
                    }
                },
            )?;
        // SAFETY: the query has no memory arguments or mutable state contract.
        let actual_host_abi = unsafe { host_abi_version() };
        if actual_host_abi != crate::MEX_HOST_ABI_VERSION {
            return Err(MexLoadError::BinaryCompatibility {
                path: path.display().to_string(),
                message: format!(
                    "module requires private host ABI {actual_host_abi}, but this RunMat executable provides {}",
                    crate::MEX_HOST_ABI_VERSION
                ),
            });
        }
        // SAFETY: these symbols are supplied by RunMat's compiled C shim and
        // have signatures fixed by RunMat's private host ABI.
        let bind =
            *unsafe { library.get::<BindHost>(b"runmatMexBindHost\0") }.map_err(|source| {
                MexLoadError::Symbol {
                    symbol: "runmatMexBindHost",
                    source,
                }
            })?;
        // SAFETY: same argument as above for the invocation entry point.
        let invoke =
            *unsafe { library.get::<InvokeMex>(b"runmatMexInvoke\0") }.map_err(|source| {
                MexLoadError::Symbol {
                    symbol: "runmatMexInvoke",
                    source,
                }
            })?;
        // SAFETY: lifecycle symbols are emitted by the same private shim.
        let is_locked =
            *unsafe { library.get::<IsLocked>(b"runmatMexIsLocked\0") }.map_err(|source| {
                MexLoadError::Symbol {
                    symbol: "runmatMexIsLocked",
                    source,
                }
            })?;
        let api_mode =
            *unsafe { library.get::<ApiMode>(b"runmatMexApiMode\0") }.map_err(|source| {
                MexLoadError::Symbol {
                    symbol: "runmatMexApiMode",
                    source,
                }
            })?;
        // SAFETY: the shim returns one of its fixed Matrix API mode tags.
        let mode = match unsafe { api_mode() } {
            0 => MxApiMode::SeparateComplex,
            1 => MxApiMode::InterleavedComplex,
            _ => {
                return Err(MexLoadError::Input(
                    "MEX module reports an unsupported Matrix API mode".into(),
                ));
            }
        };
        // SAFETY: lifecycle symbols are emitted by the same private shim.
        let unload = *unsafe { library.get::<Unload>(b"runmatMexUnload\0") }.map_err(|source| {
            MexLoadError::Symbol {
                symbol: "runmatMexUnload",
                source,
            }
        })?;
        let cpp_unload = unsafe { library.get::<CppUnload>(b"runmatMexCppUnload\0") }
            .ok()
            .map(|symbol| *symbol);
        Ok(Self {
            _in_process_lease: in_process_lease,
            _library: library,
            bind,
            invoke,
            is_locked,
            mode,
            interface,
            unload,
            cpp_unload,
            state: Mutex::new(None),
        })
    }

    pub fn api_mode(&self) -> MxApiMode {
        self.mode
    }

    #[doc(hidden)]
    pub fn boundary_interface(&self) -> MxBoundaryInterface {
        self.interface
    }

    pub fn invoke(
        &self,
        inputs: &[Value],
        output_count: usize,
        mode: MxApiMode,
    ) -> Result<MexInvocation, MexLoadError> {
        self.invoke_with_services(
            inputs,
            output_count,
            mode,
            std::rc::Rc::new(UnavailableMexHostServices),
        )
    }

    pub fn invoke_with_services(
        &self,
        inputs: &[Value],
        output_count: usize,
        mode: MxApiMode,
        services: std::rc::Rc<dyn MexHostServices>,
    ) -> Result<MexInvocation, MexLoadError> {
        if mode != self.mode {
            return Err(MexLoadError::Input(format!(
                "MEX module was built for {:?}, not {:?}",
                self.mode, mode
            )));
        }
        let _process_guard = MEX_PROCESS_GATE.lock();
        let mut state_slot = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Err(MexLoadError::ReentrantModule),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(MexLoadError::Poisoned),
        };
        let state = state_slot.get_or_insert_with(|| MexModuleState::direct(mode, self.interface));
        if state
            .call
            .lock()
            .map_err(|_| MexLoadError::Poisoned)?
            .mx
            .mode()
            != mode
        {
            return Err(MexLoadError::Input(
                "a loaded MEX module cannot switch complex API mode".into(),
            ));
        }
        state
            .call
            .set_services_for_context(services, state.values.clone());
        state.call.begin_call();
        let result = (|| {
            let mut arrays = Vec::with_capacity(inputs.len());
            for input in inputs {
                let value = value_to_mx_for_interface_in_context(
                    input,
                    mode,
                    self.interface,
                    Some(&state.values),
                )
                .map_err(|error| MexLoadError::Input(error.message))?;
                arrays.push(value);
            }
            let invocation =
                self.invoke_arrays(&state.call, state.api.as_ref(), arrays, output_count)?;
            let outputs = invocation
                .outputs
                .iter()
                .map(|value| {
                    value_from_mx_in_context(value, Some(&state.values))
                        .map_err(|error| MexLoadError::Output(error.message))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(MexInvocation {
                outputs,
                warnings: invocation.warnings,
                console: invocation.console,
            })
        })();
        state.call.finish_call();
        result
    }

    /// Invoke a module using the thread-transferable native array boundary.
    ///
    /// The caller must execute this method on the lane that owns the module.
    /// Host callbacks are delegated to `services`; they may proxy requests to
    /// an originating runtime task without moving runtime-owned values or
    /// garbage-collected handles onto this thread.
    #[doc(hidden)]
    pub fn invoke_native(
        &self,
        inputs: Vec<MxArray>,
        output_count: usize,
        mode: MxApiMode,
        services: std::sync::Arc<dyn ConcurrentMexBoundaryHostServices>,
    ) -> Result<MexNativeInvocation, MexLoadError> {
        if mode != self.mode {
            return Err(MexLoadError::Input(format!(
                "MEX module was built for {:?}, not {:?}",
                self.mode, mode
            )));
        }
        let _process_guard = MEX_PROCESS_GATE.lock();
        let mut state_slot = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Err(MexLoadError::ReentrantModule),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(MexLoadError::Poisoned),
        };
        let state = state_slot
            .get_or_insert_with(|| MexModuleState::native(mode, self.interface, services.clone()));
        if state
            .call
            .lock()
            .map_err(|_| MexLoadError::Poisoned)?
            .mx
            .mode()
            != mode
        {
            return Err(MexLoadError::Input(
                "a loaded MEX module cannot switch complex API mode".into(),
            ));
        }
        state.call.set_boundary_services(services);
        state.call.begin_call();
        let result = self.invoke_arrays(&state.call, state.api.as_ref(), inputs, output_count);
        state.call.finish_call();
        result
    }

    fn invoke_arrays(
        &self,
        state: &MexCallState,
        api: &MexHostApiV1,
        inputs: Vec<MxArray>,
        output_count: usize,
    ) -> Result<MexNativeInvocation, MexLoadError> {
        let input_pointers = {
            let mut state = state.lock().map_err(|_| MexLoadError::Poisoned)?;
            inputs
                .into_iter()
                .map(|value| state.mx.allocate(value).cast_const())
                .collect::<Vec<_>>()
        };
        let mut output_pointers = vec![std::ptr::null_mut(); output_count];
        // SAFETY: the host API is pointer-stable for the loaded module's
        // lifetime; input/output arrays have the exact lengths passed here.
        let status = unsafe {
            if (self.bind)(api) != 0 {
                1
            } else {
                (self.invoke)(
                    i32::try_from(output_pointers.len()).unwrap_or(i32::MAX),
                    output_pointers.as_mut_ptr(),
                    i32::try_from(input_pointers.len()).unwrap_or(i32::MAX),
                    input_pointers.as_ptr(),
                )
            }
        };
        let mut state = state.lock().map_err(|_| MexLoadError::Poisoned)?;
        if status != 0 || state.error.is_some() {
            let error = state.error.clone().unwrap_or(crate::MexDiagnostic {
                identifier: None,
                message: format!("gateway returned status {status}"),
            });
            return Err(invocation_error(error));
        }
        let outputs = output_pointers
            .into_iter()
            .enumerate()
            .map(|(index, pointer)| {
                state
                    .mx
                    .arena()
                    .get(pointer)
                    .cloned()
                    .map_err(|_| MexLoadError::MissingOutput(index))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(MexNativeInvocation {
            outputs,
            warnings: std::mem::take(&mut state.warnings),
            console: std::mem::take(&mut state.console),
        })
    }

    pub fn is_locked(&self) -> bool {
        // SAFETY: the no-argument shim query has no memory preconditions.
        unsafe { (self.is_locked)() != 0 }
    }

    pub fn clear(&self) -> Result<bool, MexLoadError> {
        self.clear_with_services(std::rc::Rc::new(UnavailableMexHostServices))
    }

    pub fn clear_with_services(
        &self,
        services: std::rc::Rc<dyn MexHostServices>,
    ) -> Result<bool, MexLoadError> {
        let _process_guard = MEX_PROCESS_GATE.lock();
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(MexLoadError::Poisoned),
        };
        if self.is_locked() {
            return Ok(false);
        }
        if state
            .as_ref()
            .is_some_and(|state| state.call.has_pending_async_requests().unwrap_or(true))
        {
            return Ok(false);
        }
        let lifecycle_error = if let Some(call_state) = state.as_mut() {
            call_state
                .call
                .set_services_for_context(services, call_state.values.clone());
            // SAFETY: lifecycle is serialized and the state remains alive.
            unsafe { self.unload_bound(Some(call_state)) };
            call_state
                .call
                .lock()
                .map_err(|_| MexLoadError::Poisoned)?
                .error
                .take()
        } else {
            // SAFETY: no state exists, so no registered callback can access it.
            unsafe { self.unload_bound(None) };
            None
        };
        *state = None;
        lifecycle_error.map_or(Ok(true), |error| Err(invocation_error(error)))
    }

    #[doc(hidden)]
    pub fn clear_with_boundary_services(
        &self,
        services: std::sync::Arc<dyn ConcurrentMexBoundaryHostServices>,
    ) -> Result<bool, MexLoadError> {
        let _process_guard = MEX_PROCESS_GATE.lock();
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Ok(false),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(MexLoadError::Poisoned),
        };
        if self.is_locked() {
            return Ok(false);
        }
        if state
            .as_ref()
            .is_some_and(|state| state.call.has_pending_async_requests().unwrap_or(true))
        {
            return Ok(false);
        }
        let lifecycle_error = if let Some(call_state) = state.as_mut() {
            call_state.call.set_boundary_services(services);
            // SAFETY: lifecycle is serialized and the state remains alive.
            unsafe { self.unload_bound(Some(call_state)) };
            call_state
                .call
                .lock()
                .map_err(|_| MexLoadError::Poisoned)?
                .error
                .take()
        } else {
            // SAFETY: no state exists, so no registered callback can access it.
            unsafe { self.unload_bound(None) };
            None
        };
        *state = None;
        lifecycle_error.map_or(Ok(true), |error| Err(invocation_error(error)))
    }

    /// Force final lifecycle teardown when the owning session exits.
    ///
    /// `mexLock` prevents an interactive `clear`, but cannot extend native
    /// state beyond its owning session. The supplied services remain valid
    /// while registered `mexAtExit` callbacks run.
    pub fn shutdown_with_services(
        &self,
        services: std::rc::Rc<dyn MexHostServices>,
    ) -> Result<(), MexLoadError> {
        let _process_guard = MEX_PROCESS_GATE.lock();
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Err(MexLoadError::ReentrantModule),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(MexLoadError::Poisoned),
        };
        let lifecycle_error = if let Some(call_state) = state.as_mut() {
            call_state
                .call
                .set_services_for_context(services, call_state.values.clone());
            // SAFETY: lifecycle is serialized and the state remains alive.
            unsafe { self.unload_bound(Some(call_state)) };
            call_state
                .call
                .lock()
                .map_err(|_| MexLoadError::Poisoned)?
                .error
                .take()
        } else {
            // SAFETY: no state exists, so no registered callback can reach a
            // host allocation owned by this loader.
            unsafe { self.unload_bound(None) };
            None
        };
        *state = None;
        lifecycle_error.map_or(Ok(()), |error| Err(invocation_error(error)))
    }

    #[doc(hidden)]
    pub fn shutdown_with_boundary_services(
        &self,
        services: std::sync::Arc<dyn ConcurrentMexBoundaryHostServices>,
    ) -> Result<(), MexLoadError> {
        let _process_guard = MEX_PROCESS_GATE.lock();
        let mut state = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Err(MexLoadError::ReentrantModule),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(MexLoadError::Poisoned),
        };
        let lifecycle_error = if let Some(call_state) = state.as_mut() {
            call_state.call.set_boundary_services(services);
            // SAFETY: lifecycle is serialized and the state remains alive.
            unsafe { self.unload_bound(Some(call_state)) };
            call_state
                .call
                .lock()
                .map_err(|_| MexLoadError::Poisoned)?
                .error
                .take()
        } else {
            // SAFETY: no state exists, so no registered callback can reach a
            // host allocation owned by this loader.
            unsafe { self.unload_bound(None) };
            None
        };
        *state = None;
        lifecycle_error.map_or(Ok(()), |error| Err(invocation_error(error)))
    }
}

fn invocation_error(error: crate::MexDiagnostic) -> MexLoadError {
    MexLoadError::Invocation {
        identifier: error
            .identifier
            .map(|value| format!(" ({value})"))
            .unwrap_or_default(),
        message: error.message,
    }
}

fn admit_artifact(path: &Path) -> Result<MexArtifactManifest, MexLoadError> {
    let manifest_path = MexArtifactManifest::path_for_module(path);
    let manifest_bytes =
        std::fs::read(&manifest_path).map_err(|source| MexLoadError::ArtifactManifestRead {
            path: manifest_path.display().to_string(),
            source,
        })?;
    let manifest = MexArtifactManifest::from_canonical_bytes(&manifest_bytes).map_err(|error| {
        MexLoadError::ArtifactManifest {
            path: manifest_path.display().to_string(),
            message: error.to_string(),
        }
    })?;
    let module = std::fs::read(path).map_err(|source| MexLoadError::ModuleRead {
        path: path.display().to_string(),
        source,
    })?;
    manifest
        .validate_current_module(&module)
        .map_err(|error| MexLoadError::ArtifactManifest {
            path: manifest_path.display().to_string(),
            message: error.to_string(),
        })?;
    Ok(manifest)
}

impl Drop for MexModule {
    fn drop(&mut self) {
        let _process_guard = MEX_PROCESS_GATE.lock();
        if let Ok(mut state) = self.state.lock() {
            if let Some(call_state) = state.as_mut() {
                // SAFETY: forced lifecycle is serialized and state remains alive.
                unsafe { self.unload_bound(Some(call_state)) };
                return;
            }
        }
        // SAFETY: there is no accessible state for an exit callback here.
        unsafe { self.unload_bound(None) };
    }
}

#[cfg(test)]
mod tests {
    use super::missing_dependency_from_loader_message;

    #[test]
    fn extracts_named_dependencies_from_platform_loader_diagnostics() {
        assert_eq!(
            missing_dependency_from_loader_message(
                "/tmp/module.mexa64",
                "libhelper.so: cannot open shared object file: No such file or directory",
            )
            .as_deref(),
            Some("libhelper.so")
        );
        assert_eq!(
            missing_dependency_from_loader_message(
                "/tmp/module.mexmaca64",
                "dlopen(/tmp/module.mexmaca64): Library not loaded: @rpath/libhelper.dylib\n  Referenced from: /tmp/module.mexmaca64",
            )
            .as_deref(),
            Some("@rpath/libhelper.dylib")
        );
    }

    #[test]
    fn does_not_mislabel_the_requested_module_as_a_dependency() {
        assert_eq!(
            missing_dependency_from_loader_message(
                "/tmp/module.mexa64",
                "/tmp/module.mexa64: cannot open shared object file: No such file or directory",
            ),
            None
        );
        assert_eq!(
            missing_dependency_from_loader_message(
                "module.mexw64",
                "The specified module could not be found. (os error 126)",
            ),
            None
        );
    }
}
