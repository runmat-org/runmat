use std::path::Path;
use std::sync::{LazyLock, Mutex};

use libloading::Library;
use parking_lot::ReentrantMutex;
use runmat_value::Value;
use thiserror::Error;

use crate::{
    value_from_mx, value_to_mx, MexCallState, MexHostApiV1, MexHostServices, MxApiMode, MxArray,
    UnavailableMexHostServices,
};

type BindHost = unsafe extern "C" fn(*const MexHostApiV1) -> i32;
type InvokeMex = unsafe extern "C" fn(i32, *mut *mut MxArray, i32, *const *const MxArray) -> i32;
type IsLocked = unsafe extern "C" fn() -> i32;
type Unload = unsafe extern "C" fn();

/// Legacy C MEX code can contain process-global C runtime state. Serialize
/// module loading, invocation, and unloading across sessions while allowing
/// same-thread callback reentry.
static MEX_PROCESS_GATE: LazyLock<ReentrantMutex<()>> = LazyLock::new(|| ReentrantMutex::new(()));

#[derive(Debug, Clone, PartialEq)]
pub struct MexInvocation {
    pub outputs: Vec<Value>,
    pub warnings: Vec<crate::MexDiagnostic>,
    pub console: String,
}

#[derive(Debug, Error)]
pub enum MexLoadError {
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

pub struct MexModule {
    _library: Library,
    bind: BindHost,
    invoke: InvokeMex,
    is_locked: IsLocked,
    unload: Unload,
    state: Mutex<Option<MexCallState>>,
}

impl MexModule {
    pub fn load(path: &Path) -> Result<Self, MexLoadError> {
        let _process_guard = MEX_PROCESS_GATE.lock();
        // SAFETY: the library remains owned by `Self`; all resolved function
        // pointers are copied only after their exact C signatures are checked.
        let library = unsafe { Library::new(path) }.map_err(|source| MexLoadError::Load {
            path: path.display().to_string(),
            source,
        })?;
        // SAFETY: these symbols are supplied by RunMat's compiled C shim and
        // have signatures fixed by the private host ABI header.
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
        // SAFETY: lifecycle symbols are emitted by the same private shim.
        let unload = *unsafe { library.get::<Unload>(b"runmatMexUnload\0") }.map_err(|source| {
            MexLoadError::Symbol {
                symbol: "runmatMexUnload",
                source,
            }
        })?;
        Ok(Self {
            _library: library,
            bind,
            invoke,
            is_locked,
            unload,
            state: Mutex::new(None),
        })
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
        let _process_guard = MEX_PROCESS_GATE.lock();
        let mut state_slot = match self.state.try_lock() {
            Ok(state) => state,
            Err(std::sync::TryLockError::WouldBlock) => return Err(MexLoadError::ReentrantModule),
            Err(std::sync::TryLockError::Poisoned(_)) => return Err(MexLoadError::Poisoned),
        };
        let state =
            state_slot.get_or_insert_with(|| MexCallState::with_services(mode, services.clone()));
        if state.mx.mode() != mode {
            return Err(MexLoadError::Input(
                "a loaded MEX module cannot switch complex API mode".into(),
            ));
        }
        state.set_services(services);
        state.begin_call();
        let mut input_pointers = Vec::with_capacity(inputs.len());
        for input in inputs {
            let value =
                value_to_mx(input, mode).map_err(|error| MexLoadError::Input(error.message))?;
            input_pointers.push(state.mx.allocate(value).cast_const());
        }
        let mut outputs = vec![std::ptr::null_mut(); output_count];
        let api = state.host_api();
        // SAFETY: the API and call arena live through this synchronous call;
        // input/output arrays have the exact lengths passed to the module.
        let status = unsafe {
            if (self.bind)(&api) != 0 {
                1
            } else {
                let status = (self.invoke)(
                    i32::try_from(outputs.len()).unwrap_or(i32::MAX),
                    outputs.as_mut_ptr(),
                    i32::try_from(input_pointers.len()).unwrap_or(i32::MAX),
                    input_pointers.as_ptr(),
                );
                let _ = (self.bind)(std::ptr::null());
                status
            }
        };
        if status != 0 || state.error.is_some() {
            let error = state.error.clone().unwrap_or(crate::MexDiagnostic {
                identifier: None,
                message: format!("gateway returned status {status}"),
            });
            state.finish_call();
            return Err(MexLoadError::Invocation {
                identifier: error
                    .identifier
                    .map(|value| format!(" ({value})"))
                    .unwrap_or_default(),
                message: error.message,
            });
        }
        let converted_outputs = outputs
            .into_iter()
            .enumerate()
            .map(|(index, pointer)| {
                let value = state
                    .mx
                    .arena()
                    .get(pointer)
                    .map_err(|_| MexLoadError::MissingOutput(index))?;
                value_from_mx(value).map_err(|error| MexLoadError::Output(error.message))
            })
            .collect::<Result<Vec<_>, _>>();
        let warnings = std::mem::take(&mut state.warnings);
        let console = std::mem::take(&mut state.console);
        state.finish_call();
        Ok(MexInvocation {
            outputs: converted_outputs?,
            warnings,
            console,
        })
    }

    pub fn is_locked(&self) -> bool {
        // SAFETY: the no-argument shim query has no memory preconditions.
        unsafe { (self.is_locked)() != 0 }
    }

    pub fn clear(&self) -> Result<bool, MexLoadError> {
        let _process_guard = MEX_PROCESS_GATE.lock();
        let mut state = self.state.lock().map_err(|_| MexLoadError::Poisoned)?;
        if self.is_locked() {
            return Ok(false);
        }
        if let Some(call_state) = state.as_mut() {
            let api = call_state.host_api();
            // SAFETY: at-exit runs synchronously while this state and API live.
            unsafe {
                let _ = (self.bind)(&api);
                (self.unload)();
            }
        } else {
            // SAFETY: no state exists, so no registered callback can access it.
            unsafe { (self.unload)() };
        }
        *state = None;
        Ok(true)
    }
}

impl Drop for MexModule {
    fn drop(&mut self) {
        let _process_guard = MEX_PROCESS_GATE.lock();
        if let Ok(mut state) = self.state.lock() {
            if let Some(call_state) = state.as_mut() {
                let api = call_state.host_api();
                // SAFETY: forced shutdown runs the exit hook synchronously
                // while the module, state, and API remain alive.
                unsafe {
                    let _ = (self.bind)(&api);
                    (self.unload)();
                }
                return;
            }
        }
        // SAFETY: there is no accessible state for an exit callback here.
        unsafe { (self.unload)() };
    }
}
