use std::path::Path;
use std::sync::Mutex;

use libloading::Library;
use runmat_value::Value;
use thiserror::Error;

use crate::{value_from_mx, value_to_mx, MexCallState, MexHostApiV1, MxApiMode, MxArray};

type BindHost = unsafe extern "C" fn(*const MexHostApiV1) -> i32;
type InvokeMex = unsafe extern "C" fn(i32, *mut *mut MxArray, i32, *const *const MxArray) -> i32;

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
}

pub struct MexModule {
    _library: Library,
    bind: BindHost,
    invoke: InvokeMex,
    invocation: Mutex<()>,
}

impl MexModule {
    pub fn load(path: &Path) -> Result<Self, MexLoadError> {
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
        Ok(Self {
            _library: library,
            bind,
            invoke,
            invocation: Mutex::new(()),
        })
    }

    pub fn invoke(
        &self,
        inputs: &[Value],
        output_count: usize,
        mode: MxApiMode,
    ) -> Result<MexInvocation, MexLoadError> {
        let _guard = self.invocation.lock().map_err(|_| MexLoadError::Poisoned)?;
        let mut state = MexCallState::new(mode);
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
            let error = state.error.unwrap_or(crate::MexDiagnostic {
                identifier: None,
                message: format!("gateway returned status {status}"),
            });
            return Err(MexLoadError::Invocation {
                identifier: error
                    .identifier
                    .map(|value| format!(" ({value})"))
                    .unwrap_or_default(),
                message: error.message,
            });
        }
        let outputs = outputs
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
            .collect::<Result<Vec<_>, _>>()?;
        Ok(MexInvocation {
            outputs,
            warnings: state.warnings,
            console: state.console,
        })
    }
}
