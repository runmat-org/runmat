#[cfg(target_family = "wasm")]
use std::rc::Rc;
#[cfg(not(target_family = "wasm"))]
use std::sync::Arc;
use std::time::Duration;

use crate::{MexDiagnostic, MexEngineCompletion, MxArray};

/// Native engine requests may cross worker threads. Browser requests and their
/// real result allocations belong to the originating WASM worker instead.
/// This bound does not grant concurrent-host-service access.
#[doc(hidden)]
#[cfg(not(target_family = "wasm"))]
pub trait MexAsyncThreadBound: Send + Sync {}
#[cfg(not(target_family = "wasm"))]
impl<T: Send + Sync + ?Sized> MexAsyncThreadBound for T {}

#[doc(hidden)]
#[cfg(target_family = "wasm")]
pub trait MexAsyncThreadBound {}
#[cfg(target_family = "wasm")]
impl<T: ?Sized> MexAsyncThreadBound for T {}

#[derive(Debug, Clone)]
pub enum MexAsyncOperation {
    Eval {
        command: String,
        capture_stdout: bool,
        capture_stderr: bool,
    },
    Call {
        function: String,
        arguments: Vec<MxArray>,
        requested_outputs: usize,
        capture_stdout: bool,
        capture_stderr: bool,
    },
    GetVariable {
        workspace: String,
        name: String,
    },
    PutVariable {
        workspace: String,
        name: String,
        value: MxArray,
    },
    GetObjectProperty {
        object: MxArray,
        index: usize,
        name: String,
    },
    SetObjectProperty {
        object: MxArray,
        index: usize,
        name: String,
        value: MxArray,
    },
}

pub trait MexAsyncResult: MexAsyncThreadBound {
    fn cancel(&self, allow_interrupt: bool) -> bool;
    fn is_ready(&self) -> bool;
    fn wait(&self, timeout: Option<Duration>) -> bool;
    fn result(&self) -> MexEngineCompletion<Vec<MxArray>>;
}

/// A native result can cross threads; a browser result stays on its originating
/// WASM worker with the `MxArray` allocations it retains.
#[cfg(not(target_family = "wasm"))]
pub type MexAsyncResultHandle = Arc<dyn MexAsyncResult>;
#[cfg(target_family = "wasm")]
pub type MexAsyncResultHandle = Rc<dyn MexAsyncResult>;

pub(super) fn new_async_result<T: MexAsyncResult + 'static>(result: T) -> MexAsyncResultHandle {
    #[cfg(not(target_family = "wasm"))]
    {
        Arc::new(result)
    }
    #[cfg(target_family = "wasm")]
    {
        Rc::new(result)
    }
}

pub trait MexAsyncHostServices: MexAsyncThreadBound {
    fn submit(&self, operation: MexAsyncOperation) -> Result<MexAsyncResultHandle, MexDiagnostic>;
}

#[cfg(test)]
#[path = "async_request_tests.rs"]
mod tests;
