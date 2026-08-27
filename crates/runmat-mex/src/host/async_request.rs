use std::sync::Arc;
use std::time::Duration;

use crate::{MexDiagnostic, MexEngineCompletion, MxArray};

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

pub trait MexAsyncResult: Send + Sync {
    fn cancel(&self, allow_interrupt: bool) -> bool;
    fn is_ready(&self) -> bool;
    fn wait(&self, timeout: Option<Duration>) -> bool;
    fn result(&self) -> MexEngineCompletion<Vec<MxArray>>;
}

pub trait MexAsyncHostServices: Send + Sync {
    fn submit(
        &self,
        operation: MexAsyncOperation,
    ) -> Result<Arc<dyn MexAsyncResult>, MexDiagnostic>;
}
