use runmat_value::Value;

use super::MexDiagnostic;

/// Synchronous host operations available while a C gateway is active.
///
/// Adapters must route these methods back into the same session and preserve
/// its workspace, cancellation, and reentrancy policy. The MEX layer never
/// reaches into VM or Core state directly.
pub trait MexHostServices {
    fn eval(&self, command: &str) -> Result<(), MexDiagnostic>;
    fn call(
        &self,
        function: &str,
        arguments: Vec<Value>,
        requested_outputs: usize,
    ) -> Result<Vec<Value>, MexDiagnostic>;
    fn get_variable(&self, workspace: &str, name: &str) -> Result<Option<Value>, MexDiagnostic>;
    fn put_variable(&self, workspace: &str, name: &str, value: Value) -> Result<(), MexDiagnostic>;
}

#[derive(Debug, Default)]
pub struct UnavailableMexHostServices;

impl MexHostServices for UnavailableMexHostServices {
    fn eval(&self, _command: &str) -> Result<(), MexDiagnostic> {
        Err(unavailable("mexEvalString"))
    }

    fn call(
        &self,
        _function: &str,
        _arguments: Vec<Value>,
        _requested_outputs: usize,
    ) -> Result<Vec<Value>, MexDiagnostic> {
        Err(unavailable("mexCallMATLAB"))
    }

    fn get_variable(&self, _workspace: &str, _name: &str) -> Result<Option<Value>, MexDiagnostic> {
        Err(unavailable("mexGetVariable"))
    }

    fn put_variable(
        &self,
        _workspace: &str,
        _name: &str,
        _value: Value,
    ) -> Result<(), MexDiagnostic> {
        Err(unavailable("mexPutVariable"))
    }
}

fn unavailable(operation: &str) -> MexDiagnostic {
    MexDiagnostic {
        identifier: Some("RunMat:MEX:HostServiceUnavailable".into()),
        message: format!("{operation} requires an active RunMat session host"),
    }
}
