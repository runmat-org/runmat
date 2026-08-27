use runmat_value::Value;
use std::sync::{atomic::AtomicBool, Arc};

use super::MexDiagnostic;

#[derive(Debug, Clone)]
pub struct MexEngineCompletion<T> {
    pub result: Result<T, MexDiagnostic>,
    pub stdout: String,
    pub stderr: String,
}

impl<T> MexEngineCompletion<T> {
    pub fn from_result(result: Result<T, MexDiagnostic>) -> Self {
        Self {
            result,
            stdout: String::new(),
            stderr: String::new(),
        }
    }

    pub fn map<U>(self, map: impl FnOnce(T) -> U) -> MexEngineCompletion<U> {
        MexEngineCompletion {
            result: self.result.map(map),
            stdout: self.stdout,
            stderr: self.stderr,
        }
    }
}

pub trait MexCancellationScope {}

impl MexCancellationScope for () {}

/// Synchronous host operations available while a C gateway is active.
///
/// Adapters must route these methods back into the same session and preserve
/// its workspace, cancellation, and reentrancy policy. The MEX layer never
/// reaches into VM or Core state directly.
pub trait MexHostServices {
    fn cancellation_scope(&self, _cancellation: Arc<AtomicBool>) -> Box<dyn MexCancellationScope> {
        Box::new(())
    }
    fn eval(&self, command: &str) -> Result<(), MexDiagnostic>;
    fn eval_captured(&self, command: &str) -> MexEngineCompletion<()> {
        MexEngineCompletion::from_result(self.eval(command))
    }
    fn call(
        &self,
        function: &str,
        arguments: Vec<Value>,
        requested_outputs: usize,
    ) -> Result<Vec<Value>, MexDiagnostic>;
    fn call_captured(
        &self,
        function: &str,
        arguments: Vec<Value>,
        requested_outputs: usize,
    ) -> MexEngineCompletion<Vec<Value>> {
        MexEngineCompletion::from_result(self.call(function, arguments, requested_outputs))
    }
    fn get_variable(&self, workspace: &str, name: &str) -> Result<Option<Value>, MexDiagnostic>;
    fn put_variable(&self, workspace: &str, name: &str, value: Value) -> Result<(), MexDiagnostic>;
    fn get_object_property(&self, object: Value, name: &str) -> Result<Value, MexDiagnostic> {
        let _ = (object, name);
        Err(unavailable("C++ MEX getProperty"))
    }
    fn get_object_property_at(
        &self,
        object: Value,
        index: usize,
        name: &str,
    ) -> Result<Value, MexDiagnostic> {
        if index == 0 {
            self.get_object_property(object, name)
        } else {
            Err(unavailable("indexed C++ MEX getProperty"))
        }
    }
    fn set_object_property(
        &self,
        object: Value,
        name: &str,
        value: Value,
    ) -> Result<Value, MexDiagnostic> {
        let _ = (object, name, value);
        Err(unavailable("C++ MEX setProperty"))
    }
    fn set_object_property_at(
        &self,
        object: Value,
        index: usize,
        name: &str,
        value: Value,
    ) -> Result<Value, MexDiagnostic> {
        if index == 0 {
            self.set_object_property(object, name, value)
        } else {
            Err(unavailable("indexed C++ MEX setProperty"))
        }
    }
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
