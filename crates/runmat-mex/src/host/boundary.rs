use runmat_execution::cancellation::CancellationToken;
use std::any::Any;
use std::rc::Rc;

use crate::{
    value_from_mx_in_context, value_to_mx_for_interface_in_context, MexCallOutput, MexDiagnostic,
    MexEngineCompletion, MexHostServices, MxApiMode, MxArray, MxBoundaryInterface, MxValueContext,
};

/// Native-lane callback boundary.
///
/// Values at this layer are opaque MEX arrays with thread-transferable storage
/// and origin-thread handle tokens. Runtime `Value` conversion belongs to the
/// origin adapter below, never to a worker-thread proxy.
pub trait MexBoundaryHostServices {
    fn execute_async(
        &self,
        operation: crate::MexAsyncOperation,
        _cancellation: CancellationToken,
    ) -> MexEngineCompletion<Vec<MxArray>> {
        execute_async_operation(self, operation)
    }
    fn eval(&self, command: &str) -> Result<(), MexDiagnostic>;
    fn eval_captured(&self, command: &str) -> MexEngineCompletion<()> {
        MexEngineCompletion::from_result(self.eval(command))
    }
    fn call(
        &self,
        function: &str,
        arguments: Vec<MxArray>,
        requested_outputs: usize,
    ) -> Result<Vec<MxArray>, MexDiagnostic>;
    fn call_captured(
        &self,
        function: &str,
        arguments: Vec<MxArray>,
        requested_outputs: usize,
    ) -> MexEngineCompletion<Vec<MxArray>> {
        MexEngineCompletion::from_result(self.call(function, arguments, requested_outputs))
    }
    fn get_variable(&self, workspace: &str, name: &str) -> Result<Option<MxArray>, MexDiagnostic>;
    fn put_variable(
        &self,
        workspace: &str,
        name: &str,
        value: MxArray,
    ) -> Result<(), MexDiagnostic>;
    fn get_object_property(
        &self,
        object: MxArray,
        index: usize,
        name: &str,
    ) -> Result<MxArray, MexDiagnostic>;
    fn set_object_property(
        &self,
        object: MxArray,
        index: usize,
        name: &str,
        value: MxArray,
    ) -> Result<MxArray, MexDiagnostic>;
}

#[doc(hidden)]
pub struct DirectMexBoundaryHostServices {
    services: Rc<dyn MexHostServices>,
    codec: MxBoundaryCodec,
}

/// Origin-thread MxArray conversion and token registration authority.
#[doc(hidden)]
pub struct MxBoundaryCodec {
    values: Rc<MxValueContext>,
    mode: MxApiMode,
    interface: MxBoundaryInterface,
    _origin_lease: Option<Rc<dyn Any>>,
}

impl DirectMexBoundaryHostServices {
    pub fn new(
        services: Rc<dyn MexHostServices>,
        values: Rc<MxValueContext>,
        mode: MxApiMode,
        interface: MxBoundaryInterface,
    ) -> Self {
        Self {
            services,
            codec: MxBoundaryCodec::new(values, mode, interface),
        }
    }

    /// Keep an origin-thread resource (such as the runtime's registered MEX
    /// token table) alive until this callback service itself retires. Native
    /// commands can outlive a cancelled foreground future.
    pub fn new_with_origin_lease(
        services: Rc<dyn MexHostServices>,
        values: Rc<MxValueContext>,
        mode: MxApiMode,
        interface: MxBoundaryInterface,
        origin_lease: Rc<dyn Any>,
    ) -> Self {
        Self {
            services,
            codec: MxBoundaryCodec::new_with_origin_lease(values, mode, interface, origin_lease),
        }
    }

    #[doc(hidden)]
    pub fn decode(&self, value: &MxArray) -> Result<runmat_value::Value, MexDiagnostic> {
        self.codec.decode(value)
    }

    #[doc(hidden)]
    pub fn encode(&self, value: &runmat_value::Value) -> Result<MxArray, MexDiagnostic> {
        self.codec.encode(value)
    }

    #[doc(hidden)]
    pub fn encode_one(&self, owned: Box<dyn MexCallOutput>) -> Result<MxArray, MexDiagnostic> {
        self.codec.encode_one(owned)
    }
}

impl MxBoundaryCodec {
    pub fn new(
        values: Rc<MxValueContext>,
        mode: MxApiMode,
        interface: MxBoundaryInterface,
    ) -> Self {
        Self {
            values,
            mode,
            interface,
            _origin_lease: None,
        }
    }

    pub fn new_with_origin_lease(
        values: Rc<MxValueContext>,
        mode: MxApiMode,
        interface: MxBoundaryInterface,
        origin_lease: Rc<dyn Any>,
    ) -> Self {
        Self {
            values,
            mode,
            interface,
            _origin_lease: Some(origin_lease),
        }
    }

    #[doc(hidden)]
    pub fn decode(&self, value: &MxArray) -> Result<runmat_value::Value, MexDiagnostic> {
        value_from_mx_in_context(value, Some(&self.values)).map_err(conversion_diagnostic)
    }

    #[doc(hidden)]
    pub fn encode(&self, value: &runmat_value::Value) -> Result<MxArray, MexDiagnostic> {
        value_to_mx_for_interface_in_context(value, self.mode, self.interface, Some(&self.values))
            .map_err(conversion_diagnostic)
    }

    #[doc(hidden)]
    pub fn encode_one(&self, owned: Box<dyn MexCallOutput>) -> Result<MxArray, MexDiagnostic> {
        let mut encoded = None;
        owned.publish(&mut |value| {
            if encoded.is_some() {
                return Err(MexDiagnostic {
                    identifier: Some("RunMat:MEX:CallbackProtocol".into()),
                    message: "single-value callback published multiple values".into(),
                });
            }
            encoded = Some(self.encode(&value)?);
            Ok(())
        })?;
        encoded.ok_or_else(|| MexDiagnostic {
            identifier: Some("RunMat:MEX:CallbackProtocol".into()),
            message: "single-value callback published no value".into(),
        })
    }
}

impl MexBoundaryHostServices for DirectMexBoundaryHostServices {
    fn execute_async(
        &self,
        operation: crate::MexAsyncOperation,
        cancellation: CancellationToken,
    ) -> MexEngineCompletion<Vec<MxArray>> {
        let _cancellation = self.services.cancellation_scope(cancellation);
        execute_async_operation(self, operation)
    }

    fn eval(&self, command: &str) -> Result<(), MexDiagnostic> {
        self.services.eval(command)
    }

    fn eval_captured(&self, command: &str) -> MexEngineCompletion<()> {
        self.services.eval_captured(command)
    }

    fn call(
        &self,
        function: &str,
        arguments: Vec<MxArray>,
        requested_outputs: usize,
    ) -> Result<Vec<MxArray>, MexDiagnostic> {
        let arguments = arguments
            .iter()
            .map(|argument| self.decode(argument))
            .collect::<Result<Vec<_>, _>>()?;
        let owned = self.services.call(function, arguments, requested_outputs)?;
        let mut outputs = Vec::with_capacity(requested_outputs);
        owned.publish(&mut |output| {
            outputs.push(self.encode(&output)?);
            Ok(())
        })?;
        Ok(outputs)
    }

    fn call_captured(
        &self,
        function: &str,
        arguments: Vec<MxArray>,
        requested_outputs: usize,
    ) -> MexEngineCompletion<Vec<MxArray>> {
        let arguments = arguments
            .iter()
            .map(|argument| self.decode(argument))
            .collect::<Result<Vec<_>, _>>();
        let arguments = match arguments {
            Ok(arguments) => arguments,
            Err(error) => return MexEngineCompletion::from_result(Err(error)),
        };
        let completion = self
            .services
            .call_captured(function, arguments, requested_outputs);
        let result = completion.result.and_then(|owned| {
            let mut outputs = Vec::with_capacity(requested_outputs);
            owned.publish(&mut |output| {
                outputs.push(self.encode(&output)?);
                Ok(())
            })?;
            Ok(outputs)
        });
        MexEngineCompletion {
            result,
            stdout: completion.stdout,
            stderr: completion.stderr,
        }
    }

    fn get_variable(&self, workspace: &str, name: &str) -> Result<Option<MxArray>, MexDiagnostic> {
        self.services
            .get_variable(workspace, name)?
            .map(|value| self.encode_one(value))
            .transpose()
    }

    fn put_variable(
        &self,
        workspace: &str,
        name: &str,
        value: MxArray,
    ) -> Result<(), MexDiagnostic> {
        self.services
            .put_variable(workspace, name, self.decode(&value)?)
    }

    fn get_object_property(
        &self,
        object: MxArray,
        index: usize,
        name: &str,
    ) -> Result<MxArray, MexDiagnostic> {
        let object = self.decode(&object)?;
        let value = self.services.get_object_property_at(object, index, name)?;
        self.encode_one(value)
    }

    fn set_object_property(
        &self,
        object: MxArray,
        index: usize,
        name: &str,
        value: MxArray,
    ) -> Result<MxArray, MexDiagnostic> {
        let object = self.decode(&object)?;
        let value = self.decode(&value)?;
        let object = self
            .services
            .set_object_property_at(object, index, name, value)?;
        self.encode_one(object)
    }
}

fn execute_async_operation(
    services: &(impl MexBoundaryHostServices + ?Sized),
    operation: crate::MexAsyncOperation,
) -> MexEngineCompletion<Vec<MxArray>> {
    match operation {
        crate::MexAsyncOperation::Eval {
            command,
            capture_stdout,
            capture_stderr,
        } => {
            let mut completion = if capture_stdout || capture_stderr {
                services.eval_captured(&command)
            } else {
                MexEngineCompletion::from_result(services.eval(&command))
            };
            if !capture_stdout {
                completion.stdout.clear();
            }
            if !capture_stderr {
                completion.stderr.clear();
            }
            completion.map(|()| Vec::new())
        }
        crate::MexAsyncOperation::Call {
            function,
            arguments,
            requested_outputs,
            capture_stdout,
            capture_stderr,
        } => {
            let mut completion = if capture_stdout || capture_stderr {
                services.call_captured(&function, arguments, requested_outputs)
            } else {
                MexEngineCompletion::from_result(services.call(
                    &function,
                    arguments,
                    requested_outputs,
                ))
            };
            if !capture_stdout {
                completion.stdout.clear();
            }
            if !capture_stderr {
                completion.stderr.clear();
            }
            completion
        }
        crate::MexAsyncOperation::GetVariable { workspace, name } => {
            let result = services.get_variable(&workspace, &name).and_then(|value| {
                value.map(|value| vec![value]).ok_or_else(|| MexDiagnostic {
                    identifier: Some("RunMat:MEX:VariableNotFound".into()),
                    message: format!("workspace variable '{name}' was not found"),
                })
            });
            MexEngineCompletion::from_result(result)
        }
        crate::MexAsyncOperation::PutVariable {
            workspace,
            name,
            value,
        } => MexEngineCompletion::from_result(
            services
                .put_variable(&workspace, &name, value)
                .map(|()| Vec::new()),
        ),
        crate::MexAsyncOperation::GetObjectProperty {
            object,
            index,
            name,
        } => MexEngineCompletion::from_result(
            services
                .get_object_property(object, index, &name)
                .map(|value| vec![value]),
        ),
        crate::MexAsyncOperation::SetObjectProperty {
            object,
            index,
            name,
            value,
        } => MexEngineCompletion::from_result(
            services
                .set_object_property(object, index, &name, value)
                .map(|object| vec![object]),
        ),
    }
}

fn conversion_diagnostic(error: crate::MxConversionError) -> MexDiagnostic {
    MexDiagnostic {
        identifier: Some("RunMat:MEX:Conversion".into()),
        message: error.message,
    }
}
