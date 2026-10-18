use super::*;
use crate::execution::{RootedValues, RootedValuesBuilder};
use runmat_mex::{MexAsyncOperation, MxArray, MxBoundaryCodec};
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;

/// Origin-thread work, distinct from the synchronous worker/C ABI request.
pub(crate) enum OriginMexOperation {
    Standard(Box<MexAsyncOperation>),
    OptionalGetVariable { workspace: String, name: String },
}

impl OriginMexOperation {
    pub(crate) fn standard(operation: MexAsyncOperation) -> Self {
        Self::Standard(Box::new(operation))
    }
}

pub(crate) type OriginMexCompletionFuture =
    Pin<Box<dyn Future<Output = MexEngineCompletion<Vec<MxArray>>> + 'static>>;

pub(crate) trait OriginMexBoundaryServices {
    fn prepare(
        self: Rc<Self>,
        operation: OriginMexOperation,
        cancellation: CancellationToken,
    ) -> Result<OriginMexCompletionFuture, MexDiagnostic>;
}

/// The Direct adapter remains the sole MxArray codec/token-table owner. This
/// origin adapter contributes only awaitable Runtime producers and rooted
/// admission; it never changes the native worker's synchronous ABI.
pub(crate) struct OriginMexBoundaryAdapter {
    host: Rc<RuntimeMexHostServices>,
    codec: Rc<MxBoundaryCodec>,
}

impl OriginMexBoundaryAdapter {
    pub(crate) fn new(host: Rc<RuntimeMexHostServices>, codec: Rc<MxBoundaryCodec>) -> Self {
        Self { host, codec }
    }

    fn decode_one(&self, value: &MxArray) -> Result<RootedValue, MexDiagnostic> {
        RootedValue::new(self.codec.decode(value)?)
            .map_err(|error| runtime_diagnostic(error.into_runtime_error()))
    }

    fn decode_many(&self, values: &[MxArray]) -> Result<RootedValues, MexDiagnostic> {
        let mut decoded = RootedValuesBuilder::with_capacity(values.len());
        for value in values {
            decoded
                .push(self.codec.decode(value)?)
                .map_err(|error| runtime_diagnostic(error.into_runtime_error()))?;
        }
        decoded
            .finish()
            .map_err(|error| runtime_diagnostic(error.into_runtime_error()))
    }
}

enum PreparedOriginOperation {
    Eval {
        command: String,
        stdout: bool,
        stderr: bool,
    },
    Call {
        function: String,
        arguments: RootedValues,
        requested_outputs: usize,
        stdout: bool,
        stderr: bool,
    },
    GetVariable {
        workspace: String,
        name: String,
        required: bool,
    },
    PutVariable {
        workspace: String,
        name: String,
        value: RootedValue,
    },
    GetObjectProperty {
        object: RootedValue,
        index: usize,
        name: String,
    },
    SetObjectProperty {
        object: RootedValue,
        index: usize,
        name: String,
        value: RootedValue,
    },
}

impl OriginMexBoundaryServices for OriginMexBoundaryAdapter {
    fn prepare(
        self: Rc<Self>,
        operation: OriginMexOperation,
        cancellation: CancellationToken,
    ) -> Result<OriginMexCompletionFuture, MexDiagnostic> {
        let ancestry = self
            .host
            .runtime
            .mex_callback_ancestry()
            .child()
            .ok_or_else(|| MexDiagnostic {
                identifier: Some("RunMat:MEX:CallbackDepth".into()),
                message: "callback depth limit exceeded".into(),
            })?;
        let operation = match operation {
            OriginMexOperation::OptionalGetVariable { workspace, name } => {
                PreparedOriginOperation::GetVariable {
                    workspace,
                    name,
                    required: false,
                }
            }
            OriginMexOperation::Standard(operation) => match *operation {
                MexAsyncOperation::Eval {
                    command,
                    capture_stdout,
                    capture_stderr,
                } => PreparedOriginOperation::Eval {
                    command,
                    stdout: capture_stdout,
                    stderr: capture_stderr,
                },
                MexAsyncOperation::Call {
                    function,
                    arguments,
                    requested_outputs,
                    capture_stdout,
                    capture_stderr,
                } => PreparedOriginOperation::Call {
                    function,
                    arguments: self.decode_many(&arguments)?,
                    requested_outputs,
                    stdout: capture_stdout,
                    stderr: capture_stderr,
                },
                MexAsyncOperation::GetVariable { workspace, name } => {
                    PreparedOriginOperation::GetVariable {
                        workspace,
                        name,
                        required: true,
                    }
                }
                MexAsyncOperation::PutVariable {
                    workspace,
                    name,
                    value,
                } => PreparedOriginOperation::PutVariable {
                    workspace,
                    name,
                    value: self.decode_one(&value)?,
                },
                MexAsyncOperation::GetObjectProperty {
                    object,
                    index,
                    name,
                } => PreparedOriginOperation::GetObjectProperty {
                    object: self.decode_one(&object)?,
                    index,
                    name,
                },
                MexAsyncOperation::SetObjectProperty {
                    object,
                    index,
                    name,
                    value,
                } => PreparedOriginOperation::SetObjectProperty {
                    object: self.decode_one(&object)?,
                    index,
                    name,
                    value: self.decode_one(&value)?,
                },
            },
        };
        let host = Rc::new(
            RuntimeMexHostServices::new(
                self.host
                    .runtime
                    .clone()
                    .with_mex_callback_ancestry(ancestry),
            )
            .with_request_cancellation(cancellation),
        );
        let runtime = host.runtime.clone();
        Ok(Box::pin(runtime.scope(async move {
            self.execute_prepared(host.as_ref(), operation).await
        })))
    }
}

impl OriginMexBoundaryAdapter {
    async fn execute_prepared(
        &self,
        host: &RuntimeMexHostServices,
        operation: PreparedOriginOperation,
    ) -> MexEngineCompletion<Vec<MxArray>> {
        match operation {
            PreparedOriginOperation::Eval {
                command,
                stdout,
                stderr,
            } => {
                capture_async(&host.runtime, stdout, stderr, async move {
                    host.eval_rooted(command)
                        .await
                        .map_err(runtime_diagnostic)
                        .map(|()| Vec::new())
                })
                .await
            }
            PreparedOriginOperation::Call {
                function,
                arguments,
                requested_outputs,
                stdout,
                stderr,
            } => {
                capture_async(&host.runtime, stdout, stderr, async move {
                    let sequence = arguments
                        .with_unbounded_values_async(|arguments| {
                            host.call_rooted(&function, arguments, requested_outputs)
                        })
                        .await
                        .map_err(runtime_diagnostic)?
                        .map_err(runtime_diagnostic)?;
                    let output = Box::new(RuntimeMexCallOutput {
                        sequence,
                        requested_outputs,
                    });
                    let mut encoded = Vec::with_capacity(requested_outputs);
                    output.publish(&mut |value| {
                        encoded.push(self.codec.encode(&value)?);
                        Ok(())
                    })?;
                    Ok(encoded)
                })
                .await
            }
            PreparedOriginOperation::GetVariable {
                workspace,
                name,
                required,
            } => {
                let result = host.get_variable(&workspace, &name).and_then(|value| {
                    if required && value.is_none() {
                        return Err(MexDiagnostic {
                            identifier: Some("RunMat:MEX:VariableNotFound".into()),
                            message: format!("workspace variable '{name}' was not found"),
                        });
                    }
                    value
                        .map(|value| self.codec.encode_one(value).map(|value| vec![value]))
                        .unwrap_or_else(|| Ok(Vec::new()))
                });
                MexEngineCompletion::from_result(result)
            }
            PreparedOriginOperation::PutVariable {
                workspace,
                name,
                value,
            } => MexEngineCompletion::from_result(value.with_owned(|value| {
                host.put_variable(&workspace, &name, value)
                    .map(|()| Vec::new())
            })),
            PreparedOriginOperation::GetObjectProperty {
                object,
                index,
                name,
            } => {
                let result = object
                    .with_owned_async(|object| host.get_object_property_rooted(object, index, name))
                    .await
                    .and_then(|value| {
                        self.codec
                            .encode_one(Box::new(RuntimeMexValueOutput(value)))
                            .map(|value| vec![value])
                    });
                MexEngineCompletion::from_result(result)
            }
            PreparedOriginOperation::SetObjectProperty {
                object,
                index,
                name,
                value,
            } => {
                let result = object
                    .with_owned_async(|object| {
                        value.with_owned_async(|value| {
                            host.set_object_property_rooted(object, index, name, value)
                        })
                    })
                    .await
                    .and_then(|value| {
                        self.codec
                            .encode_one(Box::new(RuntimeMexValueOutput(value)))
                            .map(|value| vec![value])
                    });
                MexEngineCompletion::from_result(result)
            }
        }
    }
}

async fn capture_async<T>(
    runtime: &RuntimeContext,
    stdout: bool,
    stderr: bool,
    operation: impl Future<Output = Result<T, MexDiagnostic>>,
) -> MexEngineCompletion<T> {
    if !stdout && !stderr {
        return MexEngineCompletion::from_result(operation.await);
    }
    if let Err(error) = super::prepare_legacy_capture_materialization(runtime) {
        return MexEngineCompletion::from_result(Err(runtime_diagnostic(error)));
    }
    let capture = match crate::console::begin_capture() {
        Ok(capture) => capture,
        Err(error) => return MexEngineCompletion::from_result(Err(runtime_diagnostic(error))),
    };
    let result = operation.await;
    let entries = capture.finish_entries();
    let mut output = MexEngineCompletion::from_result(result);
    for entry in entries {
        match entry.stream {
            crate::console::ConsoleStream::Stdout if stdout => output.stdout.push_str(&entry.text),
            crate::console::ConsoleStream::Stderr if stderr => output.stderr.push_str(&entry.text),
            _ => {}
        }
    }
    output
}

#[cfg(test)]
mod ancestry_tests {
    use super::*;
    use crate::execution::RuntimeExecutionService;
    use crate::foreign::MexCallbackAncestry;
    use runmat_execution::cancellation::CancellationSource;
    use runmat_mex::{MxApiMode, MxBoundaryInterface, MxValueContext};

    #[test]
    fn native_origin_admission_checks_logical_parent_before_preparation() {
        let runtime = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()))
            .with_mex_callback_ancestry(MexCallbackAncestry::from_wire(63).unwrap());
        let codec = Rc::new(MxBoundaryCodec::new(
            Rc::new(MxValueContext::new()),
            MxApiMode::InterleavedComplex,
            MxBoundaryInterface::CMatrix,
        ));
        let cancellation = CancellationSource::new().token();
        let admitted = Rc::new(OriginMexBoundaryAdapter::new(
            Rc::new(RuntimeMexHostServices::new(runtime.clone())),
            Rc::clone(&codec),
        ))
        .prepare(
            OriginMexOperation::OptionalGetVariable {
                workspace: "base".into(),
                name: "absent".into(),
            },
            cancellation.clone(),
        )
        .expect("the 64th logical callback is admitted");
        drop(admitted);
        assert_eq!(runtime.mex_callback_ancestry().wire(), 63);

        let maximum =
            runtime.with_mex_callback_ancestry(MexCallbackAncestry::from_wire(64).unwrap());
        let rejected = Rc::new(OriginMexBoundaryAdapter::new(
            Rc::new(RuntimeMexHostServices::new(maximum.clone())),
            codec,
        ))
        .prepare(
            OriginMexOperation::OptionalGetVariable {
                workspace: "base".into(),
                name: "absent".into(),
            },
            cancellation,
        );
        let error = match rejected {
            Ok(_) => panic!("the 65th logical callback must be rejected before effects"),
            Err(error) => error,
        };
        assert_eq!(
            error.identifier.as_deref(),
            Some("RunMat:MEX:CallbackDepth")
        );
        assert_eq!(maximum.mex_callback_ancestry().wire(), 64);
    }
}
