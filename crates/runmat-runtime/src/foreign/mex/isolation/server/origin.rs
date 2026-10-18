use super::*;
use crate::execution::{RootedValue, RootedValues, RootedValuesBuilder};
use crate::foreign::mex::services::{
    OriginMexBoundaryServices, OriginMexCompletionFuture, OriginMexOperation,
};
use runmat_mex::{MexAsyncOperation, MexEngineCompletion, MxArray};

pub(super) struct IsolatedOriginBoundaryAdapter<W: Write> {
    host: Rc<IsolatedMexHostServices<W>>,
    codec: Rc<MxBoundaryCodec>,
}

impl<W: Write> IsolatedOriginBoundaryAdapter<W> {
    pub(super) fn new(host: Rc<IsolatedMexHostServices<W>>, codec: Rc<MxBoundaryCodec>) -> Self {
        Self { host, codec }
    }

    fn decode_one(&self, value: &MxArray) -> Result<RootedValue, MexDiagnostic> {
        RootedValue::new(self.codec.decode(value)?)
            .map_err(|error| diagnostic("RunMat:MEX:ValueRoot", error.to_string()))
    }

    fn decode_many(&self, values: &[MxArray]) -> Result<RootedValues, MexDiagnostic> {
        let mut decoded = RootedValuesBuilder::with_capacity(values.len());
        for value in values {
            decoded
                .push(self.codec.decode(value)?)
                .map_err(|error| diagnostic("RunMat:MEX:ValueRoot", error.to_string()))?;
        }
        decoded
            .finish()
            .map_err(|error| diagnostic("RunMat:MEX:ValueRoot", error.to_string()))
    }

    fn callback_error(error: MexWireError) -> MexDiagnostic {
        diagnostic(&error.identifier, error.message)
    }

    fn encode_callback_values(
        &self,
        values: &[super::super::MexValueTransfer],
    ) -> Result<Vec<MxArray>, MexDiagnostic> {
        let values = decode_value_transfers_rooted(values, &self.host.snapshots)
            .map_err(Self::callback_error)?;
        values.with_owned_output_storage(|values| {
            values
                .iter()
                .map(|value| self.codec.encode(value))
                .collect()
        })
    }

    async fn execute(
        &self,
        operation: Prepared,
        cancellation: runmat_execution::cancellation::CancellationToken,
    ) -> Result<Vec<MxArray>, MexDiagnostic> {
        match operation {
            Prepared::Eval(command) => match self
                .host
                .callback(MexCallbackOperation::Eval { command }, &cancellation)
                .await?
            {
                MexCallbackOutput::Unit => Ok(Vec::new()),
                _ => Err(diagnostic(
                    "RunMat:MEX:CallbackProtocol",
                    "eval returned an invalid result",
                )),
            },
            Prepared::Call {
                function,
                arguments,
                requested_outputs,
            } => {
                let transfers = arguments
                    .values()
                    .iter()
                    .map(|value| {
                        encode_value_transfer(value, &self.host.snapshots)
                            .map_err(Self::callback_error)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let requested_outputs = u32::try_from(requested_outputs).map_err(|_| {
                    diagnostic("RunMat:MEX:CallbackOutputs", "output count exceeds u32")
                })?;
                match self
                    .host
                    .callback(
                        MexCallbackOperation::Call {
                            function,
                            arguments: transfers,
                            requested_outputs,
                        },
                        &cancellation,
                    )
                    .await?
                {
                    MexCallbackOutput::Values(values) => self.encode_callback_values(&values),
                    _ => Err(diagnostic(
                        "RunMat:MEX:CallbackProtocol",
                        "call returned an invalid result",
                    )),
                }
            }
            Prepared::GetVariable {
                workspace,
                name,
                required,
            } => {
                let requested_name = name.clone();
                match self
                    .host
                    .callback(
                        MexCallbackOperation::GetVariable { workspace, name },
                        &cancellation,
                    )
                    .await?
                {
                    MexCallbackOutput::OptionalValue(value) => {
                        if required && value.is_none() {
                            Err(diagnostic(
                                "RunMat:MEX:VariableNotFound",
                                format!("workspace variable '{requested_name}' was not found"),
                            ))
                        } else {
                            value
                                .as_ref()
                                .map(|value| {
                                    let value = decode_value_transfer_rooted_value(
                                        value,
                                        &self.host.snapshots,
                                    )
                                    .map_err(Self::callback_error)?;
                                    value
                                        .with_owned(|value| self.codec.encode(&value))
                                        .map(|value| vec![value])
                                })
                                .unwrap_or_else(|| Ok(Vec::new()))
                        }
                    }
                    _ => Err(diagnostic(
                        "RunMat:MEX:CallbackProtocol",
                        "workspace read returned an invalid result",
                    )),
                }
            }
            Prepared::PutVariable {
                workspace,
                name,
                value,
            } => {
                let value = encode_value_transfer(value.value(), &self.host.snapshots)
                    .map_err(Self::callback_error)?;
                match self
                    .host
                    .callback(
                        MexCallbackOperation::PutVariable {
                            workspace,
                            name,
                            value,
                        },
                        &cancellation,
                    )
                    .await?
                {
                    MexCallbackOutput::Unit => Ok(Vec::new()),
                    _ => Err(diagnostic(
                        "RunMat:MEX:CallbackProtocol",
                        "workspace write returned an invalid result",
                    )),
                }
            }
            Prepared::GetObjectProperty {
                object,
                index,
                name,
            } => {
                let object = encode_value_transfer(object.value(), &self.host.snapshots)
                    .map_err(Self::callback_error)?;
                let index = u64::try_from(index).map_err(|_| {
                    diagnostic("RunMat:MEX:ObjectIndex", "object index exceeds u64")
                })?;
                match self
                    .host
                    .callback(
                        MexCallbackOperation::GetObjectProperty {
                            object,
                            index,
                            name,
                        },
                        &cancellation,
                    )
                    .await?
                {
                    MexCallbackOutput::Values(values) if values.len() == 1 => {
                        self.encode_callback_values(&values)
                    }
                    _ => Err(diagnostic(
                        "RunMat:MEX:CallbackProtocol",
                        "property read returned an invalid result",
                    )),
                }
            }
            Prepared::SetObjectProperty {
                object,
                index,
                name,
                value,
            } => {
                let object = encode_value_transfer(object.value(), &self.host.snapshots)
                    .map_err(Self::callback_error)?;
                let value = encode_value_transfer(value.value(), &self.host.snapshots)
                    .map_err(Self::callback_error)?;
                let index = u64::try_from(index).map_err(|_| {
                    diagnostic("RunMat:MEX:ObjectIndex", "object index exceeds u64")
                })?;
                match self
                    .host
                    .callback(
                        MexCallbackOperation::SetObjectProperty {
                            object,
                            index,
                            name,
                            value,
                        },
                        &cancellation,
                    )
                    .await?
                {
                    MexCallbackOutput::Values(values) if values.len() == 1 => {
                        self.encode_callback_values(&values)
                    }
                    _ => Err(diagnostic(
                        "RunMat:MEX:CallbackProtocol",
                        "property write returned an invalid result",
                    )),
                }
            }
        }
    }
}

enum Prepared {
    Eval(String),
    Call {
        function: String,
        arguments: RootedValues,
        requested_outputs: usize,
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

impl<W: Write + 'static> OriginMexBoundaryServices for IsolatedOriginBoundaryAdapter<W> {
    fn prepare(
        self: Rc<Self>,
        operation: OriginMexOperation,
        cancellation: runmat_execution::cancellation::CancellationToken,
    ) -> Result<OriginMexCompletionFuture, MexDiagnostic> {
        let prepared = match operation {
            OriginMexOperation::OptionalGetVariable { workspace, name } => Prepared::GetVariable {
                workspace,
                name,
                required: false,
            },
            OriginMexOperation::Standard(operation) => match *operation {
                MexAsyncOperation::Eval { command, .. } => Prepared::Eval(command),
                MexAsyncOperation::Call {
                    function,
                    arguments,
                    requested_outputs,
                    ..
                } => Prepared::Call {
                    function,
                    arguments: self.decode_many(&arguments)?,
                    requested_outputs,
                },
                MexAsyncOperation::GetVariable { workspace, name } => Prepared::GetVariable {
                    workspace,
                    name,
                    required: true,
                },
                MexAsyncOperation::PutVariable {
                    workspace,
                    name,
                    value,
                } => Prepared::PutVariable {
                    workspace,
                    name,
                    value: self.decode_one(&value)?,
                },
                MexAsyncOperation::GetObjectProperty {
                    object,
                    index,
                    name,
                } => Prepared::GetObjectProperty {
                    object: self.decode_one(&object)?,
                    index,
                    name,
                },
                MexAsyncOperation::SetObjectProperty {
                    object,
                    index,
                    name,
                    value,
                } => Prepared::SetObjectProperty {
                    object: self.decode_one(&object)?,
                    index,
                    name,
                    value: self.decode_one(&value)?,
                },
            },
        };
        Ok(Box::pin(async move {
            MexEngineCompletion::from_result(self.execute(prepared, cancellation).await)
        }))
    }
}
