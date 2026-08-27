use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use runmat_mex::{MexDiagnostic, MexHostServices, MexModule, MxApiMode};
use runmat_process_host::ipc::{
    authenticate_host_blocking, read_payload_blocking, write_payload_blocking, FrameLimits,
    HostHandshake, SessionSecret,
};
use runmat_process_host::shared_memory::SharedSnapshotStore;
use runmat_value::Value;

use super::{
    decode_value_transfer, encode_value_transfer, DriverMessage, HostMessage, MexBinaryTier,
    MexCallbackOperation, MexCallbackOutput, MexCallbackRequest, MexInvocationOutput,
    MexInvocationRequest, MexInvocationResult, MexLifecycleOutcome, MexLifecycleResult,
    MexWireDiagnostic, MexWireError, MEX_HOST_MAX_CALLBACK_DEPTH, MEX_HOST_MAX_MESSAGE_BYTES,
    MEX_HOST_PROTOCOL, MEX_HOST_SCHEMA_VERSION, MEX_HOST_SECRET_ENV, MEX_HOST_SNAPSHOT_ROOT_ENV,
};

pub fn run_mex_extension_host() -> Result<(), String> {
    let encoded_secret = std::env::var(MEX_HOST_SECRET_ENV)
        .map_err(|_| "extension host bootstrap secret is unavailable".to_string())?;
    std::env::remove_var(MEX_HOST_SECRET_ENV);
    let secret = SessionSecret::from_hex(&encoded_secret).map_err(|error| error.to_string())?;
    drop(encoded_secret);
    let snapshot_root = std::env::var_os(MEX_HOST_SNAPSHOT_ROOT_ENV)
        .ok_or_else(|| "extension host snapshot root is unavailable".to_string())?;
    std::env::remove_var(MEX_HOST_SNAPSHOT_ROOT_ENV);
    let snapshots = Rc::new(
        SharedSnapshotStore::open_existing(PathBuf::from(snapshot_root))
            .map_err(|error| error.to_string())?,
    );

    let mut reader = std::io::stdin();
    let mut writer = std::io::stdout();
    let session = authenticate_host_blocking(
        &mut reader,
        &mut writer,
        HostHandshake::new(
            MEX_HOST_PROTOCOL,
            MEX_HOST_SCHEMA_VERSION,
            MEX_HOST_MAX_MESSAGE_BYTES,
        ),
        &secret,
    )
    .map_err(|error| error.to_string())?;
    let channel = Rc::new(RefCell::new(HostChannel {
        reader,
        writer,
        limits: session.limits,
    }));
    let mut modules = HashMap::<PathBuf, Rc<MexModule>>::new();

    loop {
        let message = channel.borrow_mut().read_driver()?;
        match message {
            DriverMessage::Invoke(request) => {
                let request_id = request.request_id;
                let outcome = invoke(
                    &mut modules,
                    request,
                    Rc::clone(&channel),
                    Rc::clone(&snapshots),
                );
                channel
                    .borrow_mut()
                    .write_host(&HostMessage::Invocation(MexInvocationResult {
                        request_id,
                        outcome,
                    }))?;
            }
            DriverMessage::Clear {
                request_id,
                module_path,
            } => {
                let outcome = clear(
                    &mut modules,
                    request_id,
                    &module_path,
                    Rc::clone(&channel),
                    Rc::clone(&snapshots),
                );
                channel
                    .borrow_mut()
                    .write_host(&HostMessage::Lifecycle(MexLifecycleResult {
                        request_id,
                        outcome,
                    }))?;
            }
            DriverMessage::Shutdown { request_id } => {
                let outcome = shutdown(
                    &mut modules,
                    request_id,
                    Rc::clone(&channel),
                    Rc::clone(&snapshots),
                );
                channel
                    .borrow_mut()
                    .write_host(&HostMessage::Lifecycle(MexLifecycleResult {
                        request_id,
                        outcome: outcome.map(|()| MexLifecycleOutcome::Shutdown),
                    }))?;
                return Ok(());
            }
            DriverMessage::CallbackResult(_) => {
                return Err("callback result arrived outside an active callback".into());
            }
        }
    }
}

fn invoke<R: Read + 'static, W: Write + 'static>(
    modules: &mut HashMap<PathBuf, Rc<MexModule>>,
    request: MexInvocationRequest,
    channel: Rc<RefCell<HostChannel<R, W>>>,
    snapshots: Rc<SharedSnapshotStore>,
) -> Result<MexInvocationOutput, MexWireError> {
    request.validate().map_err(protocol_error)?;
    let path = canonical_module_path(&request.module_path)?;
    let module = load_module(modules, &path, request.tier)?;
    let mode = request.api.map_or_else(
        || module.api_mode(),
        |api| {
            if api.uses_interleaved_complex() {
                MxApiMode::InterleavedComplex
            } else {
                MxApiMode::SeparateComplex
            }
        },
    );
    let arguments = request
        .arguments
        .iter()
        .map(|value| decode_value_transfer(value, &snapshots))
        .collect::<Result<Vec<_>, _>>()?;
    let services = Rc::new(IsolatedMexHostServices::new(
        request.request_id,
        channel,
        Rc::clone(&snapshots),
    ));
    let invocation = module
        .invoke_with_services(
            &arguments,
            request.requested_outputs as usize,
            mode,
            services,
        )
        .map_err(|error| wire_error("RunMat:MEX:Invocation", error.to_string(), None))?;
    let outputs = invocation
        .outputs
        .iter()
        .map(|value| encode_value_transfer(value, &snapshots))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(MexInvocationOutput {
        outputs,
        warnings: invocation
            .warnings
            .into_iter()
            .map(|warning| MexWireDiagnostic {
                identifier: warning.identifier,
                message: warning.message,
            })
            .collect(),
        console: invocation.console,
    })
}

fn clear<R: Read + 'static, W: Write + 'static>(
    modules: &mut HashMap<PathBuf, Rc<MexModule>>,
    request_id: u64,
    path: &str,
    channel: Rc<RefCell<HostChannel<R, W>>>,
    snapshots: Rc<SharedSnapshotStore>,
) -> Result<MexLifecycleOutcome, MexWireError> {
    let path = canonical_module_path(path)?;
    let Some(module) = modules.get(&path).cloned() else {
        return Ok(MexLifecycleOutcome::Cleared);
    };
    let services = Rc::new(IsolatedMexHostServices::new(request_id, channel, snapshots));
    match module.clear_with_services(services) {
        Ok(true) => {
            modules.remove(&path);
            Ok(MexLifecycleOutcome::Cleared)
        }
        Ok(false) => Ok(MexLifecycleOutcome::Retained),
        Err(error) => Err(wire_error("RunMat:MEX:Clear", error.to_string(), None)),
    }
}

fn shutdown<R: Read + 'static, W: Write + 'static>(
    modules: &mut HashMap<PathBuf, Rc<MexModule>>,
    request_id: u64,
    channel: Rc<RefCell<HostChannel<R, W>>>,
    snapshots: Rc<SharedSnapshotStore>,
) -> Result<(), MexWireError> {
    let loaded = modules.drain().collect::<Vec<_>>();
    let mut first_error = None;
    for (_, module) in loaded {
        let services = Rc::new(IsolatedMexHostServices::new(
            request_id,
            Rc::clone(&channel),
            Rc::clone(&snapshots),
        ));
        if let Err(error) = module.shutdown_with_services(services) {
            first_error
                .get_or_insert_with(|| wire_error("RunMat:MEX:Shutdown", error.to_string(), None));
        }
    }
    first_error.map_or(Ok(()), Err)
}

fn load_module(
    modules: &mut HashMap<PathBuf, Rc<MexModule>>,
    path: &Path,
    tier: MexBinaryTier,
) -> Result<Rc<MexModule>, MexWireError> {
    if let Some(module) = modules.get(path) {
        return Ok(Rc::clone(module));
    }
    let loaded = match tier {
        MexBinaryTier::RunMatExact => MexModule::load(path),
        MexBinaryTier::RunMatCompatibleIsolated => MexModule::load_compatible_isolated(path),
    };
    let module = Rc::new(loaded.map_err(|error| {
        wire_error(
            "RunMat:MEX:Load",
            error.to_string(),
            dependency_from_load_error(&error),
        )
    })?);
    modules.insert(path.to_path_buf(), Rc::clone(&module));
    Ok(module)
}

fn canonical_module_path(path: &str) -> Result<PathBuf, MexWireError> {
    let path = PathBuf::from(path);
    if !path.is_absolute() {
        return Err(wire_error(
            "RunMat:MEX:InvalidModulePath",
            "isolated MEX paths must be absolute",
            None,
        ));
    }
    std::fs::canonicalize(&path).map_err(|error| {
        wire_error(
            "RunMat:MEX:ModuleUnavailable",
            format!("could not resolve '{}': {error}", path.display()),
            Some(path.display().to_string()),
        )
    })
}

fn dependency_from_load_error(error: &runmat_mex::MexLoadError) -> Option<String> {
    error.missing_dependency()
}

fn protocol_error(error: impl std::fmt::Display) -> MexWireError {
    wire_error("RunMat:MEX:HostProtocol", error.to_string(), None)
}

fn wire_error(
    identifier: &str,
    message: impl Into<String>,
    dependency: Option<String>,
) -> MexWireError {
    MexWireError {
        identifier: identifier.into(),
        message: message.into(),
        dependency,
    }
}

struct HostChannel<R, W> {
    reader: R,
    writer: W,
    limits: FrameLimits,
}

impl<R: Read, W: Write> HostChannel<R, W> {
    fn read_driver(&mut self) -> Result<DriverMessage, String> {
        let payload = read_payload_blocking(&mut self.reader, self.limits)
            .map_err(|error| error.to_string())?;
        let message: DriverMessage = serde_json::from_slice(&payload)
            .map_err(|error| format!("invalid driver record: {error}"))?;
        message
            .validate()
            .map_err(|error| format!("invalid driver record: {error}"))?;
        Ok(message)
    }

    fn write_host(&mut self, message: &HostMessage) -> Result<(), String> {
        message
            .validate()
            .map_err(|error| format!("invalid host record: {error}"))?;
        let payload = serde_json::to_vec(message)
            .map_err(|error| format!("could not encode host record: {error}"))?;
        write_payload_blocking(&mut self.writer, &payload, self.limits)
            .map_err(|error| error.to_string())
    }
}

struct IsolatedMexHostServices<R, W> {
    request_id: u64,
    next_callback_id: Cell<u64>,
    depth: Cell<u16>,
    channel: Rc<RefCell<HostChannel<R, W>>>,
    snapshots: Rc<SharedSnapshotStore>,
}

impl<R, W> IsolatedMexHostServices<R, W> {
    fn new(
        request_id: u64,
        channel: Rc<RefCell<HostChannel<R, W>>>,
        snapshots: Rc<SharedSnapshotStore>,
    ) -> Self {
        Self {
            request_id,
            next_callback_id: Cell::new(1),
            depth: Cell::new(0),
            channel,
            snapshots,
        }
    }
}

impl<R: Read, W: Write> IsolatedMexHostServices<R, W> {
    fn callback(
        &self,
        operation: MexCallbackOperation,
    ) -> Result<MexCallbackOutput, MexDiagnostic> {
        let depth = self.depth.get().saturating_add(1);
        if depth > MEX_HOST_MAX_CALLBACK_DEPTH {
            return Err(diagnostic(
                "RunMat:MEX:CallbackDepth",
                "callback depth limit exceeded",
            ));
        }
        self.depth.set(depth);
        let callback_id = self.next_callback_id.get();
        self.next_callback_id.set(callback_id.saturating_add(1));
        let result = (|| {
            let mut channel = self.channel.borrow_mut();
            channel
                .write_host(&HostMessage::Callback(MexCallbackRequest {
                    request_id: self.request_id,
                    callback_id,
                    depth,
                    operation,
                }))
                .map_err(|error| diagnostic("RunMat:MEX:CallbackTransport", error))?;
            let response = channel
                .read_driver()
                .map_err(|error| diagnostic("RunMat:MEX:CallbackTransport", error))?;
            let DriverMessage::CallbackResult(response) = response else {
                return Err(diagnostic(
                    "RunMat:MEX:CallbackProtocol",
                    "host expected a callback result",
                ));
            };
            if response.request_id != self.request_id || response.callback_id != callback_id {
                return Err(diagnostic(
                    "RunMat:MEX:CallbackProtocol",
                    "callback result identity does not match the active callback",
                ));
            }
            response.outcome.map_err(|error| MexDiagnostic {
                identifier: Some(error.identifier),
                message: error.message,
            })
        })();
        self.depth.set(depth - 1);
        result
    }
}

impl<R: Read, W: Write> MexHostServices for IsolatedMexHostServices<R, W> {
    fn eval(&self, command: &str) -> Result<(), MexDiagnostic> {
        match self.callback(MexCallbackOperation::Eval {
            command: command.into(),
        })? {
            MexCallbackOutput::Unit => Ok(()),
            _ => Err(diagnostic(
                "RunMat:MEX:CallbackProtocol",
                "eval returned an invalid result",
            )),
        }
    }

    fn call(
        &self,
        function: &str,
        arguments: Vec<Value>,
        requested_outputs: usize,
    ) -> Result<Vec<Value>, MexDiagnostic> {
        let arguments = arguments
            .iter()
            .map(|value| encode_value_transfer(value, &self.snapshots))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| diagnostic(&error.identifier, error.message))?;
        let requested_outputs = u32::try_from(requested_outputs)
            .map_err(|_| diagnostic("RunMat:MEX:CallbackOutputs", "output count exceeds u32"))?;
        match self.callback(MexCallbackOperation::Call {
            function: function.into(),
            arguments,
            requested_outputs,
        })? {
            MexCallbackOutput::Values(values) => values
                .iter()
                .map(|value| decode_value_transfer(value, &self.snapshots))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| diagnostic(&error.identifier, error.message)),
            _ => Err(diagnostic(
                "RunMat:MEX:CallbackProtocol",
                "call returned an invalid result",
            )),
        }
    }

    fn get_variable(&self, workspace: &str, name: &str) -> Result<Option<Value>, MexDiagnostic> {
        match self.callback(MexCallbackOperation::GetVariable {
            workspace: workspace.into(),
            name: name.into(),
        })? {
            MexCallbackOutput::OptionalValue(value) => value
                .as_ref()
                .map(|value| decode_value_transfer(value, &self.snapshots))
                .transpose()
                .map_err(|error| diagnostic(&error.identifier, error.message)),
            _ => Err(diagnostic(
                "RunMat:MEX:CallbackProtocol",
                "workspace read returned an invalid result",
            )),
        }
    }

    fn put_variable(&self, workspace: &str, name: &str, value: Value) -> Result<(), MexDiagnostic> {
        let value = encode_value_transfer(&value, &self.snapshots)
            .map_err(|error| diagnostic(&error.identifier, error.message))?;
        match self.callback(MexCallbackOperation::PutVariable {
            workspace: workspace.into(),
            name: name.into(),
            value,
        })? {
            MexCallbackOutput::Unit => Ok(()),
            _ => Err(diagnostic(
                "RunMat:MEX:CallbackProtocol",
                "workspace write returned an invalid result",
            )),
        }
    }

    fn get_object_property_at(
        &self,
        object: Value,
        index: usize,
        name: &str,
    ) -> Result<Value, MexDiagnostic> {
        let object = encode_value_transfer(&object, &self.snapshots)
            .map_err(|error| diagnostic(&error.identifier, error.message))?;
        let index = u64::try_from(index)
            .map_err(|_| diagnostic("RunMat:MEX:ObjectIndex", "object index exceeds u64"))?;
        match self.callback(MexCallbackOperation::GetObjectProperty {
            object,
            index,
            name: name.into(),
        })? {
            MexCallbackOutput::Values(values) if values.len() == 1 => {
                decode_value_transfer(&values[0], &self.snapshots)
                    .map_err(|error| diagnostic(&error.identifier, error.message))
            }
            _ => Err(diagnostic(
                "RunMat:MEX:CallbackProtocol",
                "property read returned an invalid result",
            )),
        }
    }

    fn set_object_property_at(
        &self,
        object: Value,
        index: usize,
        name: &str,
        value: Value,
    ) -> Result<Value, MexDiagnostic> {
        let object = encode_value_transfer(&object, &self.snapshots)
            .map_err(|error| diagnostic(&error.identifier, error.message))?;
        let value = encode_value_transfer(&value, &self.snapshots)
            .map_err(|error| diagnostic(&error.identifier, error.message))?;
        let index = u64::try_from(index)
            .map_err(|_| diagnostic("RunMat:MEX:ObjectIndex", "object index exceeds u64"))?;
        match self.callback(MexCallbackOperation::SetObjectProperty {
            object,
            index,
            name: name.into(),
            value,
        })? {
            MexCallbackOutput::Values(values) if values.len() == 1 => {
                decode_value_transfer(&values[0], &self.snapshots)
                    .map_err(|error| diagnostic(&error.identifier, error.message))
            }
            _ => Err(diagnostic(
                "RunMat:MEX:CallbackProtocol",
                "property write returned an invalid result",
            )),
        }
    }
}

fn diagnostic(identifier: &str, message: impl Into<String>) -> MexDiagnostic {
    MexDiagnostic {
        identifier: Some(identifier.into()),
        message: message.into(),
    }
}
