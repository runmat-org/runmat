use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use runmat_python::{
    discover_python, PythonCall, PythonCallbackInvocation, PythonDiscoveryRequest, PythonError,
    PythonExecutionMode, PythonObjectHandle, PythonSession, PythonSessionConfig, PythonValue,
    PYTHON_ADAPTER_ID, PYTHON_ADAPTER_VERSION,
};
use runmat_types::{
    CapabilityRequirement, ForeignAffinity, ForeignCapability, ForeignLifetime, ForeignOwnership,
    ForeignTypeIdentity,
};
use runmat_value::{ForeignRef, ForeignResourceKey, ObjectInstance, Value, WeakForeignRef};

use super::super::{
    foreign_callback_request, foreign_error, invoke_foreign_callback, ForeignAdapter,
    ForeignAdapterDescriptor, ForeignAdapterFuture, ForeignErrorKind, ForeignExecutionPolicy,
    ForeignHandleRegistry, ForeignHostRegistration, ForeignHostRelease, ForeignResourceMetadata,
};
use super::conversion::{value_from_python, value_to_python};
use crate::context::{ForeignCall, RuntimeContext};
use crate::{build_runtime_error, RuntimeError};

#[derive(Debug, Default)]
struct ReleaseQueue(Mutex<Vec<u64>>);

#[derive(Clone)]
struct PythonCallbackRegistration {
    callback: Value,
    context: RuntimeContext,
    foreign: std::rc::Weak<dyn crate::context::RuntimeForeignService>,
}

impl ForeignHostRelease for ReleaseQueue {
    fn release(&self, key: &ForeignResourceKey) {
        if let Ok(mut released) = self.0.lock() {
            released.push(key.handle);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PythonRuntimeConfiguration {
    pub executable: Option<std::path::PathBuf>,
    pub version: Option<(u16, u16)>,
    pub minimum_version: (u16, u16),
    pub maximum_version: Option<(u16, u16)>,
    pub execution_mode: PythonExecutionMode,
}

impl Default for PythonRuntimeConfiguration {
    fn default() -> Self {
        Self {
            executable: None,
            version: None,
            minimum_version: (3, 9),
            maximum_version: None,
            execution_mode: PythonExecutionMode::InProcess,
        }
    }
}

pub struct PythonAdapter {
    handles: ForeignHandleRegistry,
    host_identity: String,
    session: RefCell<Option<PythonSession>>,
    configuration: RefCell<PythonRuntimeConfiguration>,
    released: Arc<ReleaseQueue>,
    resources: RefCell<BTreeMap<u64, PythonObjectHandle>>,
    python_to_foreign: RefCell<BTreeMap<PythonObjectHandle, WeakForeignRef>>,
    callbacks: RefCell<BTreeMap<u64, PythonCallbackRegistration>>,
    next_callback: Cell<u64>,
    terminated: Cell<bool>,
}

impl std::fmt::Debug for PythonAdapter {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PythonAdapter")
            .field("host_identity", &self.host_identity)
            .field("running", &self.session.borrow().is_some())
            .field("configuration", &self.configuration.borrow())
            .finish_non_exhaustive()
    }
}

impl PythonAdapter {
    pub fn new(handles: ForeignHandleRegistry) -> Result<Rc<Self>, RuntimeError> {
        Self::with_configuration(handles, PythonRuntimeConfiguration::default())
    }

    pub fn with_configuration(
        handles: ForeignHandleRegistry,
        configuration: PythonRuntimeConfiguration,
    ) -> Result<Rc<Self>, RuntimeError> {
        let host_identity = "python-session".to_owned();
        let released = Arc::new(ReleaseQueue::default());
        handles.register_host(ForeignHostRegistration {
            identity: host_identity.clone(),
            adapter: PYTHON_ADAPTER_ID.into(),
            session_identity: host_identity.clone(),
            capabilities: BTreeSet::from([
                ForeignCapability::Invoke,
                ForeignCapability::Read,
                ForeignCapability::Write,
                ForeignCapability::Callback,
                ForeignCapability::Transfer,
                ForeignCapability::ZeroCopy,
            ]),
            policy: policy(configuration.execution_mode),
            release: released.clone(),
        })?;
        Ok(Rc::new(Self {
            handles,
            host_identity,
            session: RefCell::new(None),
            configuration: RefCell::new(configuration),
            released,
            resources: RefCell::new(BTreeMap::new()),
            python_to_foreign: RefCell::new(BTreeMap::new()),
            callbacks: RefCell::new(BTreeMap::new()),
            next_callback: Cell::new(1),
            terminated: Cell::new(false),
        }))
    }

    pub fn configure(&self, configuration: PythonRuntimeConfiguration) -> Result<(), RuntimeError> {
        if self.session.borrow().is_some() {
            return Err(invalid_call(
                "Python configuration is immutable after the interpreter starts",
            ));
        }
        self.handles
            .set_host_policy(&self.host_identity, policy(configuration.execution_mode))?;
        *self.configuration.borrow_mut() = configuration;
        self.terminated.set(false);
        Ok(())
    }

    pub fn is_running(&self) -> bool {
        self.session.borrow().is_some()
    }

    fn ensure_session(&self) -> Result<PythonSession, RuntimeError> {
        if let Some(session) = self.session.borrow().clone() {
            return Ok(session);
        }
        let configuration = self.configuration.borrow().clone();
        if configuration.execution_mode == PythonExecutionMode::OutOfProcess {
            return Err(build_runtime_error(
                "the isolated Python host has not been installed for this RunMat executable",
            )
            .with_builtin("python")
            .with_identifier("RunMat:Python:IsolatedHostUnavailable")
            .build());
        }
        let installation =
            discover_python(&discovery_request(&configuration)).map_err(python_runtime_error)?;
        let session = PythonSession::start(PythonSessionConfig { installation })
            .map_err(python_runtime_error)?;
        *self.session.borrow_mut() = Some(session.clone());
        self.terminated.set(false);
        Ok(session)
    }

    fn drain_releases(&self) -> Result<(), RuntimeError> {
        let handles = {
            let mut released = self.released.0.lock().map_err(|_| {
                foreign_error(
                    ForeignErrorKind::HostUnavailable,
                    "Python release queue lock was poisoned",
                )
            })?;
            std::mem::take(&mut *released)
        };
        let Some(session) = self.session.borrow().clone() else {
            return Ok(());
        };
        for foreign_handle in handles {
            if let Some(python_handle) = self.resources.borrow_mut().remove(&foreign_handle) {
                self.python_to_foreign.borrow_mut().remove(&python_handle);
                session
                    .invoke(PythonCall::Release {
                        handle: python_handle,
                    })
                    .map_err(python_runtime_error)?;
            }
        }
        Ok(())
    }

    fn invoke_now(
        &self,
        context: &RuntimeContext,
        call: ForeignCall,
    ) -> Result<Value, RuntimeError> {
        match call.symbol.as_str() {
            "status" => return self.status_value(),
            "configure" => return self.configure_value(call.arguments),
            "terminate" => return self.terminate_value(),
            _ => {}
        }
        let session = self.ensure_session()?;
        self.drain_releases()?;
        let mut arguments = call.arguments.into_iter();
        let mut assigned_receiver = None;
        let python_call = match call.symbol.as_str() {
            "invoke_qualified" => PythonCall::InvokeQualified {
                name: string_argument(arguments.next(), "Python name")?,
                arguments: arguments
                    .map(|value| self.argument_to_python(context, value))
                    .collect::<Result<_, _>>()?,
            },
            "get_member" => PythonCall::GetMember {
                receiver: self.python_handle(&foreign_argument(arguments.next())?)?,
                name: string_argument(arguments.next(), "Python attribute")?,
            },
            "set_member" => {
                let reference = foreign_argument(arguments.next())?;
                assigned_receiver = Some(reference.clone());
                PythonCall::SetMember {
                    receiver: self.python_handle(&reference)?,
                    name: string_argument(arguments.next(), "Python attribute")?,
                    value: self.argument_to_python(
                        context,
                        arguments.next().ok_or_else(|| {
                            invalid_call("Python attribute assignment requires a value")
                        })?,
                    )?,
                }
            }
            "invoke_member" => PythonCall::InvokeMember {
                receiver: self.python_handle(&foreign_argument(arguments.next())?)?,
                name: string_argument(arguments.next(), "Python method")?,
                arguments: arguments
                    .map(|value| self.argument_to_python(context, value))
                    .collect::<Result<_, _>>()?,
            },
            "get_item" => PythonCall::GetItem {
                receiver: self.python_handle(&foreign_argument(arguments.next())?)?,
                index: python_indices(arguments.next())?,
            },
            "set_item" => PythonCall::SetItem {
                receiver: {
                    let reference = foreign_argument(arguments.next())?;
                    let receiver = self.python_handle(&reference)?;
                    assigned_receiver = Some(reference);
                    receiver
                },
                index: python_indices(arguments.next())?,
                value: self.argument_to_python(
                    context,
                    arguments
                        .next()
                        .ok_or_else(|| invalid_call("Python item assignment requires a value"))?,
                )?,
            },
            "iterate" => PythonCall::Iterate {
                receiver: self.python_handle(&foreign_argument(arguments.next())?)?,
            },
            "pyrun" => {
                let code = string_argument(arguments.next(), "Python code")?;
                let outputs = output_names(arguments.next())?;
                PythonCall::ExecutePersistent {
                    code,
                    inputs: self.named_inputs(context, arguments.collect())?,
                    outputs,
                }
            }
            "pyrunfile" => {
                let path = string_argument(arguments.next(), "Python file")?.into();
                let outputs = output_names(arguments.next())?;
                let remaining = arguments.collect::<Vec<_>>();
                let (script_arguments, inputs) = split_file_inputs(remaining)?;
                PythonCall::ExecuteFile {
                    path,
                    arguments: script_arguments,
                    inputs: self.named_inputs(context, inputs)?,
                    outputs,
                }
            }
            operation => {
                return Err(invalid_call(format!(
                    "unknown Python adapter operation {operation}"
                )));
            }
        };
        let cancellation = context.cancellation();
        let values = session
            .invoke_with_callback_and_cancellation(python_call, Some(&cancellation), |invocation| {
                self.invoke_callback(&session, invocation)
            })
            .map_err(python_runtime_error)?;
        if let Some(reference) = assigned_receiver {
            return Ok(Value::Foreign(reference));
        }
        let values = values
            .into_iter()
            .map(|value| self.result_from_python(&session, value))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(match values.len() {
            0 => Value::OutputList(Vec::new()),
            1 if call.requested_outputs <= 1 => values.into_iter().next().expect("one value"),
            _ => Value::OutputList(values),
        })
    }

    fn argument_to_python(
        &self,
        context: &RuntimeContext,
        value: Value,
    ) -> Result<PythonValue, RuntimeError> {
        if super::super::is_callable(&value) {
            let foreign = context
                .service_ports()
                .foreign()
                .map(Rc::downgrade)
                .ok_or_else(|| invalid_call("Python callback requires a foreign runtime"))?;
            let callback = self.next_callback.get();
            self.next_callback.set(
                callback
                    .checked_add(1)
                    .ok_or_else(|| invalid_call("Python callback identity space is exhausted"))?,
            );
            self.callbacks.borrow_mut().insert(
                callback,
                PythonCallbackRegistration {
                    callback: value,
                    context: context
                        .clone()
                        .with_service_ports(context.service_ports().clone().without_foreign()),
                    foreign,
                },
            );
            return Ok(PythonValue::Callback(callback));
        }
        match value {
            Value::Foreign(reference) if reference.type_identity.family == PYTHON_ADAPTER_ID => {
                self.python_handle(&reference).map(PythonValue::Object)
            }
            value => value_to_python(value),
        }
    }

    fn result_from_python(
        &self,
        session: &PythonSession,
        value: PythonValue,
    ) -> Result<Value, RuntimeError> {
        let PythonValue::Object(handle) = value else {
            return value_from_python(value);
        };
        if let Some(reference) = self
            .python_to_foreign
            .borrow()
            .get(&handle)
            .and_then(WeakForeignRef::upgrade)
        {
            return Ok(Value::Foreign(reference));
        }
        let metadata = session.metadata(handle).map_err(python_runtime_error)?;
        let qualified_type = if metadata.module == "builtins" {
            metadata.type_name
        } else {
            format!("{}.{}", metadata.module, metadata.type_name)
        };
        let reference = self.handles.register_resource(
            &self.host_identity,
            ForeignResourceMetadata {
                type_identity: ForeignTypeIdentity {
                    family: PYTHON_ADAPTER_ID.into(),
                    name: qualified_type,
                    version: PYTHON_ADAPTER_VERSION,
                },
                ownership: ForeignOwnership::Shared,
                affinity: ForeignAffinity::OriginProcess,
                lifetime: ForeignLifetime::Session,
            },
        )?;
        self.resources.borrow_mut().insert(reference.handle, handle);
        if let Some(weak) = reference.downgrade() {
            self.python_to_foreign.borrow_mut().insert(handle, weak);
        }
        Ok(Value::Foreign(reference))
    }

    fn python_handle(&self, reference: &ForeignRef) -> Result<PythonObjectHandle, RuntimeError> {
        let resolved = self.handles.resolve(reference, ForeignCapability::Invoke)?;
        self.resources
            .borrow()
            .get(&resolved.key.handle)
            .copied()
            .ok_or_else(|| {
                foreign_error(
                    ForeignErrorKind::StaleHandle,
                    format!("Python object handle {} is not live", resolved.key.handle),
                )
            })
    }

    fn named_inputs(
        &self,
        context: &RuntimeContext,
        arguments: Vec<Value>,
    ) -> Result<Vec<(String, PythonValue)>, RuntimeError> {
        if !arguments.len().is_multiple_of(2) {
            return Err(invalid_call("Python named inputs require name-value pairs"));
        }
        let mut inputs = Vec::with_capacity(arguments.len() / 2);
        let mut arguments = arguments.into_iter();
        while let Some(name) = arguments.next() {
            let name = string_argument(Some(name), "Python input name")?;
            let value = arguments.next().expect("even argument count");
            inputs.push((name, self.argument_to_python(context, value)?));
        }
        Ok(inputs)
    }

    fn invoke_callback(
        &self,
        session: &PythonSession,
        invocation: PythonCallbackInvocation,
    ) -> Result<PythonValue, PythonError> {
        let registration = self
            .callbacks
            .borrow()
            .get(&invocation.callback)
            .cloned()
            .ok_or_else(|| {
                PythonError::host(
                    "PythonCallbackError",
                    "Python callback identity is stale or belongs to another session",
                )
            })?;
        let foreign = registration.foreign.upgrade().ok_or_else(|| {
            PythonError::host(
                "PythonCallbackError",
                "callback's originating foreign runtime has ended",
            )
        })?;
        let context = registration.context.clone().with_service_ports(
            registration
                .context
                .service_ports()
                .clone()
                .with_foreign(foreign),
        );
        let arguments = invocation
            .arguments
            .into_iter()
            .map(|value| self.result_from_python(session, value))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| PythonError::host("PythonCallbackError", error.to_string()))?;
        let request = foreign_callback_request(&registration.callback, arguments, 1)
            .map_err(|error| PythonError::host("PythonCallbackError", error.to_string()))?;
        let value = pollster::block_on(invoke_foreign_callback(context.clone(), request))
            .map_err(|error| PythonError::host("PythonCallbackError", error.to_string()))?;
        self.argument_to_python(&context, value)
            .map_err(|error| PythonError::host("PythonCallbackError", error.to_string()))
    }

    fn status_value(&self) -> Result<Value, RuntimeError> {
        let configuration = self.configuration.borrow().clone();
        let installation = discover_python(&discovery_request(&configuration)).ok();
        let running = self.session.borrow().is_some();
        let mut environment = ObjectInstance::new("py.PythonEnvironment".into());
        environment.properties.insert(
            "Version".into(),
            Value::String(
                installation
                    .as_ref()
                    .map(|value| value.version.to_string())
                    .unwrap_or_default(),
            ),
        );
        environment.properties.insert(
            "Executable".into(),
            Value::String(
                installation
                    .as_ref()
                    .map(|value| value.executable.display().to_string())
                    .unwrap_or_default(),
            ),
        );
        environment.properties.insert(
            "Library".into(),
            Value::String(
                installation
                    .as_ref()
                    .map(|value| value.library.display().to_string())
                    .unwrap_or_default(),
            ),
        );
        environment.properties.insert(
            "Home".into(),
            Value::String(
                installation
                    .as_ref()
                    .map(|value| value.home.display().to_string())
                    .unwrap_or_default(),
            ),
        );
        environment.properties.insert(
            "Status".into(),
            Value::String(
                if running {
                    "Loaded"
                } else if self.terminated.get() {
                    "Terminated"
                } else {
                    "NotLoaded"
                }
                .into(),
            ),
        );
        environment.properties.insert(
            "ExecutionMode".into(),
            Value::String(configuration.execution_mode.as_compatibility_name().into()),
        );
        environment.properties.insert(
            "ProcessID".into(),
            if running {
                Value::Int(runmat_value::IntValue::U32(std::process::id()))
            } else {
                Value::Num(f64::NAN)
            },
        );
        environment.properties.insert(
            "ProcessName".into(),
            Value::String(if running { "RunMat" } else { "" }.into()),
        );
        Ok(Value::Object(environment))
    }

    fn configure_value(&self, arguments: Vec<Value>) -> Result<Value, RuntimeError> {
        if self.session.borrow().is_some() {
            return Err(invalid_call(
                "pyenv cannot change Python configuration after startup",
            ));
        }
        if arguments.is_empty() || !arguments.len().is_multiple_of(2) {
            return Err(invalid_call("pyenv requires name-value pairs"));
        }
        let mut configuration = self.configuration.borrow().clone();
        let mut arguments = arguments.into_iter();
        while let Some(name) = arguments.next() {
            let name = string_argument(Some(name), "pyenv option")?.to_ascii_lowercase();
            let value = string_argument(arguments.next(), "pyenv option value")?;
            match name.as_str() {
                "version" => {
                    let path = std::path::PathBuf::from(&value);
                    if path.is_file() || value.contains(std::path::MAIN_SEPARATOR) {
                        configuration.executable = Some(path);
                        configuration.version = None;
                    } else {
                        configuration.version = Some(parse_version_pair(&value)?);
                        configuration.executable = None;
                    }
                }
                "executionmode" => {
                    configuration.execution_mode = match value.to_ascii_lowercase().as_str() {
                        "inprocess" => PythonExecutionMode::InProcess,
                        "outofprocess" => PythonExecutionMode::OutOfProcess,
                        _ => {
                            return Err(invalid_call(
                                "Python ExecutionMode must be 'InProcess' or 'OutOfProcess'",
                            ));
                        }
                    };
                }
                _ => {
                    return Err(invalid_call(
                        "pyenv supports the 'Version' and 'ExecutionMode' options",
                    ));
                }
            }
        }
        discover_python(&discovery_request(&configuration)).map_err(python_runtime_error)?;
        self.configure(configuration)?;
        self.status_value()
    }

    fn terminate_value(&self) -> Result<Value, RuntimeError> {
        if self.configuration.borrow().execution_mode != PythonExecutionMode::OutOfProcess {
            return Err(invalid_call(
                "terminate is available only for an OutOfProcess Python environment",
            ));
        }
        self.handles.restart_host(&self.host_identity)?;
        self.session.borrow_mut().take();
        self.resources.borrow_mut().clear();
        self.python_to_foreign.borrow_mut().clear();
        if let Ok(mut released) = self.released.0.lock() {
            released.clear();
        }
        self.terminated.set(true);
        self.status_value()
    }
}

impl ForeignAdapter for PythonAdapter {
    fn descriptor(&self) -> ForeignAdapterDescriptor {
        ForeignAdapterDescriptor {
            adapter: PYTHON_ADAPTER_ID.into(),
            version: PYTHON_ADAPTER_VERSION,
            capabilities: BTreeSet::from([CapabilityRequirement::ForeignRuntime]),
            foreign_capabilities: BTreeSet::from([
                ForeignCapability::Invoke,
                ForeignCapability::Read,
                ForeignCapability::Write,
                ForeignCapability::Callback,
                ForeignCapability::Transfer,
                ForeignCapability::ZeroCopy,
            ]),
            artifact_identities: BTreeSet::new(),
            supports_wasm: false,
            supports_host_bridge: false,
            execution_stack: runmat_types::ExecutionStackRequirement::Process,
        }
    }

    fn invoke(&self, context: RuntimeContext, call: ForeignCall) -> ForeignAdapterFuture {
        let result = self.invoke_now(&context, call);
        Box::pin(async move { result })
    }

    fn is_isolated(&self) -> bool {
        self.configuration.borrow().execution_mode == PythonExecutionMode::OutOfProcess
    }
}

fn discovery_request(configuration: &PythonRuntimeConfiguration) -> PythonDiscoveryRequest {
    PythonDiscoveryRequest {
        executable: configuration.executable.clone(),
        version: configuration.version,
        minimum_version: configuration.minimum_version,
        maximum_version: configuration.maximum_version,
    }
}

fn policy(mode: PythonExecutionMode) -> ForeignExecutionPolicy {
    match mode {
        PythonExecutionMode::InProcess => ForeignExecutionPolicy::trusted_in_process(),
        PythonExecutionMode::OutOfProcess => ForeignExecutionPolicy {
            trust: super::super::ForeignTrust::Trusted,
            isolation: super::super::ForeignIsolation::IsolatedProcess,
        },
    }
}

fn foreign_argument(value: Option<Value>) -> Result<ForeignRef, RuntimeError> {
    match value {
        Some(Value::Foreign(reference)) if reference.type_identity.family == PYTHON_ADAPTER_ID => {
            Ok(reference)
        }
        _ => Err(invalid_call("Python receiver must be a Python object")),
    }
}

fn string_argument(value: Option<Value>, label: &str) -> Result<String, RuntimeError> {
    match value {
        Some(value) => String::try_from(&value)
            .ok()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| invalid_call(format!("{label} must be non-empty text"))),
        None => Err(invalid_call(format!("{label} is required"))),
    }
}

fn output_names(value: Option<Value>) -> Result<Vec<String>, RuntimeError> {
    match value {
        None => Ok(Vec::new()),
        Some(Value::String(value)) => Ok(vec![value]),
        Some(Value::CharArray(value)) => value
            .row_string()
            .map(|value| vec![value])
            .ok_or_else(|| invalid_call("Python outputs must be text")),
        Some(Value::StringArray(value)) => Ok(value.data),
        Some(Value::Cell(value)) => value
            .data
            .iter()
            .map(|value| {
                String::try_from(value)
                    .map_err(|_| invalid_call("Python output names must be text"))
            })
            .collect(),
        Some(_) => Err(invalid_call("Python outputs must be text")),
    }
}

fn split_file_inputs(arguments: Vec<Value>) -> Result<(Vec<String>, Vec<Value>), RuntimeError> {
    if arguments.len() >= 2 {
        if let Some(position) = arguments.iter().position(|value| {
            String::try_from(value).is_ok_and(|value| value.eq_ignore_ascii_case("Arguments"))
        }) {
            let Some(Value::Cell(values)) = arguments.get(position + 1) else {
                return Err(invalid_call(
                    "pyrunfile Arguments must be a cell array of text",
                ));
            };
            let script_arguments = values
                .data
                .iter()
                .map(|value| {
                    String::try_from(value)
                        .map_err(|_| invalid_call("pyrunfile Arguments must contain text"))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut inputs = arguments;
            inputs.drain(position..=position + 1);
            return Ok((script_arguments, inputs));
        }
    }
    Ok((Vec::new(), arguments))
}

fn python_index(value: Option<Value>) -> Result<PythonValue, RuntimeError> {
    let value = value.ok_or_else(|| invalid_call("Python index is required"))?;
    let one_based = match value {
        Value::Int(value) => value.to_f64(),
        Value::Num(value) => value,
        _ => return value_to_python(value),
    };
    if !one_based.is_finite() || one_based.fract() != 0.0 || one_based < 1.0 {
        return Err(invalid_call(
            "Python positional indices must be positive integers",
        ));
    }
    Ok(PythonValue::Unsigned(one_based as u64 - 1))
}

fn python_indices(value: Option<Value>) -> Result<PythonValue, RuntimeError> {
    let values = match value {
        Some(Value::OutputList(values)) => values,
        value => return python_index(value),
    };
    if values.is_empty() {
        return Err(invalid_call("Python indexing requires at least one index"));
    }
    let mut indices = values
        .into_iter()
        .map(|value| python_index(Some(value)))
        .collect::<Result<Vec<_>, _>>()?;
    if indices.len() == 1 {
        Ok(indices.pop().expect("one index"))
    } else {
        Ok(PythonValue::Tuple(indices))
    }
}

fn parse_version_pair(value: &str) -> Result<(u16, u16), RuntimeError> {
    let mut parts = value.trim_start_matches("Python ").split('.');
    let major = parts
        .next()
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| invalid_call("Python Version must identify a major and minor release"))?;
    let minor = parts
        .next()
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| invalid_call("Python Version must identify a major and minor release"))?;
    Ok((major, minor))
}

fn python_runtime_error(error: PythonError) -> RuntimeError {
    let type_name = error
        .type_name
        .chars()
        .filter(|character| character.is_ascii_alphanumeric() || *character == '_')
        .collect::<String>();
    let message = if error.formatted_traceback.is_empty() {
        format!("{}: {}", error.type_name, error.message)
    } else {
        format!(
            "{}: {}\n{}",
            error.type_name, error.message, error.formatted_traceback
        )
    };
    build_runtime_error(message)
        .with_builtin("python")
        .with_identifier(format!("RunMat:Python:{type_name}"))
        .with_source(error)
        .build()
}

fn invalid_call(message: impl Into<String>) -> RuntimeError {
    build_runtime_error(message)
        .with_builtin("python")
        .with_identifier("RunMat:Foreign:InvalidCall")
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::RuntimeExecutionService;
    use crate::foreign::{ForeignPlatform, ForeignRuntime};

    fn context(foreign: Rc<ForeignRuntime>) -> RuntimeContext {
        RuntimeContext::new(Rc::new(RuntimeExecutionService::new())).with_service_ports(
            crate::context::RuntimeServicePorts::default().with_foreign(foreign),
        )
    }

    fn adapter() -> Option<(Rc<ForeignRuntime>, Rc<PythonAdapter>)> {
        if discover_python(&PythonDiscoveryRequest::default()).is_err() {
            return None;
        }
        let foreign = Rc::new(ForeignRuntime::new(ForeignPlatform::Native));
        let adapter = PythonAdapter::new(foreign.handles().clone()).expect("create adapter");
        foreign
            .register_adapter(adapter.clone())
            .expect("register adapter");
        Some((foreign, adapter))
    }

    #[test]
    fn qualified_calls_and_object_members_use_one_identity_registry() {
        let Some((foreign, adapter)) = adapter() else {
            return;
        };
        let context = context(foreign);
        let object = adapter
            .invoke_now(
                &context,
                ForeignCall {
                    adapter: PYTHON_ADAPTER_ID.into(),
                    symbol: "invoke_qualified".into(),
                    arguments: vec![Value::String("py.types.SimpleNamespace".into())],
                    requested_outputs: 1,
                },
            )
            .expect("construct Python object");
        let Value::Foreign(reference) = object else {
            panic!("expected Python foreign reference");
        };
        adapter
            .invoke_now(
                &context,
                ForeignCall {
                    adapter: PYTHON_ADAPTER_ID.into(),
                    symbol: "set_member".into(),
                    arguments: vec![
                        Value::Foreign(reference.clone()),
                        Value::String("answer".into()),
                        Value::Int(runmat_value::IntValue::I32(42)),
                    ],
                    requested_outputs: 0,
                },
            )
            .expect("set Python attribute");
        let answer = adapter
            .invoke_now(
                &context,
                ForeignCall {
                    adapter: PYTHON_ADAPTER_ID.into(),
                    symbol: "get_member".into(),
                    arguments: vec![Value::Foreign(reference), Value::String("answer".into())],
                    requested_outputs: 1,
                },
            )
            .expect("get Python attribute");
        assert_eq!(answer, Value::Int(runmat_value::IntValue::I64(42)));
    }

    #[test]
    fn environment_status_does_not_start_python() {
        let Some((_foreign, adapter)) = adapter() else {
            return;
        };
        let Value::Object(environment) = adapter.status_value().expect("read status") else {
            panic!("expected PythonEnvironment");
        };
        assert_eq!(environment.class_name, "py.PythonEnvironment");
        assert_eq!(
            environment.properties.get("Status"),
            Some(&Value::String("NotLoaded".into()))
        );
        assert!(!adapter.is_running());
    }
}
