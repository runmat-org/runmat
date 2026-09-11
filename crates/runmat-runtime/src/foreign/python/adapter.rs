use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use runmat_python::{
    discover_python, InstalledPythonArtifacts, PythonArtifactBundle, PythonCall,
    PythonCallbackInvocation, PythonDiscoveryRequest, PythonError, PythonExecutionMode,
    PythonObjectHandle, PythonSession, PythonSessionConfig, PythonValue, PYTHON_ADAPTER_ID,
    PYTHON_ADAPTER_VERSION,
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
use super::{IsolatedPythonClient, NestedPythonCall, NestedPythonQueue};
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

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PythonRuntimeConfiguration {
    pub executable: Option<std::path::PathBuf>,
    pub version: Option<(u16, u16)>,
    pub minimum_version: (u16, u16),
    pub maximum_version: Option<(u16, u16)>,
    #[serde(default)]
    pub exact_version: Option<runmat_python::PythonVersion>,
    #[serde(default)]
    pub required_abi_tag: Option<String>,
    #[serde(default)]
    pub required_platform_tag: Option<String>,
    pub execution_mode: PythonExecutionMode,
    #[serde(default)]
    pub module_paths: Vec<std::path::PathBuf>,
    #[serde(default)]
    pub artifact_identities: BTreeSet<String>,
}

impl Default for PythonRuntimeConfiguration {
    fn default() -> Self {
        Self {
            executable: None,
            version: None,
            minimum_version: (3, 9),
            maximum_version: None,
            exact_version: None,
            required_abi_tag: None,
            required_platform_tag: None,
            execution_mode: PythonExecutionMode::InProcess,
            module_paths: Vec::new(),
            artifact_identities: BTreeSet::new(),
        }
    }
}

pub struct PythonAdapter {
    handles: ForeignHandleRegistry,
    host_identity: String,
    session: RefCell<Option<PythonSession>>,
    isolated_client: Rc<RefCell<Option<IsolatedPythonClient>>>,
    isolated_call_active: Rc<Cell<bool>>,
    nested_isolated_calls: NestedPythonQueue,
    configuration: Rc<RefCell<PythonRuntimeConfiguration>>,
    released: Arc<ReleaseQueue>,
    resources: RefCell<BTreeMap<u64, PythonObjectHandle>>,
    python_to_foreign: RefCell<BTreeMap<PythonObjectHandle, WeakForeignRef>>,
    callbacks: RefCell<BTreeMap<u64, PythonCallbackRegistration>>,
    next_callback: Cell<u64>,
    terminated: Rc<Cell<bool>>,
    installed_artifacts: RefCell<Option<InstalledPythonArtifacts>>,
}

#[derive(Clone)]
struct IsolatedPythonState {
    handles: ForeignHandleRegistry,
    host_identity: String,
    client: Rc<RefCell<Option<IsolatedPythonClient>>>,
    call_active: Rc<Cell<bool>>,
    nested_calls: NestedPythonQueue,
    configuration: Rc<RefCell<PythonRuntimeConfiguration>>,
    released: Arc<ReleaseQueue>,
    terminated: Rc<Cell<bool>>,
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
            adapter: PYTHON_ADAPTER_ID.to_string(),
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
            isolated_client: Rc::new(RefCell::new(None)),
            isolated_call_active: Rc::new(Cell::new(false)),
            nested_isolated_calls: Rc::new(RefCell::new(std::collections::VecDeque::new())),
            configuration: Rc::new(RefCell::new(configuration)),
            released,
            resources: RefCell::new(BTreeMap::new()),
            python_to_foreign: RefCell::new(BTreeMap::new()),
            callbacks: RefCell::new(BTreeMap::new()),
            next_callback: Cell::new(1),
            terminated: Rc::new(Cell::new(false)),
            installed_artifacts: RefCell::new(None),
        }))
    }

    pub fn configure(&self, configuration: PythonRuntimeConfiguration) -> Result<(), RuntimeError> {
        if self.is_running() || self.isolated_call_active.get() {
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
        self.session.borrow().is_some() || self.isolated_client.borrow().is_some()
    }

    pub fn install_artifact_bundle(
        &self,
        bundle: &PythonArtifactBundle,
    ) -> Result<(), RuntimeError> {
        if self.is_running() || self.isolated_call_active.get() {
            return Err(invalid_call(
                "Python artifacts cannot change after the interpreter starts",
            ));
        }
        bundle.validate().map_err(python_artifact_error)?;
        let configuration = self.configuration.borrow().clone();
        let installation =
            discover_python(&discovery_request(&configuration)).map_err(python_runtime_error)?;
        if let Some(required) = &bundle.environment {
            if required.implementation != "cpython"
                || required.version != installation.version
                || required.abi_tag != installation.abi_tag
                || required.platform_tag != installation.platform_tag
                || required.execution_mode != configuration.execution_mode
            {
                return Err(invalid_call(format!(
                    "Python artifact environment requires {} {} ({}, {}) in {:?} mode, but this session resolved {} {} ({}, {}) in {:?} mode",
                    required.implementation,
                    required.version,
                    required.abi_tag,
                    required.platform_tag,
                    required.execution_mode,
                    installation.implementation,
                    installation.version,
                    installation.abi_tag,
                    installation.platform_tag,
                    configuration.execution_mode
                )));
            }
        }
        let installed = bundle.install().map_err(python_artifact_error)?;
        let mut updated = configuration;
        updated.module_paths = installed.module_paths().to_vec();
        updated.artifact_identities = bundle.artifact_identities();
        *self.configuration.borrow_mut() = updated;
        *self.installed_artifacts.borrow_mut() = Some(installed);
        Ok(())
    }

    pub fn install_portable_artifact_bundle(
        &self,
        bundle: &PythonArtifactBundle,
    ) -> Result<(), RuntimeError> {
        if let Some(environment) = bundle.environment.as_ref() {
            let mut configuration = self.configuration.borrow().clone();
            configuration.version = Some((environment.version.major, environment.version.minor));
            configuration.minimum_version = (environment.version.major, environment.version.minor);
            configuration.maximum_version =
                Some((environment.version.major, environment.version.minor));
            configuration.execution_mode = environment.execution_mode;
            configuration.exact_version = Some(environment.version);
            configuration.required_abi_tag = Some(environment.abi_tag.clone());
            configuration.required_platform_tag = Some(environment.platform_tag.clone());
            configuration.module_paths.clear();
            configuration.artifact_identities.clear();
            self.configure(configuration)?;
        }
        self.install_artifact_bundle(bundle)
    }

    pub fn clear_artifact_bundle(&self) -> Result<(), RuntimeError> {
        if self.is_running() || self.isolated_call_active.get() {
            return Err(invalid_call(
                "Python artifacts cannot be cleared after the interpreter starts",
            ));
        }
        let mut configuration = self.configuration.borrow_mut();
        configuration.module_paths.clear();
        configuration.artifact_identities.clear();
        drop(configuration);
        self.installed_artifacts.borrow_mut().take();
        Ok(())
    }

    fn ensure_session(&self) -> Result<PythonSession, RuntimeError> {
        if let Some(session) = self.session.borrow().clone() {
            return Ok(session);
        }
        let configuration = self.configuration.borrow().clone();
        debug_assert_eq!(configuration.execution_mode, PythonExecutionMode::InProcess);
        let installation =
            discover_python(&discovery_request(&configuration)).map_err(python_runtime_error)?;
        let session = PythonSession::start(PythonSessionConfig {
            installation,
            module_paths: configuration.module_paths,
        })
        .map_err(python_runtime_error)?;
        *self.session.borrow_mut() = Some(session.clone());
        self.terminated.set(false);
        Ok(session)
    }

    fn drain_releases(&self) -> Result<(), RuntimeError> {
        let handles = self.take_releases()?;
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

    fn take_releases(&self) -> Result<Vec<u64>, RuntimeError> {
        let mut released = self.released.0.lock().map_err(|_| {
            foreign_error(
                ForeignErrorKind::HostUnavailable,
                "Python release queue lock was poisoned",
            )
        })?;
        Ok(std::mem::take(&mut *released))
    }

    fn invoke_now(
        &self,
        context: &RuntimeContext,
        call: ForeignCall,
    ) -> Result<runmat_value::ValueSequence, RuntimeError> {
        match call.symbol.as_str() {
            "status" => {
                return self
                    .status_value()
                    .and_then(super::super::single_output_sequence)
            }
            "configure" => {
                return self
                    .configure_value(call.arguments)
                    .and_then(super::super::single_output_sequence)
            }
            "terminate" => {
                return self
                    .terminate_value()
                    .and_then(super::super::single_output_sequence)
            }
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
                index: python_indices(arguments.collect())?,
            },
            "set_item" => {
                let reference = foreign_argument(arguments.next())?;
                let receiver = self.python_handle(&reference)?;
                let (indices, value) = split_item_assignment_arguments(arguments.collect())?;
                assigned_receiver = Some(reference);
                PythonCall::SetItem {
                    receiver,
                    index: python_indices(indices)?,
                    value: self.argument_to_python(context, value)?,
                }
            }
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
            return super::super::single_output_sequence(Value::Foreign(reference));
        }
        let values = values
            .into_iter()
            .map(|value| self.result_from_python(&session, value))
            .collect::<Result<Vec<_>, _>>()?;
        if values.len() == 1 && call.requested_outputs <= 1 {
            return super::super::single_output_sequence(
                values.into_iter().next().expect("one Python output"),
            );
        }
        runmat_value::ValueSequence::comma_separated(values)
            .map_err(crate::sequence::sequence_error_to_runtime)
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
        let sequence = pollster::block_on(invoke_foreign_callback(context.clone(), request))
            .map_err(|error| PythonError::host("PythonCallbackError", error.to_string()))?;
        let value = crate::sequence::ResolveValueSequence::resolve(
            sequence,
            runmat_types::SequenceUse::SelectPrefix { count: 1 },
            crate::sequence::SequenceResolutionContext::default(),
        )
        .map_err(|error| PythonError::host("PythonCallbackError", error.to_string()))?
        .into_iter()
        .next()
        .expect("one Python callback output was selected");
        self.argument_to_python(&context, value)
            .map_err(|error| PythonError::host("PythonCallbackError", error.to_string()))
    }

    fn status_value(&self) -> Result<Value, RuntimeError> {
        let configuration = self.configuration.borrow().clone();
        let running = self.is_running();
        let process_id = self
            .isolated_client
            .borrow()
            .as_ref()
            .and_then(IsolatedPythonClient::process_id)
            .or_else(|| running.then(std::process::id));
        Ok(python_status_value(
            &configuration,
            running,
            self.terminated.get(),
            process_id,
        ))
    }

    fn configure_value(&self, arguments: Vec<Value>) -> Result<Value, RuntimeError> {
        if self.is_running() || self.isolated_call_active.get() {
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

    fn isolated_state(&self) -> IsolatedPythonState {
        IsolatedPythonState {
            handles: self.handles.clone(),
            host_identity: self.host_identity.clone(),
            client: Rc::clone(&self.isolated_client),
            call_active: Rc::clone(&self.isolated_call_active),
            nested_calls: Rc::clone(&self.nested_isolated_calls),
            configuration: Rc::clone(&self.configuration),
            released: Arc::clone(&self.released),
            terminated: Rc::clone(&self.terminated),
        }
    }
}

fn python_status_value(
    configuration: &PythonRuntimeConfiguration,
    running: bool,
    terminated: bool,
    process_id: Option<u32>,
) -> Value {
    let installation = discover_python(&discovery_request(configuration)).ok();
    let mut environment = ObjectInstance::new("py.PythonEnvironment");
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
            } else if terminated {
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
        if let Some(process_id) = process_id {
            Value::Int(runmat_value::IntValue::U32(process_id))
        } else {
            Value::Num(f64::NAN)
        },
    );
    environment.properties.insert(
        "ProcessName".into(),
        Value::String(if running { "RunMat" } else { "" }.into()),
    );
    Value::Object(environment)
}

async fn invoke_isolated(
    state: IsolatedPythonState,
    context: RuntimeContext,
    call: ForeignCall,
) -> Result<runmat_value::ValueSequence, RuntimeError> {
    if state.call_active.replace(true) {
        let requested_outputs = call.requested_outputs;
        let (reply, response) = tokio::sync::oneshot::channel();
        state
            .nested_calls
            .borrow_mut()
            .push_back(NestedPythonCall { call, reply });
        let outputs = response.await.map_err(|_| {
            foreign_error(
                ForeignErrorKind::HostUnavailable,
                "isolated Python callback ended before its nested call completed",
            )
        })??;
        return isolated_outputs(outputs, requested_outputs);
    }
    let result = async {
        let mut client = state.client.borrow_mut().take();
        if client.is_none() {
            let configuration = state.configuration.borrow().clone();
            client = Some(
                IsolatedPythonClient::spawn(&configuration)
                    .await
                    .map_err(super::isolation::runtime_error)?,
            );
            state.terminated.set(false);
        }
        let mut client = client.expect("isolated Python client initialized");
        let released = take_releases(&state.released)?;
        let requested_outputs = call.requested_outputs;
        let outcome = client
            .invoke(
                context,
                call,
                &state.handles,
                &state.host_identity,
                released,
                &state.nested_calls,
            )
            .await
            .and_then(|outputs| isolated_outputs(outputs, requested_outputs));
        let terminal = outcome.as_ref().err().is_some_and(|error| {
            matches!(
                error.identifier(),
                Some(
                    "RunMat:Python:Cancelled"
                        | "RunMat:Python:HostCrashed"
                        | "RunMat:Python:HostTransport"
                        | "RunMat:Python:HostAuthentication"
                )
            )
        });
        if terminal {
            state.handles.restart_host(&state.host_identity)?;
            state.terminated.set(true);
        } else {
            *state.client.borrow_mut() = Some(client);
        }
        outcome
    }
    .await;
    state.call_active.set(false);
    result
}

fn isolated_outputs(
    mut outputs: Vec<Value>,
    requested_outputs: usize,
) -> Result<runmat_value::ValueSequence, RuntimeError> {
    match (requested_outputs, outputs.len()) {
        (1, 1) => {
            runmat_value::ValueSequence::single(outputs.pop().expect("one isolated Python output"))
                .map_err(|error| invalid_call(error.to_string()))
        }
        _ => runmat_value::ValueSequence::comma_separated(outputs)
            .map_err(|error| invalid_call(error.to_string())),
    }
}

async fn terminate_isolated(state: IsolatedPythonState) -> Result<Value, RuntimeError> {
    if state.call_active.replace(true) {
        return Err(invalid_call(
            "terminate cannot run during an active isolated Python call",
        ));
    }
    let result = async {
        let client = state.client.borrow_mut().take();
        let shutdown = if let Some(mut client) = client {
            let released = take_releases(&state.released)?;
            client
                .shutdown(released)
                .await
                .map_err(super::isolation::runtime_error)
        } else {
            Ok(())
        };
        state.handles.restart_host(&state.host_identity)?;
        state.nested_calls.borrow_mut().clear();
        state.terminated.set(true);
        shutdown?;
        let configuration = state.configuration.borrow().clone();
        Ok(python_status_value(&configuration, false, true, None))
    }
    .await;
    state.call_active.set(false);
    result
}

fn take_releases(released: &ReleaseQueue) -> Result<Vec<u64>, RuntimeError> {
    let mut released = released.0.lock().map_err(|_| {
        foreign_error(
            ForeignErrorKind::HostUnavailable,
            "Python release queue lock was poisoned",
        )
    })?;
    Ok(std::mem::take(&mut *released))
}

impl ForeignAdapter for PythonAdapter {
    fn descriptor(&self) -> ForeignAdapterDescriptor {
        ForeignAdapterDescriptor {
            adapter: runmat_types::ForeignAdapterId::new(PYTHON_ADAPTER_ID)
                .expect("the built-in Python adapter identity is valid"),
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
            artifact_identities: self
                .configuration
                .borrow()
                .artifact_identities
                .iter()
                .map(|identity| {
                    runmat_types::ForeignArtifactIdentity::new(identity.clone())
                        .expect("installed Python artifact identities are canonical")
                })
                .collect(),
            supports_wasm: false,
            supports_host_bridge: false,
            execution_stack: runmat_types::ExecutionStackRequirement::Process,
        }
    }

    fn invoke(&self, context: RuntimeContext, call: ForeignCall) -> ForeignAdapterFuture {
        let mode = self.configuration.borrow().execution_mode;
        if mode == PythonExecutionMode::InProcess
            || matches!(call.symbol.as_str(), "status" | "configure")
        {
            let result = self.invoke_now(&context, call);
            return Box::pin(async move { result });
        }
        if call.symbol == "terminate" {
            let state = self.isolated_state();
            return Box::pin(async move {
                terminate_isolated(state)
                    .await
                    .and_then(super::super::single_output_sequence)
            });
        }
        Box::pin(invoke_isolated(self.isolated_state(), context, call))
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
        exact_version: configuration.exact_version,
        required_abi_tag: configuration.required_abi_tag.clone(),
        required_platform_tag: configuration.required_platform_tag.clone(),
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

fn python_indices(values: Vec<Value>) -> Result<PythonValue, RuntimeError> {
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

fn split_item_assignment_arguments(
    mut arguments: Vec<Value>,
) -> Result<(Vec<Value>, Value), RuntimeError> {
    let value = arguments
        .pop()
        .ok_or_else(|| invalid_call("Python item assignment requires a value"))?;
    if arguments.is_empty() {
        return Err(invalid_call("Python indexing requires at least one index"));
    }
    Ok((arguments, value))
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

fn python_artifact_error(error: runmat_python::PythonArtifactError) -> RuntimeError {
    foreign_error(ForeignErrorKind::InvalidManifest, error.to_string())
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

    fn only_output(sequence: runmat_value::ValueSequence) -> Value {
        let [value] = sequence.into_values().try_into().expect("one output");
        value
    }

    #[test]
    fn flattened_item_arguments_preserve_selector_and_value_boundaries() {
        assert!(matches!(
            python_indices(vec![Value::Num(2.0)]).expect("one index"),
            PythonValue::Unsigned(1)
        ));
        let PythonValue::Tuple(indices) =
            python_indices(vec![Value::Num(2.0), Value::Num(4.0)]).expect("two indices")
        else {
            panic!("multiple indices must remain a tuple");
        };
        assert!(matches!(
            indices.as_slice(),
            [PythonValue::Unsigned(1), PythonValue::Unsigned(3)]
        ));
        assert!(python_indices(Vec::new()).is_err());

        let (indices, assigned) = split_item_assignment_arguments(vec![
            Value::Num(2.0),
            Value::Num(4.0),
            Value::String("assigned".into()),
        ])
        .expect("assignment arguments");
        assert_eq!(indices, vec![Value::Num(2.0), Value::Num(4.0)]);
        assert_eq!(assigned, Value::String("assigned".into()));
        assert!(split_item_assignment_arguments(Vec::new()).is_err());
        assert!(split_item_assignment_arguments(vec![Value::Num(1.0)]).is_err());
    }

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
        let object = only_output(
            adapter
                .invoke_now(
                    &context,
                    ForeignCall {
                        adapter: PYTHON_ADAPTER_ID.into(),
                        symbol: "invoke_qualified".into(),
                        arguments: vec![Value::String("py.types.SimpleNamespace".into())],
                        requested_outputs: 1,
                    },
                )
                .expect("construct Python object"),
        );
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
        let answer = only_output(
            adapter
                .invoke_now(
                    &context,
                    ForeignCall {
                        adapter: PYTHON_ADAPTER_ID.into(),
                        symbol: "get_member".into(),
                        arguments: vec![Value::Foreign(reference), Value::String("answer".into())],
                        requested_outputs: 1,
                    },
                )
                .expect("get Python attribute"),
        );
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
        assert_eq!(
            environment.class_name.display_name(),
            "py.PythonEnvironment"
        );
        assert_eq!(
            environment.properties.get("Status"),
            Some(&Value::String("NotLoaded".into()))
        );
        assert!(!adapter.is_running());
    }
}
