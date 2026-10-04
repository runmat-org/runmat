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
use runmat_value::{ForeignRef, ForeignResourceKey, Value, WeakForeignRef};

mod environment;
mod isolated;
mod prepared;
#[cfg(test)]
use environment::environment_metadata;
use environment::{issued_python_status_value, validate_environment};
use isolated::{IsolatedCallGuard, NestedCallLease};
use prepared::{prepare_python_call, PreparedPythonCall};

use super::super::{
    foreign_callback_request, foreign_error, invoke_foreign_callback, ForeignAdapter,
    ForeignAdapterDescriptor, ForeignAdapterFuture, ForeignErrorKind, ForeignExecutionPolicy,
    ForeignHandleRegistry, ForeignHostRegistration, ForeignHostRelease, ForeignResourceMetadata,
};
use super::conversion::{value_from_python, value_to_python};
use super::{IsolatedPythonClient, NestedPythonCall, NestedPythonQueue};
use crate::builtins::common::validation::value_text_scalar_string;
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
            .reconfigure_host(&self.host_identity, policy(configuration.execution_mode))?;
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
        self.install_artifact_bundle_with_configuration(
            bundle,
            self.configuration.borrow().clone(),
            false,
        )
    }

    fn install_artifact_bundle_with_configuration(
        &self,
        bundle: &PythonArtifactBundle,
        configuration: PythonRuntimeConfiguration,
        reset_terminated: bool,
    ) -> Result<(), RuntimeError> {
        if self.is_running() || self.isolated_call_active.get() {
            return Err(invalid_call(
                "Python artifacts cannot change after the interpreter starts",
            ));
        }
        self.handles
            .admit_host_reconfiguration(&self.host_identity)?;
        bundle.validate().map_err(python_artifact_error)?;
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
        self.handles
            .reconfigure_host(&self.host_identity, policy(updated.execution_mode))?;
        *self.configuration.borrow_mut() = updated;
        *self.installed_artifacts.borrow_mut() = Some(installed);
        if reset_terminated {
            self.terminated.set(false);
        }
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
            return self.install_artifact_bundle_with_configuration(bundle, configuration, true);
        }
        self.install_artifact_bundle(bundle)
    }

    pub fn clear_artifact_bundle(&self) -> Result<(), RuntimeError> {
        if self.is_running() || self.isolated_call_active.get() {
            return Err(invalid_call(
                "Python artifacts cannot be cleared after the interpreter starts",
            ));
        }
        self.handles.revoke_host_controls(&self.host_identity)?;
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
                    .terminate_value(call.arguments)
                    .and_then(super::super::single_output_sequence)
            }
            _ => {}
        }
        let prepared = prepare_python_call(&call.symbol, call.arguments)?;
        let session = self.ensure_session()?;
        self.drain_releases()?;
        let mut assigned_receiver = None;
        let python_call = match prepared {
            PreparedPythonCall::InvokeQualified { name, arguments } => {
                PythonCall::InvokeQualified {
                    name,
                    arguments: arguments
                        .into_iter()
                        .map(|value| self.argument_to_python(context, value))
                        .collect::<Result<_, _>>()?,
                }
            }
            PreparedPythonCall::GetMember { receiver, name } => PythonCall::GetMember {
                receiver: self.python_handle(&receiver)?,
                name,
            },
            PreparedPythonCall::SetMember {
                receiver,
                name,
                value,
            } => {
                assigned_receiver = Some(receiver.clone());
                PythonCall::SetMember {
                    receiver: self.python_handle(&receiver)?,
                    name,
                    value: self.argument_to_python(context, value)?,
                }
            }
            PreparedPythonCall::InvokeMember {
                receiver,
                name,
                arguments,
            } => PythonCall::InvokeMember {
                receiver: self.python_handle(&receiver)?,
                name,
                arguments: arguments
                    .into_iter()
                    .map(|value| self.argument_to_python(context, value))
                    .collect::<Result<_, _>>()?,
            },
            PreparedPythonCall::GetItem { receiver, index } => PythonCall::GetItem {
                receiver: self.python_handle(&receiver)?,
                index,
            },
            PreparedPythonCall::SetItem {
                receiver,
                index,
                value,
            } => {
                let handle = self.python_handle(&receiver)?;
                assigned_receiver = Some(receiver);
                PythonCall::SetItem {
                    receiver: handle,
                    index,
                    value: self.argument_to_python(context, value)?,
                }
            }
            PreparedPythonCall::Iterate { receiver } => PythonCall::Iterate {
                receiver: self.python_handle(&receiver)?,
            },
            PreparedPythonCall::Run {
                code,
                outputs,
                inputs,
            } => PythonCall::ExecutePersistent {
                code,
                inputs: self.convert_named_inputs(context, inputs)?,
                outputs,
            },
            PreparedPythonCall::RunFile {
                path,
                arguments,
                outputs,
                inputs,
            } => PythonCall::ExecuteFile {
                path: path.into(),
                arguments,
                inputs: self.convert_named_inputs(context, inputs)?,
                outputs,
            },
        };
        let cancellation = context.cancellation();
        let values = session
            .invoke_with_callback_and_cancellation(
                python_call,
                Some(&|| cancellation.is_cancelled()),
                |invocation| self.invoke_callback(&session, invocation),
            )
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

    fn convert_named_inputs(
        &self,
        context: &RuntimeContext,
        inputs: Vec<(String, Value)>,
    ) -> Result<Vec<(String, PythonValue)>, RuntimeError> {
        inputs
            .into_iter()
            .map(|(name, value)| Ok((name, self.argument_to_python(context, value)?)))
            .collect()
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
        issued_python_status_value(
            &self.handles,
            &self.host_identity,
            &configuration,
            running,
            self.terminated.get(),
            process_id,
        )
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

    fn terminate_value(&self, arguments: Vec<Value>) -> Result<Value, RuntimeError> {
        validate_environment(&self.handles, &self.host_identity, &arguments)?;
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

async fn invoke_isolated(
    state: IsolatedPythonState,
    context: RuntimeContext,
    call: ForeignCall,
) -> Result<runmat_value::ValueSequence, RuntimeError> {
    if state.call_active.replace(true) {
        let requested_outputs = call.requested_outputs;
        let (reply, response) = tokio::sync::oneshot::channel();
        let ownership = Rc::new(());
        state.nested_calls.borrow_mut().push_back(NestedPythonCall {
            call,
            reply,
            ownership: Rc::clone(&ownership),
        });
        let _lease = NestedCallLease::new(Rc::clone(&state.nested_calls), ownership);
        let outputs = response.await.map_err(|_| {
            foreign_error(
                ForeignErrorKind::HostUnavailable,
                "isolated Python callback ended before its nested call completed",
            )
        })??;
        return isolated_outputs(outputs, requested_outputs);
    }
    let mut guard = IsolatedCallGuard::begin(state.clone());
    if guard.client().is_none() {
        let configuration = state.configuration.borrow().clone();
        *guard.client() = Some(
            IsolatedPythonClient::spawn(&configuration)
                .await
                .map_err(super::isolation::runtime_error)?,
        );
        state.terminated.set(false);
    }
    let released = take_releases(&state.released)?;
    let requested_outputs = call.requested_outputs;
    let outcome = guard
        .client()
        .as_mut()
        .expect("isolated Python client initialized")
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
        guard.finish_terminated();
    } else {
        guard.finish_running();
    }
    outcome
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

async fn terminate_isolated(
    state: IsolatedPythonState,
    arguments: Vec<Value>,
) -> Result<Value, RuntimeError> {
    validate_environment(&state.handles, &state.host_identity, &arguments)?;
    if state.configuration.borrow().execution_mode != PythonExecutionMode::OutOfProcess {
        return Err(invalid_call(
            "terminate is available only for an OutOfProcess Python environment",
        ));
    }
    if state.call_active.replace(true) {
        return Err(invalid_call(
            "terminate cannot run during an active isolated Python call",
        ));
    }
    let mut guard = IsolatedCallGuard::begin(state.clone());
    let shutdown = if let Some(client) = guard.client().as_mut() {
        let released = take_releases(&state.released)?;
        client
            .shutdown(released)
            .await
            .map_err(super::isolation::runtime_error)
    } else {
        Ok(())
    };
    state.handles.restart_host(&state.host_identity)?;
    guard.finish_terminated();
    shutdown?;
    let configuration = state.configuration.borrow().clone();
    issued_python_status_value(
        &state.handles,
        &state.host_identity,
        &configuration,
        false,
        true,
        None,
    )
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
    fn visit_retained_values(&self, visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        let callbacks = self.callbacks.try_borrow().ok()?;
        if !callbacks.is_empty() {
            // Callback registrations retain RuntimeContexts as well as Values.
            // Their recursively injected services are not enumerable through
            // this adapter alone; never pretend callback values are all roots.
            return None;
        }
        for request in self.nested_isolated_calls.try_borrow().ok()?.iter() {
            for value in &request.call.arguments {
                visitor(value)?;
            }
        }
        Some(())
    }
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
            let arguments = call.arguments;
            return Box::pin(async move {
                terminate_isolated(state, arguments)
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
        Some(value) => value_text_scalar_string(&value)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| invalid_call(format!("{label} must be non-empty text"))),
        None => Err(invalid_call(format!("{label} is required"))),
    }
}

fn output_names(value: Option<Value>) -> Result<Vec<String>, RuntimeError> {
    match value {
        None => Ok(Vec::new()),
        Some(Value::String(value)) => Ok(vec![value]),
        Some(Value::CharArray(value)) => value_text_scalar_string(&Value::CharArray(value))
            .map(|value| vec![value])
            .ok_or_else(|| invalid_call("Python outputs must be text")),
        Some(Value::StringArray(value)) => Ok(value.data),
        Some(Value::Cell(value)) => value
            .data
            .iter()
            .map(|value| {
                value_text_scalar_string(value)
                    .ok_or_else(|| invalid_call("Python output names must be text"))
            })
            .collect(),
        Some(_) => Err(invalid_call("Python outputs must be text")),
    }
}

fn split_file_inputs(arguments: Vec<Value>) -> Result<(Vec<String>, Vec<Value>), RuntimeError> {
    if arguments.len() >= 2 {
        if let Some(position) = arguments.iter().position(|value| {
            value_text_scalar_string(value)
                .is_some_and(|value| value.eq_ignore_ascii_case("Arguments"))
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
                    value_text_scalar_string(value)
                        .ok_or_else(|| invalid_call("pyrunfile Arguments must contain text"))
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
#[path = "adapter/text_tests.rs"]
mod text_tests;

#[cfg(test)]
mod tests {
    use runmat_value::ObjectInstance;

    impl super::PythonAdapter {
        fn retained_values(&self) -> Option<Vec<runmat_value::Value>> {
            let mut values = Vec::new();
            crate::foreign::ForeignAdapter::visit_retained_values(self, &mut |value| {
                values.push(value.clone());
                Some(())
            })?;
            Some(values)
        }
    }
    #[test]
    fn retained_queue_values_disappear_on_queue_cancellation_and_callbacks_fail_closed() {
        let adapter = super::PythonAdapter::new(super::ForeignHandleRegistry::default()).unwrap();
        let (reply, _receiver) = tokio::sync::oneshot::channel();
        adapter
            .nested_isolated_calls
            .borrow_mut()
            .push_back(super::NestedPythonCall {
                call: crate::context::ForeignCall {
                    adapter: "python".into(),
                    symbol: "pending".into(),
                    arguments: vec![runmat_value::Value::String("retained-argument".into())],
                    requested_outputs: 1,
                },
                reply,
                ownership: Rc::new(()),
            });
        assert_eq!(
            adapter.retained_values(),
            Some(vec![runmat_value::Value::String(
                "retained-argument".into()
            )])
        );
        adapter.nested_isolated_calls.borrow_mut().clear();
        assert_eq!(adapter.retained_values(), Some(vec![]));
        let foreign: std::rc::Rc<dyn crate::context::RuntimeForeignService> = std::rc::Rc::new(
            crate::foreign::ForeignRuntime::new(crate::foreign::ForeignPlatform::Native),
        );
        adapter.callbacks.borrow_mut().insert(
            1,
            super::PythonCallbackRegistration {
                callback: runmat_value::Value::String("nested-capture".into()),
                context: crate::context::RuntimeContext::new(std::rc::Rc::new(
                    crate::execution::RuntimeExecutionService::new(),
                )),
                foreign: std::rc::Rc::downgrade(&foreign),
            },
        );
        assert!(
            adapter.retained_values().is_none(),
            "callback context services must not be mistaken for an empty root set"
        );
        adapter.callbacks.borrow_mut().clear();
        assert_eq!(adapter.retained_values(), Some(vec![]));
    }
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
        let later = adapter.status_value().expect("read status again");
        validate_environment(
            &adapter.handles,
            &adapter.host_identity,
            &[Value::Object(environment)],
        )
        .unwrap();
        validate_environment(&adapter.handles, &adapter.host_identity, &[later]).unwrap();
        assert!(!adapter.is_running());
    }

    #[test]
    fn environment_control_rejects_synthetic_cross_registry_and_stale_values() {
        let adapter = PythonAdapter::new(ForeignHandleRegistry::default()).unwrap();
        let original = adapter.status_value().unwrap();
        let Value::Object(original_object) = &original else {
            panic!("expected PythonEnvironment");
        };
        assert!(original_object.foreign_host_control().is_some());
        let mut synthetic = ObjectInstance::new(runmat_builtins::PYTHON_ENVIRONMENT_CLASS.owned());
        synthetic.properties = original_object.properties.clone();
        assert_eq!(
            validate_environment(
                &adapter.handles,
                &adapter.host_identity,
                &[Value::Object(synthetic)]
            )
            .unwrap_err()
            .identifier(),
            Some("RunMat:Foreign:InvalidCall")
        );
        let other = PythonAdapter::new(ForeignHandleRegistry::default()).unwrap();
        let other_status = other.status_value().unwrap();
        assert_eq!(
            validate_environment(&other.handles, &other.host_identity, &[original.clone()])
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:HandleMetadataMismatch")
        );
        validate_environment(&other.handles, &other.host_identity, &[other_status]).unwrap();

        let mut out_of_process = adapter.configuration.borrow().clone();
        out_of_process.execution_mode = PythonExecutionMode::OutOfProcess;
        adapter.configure(out_of_process).unwrap();
        assert_eq!(
            validate_environment(&adapter.handles, &adapter.host_identity, &[original])
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:StaleHandle")
        );
        let fresh = adapter.status_value().unwrap();
        validate_environment(&adapter.handles, &adapter.host_identity, &[fresh.clone()]).unwrap();
        adapter.terminate_value(vec![fresh]).unwrap();
        assert!(adapter.terminated.get());
    }

    #[test]
    fn rejected_terminate_never_fences_a_valid_environment() {
        let adapter = PythonAdapter::new(ForeignHandleRegistry::default()).unwrap();
        let mut configuration = adapter.configuration.borrow().clone();
        configuration.execution_mode = PythonExecutionMode::OutOfProcess;
        adapter.configure(configuration).unwrap();
        let valid = adapter.status_value().unwrap();
        let synthetic = Value::Object(ObjectInstance::new(
            runmat_builtins::PYTHON_ENVIRONMENT_CLASS.owned(),
        ));
        assert_eq!(
            adapter
                .terminate_value(vec![synthetic])
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:InvalidCall")
        );
        assert!(!adapter.terminated.get());
        validate_environment(&adapter.handles, &adapter.host_identity, &[valid.clone()]).unwrap();
        let terminated = adapter.terminate_value(vec![valid.clone()]).unwrap();
        assert_eq!(
            validate_environment(&adapter.handles, &adapter.host_identity, &[valid])
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:StaleHandle")
        );
        validate_environment(&adapter.handles, &adapter.host_identity, &[terminated]).unwrap();
    }

    #[test]
    fn failed_reconfiguration_preserves_ticket_and_real_python_resources_block_policy_change() {
        let adapter = PythonAdapter::new(ForeignHandleRegistry::default()).unwrap();
        let environment = adapter.status_value().unwrap();
        let object = adapter
            .handles
            .register_resource(&adapter.host_identity, environment_metadata())
            .unwrap();
        let mut configuration = adapter.configuration.borrow().clone();
        configuration.execution_mode = PythonExecutionMode::OutOfProcess;
        assert_eq!(
            adapter
                .configure(configuration.clone())
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:InvalidCall")
        );
        assert_eq!(
            adapter.clear_artifact_bundle().unwrap_err().identifier(),
            Some("RunMat:Foreign:InvalidCall")
        );
        validate_environment(
            &adapter.handles,
            &adapter.host_identity,
            &[environment.clone()],
        )
        .unwrap();
        drop(object);
        adapter.configure(configuration).unwrap();
        assert_eq!(
            validate_environment(&adapter.handles, &adapter.host_identity, &[environment])
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:StaleHandle")
        );
    }

    #[test]
    fn failed_portable_artifact_install_keeps_previous_configuration_and_ticket() {
        let adapter = PythonAdapter::new(ForeignHandleRegistry::default()).unwrap();
        let original = adapter.status_value().unwrap();
        let previous = adapter.configuration.borrow().clone();
        let mut bundle = PythonArtifactBundle::empty();
        bundle.schema_version = 0;
        bundle.environment = Some(runmat_python::PythonEnvironmentIdentity {
            implementation: "cpython".into(),
            version: runmat_python::PythonVersion {
                major: 3,
                minor: 11,
                patch: 0,
            },
            abi_tag: "invalid-test-abi".into(),
            platform_tag: "invalid-test-platform".into(),
            execution_mode: PythonExecutionMode::OutOfProcess,
        });
        assert!(adapter.install_portable_artifact_bundle(&bundle).is_err());
        assert_eq!(*adapter.configuration.borrow(), previous);
        validate_environment(&adapter.handles, &adapter.host_identity, &[original]).unwrap();
    }

    #[test]
    fn isolated_terminate_validates_again_before_shutdown_state_changes() {
        let adapter = PythonAdapter::new(ForeignHandleRegistry::default()).unwrap();
        let mut configuration = adapter.configuration.borrow().clone();
        configuration.execution_mode = PythonExecutionMode::OutOfProcess;
        adapter.configure(configuration).unwrap();
        let valid = adapter.status_value().unwrap();
        let state = adapter.isolated_state();
        let invalid = Value::Object(ObjectInstance::new(
            runmat_builtins::PYTHON_ENVIRONMENT_CLASS.owned(),
        ));
        assert_eq!(
            pollster::block_on(terminate_isolated(state.clone(), vec![invalid]))
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:InvalidCall")
        );
        assert!(!state.call_active.get());
        assert!(!state.terminated.get());
        validate_environment(&state.handles, &state.host_identity, &[valid.clone()]).unwrap();
        let result = pollster::block_on(terminate_isolated(state, vec![valid.clone()])).unwrap();
        assert!(matches!(result, Value::Object(_)));
        assert_eq!(
            validate_environment(&adapter.handles, &adapter.host_identity, &[valid])
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:StaleHandle")
        );
    }

    #[test]
    fn abandoned_isolated_call_releases_slot_and_fences_old_authority() {
        let adapter = PythonAdapter::new(ForeignHandleRegistry::default()).unwrap();
        let mut configuration = adapter.configuration.borrow().clone();
        configuration.execution_mode = PythonExecutionMode::OutOfProcess;
        adapter.configure(configuration).unwrap();
        let environment = adapter.status_value().unwrap();
        let object = adapter
            .handles
            .register_resource(&adapter.host_identity, environment_metadata())
            .unwrap();
        let state = adapter.isolated_state();
        state.call_active.set(true);
        let guard = IsolatedCallGuard::begin(state.clone());
        let (reply, _response) = tokio::sync::oneshot::channel();
        state.nested_calls.borrow_mut().push_back(NestedPythonCall {
            call: ForeignCall {
                adapter: PYTHON_ADAPTER_ID.into(),
                symbol: "pending".into(),
                arguments: vec![Value::String("retained".into())],
                requested_outputs: 1,
            },
            reply,
            ownership: Rc::new(()),
        });
        drop(guard);
        assert!(!state.call_active.get());
        assert!(state.terminated.get());
        assert!(state.nested_calls.borrow().is_empty());
        assert_eq!(
            validate_environment(&state.handles, &state.host_identity, &[environment])
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:StaleHandle")
        );
        assert_eq!(
            state
                .handles
                .resolve(&object, ForeignCapability::Invoke)
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:StaleHandle")
        );
        assert!(state.released.0.lock().unwrap().is_empty());
        let fresh = adapter.status_value().unwrap();
        validate_environment(&state.handles, &state.host_identity, &[fresh]).unwrap();
        drop(object);
        assert!(state.released.0.lock().unwrap().is_empty());
    }

    #[test]
    fn unpolled_isolated_shutdown_does_not_revoke_or_take_the_client_slot() {
        let adapter = PythonAdapter::new(ForeignHandleRegistry::default()).unwrap();
        let mut configuration = adapter.configuration.borrow().clone();
        configuration.execution_mode = PythonExecutionMode::OutOfProcess;
        adapter.configure(configuration).unwrap();
        let status = adapter.status_value().unwrap();
        let future = terminate_isolated(adapter.isolated_state(), vec![status.clone()]);
        drop(future);
        assert!(!adapter.isolated_call_active.get());
        assert!(!adapter.terminated.get());
        validate_environment(&adapter.handles, &adapter.host_identity, &[status]).unwrap();
    }

    #[tokio::test]
    async fn dropped_queued_nested_request_cannot_dispatch_or_retain_arguments() {
        let adapter = PythonAdapter::new(ForeignHandleRegistry::default()).unwrap();
        let status = adapter.status_value().unwrap();
        // A callback's outer client owns the active slot; the nested future
        // only owns its queued request, not the session or its controls.
        adapter.isolated_call_active.set(true);
        let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let mut pending = Box::pin(invoke_isolated(
            adapter.isolated_state(),
            context,
            ForeignCall {
                adapter: PYTHON_ADAPTER_ID.into(),
                symbol: "status".into(),
                arguments: vec![Value::String("must-not-dispatch".into())],
                requested_outputs: 1,
            },
        ));
        tokio::select! {
            result = &mut pending => panic!("queued nested call unexpectedly completed: {result:?}"),
            () = tokio::time::sleep(std::time::Duration::from_millis(1)) => {}
        }
        assert_eq!(adapter.nested_isolated_calls.borrow().len(), 1);
        drop(pending);
        assert!(adapter.nested_isolated_calls.borrow().is_empty());
        assert_eq!(adapter.retained_values(), Some(vec![]));
        validate_environment(&adapter.handles, &adapter.host_identity, &[status]).unwrap();
        assert!(adapter.isolated_call_active.get());
    }

    #[cfg(unix)]
    async fn pending_isolated_adapter() -> (Rc<PythonAdapter>, Value, ForeignRef, u32) {
        let adapter = PythonAdapter::new(ForeignHandleRegistry::default()).unwrap();
        let mut configuration = adapter.configuration.borrow().clone();
        configuration.execution_mode = PythonExecutionMode::OutOfProcess;
        adapter.configure(configuration).unwrap();
        let status = adapter.status_value().unwrap();
        let object = adapter
            .handles
            .register_resource(&adapter.host_identity, environment_metadata())
            .unwrap();
        let client = IsolatedPythonClient::pending_test_client().await;
        let pid = client.process_id().expect("owned test child pid");
        *adapter.isolated_client.borrow_mut() = Some(client);
        (adapter, status, object, pid)
    }

    #[cfg(unix)]
    async fn assert_abandoned_child_and_authority(
        adapter: &PythonAdapter,
        old_status: Value,
        old_object: ForeignRef,
        pid: u32,
    ) {
        assert!(!adapter.isolated_call_active.get());
        assert!(!adapter.is_running());
        assert!(adapter.terminated.get());
        assert_eq!(
            validate_environment(&adapter.handles, &adapter.host_identity, &[old_status])
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:StaleHandle")
        );
        assert_eq!(
            adapter
                .handles
                .resolve(&old_object, ForeignCapability::Invoke)
                .unwrap_err()
                .identifier(),
            Some("RunMat:Foreign:StaleHandle")
        );
        let fresh = adapter.status_value().unwrap();
        validate_environment(&adapter.handles, &adapter.host_identity, &[fresh]).unwrap();
        drop(old_object);
        assert!(adapter.released.0.lock().unwrap().is_empty());
        let mut stopped = false;
        for _ in 0..100 {
            // kill_on_drop is synchronous; Tokio may reap the child on a
            // later runtime tick, so allow a bounded reap interval.
            if unsafe { libc::kill(pid as libc::pid_t, 0) } == -1 {
                stopped = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        assert!(
            stopped,
            "abandoned owned Python child {pid} is still present"
        );
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn dropped_pending_isolated_request_kills_child_and_fences_both_leases() {
        let (adapter, status, object, pid) = pending_isolated_adapter().await;
        let context = RuntimeContext::new(Rc::new(RuntimeExecutionService::new()));
        let call = ForeignCall {
            adapter: PYTHON_ADAPTER_ID.into(),
            symbol: "status".into(),
            arguments: Vec::new(),
            requested_outputs: 1,
        };
        let mut pending = Box::pin(invoke_isolated(adapter.isolated_state(), context, call));
        tokio::select! {
            result = &mut pending => panic!("sleeping child unexpectedly replied: {result:?}"),
            () = tokio::time::sleep(std::time::Duration::from_millis(20)) => {}
        }
        assert!(adapter.isolated_call_active.get());
        drop(pending);
        assert_abandoned_child_and_authority(&adapter, status, object, pid).await;
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn dropped_pending_isolated_shutdown_kills_child_and_fences_both_leases() {
        let (adapter, status, object, pid) = pending_isolated_adapter().await;
        let mut pending = Box::pin(terminate_isolated(
            adapter.isolated_state(),
            vec![status.clone()],
        ));
        tokio::select! {
            result = &mut pending => panic!("sleeping child unexpectedly replied: {result:?}"),
            () = tokio::time::sleep(std::time::Duration::from_millis(20)) => {}
        }
        assert!(adapter.isolated_call_active.get());
        drop(pending);
        assert_abandoned_child_and_authority(&adapter, status, object, pid).await;
    }
}
