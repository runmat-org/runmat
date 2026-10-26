use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::rc::{Rc, Weak};
use std::sync::{Arc, Mutex};

use runmat_java::{
    discover_jvm, ClasspathSnapshot, JavaArtifactIdentity, JavaCallbackInvocation,
    JavaDiscoveryRequest, JavaInvocationError, JavaLane, JavaObjectHandle, JavaValue, JvmConfig,
    JvmInstallation, JvmProcess, SessionClasspath, JAVA_ADAPTER_ID, JAVA_ADAPTER_VERSION,
};
use runmat_process_host::private_bytes::PrivateByteLease;
use runmat_types::{
    CapabilityRequirement, ForeignAffinity, ForeignCapability, ForeignLifetime, ForeignOwnership,
    ForeignTypeIdentity,
};
use runmat_value::{CellArray, CharArray, ForeignRef, ForeignResourceKey, Value, WeakForeignRef};

use super::super::project_retirement::ProjectRetirementAuthority;
use super::super::{
    foreign_callback_request, foreign_error, invoke_foreign_callback, ForeignAdapter,
    ForeignAdapterDescriptor, ForeignAdapterFuture, ForeignErrorKind, ForeignExecutionPolicy,
    ForeignHandleRegistry, ForeignHostRegistration, ForeignHostRelease, ForeignResourceMetadata,
};
use super::conversion::{array_from_java, invalid_conversion, scalar_from_java, value_to_java};
use crate::builtins::common::validation::value_text_scalar_string;
use crate::context::{ForeignCall, ForeignCallCompletion, RuntimeContext};
use crate::{build_runtime_error, RuntimeError};

#[derive(Debug, Default)]
struct ReleaseQueue(Mutex<Vec<u64>>);

impl ForeignHostRelease for ReleaseQueue {
    fn release(&self, key: &ForeignResourceKey) {
        if let Ok(mut released) = self.0.lock() {
            released.push(key.handle);
        }
    }
}

pub struct JavaAdapter {
    handles: ForeignHandleRegistry,
    self_weak: Weak<JavaAdapter>,
    host_identity: String,
    session: RefCell<Option<Rc<JavaLane>>>,
    selected_byte_lease: RefCell<Option<PrivateByteLease>>,
    retirement_requested: Cell<bool>,
    project_retirement: ProjectRetirementAuthority,
    discovery: RefCell<JavaDiscoveryRequest>,
    config: RefCell<JvmConfig>,
    initial_classpath: RefCell<SessionClasspath>,
    configured_classpath: RefCell<Vec<std::path::PathBuf>>,
    configured_classpath_revision: Cell<u64>,
    released: Arc<ReleaseQueue>,
    resources: Rc<RefCell<BTreeMap<u64, JavaObjectHandle>>>,
    java_to_foreign: Rc<RefCell<BTreeMap<JavaObjectHandle, WeakForeignRef>>>,
    callbacks: RefCell<BTreeMap<u64, Rc<JavaCallbackRegistration>>>,
    next_callback: Cell<u64>,
    artifact_identities: RefCell<BTreeSet<JavaArtifactIdentity>>,
    desktop_available: Cell<bool>,
}

struct JavaCallbackRegistration {
    callable: crate::call::RetainedCallableTarget,
    origin: crate::context::RuntimeContextLocalId,
    context: RuntimeContext,
    foreign: std::rc::Weak<dyn crate::context::RuntimeForeignService>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JavaRuntimeConfiguration {
    pub home: Option<std::path::PathBuf>,
    pub minimum_version: u16,
    pub maximum_version: Option<u16>,
    pub classpath: Vec<std::path::PathBuf>,
    pub options: Vec<String>,
}

mod project_replacement;
mod project_retirement;
mod in_process;
mod lane_call;
mod prepare;
use project_replacement::prepare_configuration;
pub use project_replacement::{PreparedJavaProjectReplacement, ReadyJavaProjectReplacement};
pub use project_retirement::VerifiedJavaProjectRetirement;

impl Default for JavaRuntimeConfiguration {
    fn default() -> Self {
        Self {
            home: None,
            minimum_version: 8,
            maximum_version: None,
            classpath: Vec::new(),
            options: Vec::new(),
        }
    }
}

impl std::fmt::Debug for JavaAdapter {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("JavaAdapter")
            .field("host_identity", &self.host_identity)
            .field("running", &self.session.borrow().is_some())
            .finish_non_exhaustive()
    }
}

impl JavaAdapter {
    pub fn new(handles: ForeignHandleRegistry) -> Result<Rc<Self>, RuntimeError> {
        Self::with_configuration(
            handles,
            JavaDiscoveryRequest::from_process(),
            JvmConfig::default(),
        )
    }

    pub fn with_configuration(
        handles: ForeignHandleRegistry,
        discovery: JavaDiscoveryRequest,
        config: JvmConfig,
    ) -> Result<Rc<Self>, RuntimeError> {
        let host_identity = "java-process".to_string();
        let initial_classpath = SessionClasspath::new(config.bootstrap_classpath.clone(), [])
            .map_err(|error| invalid_conversion(error.to_string()))?;
        let released = Arc::new(ReleaseQueue::default());
        handles.register_host(ForeignHostRegistration {
            identity: host_identity.clone(),
            adapter: JAVA_ADAPTER_ID.to_string(),
            session_identity: host_identity.clone(),
            capabilities: BTreeSet::from([
                ForeignCapability::Invoke,
                ForeignCapability::Read,
                ForeignCapability::Write,
                ForeignCapability::Callback,
                ForeignCapability::Transfer,
            ]),
            policy: ForeignExecutionPolicy::trusted_in_process(),
            release: released.clone(),
        })?;
        Ok(Rc::new_cyclic(|self_weak| Self {
            handles,
            self_weak: self_weak.clone(),
            host_identity,
            session: RefCell::new(None),
            selected_byte_lease: RefCell::new(None),
            retirement_requested: Cell::new(false),
            project_retirement: ProjectRetirementAuthority::default(),
            discovery: RefCell::new(discovery),
            config: RefCell::new(config),
            initial_classpath: RefCell::new(initial_classpath),
            configured_classpath: RefCell::new(Vec::new()),
            configured_classpath_revision: Cell::new(1),
            released,
            resources: Rc::new(RefCell::new(BTreeMap::new())),
            java_to_foreign: Rc::new(RefCell::new(BTreeMap::new())),
            callbacks: RefCell::new(BTreeMap::new()),
            next_callback: Cell::new(1),
            artifact_identities: RefCell::new(BTreeSet::new()),
            desktop_available: Cell::new(false),
        }))
    }

    pub fn is_running(&self) -> bool {
        self.session.borrow().is_some()
    }

    pub fn set_desktop_available(&self, available: bool) {
        self.desktop_available.set(available);
    }

    pub fn configure(&self, configuration: JavaRuntimeConfiguration) -> Result<(), RuntimeError> {
        if self.project_retirement.is_pending() {
            return Err(invalid_conversion("Java project classloader is retiring"));
        }
        let session = self
            .session
            .try_borrow()
            .map_err(|_| invalid_conversion("Java session is busy during configuration"))?;
        if session.is_some() {
            return Err(invalid_conversion(
                "Java runtime configuration is immutable after JVM startup",
            ));
        }
        drop(session);
        let prepared =
            prepare_configuration(configuration, self.configured_classpath_revision.get())?;
        let mut classpath_slot = self
            .initial_classpath
            .try_borrow_mut()
            .map_err(|_| invalid_conversion("Java classpath is busy during configuration"))?;
        let mut identities = self
            .artifact_identities
            .try_borrow_mut()
            .map_err(|_| invalid_conversion("Java artifacts are busy during configuration"))?;
        let mut configured_slot = self.configured_classpath.try_borrow_mut().map_err(|_| {
            invalid_conversion("configured Java paths are busy during configuration")
        })?;
        let mut discovery_slot = self
            .discovery
            .try_borrow_mut()
            .map_err(|_| invalid_conversion("Java discovery is busy during configuration"))?;
        let mut config_slot = self
            .config
            .try_borrow_mut()
            .map_err(|_| invalid_conversion("Java runtime policy is busy during configuration"))?;
        *discovery_slot = prepared.discovery;
        *config_slot = prepared.config;
        *classpath_slot = prepared.classpath;
        *configured_slot = prepared.configured_classpath;
        self.configured_classpath_revision
            .set(prepared.next_revision);
        identities.clear();
        Ok(())
    }

    pub fn install_project_artifacts(
        &self,
        artifacts: &[(JavaArtifactIdentity, std::path::PathBuf)],
    ) -> Result<(), RuntimeError> {
        self.prepare_project_replacement(artifacts)?
            .admit_commit()?
            .commit();
        Ok(())
    }

    pub fn attach_selected_byte_lease(&self, lease: PrivateByteLease) -> Result<(), RuntimeError> {
        if self.session.borrow().is_some() || self.selected_byte_lease.borrow().is_some() {
            return Err(invalid_conversion(
                "Java selected byte owner may be attached only once before session startup",
            ));
        }
        *self.selected_byte_lease.borrow_mut() = Some(lease);
        Ok(())
    }

    /// Closes only this adapter's classloader. A failed close leaves the
    /// exact session and private-byte owner available for retirement retry.
    pub fn shutdown_selected_retirement(&self) -> Result<(), RuntimeError> {
        self.retirement_requested.set(true);
        let session = self.session.borrow().as_ref().cloned();
        let Some(session) = session else {
            self.selected_byte_lease.borrow_mut().take();
            self.callbacks.borrow_mut().clear();
            return Ok(());
        };
        session
            .retire_classloader()
            .map_err(java_invocation_error)?;
        self.session.borrow_mut().take();
        self.selected_byte_lease.borrow_mut().take();
        self.callbacks.borrow_mut().clear();
        Ok(())
    }

    fn ensure_session(&self) -> Result<(), RuntimeError> {
        if self.project_retirement.is_pending() {
            return Err(invalid_conversion("Java project classloader is retiring"));
        }
        if self.session.borrow().is_some() {
            return Ok(());
        }
        let installation = discover_jvm(&self.discovery.borrow()).map_err(java_runtime_error)?;
        let process =
            JvmProcess::launch(installation, &self.config.borrow()).map_err(java_runtime_error)?;
        *self.session.borrow_mut() = Some(Rc::new(
            JavaLane::start(
                process,
                self.initial_classpath.borrow().clone(),
                self.selected_byte_lease.borrow().clone(),
            )
            .map_err(java_invocation_error)?,
        ));
        Ok(())
    }

    fn invoke_control_now(
        &self,
        runtime: Option<&RuntimeContext>,
        call: ForeignCall,
    ) -> Result<runmat_value::ValueSequence, RuntimeError> {
        match call.symbol.as_str() {
            "status" => self
                .status_value(runtime)
                .and_then(super::super::single_output_sequence),
            "configure" => self
                .configure_value(runtime, call.arguments)
                .and_then(super::super::single_output_sequence),
            _ => Err(invalid_conversion(
                "unsupported synchronous Java control operation",
            )),
        }
    }
    fn value_from_java(&self, value: JavaValue) -> Result<Value, RuntimeError> {
        value_from_java_parts(
            &self.handles,
            &self.host_identity,
            &self.resources,
            &self.java_to_foreign,
            value,
        )
    }

    fn resolve_java_handle(
        &self,
        reference: &ForeignRef,
    ) -> Result<JavaObjectHandle, RuntimeError> {
        let resolved = self.handles.resolve(reference, ForeignCapability::Invoke)?;
        if resolved.adapter != JAVA_ADAPTER_ID {
            return Err(foreign_error(
                ForeignErrorKind::HandleMetadataMismatch,
                "foreign receiver is not owned by the Java adapter",
            ));
        }
        self.resources
            .borrow()
            .get(&reference.handle)
            .copied()
            .ok_or_else(|| foreign_error(ForeignErrorKind::StaleHandle, "Java object is stale"))
    }

    fn argument_to_java(
        &self,
        context: &RuntimeContext,
        value: crate::call::RootedArgumentValue,
    ) -> Result<JavaValue, RuntimeError> {
        crate::foreign::reject_scoped_callback(value.value())?;
        if super::super::is_callable(value.value()) {
            let instance = value
                .value()
                .callable_instance_key()
                .expect("every callable value has a physical instance");
            if let Some((token, _)) = self.callbacks.borrow().iter().find(|(_, registration)| {
                registration.origin == context.local_identity()
                    && registration.callable.value().callable_instance_key() == Some(instance)
            }) {
                return Ok(JavaValue::Callback(*token));
            }
            let foreign = context
                .service_ports()
                .foreign()
                .map(Rc::downgrade)
                .ok_or_else(|| invalid_conversion("Java callback requires a foreign runtime"))?;
            let token = self.next_callback.get();
            self.next_callback.set(
                token.checked_add(1).ok_or_else(|| {
                    invalid_conversion("Java callback identity space is exhausted")
                })?,
            );
            let callable = match value {
                crate::call::RootedArgumentValue::Owned(value) => {
                    crate::call::RetainedCallableTarget::Direct(
                        crate::execution::RetainedValueReadLease::prepare_owned(value, context)?,
                    )
                }
                crate::call::RootedArgumentValue::Captured(value) => {
                    crate::call::RetainedCallableTarget::Captured(value)
                }
            };
            let callback_context = context
                .clone()
                .with_service_ports(context.service_ports().clone().without_foreign());
            self.callbacks.borrow_mut().insert(
                token,
                Rc::new(JavaCallbackRegistration {
                    callable,
                    origin: context.local_identity(),
                    context: callback_context,
                    foreign,
                }),
            );
            return Ok(JavaValue::Callback(token));
        }
        let Value::Foreign(reference) = value.value() else {
            return match value.into_owned() {
                Ok(value) => value_to_java(value),
                Err(value) => super::conversion::value_ref_to_java(value.value()),
            };
        };
        let reference = reference.clone();
        if reference.type_identity.family != JAVA_ADAPTER_ID {
            return Err(invalid_conversion("foreign argument is not a Java object"));
        }
        let handle = self.resolve_java_handle(&reference)?;
        Ok(JavaValue::Object {
            handle,
            class_name: reference.type_identity.name.clone(),
        })
    }

    fn classpath_value(
        &self,
        layer_argument: Option<&Value>,
        snapshot: ClasspathSnapshot,
    ) -> Result<Value, RuntimeError> {
        let layer = layer_argument
            .map(|value| string_argument_ref(Some(value), "Java classpath layer"))
            .transpose()?;
        let layer = layer
            .as_deref()
            .map(runmat_builtins::JavaClasspathQuery::parse_wire)
            .unwrap_or(Some(runmat_builtins::JavaClasspathQuery::All))
            .ok_or_else(|| {
                invalid_conversion("Java classpath layer must be 'all', 'static', or 'dynamic'")
            })?;
        let paths = classpath_paths(&snapshot, layer);
        let length = paths.len();
        CellArray::new(
            paths
                .into_iter()
                .map(|path| Value::CharArray(CharArray::new_row(&path.to_string_lossy())))
                .collect(),
            length,
            1,
        )
        .map(Value::Cell)
        .map_err(invalid_conversion)
    }

    fn status_value(&self, runtime: Option<&RuntimeContext>) -> Result<Value, RuntimeError> {
        let running = self.session.borrow().is_some();
        let installation = if let Some(session) = self.session.borrow().as_ref() {
            Some(session.installation().clone())
        } else {
            discover_jvm(&self.discovery.borrow()).ok()
        };
        let fields = [
            (
                "Version",
                Value::string(
                    installation
                        .as_ref()
                        .map(|value| value.version.raw.clone())
                        .unwrap_or_default(),
                ),
            ),
            (
                "Home",
                Value::string(
                    installation
                        .as_ref()
                        .map(|value| value.home.display().to_string())
                        .unwrap_or_default(),
                ),
            ),
            (
                "Library",
                Value::string(
                    installation
                        .as_ref()
                        .map(|value| value.library.display().to_string())
                        .unwrap_or_default(),
                ),
            ),
            (
                "Status",
                Value::String(if running { "loaded" } else { "notloaded" }.into()),
            ),
            (
                "Configuration",
                Value::string({
                    let discovery = self.discovery.borrow();
                    discovery
                        .explicit_home
                        .as_ref()
                        .or(discovery.environment_home.as_ref())
                        .map(|path| path.display().to_string())
                        .unwrap_or_else(|| "system".into())
                }),
            ),
        ];
        crate::execution::object::construct_object_from_borrowed_fields(
            runtime,
            runmat_builtins::JAVA_ENVIRONMENT_CLASS.display_name(),
            fields.iter().map(|(name, value)| (*name, value)),
        )
        // Environment status has no GC edges; retain its original generation.
        .map(|status| status.value().clone())
    }

    fn usejava_value(&self, arguments: Vec<Value>) -> Result<Value, RuntimeError> {
        let feature = string_argument(arguments.into_iter().next(), "Java feature")?;
        let feature = runmat_builtins::JavaFeature::parse(&feature).ok_or_else(|| {
            invalid_conversion("Java feature must be 'jvm', 'awt', 'swing', or 'desktop'")
        })?;
        match feature {
            runmat_builtins::JavaFeature::Jvm => Ok(Value::Bool(
                self.session.borrow().is_some() || discover_jvm(&self.discovery.borrow()).is_ok(),
            )),
            runmat_builtins::JavaFeature::Desktop => Ok(Value::Bool(self.desktop_available.get())),
            runmat_builtins::JavaFeature::Awt | runmat_builtins::JavaFeature::Swing => Err(
                invalid_conversion("Java AWT capability requires asynchronous admission"),
            ),
        }
    }

    fn configure_value(
        &self,
        runtime: Option<&RuntimeContext>,
        arguments: crate::call::RootedFunctionArguments,
    ) -> Result<Value, RuntimeError> {
        if self.is_running() {
            return Err(invalid_conversion(
                "jenv cannot change Java configuration after JVM startup",
            ));
        }
        if arguments.len() != 2 {
            return Err(invalid_conversion(
                "jenv configuration requires one name-value pair",
            ));
        }
        let name = string_argument_ref(arguments.get(0), "jenv option")?;
        let value = string_argument_ref(arguments.get(1), "jenv option value")?;
        let control = runmat_builtins::JavaEnvironmentControl::parse(&name).ok_or_else(|| {
            invalid_conversion("jenv supports the 'Version' and 'ExecutionMode' options")
        })?;
        match control {
            runmat_builtins::JavaEnvironmentControl::Version => {
                let path = std::path::PathBuf::from(&value);
                let mut discovery = self.discovery.borrow().clone();
                let mut config = self.config.borrow().clone();
                if path.exists() || value.contains(std::path::MAIN_SEPARATOR) {
                    discovery.explicit_home = Some(path);
                    discovery.required_major = None;
                } else {
                    let version = runmat_java::JvmVersion::parse(value.clone())
                        .map_err(java_runtime_error)?;
                    discovery.explicit_home = None;
                    discovery.required_major = Some(version.major);
                    config.minimum_major = version.major;
                    config.maximum_major = Some(version.major);
                }
                let installation = discover_jvm(&discovery).map_err(java_runtime_error)?;
                if !config.accepts(&installation.version) {
                    return Err(invalid_conversion(format!(
                        "Java {} does not satisfy the configured version bounds",
                        installation.version.raw
                    )));
                }
                *self.discovery.borrow_mut() = discovery;
                *self.config.borrow_mut() = config;
            }
            runmat_builtins::JavaEnvironmentControl::ExecutionMode => {
                if runmat_builtins::JavaExecutionMode::parse(&value).is_none() {
                    return Err(invalid_conversion(
                        "RunMat currently supports Java execution mode 'InProcess'",
                    ));
                }
            }
        }
        self.status_value(runtime)
    }
}

fn value_from_java_parts(
    handles: &ForeignHandleRegistry,
    host_identity: &str,
    resources: &RefCell<BTreeMap<u64, JavaObjectHandle>>,
    java_to_foreign: &RefCell<BTreeMap<JavaObjectHandle, WeakForeignRef>>,
    value: JavaValue,
) -> Result<Value, RuntimeError> {
    if let JavaValue::Array {
        component,
        elements,
    } = value
    {
        if matches!(
            component,
            runmat_java::JavaParameterType::Object(_) | runmat_java::JavaParameterType::Array(_)
        ) {
            let length = elements.len();
            let mut converted = crate::execution::RootedValuesBuilder::with_capacity(length);
            for value in elements {
                converted
                    .push(value_from_java_parts(
                        handles,
                        host_identity,
                        resources,
                        java_to_foreign,
                        value,
                    )?)
                    .map_err(|error| error.into_runtime_error())?;
            }
            return converted
                .finish()
                .map_err(|error| error.into_runtime_error())?
                .with_unbounded_values(|values| {
                    CellArray::new(values, length, 1)
                        .map(Value::Cell)
                        .map_err(invalid_conversion)
                })?;
        }
        return array_from_java(component, elements);
    }
    let JavaValue::Object { handle, class_name } = value else {
        return scalar_from_java(value);
    };
    if let Some(reference) = java_to_foreign
        .borrow()
        .get(&handle)
        .and_then(WeakForeignRef::upgrade)
    {
        return Ok(Value::Foreign(reference));
    }
    let reference = handles.register_resource(
        host_identity,
        ForeignResourceMetadata {
            type_identity: ForeignTypeIdentity {
                family: JAVA_ADAPTER_ID.into(),
                name: class_name,
                version: JAVA_ADAPTER_VERSION,
            },
            ownership: ForeignOwnership::Shared,
            affinity: ForeignAffinity::OriginProcess,
            lifetime: ForeignLifetime::Session,
        },
    )?;
    resources.borrow_mut().insert(reference.handle, handle);
    if let Some(weak) = reference.downgrade() {
        java_to_foreign.borrow_mut().insert(handle, weak);
    }
    Ok(Value::Foreign(reference))
}

fn classpath_paths(
    snapshot: &ClasspathSnapshot,
    layer: runmat_builtins::JavaClasspathQuery,
) -> Vec<&std::path::PathBuf> {
    match layer {
        runmat_builtins::JavaClasspathQuery::All => snapshot
            .bootstrap
            .iter()
            .chain(snapshot.project.iter())
            .chain(snapshot.dynamic.iter())
            .collect(),
        runmat_builtins::JavaClasspathQuery::Static => snapshot
            .bootstrap
            .iter()
            .chain(snapshot.project.iter())
            .collect(),
        runmat_builtins::JavaClasspathQuery::Dynamic => snapshot.dynamic.iter().collect(),
    }
}

impl ForeignAdapter for JavaAdapter {
    fn visit_retained_values(&self, _visitor: &mut dyn FnMut(&Value) -> Option<()>) -> Option<()> {
        // JNI object identities and JavaValue conversions do not retain local
        // RunMat Values; invocation arguments belong to the returned future.
        Some(())
    }
    fn descriptor(&self) -> ForeignAdapterDescriptor {
        ForeignAdapterDescriptor {
            adapter: runmat_types::ForeignAdapterId::new(JAVA_ADAPTER_ID)
                .expect("the built-in Java adapter identity is valid"),
            version: JAVA_ADAPTER_VERSION,
            capabilities: BTreeSet::from([CapabilityRequirement::ForeignRuntime]),
            foreign_capabilities: BTreeSet::from([
                ForeignCapability::Invoke,
                ForeignCapability::Read,
                ForeignCapability::Write,
                ForeignCapability::Callback,
                ForeignCapability::Transfer,
            ]),
            artifact_identities: self
                .artifact_identities
                .borrow()
                .iter()
                .map(|identity| {
                    runmat_types::ForeignArtifactIdentity::new(identity.as_str())
                        .expect("installed Java artifact identities are canonical")
                })
                .collect(),
            supports_wasm: false,
            supports_host_bridge: false,
            execution_stack: runmat_types::ExecutionStackRequirement::Process,
        }
    }

    fn invoke(&self, context: RuntimeContext, call: ForeignCall) -> ForeignAdapterFuture {
        if self.retirement_requested.get() || self.project_retirement.is_pending() {
            let error = invalid_conversion("Java selected adapter is retiring");
            return Box::pin(async move { ForeignCallCompletion::invoked(Err(error)) });
        }
        if matches!(call.symbol.as_str(), "status" | "configure") {
            let result = self
                .invoke_control_now(Some(&context), call)
                .and_then(|sequence| {
                    crate::execution::RootedValueSequence::in_context(sequence, &context)
                });
            return Box::pin(async move { ForeignCallCompletion::invoked(result) });
        }
        let Some(owner) = self.self_weak.upgrade() else {
            let error = invalid_conversion("Java adapter owner has ended");
            return Box::pin(async move { ForeignCallCompletion::invoked(Err(error)) });
        };
        Box::pin(async move { owner.invoke_in_process(context, call).await })
    }
}

fn string_argument(value: Option<Value>, label: &str) -> Result<String, RuntimeError> {
    match value {
        Some(value) => value_text_scalar_string(&value)
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| invalid_conversion(format!("{label} must be non-empty text"))),
        _ => Err(invalid_conversion(format!(
            "{label} must be non-empty text"
        ))),
    }
}

fn string_argument_ref(value: Option<&Value>, label: &str) -> Result<String, RuntimeError> {
    match value {
        Some(value) => value_text_scalar_string(value)
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| invalid_conversion(format!("{label} must be non-empty text"))),
        None => Err(invalid_conversion(format!(
            "{label} must be non-empty text"
        ))),
    }
}

fn java_runtime_error(error: runmat_java::JvmError) -> RuntimeError {
    build_runtime_error(error.to_string())
        .with_builtin("java")
        .with_identifier("RunMat:Java:RuntimeUnavailable")
        .build()
}

fn java_invocation_error(error: JavaInvocationError) -> RuntimeError {
    if matches!(&error, JavaInvocationError::InvalidQualifiedName) {
        return invalid_conversion(error.to_string());
    }
    let identifier = match &error {
        JavaInvocationError::Exception(_) => "RunMat:Java:Exception",
        JavaInvocationError::ClassNotFound(_) => "RunMat:Java:ClassNotFound",
        _ => "RunMat:Java:InvocationFailed",
    };
    build_runtime_error(error.to_string())
        .with_builtin("java")
        .with_identifier(identifier)
        .build()
}

#[cfg(test)]
#[path = "adapter/tests.rs"]
mod tests;
