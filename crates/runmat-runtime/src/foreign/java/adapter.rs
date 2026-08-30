use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use runmat_java::{
    discover_jvm, ClasspathSnapshot, JavaArtifactIdentity, JavaCallbackInvocation,
    JavaDiscoveryRequest, JavaInvocationError, JavaObjectHandle, JavaSession, JavaValue, JvmConfig,
    JvmProcess, SessionClasspath, JAVA_ADAPTER_ID, JAVA_ADAPTER_VERSION,
};
use runmat_types::{
    CapabilityRequirement, ForeignAffinity, ForeignCapability, ForeignLifetime, ForeignOwnership,
    ForeignTypeIdentity,
};
use runmat_value::{
    CellArray, CharArray, ForeignRef, ForeignResourceKey, IntValue, ObjectInstance, Value,
    WeakForeignRef,
};

use super::super::{
    foreign_error, invoke_foreign_callback, ForeignAdapter, ForeignAdapterDescriptor,
    ForeignAdapterFuture, ForeignErrorKind, ForeignExecutionPolicy, ForeignHandleRegistry,
    ForeignHostRegistration, ForeignHostRelease, ForeignResourceMetadata,
};
use super::conversion::{array_from_java, invalid_conversion, scalar_from_java, value_to_java};
use crate::context::{ForeignCall, RuntimeContext};
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
    host_identity: String,
    session: RefCell<Option<Rc<JavaSession>>>,
    discovery: RefCell<JavaDiscoveryRequest>,
    config: RefCell<JvmConfig>,
    initial_classpath: RefCell<SessionClasspath>,
    released: Arc<ReleaseQueue>,
    resources: Rc<RefCell<BTreeMap<u64, JavaObjectHandle>>>,
    java_to_foreign: Rc<RefCell<BTreeMap<JavaObjectHandle, WeakForeignRef>>>,
    artifact_identities: RefCell<BTreeSet<String>>,
    desktop_available: Cell<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JavaRuntimeConfiguration {
    pub home: Option<std::path::PathBuf>,
    pub minimum_version: u16,
    pub maximum_version: Option<u16>,
    pub classpath: Vec<std::path::PathBuf>,
    pub options: Vec<String>,
}

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
        Ok(Rc::new(Self {
            handles,
            host_identity,
            session: RefCell::new(None),
            discovery: RefCell::new(discovery),
            config: RefCell::new(config),
            initial_classpath: RefCell::new(initial_classpath),
            released,
            resources: Rc::new(RefCell::new(BTreeMap::new())),
            java_to_foreign: Rc::new(RefCell::new(BTreeMap::new())),
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
        if self.is_running() {
            return Err(invalid_conversion(
                "Java runtime configuration is immutable after JVM startup",
            ));
        }
        let mut discovery = JavaDiscoveryRequest::from_process();
        discovery.explicit_home = configuration.home;
        let config = JvmConfig {
            minimum_major: configuration.minimum_version,
            maximum_major: configuration.maximum_version,
            bootstrap_classpath: Vec::new(),
            options: configuration.options,
        };
        config.validate().map_err(java_runtime_error)?;
        let classpath =
            SessionClasspath::new(config.bootstrap_classpath.clone(), configuration.classpath)
                .map_err(|error| invalid_conversion(error.to_string()))?;
        *self.discovery.borrow_mut() = discovery;
        *self.config.borrow_mut() = config;
        *self.initial_classpath.borrow_mut() = classpath;
        Ok(())
    }

    pub fn install_project_artifacts(
        &self,
        artifacts: &[(JavaArtifactIdentity, std::path::PathBuf)],
    ) -> Result<(), RuntimeError> {
        if self.is_running() {
            return Err(invalid_conversion(
                "Java project artifacts must be installed before JVM startup",
            ));
        }
        let mut identities = BTreeSet::new();
        let mut paths = self.initial_classpath.borrow().snapshot().project;
        for (identity, path) in artifacts {
            identity
                .validate_file(path)
                .map_err(|error| invalid_conversion(error.to_string()))?;
            if !identities.insert(identity.to_string()) {
                return Err(invalid_conversion(format!(
                    "Java artifact identity `{identity}` appears more than once"
                )));
            }
            if !paths.contains(path) {
                paths.push(path.clone());
            }
        }
        self.initial_classpath
            .borrow_mut()
            .replace_project(paths)
            .map_err(|error| invalid_conversion(error.to_string()))?;
        *self.artifact_identities.borrow_mut() = identities;
        Ok(())
    }

    fn ensure_session(&self) -> Result<(), RuntimeError> {
        if self.session.borrow().is_some() {
            return Ok(());
        }
        let installation = discover_jvm(&self.discovery.borrow()).map_err(java_runtime_error)?;
        let process =
            JvmProcess::launch(installation, &self.config.borrow()).map_err(java_runtime_error)?;
        *self.session.borrow_mut() = Some(Rc::new(JavaSession::with_classpath(
            process,
            self.initial_classpath.borrow().clone(),
        )));
        Ok(())
    }

    fn drain_releases(&self) -> Result<(), RuntimeError> {
        let handles = {
            let mut queue = self.released.0.lock().map_err(|_| {
                foreign_error(
                    ForeignErrorKind::HostUnavailable,
                    "Java release queue lock was poisoned",
                )
            })?;
            std::mem::take(&mut *queue)
        };
        let session = self.session.borrow();
        let Some(session) = session.as_ref() else {
            return Ok(());
        };
        for foreign_handle in handles {
            if let Some(java_handle) = self.resources.borrow_mut().remove(&foreign_handle) {
                self.java_to_foreign.borrow_mut().remove(&java_handle);
                let _ = session.release(java_handle);
            }
        }
        Ok(())
    }

    fn invoke_now(
        &self,
        context: &RuntimeContext,
        call: ForeignCall,
    ) -> Result<Value, RuntimeError> {
        if call.symbol == "classpath" {
            return self.classpath_value(call.arguments);
        }
        if call.symbol == "has_classpath" {
            let snapshot = self
                .session
                .borrow()
                .as_ref()
                .map(|session| session.classpath())
                .unwrap_or_else(|| self.initial_classpath.borrow().snapshot());
            return Ok(Value::Bool(
                !snapshot.bootstrap.is_empty()
                    || !snapshot.project.is_empty()
                    || !snapshot.dynamic.is_empty(),
            ));
        }
        if call.symbol == "status" {
            return self.status_value();
        }
        if call.symbol == "usejava" {
            return self.usejava_value(call.arguments);
        }
        if call.symbol == "configure" {
            return self.configure_value(call.arguments);
        }
        if matches!(
            call.symbol.as_str(),
            "construct_edt" | "call_static_edt" | "invoke_member_edt"
        ) && !self.desktop_available.get()
        {
            return Err(build_runtime_error(
                "Java EDT execution requires a Desktop host with Java UI support",
            )
            .with_builtin("java")
            .with_identifier("RunMat:Java:EdtUnavailable")
            .build());
        }
        self.ensure_session()?;
        self.drain_releases()?;
        let mut arguments = call.arguments.into_iter();
        let result = match call.symbol.as_str() {
            "construct" => {
                let class = string_argument(arguments.next(), "Java class")?;
                let values = arguments
                    .map(|value| self.argument_to_java(context, value))
                    .collect::<Result<Vec<_>, _>>()?;
                self.with_session(|session| session.construct_resolved(&class, &values))?
            }
            "construct_edt" => {
                let class = string_argument(arguments.next(), "Java class")?;
                let values = arguments
                    .map(|value| self.argument_to_java(context, value))
                    .collect::<Result<Vec<_>, _>>()?;
                self.with_session(|session| session.construct_resolved_on_edt(&class, &values))?
            }
            "call_static" => {
                let class = string_argument(arguments.next(), "Java class")?;
                let method = string_argument(arguments.next(), "Java method")?;
                let values = arguments
                    .map(|value| self.argument_to_java(context, value))
                    .collect::<Result<Vec<_>, _>>()?;
                self.with_session(|session| session.call_static_resolved(&class, &method, &values))?
            }
            "call_static_edt" => {
                let class = string_argument(arguments.next(), "Java class")?;
                let method = string_argument(arguments.next(), "Java method")?;
                let values = arguments
                    .map(|value| self.argument_to_java(context, value))
                    .collect::<Result<Vec<_>, _>>()?;
                self.with_session(|session| {
                    session.call_static_resolved_on_edt(&class, &method, &values)
                })?
            }
            "invoke_qualified" => {
                let name = string_argument(arguments.next(), "qualified Java name")?;
                let values = arguments
                    .map(|value| self.argument_to_java(context, value))
                    .collect::<Result<Vec<_>, _>>()?;
                if self.with_session(|session| session.class_exists(&name))? {
                    self.with_session(|session| session.construct_resolved(&name, &values))?
                } else {
                    let (class, method) = name.rsplit_once('.').ok_or_else(|| {
                        invalid_conversion("qualified Java call requires a dotted class name")
                    })?;
                    if !self.with_session(|session| session.class_exists(class))? {
                        return Err(build_runtime_error(format!(
                            "Java class {class} was not found"
                        ))
                        .with_builtin("java")
                        .with_identifier("RunMat:Java:ClassNotFound")
                        .build());
                    }
                    self.with_session(|session| {
                        session.call_static_resolved(class, method, &values)
                    })?
                }
            }
            "invoke_member" => {
                let reference = foreign_argument(arguments.next())?;
                let method = string_argument(arguments.next(), "Java method")?;
                let java_handle = self.resolve_java_handle(&reference)?;
                let values = arguments
                    .map(|value| self.argument_to_java(context, value))
                    .collect::<Result<Vec<_>, _>>()?;
                self.with_session(|session| {
                    session.call_method_resolved(java_handle, &method, &values)
                })?
            }
            "invoke_member_edt" => {
                let reference = foreign_argument(arguments.next())?;
                let method = string_argument(arguments.next(), "Java method")?;
                let java_handle = self.resolve_java_handle(&reference)?;
                let values = arguments
                    .map(|value| self.argument_to_java(context, value))
                    .collect::<Result<Vec<_>, _>>()?;
                self.with_session(|session| {
                    session.call_method_resolved_on_edt(java_handle, &method, &values)
                })?
            }
            "get_member" => {
                let reference = foreign_argument(arguments.next())?;
                let field = string_argument(arguments.next(), "Java field")?;
                let java_handle = self.resolve_java_handle(&reference)?;
                self.with_session(|session| session.get_field_resolved(java_handle, &field))?
            }
            "set_member" => {
                let reference = foreign_argument(arguments.next())?;
                let field = string_argument(arguments.next(), "Java field")?;
                let value = self.argument_to_java(
                    context,
                    arguments.next().ok_or_else(|| {
                        invalid_conversion("Java field assignment requires a value")
                    })?,
                )?;
                let java_handle = self.resolve_java_handle(&reference)?;
                self.with_session(|session| {
                    session.set_field_resolved(java_handle, &field, &value)
                })?;
                return Ok(Value::Foreign(reference));
            }
            "add_classpath" => {
                let entry = string_argument(arguments.next(), "Java classpath entry")?;
                let position = arguments
                    .next()
                    .map(|value| string_argument(Some(value), "Java classpath position"))
                    .transpose()?;
                let at_end = match position.as_deref() {
                    None | Some("begin") => false,
                    Some("end") => true,
                    Some(_) => {
                        return Err(invalid_conversion(
                            "Java classpath position must be 'begin' or 'end'",
                        ))
                    }
                };
                self.with_session(|session| session.add_dynamic_classpath_at(entry, at_end))?;
                return Ok(Value::OutputList(Vec::new()));
            }
            "remove_classpath" => {
                let entry = string_argument(arguments.next(), "Java classpath entry")?;
                self.with_session(|session| {
                    session.remove_dynamic_classpath(std::path::Path::new(&entry))
                })?;
                return Ok(Value::OutputList(Vec::new()));
            }
            "set_classpath" => {
                let entries = arguments
                    .map(|value| string_argument(Some(value), "Java classpath entry"))
                    .collect::<Result<Vec<_>, _>>()?;
                self.with_session(|session| {
                    session.replace_dynamic_classpath(
                        entries.into_iter().map(std::path::PathBuf::from),
                    )
                })?;
                return Ok(Value::OutputList(Vec::new()));
            }
            "new_array" => {
                let class = string_argument(arguments.next(), "Java array class")?;
                let dimensions = arguments
                    .map(java_dimension)
                    .collect::<Result<Vec<_>, _>>()?;
                self.with_session(|session| session.new_object_array(&class, &dimensions))?
            }
            operation => {
                return Err(foreign_error(
                    ForeignErrorKind::InvalidCall,
                    format!("unsupported Java operation {operation}"),
                ))
            }
        };
        self.value_from_java(result)
    }

    fn with_session<T>(
        &self,
        operation: impl FnOnce(&JavaSession) -> Result<T, JavaInvocationError>,
    ) -> Result<T, RuntimeError> {
        let session = self.session.borrow();
        operation(session.as_deref().expect("session initialized")).map_err(java_invocation_error)
    }

    fn with_session_rc<T>(
        &self,
        operation: impl FnOnce(&Rc<JavaSession>) -> Result<T, JavaInvocationError>,
    ) -> Result<T, RuntimeError> {
        let session = self.session.borrow();
        operation(session.as_ref().expect("session initialized")).map_err(java_invocation_error)
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
        value: Value,
    ) -> Result<JavaValue, RuntimeError> {
        if super::super::is_callable(&value) {
            let callback = value;
            let foreign = context
                .service_ports()
                .foreign()
                .map(Rc::downgrade)
                .ok_or_else(|| invalid_conversion("Java callback requires a foreign runtime"))?;
            let callback_context = context
                .clone()
                .with_service_ports(context.service_ports().clone().without_foreign());
            let handles = self.handles.clone();
            let host_identity = self.host_identity.clone();
            let resources = Rc::clone(&self.resources);
            let java_to_foreign = Rc::clone(&self.java_to_foreign);
            return self.with_session_rc(|session| {
                session.register_callback(move |invocation: JavaCallbackInvocation| {
                    let foreign = foreign.upgrade().ok_or_else(|| {
                        JavaInvocationError::Callback(
                            "callback's originating foreign runtime has ended".into(),
                        )
                    })?;
                    let context = callback_context.clone().with_service_ports(
                        callback_context
                            .service_ports()
                            .clone()
                            .with_foreign(foreign),
                    );
                    let arguments = invocation
                        .arguments
                        .into_iter()
                        .map(|value| {
                            value_from_java_parts(
                                &handles,
                                &host_identity,
                                &resources,
                                &java_to_foreign,
                                value,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|error| JavaInvocationError::Callback(error.to_string()))?;
                    let requested_outputs = usize::from(invocation.returns_value);
                    let request = super::super::foreign_callback_request(
                        &callback,
                        arguments,
                        requested_outputs,
                    )
                    .map_err(|error| JavaInvocationError::Callback(error.to_string()))?;
                    let result = pollster::block_on(invoke_foreign_callback(context, request))
                        .map_err(|error| JavaInvocationError::Callback(error.to_string()))?;
                    if !invocation.returns_value {
                        return Ok(JavaValue::Null);
                    }
                    value_to_java(result)
                        .map_err(|error| JavaInvocationError::Callback(error.to_string()))
                })
            });
        }
        let Value::Foreign(reference) = value else {
            return value_to_java(value);
        };
        if reference.type_identity.family != JAVA_ADAPTER_ID {
            return Err(invalid_conversion("foreign argument is not a Java object"));
        }
        let handle = self.resolve_java_handle(&reference)?;
        Ok(JavaValue::Object {
            handle,
            class_name: reference.type_identity.name.clone(),
        })
    }

    fn classpath_value(&self, arguments: Vec<Value>) -> Result<Value, RuntimeError> {
        let layer = arguments
            .into_iter()
            .next()
            .map(|value| string_argument(Some(value), "Java classpath layer"))
            .transpose()?
            .unwrap_or_else(|| "all".into())
            .to_ascii_lowercase();
        let snapshot = self
            .session
            .borrow()
            .as_ref()
            .map(|session| session.classpath())
            .unwrap_or_else(|| self.initial_classpath.borrow().snapshot());
        let paths = classpath_paths(&snapshot, &layer)?;
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

    fn status_value(&self) -> Result<Value, RuntimeError> {
        let running = self.session.borrow().is_some();
        let installation = if let Some(session) = self.session.borrow().as_ref() {
            Some(session.installation().clone())
        } else {
            discover_jvm(&self.discovery.borrow()).ok()
        };
        let mut status = ObjectInstance::new("matlab.javaclient.JavaEnvironment".into());
        status.properties.insert(
            "Version".into(),
            Value::String(
                installation
                    .as_ref()
                    .map(|value| value.version.raw.clone())
                    .unwrap_or_default(),
            ),
        );
        status.properties.insert(
            "Home".into(),
            Value::String(
                installation
                    .as_ref()
                    .map(|value| value.home.display().to_string())
                    .unwrap_or_default(),
            ),
        );
        status.properties.insert(
            "Library".into(),
            Value::String(
                installation
                    .as_ref()
                    .map(|value| value.library.display().to_string())
                    .unwrap_or_default(),
            ),
        );
        status.properties.insert(
            "Status".into(),
            Value::String(if running { "loaded" } else { "notloaded" }.into()),
        );
        status.properties.insert(
            "Configuration".into(),
            Value::String({
                let discovery = self.discovery.borrow();
                discovery
                    .explicit_home
                    .as_ref()
                    .or(discovery.environment_home.as_ref())
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "system".into())
            }),
        );
        Ok(Value::Object(status))
    }

    fn usejava_value(&self, arguments: Vec<Value>) -> Result<Value, RuntimeError> {
        let feature =
            string_argument(arguments.into_iter().next(), "Java feature")?.to_ascii_lowercase();
        match feature.as_str() {
            "jvm" => Ok(Value::Bool(
                self.session.borrow().is_some() || discover_jvm(&self.discovery.borrow()).is_ok(),
            )),
            "desktop" => Ok(Value::Bool(self.desktop_available.get())),
            "awt" | "swing" => {
                self.ensure_session()?;
                let headless = self.with_session(|session| {
                    session.call_static_resolved("java.awt.GraphicsEnvironment", "isHeadless", &[])
                })?;
                let JavaValue::Boolean(headless) = headless else {
                    return Err(invalid_conversion(
                        "Java graphics environment returned an invalid capability value",
                    ));
                };
                Ok(Value::Bool(!headless))
            }
            _ => Err(invalid_conversion(
                "Java feature must be 'jvm', 'awt', 'swing', or 'desktop'",
            )),
        }
    }

    fn configure_value(&self, arguments: Vec<Value>) -> Result<Value, RuntimeError> {
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
        let mut arguments = arguments.into_iter();
        let name = string_argument(arguments.next(), "jenv option")?.to_ascii_lowercase();
        let value = string_argument(arguments.next(), "jenv option value")?;
        match name.as_str() {
            "version" => {
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
            "executionmode" => {
                if !value.eq_ignore_ascii_case("inprocess") {
                    return Err(invalid_conversion(
                        "RunMat currently supports Java execution mode 'InProcess'",
                    ));
                }
            }
            _ => {
                return Err(invalid_conversion(
                    "jenv supports the 'Version' and 'ExecutionMode' options",
                ))
            }
        }
        self.status_value()
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
            return CellArray::new(
                elements
                    .into_iter()
                    .map(|value| {
                        value_from_java_parts(
                            handles,
                            host_identity,
                            resources,
                            java_to_foreign,
                            value,
                        )
                    })
                    .collect::<Result<_, _>>()?,
                length,
                1,
            )
            .map(Value::Cell)
            .map_err(invalid_conversion);
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

fn classpath_paths<'a>(
    snapshot: &'a ClasspathSnapshot,
    layer: &str,
) -> Result<Vec<&'a std::path::PathBuf>, RuntimeError> {
    Ok(match layer {
        "all" => snapshot
            .bootstrap
            .iter()
            .chain(snapshot.project.iter())
            .chain(snapshot.dynamic.iter())
            .collect(),
        "static" | "bootstrap" => snapshot
            .bootstrap
            .iter()
            .chain(snapshot.project.iter())
            .collect(),
        "dynamic" => snapshot.dynamic.iter().collect(),
        _ => {
            return Err(invalid_conversion(
                "Java classpath layer must be 'all', 'static', or 'dynamic'",
            ))
        }
    })
}

impl ForeignAdapter for JavaAdapter {
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
                    runmat_types::ForeignArtifactIdentity::new(identity.clone())
                        .expect("installed Java artifact identities are canonical")
                })
                .collect(),
            supports_wasm: false,
            supports_host_bridge: false,
            execution_stack: runmat_types::ExecutionStackRequirement::Process,
        }
    }

    fn invoke(&self, context: RuntimeContext, call: ForeignCall) -> ForeignAdapterFuture {
        let result = self.invoke_now(&context, call);
        Box::pin(async move { result })
    }
}

fn string_argument(value: Option<Value>, label: &str) -> Result<String, RuntimeError> {
    match value {
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(value),
        _ => Err(invalid_conversion(format!(
            "{label} must be non-empty text"
        ))),
    }
}

fn foreign_argument(value: Option<Value>) -> Result<ForeignRef, RuntimeError> {
    match value {
        Some(Value::Foreign(reference)) if reference.type_identity.family == JAVA_ADAPTER_ID => {
            Ok(reference)
        }
        _ => Err(invalid_conversion("Java receiver must be a Java object")),
    }
}

fn java_dimension(value: Value) -> Result<usize, RuntimeError> {
    match value {
        Value::Int(value) => match value {
            IntValue::I8(value) => usize::try_from(value),
            IntValue::I16(value) => usize::try_from(value),
            IntValue::I32(value) => usize::try_from(value),
            IntValue::I64(value) => usize::try_from(value),
            IntValue::U8(value) => Ok(usize::from(value)),
            IntValue::U16(value) => Ok(usize::from(value)),
            IntValue::U32(value) => usize::try_from(value),
            IntValue::U64(value) => usize::try_from(value),
        }
        .map_err(|_| invalid_conversion("Java array dimensions must be nonnegative integers")),
        Value::Num(value)
            if value.is_finite()
                && value >= 0.0
                && value.fract() == 0.0
                && value <= usize::MAX as f64 =>
        {
            Ok(value as usize)
        }
        _ => Err(invalid_conversion(
            "Java array dimensions must be nonnegative integer scalars",
        )),
    }
}

fn java_runtime_error(error: runmat_java::JvmError) -> RuntimeError {
    build_runtime_error(error.to_string())
        .with_builtin("java")
        .with_identifier("RunMat:Java:RuntimeUnavailable")
        .build()
}

fn java_invocation_error(error: JavaInvocationError) -> RuntimeError {
    let identifier = if matches!(error, JavaInvocationError::Exception(_)) {
        "RunMat:Java:Exception"
    } else {
        "RunMat:Java:InvocationFailed"
    };
    build_runtime_error(error.to_string())
        .with_builtin("java")
        .with_identifier(identifier)
        .build()
}
