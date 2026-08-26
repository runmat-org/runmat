use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use runmat_java::{
    discover_jvm, JavaDiscoveryRequest, JavaInvocationError, JavaObjectHandle, JavaSession,
    JavaValue, JvmConfig, JvmProcess, JAVA_ADAPTER_ID, JAVA_ADAPTER_VERSION,
};
use runmat_types::{
    CapabilityRequirement, ForeignAffinity, ForeignCapability, ForeignLifetime, ForeignOwnership,
    ForeignTypeIdentity,
};
use runmat_value::{ForeignRef, ForeignResourceKey, Value, WeakForeignRef};

use super::super::{
    foreign_error, ForeignAdapter, ForeignAdapterDescriptor, ForeignAdapterFuture,
    ForeignErrorKind, ForeignExecutionPolicy, ForeignHandleRegistry, ForeignHostRegistration,
    ForeignHostRelease, ForeignResourceMetadata,
};
use super::conversion::{invalid_conversion, scalar_from_java, value_to_java};
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
    session: RefCell<Option<JavaSession>>,
    discovery: JavaDiscoveryRequest,
    config: JvmConfig,
    released: Arc<ReleaseQueue>,
    resources: RefCell<BTreeMap<u64, JavaObjectHandle>>,
    java_to_foreign: RefCell<BTreeMap<JavaObjectHandle, WeakForeignRef>>,
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
        let released = Arc::new(ReleaseQueue::default());
        handles.register_host(ForeignHostRegistration {
            identity: host_identity.clone(),
            adapter: JAVA_ADAPTER_ID.into(),
            session_identity: host_identity.clone(),
            capabilities: BTreeSet::from([
                ForeignCapability::Invoke,
                ForeignCapability::Read,
                ForeignCapability::Write,
                ForeignCapability::Transfer,
            ]),
            policy: ForeignExecutionPolicy::trusted_in_process(),
            release: released.clone(),
        })?;
        Ok(Rc::new(Self {
            handles,
            host_identity,
            session: RefCell::new(None),
            discovery,
            config,
            released,
            resources: RefCell::new(BTreeMap::new()),
            java_to_foreign: RefCell::new(BTreeMap::new()),
        }))
    }

    pub fn is_running(&self) -> bool {
        self.session.borrow().is_some()
    }

    fn ensure_session(&self) -> Result<(), RuntimeError> {
        if self.session.borrow().is_some() {
            return Ok(());
        }
        let installation = discover_jvm(&self.discovery).map_err(java_runtime_error)?;
        let process = JvmProcess::launch(installation, &self.config).map_err(java_runtime_error)?;
        *self.session.borrow_mut() = Some(JavaSession::new(process));
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

    fn invoke_now(&self, call: ForeignCall) -> Result<Value, RuntimeError> {
        self.ensure_session()?;
        self.drain_releases()?;
        let mut arguments = call.arguments.into_iter();
        let result = match call.symbol.as_str() {
            "construct" => {
                let class = string_argument(arguments.next(), "Java class")?;
                let values = arguments
                    .map(|value| self.argument_to_java(value))
                    .collect::<Result<Vec<_>, _>>()?;
                self.with_session(|session| session.construct_resolved(&class, &values))?
            }
            "call_static" => {
                let class = string_argument(arguments.next(), "Java class")?;
                let method = string_argument(arguments.next(), "Java method")?;
                let values = arguments
                    .map(|value| self.argument_to_java(value))
                    .collect::<Result<Vec<_>, _>>()?;
                self.with_session(|session| session.call_static_resolved(&class, &method, &values))?
            }
            "invoke_member" => {
                let reference = foreign_argument(arguments.next())?;
                let method = string_argument(arguments.next(), "Java method")?;
                let java_handle = self.resolve_java_handle(&reference)?;
                let values = arguments
                    .map(|value| self.argument_to_java(value))
                    .collect::<Result<Vec<_>, _>>()?;
                self.with_session(|session| {
                    session.call_method_resolved(java_handle, &method, &values)
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
                let value = self.argument_to_java(arguments.next().ok_or_else(|| {
                    invalid_conversion("Java field assignment requires a value")
                })?)?;
                let java_handle = self.resolve_java_handle(&reference)?;
                self.with_session(|session| {
                    session.set_field_resolved(java_handle, &field, &value)
                })?;
                return Ok(Value::Foreign(reference));
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
        operation(session.as_ref().expect("session initialized")).map_err(java_invocation_error)
    }

    fn value_from_java(&self, value: JavaValue) -> Result<Value, RuntimeError> {
        let JavaValue::Object { handle, class_name } = value else {
            return scalar_from_java(value);
        };
        if let Some(reference) = self
            .java_to_foreign
            .borrow()
            .get(&handle)
            .and_then(WeakForeignRef::upgrade)
        {
            return Ok(Value::Foreign(reference));
        }
        let reference = self.handles.register_resource(
            &self.host_identity,
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
        self.resources.borrow_mut().insert(reference.handle, handle);
        if let Some(weak) = reference.downgrade() {
            self.java_to_foreign.borrow_mut().insert(handle, weak);
        }
        Ok(Value::Foreign(reference))
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

    fn argument_to_java(&self, value: Value) -> Result<JavaValue, RuntimeError> {
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
}

impl ForeignAdapter for JavaAdapter {
    fn descriptor(&self) -> ForeignAdapterDescriptor {
        ForeignAdapterDescriptor {
            adapter: JAVA_ADAPTER_ID.into(),
            version: JAVA_ADAPTER_VERSION,
            capabilities: BTreeSet::from([CapabilityRequirement::ForeignRuntime]),
            foreign_capabilities: BTreeSet::from([
                ForeignCapability::Invoke,
                ForeignCapability::Read,
                ForeignCapability::Write,
                ForeignCapability::Transfer,
            ]),
            artifact_identities: BTreeSet::new(),
            supports_wasm: false,
            supports_host_bridge: false,
        }
    }

    fn invoke(&self, _context: RuntimeContext, call: ForeignCall) -> ForeignAdapterFuture {
        let result = self.invoke_now(call);
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
