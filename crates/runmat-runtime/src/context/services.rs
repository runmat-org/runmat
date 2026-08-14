use super::{RuntimeCapability, RuntimeCapabilityError};
use crate::class_registry::RuntimeClass;
use crate::warning_store::RuntimeWarning;
use crate::RuntimeError;
use runmat_types::{CallableIdentity, SourceId};
use runmat_value::Value;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;

pub type RuntimeServiceFuture<T> = Pin<Box<dyn Future<Output = T> + 'static>>;

#[derive(Debug, Clone)]
pub struct RuntimeCallRequest {
    pub identity: CallableIdentity,
    pub arguments: Vec<Value>,
    pub requested_outputs: usize,
}

pub trait RuntimeCallService {
    fn resolve(&self, name: &str) -> Option<usize>;

    fn invoke(
        &self,
        request: RuntimeCallRequest,
    ) -> RuntimeServiceFuture<Result<Value, RuntimeError>>;

    fn source_functions(&self, _source_id: SourceId) -> Vec<(String, usize)> {
        Vec::new()
    }
}

pub trait RuntimeWorkspaceService {
    fn lookup(&self, name: &str) -> Option<Value>;
    fn snapshot(&self) -> Vec<(String, Value)>;
    fn global_names(&self) -> Vec<String>;
    fn assign(&self, name: &str, value: Value) -> Result<(), RuntimeError>;
    fn clear(&self) -> Result<(), RuntimeError>;
    fn remove(&self, name: &str) -> Result<(), RuntimeError>;
}

pub trait RuntimeObjectService {
    fn class(&self, name: &str) -> Option<RuntimeClass>;
    fn register_class(&self, class: RuntimeClass) -> Result<(), RuntimeError>;
    fn static_property(&self, class: &str, property: &str) -> Option<Value>;
    fn set_static_property(
        &self,
        class: &str,
        property: &str,
        value: Value,
    ) -> Result<(), RuntimeError>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostInteraction {
    Line { prompt: String, echo: bool },
    KeyPress { prompt: String },
}

pub trait RuntimeHostService {
    fn console(&self, stream: crate::console::ConsoleStream, text: String);

    fn interact(
        &self,
        request: HostInteraction,
    ) -> RuntimeServiceFuture<Result<crate::interaction::InteractionResponse, RuntimeError>>;

    fn warning(&self, warning: RuntimeWarning);
}

pub trait RuntimeErrorService {
    fn report(&self, error: &RuntimeError);
}

pub trait RuntimeAccelerationService {
    fn supports_operation(&self, operation: &str) -> bool;
}

/// Session-owned execution-placement authority. The runtime exposes only
/// executor-neutral contracts; candidate generation and policy remain in their
/// owning executor/acceleration crates.
pub trait RuntimePlacementService {
    fn plan(
        &self,
        request: runmat_execution::PlacementPlanRequest,
    ) -> Result<runmat_execution::PlacementDecision, RuntimeError>;

    fn observe(&self, feedback: runmat_execution::PlacementFeedback) -> Result<(), RuntimeError>;

    fn invalidate(&self, invalidation: runmat_execution::PlacementInvalidation);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeCapability {
    SharedLibrary,
    ExecutableMemory,
    ObjectEmission,
}

pub trait RuntimeNativeService {
    fn supports(&self, capability: NativeCapability) -> bool;
}

#[derive(Debug, Clone)]
pub struct ForeignCall {
    pub adapter: String,
    pub symbol: String,
    pub arguments: Vec<Value>,
    pub requested_outputs: usize,
}

pub trait RuntimeForeignService {
    fn invoke(&self, call: ForeignCall) -> RuntimeServiceFuture<Result<Value, RuntimeError>>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelCapability {
    Pool,
    Parfor,
    Spmd,
    DistributedValues,
    Collectives,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeParallelResources {
    pub cpu_millicores_available: u32,
    pub memory_available_bytes: Option<u64>,
    pub epoch: u64,
}

impl Default for RuntimeParallelResources {
    fn default() -> Self {
        Self {
            cpu_millicores_available: 1_000,
            memory_available_bytes: None,
            epoch: 0,
        }
    }
}

pub trait RuntimeParallelService {
    fn supports(&self, capability: ParallelCapability) -> bool;

    /// Side-effect-free scheduler capacity for placement admission. A future
    /// RM-1067 pool/scheduler adapter overrides this with its current lease;
    /// absence preserves the single-core local runtime budget.
    fn placement_resources(&self) -> RuntimeParallelResources {
        RuntimeParallelResources::default()
    }
}

/// Narrow, typed ports composed by the host. An absent port is meaningful and
/// produces a stable capability error through the corresponding `require_*`
/// accessor; there is no string-keyed service locator.
#[derive(Clone, Default)]
pub struct RuntimeServicePorts {
    call: Option<Rc<dyn RuntimeCallService>>,
    workspace: Option<Rc<dyn RuntimeWorkspaceService>>,
    object: Option<Rc<dyn RuntimeObjectService>>,
    host: Option<Rc<dyn RuntimeHostService>>,
    error: Option<Rc<dyn RuntimeErrorService>>,
    acceleration: Option<Rc<dyn RuntimeAccelerationService>>,
    placement: Option<Rc<dyn RuntimePlacementService>>,
    native: Option<Rc<dyn RuntimeNativeService>>,
    foreign: Option<Rc<dyn RuntimeForeignService>>,
    parallel: Option<Rc<dyn RuntimeParallelService>>,
}

impl std::fmt::Debug for RuntimeServicePorts {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RuntimeServicePorts")
            .field("call", &self.call.is_some())
            .field("workspace", &self.workspace.is_some())
            .field("object", &self.object.is_some())
            .field("host", &self.host.is_some())
            .field("error", &self.error.is_some())
            .field("acceleration", &self.acceleration.is_some())
            .field("placement", &self.placement.is_some())
            .field("native", &self.native.is_some())
            .field("foreign", &self.foreign.is_some())
            .field("parallel", &self.parallel.is_some())
            .finish()
    }
}

macro_rules! port_accessors {
    ($with:ident, $get:ident, $require:ident, $field:ident, $trait_name:ident, $cap:ident) => {
        pub fn $with(mut self, service: Rc<dyn $trait_name>) -> Self {
            self.$field = Some(service);
            self
        }

        pub fn $get(&self) -> Option<&Rc<dyn $trait_name>> {
            self.$field.as_ref()
        }

        pub fn $require(
            &self,
            operation: impl Into<String>,
        ) -> Result<&Rc<dyn $trait_name>, RuntimeCapabilityError> {
            self.$field
                .as_ref()
                .ok_or_else(|| RuntimeCapabilityError::new(RuntimeCapability::$cap, operation))
        }
    };
}

impl RuntimeServicePorts {
    port_accessors!(
        with_call,
        call,
        require_call,
        call,
        RuntimeCallService,
        Call
    );
    port_accessors!(
        with_workspace,
        workspace,
        require_workspace,
        workspace,
        RuntimeWorkspaceService,
        Workspace
    );
    port_accessors!(
        with_object,
        object,
        require_object,
        object,
        RuntimeObjectService,
        Object
    );
    port_accessors!(
        with_host,
        host,
        require_host,
        host,
        RuntimeHostService,
        Host
    );
    port_accessors!(
        with_error,
        error,
        require_error,
        error,
        RuntimeErrorService,
        Error
    );
    port_accessors!(
        with_acceleration,
        acceleration,
        require_acceleration,
        acceleration,
        RuntimeAccelerationService,
        Acceleration
    );
    port_accessors!(
        with_placement,
        placement,
        require_placement,
        placement,
        RuntimePlacementService,
        Placement
    );
    port_accessors!(
        with_native,
        native,
        require_native,
        native,
        RuntimeNativeService,
        Native
    );
    port_accessors!(
        with_foreign,
        foreign,
        require_foreign,
        foreign,
        RuntimeForeignService,
        Foreign
    );
    port_accessors!(
        with_parallel,
        parallel,
        require_parallel,
        parallel,
        RuntimeParallelService,
        Parallel
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_ports_report_stable_typed_capabilities() {
        fn missing<T>(result: Result<T, RuntimeCapabilityError>) -> RuntimeCapabilityError {
            match result {
                Ok(_) => panic!("expected absent runtime port"),
                Err(error) => error,
            }
        }
        let ports = RuntimeServicePorts::default();
        let failures = [
            missing(ports.require_call("invoke")),
            missing(ports.require_workspace("lookup")),
            missing(ports.require_object("class lookup")),
            missing(ports.require_host("console write")),
            missing(ports.require_error("report error")),
            missing(ports.require_acceleration("resident operation")),
            missing(ports.require_placement("placement plan")),
            missing(ports.require_native("load library")),
            missing(ports.require_foreign("foreign call")),
            missing(ports.require_parallel("parfor")),
        ];
        assert_eq!(
            failures
                .iter()
                .map(|failure| failure.capability)
                .collect::<Vec<_>>(),
            vec![
                RuntimeCapability::Call,
                RuntimeCapability::Workspace,
                RuntimeCapability::Object,
                RuntimeCapability::Host,
                RuntimeCapability::Error,
                RuntimeCapability::Acceleration,
                RuntimeCapability::Placement,
                RuntimeCapability::Native,
                RuntimeCapability::Foreign,
                RuntimeCapability::Parallel,
            ]
        );
        assert!(failures.iter().all(|failure| {
            failure.clone().into_runtime_error().identifier()
                == Some(RuntimeCapabilityError::IDENTIFIER)
        }));
    }
}
