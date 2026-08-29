use super::{RuntimeCapability, RuntimeCapabilityError};
use crate::builtin::RuntimeBuiltinBinding;
use crate::class_registry::RuntimeClass;
use crate::warning_store::RuntimeWarning;
use crate::RuntimeError;
use runmat_types::{
    BuiltinId, CallableFallbackPolicy, CallableIdentity, DistributedValueContract,
    DistributionScheme, LabRank, SourceId,
};
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

/// Canonical call router for callbacks that re-enter the active RunMat
/// executor. The executor installs its semantic and external invokers in the
/// invocation context; this service keeps foreign adapters independent of VM,
/// JIT, and AOT implementation details.
#[derive(Debug, Default)]
pub struct RuntimeCallRouter;

impl RuntimeCallService for RuntimeCallRouter {
    fn resolve(&self, name: &str) -> Option<usize> {
        crate::user_functions::resolve_semantic_function_by_name(name)
    }

    fn invoke(
        &self,
        request: RuntimeCallRequest,
    ) -> RuntimeServiceFuture<Result<Value, RuntimeError>> {
        let fallback_policy = match &request.identity {
            CallableIdentity::DynamicName(_)
            | CallableIdentity::Imported(_)
            | CallableIdentity::Method(_) => CallableFallbackPolicy::RuntimeNameResolution,
            CallableIdentity::ExternalName(_) => CallableFallbackPolicy::ExternalBoundary,
            _ => CallableFallbackPolicy::None,
        };
        Box::pin(async move {
            crate::call::descriptor::execute_callable_descriptor(
                crate::call::descriptor::CallableDescriptor::resolved(
                    request.identity,
                    request.arguments,
                    request.requested_outputs,
                    fallback_policy,
                    crate::call::descriptor::CallableCallKind::Direct,
                ),
            )
            .await
        })
    }

    fn source_functions(&self, source_id: SourceId) -> Vec<(String, usize)> {
        crate::user_functions::source_functions_for(source_id)
            .into_iter()
            .map(|function| (function.name, function.function))
            .collect()
    }
}

/// Invocation-scoped builtin authority used by exact compiled products.
/// When installed, the dispatcher must not fall back to process-global
/// discovery for a missing name.
pub trait RuntimeBuiltinService {
    fn bindings_by_name(&self, name: &str) -> Vec<RuntimeBuiltinBinding>;
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
    fn execution_stack_requirement(&self) -> runmat_types::ExecutionStackRequirement {
        runmat_types::ExecutionStackRequirement::Any
    }

    fn invoke(
        &self,
        context: super::RuntimeContext,
        call: ForeignCall,
    ) -> RuntimeServiceFuture<Result<Value, RuntimeError>>;
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

#[derive(Clone)]
pub struct RuntimeSpmdAdmission {
    pub gang: runmat_execution::GangSnapshot,
    pub labs: Vec<Rc<dyn RuntimeCollectiveService>>,
}

#[derive(Debug, Clone)]
pub struct RuntimeSpmdOutput {
    pub value: runmat_types::RegionValueId,
    pub fact: runmat_types::ValueFact,
    pub entries: Vec<Option<runmat_execution::value::ValuePayload>>,
}

impl RuntimeSpmdAdmission {
    pub fn validate(&self) -> Result<(), RuntimeError> {
        self.gang.validate().map_err(|error| {
            crate::runtime_error::semantic_error(
                "RunMat:parallel:InvalidGangAdmission",
                error.to_string(),
            )
        })?;
        if self.labs.len() != self.gang.handle.labs.0 as usize
            || self
                .labs
                .iter()
                .zip(&self.gang.ranks)
                .any(|(service, rank)| {
                    service.context().gang != self.gang.handle || service.context().rank != *rank
                })
        {
            return Err(crate::runtime_error::semantic_error(
                "RunMat:parallel:InvalidGangAdmission",
                "SPMD admission must provide one ordered collective context per stable lab rank",
            ));
        }
        Ok(())
    }
}

/// Host-owned admission and lifecycle authority for one SPMD gang. Collective
/// services are scoped per lab while the coordinator remains shared.
pub trait RuntimeSpmdService {
    fn admit(
        &self,
        request: runmat_execution::GangRequest,
        available_labs: runmat_types::LabCount,
        region: runmat_types::ParallelRegionId,
    ) -> RuntimeServiceFuture<Result<RuntimeSpmdAdmission, RuntimeError>>;

    fn retire(
        &self,
        gang: runmat_execution::GangHandle,
    ) -> RuntimeServiceFuture<Result<(), RuntimeError>>;

    fn retain_outputs(
        &self,
        gang: runmat_execution::GangHandle,
        region: runmat_types::ParallelRegionId,
        outputs: Vec<RuntimeSpmdOutput>,
    ) -> RuntimeServiceFuture<Result<Vec<runmat_execution::CompositeHandle>, RuntimeError>>;
}

#[derive(Debug, Clone)]
pub struct RuntimeDistributedCallRequest {
    pub builtin: BuiltinId,
    pub arguments: Vec<Value>,
    pub requested_outputs: usize,
    pub output: runmat_types::ValueFact,
}

/// Live distributed metadata visible to language/runtime consumers. Payloads
/// remain behind the execution-owned store; unlike a transport snapshot this
/// record never invents object references for locally retained partitions.
#[derive(Debug, Clone)]
pub struct RuntimeDistributedSnapshot {
    pub handle: runmat_execution::DistributedValueHandle,
    pub partitions: Vec<runmat_execution::DistributedPartitionLayout>,
}

/// Runtime operations over execution-service-owned distributed values.
/// Implementations preserve partition ownership and decide locality; callers
/// never obtain raw partition storage through this port.
pub trait RuntimeDistributedService {
    fn create(
        &self,
        contract: DistributedValueContract,
        input: Value,
        pool: runmat_execution::PoolSnapshot,
    ) -> RuntimeServiceFuture<Result<runmat_execution::DistributedValueHandle, RuntimeError>>;

    fn inspect(
        &self,
        handle: runmat_execution::DistributedValueHandle,
    ) -> RuntimeServiceFuture<Result<RuntimeDistributedSnapshot, RuntimeError>>;

    fn local_part(
        &self,
        handle: runmat_execution::DistributedValueHandle,
        rank: LabRank,
    ) -> RuntimeServiceFuture<Result<Value, RuntimeError>>;

    fn materialize(
        &self,
        handle: runmat_execution::DistributedValueHandle,
    ) -> RuntimeServiceFuture<Result<Value, RuntimeError>>;

    fn redistribute(
        &self,
        handle: runmat_execution::DistributedValueHandle,
        scheme: DistributionScheme,
    ) -> RuntimeServiceFuture<Result<runmat_execution::DistributedValueHandle, RuntimeError>>;

    fn invoke(
        &self,
        request: RuntimeDistributedCallRequest,
    ) -> RuntimeServiceFuture<Result<Value, RuntimeError>>;

    fn composite_entry(
        &self,
        handle: runmat_execution::CompositeHandle,
        rank: LabRank,
    ) -> RuntimeServiceFuture<Result<Option<Value>, RuntimeError>>;
}

/// SPMD worker communication. The installed service owns the current rank and
/// transports typed collective payloads through the admitted gang.
pub trait RuntimeCollectiveService {
    fn context(&self) -> &runmat_execution::SpmdTaskContext;

    /// Allocate the next invocation sequence for one compiler-owned call site.
    /// Sequence state is scoped to the lab service, so repeated calls in loops
    /// cannot collide while every rank retains the same deterministic order.
    fn next_sequence(
        &self,
        id: runmat_types::CollectiveId,
    ) -> Result<runmat_execution::CollectiveSequence, RuntimeError>;

    fn execute(
        &self,
        request: runmat_execution::CollectiveRequest,
    ) -> RuntimeServiceFuture<Result<runmat_execution::CollectiveResponse, RuntimeError>>;
}

/// Narrow, typed ports composed by the host. An absent port is meaningful and
/// produces a stable capability error through the corresponding `require_*`
/// accessor; there is no string-keyed service locator.
#[derive(Clone, Default)]
pub struct RuntimeServicePorts {
    call: Option<Rc<dyn RuntimeCallService>>,
    builtin: Option<Rc<dyn RuntimeBuiltinService>>,
    workspace: Option<Rc<dyn RuntimeWorkspaceService>>,
    object: Option<Rc<dyn RuntimeObjectService>>,
    host: Option<Rc<dyn RuntimeHostService>>,
    error: Option<Rc<dyn RuntimeErrorService>>,
    acceleration: Option<Rc<dyn RuntimeAccelerationService>>,
    placement: Option<Rc<dyn RuntimePlacementService>>,
    native: Option<Rc<dyn RuntimeNativeService>>,
    foreign: Option<Rc<dyn RuntimeForeignService>>,
    parallel: Option<Rc<dyn RuntimeParallelService>>,
    spmd: Option<Rc<dyn RuntimeSpmdService>>,
    distributed: Option<Rc<dyn RuntimeDistributedService>>,
    collective: Option<Rc<dyn RuntimeCollectiveService>>,
}

impl std::fmt::Debug for RuntimeServicePorts {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RuntimeServicePorts")
            .field("call", &self.call.is_some())
            .field("builtin", &self.builtin.is_some())
            .field("workspace", &self.workspace.is_some())
            .field("object", &self.object.is_some())
            .field("host", &self.host.is_some())
            .field("error", &self.error.is_some())
            .field("acceleration", &self.acceleration.is_some())
            .field("placement", &self.placement.is_some())
            .field("native", &self.native.is_some())
            .field("foreign", &self.foreign.is_some())
            .field("parallel", &self.parallel.is_some())
            .field("spmd", &self.spmd.is_some())
            .field("distributed", &self.distributed.is_some())
            .field("collective", &self.collective.is_some())
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
        with_builtin,
        builtin,
        require_builtin,
        builtin,
        RuntimeBuiltinService,
        Builtin
    );
    port_accessors!(
        with_workspace,
        workspace,
        require_workspace,
        workspace,
        RuntimeWorkspaceService,
        Workspace
    );

    /// Remove an inherited caller-workspace service before entering an
    /// isolated procedure frame. The callee may then install its own scoped
    /// workspace authority without exposing or mutating the caller frame.
    pub fn without_workspace(mut self) -> Self {
        self.workspace = None;
        self
    }
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

    /// Remove the strong foreign-runtime edge while retaining every other
    /// session authority. A long-lived foreign callback can retain this
    /// context and restore the foreign port from a weak reference while it
    /// executes without creating a host/session ownership cycle.
    pub fn without_foreign(mut self) -> Self {
        self.foreign = None;
        self
    }
    port_accessors!(
        with_parallel,
        parallel,
        require_parallel,
        parallel,
        RuntimeParallelService,
        Parallel
    );
    port_accessors!(
        with_spmd,
        spmd,
        require_spmd,
        spmd,
        RuntimeSpmdService,
        Spmd
    );
    port_accessors!(
        with_distributed,
        distributed,
        require_distributed,
        distributed,
        RuntimeDistributedService,
        Distributed
    );
    port_accessors!(
        with_collective,
        collective,
        require_collective,
        collective,
        RuntimeCollectiveService,
        Collective
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
            missing(ports.require_builtin("resolve builtin")),
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
            missing(ports.require_distributed("distributed array")),
            missing(ports.require_collective("SPMD barrier")),
        ];
        assert_eq!(
            failures
                .iter()
                .map(|failure| failure.capability)
                .collect::<Vec<_>>(),
            vec![
                RuntimeCapability::Builtin,
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
                RuntimeCapability::Distributed,
                RuntimeCapability::Collective,
            ]
        );
        assert!(failures.iter().all(|failure| {
            failure.clone().into_runtime_error().identifier()
                == Some(RuntimeCapabilityError::IDENTIFIER)
        }));
    }
}
