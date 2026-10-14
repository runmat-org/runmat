use runmat_execution::value::{GraphCallableOwner, GraphNode, GraphNodeRef};
use runmat_execution::{ExecutableSourceMapV2, ProgramCallable};
use runmat_execution_artifact::{
    GraphExecutionRequest, GraphInvocationRoots, ProgramExecutionResponse,
};
use runmat_runtime::context::RuntimeForeignService;
use runmat_runtime::execution::value_codec::GraphSourceIssuer;
use runmat_runtime::execution::{RootedValueSequence, RootedValues};
use runmat_value::{Value, ValueSequence};
use std::rc::Rc;
use std::sync::Arc;

use super::{recipe_supports_capabilities, runtime_failure_response};

#[path = "graph_request/selected.rs"]
mod selected;
use selected::SelectedWorkerGraph;
#[path = "graph_request/prepared.rs"]
mod prepared;
pub use prepared::{
    execute_prepared_graph_request, prepare_graph_request_with_selected_foreign_ports_recording,
    PreparedSelectedGraph, SelectedGraphCleanupContexts,
};

/// Foreign services materialized for the exact ordered Graph V3 units.
/// Position zero is the primary unit; later positions match retained units.
/// A missing port never inherits the host's ambient foreign catalog.
#[derive(Clone)]
pub struct SelectedGraphForeignPorts(Vec<Option<Rc<dyn RuntimeForeignService>>>);

impl SelectedGraphForeignPorts {
    pub fn new(ordered: Vec<Option<Rc<dyn RuntimeForeignService>>>) -> Self {
        Self(ordered)
    }

    fn as_slice(&self) -> &[Option<Rc<dyn RuntimeForeignService>>] {
        &self.0
    }
}

fn failure(message: impl Into<String>) -> ProgramExecutionResponse {
    ProgramExecutionResponse::Failure {
        message: message.into(),
    }
}

fn graph_error(message: impl Into<String>) -> runmat_runtime::RuntimeError {
    runmat_runtime::runtime_error::semantic_error("WorkerGraph", message)
}

fn graph_runtime_failure_response(
    error: runmat_runtime::RuntimeError,
    runtime: &runmat_runtime::context::RuntimeContext,
    selected: &SelectedWorkerGraph,
) -> ProgramExecutionResponse {
    graph_runtime_failure_with_sources(error, runtime, &selected.issuer, &selected.source_map)
}

fn graph_runtime_failure_with_sources(
    error: runmat_runtime::RuntimeError,
    runtime: &runmat_runtime::context::RuntimeContext,
    issuer: &GraphSourceIssuer,
    source_map: &ExecutableSourceMapV2,
) -> ProgramExecutionResponse {
    match runmat_runtime::execution::encode_runtime_failure_with_source_issuer_in_context(
        &error, runtime, issuer, source_map,
    ) {
        Ok(failure) => ProgramExecutionResponse::RuntimeFailure { failure },
        Err(protocol_error) => failure(format!(
            "worker could not encode its structured graph runtime failure: {protocol_error}"
        )),
    }
}

async fn invoke_selected_nested_bound(
    selected: &SelectedWorkerGraph,
    caller_runtime: runmat_runtime::context::RuntimeContext,
    binding: runmat_value::BoundFunctionBinding,
    arguments: RootedValues,
    requested_outputs: usize,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
) -> Result<RootedValueSequence, runmat_runtime::RuntimeError> {
    let (_caller_runtime, selected_method) =
        caller_runtime.take_selected_method_body_for(&binding)?;
    let owner = selected.owner_for_binding(&binding).map_err(graph_error)?;
    let unit = selected.unit(owner).map_err(graph_error)?;
    let mut view = unit
        .registry()
        .owned_view(runmat_hir::FunctionId(binding.function), &binding.authority)
        .ok_or_else(|| graph_error("selected static target definition graph is incomplete"))?;
    let mut runtime = selected.runtime_for_unit(owner.unit).map_err(graph_error)?;
    if let Some(product) = selected.class_products.get(&owner) {
        view.attach_selected_class_product(
            runmat_hir::FunctionId(binding.function),
            &binding.authority,
            Rc::clone(product),
        )
        .map_err(graph_error)?;
        runtime = product.invocation_runtime(runtime)?;
    }
    if let Some((facts, method_owner)) = selected_method {
        let constructor = selected
            .owner_for_binding(facts.constructor())
            .map_err(graph_error)?;
        let issued_facts = selected
            .object_classes
            .get(&constructor)
            .ok_or_else(|| graph_error("selected receiver constructor is absent"))?;
        let product = selected
            .class_products
            .get(&constructor)
            .ok_or_else(|| graph_error("selected receiver class product is absent"))?;
        let body = view
            .get(runmat_hir::FunctionId(binding.function))
            .ok_or_else(|| graph_error("selected receiver method body is absent"))?;
        if !Rc::ptr_eq(&facts, issued_facts)
            || method_owner.role != runmat_types::ClassMethodRole::Ordinary
            || method_owner.is_static
            || body.class_method_owner.as_ref() != Some(&method_owner)
            || facts.method_binding(&method_owner.method).as_ref() != Some(&binding)
            || !product.validates_receiver_method(&facts, &method_owner)
            || body.selected_source_owner.as_ref().is_none_or(|source| {
                unit.selected_source_owner()
                    .is_none_or(|selected| !source.same_issuance(selected))
            })
        {
            return Err(graph_error(
                "selected receiver method differs from its admitted class/body",
            ));
        }
        runtime = product
            .invocation_runtime(runtime)?
            .with_selected_object_class(facts);
    }
    crate::interpreter::runner::invoke_selected_graph_bound_in_context(
        runtime,
        binding,
        arguments,
        requested_outputs,
        view,
        lexical_environment,
    )
    .await
}

struct SelectedFailureSources {
    issuer: GraphSourceIssuer,
    source_map: Arc<ExecutableSourceMapV2>,
}

#[derive(Default)]
struct GraphFailureRecorderState {
    claimed: bool,
    selected: Option<SelectedFailureSources>,
}

/// Carries the exact admitted V8 source projection across host-owned cleanup.
/// One recorder belongs to one request and contains no runtime context,
/// service port, or mutable source lookup.
#[derive(Clone, Default)]
pub struct GraphFailureSourceRecorder(std::rc::Rc<std::cell::RefCell<GraphFailureRecorderState>>);

impl GraphFailureSourceRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    fn claim(&self) -> bool {
        let mut state = self.0.borrow_mut();
        if state.claimed {
            return false;
        }
        state.claimed = true;
        true
    }

    fn record(&self, selected: &SelectedWorkerGraph) -> bool {
        let mut state = self.0.borrow_mut();
        if !state.claimed || state.selected.is_some() {
            return false;
        }
        state.selected = Some(SelectedFailureSources {
            issuer: selected.issuer.clone(),
            source_map: selected.source_map.clone(),
        });
        true
    }

    pub fn encode_runtime_failure(
        &self,
        error: runmat_runtime::RuntimeError,
        runtime: &runmat_runtime::context::RuntimeContext,
    ) -> ProgramExecutionResponse {
        let state = self.0.borrow();
        graph_host_failure_response(error, runtime, state.selected.as_ref())
    }
}

fn graph_host_failure_response(
    error: runmat_runtime::RuntimeError,
    runtime: &runmat_runtime::context::RuntimeContext,
    selected: Option<&SelectedFailureSources>,
) -> ProgramExecutionResponse {
    match selected {
        Some(selected) => graph_runtime_failure_with_sources(
            error,
            runtime,
            &selected.issuer,
            &selected.source_map,
        ),
        None => runtime_failure_response(error, runtime),
    }
}

/// Host-owned Graph entrypoint. Direct V7 keeps its own manifest-entrypoint
/// policy; a graph request executes only its reconstructed typed root.
pub async fn execute_graph_request(request: GraphExecutionRequest) -> ProgramExecutionResponse {
    if let Err(error) = request.validate() {
        return failure(format!("worker rejected invalid graph request: {error}"));
    }
    let selected_requirements = match request.selected_unit_host_requirements() {
        Ok(requirements) => requirements,
        Err(error) => return failure(format!("worker rejected selected graph interop: {error}")),
    };
    if selected_requirements.iter().any(|requirement| {
        !requirement.interop.foreign_types.is_empty()
            || !requirement.interop.adapters.is_empty()
            || !requirement.interop.adapter_contracts.is_empty()
    }) {
        return failure("graph interop requires an explicitly provisioned worker host");
    }
    let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ));
    let workspace_owner = runmat_runtime::workspace::BaseWorkspaceOwner::new();
    let workspace_service = std::rc::Rc::new(runmat_runtime::workspace::BaseWorkspaceService::new(
        workspace_owner.clone(),
    ));
    #[cfg(not(target_arch = "wasm32"))]
    let supervisor = std::rc::Rc::new(runmat_runtime::context::RuntimeLocalTaskSupervisor::new());
    #[cfg(target_arch = "wasm32")]
    let supervisor = std::rc::Rc::new(runmat_runtime::context::BrowserLocalTaskSupervisor::new());
    let ports = runtime
        .service_ports()
        .clone()
        .with_workspace(workspace_service.clone())
        .with_base_workspace(workspace_service)
        .with_local_tasks(supervisor.submission_port(runtime.local_identity()));
    let runtime = runtime.with_service_ports(ports);
    let host = match runmat_runtime::context::RuntimeLifecycleHost::new(
        &runtime,
        supervisor.clone(),
    ) {
        Ok(host) => host,
        Err(error) => {
            let response = runtime_failure_response(error, &runtime);
            if let Err(close_error) = workspace_owner.close() {
                tracing::warn!(%close_error, "workspace close failed after graph host admission error");
            }
            return response;
        }
    };
    let selected_failure_sources = GraphFailureSourceRecorder::new();
    let complete = async {
        let result = execute_graph_request_with_context_recorded(
            request,
            runtime.clone(),
            None,
            &selected_failure_sources,
        )
        .await;
        match host.retire_workspace_and_drain(&runtime).await {
            Ok(()) => result,
            Err(error)
                if matches!(
                    &result,
                    ProgramExecutionResponse::Failure { .. }
                        | ProgramExecutionResponse::RuntimeFailure { .. }
                ) =>
            {
                tracing::warn!(%error, "cleanup callback failed after graph worker error");
                result
            }
            Err(error) => selected_failure_sources.encode_runtime_failure(error, &runtime),
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    let result = match supervisor.participate_foreground(complete).await {
        Ok(result) => result,
        Err(error) => selected_failure_sources.encode_runtime_failure(error, &runtime),
    };
    #[cfg(target_arch = "wasm32")]
    let result = complete.await;
    host.force_close();
    match workspace_owner.close() {
        Ok(_) => result,
        Err(error)
            if matches!(
                &result,
                ProgramExecutionResponse::Failure { .. }
                    | ProgramExecutionResponse::RuntimeFailure { .. }
            ) =>
        {
            tracing::warn!(%error, "workspace close failed after graph worker error");
            result
        }
        Err(error) => selected_failure_sources.encode_runtime_failure(
            runmat_runtime::workspace::base_workspace_error_to_runtime(error),
            &runtime,
        ),
    }
}

/// A foreign/collective host supplies its explicit execution context. The
/// selected bundle and graph are still admitted afresh in that session.
pub async fn execute_graph_request_with_context(
    request: GraphExecutionRequest,
    runtime: runmat_runtime::context::RuntimeContext,
) -> ProgramExecutionResponse {
    execute_graph_request_with_context_recording(
        request,
        runtime,
        GraphFailureSourceRecorder::new(),
    )
    .await
}

/// Explicit host route. The host retains `failure_sources` across its own
/// foreground drain and close so post-admission errors use the selected union.
pub async fn execute_graph_request_with_context_recording(
    request: GraphExecutionRequest,
    runtime: runmat_runtime::context::RuntimeContext,
    failure_sources: GraphFailureSourceRecorder,
) -> ProgramExecutionResponse {
    execute_graph_request_with_context_recorded(request, runtime, None, &failure_sources).await
}

/// The host has already validated and materialized one foreign-artifact
/// receipt per selected V3 unit. Only that unit's foreign service varies;
/// VM derives every execution context from the same session-owning `runtime`.
pub async fn execute_graph_request_with_selected_foreign_ports_recording(
    request: GraphExecutionRequest,
    runtime: runmat_runtime::context::RuntimeContext,
    selected_foreign_ports: SelectedGraphForeignPorts,
    failure_sources: GraphFailureSourceRecorder,
) -> ProgramExecutionResponse {
    execute_graph_request_with_context_recorded(
        request,
        runtime,
        Some(selected_foreign_ports),
        &failure_sources,
    )
    .await
}

async fn execute_graph_request_with_context_recorded(
    request: GraphExecutionRequest,
    runtime: runmat_runtime::context::RuntimeContext,
    selected_foreign_ports: Option<SelectedGraphForeignPorts>,
    failure_sources: &GraphFailureSourceRecorder,
) -> ProgramExecutionResponse {
    let (admitted, bundles) = match request.into_validated() {
        Ok(admitted) => admitted.into_admitted_with_bundles(),
        Err(error) => return failure(format!("worker rejected invalid graph request: {error}")),
    };
    let (prepared, _contexts) = match prepared::prepare(
        admitted,
        bundles,
        runtime,
        selected_foreign_ports,
        failure_sources,
    ) {
        Ok(prepared) => prepared,
        Err(response) => return response,
    };
    prepared::execute_prepared_graph_request(prepared).await
}

async fn invoke_selected_root(
    request: &GraphExecutionRequest,
    selected: &SelectedWorkerGraph,
    runtime: &runmat_runtime::context::RuntimeContext,
    values: &RootedValues,
) -> Result<RootedValueSequence, ProgramExecutionResponse> {
    match &request.roots {
        GraphInvocationRoots::Callable { target, .. } => {
            invoke_bound_root(request, selected, values, *target)
                .await
                .map_err(|error| graph_runtime_failure_response(error, runtime, selected))
        }
        GraphInvocationRoots::Builtin { name, .. } => {
            if !runmat_builtins::builtin_name_is_known(&name.0) {
                return Err(failure("graph builtin is absent from the worker catalog"));
            }
            let required = runmat_builtins::builtin_required_capabilities(&name.0);
            if !recipe_supports_capabilities(&request.recipe, &required) {
                return Err(failure(
                    "graph recipe weakens builtin accelerator requirements",
                ));
            }
            let sequence = runmat_runtime::call_builtin_sequence_async_with_outputs_in_context(
                runtime,
                &name.0,
                values.values(),
                usize::from(request.requested_outputs),
            )
            .await
            .map_err(|error| graph_runtime_failure_response(error, runtime, selected))?;
            Ok(sequence)
        }
        GraphInvocationRoots::ParallelRegion { region, .. } => {
            let Some(primary) = selected.units.first() else {
                return Err(failure("parallel graph has no primary selected unit"));
            };
            let callable = ProgramCallable::parallel_region(*region);
            let value = super::execute_parallel_region_values(
                &callable,
                &request.context,
                request.requested_outputs,
                &request.recipe,
                primary.bytecode(),
                primary.runtime(),
                values.values().to_vec(),
            )
            .await
            .map_err(|error| match error {
                super::ParallelRegionError::Contract(response) => response,
                super::ParallelRegionError::Runtime(error) => {
                    graph_runtime_failure_response(error, runtime, selected)
                }
            })?;
            value.with_owned(|value| {
                let sequence = ValueSequence::single(value).map_err(|error| {
                    failure(format!("parallel graph result is invalid: {error}"))
                })?;
                RootedValueSequence::new(sequence).map_err(|error| {
                    graph_runtime_failure_response(error.into_runtime_error(), runtime, selected)
                })
            })
        }
    }
}

async fn invoke_bound_root(
    request: &GraphExecutionRequest,
    selected: &SelectedWorkerGraph,
    values: &RootedValues,
    target: GraphNodeRef,
) -> Result<RootedValueSequence, runmat_runtime::RuntimeError> {
    let Some(GraphNode::Callable(node)) = request.graph.nodes.get(target.0 as usize) else {
        return Err(runmat_runtime::runtime_error::semantic_error(
            "WorkerGraph",
            "graph callable target is absent",
        ));
    };
    let GraphCallableOwner::Bound(owner) = node.owner else {
        return Err(runmat_runtime::runtime_error::semantic_error(
            "WorkerGraph",
            "graph callable has no admitted bound owner",
        ));
    };
    let unit = selected
        .unit(owner)
        .map_err(|message| runmat_runtime::runtime_error::semantic_error("WorkerGraph", message))?;
    let binding = selected.bindings.get(&owner).ok_or_else(|| {
        runmat_runtime::runtime_error::semantic_error(
            "WorkerGraph",
            "graph target definition is absent",
        )
    })?;
    let target_value = values.values().first().ok_or_else(|| {
        runmat_runtime::runtime_error::semantic_error("WorkerGraph", "graph target root is absent")
    })?;
    let (mut arguments, environment) = match target_value {
        Value::Closure(closure) if closure.binding.as_ref() == Some(binding) => (
            closure.captures.clone(),
            closure.lexical_environment.clone(),
        ),
        Value::BoundFunctionHandle(handle)
            if handle.function == binding.function && handle.authority == binding.authority =>
        {
            (Vec::new(), None)
        }
        _ => {
            return Err(runmat_runtime::runtime_error::semantic_error(
                "WorkerGraph",
                "reconstructed target differs from selected bound callable",
            ))
        }
    };
    arguments.extend(values.values()[1..].iter().cloned());
    let mut view = unit
        .registry()
        .owned_view(runmat_hir::FunctionId(binding.function), &binding.authority)
        .ok_or_else(|| {
            runmat_runtime::runtime_error::semantic_error(
                "WorkerGraph",
                "selected callable definition graph is incomplete",
            )
        })?;
    let required = view
        .get(runmat_hir::FunctionId(binding.function))
        .ok_or_else(|| {
            runmat_runtime::runtime_error::semantic_error(
                "WorkerGraph",
                "selected callable body is absent",
            )
        })?
        .capabilities
        .clone();
    if !recipe_supports_capabilities(&request.recipe, &required) {
        return Err(runmat_runtime::runtime_error::semantic_error(
            "WorkerCapability",
            "graph recipe weakens callable accelerator requirements",
        ));
    }
    let selected_runtime = if let Some(product) = selected.class_products.get(&owner) {
        view.attach_selected_class_product(
            runmat_hir::FunctionId(binding.function),
            &binding.authority,
            std::rc::Rc::clone(product),
        )
        .map_err(|message| runmat_runtime::runtime_error::semantic_error("WorkerGraph", message))?;
        let runtime = selected.runtime_for_unit(owner.unit).map_err(|message| {
            runmat_runtime::runtime_error::semantic_error("WorkerGraph", message)
        })?;
        if let Some(facts) = selected.object_classes.get(&owner) {
            product
                .invocation_runtime(runtime)?
                .with_selected_object_class(Rc::clone(facts))
        } else {
            product.invocation_runtime(runtime)?
        }
    } else {
        selected.runtime_for_unit(owner.unit).map_err(|message| {
            runmat_runtime::runtime_error::semantic_error("WorkerGraph", message)
        })?
    };
    crate::invoke_semantic_function_value_with_environment_in_context(
        binding.function,
        &arguments,
        usize::from(request.requested_outputs),
        &view,
        selected_runtime,
        environment,
    )
    .await
}
