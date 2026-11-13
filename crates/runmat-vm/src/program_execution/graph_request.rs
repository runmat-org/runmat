use runmat_execution::value::{GraphCallableOwner, GraphNode, GraphNodeRef};
use runmat_execution::ExecutableSourceMapV2;
use runmat_execution_artifact::{GraphExecutionRequest, ProgramExecutionResponse};
use runmat_runtime::context::RuntimeForeignService;
use runmat_runtime::execution::value_codec::GraphSourceIssuer;
use runmat_runtime::execution::{RootedValueSequence, RootedValues};
use runmat_value::Value;
use std::rc::Rc;
use std::sync::Arc;

use super::{recipe_supports_capabilities, runtime_failure_response};

/// Software schemas implemented by this compiled selected Graph decoder and
/// executor. These are not service permissions or resource grants: original
/// host requirements and assignment admission remain independent authorities.
pub fn compiled_graph_execution_capabilities(
) -> std::collections::BTreeSet<runmat_execution::resource::Capability> {
    use runmat_execution::value::GraphRequiredSchema;
    [
        GraphRequiredSchema::SelectedValuesV11,
        GraphRequiredSchema::CallableArgumentsV12,
        #[cfg(not(target_arch = "wasm32"))]
        GraphRequiredSchema::ProjectRuntimePolicyV13,
        GraphRequiredSchema::InvocationControlsV14,
    ]
    .into_iter()
    .filter_map(GraphRequiredSchema::required_capability)
    .collect()
}

#[path = "graph_request/selected.rs"]
mod selected;
use selected::SelectedWorkerGraph;
#[path = "graph_request/invocation.rs"]
mod invocation;
use invocation::invoke_selected_root;
#[path = "graph_request/restored.rs"]
mod restored;
pub use restored::RestoredSelectedGraph;
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
    arguments: runmat_runtime::call::RootedFunctionArguments,
    requested_outputs: usize,
    lexical_environment: Option<runmat_value::LexicalEnvironment>,
) -> Result<RootedValueSequence, runmat_runtime::RuntimeError> {
    let owner = selected.owner_for_binding(&binding).map_err(graph_error)?;
    let foreign = selected
        .active_foreign_for_unit(owner.unit)
        .map_err(graph_error)?;
    invoke_selected_code_bound(
        selected,
        crate::interpreter::runner::BoundInvocation {
            runtime: caller_runtime,
            binding,
            args: arguments,
            requested_outputs,
            driver: None,
            lexical_environment,
        },
        foreign,
    )
    .await
}

#[inline(always)]
fn invoke_selected_code_bound<'a>(
    selected: &'a selected::RetainedSelectedGraph,
    invocation: crate::interpreter::runner::BoundInvocation,
    selected_foreign: Option<&'a Rc<dyn RuntimeForeignService>>,
) -> impl std::future::Future<Output = Result<RootedValueSequence, runmat_runtime::RuntimeError>> + 'a
{
    runmat_runtime::execution::after_result(
        async move {
            let (runtime, registry) = prepare_selected_bound(
                selected,
                invocation.runtime,
                &invocation.binding,
                selected_foreign,
            )?;
            Ok((
                registry,
                crate::interpreter::runner::BoundInvocation {
                    runtime,
                    ..invocation
                },
            ))
        },
        move |(registry, invocation)| {
            crate::interpreter::runner::invoke_selected_graph_bound_in_context(
                invocation.runtime,
                invocation.binding,
                invocation.args,
                invocation.requested_outputs,
                registry,
                invocation.lexical_environment,
                invocation.driver,
            )
        },
    )
}

// Source/class selection retires before the selected frame begins execution.
// The successor retains the exact owned registry and invocation runtime.
fn prepare_selected_bound(
    selected: &selected::RetainedSelectedGraph,
    caller_runtime: runmat_runtime::context::RuntimeContext,
    binding: &runmat_value::BoundFunctionBinding,
    selected_foreign: Option<&Rc<dyn RuntimeForeignService>>,
) -> Result<
    (
        runmat_runtime::context::RuntimeContext,
        crate::FunctionRegistry,
    ),
    runmat_runtime::RuntimeError,
> {
    let (caller_runtime, selected_method) =
        caller_runtime.take_selected_method_body_for(binding)?;
    let retained_super_body = caller_runtime.has_retained_super_body_for(binding);
    let caller_runtime = if selected_method.is_some()
        || caller_runtime.has_admitted_selected_method_frame_for(binding)
        || retained_super_body
    {
        caller_runtime
    } else {
        caller_runtime.without_selected_object_class()
    };
    let owner = selected.owner_for_binding(binding).map_err(graph_error)?;
    let unit = selected.unit(owner).map_err(graph_error)?;
    let mut view = unit
        .registry()
        .owned_view(runmat_hir::FunctionId(binding.function), &binding.authority)
        .ok_or_else(|| graph_error("selected static target definition graph is incomplete"))?;
    let mut runtime = selected
        .project_runtime(owner.unit, &caller_runtime, selected_foreign)
        .map_err(graph_error)?;
    if let Some(product) = selected.class_products.get(&owner) {
        view.attach_selected_class_product(
            runmat_hir::FunctionId(binding.function),
            &binding.authority,
            Rc::clone(product),
        )
        .map_err(graph_error)?;
        runtime = if retained_super_body {
            // A parent body operates on the existing Derived receiver. It is
            // not a fresh Base construction and must not replace those facts.
            product.invocation_runtime(runtime)?
        } else {
            let facts = selected
                .object_classes
                .get(&owner)
                .ok_or_else(|| graph_error("selected constructor receiver facts are absent"))?;
            if facts.constructor() != binding || facts.class() != product.class() {
                return Err(graph_error(
                    "selected constructor receiver facts differ from its exact graph binding",
                ));
            }
            // Cross-unit ordinary methods have independently admitted bindings.
            // Carry that sealed owner rather than rebuilding from this unit's view.
            product.invocation_runtime_with_binding(
                runtime.with_selected_object_class(Rc::clone(facts)),
                &view,
                binding,
            )?
        };
    }
    if let Some((facts, method_owner, dispatch)) = selected_method {
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
            || match dispatch {
                runmat_runtime::context::SelectedMethodDispatch::Virtual => {
                    facts.method(&method_owner.method)
                }
                runmat_runtime::context::SelectedMethodDispatch::Superclass => {
                    facts.method_by_owner(&method_owner)
                }
            }
            .is_none_or(|method| method.binding.as_ref() != Some(binding) || method.is_abstract)
            || !match dispatch {
                runmat_runtime::context::SelectedMethodDispatch::Virtual => {
                    product.validates_receiver_method(&facts, &method_owner)
                }
                runmat_runtime::context::SelectedMethodDispatch::Superclass => {
                    product.validates_receiver_owned_method(&facts, &method_owner)
                }
            }
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
        runtime =
            runtime.with_admitted_selected_method_frame(method_owner, binding.clone(), dispatch)?;
    }
    Ok((runtime, view))
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CallableArgumentLayout {
    LegacyFlattened,
    SuppliedOnly,
}

fn callable_argument_layout(version: u16) -> Option<CallableArgumentLayout> {
    use runmat_execution::value::GraphRequiredSchema;

    match GraphRequiredSchema::from_version(version)? {
        GraphRequiredSchema::V10 | GraphRequiredSchema::SelectedValuesV11 => {
            Some(CallableArgumentLayout::LegacyFlattened)
        }
        GraphRequiredSchema::CallableArgumentsV12
        | GraphRequiredSchema::ProjectRuntimePolicyV13
        | GraphRequiredSchema::InvocationControlsV14 => Some(CallableArgumentLayout::SuppliedOnly),
    }
}

async fn invoke_bound_root(
    request: &GraphExecutionRequest,
    selected: &SelectedWorkerGraph,
    values: RootedValues,
    target: GraphNodeRef,
) -> Result<RootedValueSequence, runmat_runtime::RuntimeError> {
    let (binding, view, selected_runtime) = prepare_bound_root(request, selected, target)?;
    values
        .with_owned_output_storage_async(move |storage| async move {
            let mut owned = storage.try_into_owned().map_err(|_| {
                graph_error("graph arguments require their original unique collection owner")
            })?;
            let target_value = owned
                .try_remove(0)
                .ok_or_else(|| graph_error("graph target root is absent"))?;
            let supplied = owned.try_into_storage().map_err(|_| {
                graph_error("graph arguments could not republish their original collection owner")
            })?;
            let supplied =
                RootedValues::from_output_storage_in_context(supplied, &selected_runtime)?;
            let (arguments, environment) = match target_value {
                Value::Closure(closure) if closure.binding.as_ref() == Some(&binding) => {
                    let environment = closure.lexical_environment.clone();
                    let callable =
                        runmat_runtime::execution::RetainedValueReadLease::prepare_owned(
                            Value::Closure(closure),
                            &selected_runtime,
                        )?;
                    (
                        runmat_runtime::call::RootedFunctionArguments::captured(
                            callable, supplied,
                        )?,
                        environment,
                    )
                }
                Value::BoundFunctionHandle(handle)
                    if handle.function == binding.function
                        && handle.authority == binding.authority =>
                {
                    (
                        runmat_runtime::call::RootedFunctionArguments::ordinary(supplied),
                        None,
                    )
                }
                _ => {
                    return Err(runmat_runtime::runtime_error::semantic_error(
                        "WorkerGraph",
                        "reconstructed target differs from selected bound callable",
                    ))
                }
            };
            crate::interpreter::runner::invoke_selected_graph_bound_in_context(
                selected_runtime,
                binding,
                arguments,
                usize::from(request.requested_outputs),
                view,
                environment,
                None,
            )
            .await
        })
        .await
}

fn prepare_bound_root(
    request: &GraphExecutionRequest,
    selected: &SelectedWorkerGraph,
    target: GraphNodeRef,
) -> Result<
    (
        runmat_value::BoundFunctionBinding,
        crate::FunctionRegistry,
        runmat_runtime::context::RuntimeContext,
    ),
    runmat_runtime::RuntimeError,
> {
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
    Ok((binding.clone(), view, selected_runtime))
}
