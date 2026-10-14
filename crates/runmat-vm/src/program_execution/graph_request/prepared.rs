//! One selected Graph admission shared by host install, dispatch, and cleanup.

use super::*;
use runmat_execution::value::ValueLimits;
use runmat_execution::{JobId, ProgramExecutionAssignment};
use runmat_execution_artifact::{AdmittedGraphUnitBundle, ArchiveAdmittedGraphExecutionRequest};
use runmat_runtime::context::RuntimeContext;
use std::future::Future;

/// A single-use selected graph. Its V3/source/class admission is never repeated
/// after the foreign host has begun installing unit-local adapters.
pub struct PreparedSelectedGraph {
    admitted: ArchiveAdmittedGraphExecutionRequest,
    selected: SelectedWorkerGraph,
    runtime: RuntimeContext,
}

/// Short-lived exact contexts for unit-owned foreign install and retirement.
/// The host may enter one unit's assignment/job scope around a callback, but
/// no scope guard remains active between preparation and execution.
pub struct SelectedGraphCleanupContexts {
    units: Vec<RuntimeContext>,
    assignment: Option<ProgramExecutionAssignment>,
    job_id: Option<JobId>,
}

impl SelectedGraphCleanupContexts {
    pub fn len(&self) -> usize {
        self.units.len()
    }

    pub fn is_empty(&self) -> bool {
        self.units.is_empty()
    }

    pub async fn with_unit_context<F, Fut, T>(
        &self,
        index: usize,
        operation: F,
    ) -> Result<T, String>
    where
        F: FnOnce(RuntimeContext) -> Fut,
        Fut: Future<Output = T>,
    {
        let context = self
            .units
            .get(index)
            .ok_or("selected graph cleanup context is absent")?
            .clone();
        let _assignment = context.enter_execution_assignment(self.assignment.clone());
        let _job = context.enter_execution_job(self.job_id);
        let callback = {
            let _active = context.enter();
            operation(context.clone())
        };
        Ok(context.scope(callback).await)
    }
}

pub fn prepare_graph_request_with_selected_foreign_ports_recording(
    admitted: ArchiveAdmittedGraphExecutionRequest,
    bundles: Vec<AdmittedGraphUnitBundle>,
    runtime: RuntimeContext,
    selected_foreign_ports: SelectedGraphForeignPorts,
    failure_sources: &GraphFailureSourceRecorder,
) -> Result<(PreparedSelectedGraph, SelectedGraphCleanupContexts), ProgramExecutionResponse> {
    prepare(
        admitted,
        bundles,
        runtime,
        Some(selected_foreign_ports),
        failure_sources,
    )
}

pub(super) fn prepare(
    admitted: ArchiveAdmittedGraphExecutionRequest,
    bundles: Vec<AdmittedGraphUnitBundle>,
    runtime: RuntimeContext,
    selected_foreign_ports: Option<SelectedGraphForeignPorts>,
    failure_sources: &GraphFailureSourceRecorder,
) -> Result<(PreparedSelectedGraph, SelectedGraphCleanupContexts), ProgramExecutionResponse> {
    if !failure_sources.claim() {
        return Err(failure(
            "graph failure-source recorder is already assigned to a request",
        ));
    }
    let request = admitted.request();
    if let Err(error) = request.recipe.program_revision.validate_current_compiler() {
        return Err(failure(format!(
            "worker rejected compiler-incompatible graph: {error}"
        )));
    }
    let _assignment = runtime.enter_execution_assignment(request.assignment.clone());
    let _job = runtime.enter_execution_job(request.job_id);
    let selected = SelectedWorkerGraph::admit(
        &admitted,
        bundles,
        &runtime,
        selected_foreign_ports
            .as_ref()
            .map(SelectedGraphForeignPorts::as_slice),
    )
    .map_err(|error| failure(format!("worker rejected selected graph units: {error}")))?;
    if !failure_sources.record(&selected) {
        return Err(failure(
            "graph failure-source recorder already has a selected admission",
        ));
    }
    let contexts = SelectedGraphCleanupContexts {
        units: selected.unit_runtimes.to_vec(),
        assignment: request.assignment.clone(),
        job_id: request.job_id,
    };
    Ok((
        PreparedSelectedGraph {
            admitted,
            selected,
            runtime,
        },
        contexts,
    ))
}

pub async fn execute_prepared_graph_request(
    prepared: PreparedSelectedGraph,
) -> ProgramExecutionResponse {
    let PreparedSelectedGraph {
        admitted,
        selected,
        runtime,
    } = prepared;
    let selected = std::rc::Rc::new(selected);
    let request = admitted.request();
    let _assignment = runtime.enter_execution_assignment(request.assignment.clone());
    let _job = runtime.enter_execution_job(request.job_id);
    if selected.units.is_empty() {
        return failure("worker graph has no admitted primary unit");
    }
    let selected_runtime =
        match selected.runtime_for_unit(runmat_execution::value::GraphProgramUnitId(0)) {
            Ok(runtime) => runtime,
            Err(message) => return failure(message),
        };
    let values = match runmat_runtime::execution::value_codec::decode_payload_graph_with_selected_classes_in_context(
        &request.graph,
        &selected_runtime,
        &selected.source_map,
        &selected.issuer,
        &selected.layouts,
        &selected.proofs,
        &selected.bindings,
        &selected.object_classes,
        ValueLimits::default(),
    ) {
        Ok(values) => values,
        Err(error) => return failure(format!("worker rejected graph input: {error}")),
    };
    let nested_selected = std::rc::Rc::clone(&selected);
    let active_runtime = selected_runtime.enter();
    let _selected_bound_invoker =
        runmat_runtime::user_functions::install_local_bound_function_invoker(std::rc::Rc::new(
            move |caller_runtime, binding, arguments, outputs, environment| {
                let selected = std::rc::Rc::clone(&nested_selected);
                Box::pin(async move {
                    super::invoke_selected_nested_bound(
                        &selected,
                        caller_runtime,
                        binding,
                        arguments,
                        outputs,
                        environment,
                    )
                    .await
                })
            },
        ));
    drop(active_runtime);
    let result = match selected_runtime
        .scope(invoke_selected_root(
            request,
            &selected,
            &selected_runtime,
            &values,
        ))
        .await
    {
        Ok(values) => values,
        Err(response) => return response,
    };
    let response = result.with_owned_sequence(|sequence| {
        let preliminary = &selected.issuer;
        let prepared = runmat_runtime::execution::value_codec::prepare_payload_graph(
            sequence.as_slice(),
            preliminary,
            ValueLimits::default(),
        )
        .map_err(|error| error.to_string())?;
        let mut node_units = std::collections::HashMap::new();
        for (node, origin, authority) in prepared.bound_authorities() {
            let selected_owner = selected.owner_for_authority(*origin, authority)?;
            if node_units.insert(*node, selected_owner.unit).is_some() {
                return Err("one result node repeats its selected authority".to_string());
            }
        }
        let graph = prepared
            .finish(
                &selected.issuer,
                &selected.source_map,
                &selected.layouts,
                &selected.proofs,
                &node_units,
            )
            .map_err(|error| error.to_string())?;
        Ok::<_, String>(ProgramExecutionResponse::GraphSuccess { graph })
    });
    let response = match response {
        Ok(response) => response,
        Err(error) => return failure(format!("worker could not transfer graph result: {error}")),
    };
    if let Err(error) = admitted.validate_response(&response) {
        return failure(format!(
            "worker graph result violates its output contract: {error}"
        ));
    }
    response
}

#[cfg(test)]
#[path = "prepared/tests.rs"]
mod tests;
