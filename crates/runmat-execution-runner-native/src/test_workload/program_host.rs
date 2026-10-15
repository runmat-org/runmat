//! Common Direct/Graph host route. Admission and owned host setup are synchronous
//! so a worker poll reserves only the selected continuation's frame.

use std::rc::Rc;
use std::sync::Arc;

use crate::materialized_project::{MaterializedProject, SelectedMaterializedView};
use runmat_execution_artifact::{
    ExecutableForm, ExecutionBundle, GraphExecutionRequest, ProgramExecutionRequest,
    ProgramExecutionResponse,
};

mod execution;
mod install;
use execution::boxed_unit_execution;
use install::boxed_selected_install;

pub(super) enum HostUnitRequest {
    Direct(ProgramExecutionRequest),
    Graph(GraphExecutionRequest),
}

struct HostPreparationError {
    message: String,
}

struct SelectedDirectRoute<'a> {
    prepared: Option<Arc<runmat_core::foreign_artifacts::PreparedSelectedForeignInput>>,
    bundle: Option<Box<ExecutionBundle>>,
    view: SelectedMaterializedView<'a>,
    handoff: Option<&'a runmat_package::FrozenProjectHandoff>,
}

struct ForeignHostRoute<'a> {
    request: HostUnitRequest,
    interop: runmat_types::InteropManifest,
    selected_direct: Option<SelectedDirectRoute<'a>>,
    session: runmat_core::RunMatSession,
    collective: Option<Rc<dyn runmat_runtime::context::RuntimeCollectiveService>>,
}

type HostResponseFuture<'a> =
    std::pin::Pin<Box<dyn std::future::Future<Output = ProgramExecutionResponse> + 'a>>;

pub(super) fn execute_host_unit_request<'a>(
    request: HostUnitRequest,
    materialized: Option<&'a MaterializedProject>,
    collective: Option<Rc<dyn runmat_runtime::context::RuntimeCollectiveService>>,
    bundle: Option<Arc<ExecutionBundle>>,
) -> HostResponseFuture<'a> {
    execute_host_unit_request_with_graphics(
        request,
        materialized,
        collective,
        bundle,
        &runmat_core::SessionGraphicsHost::Unavailable,
    )
}

pub(super) fn execute_host_unit_request_with_graphics<'a>(
    request: HostUnitRequest,
    materialized: Option<&'a MaterializedProject>,
    collective: Option<Rc<dyn runmat_runtime::context::RuntimeCollectiveService>>,
    bundle: Option<Arc<ExecutionBundle>>,
    graphics: &runmat_core::SessionGraphicsHost,
) -> HostResponseFuture<'a> {
    match prepare_host_unit_route(request, materialized, collective, bundle, graphics) {
        Ok(execution) => execution,
        Err(error) => Box::pin(std::future::ready(ProgramExecutionResponse::Failure {
            message: error.message,
        })),
    }
}

#[inline(never)]
fn prepare_host_unit_route<'a>(
    request: HostUnitRequest,
    materialized: Option<&'a MaterializedProject>,
    collective: Option<Rc<dyn runmat_runtime::context::RuntimeCollectiveService>>,
    bundle: Option<Arc<ExecutionBundle>>,
    graphics: &runmat_core::SessionGraphicsHost,
) -> Result<HostResponseFuture<'a>, HostPreparationError> {
    let failure = |message| HostPreparationError { message };
    if let HostUnitRequest::Graph(graph) = &request {
        graph.validate().map_err(|error| {
            failure(format!("worker rejected an invalid graph request: {error}"))
        })?;
    }
    let interop = match &request {
        HostUnitRequest::Direct(request) => match request.artifact.executable_unit() {
            Ok(Some(envelope)) => envelope.manifest.interop.clone(),
            Ok(None) => runmat_types::InteropManifest::empty(),
            Err(error) => {
                return Err(failure(format!(
                    "worker rejected an invalid executable unit: {error}"
                )))
            }
        },
        HostUnitRequest::Graph(request) => match request.selected_unit_host_requirements() {
            Ok(selected) if selected.iter().skip(1).all(|unit| {
                unit.interop.foreign_types.is_empty()
                    && unit.interop.adapters.is_empty()
                    && unit.interop.adapter_contracts.is_empty()
            }) => selected[0].interop.clone(),
            Ok(_) => {
                return Err(failure(
                    "worker has no ordered materialized bundle receipts for retained Graph foreign units".into(),
                ))
            }
            Err(error) => {
                return Err(failure(format!("worker rejected an invalid graph unit: {error}")))
            }
        },
    };

    // The original validated bundle, not a private path, names the selected
    // Direct recipe, artifact, foreign closure and captured policy.
    let selected_direct = match &request {
        HostUnitRequest::Direct(direct)
            if direct.artifact.form == ExecutableForm::ExecutableUnitV3
                && (interop != runmat_types::InteropManifest::empty()
                    || direct.recipe.foreign_policy.is_some()) =>
        {
            direct.validate_for_portable_host().map_err(|error| {
                failure(format!(
                    "worker rejected invalid Direct V3 before installation: {error}"
                ))
            })?;
            let (Some(materialized), Some(bundle)) = (materialized, bundle.as_ref()) else {
                let requested_identity = interop
                    .adapters
                    .iter()
                    .flat_map(|adapter| adapter.artifact_identities.iter())
                    .next();
                let message = match requested_identity {
                    Some(identity) => format!(
                        "selected Direct V3 interop requires its exact validated bundle and materialization for foreign artifact {identity}"
                    ),
                    None => "selected Direct V3 interop requires its exact validated bundle and materialization".into(),
                };
                return Err(failure(message));
            };
            let same_bundle = materialized.matches_bundle(bundle).map_err(|error| {
                failure(format!(
                    "worker could not verify materialized Direct bundle: {error}"
                ))
            })?;
            if !same_bundle
                || !bundle.manifest.recipes.contains(&direct.recipe)
                || !bundle.manifest.artifacts.contains(&direct.artifact)
            {
                return Err(failure(
                    "selected Direct V3 differs from its materialized bundle".into(),
                ));
            }
            let handoff = materialized.handoff();
            let prepared = handoff
                .map(|handoff| {
                    handoff
                        .validate()
                        .map_err(|error| failure(error.to_string()))?;
                    runmat_core::foreign_artifacts::PreparedSelectedForeignInput::from_validated_bundle(
                        Arc::clone(bundle),
                        &direct.recipe,
                        Arc::new(handoff.clone()),
                        materialized.private_root(),
                    )
                    .map(Some)
                    .map_err(failure)
                })
                .transpose()?
                .flatten();
            let selected_bundle = bundle
                .selected_unit_bundle_for(&direct.recipe, &direct.artifact.executable_bytes)
                .map_err(|error| failure(error.to_string()))?;
            let view = SelectedMaterializedView::from_admitted_recipe(
                materialized,
                bundle,
                &direct.recipe,
            )
            .map_err(failure)?;
            Some(SelectedDirectRoute {
                prepared,
                bundle: Some(Box::new(selected_bundle)),
                view,
                handoff,
            })
        }
        _ => None,
    };

    if interop == runmat_types::InteropManifest::empty()
        && collective.is_none()
        && selected_direct.is_none()
        && matches!(graphics, runmat_core::SessionGraphicsHost::Unavailable)
    {
        return Ok(match request {
            HostUnitRequest::Direct(request) => {
                Box::pin(runmat_vm::execute_program_request(request))
            }
            HostUnitRequest::Graph(request) => Box::pin(runmat_vm::execute_graph_request(request)),
        });
    }
    if selected_direct.is_none() && interop != runmat_types::InteropManifest::empty() {
        return Err(failure(
            "foreign interop requires an exact selected Direct or Graph archive".into(),
        ));
    }

    let mut session = runmat_core::RunMatSession::with_options(true, false).map_err(|error| {
        failure(format!(
            "failed to initialize foreign execution session: {error}"
        ))
    })?;
    if let Err(error) = session.attach_graphics_host(graphics) {
        let revision = match &request {
            HostUnitRequest::Direct(request) => request.recipe.program_revision.clone(),
            HostUnitRequest::Graph(request) => request.recipe.program_revision.clone(),
        };
        let runtime = session.execution_runtime_context(revision);
        let response = runmat_vm::runtime_failure_response(error, &runtime);
        return Ok(Box::pin(std::future::ready(response)));
    }
    if let Some(materialized) = materialized {
        session
            .attach_foreign_private_root(materialized.private_root())
            .map_err(|error| {
                failure(format!(
                    "worker could not bind exact foreign bytes: {error}"
                ))
            })?;
    }
    Ok(Box::pin(execute_foreign_host_route(ForeignHostRoute {
        request,
        interop,
        selected_direct,
        session,
        collective,
    })))
}

async fn execute_foreign_host_route(route: ForeignHostRoute<'_>) -> ProgramExecutionResponse {
    let ForeignHostRoute {
        request,
        interop,
        mut selected_direct,
        mut session,
        collective,
    } = route;
    if let Some(selected) = selected_direct.as_ref() {
        if let Err(response) = boxed_selected_install(&mut session, selected, &interop).await {
            return response;
        }
    }
    let revision = match &request {
        HostUnitRequest::Direct(request) => request.recipe.program_revision.clone(),
        HostUnitRequest::Graph(request) => request.recipe.program_revision.clone(),
    };
    let runtime = session.execution_runtime_context(revision);
    let runtime = if let Some(collective) = collective {
        let ports = runtime.service_ports().clone().with_collective(collective);
        runtime.with_service_ports(ports)
    } else {
        runtime
    };
    let driver = session.local_task_driver();
    let graph_request = matches!(&request, HostUnitRequest::Graph(_));
    let graph_failure_sources = runmat_vm::GraphFailureSourceRecorder::new();
    let selected_bundle = selected_direct
        .as_mut()
        .and_then(|selected| selected.bundle.take());
    let execution = boxed_unit_execution(
        request,
        runtime.clone(),
        selected_bundle,
        graph_failure_sources.clone(),
    );
    let callbacks = session.graphics_callback_queue().cloned();
    let execution = async move {
        match callbacks {
            Some(queue) => queue.drive_execution_with_host_callbacks(execution).await,
            None => Ok(execution.await),
        }
    };
    let mut response = match driver
        .participate_foreground(execution)
        .await
        .and_then(|result| result)
    {
        Ok(response) => response,
        Err(error) if graph_request => {
            graph_failure_sources.encode_runtime_failure(error, &runtime)
        }
        Err(error) => runmat_vm::runtime_failure_response(error, &runtime),
    };
    if let Err(error) = session.flush_graphics().await {
        if !matches!(
            &response,
            ProgramExecutionResponse::Failure { .. }
                | ProgramExecutionResponse::RuntimeFailure { .. }
        ) {
            response = runmat_vm::runtime_failure_response(error, &runtime);
        }
    }
    if let Err(error) = session.close_gracefully().await {
        if matches!(
            &response,
            ProgramExecutionResponse::Failure { .. }
                | ProgramExecutionResponse::RuntimeFailure { .. }
        ) {
            eprintln!("cleanup callback failed after worker execution error: {error}");
        } else {
            response = if graph_request {
                graph_failure_sources.encode_runtime_failure(error, &runtime)
            } else {
                runmat_vm::runtime_failure_response(error, &runtime)
            };
        }
    }
    response
}
