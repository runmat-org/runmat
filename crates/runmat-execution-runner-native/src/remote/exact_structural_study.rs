//! Driver-owned whole-study composition for exact CAD and structural-static analysis.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use runmat_analysis_execution::{
    build_analysis_task_submission, import_structural_linear_result, AnalysisExecutionContext,
};
use runmat_analysis_fea::execution::DEFAULT_ANALYSIS_CHUNK_BYTES;
use runmat_execution::identity::ArtifactId;
use runmat_execution::task::RetryPolicy;
use runmat_execution::value::ValuePayload;
use runmat_execution::Digest;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ProgramExecutionRequest, ProgramExecutionResponse};
use runmat_execution_transport_native::control::ResourceRequest as AllocationResources;

use super::exact_meshing::RemoteExactMeshingComposition;
use super::pool_execution::RemotePoolExecutionOutcome;
use super::pool_tasks::{execution_failure_outcome, packaged, task_failure_outcome};
use super::RemotePoolDriver;
use crate::{BoundaryRegionResolution, NativeExecutionError, NativeExecutionResult};

#[derive(Clone, Debug)]
pub(super) enum RemoteExactStructuralStudyComposition {
    Completed(ProgramExecutionResponse),
    Failed(String),
    Cancelled,
    Indeterminate(String),
}

pub(crate) async fn execute(
    pool: Arc<RemotePoolDriver>,
    request: ProgramExecutionRequest,
    run_id: String,
    resources: AllocationResources,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    match execute_for_composition(Arc::clone(&pool), &request, &run_id, &resources).await? {
        RemoteExactStructuralStudyComposition::Completed(response) => {
            packaged(response, pool).map(RemotePoolExecutionOutcome::Completed)
        }
        RemoteExactStructuralStudyComposition::Failed(message) => {
            execution_failure_outcome(pool, message)
        }
        RemoteExactStructuralStudyComposition::Cancelled => {
            Ok(RemotePoolExecutionOutcome::Cancelled)
        }
        RemoteExactStructuralStudyComposition::Indeterminate(message) => {
            Ok(RemotePoolExecutionOutcome::Indeterminate(message))
        }
    }
}

pub(super) async fn execute_for_composition(
    pool: Arc<RemotePoolDriver>,
    request: &ProgramExecutionRequest,
    run_id: &str,
    resources: &AllocationResources,
) -> NativeExecutionResult<RemoteExactStructuralStudyComposition> {
    let decoded = crate::decode_exact_structural_study_root(request, pool.as_ref())?;
    let completed = match super::exact_meshing::execute_for_composition(
        Arc::clone(&pool),
        &decoded.exact_meshing,
        run_id,
        resources,
    )
    .await?
    {
        RemoteExactMeshingComposition::Completed(completed) => completed,
        RemoteExactMeshingComposition::Terminal(outcome) => return Ok(composition(outcome)),
    };
    execute_with_completed_meshing(pool, request, decoded, completed, resources).await
}

pub(super) async fn execute_with_completed_meshing(
    pool: Arc<RemotePoolDriver>,
    request: &ProgramExecutionRequest,
    decoded: crate::DecodedExactStructuralStudyRoot,
    completed: Box<super::exact_meshing::CompletedRemoteExactMeshing>,
    resources: &AllocationResources,
) -> NativeExecutionResult<RemoteExactStructuralStudyComposition> {
    if completed.result.artifact.geometry.source_digest.bytes()
        != decoded.geometry_source_digest.bytes()
    {
        return Ok(RemoteExactStructuralStudyComposition::Failed(
            "meshed geometry differs from the whole-study source identity".into(),
        ));
    }

    let aliases = completed
        .source_face_ids
        .iter()
        .map(|(index, identity)| {
            (
                format!("face_{:06}", index.saturating_add(1)),
                identity.clone(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let resolution = BoundaryRegionResolution {
        schema_version: 1,
        geometry_source_digest: decoded.geometry_source_digest,
        solver_mesh_digest: Digest::from_bytes(*completed.result.artifact.canonical_digest.bytes()),
        aliases,
    };
    resolution.validate()?;
    let boundary_objects = crate::exact_structural_study_root::prepare_boundary_resolution_objects(
        &resolution,
        &decoded.artifact_access,
        DEFAULT_ANALYSIS_CHUNK_BYTES,
        ObjectInventoryLimits::default(),
    )?;

    let mut workload = decoded.workload_template.clone();
    if let Err(error) = resolve_workload_boundary_aliases(&mut workload, &resolution) {
        return Ok(RemoteExactStructuralStudyComposition::Failed(error));
    }
    let unresolved_workload_digest = Digest::sha256(
        &decoded
            .workload_template
            .canonical_bytes()
            .map_err(protocol)?,
    );
    let prepared = crate::exact_structural_study_root::prepare_structural_study_analysis_program(
        decoded.authored_study_digest,
        decoded.frozen_project_digest,
        request.recipe.program_revision.clone(),
        decoded.artifact_access.clone(),
        workload,
        completed.result.artifact.clone(),
    )
    .map_err(protocol)?;
    for object in &prepared.program.objects {
        pool.register_execution_object(object.reference.clone(), object.bytes.clone())?;
    }
    let roots = prepared
        .program
        .request
        .arguments
        .iter()
        .map(|argument| match argument {
            ValuePayload::Object(reference) => Ok(reference.as_ref().clone()),
            ValuePayload::Inline(_) => Err(protocol("analysis inputs must be externalized")),
        })
        .collect::<NativeExecutionResult<Vec<_>>>()?;
    let mut task_resources = super::pool_resources::task_resources(resources)?;
    task_resources.required_capabilities = prepared
        .host
        .workload
        .generic_capabilities()
        .map_err(protocol)?;
    let submission = build_analysis_task_submission(
        &prepared.host,
        roots,
        Vec::new(),
        BTreeSet::new(),
        AnalysisExecutionContext {
            scope_id: pool.scope_id(),
            pool_id: pool.pool_id(),
            program_artifact_id: ArtifactId::derive(&[prepared
                .program
                .request
                .artifact
                .id
                .0
                .bytes()]),
            resources: task_resources,
            retry: RetryPolicy::ExplicitlyIdempotent {
                max_attempts: runmat_analysis_execution::DEFAULT_ANALYSIS_MAXIMUM_ATTEMPTS,
            },
            deadline_unix_millis: Some(super::pool_resources::task_deadline_unix_millis(
                resources,
            )?),
            priority: 0,
        },
    )
    .map_err(protocol)?;
    let task_id = submission.request.id;
    let completion = pool.submit(submission, prepared.program.request.clone())?;
    let success = match completion.wait().await {
        Ok(success) => success,
        Err(message) => return task_failure_outcome(pool, task_id, message).map(composition),
    };
    let analysis_response = ProgramExecutionResponse::ExternalizedSuccess {
        outputs: success.outputs,
        result_objects: success.result_objects,
    };
    if let Err(error) = import_structural_linear_result(
        pool.as_ref(),
        analysis_response.clone(),
        &prepared,
        ObjectInventoryLimits::default(),
    ) {
        return Ok(RemoteExactStructuralStudyComposition::Failed(
            error.to_string(),
        ));
    }
    let (response, objects) =
        crate::exact_structural_study_root::package_exact_structural_study_result(
            request,
            &decoded,
            &completed.result,
            &boundary_objects,
            &analysis_response,
            unresolved_workload_digest,
        )?;
    for object in objects {
        pool.register_execution_object(object.reference, object.bytes)?;
    }
    Ok(RemoteExactStructuralStudyComposition::Completed(response))
}

pub(super) fn composition(
    outcome: RemotePoolExecutionOutcome,
) -> RemoteExactStructuralStudyComposition {
    match outcome {
        RemotePoolExecutionOutcome::Completed(result) => match result.response {
            ProgramExecutionResponse::Failure { message } => {
                RemoteExactStructuralStudyComposition::Failed(message)
            }
            response => RemoteExactStructuralStudyComposition::Completed(response),
        },
        RemotePoolExecutionOutcome::Cancelled => RemoteExactStructuralStudyComposition::Cancelled,
        RemotePoolExecutionOutcome::Indeterminate(message) => {
            RemoteExactStructuralStudyComposition::Indeterminate(message)
        }
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
}

fn resolve_workload_boundary_aliases(
    workload: &mut runmat_analysis_fea::execution::StructuralLinearWorkload,
    resolution: &BoundaryRegionResolution,
) -> Result<(), String> {
    let aliases = runmat_analysis_core::BoundaryRegionAliases::new(resolution.aliases.clone())
        .map_err(|error| error.to_string())?;
    workload.model =
        runmat_analysis_core::resolve_boundary_region_aliases(&workload.model, &aliases)
            .map_err(|error| error.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use runmat_analysis_fea::execution::StructuralLinearWorkload;
    use runmat_analysis_fea::fixtures::{fixture_model, FixtureId};
    use runmat_analysis_fea::solve::backend::kind::LinearAlgebraBackendKind;
    use runmat_analysis_fea::solve::preconditioner::SpdPreconditionerKind;
    use runmat_analysis_fea::ComputeBackend;
    use runmat_execution::Digest;
    use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};

    use super::*;

    #[test]
    fn unresolved_importer_alias_is_a_coordinator_resolution_failure() {
        let mut workload = StructuralLinearWorkload {
            schema_version: 1,
            model: fixture_model(FixtureId::CantileverLinearStatic),
            backend: ComputeBackend::Cpu,
            formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
            preconditioner: SpdPreconditionerKind::Jacobi,
            algebra_backend: LinearAlgebraBackendKind::CpuReference,
            require_solver_mesh_for_solid: true,
            thermo_mechanical_context: None,
            electro_thermal_context: None,
        };
        workload.model.constraints[0].visit_targets_mut(|target| {
            *target = runmat_analysis_core::RegionTarget::boundary("face_000002")
        });
        let resolution = BoundaryRegionResolution {
            schema_version: 1,
            geometry_source_digest: Digest::sha256(b"geometry"),
            solver_mesh_digest: Digest::sha256(b"mesh"),
            aliases: BTreeMap::from([(
                "face_000001".into(),
                PersistentEntityId {
                    kind: PersistentEntityKind::Face,
                    source_topology_id: "face-a".into(),
                    assembly_path: vec!["root".into()],
                },
            )]),
        };

        let error = resolve_workload_boundary_aliases(&mut workload, &resolution).unwrap_err();
        assert!(error.contains("face_000002"));
    }
}
