//! Driver-owned whole-study composition for exact CAD and structural-static analysis.

use std::collections::BTreeMap;
use std::sync::Arc;

use runmat_analysis_execution::StructuralSequencePreparation;
use runmat_analysis_fea::execution::DEFAULT_ANALYSIS_CHUNK_BYTES;
use runmat_execution::state::TaskState;
use runmat_execution::task::RetryPolicy;
use runmat_execution::Digest;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ProgramExecutionRequest, ProgramExecutionResponse};
use runmat_execution_transport_native::control::ResourceRequest as AllocationResources;

use super::exact_meshing::RemoteExactMeshingComposition;
use super::pool_execution::RemotePoolExecutionOutcome;
use super::pool_tasks::{execution_failure_outcome, packaged};
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
    let signal_materializations = decoded.signal_materializations.clone();
    let artifact_access = decoded.artifact_access.clone();
    let meshing = super::exact_meshing::execute_for_composition(
        Arc::clone(&pool),
        &decoded.exact_meshing,
        run_id,
        resources,
    );
    let signals = super::signal_materialization::execute(
        Arc::clone(&pool),
        signal_materializations,
        &artifact_access,
        resources,
    );
    let (meshing, signal_outcomes) = tokio::join!(meshing, signals);
    let completed = match meshing? {
        RemoteExactMeshingComposition::Completed(completed) => completed,
        RemoteExactMeshingComposition::Terminal(outcome) => return Ok(composition(outcome)),
    };
    execute_with_completed_meshing(
        pool,
        request,
        decoded,
        completed,
        &signal_outcomes?,
        resources,
    )
    .await
}

pub(super) async fn execute_with_completed_meshing(
    pool: Arc<RemotePoolDriver>,
    request: &ProgramExecutionRequest,
    decoded: crate::DecodedExactStructuralStudyRoot,
    completed: Box<super::exact_meshing::CompletedRemoteExactMeshing>,
    signal_outcomes: &BTreeMap<
        runmat_analysis_fea::execution::SignalMaterializationIdentity,
        super::signal_materialization::RemoteSignalMaterializationOutcome,
    >,
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

    let unresolved_workloads_digest =
        crate::exact_structural_study_root::structural_workload_inventory_digest(
            &decoded.workloads,
        )?;
    let mut workloads = decoded.workloads.clone();
    let mut completed_signals = None;
    for workload in workloads.values_mut() {
        let completed_for_step = match super::signal_materialization::apply_to_workload(
            workload,
            &decoded.signal_materializations,
            signal_outcomes,
        ) {
            Ok(completed) => completed,
            Err(terminal) => return Ok(signal_composition(terminal)),
        };
        completed_signals.get_or_insert(completed_for_step);
        if let Err(error) = resolve_workload_boundary_aliases(workload, &resolution) {
            return Ok(RemoteExactStructuralStudyComposition::Failed(error));
        }
    }
    let completed_signals = completed_signals
        .ok_or_else(|| protocol("remote structural sequence contains no workload"))?;
    let sequence = match crate::execute_structural_sequence_remotely(
        Arc::clone(&pool),
        StructuralSequencePreparation {
            authored_study_digest: decoded.authored_study_digest,
            frozen_project_digest: decoded.frozen_project_digest,
            program_revision: request.recipe.program_revision.clone(),
            artifact_access: decoded.artifact_access.clone(),
            workloads,
            mesh: completed.result.artifact.clone(),
            chunk_bytes: DEFAULT_ANALYSIS_CHUNK_BYTES,
            inventory_limits: ObjectInventoryLimits::default(),
            required_capabilities: Default::default(),
        },
        crate::StructuralSequenceExecutionPolicy {
            resources: super::pool_resources::task_resources(resources)?,
            retry: RetryPolicy::ExplicitlyIdempotent {
                max_attempts: runmat_analysis_execution::DEFAULT_ANALYSIS_MAXIMUM_ATTEMPTS,
            },
            deadline_unix_millis: Some(super::pool_resources::task_deadline_unix_millis(
                resources,
            )?),
            priority: 0,
        },
    )
    .await
    {
        Ok(sequence) => sequence,
        Err(error) => return Ok(sequence_error_composition(&pool, error.to_string())),
    };
    let (response, objects) =
        crate::exact_structural_study_root::package_exact_structural_study_result(
            request,
            &decoded,
            &completed.result,
            &boundary_objects,
            &completed_signals,
            &sequence,
            unresolved_workloads_digest,
        )?;
    for object in objects {
        pool.register_execution_object(object.reference, object.bytes)?;
    }
    Ok(RemoteExactStructuralStudyComposition::Completed(response))
}

fn sequence_error_composition(
    pool: &RemotePoolDriver,
    message: String,
) -> RemoteExactStructuralStudyComposition {
    let snapshot = pool.snapshot();
    if snapshot
        .tasks
        .values()
        .any(|task| task.state == TaskState::Indeterminate)
    {
        RemoteExactStructuralStudyComposition::Indeterminate(message)
    } else if snapshot
        .tasks
        .values()
        .any(|task| task.state == TaskState::Cancelled)
    {
        RemoteExactStructuralStudyComposition::Cancelled
    } else {
        RemoteExactStructuralStudyComposition::Failed(message)
    }
}

fn signal_composition(
    outcome: super::signal_materialization::RemoteSignalMaterializationOutcome,
) -> RemoteExactStructuralStudyComposition {
    match outcome {
        super::signal_materialization::RemoteSignalMaterializationOutcome::Completed(_) => {
            RemoteExactStructuralStudyComposition::Failed(
                "completed signal materialization was used as a terminal outcome".into(),
            )
        }
        super::signal_materialization::RemoteSignalMaterializationOutcome::Failed(message) => {
            RemoteExactStructuralStudyComposition::Failed(message)
        }
        super::signal_materialization::RemoteSignalMaterializationOutcome::Cancelled => {
            RemoteExactStructuralStudyComposition::Cancelled
        }
        super::signal_materialization::RemoteSignalMaterializationOutcome::Indeterminate(
            message,
        ) => RemoteExactStructuralStudyComposition::Indeterminate(message),
    }
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
        let model = fixture_model(FixtureId::CantileverLinearStatic);
        let step_id = model.sequence.steps[0].step_id.clone();
        let mut workload = StructuralLinearWorkload {
            schema_version: 1,
            model,
            step_id,
            backend: ComputeBackend::Cpu,
            formulation_requirement: runmat_analysis_core::FormulationRequirement::Experimental,
            preconditioner: SpdPreconditionerKind::Jacobi,
            algebra_backend: LinearAlgebraBackendKind::CpuReference,
            require_solver_mesh_for_solid: true,
            initial_guess_transition: None,
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
