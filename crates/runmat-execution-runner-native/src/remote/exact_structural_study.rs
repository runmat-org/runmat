//! Driver-owned whole-study composition for exact CAD and structural-static analysis.

use std::collections::BTreeMap;
use std::sync::Arc;

use runmat_analysis_execution::{
    StructuralSequencePreparation, StructuralStudyWorkloads, StructuralTransientRecoveryClosure,
    StructuralTransientSequencePreparationInput,
};
use runmat_analysis_fea::execution::{StructuralFrequencySolverPlan, DEFAULT_ANALYSIS_CHUNK_BYTES};
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
use crate::StructuralTransientRecoveryPublisher;
use crate::{NativeExecutionError, NativeExecutionResult, TopologyRegionResolution};

mod heterogeneous;
mod standalone;
mod thermal;

#[derive(Clone, Debug)]
pub(crate) enum RemoteExactStructuralStudyComposition {
    Completed(ProgramExecutionResponse),
    Failed(String),
    Cancelled,
    Indeterminate(String),
}

pub(super) enum StructuralTransientRecoveryAdmission<'a> {
    Unavailable,
    Available {
        restored: Option<&'a super::recovery::RestoredObjectClosure>,
        publisher: &'a mut dyn StructuralTransientRecoveryPublisher,
    },
}

pub(super) async fn execute(
    pool: Arc<RemotePoolDriver>,
    request: ProgramExecutionRequest,
    run_id: String,
    resources: AllocationResources,
    recovered: Option<&super::recovery::RestoredObjectClosure>,
    recovery_publisher: &mut super::recovery::RemoteRecoveryPublisher<'_>,
) -> NativeExecutionResult<RemotePoolExecutionOutcome> {
    match execute_for_composition(
        Arc::clone(&pool),
        &request,
        &run_id,
        &resources,
        recovered,
        Some(recovery_publisher),
    )
    .await?
    {
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
    recovered: Option<&super::recovery::RestoredObjectClosure>,
    recovery_publisher: Option<&mut dyn StructuralTransientRecoveryPublisher>,
) -> NativeExecutionResult<RemoteExactStructuralStudyComposition> {
    let decoded = crate::decode_exact_structural_study_root(request, pool.as_ref())?;
    let artifact_access = decoded.artifact_access.clone();
    let meshing = super::exact_meshing::execute_for_composition(
        Arc::clone(&pool),
        &decoded.exact_meshing,
        run_id,
        resources,
    );
    let inputs = super::input_materialization::execute_ready(
        Arc::clone(&pool),
        &decoded.input_materializations,
        &artifact_access,
        resources,
    );
    let (meshing, input_outcomes) = tokio::join!(meshing, inputs);
    let completed = match meshing? {
        RemoteExactMeshingComposition::Completed(completed) => completed,
        RemoteExactMeshingComposition::Terminal(outcome) => return Ok(composition(*outcome)),
    };
    let mut input_outcomes = input_outcomes?;
    input_outcomes.extend(
        super::input_materialization::execute_deferred(
            Arc::clone(&pool),
            &decoded.input_materializations,
            &completed.result.artifact,
            &artifact_access,
            resources,
        )
        .await?,
    );
    let recovery = match recovery_publisher {
        Some(publisher) => StructuralTransientRecoveryAdmission::Available {
            restored: recovered,
            publisher,
        },
        None if recovered.is_none() => StructuralTransientRecoveryAdmission::Unavailable,
        None => {
            return Err(protocol(
                "restored structural transient state requires a recovery publisher",
            ))
        }
    };
    Box::pin(execute_with_completed_meshing(
        pool,
        request,
        decoded,
        completed,
        &input_outcomes,
        resources,
        recovery,
    ))
    .await
}

#[cfg(all(test, feature = "occt-native"))]
pub(crate) async fn execute_for_test(
    pool: Arc<RemotePoolDriver>,
    request: &ProgramExecutionRequest,
    run_id: &str,
    resources: &AllocationResources,
    publisher: &mut dyn StructuralTransientRecoveryPublisher,
) -> NativeExecutionResult<Result<ProgramExecutionResponse, String>> {
    Ok(
        match execute_for_composition(pool, request, run_id, resources, None, Some(publisher))
            .await?
        {
            RemoteExactStructuralStudyComposition::Completed(response) => Ok(response),
            RemoteExactStructuralStudyComposition::Failed(message)
            | RemoteExactStructuralStudyComposition::Indeterminate(message) => Err(message),
            RemoteExactStructuralStudyComposition::Cancelled => {
                Err("exact structural study was cancelled".into())
            }
        },
    )
}

pub(super) async fn execute_with_completed_meshing(
    pool: Arc<RemotePoolDriver>,
    request: &ProgramExecutionRequest,
    decoded: crate::DecodedExactStructuralStudyRoot,
    completed: Box<super::exact_meshing::CompletedRemoteExactMeshing>,
    input_outcomes: &BTreeMap<
        runmat_execution::Digest,
        super::input_materialization::RemoteInputMaterializationOutcome,
    >,
    resources: &AllocationResources,
    recovery: StructuralTransientRecoveryAdmission<'_>,
) -> NativeExecutionResult<RemoteExactStructuralStudyComposition> {
    if completed.result.artifact.geometry.source_digest.bytes()
        != decoded.geometry_source_digest.bytes()
    {
        return Ok(RemoteExactStructuralStudyComposition::Failed(
            "meshed geometry differs from the whole-study source identity".into(),
        ));
    }

    let aliases = match &completed.regions {
        super::exact_meshing::MeshedGeometryRegions::PersistentEntities(aliases) => aliases,
        super::exact_meshing::MeshedGeometryRegions::Materialized(regions) => {
            return Ok(RemoteExactStructuralStudyComposition::Failed(format!(
                "{} typed geometry regions require mesh-property compilation before structural execution",
                regions.len()
            )))
        }
    };
    let resolution = TopologyRegionResolution {
        schema_version: 1,
        geometry_source_digest: decoded.geometry_source_digest,
        solver_mesh_digest: Digest::from_bytes(*completed.result.artifact.canonical_digest.bytes()),
        aliases: aliases.clone(),
    };
    resolution.validate()?;
    let region_resolution_objects =
        crate::exact_structural_study_root::prepare_topology_resolution_objects(
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
    let mut resolved_model = workloads.authored_model().map_err(protocol)?.clone();
    let input_materializations = decoded
        .input_materializations
        .iter()
        .map(|plan| plan.resolve(&completed.result.artifact))
        .collect::<Result<Vec<_>, _>>()
        .map_err(protocol)?;
    let completed_inputs = match super::input_materialization::apply_to_model(
        &mut resolved_model,
        &input_materializations,
        input_outcomes,
    ) {
        Ok(completed) => completed,
        Err(terminal) => return Ok(signal_composition(terminal)),
    };
    if let Err(error) = resolve_model_boundary_aliases(&mut resolved_model, &resolution) {
        return Ok(RemoteExactStructuralStudyComposition::Failed(error));
    }
    workloads.replace_model(resolved_model).map_err(protocol)?;
    let task_resources = match admit_analysis_task_resources(
        resources,
        &decoded.analysis_accelerator_requests,
        decoded.local_execution,
    ) {
        Ok(resources) => resources,
        Err(error) => return Ok(RemoteExactStructuralStudyComposition::Failed(error)),
    };
    let policy = crate::StructuralSequenceExecutionPolicy {
        resources: task_resources,
        retry: RetryPolicy::ExplicitlyIdempotent {
            max_attempts: runmat_analysis_execution::DEFAULT_ANALYSIS_MAXIMUM_ATTEMPTS,
        },
        deadline_unix_millis: Some(super::pool_resources::task_deadline_unix_millis(resources)?),
        priority: 0,
    };
    if matches!(&workloads, StructuralStudyWorkloads::HeterogeneousSequence { workloads }
        if workloads.values().any(|step| matches!(step, runmat_analysis_fea::execution::StructuralStepWorkload::Thermal { .. })))
    {
        if matches!(
            &recovery,
            StructuralTransientRecoveryAdmission::Available {
                restored: Some(_),
                ..
            }
        ) {
            return Err(protocol(
                "thermal prestress cannot consume a structural transient recovery closure",
            ));
        }
        return thermal::execute(thermal::Composition {
            pool,
            request,
            decoded: &decoded,
            meshing: &completed.result,
            region_resolution: &region_resolution_objects,
            inputs: &completed_inputs,
            workloads,
            unresolved_workloads_digest,
            policy,
        })
        .await;
    }
    // Keep each family's poll frame separate: large unrelated analysis futures must not
    // consume the default worker stack while another family is executing.
    let input = Composition {
        pool,
        request,
        decoded,
        completed,
        region_resolution_objects,
        completed_inputs,
        unresolved_workloads_digest,
        policy,
        recovery,
    };
    match workloads {
        StructuralStudyWorkloads::HeterogeneousSequence { workloads } => {
            Box::pin(heterogeneous::heterogeneous(input, workloads)).await
        }
        StructuralStudyWorkloads::StructuralModal { workload } => {
            Box::pin(standalone::modal(input, workload)).await
        }
        StructuralStudyWorkloads::LinearSequence { workloads } => {
            Box::pin(standalone::linear(input, workloads)).await
        }
        StructuralStudyWorkloads::StructuralTransient {
            workload,
            block_plan,
        } => Box::pin(standalone::transient(input, workload, block_plan)).await,
        StructuralStudyWorkloads::StructuralFrequency { workload } => {
            Box::pin(standalone::frequency(input, workload)).await
        }
    }
}

struct Composition<'a, 'recovery> {
    pool: Arc<RemotePoolDriver>,
    request: &'a ProgramExecutionRequest,
    decoded: crate::DecodedExactStructuralStudyRoot,
    completed: Box<super::exact_meshing::CompletedRemoteExactMeshing>,
    region_resolution_objects: runmat_analysis_execution::PreparedAnalysisObjects,
    completed_inputs: Vec<crate::input_materialization::CompletedInputMaterialization>,
    unresolved_workloads_digest: Digest,
    policy: crate::StructuralSequenceExecutionPolicy,
    recovery: StructuralTransientRecoveryAdmission<'recovery>,
}

pub(super) fn admit_analysis_task_resources(
    resources: &AllocationResources,
    expected_accelerators: &[runmat_execution::resource::AcceleratorRequest],
    local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
) -> Result<runmat_execution::resource::ResourceRequest, String> {
    let task_resources =
        super::pool_resources::task_resources(resources).map_err(|error| error.to_string())?;
    if task_resources.accelerators != expected_accelerators {
        return Err(
            "allocated accelerators differ from the immutable structural-study request".into(),
        );
    }
    let expected_cpu = local_execution
        .checked_cpu_millicores()
        .map_err(|error| error.to_string())?;
    if task_resources.cpu_millicores != expected_cpu {
        return Err("allocated CPU differs from the immutable structural-study request".into());
    }
    Ok(task_resources)
}

#[cfg(test)]
mod allocation_tests {
    use super::*;

    fn resources(accelerator_count: u32) -> AllocationResources {
        AllocationResources {
            cpu_millicores: 1_000,
            memory_bytes: 1 << 20,
            scratch_bytes: 1 << 20,
            accelerators: if accelerator_count == 0 {
                Vec::new()
            } else {
                vec![runmat_execution::resource::AcceleratorRequest {
                    class: runmat_execution::resource::AcceleratorClass::new("gpu.test").unwrap(),
                    count: u16::try_from(accelerator_count).unwrap(),
                    minimum_allocation_bytes: 1 << 18,
                    provider: None,
                    required_features: Default::default(),
                }]
            },
            maximum_wall_millis: 10_000,
        }
    }

    #[test]
    fn remote_allocation_must_match_the_immutable_study_request() {
        let serial = runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial();
        assert!(admit_analysis_task_resources(&resources(0), &[], serial).is_ok());
        let mut fractional = resources(0);
        fractional.cpu_millicores = 500;
        assert!(admit_analysis_task_resources(&fractional, &[], serial)
            .unwrap_err()
            .contains("allocated CPU differs"));
        let parallel = runmat_analysis_fea::planning::resource::LocalExecutionPlan::fixed(
            runmat_analysis_fea::planning::resource::LocalThreadCount::new(2).unwrap(),
        )
        .unwrap();
        assert!(admit_analysis_task_resources(&resources(0), &[], parallel)
            .unwrap_err()
            .contains("allocated CPU differs"));
        assert!(admit_analysis_task_resources(&resources(1), &[], serial)
            .unwrap_err()
            .contains("immutable structural-study request"));
        let requested = runmat_execution::resource::AcceleratorRequest {
            class: runmat_execution::resource::AcceleratorClass::new("gpu.test").unwrap(),
            count: 1,
            minimum_allocation_bytes: 1 << 18,
            provider: None,
            required_features: Default::default(),
        };
        assert!(admit_analysis_task_resources(
            &resources(0),
            std::slice::from_ref(&requested),
            serial,
        )
        .unwrap_err()
        .contains("immutable structural-study request"));
        assert_eq!(
            admit_analysis_task_resources(&resources(1), &[requested], serial)
                .unwrap()
                .accelerators
                .len(),
            1
        );
    }
}

pub(super) fn sequence_error_composition(
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

pub(super) fn signal_composition(
    outcome: super::input_materialization::RemoteInputMaterializationOutcome,
) -> RemoteExactStructuralStudyComposition {
    match outcome {
        super::input_materialization::RemoteInputMaterializationOutcome::Completed(_) => {
            RemoteExactStructuralStudyComposition::Failed(
                "completed signal materialization was used as a terminal outcome".into(),
            )
        }
        super::input_materialization::RemoteInputMaterializationOutcome::Failed(message) => {
            RemoteExactStructuralStudyComposition::Failed(message)
        }
        super::input_materialization::RemoteInputMaterializationOutcome::Cancelled => {
            RemoteExactStructuralStudyComposition::Cancelled
        }
        super::input_materialization::RemoteInputMaterializationOutcome::Indeterminate(message) => {
            RemoteExactStructuralStudyComposition::Indeterminate(message)
        }
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

fn resolve_model_boundary_aliases(
    model: &mut runmat_analysis_core::AnalysisModel,
    resolution: &TopologyRegionResolution,
) -> Result<(), String> {
    let aliases = runmat_analysis_core::GeometryRegionAliases::new(resolution.aliases.clone())
        .map_err(|error| error.to_string())?;
    *model = runmat_analysis_core::resolve_geometry_region_aliases(model, &aliases)
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
            schema_version: 2,
            model,
            step_id,
            backend: ComputeBackend::Cpu,
            preconditioner: SpdPreconditionerKind::Auto,
            algebra_backend: LinearAlgebraBackendKind::CpuReference,
            observation_execution:
                runmat_analysis_fea::results::observation::ObservationExecutionPolicy::scalable(),
            require_solver_mesh_for_solid: true,
            state_input: None,
        };
        workload.model.constraints[0].visit_targets_mut(|target| {
            *target = runmat_analysis_core::RegionTarget::boundary("face_000002")
        });
        let resolution = TopologyRegionResolution {
            schema_version: 1,
            geometry_source_digest: Digest::sha256(b"geometry"),
            solver_mesh_digest: Digest::sha256(b"mesh"),
            aliases: BTreeMap::from([(
                "face_000001".into(),
                PersistentEntityId::new(
                    PersistentEntityKind::Face,
                    "face-a".into(),
                    vec!["root".into()],
                ),
            )]),
        };

        let error = resolve_model_boundary_aliases(&mut workload.model, &resolution).unwrap_err();
        assert!(error.contains("face_000002"));
    }
}
