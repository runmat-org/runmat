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
            packaged(response, pool).map(Ok)
        }
        RemoteExactStructuralStudyComposition::Failed(message) => {
            execution_failure_outcome(pool, message)
        }
        RemoteExactStructuralStudyComposition::Cancelled => Ok(Err(
            crate::remote::pool_execution::RemoteExecutionStop::Cancelled,
        )),
        RemoteExactStructuralStudyComposition::Indeterminate(message) => Ok(Err(
            crate::remote::pool_execution::RemoteExecutionStop::Indeterminate(message),
        )),
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

    match &completed.regions {
        super::exact_meshing::MeshedGeometryRegions::PersistentEntities => {}
        super::exact_meshing::MeshedGeometryRegions::Materialized(regions) => {
            return Ok(RemoteExactStructuralStudyComposition::Failed(format!(
                "{} typed geometry regions require mesh-property compilation before structural execution",
                regions.len()
            )))
        }
    }
    let resolution = TopologyRegionResolution {
        schema_version: TopologyRegionResolution::SCHEMA_VERSION,
        geometry_source_digest: decoded.geometry_source_digest,
        solver_mesh_digest: Digest::from_bytes(*completed.result.artifact.canonical_digest.bytes()),
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
    workloads.replace_model(resolved_model).map_err(protocol)?;
    let task_resources = match admit_analysis_task_resources(
        resources,
        &decoded.analysis_accelerator_requests,
        decoded.local_execution,
        decoded.memory_per_worker_bytes,
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
        decoded: &decoded,
        completed: &completed,
        region_resolution_objects: &region_resolution_objects,
        completed_inputs: &completed_inputs,
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
    // The admission frame retains these immutable owners through the awaited
    // consumer. Forward references rather than moving their inline metadata
    // through every nested async dispatch frame.
    decoded: &'a crate::DecodedExactStructuralStudyRoot,
    completed: &'a super::exact_meshing::CompletedRemoteExactMeshing,
    region_resolution_objects: &'a runmat_analysis_execution::PreparedAnalysisObjects,
    completed_inputs: &'a [crate::input_materialization::CompletedInputMaterialization],
    unresolved_workloads_digest: Digest,
    policy: crate::StructuralSequenceExecutionPolicy,
    recovery: StructuralTransientRecoveryAdmission<'recovery>,
}

/// The task envelope of one analysis worker inside the remote allocation. As in local
/// placement, the execution policy's `memory_per_worker` bounds each worker's resident memory:
/// the request is min(allocation, memory_per_worker), and the solve path is planned within it.
pub(super) fn admit_analysis_task_resources(
    resources: &AllocationResources,
    expected_accelerators: &[runmat_execution::resource::AcceleratorRequest],
    local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
    memory_per_worker_bytes: Option<u64>,
) -> Result<runmat_execution::resource::ResourceRequest, String> {
    let mut task_resources =
        super::pool_resources::task_resources(resources).map_err(|error| error.to_string())?;
    if let Some(maximum) = memory_per_worker_bytes {
        task_resources.memory_bytes = task_resources.memory_bytes.min(maximum);
    }
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
        Ok(result) => match result.response {
            ProgramExecutionResponse::Failure { message } => {
                RemoteExactStructuralStudyComposition::Failed(message)
            }
            response => RemoteExactStructuralStudyComposition::Completed(response),
        },
        Err(crate::remote::pool_execution::RemoteExecutionStop::Cancelled) => {
            RemoteExactStructuralStudyComposition::Cancelled
        }
        Err(crate::remote::pool_execution::RemoteExecutionStop::Indeterminate(message)) => {
            RemoteExactStructuralStudyComposition::Indeterminate(message)
        }
    }
}

fn protocol(error: impl std::fmt::Display) -> NativeExecutionError {
    NativeExecutionError::Protocol(error.to_string())
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
        assert!(admit_analysis_task_resources(&resources(0), &[], serial, None).is_ok());
        let mut fractional = resources(0);
        fractional.cpu_millicores = 500;
        assert!(
            admit_analysis_task_resources(&fractional, &[], serial, None)
                .unwrap_err()
                .contains("allocated CPU differs")
        );
        let parallel = runmat_analysis_fea::planning::resource::LocalExecutionPlan::fixed(
            runmat_analysis_fea::planning::resource::LocalThreadCount::new(2).unwrap(),
        )
        .unwrap();
        assert!(
            admit_analysis_task_resources(&resources(0), &[], parallel, None)
                .unwrap_err()
                .contains("allocated CPU differs")
        );
        assert!(
            admit_analysis_task_resources(&resources(1), &[], serial, None)
                .unwrap_err()
                .contains("immutable structural-study request")
        );
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
            None,
        )
        .unwrap_err()
        .contains("immutable structural-study request"));
        assert_eq!(
            admit_analysis_task_resources(&resources(1), &[requested], serial, None)
                .unwrap()
                .accelerators
                .len(),
            1
        );
    }

    #[test]
    fn memory_per_worker_bounds_the_remote_analysis_task() {
        let serial = runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial();
        let allocation = resources(0);
        let unbounded = admit_analysis_task_resources(&allocation, &[], serial, None).unwrap();
        assert_eq!(unbounded.memory_bytes, allocation.memory_bytes);
        let capped =
            admit_analysis_task_resources(&allocation, &[], serial, Some(1 << 19)).unwrap();
        assert_eq!(capped.memory_bytes, 1 << 19);
        // Scratch and CPU stay the allocation's; only resident memory is the policy's.
        assert_eq!(capped.scratch_bytes, unbounded.scratch_bytes);
        assert_eq!(capped.cpu_millicores, unbounded.cpu_millicores);
        // A policy above the allocation never raises the request past what was granted.
        let above = admit_analysis_task_resources(&allocation, &[], serial, Some(1 << 30)).unwrap();
        assert_eq!(above.memory_bytes, allocation.memory_bytes);
    }
}
