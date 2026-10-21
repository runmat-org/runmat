//! Recording substrate tests execute real numerical programs; reserved accounting is test data.
use super::*;
use crate::analysis_thermal::tests::prestress;
use runmat_analysis_fea::{
    planning::resource::LocalExecutionPlan,
    study::{
        plan_structural_study, StructuralNumericalPlanningPolicy, StructuralStudyPlanningPolicy,
    },
};
use runmat_execution::{resource::ResourceRequest, task::RetryPolicy, Digest, TaskId};
use runmat_meshing_core::ElementOrder;
pub(crate) mod pool;

pub(crate) fn preparation(branch: usize, order: ElementOrder) -> ThermalPrestressStudyPreparation {
    let heat = match branch {
        0 => prestress::setup(order, true),
        1 => prestress::setup(order, false),
        2 => prestress::frequency::frequency_setup(order),
        _ => prestress::frequency::reduced::reduced_setup(order),
    };
    let workloads = plan_structural_study(
        &heat.product.compiled,
        StructuralStudyPlanningPolicy {
            backend: runmat_analysis_fea::ComputeBackend::Cpu,
            numerics: StructuralNumericalPlanningPolicy::production(),
        },
    )
    .unwrap();
    ThermalPrestressStudyPreparation {
        structural: StructuralPrestressedFrequencyStudyPreparation {
            authored_study_digest: heat.product.compiled.definition_digest().unwrap(),
            frozen_project_digest: heat.frozen_project_digest,
            workloads,
            mesh: heat.mesh,
            artifact_access: heat.artifact_access,
            program_revision: heat.program_revision,
            chunk_bytes: heat.output_policy.chunk_bytes(),
            inventory_limits: heat.output_policy.inventory_limits(),
            required_capabilities: heat.required_capabilities,
            accelerator_requests: Vec::new(),
            local_execution: LocalExecutionPlan::serial(),
        },
        compiled: heat.product.compiled,
    }
}

#[tokio::test]
async fn same_pool_named_thermal_publication_precedes_all_four_consumer_branches() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        for branch in 0..4 {
            let preparation = preparation(branch, order);
            let directory = tempfile::tempdir().unwrap();
            let mut pool = pool::RecordingPool::new(directory.path());
            let completion = execute_with_pool(
                &mut pool,
                preparation,
                StructuralSequenceExecutionPolicy {
                    resources: ResourceRequest {
                        cpu_millicores: 1000,
                        memory_bytes: 64 * 1024 * 1024,
                        scratch_bytes: 0,
                        max_wall_millis: 60_000,
                        max_artifact_bytes: 64 * 1024 * 1024,
                        max_egress_bytes: 0,
                        max_relay_bytes: 0,
                        accelerators: Vec::new(),
                        required_capabilities: Default::default(),
                    },
                    retry: RetryPolicy::IdempotentInfrastructure,
                    deadline_unix_millis: None,
                    priority: 0,
                },
            )
            .await
            .unwrap();
            let sealed = ThermalPrestressSource::import(
                &pool,
                completion.thermal.preparation(),
                completion.thermal.result_closures().to_vec(),
            )
            .unwrap();
            let selected = sealed
                .state_publication_dependency(&runmat_analysis_core::StateId(
                    "second-temperature".into(),
                ))
                .unwrap();
            let selected_task = TaskId::derive(&[selected.digest().bytes()]);
            // Five thermal tasks: solve, accuracy, both named states, terminal publication.
            // Replay/sealing adds no execution, and preload names the second state, not receipt.
            assert_eq!(pool.tasks[3].request.id, selected_task);
            assert_eq!(
                pool.tasks[5].dependencies,
                [selected_task].into_iter().collect()
            );
            assert_ne!(pool.tasks[4].request.id, selected_task);
            let preload = match &completion.structural {
                ThermalPrestressStructuralCompletion::Buckling(value) => &value.preload,
                ThermalPrestressStructuralCompletion::Modal(value) => &value.preload,
                ThermalPrestressStructuralCompletion::Frequency(value) => &value.preload,
                ThermalPrestressStructuralCompletion::ModalFrequency(value) => &value.preload,
            };
            assert_eq!(
                preload
                    .result
                    .linearization
                    .as_ref()
                    .unwrap()
                    .thermal_source()
                    .unwrap()
                    .transition()
                    .source_state,
                completion.thermal.states[1].state.identity
            );
            assert!(pool.tasks.len() >= 8);
            let task_count = pool.tasks.len();
            let wrong_pool = BoundThermalExecution {
                source: sealed,
                dependency: AnalysisTaskDependency {
                    identity: selected,
                    task_id: pool.tasks[3].request.id,
                },
                scope_id: pool.scope_id(),
                pool_id: PoolId::derive(&[b"foreign-pool"]),
            };
            assert!(wrong_pool.into_parts(&pool).is_err());
            assert_eq!(pool.tasks.len(), task_count);
        }
    }
}
