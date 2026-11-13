use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_core::{
    AnalysisSequence, IndependentDomain, ProcedureStep, ProcedureStepId, ReferenceStatePolicy,
    SequenceDependency, SequenceDependencyReason, StateId, StateTransition, StateTransitionMethod,
    TopologyTransitionPolicy, TransitionId,
};
use runmat_analysis_execution::StructuralModalTransientStudyPreparation;
use runmat_analysis_fea::{
    execution::{
        StructuralModalTransientTemplate, StructuralModalWorkload, StructuralModalWorkloadInput,
        StructuralStepWorkload, StructuralStudyWorkloads, StructuralTransientBlockPlan,
    },
    pipeline::modal_governance::{ModalConvergencePolicy, ModalTruncationPolicy},
    results::observation::ObservationExecutionPolicy,
    solve::transient::StructuralTimeStepPolicy,
    ComputeBackend,
};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::{object::ObjectInventoryLimits, ArtifactAccess};

use super::transient::structural_transient_fixture;

pub(super) fn preparation(
    revision: ProgramRevision,
    authorization_scope: &str,
    encryption_context: &str,
) -> StructuralModalTransientStudyPreparation {
    let (mut model, mesh) = structural_transient_fixture();
    let mut modal_step = ProcedureStep::structural_modal("modes");
    modal_step.independent_domain = IndependentDomain::Eigenvalue { requested_count: 3 };
    let mut transient_step =
        ProcedureStep::structural_modal_transient("response", "modes", "modal_basis");
    transient_step.independent_domain = IndependentDomain::TimeSeconds {
        origin: 0.0,
        start: 0.0,
        end: 1.0,
        interval_count: 10,
    };
    let mut sequence = AnalysisSequence::from_steps(
        "process-modal-transient",
        vec![modal_step.clone(), transient_step.clone()],
    );
    sequence.dependencies = vec![SequenceDependency {
        predecessor: modal_step.step_id.clone(),
        successor: transient_step.step_id.clone(),
        reason: SequenceDependencyReason::StateTransfer,
    }];
    sequence.transitions = vec![StateTransition {
        transition_id: TransitionId("modal-basis-to-transient".into()),
        producer_step: modal_step.step_id.clone(),
        consumer_step: transient_step.step_id.clone(),
        state_id: StateId("modal_basis".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    }];
    model.sequence = sequence;
    let modal = StructuralModalWorkload::new(StructuralModalWorkloadInput {
        model: model.clone(),
        step_id: modal_step.step_id.clone(),
        backend: ComputeBackend::Cpu,
        observation_execution: ObservationExecutionPolicy::scalable(),
        truncation: ModalTruncationPolicy::requested(3),
        convergence: ModalConvergencePolicy::exploratory(),
        residual_vector_directions: Vec::new(),
        prestress_transition: None,
    })
    .unwrap();
    let transient = StructuralModalTransientTemplate {
        schema_version: 2,
        initial_fields: Vec::new(),
        model,
        step_id: transient_step.step_id.clone(),
        backend: ComputeBackend::Cpu,
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 4 },
        residual_target: 1.0e-8,
    };
    StructuralModalTransientStudyPreparation {
        local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan::serial(),
        authored_study_digest: Digest::sha256(b"process-modal-transient-study"),
        frozen_project_digest: None,
        workloads: StructuralStudyWorkloads::HeterogeneousSequence {
            workloads: BTreeMap::from([
                (
                    ProcedureStepId("modes".into()),
                    StructuralStepWorkload::ModalExtraction {
                        workload: Box::new(modal),
                    },
                ),
                (
                    ProcedureStepId("response".into()),
                    StructuralStepWorkload::ModalTransient {
                        template: Box::new(transient),
                        block_plan: StructuralTransientBlockPlan::new(2).unwrap(),
                    },
                ),
            ]),
        },
        mesh,
        artifact_access: ArtifactAccess {
            authorization_scope: authorization_scope.into(),
            encryption_context: Digest::sha256(encryption_context),
        },
        program_revision: revision,
        chunk_bytes: 256,
        inventory_limits: ObjectInventoryLimits::default(),
        required_capabilities: BTreeSet::new(),
        accelerator_requests: Vec::new(),
    }
}

pub(super) fn assert_numerical_equality(
    actual: &runmat_execution_runner_native::CompletedStructuralModalTransientStudy,
    expected: &runmat_execution_runner_native::CompletedStructuralModalTransientStudy,
) {
    assert_eq!(actual.modal.result, expected.modal.result);
    assert_eq!(
        actual.modal.numerical_receipt,
        expected.modal.numerical_receipt
    );
    assert_eq!(actual.basis_state.state, expected.basis_state.state);
    assert_eq!(
        actual.transient.result.result,
        expected.transient.result.result
    );
    assert_eq!(
        actual.transient.result.numerical_receipt,
        expected.transient.result.numerical_receipt
    );
    assert_eq!(
        actual.basis_state.state.source_result,
        actual.modal.numerical_receipt.source_result()
    );
    assert!(actual
        .transient
        .result
        .numerical_receipt
        .blocks
        .iter()
        .all(|block| block.solver_method
            == runmat_analysis_fea::contracts::SolverMethod::StructuralModalTransientNewmarkDenseFactorization));
}
