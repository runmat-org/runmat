use runmat_analysis_core::{
    IndependentDomain, ProcedureStep, ReferenceStatePolicy, SequenceDependency,
    SequenceDependencyReason, StateId, StateTransition, StateTransitionMethod,
    TopologyTransitionPolicy, TransitionId,
};

use crate::execution::{
    StructuralFrequencySolverPlan, StructuralFrequencyWorkload, StructuralFrequencyWorkloadInput,
    StructuralModalTransientTemplate, StructuralModalWorkloadInput, StructuralTransientWorkload,
};
use crate::fixtures::{fixture_model, FixtureId};
use crate::frequency::FrequencyResponsePlan;
use crate::results::observation::ObservationExecutionPolicy;
use crate::solve::dynamics::DirectHarmonicPolicy;
use crate::solve::transient::StructuralTimeStepPolicy;
use crate::ComputeBackend;

use super::*;

fn heterogeneous_modal_sequence() -> BTreeMap<ProcedureStepId, StructuralStepWorkload> {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut modal = ProcedureStep::structural_modal("z_modes");
    modal.independent_domain = IndependentDomain::Eigenvalue { requested_count: 3 };
    let transient =
        ProcedureStep::structural_modal_transient("a_response", "z_modes", "modal_basis");
    model.sequence.steps = vec![modal, transient];
    model.sequence.dependencies = vec![SequenceDependency {
        predecessor: ProcedureStepId("z_modes".into()),
        successor: ProcedureStepId("a_response".into()),
        reason: SequenceDependencyReason::StateTransfer,
    }];
    model.sequence.transitions = vec![StateTransition {
        transition_id: TransitionId("modal-basis-transfer".into()),
        producer_step: ProcedureStepId("z_modes".into()),
        consumer_step: ProcedureStepId("a_response".into()),
        state_id: StateId("modal_basis".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    }];
    let modal = StructuralModalWorkload::new(StructuralModalWorkloadInput {
        model: model.clone(),
        step_id: ProcedureStepId("z_modes".into()),
        backend: ComputeBackend::Cpu,
        observation_execution: ObservationExecutionPolicy::scalable(),
        truncation: crate::ModalTruncationPolicy::requested(3),
        convergence: crate::ModalConvergencePolicy::exploratory(),
        residual_vector_directions: Vec::new(),
        prestress_transition: None,
    })
    .unwrap();
    let transient = StructuralModalTransientTemplate {
        schema_version: 2,
        initial_fields: Vec::new(),
        model,
        step_id: ProcedureStepId("a_response".into()),
        backend: ComputeBackend::Cpu,
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 8 },
        residual_target: 1.0e-10,
    };
    BTreeMap::from([
        (
            ProcedureStepId("a_response".into()),
            StructuralStepWorkload::ModalTransient {
                template: Box::new(transient),
                block_plan: StructuralTransientBlockPlan::new(3).unwrap(),
            },
        ),
        (
            ProcedureStepId("z_modes".into()),
            StructuralStepWorkload::ModalExtraction {
                workload: Box::new(modal),
            },
        ),
    ])
}

#[test]
fn transient_block_plan_is_bounded_and_closes_the_terminal_schedule() {
    let plan = StructuralTransientBlockPlan::new(3).unwrap();
    assert_eq!(plan.terminal_schedule_ordinals(8).unwrap(), vec![3, 6, 8]);
    assert!(plan.terminal_schedule_ordinals(0).is_err());
    assert!(StructuralTransientBlockPlan::new(0).is_err());
    assert!(StructuralTransientBlockPlan::new(MAXIMUM_TRANSIENT_BLOCK_INTERVALS + 1).is_err());
}

#[test]
fn direct_transient_workloads_are_strongly_typed_and_model_replaceable() {
    let model = fixture_model(FixtureId::StructuralTransientTet4Reference);
    let workload = StructuralTransientWorkload {
        schema_version: 2,
        initial_fields: Vec::new(),
        step_id: model.sequence.steps[0].step_id.clone(),
        model,
        backend: ComputeBackend::Cpu,
        solver: crate::execution::StructuralTransientSolverPlan::Direct,
        time_step_policy: StructuralTimeStepPolicy::Fixed { interval_count: 8 },
        residual_target: 1.0e-10,
    };
    let mut workloads = StructuralStudyWorkloads::StructuralTransient {
        workload: Box::new(workload),
        block_plan: StructuralTransientBlockPlan::new(3).unwrap(),
    };
    let digest = workloads.canonical_digest().unwrap();
    let mut replacement = workloads.authored_model().unwrap().clone();
    replacement.model_id.0 = "replacement-model".into();
    workloads.replace_model(replacement.clone()).unwrap();
    assert_eq!(workloads.authored_model().unwrap(), &replacement);
    assert_ne!(workloads.canonical_digest().unwrap(), digest);
}

#[test]
fn structural_frequency_workload_is_canonical_and_model_replaceable() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut step = ProcedureStep::structural_harmonic("frequency");
    step.independent_domain = IndependentDomain::FrequencyHertz(vec![20.0, 40.0]);
    model.sequence.steps = vec![step];
    let workload = StructuralFrequencyWorkload::new(StructuralFrequencyWorkloadInput {
        model,
        step_id: ProcedureStepId("frequency".into()),
        backend: ComputeBackend::Cpu,
        observation_execution: ObservationExecutionPolicy::scalable(),
        plan: FrequencyResponsePlan::new(20.0, &[40.0], &[]).unwrap(),
        solver: StructuralFrequencySolverPlan::Direct {
            policy: DirectHarmonicPolicy::default(),
        },
        prestress_transition: None,
    })
    .unwrap();
    let mut workloads = StructuralStudyWorkloads::StructuralFrequency {
        workload: Box::new(workload),
    };
    let canonical = workloads.canonical_digest().unwrap();
    assert_eq!(
        workloads.step_ids().unwrap(),
        [ProcedureStepId("frequency".into())]
    );
    assert_eq!(workloads.backends().unwrap(), [ComputeBackend::Cpu]);

    let mut replacement = workloads.authored_model().unwrap().clone();
    replacement.model_id.0 = "resolved-frequency-model".into();
    workloads.replace_model(replacement.clone()).unwrap();
    assert_eq!(workloads.authored_model().unwrap(), &replacement);
    assert_ne!(workloads.canonical_digest().unwrap(), canonical);
}

#[test]
fn standalone_structural_frequency_rejects_a_mixed_sequence() {
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    let mut frequency = ProcedureStep::structural_harmonic("frequency");
    frequency.independent_domain = IndependentDomain::FrequencyHertz(vec![20.0]);
    model.sequence.steps.push(frequency.clone());
    let workload = StructuralFrequencyWorkload::new(StructuralFrequencyWorkloadInput {
        model,
        step_id: frequency.step_id,
        backend: ComputeBackend::Cpu,
        observation_execution: ObservationExecutionPolicy::scalable(),
        plan: FrequencyResponsePlan::new(20.0, &[], &[]).unwrap(),
        solver: StructuralFrequencySolverPlan::Direct {
            policy: DirectHarmonicPolicy::default(),
        },
        prestress_transition: None,
    })
    .unwrap();
    assert!(StructuralStudyWorkloads::StructuralFrequency {
        workload: Box::new(workload),
    }
    .validate()
    .is_err());
}

#[test]
fn heterogeneous_inventory_round_trips_and_reports_authored_order() {
    let study = StructuralStudyWorkloads::HeterogeneousSequence {
        workloads: heterogeneous_modal_sequence(),
    };
    assert_eq!(
        study.step_ids().unwrap(),
        [
            ProcedureStepId("z_modes".into()),
            ProcedureStepId("a_response".into())
        ]
    );
    let bytes = study.canonical_bytes().unwrap();
    let restored = StructuralStudyWorkloads::from_canonical_bytes(&bytes).unwrap();
    assert_eq!(restored, study);
    assert_eq!(restored.canonical_digest().unwrap(), Digest::sha256(bytes));
}

#[test]
fn failed_model_replacement_preserves_the_complete_workload_inventory() {
    let mut study = StructuralStudyWorkloads::HeterogeneousSequence {
        workloads: heterogeneous_modal_sequence(),
    };
    let original = study.clone();
    let mut invalid_model = study.authored_model().unwrap().clone();
    invalid_model.sequence.steps[0].state_outputs.clear();

    assert!(study.replace_model(invalid_model).is_err());
    assert_eq!(study, original);
    study.validate().unwrap();
}

#[test]
fn heterogeneous_inventory_rejects_missing_and_substituted_steps() {
    let mut missing = heterogeneous_modal_sequence();
    missing.remove(&ProcedureStepId("a_response".into()));
    assert!(
        StructuralStudyWorkloads::HeterogeneousSequence { workloads: missing }
            .validate()
            .is_err()
    );

    let mut substituted = heterogeneous_modal_sequence();
    let response = substituted
        .remove(&ProcedureStepId("a_response".into()))
        .unwrap();
    substituted.insert(ProcedureStepId("z_modes".into()), response);
    assert!(StructuralStudyWorkloads::HeterogeneousSequence {
        workloads: substituted,
    }
    .validate()
    .is_err());
}

#[test]
fn modal_transient_inventory_is_an_unbound_template() {
    let workloads = heterogeneous_modal_sequence();
    let StructuralStepWorkload::ModalTransient { template, .. } =
        &workloads[&ProcedureStepId("a_response".into())]
    else {
        panic!("response step must remain a modal-transient template");
    };
    template.validate().unwrap();
    let encoded = serde_json::to_value(&workloads).unwrap();
    let encoded = encoded.to_string();
    assert!(!encoded.contains("basis_transition"));
    assert!(!encoded.contains("source_state"));
}
