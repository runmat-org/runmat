//! Original solved temperature feeds the supported thermal-preload-modal chain.
use super::*;
use crate::execution::{StructuralStepWorkload, StructuralStudyWorkloads};
use crate::study::{
    plan_structural_study, plan_structural_study_with_fields, StructuralNumericalPlanningPolicy,
    StructuralStudyPlanningPolicy,
};

pub(super) fn verify_source_selection(
    original: &crate::study::CompiledAnalysisStudy,
    fields: &[InitialFieldBinding],
    thermal_policy: crate::study::ThermalStudyPlanningPolicy,
) {
    let mut compiled = original.clone();
    let heat = compiled.model.sequence.steps[0].clone();
    compiled.model.thermo_mechanical = Some(ThermoMechanicalDomain {
        reference_temperature: Temperature::new(293.15),
    });
    let mut preload = ProcedureStep::structural_preload("preload");
    preload.family = PhysicsFamily::Coupled(CoupledPhysicsFamily::Thermomechanical);
    preload.state_inputs.push(StateInput {
        input_id: "original-temperature-state".into(),
        required_kind: StateKind::Thermal,
        source: StateSource::Upstream {
            producer_step: heat.step_id.clone(),
            state_id: heat.state_outputs[0].state_id.clone(),
        },
        use_as: StateUse::Prestress,
        conflict: InitialStateConflictPolicy::Reject,
    });
    let consumer =
        ProcedureStep::structural_prestressed_modal("modes", "preload", "geometric_stiffness");
    compiled.model.sequence = AnalysisSequence::from_steps(
        "original-thermal-prestress",
        vec![heat.clone(), preload.clone(), consumer.clone()],
    );
    for dependency in &mut compiled.model.sequence.dependencies {
        dependency.reason = SequenceDependencyReason::StateTransfer;
    }
    compiled.model.sequence.transitions = vec![
        StateTransition {
            transition_id: TransitionId("thermal-preload".into()),
            producer_step: heat.step_id.clone(),
            consumer_step: preload.step_id,
            state_id: heat.state_outputs[0].state_id.clone(),
            method: StateTransitionMethod::Reuse,
            topology: TopologyTransitionPolicy::RequireIdentical,
            reference: ReferenceStatePolicy::Preserve,
        },
        StateTransition {
            transition_id: TransitionId("preload-modes".into()),
            producer_step: ProcedureStepId("preload".into()),
            consumer_step: consumer.step_id,
            state_id: StateId("geometric_stiffness".into()),
            method: StateTransitionMethod::Reuse,
            topology: TopologyTransitionPolicy::RequireIdentical,
            reference: ReferenceStatePolicy::Preserve,
        },
    ];
    let policy = StructuralStudyPlanningPolicy {
        backend: thermal_policy.backend,
        numerics: StructuralNumericalPlanningPolicy::production(),
    };
    assert!(plan_structural_study(&compiled, policy).is_err());
    let planned = plan_structural_study_with_fields(&compiled, policy, fields).unwrap();
    let cold = StructuralStudyWorkloads::from_canonical_bytes(&planned.canonical_bytes().unwrap())
        .unwrap();
    assert_eq!(planned, cold);
    let StructuralStudyWorkloads::HeterogeneousSequence { workloads } = cold else {
        panic!("thermal sequence")
    };
    assert_eq!(workloads.len(), 3);
    let StructuralStepWorkload::Thermal { workload } = &workloads[&heat.step_id] else {
        panic!("thermal producer")
    };
    assert_eq!(workload.initial_fields, fields);
    assert_eq!(workload.model, compiled.model);
    compiled.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(1);
    assert!(plan_structural_study_with_fields(&compiled, policy, fields).is_err());
}
