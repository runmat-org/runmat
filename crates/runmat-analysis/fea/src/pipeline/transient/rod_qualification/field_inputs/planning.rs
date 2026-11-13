//! The planner must preserve original solved sources before geometry-dependent admission.
use super::*;
use crate::execution::{StructuralStepWorkload, StructuralStudyWorkloads};
use crate::study::{
    plan_structural_study, plan_structural_study_with_fields, CompiledAnalysisStudy,
    StructuralNumericalPlanningPolicy, StructuralStudyPlanningPolicy,
};

pub(super) fn verify_original_sources(
    expected: &StructuralTransientWorkload,
    mesh: &SolverMeshArtifact,
) {
    let policy = StructuralStudyPlanningPolicy {
        backend: expected.backend,
        numerics: StructuralNumericalPlanningPolicy::production(),
    };
    let mut compiled = CompiledAnalysisStudy {
        module_path: "tests/original-rod.fea".into(),
        study_id: "original-rod".into(),
        study_inputs: Vec::new(),
        assessments: Vec::new(),
        outputs: Vec::new(),
        properties: crate::properties::compile_property_map(&expected.model, Some(mesh)).unwrap(),
        model: expected.model.clone(),
        solver_mesh_digest: mesh.canonical_digest,
    };
    assert!(plan_structural_study(&compiled, policy).is_err());
    assert!(
        plan_structural_study_with_fields(&compiled, policy, &expected.initial_fields[..1])
            .is_err()
    );
    let mut duplicate = expected.initial_fields.clone();
    duplicate.push(duplicate[0].clone());
    assert!(plan_structural_study_with_fields(&compiled, policy, &duplicate).is_err());
    let mut foreign = expected.initial_fields.clone();
    let mut reference = foreign[0].reference().clone();
    reference.artifact_id = "different-published-source".into();
    foreign[0] = InitialFieldBinding::new(reference, foreign[0].artifact().clone()).unwrap();
    assert!(plan_structural_study_with_fields(&compiled, policy, &foreign).is_err());
    let StructuralStudyWorkloads::StructuralTransient { workload, .. } =
        plan_structural_study_with_fields(&compiled, policy, &expected.initial_fields).unwrap()
    else {
        panic!("direct transient")
    };
    assert_eq!(*workload, *expected);
    let mut capped = compiled.clone();
    capped.model.sequence.steps[0]
        .accuracy
        .resources
        .maximum_resident_bytes = Some(1);
    assert!(plan_structural_study_with_fields(&capped, policy, &expected.initial_fields).is_err());
    let mut unselected = compiled.clone();
    unselected.model.initial_conditions.clear();
    assert!(
        plan_structural_study_with_fields(&unselected, policy, &expected.initial_fields).is_err()
    );

    let mut modes = ProcedureStep::structural_modal("modes");
    modes.independent_domain = IndependentDomain::Eigenvalue { requested_count: 2 };
    let mut response = compiled.model.sequence.steps[0].clone();
    let modal =
        ProcedureStep::structural_modal_transient(&response.step_id.0, "modes", "modal_basis");
    response.solution_space = modal.solution_space;
    response.state_inputs = modal.state_inputs;
    compiled.model.sequence =
        AnalysisSequence::from_steps("modal-original-fields", vec![modes, response]);
    compiled.model.sequence.dependencies[0].reason = SequenceDependencyReason::StateTransfer;
    compiled.model.sequence.transitions.push(StateTransition {
        transition_id: TransitionId("original-basis".into()),
        producer_step: ProcedureStepId("modes".into()),
        consumer_step: expected.step_id.clone(),
        state_id: StateId("modal_basis".into()),
        method: StateTransitionMethod::Reuse,
        topology: TopologyTransitionPolicy::RequireIdentical,
        reference: ReferenceStatePolicy::Preserve,
    });
    let planned =
        plan_structural_study_with_fields(&compiled, policy, &expected.initial_fields).unwrap();
    let cold = StructuralStudyWorkloads::from_canonical_bytes(&planned.canonical_bytes().unwrap())
        .unwrap();
    assert_eq!(cold, planned);
    let StructuralStudyWorkloads::HeterogeneousSequence { workloads } = cold else {
        panic!("modal sequence")
    };
    let StructuralStepWorkload::ModalTransient { template, .. } = &workloads[&expected.step_id]
    else {
        panic!("modal consumer")
    };
    assert_eq!(template.initial_fields, expected.initial_fields);
    assert!(matches!(
        workloads[&ProcedureStepId("modes".into())],
        StructuralStepWorkload::ModalExtraction { .. }
    ));
    // Sources selected on a producer that cannot consume them must not disappear.
    for condition in &mut compiled.model.initial_conditions {
        match condition {
            InitialCondition::Displacement(condition) => {
                condition.applies_before_step_id = "modes".into()
            }
            InitialCondition::Velocity(condition) => {
                condition.applies_before_step_id = "modes".into()
            }
            _ => unreachable!(),
        }
    }
    assert!(
        plan_structural_study_with_fields(&compiled, policy, &expected.initial_fields).is_err()
    );
}
