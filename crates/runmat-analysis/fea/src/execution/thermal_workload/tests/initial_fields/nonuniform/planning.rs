//! Exercise original solved temperature inputs through shared authored product admission.
use super::*;
mod mixed;
pub(super) fn through_authored_product(
    mut expected: ThermalWorkload,
    mesh: &SolverMeshArtifact,
) -> ThermalWorkload {
    for criterion in &mut expected.model.sequence.steps[0].accuracy.criteria {
        if criterion.contribution == ErrorContributionKind::Algebraic {
            criterion.tolerance = AccuracyTolerance::Relative(expected.residual_target);
        }
    }
    let compiled = crate::study::CompiledAnalysisStudy {
        module_path: "tests/original-temperature.fea".into(),
        study_id: "original-temperature".into(),
        study_inputs: Vec::new(),
        outputs: Vec::new(),
        assessments: Vec::new(),
        properties: crate::properties::compile_property_map(&expected.model, Some(mesh)).unwrap(),
        model: expected.model.clone(),
        solver_mesh_digest: mesh.canonical_digest,
    };
    let policy = crate::study::ThermalStudyPlanningPolicy {
        backend: expected.backend,
        element_heat_source: None,
    };
    mixed::verify_source_selection(&compiled, &expected.initial_fields, policy.clone());
    assert!(crate::study::plan_thermal_study(&compiled, policy.clone()).is_err());
    let product = crate::study::plan_thermal_study_product_with_fields(
        compiled,
        mesh,
        policy,
        expected.initial_fields.clone(),
    )
    .unwrap();
    assert_eq!(product.workload, expected);
    product.workload
}
