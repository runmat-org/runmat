use super::*;
use crate::physics::magnetic::test_support::authored_model;
use runmat_analysis_core::{AnalysisConstraint, MagneticConstraintKind, RegionTarget};
use runmat_meshing_core::{ElementOrder, SolverMeshArtifact};

fn mesh(order: ElementOrder, cells: usize) -> SolverMeshArtifact {
    crate::physics::electrical::tests::mesh(order, cells)
}

fn compiled(
    model: runmat_analysis_core::AnalysisModel,
    mesh: &SolverMeshArtifact,
) -> CompiledAnalysisStudy {
    let properties = crate::properties::compile_property_map(&model, Some(mesh)).unwrap();
    CompiledAnalysisStudy {
        module_path: "tests/magnetic.fea".into(),
        study_id: "magnetic".into(),
        study_inputs: vec![],
        assessments: Vec::new(),
        outputs: vec![],
        model,
        properties,
        solver_mesh_digest: mesh.canonical_digest,
    }
}

fn policy() -> MagneticStudyPlanningPolicy {
    MagneticStudyPlanningPolicy {}
}

#[test]
fn both_affine_orders_preserve_complete_compiled_sources_without_solving() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = mesh(order, 2);
        let study = compiled(authored_model(-2.0), &mesh);
        let before = study.clone();
        let product = plan_magnetic_study_product(study, &mesh, policy()).unwrap();
        assert_eq!(product.compiled, before);
        assert_eq!(product.workload.model, before.model);
        product.compiled.validate_against(&mesh).unwrap();
    }
}

#[test]
fn altered_mesh_or_property_ownership_is_rejected() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let original = mesh(order, 2);
        let study = compiled(authored_model(1.0), &original);
        assert!(matches!(
            plan_magnetic_study_product(study.clone(), &mesh(order, 3), policy()),
            Err(MagneticStudyProductError::Compilation(_))
        ));
        let mut changed = study;
        changed.model.materials[0]
            .electrical
            .as_mut()
            .unwrap()
            .relative_permeability = runmat_types::physical::RelativePermeability::new(2.0);
        assert!(matches!(
            plan_magnetic_study_product(changed, &original, policy()),
            Err(MagneticStudyProductError::Compilation(_))
        ));
    }
}

#[test]
fn post_mesh_ground_gauge_and_material_limits_are_not_inferred() {
    let mesh = mesh(ElementOrder::Tet4, 2);
    let model = authored_model(1.0);
    let mut missing_ground = model.clone();
    missing_ground.constraints.retain(|c| {
        !matches!(c, AnalysisConstraint::Magnetic(m)
            if matches!(m.kind, MagneticConstraintKind::VectorPotentialGround { .. }))
    });
    let mut missing_gauge = model.clone();
    missing_gauge.constraints.pop();
    let mut foreign_ground = model.clone();
    if let AnalysisConstraint::Magnetic(ground) = &mut foreign_ground.constraints[0] {
        ground.kind = MagneticConstraintKind::VectorPotentialGround {
            target: RegionTarget::boundary(runmat_meshing_core::fixtures::fixture_face(
                "unknown-face",
            )),
        };
    }
    let mut coupled_material = model;
    coupled_material.materials[0]
        .electrical
        .as_mut()
        .unwrap()
        .resistive_heating_coefficient = runmat_types::physical::InverseTemperature::new(1.0);
    // Magnetostatics uses only the permeability; a conductivity law is not an error.
    assert!(
        plan_magnetic_study_product(compiled(coupled_material, &mesh), &mesh, policy()).is_ok()
    );
    for changed in [missing_ground, missing_gauge, foreign_ground] {
        assert!(matches!(
            plan_magnetic_study_product(compiled(changed, &mesh), &mesh, policy()),
            Err(MagneticStudyProductError::Admission(_))
        ));
    }
}
