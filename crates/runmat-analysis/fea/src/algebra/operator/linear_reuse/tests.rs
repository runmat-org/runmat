//! Numerical equivalence, exact invalidation and fresh RHS/nullspace binding.
use super::*;
use crate::algebra::{MatrixProperties, OperatorRole};

fn system(diagonal: f64, constrained: Vec<bool>, load: Vec<f64>) -> AssembledRealSystem {
    let matrix = RealSparseMatrix::from_dense(
        &[diagonal, -1.0, -1.0, 3.0],
        2,
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap();
    let mass = RealSparseMatrix::diagonal(
        vec![1.0; 2],
        OperatorRole::Mass,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap();
    let damping = RealSparseMatrix::diagonal(
        vec![0.0; 2],
        OperatorRole::Damping,
        MatrixProperties::SYMMETRIC_POSITIVE_SEMIDEFINITE,
    )
    .unwrap();
    AssembledRealSystem::try_new(
        CanonicalRealOperators::try_new(matrix, mass, damping).unwrap(),
        constrained,
        load,
    )
    .unwrap()
}

#[test]
fn exact_operator_reuses_proof_but_rebinds_dirichlet_rhs_and_prepared_identity() {
    let mut reuse = LinearPreparationReuse::default();
    let first = system(4.0, vec![true, false], vec![1.0, 2.0]);
    let before = first.prepare_linear_reusing(&mut reuse).unwrap();
    let certificate = reuse.certificate.clone();
    let changed = first.with_load(vec![2.0, 5.0]).unwrap();
    let after = changed.prepare_linear_reusing(&mut reuse).unwrap();
    assert_eq!(after, changed.prepare_linear().unwrap());
    assert_eq!(after.rhs(), &[2.0, 7.0]);
    assert_eq!(reuse.certificate, certificate);
    assert_ne!(before.admissibility(), after.admissibility());
    assert!(super::super::super::validate_prepared_solver_compatibility(
        &after,
        crate::algebra::RealSolver::ConjugateGradient,
        crate::algebra::RealPreconditioner::Jacobi
    )
    .is_ok());
    let cold: PreparedRealLinearSystem =
        serde_json::from_value(serde_json::to_value(after).unwrap()).unwrap();
    assert_eq!(cold, changed.prepare_linear().unwrap());
}

#[test]
fn one_bit_value_and_constraint_changes_cannot_reuse_foreign_proof() {
    let mut reuse = LinearPreparationReuse::default();
    system(4.0, vec![false; 2], vec![1.0, 2.0])
        .prepare_linear_reusing(&mut reuse)
        .unwrap();
    let first = reuse.certificate.clone();
    let changed = system(
        f64::from_bits(4.0_f64.to_bits() + 1),
        vec![false; 2],
        vec![1.0, 2.0],
    );
    assert_eq!(
        changed.prepare_linear_reusing(&mut reuse).unwrap(),
        changed.prepare_linear().unwrap()
    );
    assert_ne!(first, reuse.certificate);
    let second = reuse.certificate.clone();
    let changed = system(4.0, vec![true, false], vec![1.0, 2.0]);
    assert_eq!(
        changed.prepare_linear_reusing(&mut reuse).unwrap(),
        changed.prepare_linear().unwrap()
    );
    assert_ne!(second, reuse.certificate);
    let indefinite = system(-4.0, vec![false; 2], vec![1.0, 2.0]);
    assert!(indefinite.prepare_linear_reusing(&mut reuse).is_err());
    assert!(reuse.certificate.is_none());
}

#[test]
fn physical_nullspace_provenance_is_rebound_from_the_current_original_source() {
    use crate::{
        assembly::{assemble_structural_system, StructuralFormulationRequirement},
        excitation::{realize_step_excitations, ExcitationSamplePoint},
        fixtures::{
            fixture_model, replace_fixture_excitations, FixtureExcitation, FixtureExcitationKind,
            FixtureId,
        },
    };
    let mut model = fixture_model(FixtureId::CantileverLinearStatic);
    model.material_assignments[0].target =
        runmat_analysis_core::PropertyRegionTarget::solid("slab");
    let runmat_analysis_core::AnalysisConstraint::Structural(constraint) =
        &mut model.constraints[0]
    else {
        panic!("structural fixture");
    };
    let runmat_analysis_core::StructuralConstraintKind::Fixed { target } = &mut constraint.kind
    else {
        panic!("fixed fixture");
    };
    *target = runmat_analysis_core::RegionTarget::boundary("slab_left");
    replace_fixture_excitations(
        &mut model,
        vec![FixtureExcitation {
            id: "original-tip".into(),
            region_id: "slab_right".into(),
            kind: FixtureExcitationKind::Traction([0.0, -1.0, 0.0]),
        }],
    );
    let loads = realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        ExcitationSamplePoint::Static,
    )
    .unwrap();
    let mesh = crate::topology::test_meshes::slab_mesh(1);
    let summary = assemble_structural_system(
        &model,
        &loads,
        Some(mesh),
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    assert!(summary.operator.nullspace_evidence().is_some());
    let mut reuse = LinearPreparationReuse::default();
    let first = summary.operator.prepare_linear_reusing(&mut reuse).unwrap();
    let changed = summary
        .operator
        .with_load(
            summary
                .operator
                .load()
                .iter()
                .map(|value| value * 2.0)
                .collect(),
        )
        .unwrap();
    let after = changed.prepare_linear_reusing(&mut reuse).unwrap();
    assert_eq!(after, changed.prepare_linear().unwrap());
    assert_ne!(first.rhs(), after.rhs());
}
