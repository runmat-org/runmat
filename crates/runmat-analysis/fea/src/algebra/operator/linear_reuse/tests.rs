//! Numerical equivalence, exact invalidation and fresh RHS/nullspace binding.
use super::*;
use crate::algebra::{MatrixProperties, OperatorRole};

fn prepare_without_host_wait(
    reuse: &mut LinearPreparationReuse,
    matrix: &RealSparseMatrix,
    rhs: &[f64],
) -> Result<PreparedSolverAdmissibility, CoercivityError> {
    use crate::algebra::coercivity::NumericalCapacityError;
    futures::executor::block_on(reuse.prepare_admitted_with_owner(
        matrix,
        rhs,
        None,
        None,
        |_, _| std::future::ready(Ok::<(), std::convert::Infallible>(())),
    ))
    .map_err(|error| match error {
        NumericalCapacityError::Numerical(error) => error,
        NumericalCapacityError::Admission(never) => match never {},
    })
}

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
    model.material_assignments[0].target = runmat_analysis_core::PropertyRegionTarget::solid(
        runmat_meshing_core::fixtures::fixture_solid("slab"),
    );
    let runmat_analysis_core::AnalysisConstraint::Structural(constraint) =
        &mut model.constraints[0]
    else {
        panic!("structural fixture");
    };
    let runmat_analysis_core::StructuralConstraintKind::Fixed { target } = &mut constraint.kind
    else {
        panic!("fixed fixture");
    };
    *target = runmat_analysis_core::RegionTarget::boundary(
        runmat_meshing_core::fixtures::fixture_face("slab_left"),
    );
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

#[test]
fn source_owned_energy_proof_rebinds_rhs_and_invalidates_changed_operator() {
    use crate::algebra::anchored_energy::AnchoredEnergySource;
    let operator = RealSparseMatrix::from_dense(
        &[1., 0., 0., 0., 2., -1., 0., -1., 1.],
        3,
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap();
    let bytes = AnchoredEnergySource::resident_bytes_for_widths([2, 2].into_iter()).unwrap();
    let mut source = AnchoredEnergySource::new([2, 2].into_iter(), bytes).unwrap();
    for nodes in [[0, 1], [1, 2]] {
        source
            .capture_next(&nodes)
            .unwrap()
            .copy_from_slice(&[1., -1., -1., 1.]);
    }
    let mut reuse = LinearPreparationReuse::with_local_energy(source, vec![true, false, false], 1.);
    let before = prepare_without_host_wait(&mut reuse, &operator, &[0., 1., 0.]).unwrap();
    assert!(reuse.local_energy.is_none());
    let after = prepare_without_host_wait(&mut reuse, &operator, &[0., 2., 0.]).unwrap();
    assert_eq!(before.coercivity(), after.coercivity());
    assert_ne!(before, after);
    assert_eq!(
        after.validate_for(&operator, &[0., 1., 0.]),
        Err(CoercivityError::BindingMismatch)
    );
    let corrupt = RealSparseMatrix::from_dense(
        &[1., 0., 0., 0., 0.1, -1., 0., -1., 1.],
        3,
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap();
    assert!(prepare_without_host_wait(&mut reuse, &corrupt, &[0., 2., 0.]).is_err());
    assert!(reuse.certificate.is_none() && reuse.local_energy.is_none());
}

/// Tet10 slivers admitted by the mesher can have element energies whose positivity the local
/// LDL bound cannot certify. That leaves the local proof undecided, and the sparse Cholesky
/// proof certifies the assembled operator (ev-busbar: `Coercivity(Energy(InvalidBlock))`).
#[test]
fn uncertifiable_local_energy_falls_back_to_the_factorization_proof() {
    use crate::algebra::anchored_energy::AnchoredEnergySource;
    let operator = RealSparseMatrix::from_dense(
        &[1., 0., 0., 0., 1.5, -1., 0., -1., 3.],
        3,
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap();
    let bytes = AnchoredEnergySource::resident_bytes_for_widths([3, 2].into_iter()).unwrap();
    let mut source = AnchoredEnergySource::new([3, 2].into_iter(), bytes).unwrap();
    // Indefinite on its free block; the second element makes the assembled sum positive.
    source
        .capture_next(&[0, 1, 2])
        .unwrap()
        .copy_from_slice(&[0., 0., 0., 0., -0.5, 0., 0., 0., 1.]);
    source
        .capture_next(&[1, 2])
        .unwrap()
        .copy_from_slice(&[2., -1., -1., 2.]);
    let mut reuse = LinearPreparationReuse::with_local_energy(source, vec![true, false, false], 1.);
    let prepared = prepare_without_host_wait(&mut reuse, &operator, &[0., 1., 0.]).unwrap();
    assert!(
        reuse.factor.is_some(),
        "the Cholesky proof certified the operator"
    );
    assert_eq!(reuse.certificate(), Some(prepared.coercivity()));
}
