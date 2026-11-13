use runmat_execution::Digest;

use super::*;
use crate::{
    assembly::{assemble_structural_system, StructuralFormulationRequirement},
    excitation::{realize_step_excitations, ExcitationSamplePoint},
    fixtures::{
        fixture_model, replace_fixture_excitations, FixtureExcitation, FixtureExcitationKind,
        FixtureId,
    },
};

#[test]
fn tiny_physical_load_keeps_a_unit_sized_relative_dynamic_residual() {
    let mut model = fixture_model(FixtureId::StructuralBeamCantileverEndMomentReference);
    replace_fixture_excitations(
        &mut model,
        vec![FixtureExcitation {
            id: "tiny_tip_moment".into(),
            region_id: "node:2".into(),
            kind: FixtureExcitationKind::Moment([0.0, 0.0, 1.0e-9]),
        }],
    );
    let excitations = realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        ExcitationSamplePoint::Static,
    )
    .unwrap();
    let summary = assemble_structural_system(
        &model,
        &excitations,
        None,
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    let zero_state = SecondOrderState::try_new(
        vec![0.0; summary.dof_count],
        vec![0.0; summary.dof_count],
        vec![0.0; summary.dof_count],
    )
    .unwrap();
    let balance = StructuralDynamicBalanceSnapshot::evaluate(
        summary.operator.operators(),
        &zero_state,
        &summary.physical_load,
    )
    .unwrap();
    assert_eq!(
        project_residual(&summary, &balance)
            .unwrap()
            .relative_norm(),
        1.0
    );

    replace_fixture_excitations(&mut model, Vec::new());
    let zero_excitations = realize_step_excitations(
        &model,
        &model.sequence.steps[0],
        ExcitationSamplePoint::Static,
    )
    .unwrap();
    let unloaded = assemble_structural_system(
        &model,
        &zero_excitations,
        None,
        StructuralFormulationRequirement::PhysicalDiscretization,
    )
    .unwrap();
    let unloaded_balance = StructuralDynamicBalanceSnapshot::evaluate(
        unloaded.operator.operators(),
        &zero_state,
        &unloaded.physical_load,
    )
    .unwrap();
    assert_eq!(
        project_residual(&unloaded, &unloaded_balance)
            .unwrap()
            .relative_norm(),
        0.0
    );

    let mut inconsistent = unloaded_balance;
    let free_dof = unloaded
        .operator
        .constrained_dofs()
        .iter()
        .position(|constrained| !constrained)
        .unwrap();
    inconsistent.equilibrium_residual[free_dof] = 1.0e-9;
    assert!(matches!(
        project_residual(&unloaded, &inconsistent),
        Err(EffectiveDynamicsSolveError::InvalidSolution)
    ));
}

#[test]
fn correction_equation_normalization_is_invariant_to_physical_scale() {
    for scale in [1.0e-12, 1.0, 1.0e12] {
        let operator = RealSparseMatrix::diagonal(
            vec![2.0 * scale],
            OperatorRole::EffectiveTangent,
            MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
        )
        .unwrap();
        let (operator_scale, solution_scale) =
            active_correction_scales(&operator, &[6.0 * scale], &[false], None).unwrap();

        assert_eq!(operator_scale, 2.0 * scale);
        assert_eq!(solution_scale, 3.0);
    }
}

#[test]
fn direct_constraint_reactions_do_not_set_correction_scale() {
    let operator = RealSparseMatrix::diagonal(
        vec![1.0e12, 2.0],
        OperatorRole::EffectiveTangent,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap();
    let (operator_scale, solution_scale) =
        active_correction_scales(&operator, &[1.0e12, 6.0], &[true, false], None).unwrap();

    assert_eq!(operator_scale, 2.0);
    assert_eq!(solution_scale, 3.0);
}

#[test]
fn affine_constraint_reactions_do_not_set_correction_scale() {
    let operator = RealSparseMatrix::diagonal(
        vec![2.0, 2.0],
        OperatorRole::EffectiveTangent,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    )
    .unwrap();
    let constraint = ResolvedConstraintMatrix {
        full_dof_count: 2,
        row_offsets: vec![0, 2],
        column_indices: vec![0, 1],
        coefficients: vec![1.0, -1.0],
        right_hand_side: vec![0.0],
        row_identities: vec![Digest::sha256(b"equal-motion")],
        constraint_ids: vec!["equal_motion".into()],
    };
    let reduction = constraint.affine_reduction().unwrap();
    let (operator_scale, solution_scale) = active_correction_scales(
        &operator,
        &[1.0e12, -1.0e12 + 6.0],
        &[false, false],
        Some(&reduction),
    )
    .unwrap();

    assert_eq!(operator_scale, 4.0);
    assert_eq!(solution_scale, 1.5);
}
