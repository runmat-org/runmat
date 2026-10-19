use runmat_execution::Digest;

use crate::constraints::ResolvedConstraintMatrix;

use super::*;

#[test]
fn nonzero_direct_constraint_supplies_default_and_rejects_authored_conflicts() {
    let mut model = beam_model();
    let layout = beam_layout();
    let mut constrained = vec![false; layout.total_dof_count()];
    constrained[0] = true;
    let mut prescribed = vec![0.0; layout.total_dof_count()];
    prescribed[0] = 0.25;
    let view = StructuralInitialConstraintView {
        displacement_constraints: None,
        velocity_constraints: None,
        affine_reduction: None,
        direct_constrained_dofs: &constrained,
        direct_prescribed_displacement: &prescribed,
    };

    let state = realize_with_constraints(
        &model,
        None,
        &layout,
        StructuralInitialStateRealizationBasis::CanonicalTopology,
        view,
        STEP_ID,
    )
    .expect("direct constraint default");
    assert_eq!(state.displacement[0], 0.25);
    assert_eq!(state.velocity[0], 0.0);
    assert_eq!(state.evidence.direct_constrained_dof_count, 1);
    assert_eq!(state.evidence.authored_displacement_dof_count, 0);
    assert_eq!(state.evidence.constraint_derived_displacement_dof_count, 1);
    assert_eq!(
        state.evidence.zero_default_displacement_dof_count,
        layout.total_dof_count() - 1
    );
    assert_eq!(state.evidence.authored_velocity_dof_count, 0);
    assert_eq!(state.evidence.constraint_derived_velocity_dof_count, 1);
    assert_eq!(
        state.evidence.zero_default_velocity_dof_count,
        layout.total_dof_count() - 1
    );

    model.initial_conditions = vec![
        displacement(
            "agreeing-u",
            InitialConditionTarget::WholeModel,
            [0.25, 0.0, 0.0],
        ),
        velocity(
            "agreeing-v",
            InitialConditionTarget::WholeModel,
            [0.0, 0.0, 0.0],
        ),
    ];
    realize_with_constraints(
        &model,
        None,
        &layout,
        StructuralInitialStateRealizationBasis::CanonicalTopology,
        view,
        STEP_ID,
    )
    .expect("authored values may agree with direct constraints");

    model.initial_conditions = vec![displacement(
        "conflicting-u",
        InitialConditionTarget::WholeModel,
        [0.5, 0.0, 0.0],
    )];
    assert!(matches!(
        realize_with_constraints(
            &model,
            None,
            &layout,
            StructuralInitialStateRealizationBasis::CanonicalTopology,
            view,
            STEP_ID,
        ),
        Err(StructuralInitialStateError::PrescribedDisplacementConflict {
            initial_condition_id,
            row: 0,
        }) if initial_condition_id == "conflicting-u"
    ));

    model.initial_conditions = vec![velocity(
        "conflicting-v",
        InitialConditionTarget::WholeModel,
        [0.5, 0.0, 0.0],
    )];
    assert!(matches!(
        realize_with_constraints(
            &model,
            None,
            &layout,
            StructuralInitialStateRealizationBasis::CanonicalTopology,
            view,
            STEP_ID,
        ),
        Err(StructuralInitialStateError::PrescribedVelocityConflict {
            initial_condition_id,
            row: 0,
        }) if initial_condition_id == "conflicting-v"
    ));
}

#[test]
fn affine_constraint_recovers_dependent_initial_state_and_validates_both_kinematics() {
    let mut model = beam_model();
    let layout = beam_layout();
    let matrix = ResolvedConstraintMatrix {
        full_dof_count: layout.total_dof_count(),
        row_offsets: vec![0, 2],
        column_indices: vec![0, 6],
        coefficients: vec![1.0, -1.0],
        right_hand_side: vec![0.25],
        row_identities: vec![Digest::sha256(b"relative-x")],
        constraint_ids: vec!["relative-x".into()],
    };
    let reduction = matrix.affine_reduction().expect("affine reduction");
    let view = StructuralInitialConstraintView {
        displacement_constraints: Some(&matrix),
        velocity_constraints: None,
        affine_reduction: Some(&reduction),
        direct_constrained_dofs: &[],
        direct_prescribed_displacement: &[],
    };
    model.initial_conditions = vec![
        displacement(
            "tip-u",
            InitialConditionTarget::Region(runmat_analysis_core::RegionTarget::point("node:2")),
            [1.0, 0.0, 0.0],
        ),
        velocity(
            "tip-v",
            InitialConditionTarget::Region(runmat_analysis_core::RegionTarget::point("node:2")),
            [2.0, 0.0, 0.0],
        ),
    ];
    let state = realize_with_constraints(
        &model,
        None,
        &layout,
        StructuralInitialStateRealizationBasis::CanonicalTopology,
        view,
        STEP_ID,
    )
    .expect("affine-compatible initial state");
    assert_eq!(state.displacement[0], 1.25);
    assert_eq!(state.displacement[6], 1.0);
    assert_eq!(state.velocity[0], 2.0);
    assert_eq!(state.velocity[6], 2.0);
    assert_eq!(state.evidence.constraint_equation_count, 1);
    assert_eq!(
        state.evidence.maximum_displacement_constraint_residual_bits,
        0.0_f64.to_bits()
    );
    assert_eq!(
        state.evidence.maximum_velocity_constraint_residual_bits,
        0.0_f64.to_bits()
    );

    model.initial_conditions.push(displacement(
        "root-u",
        InitialConditionTarget::Region(runmat_analysis_core::RegionTarget::point("node:1")),
        [9.0, 0.0, 0.0],
    ));
    assert!(matches!(
        realize_with_constraints(
            &model,
            None,
            &layout,
            StructuralInitialStateRealizationBasis::CanonicalTopology,
            view,
            STEP_ID,
        ),
        Err(StructuralInitialStateError::DisplacementConstraintViolation {
            constraint_id,
            authored_initial_condition_ids,
            ..
        }) if constraint_id == "relative-x"
            && authored_initial_condition_ids == vec!["root-u".to_string(), "tip-u".to_string()]
    ));

    model.initial_conditions = vec![
        velocity(
            "root-v",
            InitialConditionTarget::Region(runmat_analysis_core::RegionTarget::point("node:1")),
            [9.0, 0.0, 0.0],
        ),
        velocity(
            "tip-v",
            InitialConditionTarget::Region(runmat_analysis_core::RegionTarget::point("node:2")),
            [2.0, 0.0, 0.0],
        ),
    ];
    assert!(matches!(
        realize_with_constraints(
            &model,
            None,
            &layout,
            StructuralInitialStateRealizationBasis::CanonicalTopology,
            view,
            STEP_ID,
        ),
        Err(StructuralInitialStateError::VelocityConstraintViolation {
            constraint_id,
            authored_initial_condition_ids,
            ..
        }) if constraint_id == "relative-x"
            && authored_initial_condition_ids == vec!["root-v".to_string(), "tip-v".to_string()]
    ));
}

#[test]
fn affine_constraint_recovers_nonzero_prescribed_initial_velocity() {
    let model = beam_model();
    let layout = beam_layout();
    let displacement = ResolvedConstraintMatrix {
        full_dof_count: layout.total_dof_count(),
        row_offsets: vec![0, 2],
        column_indices: vec![0, 6],
        coefficients: vec![1.0, -1.0],
        right_hand_side: vec![0.25],
        row_identities: vec![Digest::sha256(b"moving-relative-x")],
        constraint_ids: vec!["moving-relative-x".into()],
    };
    let mut velocity = displacement.clone();
    velocity.right_hand_side = vec![-0.5];
    let reduction = displacement.affine_reduction().unwrap();
    let state = realize_with_constraints(
        &model,
        None,
        &layout,
        StructuralInitialStateRealizationBasis::CanonicalTopology,
        StructuralInitialConstraintView {
            displacement_constraints: Some(&displacement),
            velocity_constraints: Some(&velocity),
            affine_reduction: Some(&reduction),
            direct_constrained_dofs: &[],
            direct_prescribed_displacement: &[],
        },
        STEP_ID,
    )
    .expect("nonzero affine velocity should be realized from its exact row space");
    assert_eq!(state.displacement[0], 0.25);
    assert_eq!(state.displacement[6], 0.0);
    assert_eq!(state.velocity[0], -0.5);
    assert_eq!(state.velocity[6], 0.0);
    assert_eq!(
        state.evidence.maximum_velocity_constraint_residual_bits,
        0.0_f64.to_bits()
    );

    let mut incompatible_velocity = velocity;
    incompatible_velocity.coefficients[1] = 1.0;
    assert!(matches!(
        realize_with_constraints(
            &model,
            None,
            &layout,
            StructuralInitialStateRealizationBasis::CanonicalTopology,
            StructuralInitialConstraintView {
                displacement_constraints: Some(&displacement),
                velocity_constraints: Some(&incompatible_velocity),
                affine_reduction: Some(&reduction),
                direct_constrained_dofs: &[],
                direct_prescribed_displacement: &[],
            },
            STEP_ID,
        ),
        Err(StructuralInitialStateError::InvalidConstraintInputs { .. })
    ));
}
