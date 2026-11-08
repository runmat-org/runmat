use super::*;
mod contact;
use crate::{
    assembly::{dofs::StructuralDofKind, nonlinear_solid::tests::fixture},
    constraints::ResolvedConstraintMatrix,
};
use runmat_meshing_core::ElementOrder;

fn policy() -> SolidNewtonPolicy {
    SolidNewtonPolicy {
        max_iterations: 24,
        relative_residual_tolerance: 1e-9,
        linear_relative_tolerance: 1e-12,
        line_search: true,
        max_backtracks: 12,
        backtrack_reduction: 0.5,
        max_cutbacks: 8,
        max_accepted_subincrements: 1024,
        minimum_load_increment: 1e-8,
    }
}

fn anchored_uniaxial_summary(
    mesh: &runmat_meshing_core::SolverMeshArtifact,
    mut summary: AssemblySummary,
    stress: f64,
) -> AssemblySummary {
    // Independent divergence-theorem load: sigma_xx traction on the inclined
    // face of the unit tetrahedron. Outward x-area is 1/2; x=0 is the reaction face.
    // Tri3 integrates each shape to area/3; affine Tri6 corner integrals vanish.
    let mut rows = Vec::new();
    let mut force = vec![0.0; summary.dof_count];
    for (node_index, node) in mesh.topology.nodes.iter().enumerate() {
        for (axis, kind) in [
            StructuralDofKind::Ux,
            StructuralDofKind::Uy,
            StructuralDofKind::Uz,
        ]
        .into_iter()
        .enumerate()
        {
            if node.coordinates_m.canonical_value()[axis] == 0.0 {
                rows.push(
                    summary
                        .structural_dof_layout
                        .index(node_index, kind)
                        .unwrap(),
                );
            }
        }
        let x = node.coordinates_m.canonical_value();
        if (x.iter().sum::<f64>() - 1.0).abs() < 1e-14 {
            let weight = match mesh.topology.volume_elements[0].order {
                ElementOrder::Tet4 => 1.0 / 6.0,
                ElementOrder::Tet10 if x.iter().filter(|v| **v == 0.5).count() == 2 => 1.0 / 6.0,
                ElementOrder::Tet10 => 0.0,
            };
            force[summary
                .structural_dof_layout
                .index(node_index, StructuralDofKind::Ux)
                .unwrap()] = weight * stress;
        }
    }
    rows.sort_unstable();
    summary.constraint_matrix = Some(ResolvedConstraintMatrix {
        full_dof_count: summary.dof_count,
        row_offsets: (0..=rows.len()).collect(),
        column_indices: rows.clone(),
        coefficients: vec![1.0; rows.len()],
        right_hand_side: vec![0.0; rows.len()],
        row_identities: rows
            .iter()
            .map(|i| Digest::sha256(i.to_le_bytes()))
            .collect(),
        constraint_ids: rows.iter().map(|i| format!("plane_anchor_{i}")).collect(),
    });
    summary.affine_constraint_reduction = None;
    summary.physical_load = force;
    summary
}

// witness: FEA-NONLINEAR-OPERATOR-001/point_state_constitutive_laws @
//   small_strain_j2/tet4|tet10/numerical_kernel; coverage: partial;
//   boundary: Physical Newton matches an independent uniaxial plastic bar and a cold restart for
//   both orders; cyclic paths and other laws are not covered.
// witness: FEA-NONLINEAR-OPERATOR-001/references @ small_strain_j2/tet4|tet10/numerical_kernel;
//   coverage: partial;
//   boundary: Independent uniaxial plastic bar reference with cold restart for both orders; the
//   rest of the listed references (cyclic paths, thick cylinder) are not covered.
// witness: FEA-NONLINEAR-OPERATOR-001/canonical_state_physical_cutbacks @
//   small_strain_j2/tet4|tet10/numerical_kernel; coverage: partial;
//   check: j2-uniaxial-bar; boundary: The plastic bar solve restarts cold from persisted state for both orders; adaptation
//   transfer of state is not covered.
#[test]
fn physical_newton_matches_independent_uniaxial_plastic_bar_and_cold_restart_both_orders() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (model, mesh, summary) = fixture(order, true);
        let summary = anchored_uniaxial_summary(&mesh, summary, 5.6);
        let basis = CanonicalNonlinearSolidBasis::prepare(&model, &mesh, &summary).unwrap();
        let problem = SolidNewtonProblem::prepare(&basis, &summary, policy()).unwrap();
        let initial = problem.initial_state().unwrap();
        let initial_before = initial.clone();
        let increments = problem.advance(&initial, 1.0).unwrap();
        assert_eq!(initial, initial_before);
        let final_state = &increments.last().unwrap().state;
        let alpha = (5.6 - 5.0) / 30.0;
        let final_increment = increments.last().unwrap();
        assert!(final_increment.relative_residual <= policy().relative_residual_tolerance);
        assert!(final_increment.relative_increment <= policy().relative_residual_tolerance);
        assert!(final_increment.attempted_work.linear_solves > final_increment.iterations);
        assert!(final_increment.correction_norm.is_finite());
        assert!(final_increment.iterations > 0);
        assert!(final_increment.backtracks <= final_increment.iterations * policy().max_backtracks);
        let evaluation = &final_increment.evaluation;
        let mut volume = 0.0;
        for (point, identity) in evaluation
            .point_evidence
            .iter()
            .zip(&final_state.constitutive.point_identities)
        {
            assert_eq!(&point.identity, identity);
            assert!(point.position_m.iter().all(|x| *x >= 0.0));
            assert!(point.position_m.iter().sum::<f64>() <= 1.0 + 1e-14);
            volume += point.volume_weight_m3;
            assert!((point.stress_pa[0] - 5.6).abs() < 2e-10);
            assert!(point.stress_pa[1..].iter().all(|s| s.abs() < 2e-10));
            assert!((point.elastic_energy_density_pa - 0.5 * 5.6 * 5.6 / 1200.0).abs() < 2e-10);
        }
        assert!((volume - 1.0 / 6.0).abs() < 2e-13);
        for element in &evaluation.element_evidence {
            assert!(final_state
                .constitutive
                .point_identities
                .iter()
                .any(|point| point.element_identity == element.element_identity));
            assert!(
                (element.elastic_energy_j.canonical_value() - 0.5 * 5.6 * 5.6 / 1200.0 / 6.0).abs()
                    < 2e-10
            );
            assert!(
                (element.hardening_energy_j.canonical_value() - 0.5 * 30.0 * alpha * alpha / 6.0)
                    .abs()
                    < 2e-10
            );
        }
        let strain = [
            5.6 / 1200.0 + alpha,
            -0.25 * 5.6 / 1200.0 - alpha / 2.0,
            -0.25 * 5.6 / 1200.0 - alpha / 2.0,
        ];
        for (index, node) in mesh.topology.nodes.iter().enumerate() {
            for (axis, kind) in [
                StructuralDofKind::Ux,
                StructuralDofKind::Uy,
                StructuralDofKind::Uz,
            ]
            .into_iter()
            .enumerate()
            {
                let actual = final_state.displacement
                    [summary.structural_dof_layout.index(index, kind).unwrap()];
                assert!(
                    (actual - strain[axis] * node.coordinates_m.canonical_value()[axis]).abs()
                        < 2e-10,
                    "order={order:?}, node={index}, axis={axis}, actual={actual}"
                );
            }
        }
        for point in &final_state.constitutive.points {
            assert!((point.equivalent_plastic_strain - alpha).abs() < 2e-10);
        }
        assert!((final_state.accumulated_plastic_dissipation_j - 5.0 * alpha / 6.0).abs() < 2e-10);
        problem.validate_accepted(final_state).unwrap();
        let prefix = problem.advance(&initial, 0.95).unwrap();
        let warm = &prefix.last().unwrap().state;
        // Actual cold typed serialization, without re-solving the accepted prefix.
        let bytes = serde_json::to_vec(warm).unwrap();
        let cold: AcceptedSolidState = serde_json::from_slice(&bytes).unwrap();
        let resumed = problem.advance(&cold, 1.0).unwrap();
        for (actual, expected) in resumed
            .last()
            .unwrap()
            .state
            .displacement
            .iter()
            .zip(&final_state.displacement)
        {
            assert!((actual - expected).abs() < 2e-10);
        }
        let mut foreign = cold.clone();
        foreign.scope_identity = Digest::sha256(b"foreign-fixed-load");
        assert!(problem.advance(&foreign, 1.0).is_err());
        let mut foreign_rule = cold.clone();
        foreign_rule.constitutive.integration_identity = Digest::sha256(b"foreign-integration");
        assert!(problem.advance(&foreign_rule, 1.0).is_err());
        let richer_plan = crate::topology::CanonicalSolidIntegrationPlan::derive(
            &mesh,
            crate::topology::CanonicalSolidIntegrationRule::Duffy5,
        )
        .unwrap();
        let richer_basis = CanonicalNonlinearSolidBasis::prepare_with_integration(
            &model,
            &mesh,
            &summary,
            &richer_plan,
        )
        .unwrap();
        let richer_problem =
            SolidNewtonProblem::prepare(&richer_basis, &summary, policy()).unwrap();
        assert!(richer_problem.advance(&cold, 1.0).is_err());
        let mut non_equilibrium = cold.clone();
        non_equilibrium.load_factor *= 0.9;
        assert!(problem.validate_accepted(&non_equilibrium).is_err());
        let mut foreign_energy = cold.clone();
        foreign_energy.accumulated_plastic_dissipation_j += 1.0;
        assert!(problem.validate_accepted(&foreign_energy).is_err());
        let mut overflow = cold;
        overflow.load_factor = f64::MAX;
        assert!(problem.validate_accepted(&overflow).is_err());
    }
}

#[test]
fn small_force_defect_can_require_a_large_pending_correction() {
    use crate::algebra::{MatrixProperties, OperatorRole, RealSparseAssembler};
    let (model, mesh, summary) = fixture(ElementOrder::Tet4, false);
    let summary = anchored_uniaxial_summary(&mesh, summary, 1.0);
    let basis = CanonicalNonlinearSolidBasis::prepare(&model, &mesh, &summary).unwrap();
    let problem = SolidNewtonProblem::prepare(&basis, &summary, policy()).unwrap();
    let free = problem.reduction.reduced_dof_count;
    assert_eq!(free, 3);
    // A controlled SPD operator isolates the acceptance criterion: the soft mode
    // has a tiny force defect but a displacement correction of 1e-3. This is
    // solver evidence, not a continuum qualification reference.
    let mut tangent = RealSparseAssembler::square(
        summary.dof_count,
        OperatorRole::Stiffness,
        MatrixProperties::SYMMETRIC_POSITIVE_DEFINITE,
    );
    for index in 0..summary.dof_count {
        tangent.add(index, index, 1e-8);
    }
    let mut residual = vec![0.0; free];
    residual[0] = 1e-11;
    assert!(relative_norm(&residual, 1.0) < policy().relative_residual_tolerance);
    let mut work = SolidAttemptWork::default();
    let correction = problem
        .correction(&tangent.finish().unwrap(), &residual, 1.0, &mut work)
        .unwrap();
    assert!((norm(&correction) - 1e-3).abs() < 1e-15);
    assert!(relative_norm(&correction, 1.0) > policy().relative_residual_tolerance);
    assert_eq!(work.linear_solves, 1);
}

// witness: FEA-NONLINEAR-STATE-001/rollback @ small_strain_j2/tet4/prepared_controller;
//   boundary: Caller history remains unchanged after failed Newton/cutback.
#[test]
fn failed_newton_and_cutback_do_not_commit_caller_history() {
    let (model, mesh, summary) = fixture(ElementOrder::Tet4, true);
    let summary = anchored_uniaxial_summary(&mesh, summary, 5.6);
    let basis = CanonicalNonlinearSolidBasis::prepare(&model, &mesh, &summary).unwrap();
    let mut short = policy();
    short.max_iterations = 1;
    short.max_cutbacks = 0;
    let problem = SolidNewtonProblem::prepare(&basis, &summary, short).unwrap();
    let accepted = problem.initial_state().unwrap();
    let before = accepted.clone();
    assert!(matches!(
        problem.advance(&accepted, 1.0),
        Err(SolidNewtonError::NotConverged { .. })
    ));
    assert_eq!(accepted, before);
    let mut with_cutback = short;
    with_cutback.max_cutbacks = 4;
    let problem = SolidNewtonProblem::prepare(&basis, &summary, with_cutback).unwrap();
    assert!(problem.advance(&accepted, 1.0).is_err());
    assert_eq!(accepted, before);
}

// witness: FEA-NONLINEAR-STATE-001/rollback @ small_strain_j2/tet4|tet10/prepared_controller;
//   boundary: Both-order point-evaluation cancellation preserves caller history; not complete
//   failed-Newton, line-search or controller suffix coverage.
#[test]
fn cancelled_point_evaluation_reaches_newton_without_mutating_history() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let (model, mesh, summary) = fixture(order, true);
        let summary = anchored_uniaxial_summary(&mesh, summary, 20.0);
        let basis = CanonicalNonlinearSolidBasis::prepare(&model, &mesh, &summary).unwrap();
        let problem = SolidNewtonProblem::prepare(&basis, &summary, policy()).unwrap();
        let state = problem.initial_state().unwrap();
        let before = state.clone();
        let context = crate::progress::FeaProgressContext::default();
        context.cancellation.cancel();
        let guard = crate::progress::replace_fea_progress_context(context);
        assert!(matches!(
            problem.evaluate(&state.displacement, &state.constitutive),
            Err(SolidNewtonError::Cancelled)
        ));
        assert_eq!(state, before);
        drop(guard);
        assert!(problem
            .evaluate(&state.displacement, &state.constitutive)
            .is_ok());
    }
}
