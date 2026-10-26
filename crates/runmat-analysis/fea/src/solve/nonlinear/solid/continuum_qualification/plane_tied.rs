//! Test-only axial-subspace counterfactual, not a replacement qualification fixture.
//! Original loads, mesh, material, integration rule and schedules remain unchanged.
//! Additional equalities remove Ux(y,z) variation, hence induced xy/xz shear, while
//! the unaligned intermediate yield front remains nonrepresentable on the same mesh.
use super::*;
use crate::constraints::ResolvedConstraintMatrix;
use runmat_execution::Digest;
use std::collections::BTreeMap;

fn tie_planes(mesh: &SolverMeshArtifact, summary: &mut AssemblySummary) {
    let mut matrix = summary.constraint_matrix.clone().unwrap_or_else(|| {
        let mut matrix = ResolvedConstraintMatrix {
            full_dof_count: summary.dof_count,
            row_offsets: vec![0],
            column_indices: Vec::new(),
            coefficients: Vec::new(),
            right_hand_side: Vec::new(),
            row_identities: Vec::new(),
            constraint_ids: Vec::new(),
        };
        for (row, &fixed) in summary.operator.constrained_dofs().iter().enumerate() {
            if fixed {
                // This independent fixture has only homogeneous prescribed DOFs.
                assert_eq!(summary.operator.load()[row], 0.0);
                append_row(&mut matrix, &[(row, 1.0)], format!("original_pin_{row}"));
            }
        }
        matrix
    });
    assert!(matrix.right_hand_side.iter().all(|v| *v == 0.0));
    let original_reduction = matrix.affine_reduction().unwrap();
    let mut planes = BTreeMap::<u64, Vec<usize>>::new();
    for (node, entry) in mesh.topology.nodes.iter().enumerate() {
        let x = entry.coordinates_m.canonical_value()[0];
        assert!(x.is_finite() && (0.0..=1.0).contains(&x));
        let row = summary
            .structural_dof_layout
            .index(node, StructuralDofKind::Ux)
            .unwrap();
        if x == 0.0 {
            // Every Ux on this face is already pinned: avoid redundant equations.
            assert_eq!(original_reduction.offset[row], 0.0);
            assert_eq!(
                original_reduction.recovery_row_offsets[row],
                original_reduction.recovery_row_offsets[row + 1]
            );
        } else {
            planes.entry(x.to_bits()).or_default().push(row);
        }
    }
    for (plane, rows) in &mut planes {
        rows.sort_unstable();
        let primary = rows[0];
        for &secondary in &rows[1..] {
            append_row(
                &mut matrix,
                &[(primary, -1.0), (secondary, 1.0)],
                format!("diagnostic_axial_plane_{plane}_{secondary}"),
            );
        }
    }
    matrix.validate().unwrap();
    let reduction = matrix.affine_reduction().unwrap();
    // One free axial value per nonzero x plane; all original transverse pins remain.
    assert_eq!(reduction.reduced_dof_count, planes.len());
    summary.constraint_matrix = Some(matrix);
    summary.affine_constraint_reduction = Some(reduction);
}

fn append_row(matrix: &mut ResolvedConstraintMatrix, terms: &[(usize, f64)], id: String) {
    assert!(terms.windows(2).all(|p| p[0].0 < p[1].0));
    for &(column, coefficient) in terms {
        matrix.column_indices.push(column);
        matrix.coefficients.push(coefficient);
    }
    matrix.row_offsets.push(matrix.column_indices.len());
    matrix.right_hand_side.push(0.0);
    matrix.row_identities.push(Digest::sha256(id.as_bytes()));
    matrix.constraint_ids.push(id);
}

#[test]
fn axially_tied_tet10_trajectories_isolate_nonradial_history() {
    // Registered before execution: final original 1e-7 field/energy/reaction bounds,
    // zero shear and radial plastic direction to arithmetic roundoff, monotone e_xx,
    // and the unchanged original Newton policy. No result-based tolerance selection.
    for cells in [1, 2, 4] {
        for requested_increments in [1, 8] {
            let (model, mesh, mut summary) = fixture::prepare(cells, true);
            tie_planes(&mesh, &mut summary);
            let basis = CanonicalNonlinearSolidBasis::prepare(&model, &mesh, &summary).unwrap();
            let problem =
                SolidNewtonProblem::prepare(&basis, &summary, execution::policy()).unwrap();
            let mut state = problem.initial_state().unwrap();
            let mut previous_strains = vec![0.0_f64; basis.point_count()];
            let mut final_evaluation = None;
            let mut accepted_count = 0;
            let mut maximum_relative_shear = 0.0_f64;
            let mut maximum_relative_direction = 0.0_f64;
            for target in (1..=requested_increments).map(|i| i as f64 / requested_increments as f64)
            {
                let before = state.clone();
                let accepted = problem.advance(&state, target).unwrap();
                assert_eq!(state, before, "advance mutated its committed input");
                for increment in accepted {
                    assert!(increment.state.load_factor > state.load_factor);
                    assert!(increment.state.load_factor <= target);
                    problem.validate_accepted(&increment.state).unwrap();
                    assert!(
                        increment.relative_residual
                            <= execution::policy().relative_residual_tolerance
                    );
                    assert_eq!(
                        increment.evaluation.point_evidence.len(),
                        previous_strains.len()
                    );
                    for ((point, history), previous) in increment
                        .evaluation
                        .point_evidence
                        .iter()
                        .zip(&increment.state.constitutive.points)
                        .zip(&mut previous_strains)
                    {
                        let strain = point.total_engineering_strain;
                        let strain_scale = strain[0].abs().max(previous.abs());
                        let roundoff = 4096.0 * f64::EPSILON * strain_scale;
                        assert!(strain[0] - *previous >= -256.0 * f64::EPSILON * strain_scale,
                            "axial reversal cells={cells} schedule={requested_increments} factor={} previous={previous} strain={strain:?}", increment.state.load_factor);
                        assert!(
                            strain[1..].iter().all(|s| s.abs() <= roundoff),
                            "plane ties failed to remove nonaxial strain: {strain:?}"
                        );
                        if strain_scale > 0.0 {
                            maximum_relative_shear = maximum_relative_shear.max(
                                strain[3..].iter().fold(0.0_f64, |n, s| n.hypot(*s)) / strain_scale,
                            );
                        }
                        let alpha = history.equivalent_plastic_strain;
                        let radial = [alpha, -0.5 * alpha, -0.5 * alpha, 0.0, 0.0, 0.0];
                        let direction_error = history
                            .engineering_plastic_strain
                            .iter()
                            .zip(radial)
                            .fold(0.0_f64, |n, (actual, expected)| n.hypot(actual - expected));
                        assert!(
                            direction_error <= 4096.0 * f64::EPSILON * strain_scale.max(alpha),
                            "nonradial axial-subspace history: {history:?}"
                        );
                        if alpha > 0.0 {
                            maximum_relative_direction =
                                maximum_relative_direction.max(direction_error / alpha);
                        }
                        *previous = strain[0];
                    }
                    accepted_count += 1;
                    state = increment.state;
                    final_evaluation = Some(increment.evaluation);
                }
                assert_eq!(state.load_factor, target);
            }
            assert!(accepted_count >= requested_increments);
            let evaluation = final_evaluation.unwrap();
            let errors = evidence::measure(
                &mesh,
                &summary,
                &state.displacement,
                &state.constitutive.points,
                state.accumulated_plastic_dissipation_j,
                &evaluation,
            );
            eprintln!("axial-subspace cells={cells} schedule={requested_increments} accepted={accepted_count} shear={maximum_relative_shear:e} plastic_direction={maximum_relative_direction:e} errors={errors:?}");
            assert!(
                errors.displacement < 1e-7 && errors.stress < 1e-7 && errors.alpha < 1e-7,
                "registered final axial-subspace field bounds: {errors:?}"
            );
            assert!(
                errors.energies.iter().all(|e| *e < 1e-7) && errors.lateral_moment < 1e-7,
                "registered final axial-subspace energy/reaction bounds: {errors:?}"
            );
        }
    }
}
