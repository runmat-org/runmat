use super::*;

#[derive(Debug, Clone, Copy)]
struct HistoryErrors {
    displacement: f64,
    stress: f64,
    /// Absolute L2 history error normalized by the fully yielded reference inventory.
    /// The normalization remains defined in the initial entirely elastic interval.
    alpha: f64,
    energies: [f64; 3],
    lateral_moment: f64,
    cell_norm: cell_norm::CellNorm,
}

fn measure(
    mesh: &SolverMeshArtifact,
    summary: &AssemblySummary,
    u: &[f64],
    evaluation: &NonlinearSolidEvaluation,
    factor: f64,
    accumulated_dissipation: f64,
) -> HistoryErrors {
    let projection = AnalysisProjection::derive(mesh).unwrap();
    let mut u_error = 0.0;
    let mut u_norm = 0.0;
    for element in &projection.elements {
        projection
            .visit_element_quadrature(element, |point| {
                let actual = element
                    .node_indices
                    .iter()
                    .enumerate()
                    .map(|(local, &node)| {
                        point.shape_values()[local]
                            * u[summary
                                .structural_dof_layout
                                .index(node, StructuralDofKind::Ux)
                                .unwrap()]
                    })
                    .sum::<f64>();
                let expected = reference(point.position_m[0], factor).displacement;
                u_error += point.volume_weight_m3 * (actual - expected).powi(2);
                u_norm += point.volume_weight_m3 * expected.powi(2);
            })
            .unwrap();
    }
    let mut stress_error = 0.0;
    let mut stress_norm = 0.0;
    let mut alpha_error = 0.0;
    let mut alpha_norm = 0.0;
    assert_eq!(
        evaluation.point_evidence.len(),
        evaluation.trial_state.points.len()
    );
    assert_eq!(
        evaluation.point_evidence.len(),
        evaluation.trial_state.point_identities.len()
    );
    for (point, state) in evaluation
        .point_evidence
        .iter()
        .zip(&evaluation.trial_state.points)
    {
        let expected = reference(point.position_m[0], factor);
        for (actual, expected) in point.stress_pa.iter().zip(expected.stress) {
            stress_error += point.volume_weight_m3 * (actual - expected).powi(2);
            stress_norm += point.volume_weight_m3 * expected.powi(2);
        }
        alpha_error +=
            point.volume_weight_m3 * (state.equivalent_plastic_strain - expected.alpha).powi(2);
        alpha_norm += point.volume_weight_m3 * alpha(point.position_m[0]).powi(2);
        if factor <= 0.125 {
            assert_eq!(
                state.equivalent_plastic_strain, 0.0,
                "premature numerical yield"
            );
        }
    }
    let axial_reaction = mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| node.coordinates_m[0] == 0.0)
        .map(|(i, _)| {
            let row = summary
                .structural_dof_layout
                .index(i, StructuralDofKind::Ux)
                .unwrap();
            evaluation.internal_force[row] - factor * summary.physical_load[row]
        })
        .sum::<f64>();
    let expected_reaction = -factor * stress(0.0)[0];
    assert!(
        (axial_reaction / expected_reaction - 1.0).abs() < 1e-8,
        "history reaction factor={factor} actual={axial_reaction} expected={expected_reaction}"
    );
    let (expected_energy, expected_moment) = integrals(factor);
    let history_dissipation = evaluation
        .trial_state
        .points
        .iter()
        .zip(&evaluation.point_evidence)
        .map(|(s, p)| YIELD * s.equivalent_plastic_strain * p.volume_weight_m3)
        .sum::<f64>();
    assert!(
        (history_dissipation - accumulated_dissipation).abs() < 1e-10,
        "accumulated accepted work differs from actual material history"
    );
    let actual_energy = [
        evaluation
            .element_evidence
            .iter()
            .map(|e| e.elastic_energy_j)
            .sum::<f64>(),
        evaluation
            .element_evidence
            .iter()
            .map(|e| e.hardening_energy_j)
            .sum::<f64>(),
        accumulated_dissipation,
    ];
    let final_energy = integrals(1.0).0;
    let energies =
        std::array::from_fn(|i| (actual_energy[i] - expected_energy[i]).abs() / final_energy[i]);
    let actual_moment = mesh
        .topology
        .nodes
        .iter()
        .enumerate()
        .map(|(node_index, node)| {
            [StructuralDofKind::Uy, StructuralDofKind::Uz]
                .into_iter()
                .enumerate()
                .map(|(axis, kind)| {
                    let row = summary
                        .structural_dof_layout
                        .index(node_index, kind)
                        .unwrap();
                    node.coordinates_m[axis + 1]
                        * (evaluation.internal_force[row] - factor * summary.physical_load[row])
                })
                .sum::<f64>()
        })
        .sum::<f64>();
    HistoryErrors {
        displacement: (u_error / u_norm).sqrt(),
        stress: (stress_error / stress_norm).sqrt(),
        alpha: (alpha_error / alpha_norm).sqrt(),
        energies,
        lateral_moment: (actual_moment / expected_moment - 1.0).abs(),
        cell_norm: cell_norm::measure(mesh, evaluation, factor),
    }
}

fn history(cells: usize, quadratic: bool) -> Vec<HistoryErrors> {
    history_schedule(
        cells,
        quadratic,
        &(1..=8).map(|i| i as f64 / 8.0).collect::<Vec<_>>(),
        false,
    )
}

fn history_schedule(
    cells: usize,
    quadratic: bool,
    factors: &[f64],
    aligned: bool,
) -> Vec<HistoryErrors> {
    let (model, mesh, summary) = fixture::prepare(cells, quadratic);
    let basis = CanonicalNonlinearSolidBasis::prepare(&model, &mesh, &summary).unwrap();
    let problem = SolidNewtonProblem::prepare(&basis, &summary, execution::policy()).unwrap();
    let mut state = problem.initial_state().unwrap();
    let mut evidence = Vec::new();
    for &factor in factors {
        let accepted = problem.advance(&state, factor).unwrap();
        if aligned {
            // An unaligned accepted cutback changes the history and invalidates the
            // exact representability argument. Do not hide it in a final-state check.
            assert_eq!(
                accepted.len(),
                1,
                "front-aligned trajectory acquired subincrements"
            );
            assert_eq!(accepted[0].cutbacks, 0);
        }
        for increment in &accepted {
            problem.validate_accepted(&increment.state).unwrap();
            assert!(increment.relative_residual <= 1e-9);
            let errors = measure(
                &mesh,
                &summary,
                &increment.state.displacement,
                &increment.evaluation,
                increment.state.load_factor,
                increment.state.accumulated_plastic_dissipation_j,
            );
            eprintln!(
                "accepted-front cells={cells} quadratic={quadratic} factor={} errors={errors:?}",
                increment.state.load_factor
            );
        }
        let completion = accepted.into_iter().last().unwrap();
        assert_eq!(completion.state.load_factor, factor);
        let errors = measure(
            &mesh,
            &summary,
            &completion.state.displacement,
            &completion.evaluation,
            factor,
            completion.state.accumulated_plastic_dissipation_j,
        );
        eprintln!("moving-front cells={cells} quadratic={quadratic} factor={factor} front={} errors={errors:?}", front(factor));
        evidence.push(errors);
        state = completion.state;
    }
    evidence
}

#[test]
fn tet10_front_aligned_trajectory_preserves_exact_accepted_fields() {
    let cells = 4;
    let mut factors = (0..=cells)
        .map(|i| {
            let r = 1.0 - i as f64 / cells as f64;
            elastic_tangent() * YIELD / (2.0 * mu() * stress(r)[0])
        })
        .collect::<Vec<_>>();
    factors.push(1.0);
    let evidence = history_schedule(cells, true, &factors, true);
    for (&factor, errors) in factors.iter().zip(evidence) {
        assert!(
            errors.displacement < 1e-7
                && errors.stress < 1e-7
                && errors.alpha < 1e-7
                && errors.energies.iter().all(|e| *e < 1e-7)
                && errors.lateral_moment < 1e-7,
            "exact front-aligned history factor={factor} errors={errors:?}"
        );
    }
}

#[test]
fn tet4_accepted_history_converges_through_the_moving_yield_front() {
    check(false);
}

#[test]
fn tet10_accepted_history_converges_through_the_moving_yield_front() {
    check(true);
}

#[test]
fn tet4_refined_history_satisfies_the_registered_bounds_and_asymptotic_rates() {
    // Additional resolution, not a replacement or relaxation of the failed cells4
    // gate. Freeze the same trajectory bounds and original asymptotic rate targets.
    let coarse = history(4, false);
    let fine = history(8, false);
    for errors in &fine {
        assert!(
            errors.displacement < 0.02 && errors.stress < 0.05 && errors.alpha < 0.08,
            "additional Tet4 resolution: {errors:?}"
        );
        assert!(errors.energies.iter().all(|e| *e < 0.08) && errors.lateral_moment < 0.08);
    }
    let coarse = coarse.last().unwrap();
    let fine = fine.last().unwrap();
    assert!(coarse.displacement / fine.displacement > 3.0);
    assert!(coarse.stress / fine.stress > 1.5 && coarse.alpha / fine.alpha > 1.5);
}

fn check(quadratic: bool) {
    let runs = [1, 2, 4].map(|cells| history(cells, quadratic));
    for (index, run) in runs.iter().enumerate() {
        for (increment, errors) in run.iter().enumerate() {
            let norms = errors.cell_norm;
            assert!(
                (norms.stress.powi(2)
                    - norms.stress_mean_bias.powi(2)
                    - norms.stress_projection.powi(2))
                .abs()
                    < 1e-12
            );
            assert!(
                (norms.alpha.powi(2)
                    - norms.alpha_mean_bias.powi(2)
                    - norms.alpha_projection.powi(2))
                .abs()
                    < 1e-12
            );
            eprintln!("cell-output decomposition quadratic={quadratic} cells={} increment={} norms={norms:?}", [1, 2, 4][index], increment + 1);
        }
    }
    // Independent geometric integrals of the piecewise-constant output fields,
    // registered before executing this diagnostic. Do not replace the point gates.
    for i in 0..8 {
        for (name, values) in [
            ("stress", runs.each_ref().map(|r| r[i].cell_norm.stress)),
            ("alpha", runs.each_ref().map(|r| r[i].cell_norm.alpha)),
        ] {
            assert!(values.windows(2).all(|p| p[1] < p[0] || (p[0] < 1e-10 && p[1] < 1e-10)),
                "split cell-average refinement quadratic={quadratic} increment={} field={name} errors={values:?}", i + 1);
        }
    }
    // Registered before execution: retain the pre-existing final fine-grid stepped
    // bounds across the entire trajectory. Do not assert polynomial exactness after
    // a non-mesh-aligned plastic front, or relax the separate original failed gates.
    for (i, fine) in runs[2].iter().enumerate() {
        assert!(
            fine.displacement < 0.02 && fine.stress < 0.05 && fine.alpha < 0.08,
            "history fine quadratic={quadratic} factor={} errors={fine:?}",
            (i + 1) as f64 / 8.0
        );
        assert!(
            fine.energies.iter().all(|e| *e < 0.08) && fine.lateral_moment < 0.08,
            "history fine energies/reactions quadratic={quadratic} factor={} errors={fine:?}",
            (i + 1) as f64 / 8.0
        );
        let coarse = runs[0][i];
        let middle = runs[1][i];
        for (name, values) in [
            (
                "displacement",
                [coarse.displacement, middle.displacement, fine.displacement],
            ),
            ("stress", [coarse.stress, middle.stress, fine.stress]),
            ("alpha", [coarse.alpha, middle.alpha, fine.alpha]),
        ] {
            assert!(values.windows(2).all(|p| p[1] < p[0] || (p[0] < 1e-10 && p[1] < 1e-10)),
                "history refinement quadratic={quadratic} increment={} field={name} errors={values:?}", i + 1);
        }
    }
    if quadratic {
        let elastic = runs[0][0];
        assert!(
            elastic.displacement < 1e-7 && elastic.stress < 1e-7 && elastic.alpha == 0.0,
            "Tet10 pre-yield exact quadratic: {elastic:?}"
        );
    }
}
