use super::*;

pub(super) fn policy() -> SolidNewtonPolicy {
    SolidNewtonPolicy {
        max_iterations: 32,
        relative_residual_tolerance: 1e-9,
        linear_relative_tolerance: 1e-12,
        line_search: true,
        max_backtracks: 16,
        backtrack_reduction: 0.5,
        max_cutbacks: 8,
        max_accepted_subincrements: 1024,
        minimum_load_increment: 1e-8,
    }
}

pub(super) fn run(cells: usize, quadratic: bool, requested_increments: usize) -> Run {
    run_with_rule(
        cells,
        quadratic,
        requested_increments,
        crate::topology::CanonicalSolidIntegrationRule::Keast15,
    )
}

pub(super) fn run_with_rule(
    cells: usize,
    quadratic: bool,
    requested_increments: usize,
    rule: crate::topology::CanonicalSolidIntegrationRule,
) -> Run {
    let (model, mesh, summary) = fixture::prepare(cells, quadratic);
    let plan = crate::topology::CanonicalSolidIntegrationPlan::derive(&mesh, rule).unwrap();
    let basis =
        CanonicalNonlinearSolidBasis::prepare_with_integration(&model, &mesh, &summary, &plan)
            .unwrap();
    let problem = SolidNewtonProblem::prepare(&basis, &summary, policy()).unwrap();
    let mut state = problem.initial_state().unwrap();
    let mut final_evaluation = None;
    let mut previous_strains = vec![0.0_f64; basis.point_count()];
    let mut minimum_strain_increment = f64::INFINITY;
    for increment in 1..=requested_increments {
        let before = state.clone();
        let mut accepted = problem
            .advance(&state, increment as f64 / requested_increments as f64)
            .unwrap();
        assert_eq!(state, before);
        for subincrement in &accepted {
            assert_eq!(
                subincrement.evaluation.point_evidence.len(),
                previous_strains.len()
            );
            for (point, previous) in subincrement
                .evaluation
                .point_evidence
                .iter()
                .zip(&mut previous_strains)
            {
                let actual = point.total_engineering_strain[0];
                let increment = actual - *previous;
                minimum_strain_increment = minimum_strain_increment.min(increment);
                let roundoff = 256.0 * f64::EPSILON * actual.abs().max(previous.abs());
                assert!(increment >= -roundoff,
                    "strain reversal: cells={cells}, quadratic={quadratic}, requested={requested_increments}, previous={previous}, actual={actual}, increment={increment}");
                *previous = actual;
            }
        }
        let completion = accepted.pop().unwrap();
        assert!(completion.relative_residual <= policy().relative_residual_tolerance);
        state = completion.state;
        final_evaluation = Some(completion.evaluation);
    }
    assert_eq!(state.load_factor, 1.0);
    problem.validate_accepted(&state).unwrap();
    assert_eq!(state.constitutive.basis_identity, basis.identity());
    assert_eq!(state.constitutive.mesh_digest, mesh.canonical_digest);
    let evaluation = final_evaluation.unwrap();
    let errors = evidence::measure(
        &mesh,
        &summary,
        &state.displacement,
        &state.constitutive.points,
        state.accumulated_plastic_dissipation_j,
        &evaluation,
    );
    // Proportionality diagnostic: uniaxial strain keeps plastic flow along [1, -1/2, -1/2, 0, 0, 0].
    let plastic_off_axis = state
        .constitutive
        .points
        .iter()
        .map(|point| {
            let p = point.engineering_plastic_strain;
            let axial = p[0];
            [p[1] + 0.5 * axial, p[2] + 0.5 * axial, p[3], p[4], p[5]]
                .iter()
                .fold(0.0_f64, |m, v| m.max(v.abs()))
                / point.equivalent_plastic_strain.max(f64::MIN_POSITIVE)
        })
        .fold(0.0_f64, f64::max);
    eprintln!("nonlinear cube rule={rule:?} cells={cells} quadratic={quadratic} requestedincrements={requested_increments} minimumstrainincrement={minimum_strain_increment} plasticoffaxis={plastic_off_axis:e} errors={errors:?}");
    Run {
        errors,
        plastic_off_axis,
        displacement: state.displacement,
        alpha: state
            .constitutive
            .points
            .iter()
            .map(|p| p.equivalent_plastic_strain)
            .collect(),
    }
}
