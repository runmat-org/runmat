//! Diagnose original accepted FE trajectories without changing equilibrium or acceptance.
//! Kernel replay is bookkeeping evidence, NOT an independent constitutive benchmark.
use super::*;
use crate::physics::structural::plasticity::{IsotropicConstitutiveModel, PlasticityPointState};

mod quadrature;

fn norm(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |n, v| n.hypot(*v))
}

#[derive(Clone, Copy, Default)]
struct AxialHistory {
    plastic: f64,
    alpha: f64,
}

impl AxialHistory {
    /// Independent scalar uniaxial-strain counterfactual driven by captured e_xx.
    /// This deliberately removes FE shear; it is not a replacement for the actual state.
    fn advance(&mut self, strain: f64) {
        let trial_difference = 2.0 * mu() * (strain - 1.5 * self.plastic);
        let excess = trial_difference.abs() - YIELD - HARDENING * self.alpha;
        if excess > 0.0 {
            let increment = excess / (3.0 * mu() + HARDENING);
            self.plastic += trial_difference.signum() * increment;
            self.alpha += increment;
        }
    }
}

#[test]
fn original_accepted_paths_preserve_point_history_and_expose_nonradial_strain() {
    for quadratic in [false, true] {
        for cells in [1, 2, 4] {
            let (model, mesh, summary) = fixture::prepare(cells, quadratic);
            let basis = CanonicalNonlinearSolidBasis::prepare(&model, &mesh, &summary).unwrap();
            let problem =
                SolidNewtonProblem::prepare(&basis, &summary, execution::policy()).unwrap();
            let material = IsotropicConstitutiveModel::from_material(&model.materials[0]).unwrap();
            let mut state = problem.initial_state().unwrap();
            let mut replay = vec![PlasticityPointState::default(); basis.point_count()];
            let mut axial = vec![AxialHistory::default(); basis.point_count()];
            let mut snapshots = Vec::new();
            let mut replay_work = 0.0;
            let mut accepted_count = 0;
            for target in (1..=8).map(|i| i as f64 / 8.0) {
                // Keep EVERY accepted cutback, not just the requested target endpoint.
                let increments = problem.advance(&state, target).unwrap();
                for increment in increments {
                    assert!(increment.state.load_factor > state.load_factor);
                    assert!(increment.state.load_factor <= target);
                    problem.validate_accepted(&increment.state).unwrap();
                    assert!(increment.relative_residual <= 1e-9);
                    snapshots.push(quadrature::Snapshot {
                        factor: increment.state.load_factor,
                        displacement: increment.state.displacement.clone(),
                        reduced_force: problem
                            .reduction
                            .transpose_apply(&increment.evaluation.internal_force)
                            .unwrap(),
                    });
                    assert_eq!(
                        state.constitutive.point_identities,
                        increment.state.constitutive.point_identities
                    );
                    assert_eq!(increment.evaluation.point_evidence.len(), replay.len());
                    let mut axial_norm = 0.0;
                    let mut shear_norm = 0.0;
                    let mut direction_error = 0.0;
                    let mut shear_history_error = 0.0;
                    let mut continuum_history_error = 0.0;
                    let mut final_history_norm = 0.0;
                    for (ordinal, point) in increment.evaluation.point_evidence.iter().enumerate() {
                        assert_eq!(point.identity, state.constitutive.point_identities[ordinal]);
                        assert_eq!(replay[ordinal], state.constitutive.points[ordinal]);
                        let response = material
                            .update(point.total_engineering_strain, &replay[ordinal])
                            .unwrap();
                        // Same original inputs must recover the SAME six components/alpha,
                        // stress and work. This checks mapping/commit, not physical accuracy.
                        assert_eq!(response.state, increment.state.constitutive.points[ordinal]);
                        assert_eq!(response.stress_pa, point.stress_pa);
                        assert_eq!(
                            response.plastic_dissipation_increment_pa,
                            point.plastic_dissipation_increment_pa
                        );
                        let weight = point.volume_weight_m3;
                        replay_work += weight * response.plastic_dissipation_increment_pa;
                        replay[ordinal] = response.state;
                        let strain = point.total_engineering_strain;
                        // Uy/Uz are constrained, but Ux(y,z) can induce xz/xy shear.
                        assert!(strain[1].abs() < 1e-12 && strain[2].abs() < 1e-12);
                        axial_norm += weight * strain[0].powi(2);
                        shear_norm += weight * 0.5 * strain[3..].iter().map(|v| v * v).sum::<f64>();
                        axial[ordinal].advance(strain[0]);
                        let history = &replay[ordinal];
                        let radial = [
                            history.equivalent_plastic_strain,
                            -history.equivalent_plastic_strain / 2.0,
                            -history.equivalent_plastic_strain / 2.0,
                            0.0,
                            0.0,
                            0.0,
                        ];
                        direction_error += weight
                            * history
                                .engineering_plastic_strain
                                .iter()
                                .zip(radial)
                                .enumerate()
                                .map(|(i, (actual, radial))| {
                                    (actual - radial).powi(2) * if i < 3 { 1.0 } else { 0.5 }
                                })
                                .sum::<f64>();
                        shear_history_error += weight
                            * (history.equivalent_plastic_strain - axial[ordinal].alpha).powi(2);
                        continuum_history_error += weight
                            * (history.equivalent_plastic_strain
                                - reference(point.position_m[0], increment.state.load_factor)
                                    .alpha)
                                .powi(2);
                        final_history_norm += weight * alpha(point.position_m[0]).powi(2);
                    }
                    assert!(
                        (replay_work - increment.state.accumulated_plastic_dissipation_j).abs()
                            < 1e-10
                    );
                    assert!(axial_norm > 0.0 && final_history_norm > 0.0);
                    let shear = (shear_norm / axial_norm).sqrt();
                    let direction = (direction_error / final_history_norm).sqrt();
                    let shear_history = (shear_history_error / final_history_norm).sqrt();
                    let continuum_history = (continuum_history_error / final_history_norm).sqrt();
                    assert!([shear, direction, shear_history, continuum_history]
                        .iter()
                        .all(|v| v.is_finite()));
                    eprintln!("captured-path cells={cells} quadratic={quadratic} factor={} shear={shear:e} plastic_direction={direction:e} shear_alpha={shear_history:e} continuum_alpha={continuum_history:e}", increment.state.load_factor);
                    accepted_count += 1;
                    state = increment.state;
                }
                assert_eq!(state.load_factor, target);
            }
            assert!(accepted_count >= 8);
            quadrature::compare(
                &mesh, &summary, &problem, &snapshots, &material, cells, quadratic,
            );
        }
    }
}
