use super::*;
use crate::physics::structural::plasticity::{IsotropicConstitutiveModel, PlasticityPointState};

#[test]
fn radial_material_history_matches_the_independent_front_on_all_schedules() {
    let (model, _, _) = fixture::prepare(1, false);
    let material = IsotropicConstitutiveModel::from_material(&model.materials[0]).unwrap();
    for increments in [1, 8, 64] {
        for x in [0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0] {
            let mut state = PlasticityPointState::default();
            let mut dissipation = 0.0;
            for increment in 1..=increments {
                let factor = increment as f64 / increments as f64;
                let expected = reference(x, factor);
                let before = state.clone();
                let response = material
                    .update([expected.strain, 0.0, 0.0, 0.0, 0.0, 0.0], &state)
                    .unwrap();
                assert_eq!(state, before, "a trial update changed committed history");
                assert!((response.state.equivalent_plastic_strain - expected.alpha).abs() < 1e-12);
                let expected_plastic = [
                    expected.alpha,
                    -expected.alpha / 2.0,
                    -expected.alpha / 2.0,
                    0.0,
                    0.0,
                    0.0,
                ];
                for (actual, expected) in response
                    .state
                    .engineering_plastic_strain
                    .iter()
                    .zip(expected_plastic)
                {
                    assert!((actual - expected).abs() < 1e-12);
                }
                for (actual, expected) in response.stress_pa.iter().zip(expected.stress) {
                    assert!((actual - expected).abs() < 1e-10);
                }
                let elastic = 0.5 * bulk() * expected.strain.powi(2)
                    + 2.0 * mu() / 3.0 * (expected.strain - 1.5 * expected.alpha).powi(2);
                let hardening = 0.5 * HARDENING * expected.alpha.powi(2);
                assert!((response.elastic_energy_density_pa - elastic).abs() < 1e-12);
                assert!((response.hardening_energy_density_pa - hardening).abs() < 1e-12);
                assert!(response.plastic_dissipation_increment_pa >= 0.0);
                dissipation += response.plastic_dissipation_increment_pa;
                assert!((dissipation - YIELD * expected.alpha).abs() < 1e-12);
                state = response.state;
            }
        }
    }
}
