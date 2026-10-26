use super::fixture::{
    channel, complete_physical_basis, constant_spectrum, converged_policy, full_frequency_bounds,
};
use crate::{
    pipeline::modal_basis::ModalBasisArtifact, solve::dynamics::solve_modal_random_response,
};

#[test]
fn cumulative_mass_threshold_selects_the_expected_modal_prefix() {
    let (summary, basis) = complete_physical_basis();
    let direction = direction_with_distinct_modal_contributions(&basis);
    let cumulative = cumulative_mass_ratios(&basis, direction);
    let excitation = constant_spectrum(
        full_frequency_bounds(&basis),
        vec![channel("prefix", direction)],
        runmat_analysis_core::CrossPowerSpectralDensity::diagonal(
            &runmat_types::physical::Quantity::new([2.0e-4]),
        ),
    );

    for expected_count in 1..=basis.modes().len() {
        let threshold = if expected_count == 1 {
            cumulative[0] * 0.5
        } else {
            0.5 * (cumulative[expected_count - 2] + cumulative[expected_count - 1])
        };
        let result =
            solve_modal_random_response(&summary, &basis, &excitation, converged_policy(threshold))
                .unwrap();
        // The selected prefix is closed over a repeated-frequency cluster.
        let modes = basis.modes();
        let cluster_frequency = modes[expected_count - 1].frequency_hz();
        let closed_count = expected_count
            + modes[expected_count..]
                .iter()
                .take_while(|mode| {
                    (mode.frequency_hz() - cluster_frequency).abs() <= 1.0e-8 * cluster_frequency
                })
                .count();
        assert_eq!(result.selected_mode_count(), closed_count);
        assert!(
            (result.captured_mass_ratio_by_channel()[0] - cumulative[closed_count - 1]).abs()
                < 1.0e-12
        );
    }
}

fn direction_with_distinct_modal_contributions(basis: &ModalBasisArtifact) -> [f64; 3] {
    let candidates = [
        [0.73, -0.41, 0.29],
        [0.21, 0.67, -0.71],
        [-0.49, 0.31, 0.81],
        [1.0, 1.0, 1.0],
    ];
    candidates
        .into_iter()
        .find(|direction| {
            let cumulative = cumulative_mass_ratios(basis, *direction);
            cumulative.windows(2).all(|pair| pair[1] - pair[0] > 1.0e-6) && cumulative[0] > 1.0e-6
        })
        .expect("qualification basis should have a generic direction exciting every mode")
}

fn cumulative_mass_ratios(basis: &ModalBasisArtifact, direction: [f64; 3]) -> Vec<f64> {
    let norm = direction
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let direction = direction.map(|value| value / norm);
    let total = basis.total_free_mass_in_direction(direction);
    let mut captured = 0.0;
    basis
        .modes()
        .iter()
        .map(|mode| {
            let participation = direction
                .iter()
                .zip(mode.directional())
                .map(|(component, metric)| component * metric.participation_factor())
                .sum::<f64>();
            captured += participation * participation * mode.modal_mass();
            captured / total
        })
        .collect()
}
