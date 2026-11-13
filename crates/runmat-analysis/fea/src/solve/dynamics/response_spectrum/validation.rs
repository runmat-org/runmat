use runmat_analysis_core::{ModalTruncationCorrection, RigidResponsePolicy};

use super::{ModalResponseSpectrumContribution, ModalResponseSpectrumModeKind};

pub(super) fn contribution_is_valid(
    contribution: &ModalResponseSpectrumContribution,
    response_length: usize,
) -> bool {
    let truncation = &contribution.truncation;
    let summed_mass = contribution
        .modes
        .iter()
        .map(|mode| mode.directional_effective_mass)
        .sum::<f64>();
    let expected_ratio = summed_mass / truncation.directional_total_free_mass;
    let direction_norm = contribution
        .direction
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let response_vectors_valid = [
        contribution.periodic_response.as_slice(),
        contribution.rigid_response.as_slice(),
        truncation.correction_response.as_slice(),
        contribution.combined_response.as_slice(),
    ]
    .iter()
    .all(|vector| vector.len() == response_length && vector.iter().all(|value| value.is_finite()));
    let response_is_consistent = (0..response_length).all(|index| {
        let expected = (contribution.periodic_response[index].powi(2)
            + if contribution.rigid_response_policy == RigidResponsePolicy::Disabled {
                truncation.correction_response[index].powi(2)
            } else {
                contribution.rigid_response[index].powi(2)
            })
        .sqrt();
        scalar_close(expected, contribution.combined_response[index])
    });
    !contribution.spectrum_id.0.trim().is_empty()
        && !contribution.spectrum_provenance_digest.trim().is_empty()
        && contribution.direction.iter().all(|value| value.is_finite())
        && scalar_close(direction_norm, 1.0)
        && contribution.damping_ratio.canonical_value().is_finite()
        && (0.0..1.0).contains(&contribution.damping_ratio.canonical_value())
        && !contribution.modes.is_empty()
        && contribution.rigid_coefficients.len() == contribution.modes.len()
        && contribution
            .rigid_coefficients
            .iter()
            .all(|coefficient| coefficient.is_finite() && (0.0..=1.0).contains(coefficient))
        && response_vectors_valid
        && response_is_consistent
        && contribution
            .combined_response
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
        && contribution.modes.iter().enumerate().all(|(index, mode)| {
            mode.mode_index == index + 1
                && (index < truncation.selected_eigenmode_count
                    && mode.kind == ModalResponseSpectrumModeKind::Eigenmode
                    || index >= truncation.selected_eigenmode_count
                        && mode.kind == ModalResponseSpectrumModeKind::ResidualVector)
                && mode.frequency_hz.is_finite()
                && mode.frequency_hz >= 0.0
                && mode.interpolated_ordinate.is_finite()
                && mode.interpolated_ordinate >= 0.0
                && mode.directional_participation_factor.is_finite()
                && mode.directional_effective_mass.is_finite()
                && mode.directional_effective_mass >= 0.0
                && mode.cumulative_mass_ratio.is_finite()
                && mode.cumulative_mass_ratio >= 0.0
                && mode.generalized_peak.is_finite()
        })
        && contribution.modes.windows(2).all(|pair| {
            pair[0].frequency_hz <= pair[1].frequency_hz
                && pair[0].cumulative_mass_ratio <= pair[1].cumulative_mass_ratio
        })
        && truncation.selected_eigenmode_count + truncation.residual_vector_count
            == contribution.modes.len()
        && truncation.available_eigenmode_count >= truncation.selected_eigenmode_count
        && truncation.required_cumulative_mass_ratio.is_finite()
        && (0.0..=1.0).contains(&truncation.required_cumulative_mass_ratio)
        && truncation.achieved_cumulative_mass_ratio.is_finite()
        && truncation.achieved_cumulative_mass_ratio >= 0.0
        && truncation.directional_total_free_mass.is_finite()
        && truncation.directional_total_free_mass > 0.0
        && truncation.directional_captured_mass.is_finite()
        && truncation.directional_captured_mass >= 0.0
        && match truncation.correction {
            ModalTruncationCorrection::ResidualMode => {
                truncation.residual_vector_count == 1
                    && truncation
                        .correction_response
                        .iter()
                        .all(|value| *value == 0.0)
            }
            ModalTruncationCorrection::None => {
                truncation.residual_vector_count == 0
                    && truncation
                        .correction_response
                        .iter()
                        .all(|value| *value == 0.0)
            }
            ModalTruncationCorrection::MissingMass { .. } => truncation.residual_vector_count == 0,
        }
        && (contribution.rigid_response_policy != RigidResponsePolicy::Disabled
            || (contribution
                .rigid_coefficients
                .iter()
                .all(|coefficient| *coefficient == 0.0)
                && contribution
                    .rigid_response
                    .iter()
                    .all(|value| *value == 0.0)))
        && scalar_close(summed_mass, truncation.directional_captured_mass)
        && scalar_close(expected_ratio, truncation.achieved_cumulative_mass_ratio)
        && truncation.achieved_cumulative_mass_ratio <= 1.0 + 1.0e-10
}

pub(super) fn vectors_close(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| scalar_close(*left, *right))
}

fn scalar_close(left: f64, right: f64) -> bool {
    (left - right).abs() <= 128.0 * f64::EPSILON * left.abs().max(right.abs()).max(1.0)
}
