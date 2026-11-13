use runmat_analysis_core::{
    CoordinateFrame, ModalCombination, ModalTruncationCorrection, ResponseSpectrum,
    RigidResponsePolicy, SpectrumDamping, SpectrumEnvelope,
};

use crate::{assembly::AssemblySummary, contracts::FeaRunError, ModalBasisArtifact};

use super::{
    combine::{combine_modal_responses, combine_periodic_and_rigid_responses},
    correction::{correction_response, rigid_coefficients},
    interpolate::interpolate_spectrum,
    validation::vectors_close,
    ModalResponseSpectrumContribution, ModalResponseSpectrumModeContribution,
    ModalResponseSpectrumModeKind, ModalResponseSpectrumTruncationEvidence,
};

pub(super) fn solve_spectrum_direction(
    summary: Option<&AssemblySummary>,
    basis: &ModalBasisArtifact,
    spectrum: &ResponseSpectrum,
) -> Result<ModalResponseSpectrumContribution, FeaRunError> {
    if spectrum.direction.frame != CoordinateFrame::Global {
        return Err(FeaRunError::Solve(
            "modal response currently requires spectrum directions in the global frame".into(),
        ));
    }
    if matches!(
        spectrum.truncation.correction,
        ModalTruncationCorrection::MissingMass { .. }
    ) && summary.is_none()
    {
        return Err(FeaRunError::Solve(
            "missing-mass correction requires the exact physical operator".into(),
        ));
    }
    let direction = normalized_direction(spectrum.direction.vector)?;
    let damping_ratio = damping_ratio(spectrum.damping)?;
    let total_free_mass = basis.total_free_mass_in_direction(direction);
    if !total_free_mass.is_finite() || total_free_mass <= 0.0 {
        return Err(FeaRunError::Solve(
            "spectrum direction has no positive free structural mass".into(),
        ));
    }

    let mut captured_mass = 0.0;
    let mut modes = Vec::new();
    let mut modal_vectors = Vec::new();
    for (mode_offset, mode) in basis.modes().iter().enumerate() {
        let participation = direction
            .iter()
            .zip(mode.directional())
            .map(|(component, metric)| component * metric.participation_factor())
            .sum::<f64>();
        let effective_mass = participation * participation * mode.modal_mass();
        captured_mass += effective_mass;
        let cumulative_mass_ratio = captured_mass / total_free_mass;
        let ordinate = interpolate_spectrum(spectrum, mode.frequency_hz())?;
        let envelope_sign = match spectrum.envelope {
            SpectrumEnvelope::Negative => -1.0,
            SpectrumEnvelope::Positive | SpectrumEnvelope::Maximax => 1.0,
        };
        let generalized_peak = envelope_sign * participation * ordinate;
        modes.push(ModalResponseSpectrumModeContribution {
            kind: ModalResponseSpectrumModeKind::Eigenmode,
            mode_index: mode_offset + 1,
            frequency_hz: mode.frequency_hz(),
            interpolated_ordinate: ordinate,
            directional_participation_factor: participation,
            directional_effective_mass: effective_mass,
            cumulative_mass_ratio,
            generalized_peak,
        });
        modal_vectors.push(
            mode.dof_shape()
                .iter()
                .map(|shape| shape * generalized_peak)
                .collect::<Vec<_>>(),
        );
        if spectrum.truncation.correction != ModalTruncationCorrection::ResidualMode
            && cumulative_mass_ratio + 64.0 * f64::EPSILON
                >= spectrum.truncation.minimum_cumulative_mass_ratio
        {
            break;
        }
    }
    let selected_eigenmode_count = modes.len();
    if spectrum.truncation.correction == ModalTruncationCorrection::ResidualMode {
        let residual = basis
            .residual_vectors()
            .iter()
            .find(|residual| vectors_close(&residual.direction(), &direction))
            .ok_or_else(|| {
                FeaRunError::Solve(
                    "modal basis does not contain the requested directional residual vector".into(),
                )
            })?;
        let ordinate = interpolate_spectrum(spectrum, residual.frequency_hz())?;
        let envelope_sign = if spectrum.envelope == SpectrumEnvelope::Negative {
            -1.0
        } else {
            1.0
        };
        let generalized_peak = envelope_sign * residual.participation_factor() * ordinate;
        captured_mass += residual.effective_mass();
        modes.push(ModalResponseSpectrumModeContribution {
            kind: ModalResponseSpectrumModeKind::ResidualVector,
            mode_index: modes.len() + 1,
            frequency_hz: residual.frequency_hz(),
            interpolated_ordinate: ordinate,
            directional_participation_factor: residual.participation_factor(),
            directional_effective_mass: residual.effective_mass(),
            cumulative_mass_ratio: captured_mass / total_free_mass,
            generalized_peak,
        });
        modal_vectors.push(
            residual
                .dof_shape()
                .iter()
                .map(|shape| shape * generalized_peak)
                .collect(),
        );
    }
    let achieved = modes
        .last()
        .map(|mode| mode.cumulative_mass_ratio)
        .unwrap_or(0.0);
    if achieved + 64.0 * f64::EPSILON < spectrum.truncation.minimum_cumulative_mass_ratio {
        return Err(FeaRunError::Solve(format!(
            "modal basis captures {:.6} of directional mass but spectrum requires {:.6}",
            achieved, spectrum.truncation.minimum_cumulative_mass_ratio
        )));
    }
    let frequencies = modes
        .iter()
        .map(|mode| mode.frequency_hz)
        .collect::<Vec<_>>();
    if spectrum.modal_combination == ModalCombination::Srss
        && frequencies.windows(2).any(|pair| {
            (pair[1] - pair[0]).abs() / pair[0].abs().max(pair[1].abs()).max(1.0e-12) <= 0.1
        })
    {
        return Err(FeaRunError::Solve(
            "SRSS requires every selected mode to be separated by more than ten percent".into(),
        ));
    }
    let rigid_coefficients = rigid_coefficients(spectrum, &modes)?;
    let (periodic_response, mut rigid_response) =
        if spectrum.rigid_response == RigidResponsePolicy::Disabled {
            (
                combine_modal_responses(
                    &modal_vectors,
                    &frequencies,
                    damping_ratio,
                    spectrum.modal_combination,
                )?,
                vec![0.0; modal_vectors[0].len()],
            )
        } else {
            combine_periodic_and_rigid_responses(
                &modal_vectors,
                &frequencies,
                &rigid_coefficients,
                damping_ratio,
                spectrum.modal_combination,
            )?
        };
    let correction_response = correction_response(
        summary,
        basis,
        spectrum,
        direction,
        &modes,
        modal_vectors[0].len(),
    )?;
    let combined_response = if spectrum.rigid_response == RigidResponsePolicy::Disabled {
        periodic_response
            .iter()
            .zip(&correction_response)
            .map(|(periodic, correction)| (periodic * periodic + correction * correction).sqrt())
            .collect()
    } else {
        for (rigid, correction) in rigid_response.iter_mut().zip(&correction_response) {
            *rigid += correction;
        }
        periodic_response
            .iter()
            .zip(&rigid_response)
            .map(|(periodic, rigid)| (periodic * periodic + rigid * rigid).sqrt())
            .collect()
    };
    Ok(ModalResponseSpectrumContribution {
        spectrum_id: spectrum.spectrum_id.clone(),
        spectrum_provenance_digest: spectrum
            .ordinates
            .numerical_digest()
            .map_err(FeaRunError::Solve)?
            .to_owned(),
        direction,
        damping_ratio: runmat_types::physical::DampingRatio::new(damping_ratio),
        modal_combination: spectrum.modal_combination,
        rigid_response_policy: spectrum.rigid_response,
        rigid_coefficients,
        truncation: ModalResponseSpectrumTruncationEvidence {
            selected_eigenmode_count,
            available_eigenmode_count: basis.truncation().available_mode_count(),
            residual_vector_count: modes.len() - selected_eigenmode_count,
            required_cumulative_mass_ratio: spectrum.truncation.minimum_cumulative_mass_ratio,
            achieved_cumulative_mass_ratio: achieved,
            directional_total_free_mass: total_free_mass,
            directional_captured_mass: captured_mass,
            correction: spectrum.truncation.correction,
            correction_response,
        },
        modes,
        periodic_response,
        rigid_response,
        combined_response,
    })
}

fn normalized_direction(vector: [f64; 3]) -> Result<[f64; 3], FeaRunError> {
    let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(FeaRunError::Solve(
            "spectrum direction must be finite and nonzero".into(),
        ));
    }
    Ok(vector.map(|value| value / norm))
}

fn damping_ratio(damping: SpectrumDamping) -> Result<f64, FeaRunError> {
    let ratio = match damping {
        SpectrumDamping::Ratio(value) => value,
        SpectrumDamping::QualityFactor(value) => 1.0 / (2.0 * value),
    };
    if !ratio.is_finite() || !(0.0..1.0).contains(&ratio) {
        return Err(FeaRunError::Solve(
            "modal response requires a damping ratio in [0, 1)".into(),
        ));
    }
    Ok(ratio)
}
