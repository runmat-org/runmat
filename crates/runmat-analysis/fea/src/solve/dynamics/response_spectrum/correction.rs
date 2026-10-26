use runmat_analysis_core::{
    ModalTruncationCorrection, ResponseSpectrum, RigidResponsePolicy, SpectrumAbscissa,
    SpectrumEnvelope,
};

use crate::{
    assembly::AssemblySummary, contracts::FeaRunError,
    pipeline::modal_basis::directional_influence_vector, ModalBasisArtifact,
};

use super::ModalResponseSpectrumModeContribution;

pub(super) fn rigid_coefficients(
    spectrum: &ResponseSpectrum,
    modes: &[ModalResponseSpectrumModeContribution],
) -> Result<Vec<f64>, FeaRunError> {
    let coefficients = match spectrum.rigid_response {
        RigidResponsePolicy::Disabled => vec![0.0; modes.len()],
        RigidResponsePolicy::Gupta {
            periodic_frequency_hz,
            rigid_frequency_hz,
        } => modes
            .iter()
            .map(|mode| {
                if mode.frequency_hz <= periodic_frequency_hz {
                    0.0
                } else if mode.frequency_hz >= rigid_frequency_hz {
                    1.0
                } else {
                    (mode.frequency_hz / periodic_frequency_hz).ln()
                        / (rigid_frequency_hz / periodic_frequency_hz).ln()
                }
            })
            .collect(),
        RigidResponsePolicy::LindleyYow {
            zero_period_acceleration,
        } => {
            let ordinates = spectrum.resolved_ordinates().map_err(FeaRunError::Solve)?;
            let maximum = ordinates.iter().copied().fold(0.0_f64, f64::max);
            let peak_onset_hz = ordinates
                .iter()
                .enumerate()
                .filter(|(_, ordinate)| **ordinate == maximum)
                .map(|(index, _)| match &spectrum.abscissa {
                    SpectrumAbscissa::FrequencyHertz(values) => values.canonical_values()[index],
                    SpectrumAbscissa::PeriodSeconds(values) => {
                        1.0 / values.canonical_values()[index]
                    }
                })
                .fold(f64::INFINITY, f64::min);
            modes
                .iter()
                .map(|mode| {
                    if mode.frequency_hz < peak_onset_hz {
                        0.0
                    } else if mode.interpolated_ordinate == 0.0 {
                        f64::from(zero_period_acceleration > 0.0)
                    } else {
                        (zero_period_acceleration / mode.interpolated_ordinate).clamp(0.0, 1.0)
                    }
                })
                .collect()
        }
    };
    if coefficients
        .iter()
        .any(|coefficient| !coefficient.is_finite() || !(0.0..=1.0).contains(coefficient))
    {
        return Err(FeaRunError::Solve(
            "rigid-response policy produced an invalid coefficient".into(),
        ));
    }
    Ok(coefficients)
}

pub(super) fn correction_response(
    summary: Option<&AssemblySummary>,
    basis: &ModalBasisArtifact,
    spectrum: &ResponseSpectrum,
    direction: [f64; 3],
    modes: &[ModalResponseSpectrumModeContribution],
    dof_count: usize,
) -> Result<Vec<f64>, FeaRunError> {
    match spectrum.truncation.correction {
        ModalTruncationCorrection::None => Ok(vec![0.0; dof_count]),
        ModalTruncationCorrection::ResidualMode => Ok(vec![0.0; dof_count]),
        ModalTruncationCorrection::MissingMass {
            zero_period_acceleration,
        } => {
            let summary = summary.ok_or_else(|| {
                FeaRunError::Solve("missing-mass correction requires the physical operator".into())
            })?;
            let mut residual = directional_influence_vector(summary, direction);
            if residual.len() != dof_count {
                return Err(FeaRunError::Solve(
                    "missing-mass influence vector has the wrong DOF dimension".into(),
                ));
            }
            for (mode, contribution) in basis.modes().iter().zip(modes) {
                for (residual, shape) in residual.iter_mut().zip(mode.dof_shape()) {
                    *residual -= contribution.directional_participation_factor * shape;
                }
            }
            let sign = if spectrum.envelope == SpectrumEnvelope::Negative {
                -1.0
            } else {
                1.0
            };
            Ok(residual
                .into_iter()
                .map(|value| sign * zero_period_acceleration * value)
                .collect())
        }
    }
}
