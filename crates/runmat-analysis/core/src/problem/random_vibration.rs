use serde::{Deserialize, Serialize};

use super::response_spectrum::SpectrumDirection;

const HERMITIAN_TOLERANCE_FACTOR: f64 = 128.0;
const POSITIVE_SEMIDEFINITE_TOLERANCE_FACTOR: f64 = 512.0;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RandomExcitationId(pub String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RandomExcitationChannelId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpectralDensityConvention {
    /// One-sided density over nonnegative ordinary frequency, in quantity squared per hertz.
    OneSidedPerHertz,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RandomSpectrumInterpolation {
    /// Componentwise linear interpolation. Convex interpolation preserves Hermitian PSD matrices.
    LinearFrequency,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RandomSpectrumExtrapolation {
    Reject,
    HoldEndpoint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RandomExcitationQuantity {
    BaseAcceleration,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RandomExcitationChannel {
    pub channel_id: RandomExcitationChannelId,
    pub quantity: RandomExcitationQuantity,
    pub direction: SpectrumDirection,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComplexSpectralDensity {
    pub real: f64,
    pub imaginary: f64,
}

impl ComplexSpectralDensity {
    pub const fn new(real: f64, imaginary: f64) -> Self {
        Self { real, imaginary }
    }

    fn norm(self) -> f64 {
        self.real.hypot(self.imaginary)
    }
}

/// One row-major cross-power spectral-density matrix at one frequency sample.
///
/// The diagonal contains auto spectra. Off-diagonal entry `(row, column)` is the cross spectrum
/// `S_row,column`; the complete matrix is retained so canonical identity and Hermitian validation
/// do not depend on an implicit packed-storage convention.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrossPowerSpectralDensity {
    pub values: Vec<ComplexSpectralDensity>,
}

impl CrossPowerSpectralDensity {
    pub fn diagonal(auto_spectra: &[f64]) -> Self {
        let width = auto_spectra.len();
        let mut values = vec![ComplexSpectralDensity::new(0.0, 0.0); width * width];
        for (index, value) in auto_spectra.iter().copied().enumerate() {
            values[index * width + index] = ComplexSpectralDensity::new(value, 0.0);
        }
        Self { values }
    }

    pub fn value(&self, width: usize, row: usize, column: usize) -> ComplexSpectralDensity {
        self.values[row * width + column]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RandomSpectrumEvidence {
    /// Maximum source-estimation or authoring error in density units, when known.
    pub max_absolute_density_error: Option<f64>,
    /// Effective independent averages for an estimated spectrum, when applicable.
    pub equivalent_degrees_of_freedom: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RandomExcitationSpectrum {
    pub excitation_id: RandomExcitationId,
    pub frequencies_hz: Vec<f64>,
    pub channels: Vec<RandomExcitationChannel>,
    pub spectral_density: Vec<CrossPowerSpectralDensity>,
    pub convention: SpectralDensityConvention,
    pub interpolation: RandomSpectrumInterpolation,
    pub extrapolation: RandomSpectrumExtrapolation,
    pub evidence: RandomSpectrumEvidence,
    pub provenance_digest: String,
}

impl RandomExcitationSpectrum {
    pub fn validate_intrinsic(&self) -> Result<(), String> {
        if self.excitation_id.0.trim().is_empty() {
            return Err("random-excitation id must be non-empty".into());
        }
        if self.frequencies_hz.len() < 2
            || self
                .frequencies_hz
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            || self
                .frequencies_hz
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(
                "random-excitation frequencies must contain at least two finite, nonnegative, strictly increasing samples"
                    .into(),
            );
        }
        if self.channels.is_empty() {
            return Err("random excitation requires at least one channel".into());
        }
        let mut channel_ids = std::collections::HashSet::new();
        for channel in &self.channels {
            if channel.channel_id.0.trim().is_empty()
                || !channel_ids.insert(channel.channel_id.0.as_str())
            {
                return Err("random-excitation channel ids must be non-empty and unique".into());
            }
            crate::validate_coordinate_frame(&channel.direction.frame)
                .map_err(|error| error.to_string())?;
            if channel
                .direction
                .vector
                .iter()
                .any(|value| !value.is_finite())
                || channel
                    .direction
                    .vector
                    .iter()
                    .map(|value| value * value)
                    .sum::<f64>()
                    <= f64::EPSILON
            {
                return Err("random-excitation directions must be finite and nonzero".into());
            }
        }
        if self.spectral_density.len() != self.frequencies_hz.len() {
            return Err("random-excitation frequency and matrix counts differ".into());
        }
        for (sample_index, matrix) in self.spectral_density.iter().enumerate() {
            validate_spectral_matrix(matrix, self.channels.len()).map_err(|detail| {
                format!("random-excitation spectral matrix {sample_index} is invalid: {detail}")
            })?;
        }
        if self
            .evidence
            .max_absolute_density_error
            .is_some_and(|value| !value.is_finite() || value < 0.0)
            || self
                .evidence
                .equivalent_degrees_of_freedom
                .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err("random-spectrum evidence must be finite and physically bounded".into());
        }
        if self.provenance_digest.trim().is_empty() {
            return Err("random-excitation provenance digest must be non-empty".into());
        }
        Ok(())
    }
}

fn validate_spectral_matrix(
    matrix: &CrossPowerSpectralDensity,
    width: usize,
) -> Result<(), String> {
    if matrix.values.len() != width * width
        || matrix
            .values
            .iter()
            .any(|value| !value.real.is_finite() || !value.imaginary.is_finite())
    {
        return Err("matrix shape or values are invalid".into());
    }
    let matrix_scale = matrix
        .values
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max);
    let scale = if matrix_scale == 0.0 {
        1.0
    } else {
        matrix_scale
    };
    let hermitian_tolerance = HERMITIAN_TOLERANCE_FACTOR * f64::EPSILON * scale * width as f64;
    for row in 0..width {
        let diagonal = matrix.value(width, row, row);
        if diagonal.real < -hermitian_tolerance || diagonal.imaginary.abs() > hermitian_tolerance {
            return Err("auto spectra must be real and nonnegative".into());
        }
        for column in 0..row {
            let lower = matrix.value(width, row, column);
            let upper = matrix.value(width, column, row);
            if (lower.real - upper.real).abs() > hermitian_tolerance
                || (lower.imaginary + upper.imaginary).abs() > hermitian_tolerance
            {
                return Err("cross spectra must form a Hermitian matrix".into());
            }
        }
    }
    validate_positive_semidefinite(matrix, width, scale)
}

/// Scale-aware unpivoted LDLᴴ admission for a Hermitian positive-semidefinite matrix.
///
/// For a zero semidefinite pivot, every remaining entry in that residual column must also be zero;
/// otherwise a 2x2 principal minor is negative. This permits exactly rank-deficient spectra without
/// weakening admission of an indefinite cross-spectrum matrix.
fn validate_positive_semidefinite(
    matrix: &CrossPowerSpectralDensity,
    width: usize,
    scale: f64,
) -> Result<(), String> {
    let tolerance = POSITIVE_SEMIDEFINITE_TOLERANCE_FACTOR * f64::EPSILON * scale * width as f64;
    let mut lower = vec![ComplexSpectralDensity::new(0.0, 0.0); width * width];
    let mut diagonal = vec![0.0; width];
    for pivot in 0..width {
        let mut pivot_value = matrix.value(width, pivot, pivot).real;
        for prior in 0..pivot {
            let value = lower[pivot * width + prior];
            pivot_value -= value.norm().powi(2) * diagonal[prior];
        }
        if pivot_value < -tolerance {
            return Err("cross-spectral matrix is not positive semidefinite".into());
        }
        if pivot_value.abs() <= tolerance {
            diagonal[pivot] = 0.0;
            for row in (pivot + 1)..width {
                let residual = schur_entry(matrix, &lower, &diagonal, width, row, pivot);
                if residual.norm() > tolerance {
                    return Err(
                        "cross-spectral matrix has a nonzero column at a zero PSD pivot".into(),
                    );
                }
            }
            continue;
        }
        diagonal[pivot] = pivot_value;
        lower[pivot * width + pivot] = ComplexSpectralDensity::new(1.0, 0.0);
        for row in (pivot + 1)..width {
            let residual = schur_entry(matrix, &lower, &diagonal, width, row, pivot);
            lower[row * width + pivot] = ComplexSpectralDensity::new(
                residual.real / pivot_value,
                residual.imaginary / pivot_value,
            );
        }
    }
    Ok(())
}

fn schur_entry(
    matrix: &CrossPowerSpectralDensity,
    lower: &[ComplexSpectralDensity],
    diagonal: &[f64],
    width: usize,
    row: usize,
    column: usize,
) -> ComplexSpectralDensity {
    let mut residual = matrix.value(width, row, column);
    for prior in 0..column {
        let left = lower[row * width + prior];
        let right = lower[column * width + prior];
        let weight = diagonal[prior];
        residual.real -= weight * (left.real * right.real + left.imaginary * right.imaginary);
        residual.imaginary -= weight * (left.imaginary * right.real - left.real * right.imaginary);
    }
    residual
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CoordinateFrame;

    fn channel(id: &str, direction: [f64; 3]) -> RandomExcitationChannel {
        RandomExcitationChannel {
            channel_id: RandomExcitationChannelId(id.into()),
            quantity: RandomExcitationQuantity::BaseAcceleration,
            direction: SpectrumDirection {
                frame: CoordinateFrame::Global,
                vector: direction,
            },
        }
    }

    fn fixture(matrix: CrossPowerSpectralDensity) -> RandomExcitationSpectrum {
        RandomExcitationSpectrum {
            excitation_id: RandomExcitationId("launch-random".into()),
            frequencies_hz: vec![10.0, 20.0],
            channels: vec![channel("x", [1.0, 0.0, 0.0]), channel("y", [0.0, 1.0, 0.0])],
            spectral_density: vec![matrix.clone(), matrix],
            convention: SpectralDensityConvention::OneSidedPerHertz,
            interpolation: RandomSpectrumInterpolation::LinearFrequency,
            extrapolation: RandomSpectrumExtrapolation::Reject,
            evidence: RandomSpectrumEvidence {
                max_absolute_density_error: Some(0.0),
                equivalent_degrees_of_freedom: None,
            },
            provenance_digest: "sha256:launch-random".into(),
        }
    }

    #[test]
    fn correlated_rank_one_spectrum_is_admitted() {
        let matrix = CrossPowerSpectralDensity {
            values: vec![
                ComplexSpectralDensity::new(4.0, 0.0),
                ComplexSpectralDensity::new(0.0, 2.0),
                ComplexSpectralDensity::new(0.0, -2.0),
                ComplexSpectralDensity::new(1.0, 0.0),
            ],
        };
        fixture(matrix).validate_intrinsic().unwrap();
    }

    #[test]
    fn indefinite_or_nonhermitian_cross_spectrum_is_rejected() {
        let indefinite = CrossPowerSpectralDensity {
            values: vec![
                ComplexSpectralDensity::new(1.0, 0.0),
                ComplexSpectralDensity::new(2.0, 0.0),
                ComplexSpectralDensity::new(2.0, 0.0),
                ComplexSpectralDensity::new(1.0, 0.0),
            ],
        };
        assert!(fixture(indefinite)
            .validate_intrinsic()
            .unwrap_err()
            .contains("positive semidefinite"));

        let nonhermitian = CrossPowerSpectralDensity {
            values: vec![
                ComplexSpectralDensity::new(1.0, 0.0),
                ComplexSpectralDensity::new(0.0, 0.5),
                ComplexSpectralDensity::new(0.0, 0.5),
                ComplexSpectralDensity::new(1.0, 0.0),
            ],
        };
        assert!(fixture(nonhermitian)
            .validate_intrinsic()
            .unwrap_err()
            .contains("Hermitian"));
    }
}
