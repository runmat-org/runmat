use runmat_analysis_core::{
    ComplexSpectralDensity, RandomExcitationSpectrum, RandomSpectrumInterpolation,
};

use crate::contracts::FeaRunError;

use super::complex::Complex;

pub(super) fn interpolate_spectral_matrix(
    spectrum: &RandomExcitationSpectrum,
    frequency_hz: f64,
) -> Result<Vec<Complex>, FeaRunError> {
    if !frequency_hz.is_finite() {
        return Err(FeaRunError::Solve(
            "random-response evaluation frequency is not finite".into(),
        ));
    }
    let frequencies = spectrum.frequencies_hz.canonical_values();
    let spectral_density = spectrum
        .resolved_spectral_density()
        .map_err(FeaRunError::Solve)?;
    let last = frequencies.len() - 1;
    let (lower, upper, fraction) =
        match frequencies.binary_search_by(|sample| sample.total_cmp(&frequency_hz)) {
            Ok(index) => (index, index, 0.0),
            Err(0) => return extrapolated_matrix(spectrum, 0, frequency_hz),
            Err(index) if index > last => return extrapolated_matrix(spectrum, last, frequency_hz),
            Err(upper) => {
                let lower = upper - 1;
                let fraction =
                    (frequency_hz - frequencies[lower]) / (frequencies[upper] - frequencies[lower]);
                (lower, upper, fraction)
            }
        };
    if lower == upper {
        return Ok(matrix_values(
            spectral_density[lower].values.canonical_values(),
        ));
    }
    match spectrum.interpolation {
        RandomSpectrumInterpolation::LinearFrequency => Ok(spectral_density[lower]
            .values
            .canonical_values()
            .iter()
            .zip(spectral_density[upper].values.canonical_values())
            .map(|(left, right)| {
                Complex::new(
                    left.real + fraction * (right.real - left.real),
                    left.imaginary + fraction * (right.imaginary - left.imaginary),
                )
            })
            .collect()),
    }
}

fn extrapolated_matrix(
    spectrum: &RandomExcitationSpectrum,
    endpoint: usize,
    frequency_hz: f64,
) -> Result<Vec<Complex>, FeaRunError> {
    use runmat_analysis_core::RandomSpectrumExtrapolation;

    match spectrum.extrapolation {
        RandomSpectrumExtrapolation::Reject => Err(FeaRunError::Solve(format!(
            "frequency {frequency_hz} Hz is outside random-spectrum support"
        ))),
        RandomSpectrumExtrapolation::HoldEndpoint => {
            let spectral_density = spectrum
                .resolved_spectral_density()
                .map_err(FeaRunError::Solve)?;
            Ok(matrix_values(
                spectral_density[endpoint].values.canonical_values(),
            ))
        }
    }
}

fn matrix_values(values: &[ComplexSpectralDensity]) -> Vec<Complex> {
    values
        .iter()
        .map(|value| Complex::new(value.real, value.imaginary))
        .collect()
}

/// Evaluates `transfer * spectrum * transferᴴ` and admits only scale-sized roundoff.
pub(super) fn response_density(
    transfer: &[Complex],
    spectral_matrix: &[Complex],
) -> Result<f64, FeaRunError> {
    let width = transfer.len();
    if width == 0 || spectral_matrix.len() != width * width {
        return Err(FeaRunError::Solve(
            "random-response transfer and spectral matrix shapes differ".into(),
        ));
    }
    let mut result = Complex::default();
    let mut scale = 0.0;
    for row in 0..width {
        for column in 0..width {
            let term = transfer[row]
                * spectral_matrix[row * width + column]
                * transfer[column].conjugate();
            result += term;
            scale += term.norm();
        }
    }
    let tolerance = 512.0 * f64::EPSILON * scale.max(f64::MIN_POSITIVE) * width as f64;
    if !result.real.is_finite()
        || !result.imaginary.is_finite()
        || result.imaginary.abs() > tolerance
        || result.real < -tolerance
    {
        return Err(FeaRunError::Solve(
            "random-response spectral projection is not a real nonnegative density".into(),
        ));
    }
    Ok(result.real.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correlated_quadratic_form_retains_phase() {
        let transfer = [Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)];
        let spectral_matrix = [
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(0.0, -1.0),
            Complex::new(1.0, 0.0),
        ];
        assert!((response_density(&transfer, &spectral_matrix).unwrap() - 4.0).abs() < 1.0e-14);
    }
}
