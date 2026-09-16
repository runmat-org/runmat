use runmat_analysis_core::{
    ResponseSpectrum, SpectrumAbscissa, SpectrumExtrapolation, SpectrumInterpolation,
};

use crate::contracts::FeaRunError;

pub(super) fn interpolate_spectrum(
    spectrum: &ResponseSpectrum,
    frequency_hz: f64,
) -> Result<f64, FeaRunError> {
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return Err(FeaRunError::Solve(
            "response-spectrum interpolation requires a positive modal frequency".into(),
        ));
    }
    let query = match spectrum.abscissa {
        SpectrumAbscissa::FrequencyHertz(_) => frequency_hz,
        SpectrumAbscissa::PeriodSeconds(_) => 1.0 / frequency_hz,
    };
    let axis = match &spectrum.abscissa {
        SpectrumAbscissa::FrequencyHertz(values) | SpectrumAbscissa::PeriodSeconds(values) => {
            values
        }
    };
    let index = match axis.binary_search_by(|value| value.total_cmp(&query)) {
        Ok(index) => return Ok(spectrum.ordinates[index]),
        Err(0) => return extrapolate(spectrum, 0),
        Err(index) if index == axis.len() => return extrapolate(spectrum, axis.len() - 1),
        Err(index) => index,
    };
    let x0 = axis[index - 1];
    let x1 = axis[index];
    let y0 = spectrum.ordinates[index - 1];
    let y1 = spectrum.ordinates[index];
    let value = match spectrum.interpolation {
        SpectrumInterpolation::LinearFrequency | SpectrumInterpolation::LinearPeriod => {
            y0 + (query - x0) * (y1 - y0) / (x1 - x0)
        }
        SpectrumInterpolation::LogLog => {
            let fraction = (query / x0).ln() / (x1 / x0).ln();
            (y0.ln() + fraction * (y1 / y0).ln()).exp()
        }
    };
    if !value.is_finite() || value < 0.0 {
        return Err(FeaRunError::Solve(
            "response-spectrum interpolation produced an invalid ordinate".into(),
        ));
    }
    Ok(value)
}

fn extrapolate(spectrum: &ResponseSpectrum, endpoint: usize) -> Result<f64, FeaRunError> {
    match spectrum.extrapolation {
        SpectrumExtrapolation::Reject => Err(FeaRunError::Solve(
            "modal frequency lies outside the response-spectrum domain".into(),
        )),
        SpectrumExtrapolation::HoldEndpoint => Ok(spectrum.ordinates[endpoint]),
    }
}
