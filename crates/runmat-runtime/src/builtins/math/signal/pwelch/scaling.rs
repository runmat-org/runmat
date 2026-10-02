//! Scale-invariant window normalization shared by host and resident Welch paths.

use super::*;

pub(super) struct NormalizedWindow {
    pub values: Vec<f64>,
    pub denominator: f64,
}

pub(super) async fn normalized_window(
    window: &[f64],
    scale: SpectrumScale,
    fs: Option<f64>,
    cooperation: &mut SignalCooperation,
) -> BuiltinResult<NormalizedWindow> {
    let mut maximum = 0.0_f64;
    let mut work = CompletedWork::default();
    for &value in window {
        if !value.is_finite() {
            return Err(pwelch_error(&PWELCH_ERROR_INVALID_WINDOW));
        }
        maximum = maximum.max(value.abs());
        if let Some(units) = work.record(1) {
            cooperation.completed(units).await?;
        }
    }
    work.finish(cooperation).await?;
    if maximum == 0.0 {
        return Err(pwelch_error(&PWELCH_ERROR_INVALID_WINDOW));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(window.len())
        .map_err(|cause| window_allocation_error(BUILTIN_NAME, &PWELCH_ERROR_INTERNAL, cause))?;
    let mut energy = 0.0;
    let mut sum = 0.0;
    for &value in window {
        let normalized = value / maximum;
        values.push(normalized);
        energy += normalized * normalized;
        sum += normalized;
        if let Some(units) = work.record(1) {
            cooperation.completed(units).await?;
        }
    }
    work.finish(cooperation).await?;
    let denominator = match scale {
        SpectrumScale::Psd => fs.unwrap_or(2.0 * std::f64::consts::PI) * energy,
        SpectrumScale::Power => sum * sum,
    };
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(pwelch_error(&PWELCH_ERROR_INVALID_WINDOW));
    }
    Ok(NormalizedWindow {
        values,
        denominator,
    })
}
