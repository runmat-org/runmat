//! Prepare numerical marginal geometry before mutating the figure.
use super::*;

#[derive(Clone, Copy)]
pub(super) enum MarginalAxis {
    X,
    Y,
}

pub(super) struct PreparedMarginal {
    pub(super) kde: Option<(Vec<f64>, Vec<f64>)>,
    pub(super) counts: Option<Vec<f64>>,
}

pub(super) fn prepare_marginals(
    series: &[GroupSeries],
    options: &ScatterhistOptions,
    edges: &[f64],
    axis: MarginalAxis,
    scale: f64,
) -> BuiltinResult<Vec<PreparedMarginal>> {
    let mut prepared = Vec::new();
    prepared
        .try_reserve_exact(series.len())
        .map_err(|_| invalid("scatterhist: marginal allocation failed"))?;
    for item in series {
        let (samples, bandwidth) = match axis {
            MarginalAxis::X => (&item.marginal_x, item.style.bandwidth.map(|b| b[0])),
            MarginalAxis::Y => (&item.marginal_y, item.style.bandwidth.map(|b| b[1])),
        };
        let kde = if matches!(options.kernel, KernelMode::On | KernelMode::Overlay) {
            let (centers, density) = kde_line(samples, edges, bandwidth)?;
            let density = scale_values(density, scale);
            if density.iter().any(|value| !value.is_finite()) {
                return Err(invalid("scatterhist: marginal density is not finite"));
            }
            Some(match axis {
                MarginalAxis::X => (centers, density),
                MarginalAxis::Y => (density, centers),
            })
        } else {
            None
        };
        let counts = if matches!(options.kernel, KernelMode::Off | KernelMode::Overlay) {
            let counts = scale_values(histogram_counts(samples, edges), scale);
            if counts.iter().any(|value| !value.is_finite()) {
                return Err(invalid("scatterhist: marginal count is not finite"));
            }
            Some(counts)
        } else {
            None
        };
        prepared.push(PreparedMarginal { kde, counts });
    }
    Ok(prepared)
}
