//! Scatterhist numeric domain behavior.
use super::*;

pub(super) fn scale_values(mut values: Vec<f64>, scale: f64) -> Vec<f64> {
    if scale != 1.0 {
        for value in &mut values {
            *value *= scale;
        }
    }
    values
}

pub(super) fn stairs_from_counts(
    edges: &[f64],
    counts: &[f64],
    horizontal: bool,
) -> (Vec<f64>, Vec<f64>) {
    let mut x = Vec::with_capacity(counts.len() * 2);
    let mut y = Vec::with_capacity(counts.len() * 2);
    for (idx, count) in counts.iter().copied().enumerate() {
        let left = edges[idx];
        let right = edges[idx + 1];
        if horizontal {
            x.push(count);
            y.push(left);
            x.push(count);
            y.push(right);
        } else {
            x.push(left);
            y.push(count);
            x.push(right);
            y.push(count);
        }
    }
    (x, y)
}

pub(super) fn combined_limits(values: impl Iterator<Item = f64>) -> BuiltinResult<(f64, f64)> {
    let mut lo: Option<f64> = None;
    let mut hi: Option<f64> = None;
    for value in values.filter(|value| value.is_finite()) {
        lo = Some(lo.map_or(value, |current| current.min(value)));
        hi = Some(hi.map_or(value, |current| current.max(value)));
    }
    let (mut lo, mut hi) = match (lo, hi) {
        (Some(lo), Some(hi)) => (lo, hi),
        _ => (0.0, 1.0),
    };
    if lo == hi {
        let pad = lo.abs().max(1.0) * 0.5;
        let lower = lo - pad;
        let upper = hi + pad;
        if lower.is_finite() {
            lo = lower;
        }
        if upper.is_finite() {
            hi = upper;
        }
    }
    if !lo.is_finite() || !hi.is_finite() || lo >= hi {
        return Err(invalid(
            "scatterhist: finite observations cannot form distinct histogram limits",
        ));
    }
    Ok((lo, hi))
}

pub(super) fn bin_edges((lo, hi): (f64, f64), bins: usize) -> BuiltinResult<Vec<f64>> {
    if bins == 0 || bins > MAX_BINS || !lo.is_finite() || !hi.is_finite() || lo >= hi {
        return Err(invalid("scatterhist: invalid histogram boundaries"));
    }
    let count = bins
        .checked_add(1)
        .ok_or_else(|| invalid("scatterhist: histogram boundary count overflow"))?;
    let mut edges = Vec::new();
    edges
        .try_reserve_exact(count)
        .map_err(|_| invalid("scatterhist: histogram boundary allocation failed"))?;
    for index in 0..=bins {
        let fraction = index as f64 / bins as f64;
        let edge = if index == 0 {
            lo
        } else if index == bins {
            hi
        } else {
            lo.mul_add(1.0 - fraction, hi * fraction)
        };
        if !edge.is_finite() || edges.last().is_some_and(|previous| edge <= *previous) {
            return Err(invalid(
                "scatterhist: histogram boundaries are not distinct at this precision",
            ));
        }
        edges.push(edge);
    }
    Ok(edges)
}

pub(super) fn auto_bin_count(data: &[f64], limits: (f64, f64)) -> usize {
    let finite = data
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value >= limits.0 && *value <= limits.1)
        .collect::<Vec<_>>();
    if finite.len() < 2 {
        return 2;
    }
    let width = scott_bin_width(&finite);
    if !width.is_finite() || width <= 0.0 {
        return 2;
    }
    let bins = ((limits.1 - limits.0) / width).ceil().max(2.0) as usize;
    bins.clamp(2, DEFAULT_AUTO_BIN_CAP.min(MAX_BINS))
}

pub(super) fn scott_bin_width(data: &[f64]) -> f64 {
    let factor = 3.5 / (data.len() as f64).powf(1.0 / 3.0);
    super::super::dispersion::finite_sample_std_scaled(data, factor).unwrap_or(0.0)
}

pub(super) fn histogram_counts(data: &[f64], edges: &[f64]) -> Vec<f64> {
    let bins = edges.len().saturating_sub(1);
    let mut counts = vec![0.0; bins];
    for value in data.iter().copied().filter(|value| value.is_finite()) {
        if value < edges[0] || value > edges[bins] {
            continue;
        }
        let idx = if value == edges[bins] {
            bins - 1
        } else {
            edges
                .partition_point(|edge| *edge <= value)
                .saturating_sub(1)
        };
        if let Some(count) = counts.get_mut(idx) {
            *count += 1.0;
        }
    }
    counts
}

pub(super) fn kde_line(
    data: &[f64],
    edges: &[f64],
    bandwidth: Option<f64>,
) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
    let centers = edges
        .windows(2)
        .map(|pair| pair[0] * 0.5 + pair[1] * 0.5)
        .collect::<Vec<_>>();
    let finite = data
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return Ok((centers, vec![0.0; edges.len().saturating_sub(1)]));
    }
    let gap = edges[1] - edges[0];
    let half_gap = if gap.is_finite() {
        gap.abs() / 2.0
    } else {
        ((edges[1] / 2.0) - (edges[0] / 2.0)).abs()
    };
    let bandwidth = bandwidth.unwrap_or_else(|| scott_bandwidth(&finite).max(half_gap));
    if !bandwidth.is_finite() || bandwidth <= 0.0 {
        return Err(invalid(
            "scatterhist: automatic KDE bandwidth is not finite and positive",
        ));
    }
    let log_norm = -bandwidth.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln();
    let mut values = Vec::with_capacity(centers.len());
    for center in &centers {
        let density = finite
            .iter()
            .map(|value| {
                let difference = *center - *value;
                let z = if difference.is_finite() {
                    difference / bandwidth
                } else {
                    *center / bandwidth - *value / bandwidth
                };
                (-0.5 * z * z + log_norm).exp()
            })
            .sum::<f64>();
        let count_scaled = density * finite.len() as f64;
        if !count_scaled.is_finite() {
            return Err(invalid("scatterhist: marginal density is not finite"));
        }
        values.push(count_scaled);
    }
    Ok((centers, values))
}

pub(super) fn scott_bandwidth(data: &[f64]) -> f64 {
    if data.len() == 1 {
        // Preserve the established singleton sample deviation of one.
        return 1.06;
    }
    let factor = 1.06 * (data.len() as f64).powf(-0.2);
    super::super::dispersion::finite_sample_std_scaled(data, factor)
        .filter(|bandwidth| *bandwidth > 0.0)
        .unwrap_or(1.0)
}
