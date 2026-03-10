use crate::{FeaThermoFieldInterpolationMode, FeaThermoMechanicalContext};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ThermoTimeSample {
    pub scale: f64,
    pub extrapolated: bool,
    pub clamped: bool,
}

pub(crate) fn temporal_profile_variation(context: &FeaThermoMechanicalContext) -> f64 {
    if context.time_profile.len() < 2 {
        return 0.0;
    }
    let mut min_scale = f64::INFINITY;
    let mut max_scale = -f64::INFINITY;
    for point in &context.time_profile {
        min_scale = min_scale.min(point.scale);
        max_scale = max_scale.max(point.scale);
    }
    if !min_scale.is_finite() || !max_scale.is_finite() {
        return 0.0;
    }
    ((max_scale - min_scale).abs() / 2.0).clamp(0.0, 1.0)
}

pub(crate) fn sample_time_profile_scale(
    context: &FeaThermoMechanicalContext,
    normalized_time: f64,
) -> ThermoTimeSample {
    if context.time_profile.is_empty() {
        return ThermoTimeSample {
            scale: 1.0,
            extrapolated: false,
            clamped: false,
        };
    }
    let mut points = context.time_profile.clone();
    points.sort_by(|a, b| a.normalized_time.total_cmp(&b.normalized_time));
    let t = normalized_time.clamp(0.0, 1.0);

    let interpolation = context
        .field_source
        .as_ref()
        .and_then(|source| source.interpolation_mode)
        .unwrap_or(FeaThermoFieldInterpolationMode::Linear);

    if t <= points[0].normalized_time {
        let raw = points[0].scale;
        let scale = raw.clamp(0.2, 2.0);
        return ThermoTimeSample {
            scale,
            extrapolated: t < points[0].normalized_time,
            clamped: (scale - raw).abs() > 0.0,
        };
    }

    for pair in points.windows(2) {
        let a = &pair[0];
        let b = &pair[1];
        if t >= a.normalized_time && t <= b.normalized_time {
            let raw = match interpolation {
                FeaThermoFieldInterpolationMode::Linear => {
                    let span = (b.normalized_time - a.normalized_time).abs().max(1.0e-9);
                    let alpha = (t - a.normalized_time) / span;
                    a.scale + (b.scale - a.scale) * alpha
                }
                FeaThermoFieldInterpolationMode::Step => a.scale,
            };
            let scale = raw.clamp(0.2, 2.0);
            return ThermoTimeSample {
                scale,
                extrapolated: false,
                clamped: (scale - raw).abs() > 0.0,
            };
        }
    }

    let raw = points.last().map(|p| p.scale).unwrap_or(1.0);
    let scale = raw.clamp(0.2, 2.0);
    ThermoTimeSample {
        scale,
        extrapolated: t > points.last().map(|p| p.normalized_time).unwrap_or(1.0),
        clamped: (scale - raw).abs() > 0.0,
    }
}
