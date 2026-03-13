use crate::{FeaThermoFieldInterpolationMode, FeaThermoMechanicalContext};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeProfileSample {
    pub scale: f64,
    pub extrapolated: bool,
    pub clamped: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransientPolicy {
    pub effective_residual_target: f64,
    pub growth_limit: f64,
    pub nonconverged_shrink: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NonlinearPolicy {
    pub convergence_residual_target: f64,
    pub convergence_increment_target: f64,
}

pub fn severity(context: Option<&FeaThermoMechanicalContext>) -> f64 {
    let Some(context) = context else {
        return 0.0;
    };
    if !context.enabled {
        return 0.0;
    }
    let thermal_strain = (context.thermal_expansion_coefficient
        * context.applied_temperature_delta_k.abs())
    .clamp(0.0, 0.05);
    (thermal_strain / 0.05).clamp(0.0, 1.0)
}

pub fn sample_time_profile(
    context: Option<&FeaThermoMechanicalContext>,
    normalized_time: f64,
) -> TimeProfileSample {
    let Some(context) = context else {
        return TimeProfileSample {
            scale: 1.0,
            extrapolated: false,
            clamped: false,
        };
    };
    if context.time_profile.is_empty() {
        return TimeProfileSample {
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
        return TimeProfileSample {
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
            return TimeProfileSample {
                scale,
                extrapolated: false,
                clamped: (scale - raw).abs() > 0.0,
            };
        }
    }

    let raw = points.last().map(|p| p.scale).unwrap_or(1.0);
    let scale = raw.clamp(0.2, 2.0);
    TimeProfileSample {
        scale,
        extrapolated: t > points.last().map(|p| p.normalized_time).unwrap_or(1.0),
        clamped: (scale - raw).abs() > 0.0,
    }
}

pub fn temporal_profile_variation(context: Option<&FeaThermoMechanicalContext>) -> f64 {
    let Some(context) = context else {
        return 0.0;
    };
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

pub fn transient_policy(base_residual_target: f64, severity: f64) -> TransientPolicy {
    let thermo_residual_relaxation = 1.0 + 1.5 * severity;
    TransientPolicy {
        effective_residual_target: base_residual_target * thermo_residual_relaxation,
        growth_limit: (1.0 - 0.12 * severity).clamp(0.75, 1.0),
        nonconverged_shrink: (1.0 - 0.20 * severity).clamp(0.65, 1.0),
    }
}

pub fn nonlinear_policy(
    tolerance: f64,
    residual_convergence_factor: f64,
    increment_norm_tolerance: f64,
    severity: f64,
) -> NonlinearPolicy {
    let thermo_residual_relaxation = 1.0 + 2.0 * severity;
    let thermo_increment_relaxation = 1.0 + 1.4 * severity;
    NonlinearPolicy {
        convergence_residual_target: tolerance
            * residual_convergence_factor.max(1.0)
            * thermo_residual_relaxation,
        convergence_increment_target: increment_norm_tolerance * thermo_increment_relaxation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thermo_policies_remain_bounded() {
        let transient = transient_policy(1.0e-6, 0.8);
        assert!(transient.effective_residual_target >= 1.0e-6);
        assert!((0.75..=1.0).contains(&transient.growth_limit));
        assert!((0.65..=1.0).contains(&transient.nonconverged_shrink));

        let nonlinear = nonlinear_policy(1.0e-6, 5.0, 1.0e-7, 0.8);
        assert!(nonlinear.convergence_residual_target >= 1.0e-6);
        assert!(nonlinear.convergence_increment_target >= 1.0e-7);
    }
}
