use crate::FeaElectroThermalContext;

pub fn severity(context: Option<FeaElectroThermalContext>) -> f64 {
    let Some(context) = context else {
        return 0.0;
    };
    if !context.enabled {
        return 0.0;
    }
    (context.applied_voltage_v.powi(2)
        * context.base_electrical_conductivity_s_per_m.max(1.0e-9)
        * context.resistive_heating_coefficient.max(0.0)
        / 1.0e7)
        .clamp(0.0, 1.0)
}

pub fn time_scale(context: Option<FeaElectroThermalContext>, normalized_time: f64) -> f64 {
    let Some(context) = context else {
        return 1.0;
    };
    if context.time_profile.is_empty() {
        return 1.0;
    }
    let t = normalized_time.clamp(0.0, 1.0);
    let mut points = context.time_profile;
    points.sort_by(|a, b| a.normalized_time.total_cmp(&b.normalized_time));
    if t <= points[0].normalized_time {
        return points[0].current_scale.clamp(0.2, 2.0);
    }
    for pair in points.windows(2) {
        let a = &pair[0];
        let b = &pair[1];
        if t >= a.normalized_time && t <= b.normalized_time {
            let span = (b.normalized_time - a.normalized_time).abs().max(1.0e-9);
            let alpha = (t - a.normalized_time) / span;
            return (a.current_scale + (b.current_scale - a.current_scale) * alpha).clamp(0.2, 2.0);
        }
    }
    points
        .last()
        .map(|p| p.current_scale.clamp(0.2, 2.0))
        .unwrap_or(1.0)
}

pub fn temporal_profile_variation(context: Option<FeaElectroThermalContext>) -> f64 {
    let Some(context) = context else {
        return 0.0;
    };
    if context.time_profile.len() < 2 {
        return 0.0;
    }
    let mut min_scale = f64::INFINITY;
    let mut max_scale = -f64::INFINITY;
    for point in &context.time_profile {
        min_scale = min_scale.min(point.current_scale);
        max_scale = max_scale.max(point.current_scale);
    }
    if !min_scale.is_finite() || !max_scale.is_finite() {
        return 0.0;
    }
    ((max_scale - min_scale).abs() / 2.0).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn electro_time_scale_interpolates_in_range() {
        let context = FeaElectroThermalContext {
            enabled: true,
            reference_temperature_k: 293.15,
            applied_voltage_v: 24.0,
            base_electrical_conductivity_s_per_m: 3.0e7,
            resistive_heating_coefficient: 5.0e-4,
            region_conductivity_scales: Vec::new(),
            time_profile: vec![
                crate::FeaElectroTimeProfilePoint {
                    normalized_time: 0.0,
                    current_scale: 0.5,
                },
                crate::FeaElectroTimeProfilePoint {
                    normalized_time: 1.0,
                    current_scale: 1.5,
                },
            ],
        };
        let scale = time_scale(Some(context), 0.5);
        assert!(scale >= 0.5 && scale <= 1.5);
    }
}
