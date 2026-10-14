//! Display-limit normalization shared by scene queries and the renderer.
use super::AxesMetadata;
use crate::core::BoundingBox;
use glam::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AxisLimitAxis {
    X,
    Y,
    Z,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AxisLimitMethod {
    #[default]
    TickAligned,
    Tight,
    Padded,
}

impl AxisLimitMethod {
    pub fn is_default(&self) -> bool {
        matches!(self, Self::TickAligned)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::TickAligned => "tickaligned",
            Self::Tight => "tight",
            Self::Padded => "padded",
        }
    }
}

pub fn resolve_axis_bounds(
    lower: f64,
    upper: f64,
    explicit: Option<(f64, f64)>,
    method: AxisLimitMethod,
) -> (f64, f64) {
    if let Some(explicit) = explicit {
        return explicit;
    }
    if !lower.is_finite() || !upper.is_finite() {
        return (0.0, 1.0);
    }
    let (lower, upper) = if lower < upper {
        (lower, upper)
    } else {
        let padding = (lower.abs() * 0.05).max(0.5);
        (lower - padding, upper + padding)
    };
    match method {
        AxisLimitMethod::Tight => (lower, upper),
        AxisLimitMethod::Padded => {
            let padding = (upper - lower) * 0.07;
            if padding.is_finite() {
                (lower - padding, upper + padding)
            } else {
                (lower, upper)
            }
        }
        AxisLimitMethod::TickAligned => {
            let rough_step = (upper - lower) / 5.0;
            if !rough_step.is_finite() || rough_step <= f64::MIN_POSITIVE {
                return (lower, upper);
            }
            let magnitude = 10_f64.powf(rough_step.log10().floor());
            if !magnitude.is_finite() || magnitude <= 0.0 {
                return (lower, upper);
            }
            let step = [1.0, 2.0, 5.0, 10.0]
                .into_iter()
                .map(|factor| factor * magnitude)
                .find(|candidate| *candidate >= rough_step)
                .unwrap_or(10.0 * magnitude);
            let aligned = ((lower / step).floor() * step, (upper / step).ceil() * step);
            if aligned.0.is_finite() && aligned.1.is_finite() && aligned.0 < aligned.1 {
                aligned
            } else {
                (lower, upper)
            }
        }
    }
}

/// Apply explicit bounds and the axes aspect policy before cameras or export
/// consume the box. Degenerate dimensions receive the same minimum span on
/// every path, including manual limit capture.
pub fn display_bounds_3d(bounds: BoundingBox, metadata: Option<&AxesMetadata>) -> BoundingBox {
    let mut min = bounds.min;
    let mut max = bounds.max;
    if let Some(meta) = metadata {
        for (index, (limits, method)) in [
            (meta.x_limits, meta.x_limit_method),
            (meta.y_limits, meta.y_limit_method),
            (meta.z_limits, meta.z_limit_method),
        ]
        .into_iter()
        .enumerate()
        {
            let (lower, upper) =
                resolve_axis_bounds(f64::from(min[index]), f64::from(max[index]), limits, method);
            min[index] = lower as f32;
            max[index] = upper as f32;
        }
    }
    for index in 0..3 {
        if min[index].is_finite() && max[index] == min[index] {
            min[index] -= 0.05;
            max[index] += 0.05;
        }
    }
    if let Some(meta) = metadata {
        if meta.axis_equal || meta.data_aspect_ratio_mode == "manual" {
            (min, max) = data_aspect_adjusted_bounds_3d(min, max, meta.data_aspect_ratio);
        }
    }
    BoundingBox { min, max }
}

fn data_aspect_adjusted_bounds_3d(min: Vec3, max: Vec3, ratio: [f64; 3]) -> (Vec3, Vec3) {
    let aspect = Vec3::new(
        ratio[0].abs().max(1.0e-12) as f32,
        ratio[1].abs().max(1.0e-12) as f32,
        ratio[2].abs().max(1.0e-12) as f32,
    );
    let center = (min + max) * 0.5;
    let span = (max - min).abs().max(Vec3::splat(0.1));
    let units = (span.x / aspect.x)
        .max(span.y / aspect.y)
        .max(span.z / aspect.z)
        .max(0.1);
    let next = aspect * units;
    (center - next * 0.5, center + next * 0.5)
}

pub fn positive_log_bounds(lo: f64, hi: f64) -> (f64, f64) {
    if lo.is_finite() && hi.is_finite() && lo > 0.0 && hi > 0.0 && lo != hi {
        return (lo, hi);
    }
    let upper = if hi.is_finite() && hi > 0.0 {
        hi
    } else if lo.is_finite() && lo > 0.0 {
        lo
    } else {
        1.0
    };
    let lower = (upper / 1000.0).max(f64::MIN_POSITIVE);
    if lower < upper {
        (lower, upper)
    } else {
        (upper * 0.1, upper * 10.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manual_capture_preserves_tiny_and_asymmetric_three_dimensional_display() {
        for (min, max) in [
            (Vec3::ZERO, Vec3::new(0.1, 0.1, 0.)),
            (Vec3::new(-3., 2., 20.), Vec3::new(4., 8., 40.)),
        ] {
            for ratio in [[1., 1., 1.], [1., 2., 3.]] {
                let mut metadata = AxesMetadata {
                    axis_equal: true,
                    data_aspect_ratio: ratio,
                    ..Default::default()
                };
                let before = display_bounds_3d(BoundingBox { min, max }, Some(&metadata));
                metadata.x_limits = Some((before.min.x as f64, before.max.x as f64));
                metadata.y_limits = Some((before.min.y as f64, before.max.y as f64));
                metadata.z_limits = Some((before.min.z as f64, before.max.z as f64));
                metadata.y_direction = super::super::YDirection::Reverse;
                let after = display_bounds_3d(
                    BoundingBox {
                        min: Vec3::splat(-100.),
                        max: Vec3::splat(100.),
                    },
                    Some(&metadata),
                );
                assert!((before.min - after.min).abs().max_element() < 1e-5);
                assert!((before.max - after.max).abs().max_element() < 1e-5);
            }
        }
    }
}
