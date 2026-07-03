use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeshingTolerance {
    pub absolute_m: f64,
    pub relative: f64,
}

impl Default for MeshingTolerance {
    fn default() -> Self {
        Self {
            absolute_m: 1.0e-12,
            relative: 1.0e-10,
        }
    }
}

impl MeshingTolerance {
    pub fn from_bounds(bounds_min_m: [f64; 3], bounds_max_m: [f64; 3]) -> Self {
        let span = (0..3)
            .map(|axis| bounds_max_m[axis] - bounds_min_m[axis])
            .filter(|span| span.is_finite())
            .fold(0.0_f64, f64::max)
            .max(1.0);
        Self {
            absolute_m: (span * 1.0e-10).max(1.0e-12),
            relative: 1.0e-10,
        }
    }

    pub fn length_epsilon(self, scale_m: f64) -> f64 {
        let scale_m = if scale_m.is_finite() {
            scale_m.abs()
        } else {
            1.0
        };
        self.absolute_m.max(scale_m * self.relative)
    }

    pub fn volume_epsilon(self, scale_m: f64) -> f64 {
        self.length_epsilon(scale_m).powi(3)
    }

    pub fn nearly_equal(self, left: f64, right: f64, scale: f64) -> bool {
        (left - right).abs() <= self.length_epsilon(scale)
    }

    pub fn point_nearly_equal(self, left: [f64; 3], right: [f64; 3], scale_m: f64) -> bool {
        super::predicate::distance(left, right) <= self.length_epsilon(scale_m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tolerance_scales_from_bounds() {
        let tolerance = MeshingTolerance::from_bounds([0.0, 0.0, 0.0], [10.0, 2.0, 1.0]);

        assert!(tolerance.absolute_m >= 1.0e-9);
        assert!(tolerance.nearly_equal(1.0, 1.0 + 1.0e-10, 10.0));
        assert!(!tolerance.nearly_equal(1.0, 1.0 + 1.0e-4, 10.0));
    }
}
