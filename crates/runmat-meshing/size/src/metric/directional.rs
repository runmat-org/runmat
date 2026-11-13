//! One-dimensional traces retain every baseline tensor constraint.
use super::{MetricContractError, MetricTensor3};
pub(super) fn unit_direction(v: [f64; 3]) -> Result<[f64; 3], MetricContractError> {
    let norm = v[0].hypot(v[1]).hypot(v[2]);
    if v.iter().any(|x| !x.is_finite()) || !norm.is_finite() || norm <= 0.0 {
        return Err(MetricContractError::invalid(
            "metric direction",
            "must be finite and nonzero",
        ));
    }
    Ok(v.map(|x| x / norm))
}
fn quadratic(m: MetricTensor3, [x, y, z]: [f64; 3]) -> f64 {
    x * (m.xx * x + m.xy * y + m.xz * z)
        + y * (m.xy * x + m.yy * y + m.yz * z)
        + z * (m.xz * x + m.yz * y + m.zz * z)
}
impl MetricTensor3 {
    /// The physical spacing tensor maps its target edge to unit metric length.
    pub const NORMALIZED_TARGET_LENGTH: f64 = 1.0;
    pub fn directional_density(self, direction: [f64; 3]) -> Result<f64, MetricContractError> {
        self.validate()?;
        let value = quadratic(self, unit_direction(direction)?);
        if value.is_finite() && value > 0.0 {
            Ok(value)
        } else {
            Err(MetricContractError::invalid(
                "directional metric density",
                "must be finite and positive",
            ))
        }
    }
    /// Tighten one trace without accumulating unrelated face tensors or
    /// weakening any baseline direction. Each query starts from its own base.
    pub fn with_directional_density(
        self,
        direction: [f64; 3],
        required: f64,
    ) -> Result<Self, MetricContractError> {
        self.validate()?;
        let [x, y, z] = unit_direction(direction)?;
        if !required.is_finite() || required <= 0.0 {
            return Err(MetricContractError::invalid(
                "required directional density",
                "must be finite and positive",
            ));
        }
        let current = quadratic(self, [x, y, z]);
        if !current.is_finite() || current <= 0.0 {
            return Err(MetricContractError::invalid(
                "baseline directional density",
                "must be finite and positive",
            ));
        }
        if current >= required {
            return Ok(self);
        }
        let delta = (required - current).next_up();
        let metric = Self {
            xx: self.xx + delta * x * x,
            yy: self.yy + delta * y * y,
            zz: self.zz + delta * z * z,
            xy: self.xy + delta * x * y,
            xz: self.xz + delta * x * z,
            yz: self.yz + delta * y * z,
        };
        metric.validate()?;
        Ok(metric)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn trace_tightening_preserves_baseline_and_does_not_accumulate_across_queries() {
        let base = MetricTensor3::isotropic_length_m(1.).unwrap();
        for direction in [[1., 0., 0.], [0.6, 0.8, 0.], [0., -0.6, 0.8]] {
            let metric = base.with_directional_density(direction, 400.).unwrap();
            assert!(metric.directional_density(direction).unwrap() >= 400. * (1. - 1e-12));
            assert_eq!(
                base.with_directional_density(direction, 400.).unwrap(),
                metric
            );
            assert_eq!(
                metric.with_directional_density(direction, 1.).unwrap(),
                metric
            );
        }
        assert!(base.with_directional_density([0.; 3], 400.).is_err());
        assert!(base
            .with_directional_density([1., 0., 0.], f64::NAN)
            .is_err());
    }
}
