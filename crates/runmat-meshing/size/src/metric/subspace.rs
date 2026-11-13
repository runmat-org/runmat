//! Bounded conservative envelope of sampled directional constraints.
//! At most three orthonormal directions are retained. A rotating tangent does
//! not repeatedly add its finest density to the same curved plane.
use super::{MetricContractError, MetricTensor3};

#[derive(Clone, Copy, Debug)]
pub struct DirectionalMetricEnvelope {
    base: MetricTensor3,
    basis: [[f64; 3]; 3],
    rank: usize,
    required_density: f64,
}
impl DirectionalMetricEnvelope {
    pub fn new(base: MetricTensor3) -> Result<Self, MetricContractError> {
        base.validate()?;
        Ok(Self {
            base,
            basis: [[0.0; 3]; 3],
            rank: 0,
            required_density: 0.0,
        })
    }
    pub fn include(
        &mut self,
        direction: [f64; 3],
        length_m: f64,
    ) -> Result<(), MetricContractError> {
        let direction = super::directional::unit_direction(direction)?;
        let required = MetricTensor3::isotropic_length_m(length_m)?.xx;
        self.include_density(direction, required)
    }
    pub(super) fn include_density(
        &mut self,
        direction: [f64; 3],
        required: f64,
    ) -> Result<(), MetricContractError> {
        let direction = super::directional::unit_direction(direction)?;
        if !required.is_finite() || required <= 0.0 {
            return Err(MetricContractError::invalid(
                "directional density",
                "must be finite and positive",
            ));
        }
        let mut residual = direction;
        // Reorthogonalization prevents repeated, rotated samples from inventing
        // a third direction solely from orthogonalization roundoff.
        for _ in 0..2 {
            for basis in &self.basis[..self.rank] {
                let dot: f64 = residual.iter().zip(basis).map(|(a, b)| a * b).sum();
                for axis in 0..3 {
                    residual[axis] -= dot * basis[axis];
                }
            }
        }
        let remaining = residual[0].hypot(residual[1]).hypot(residual[2]);
        if self.rank < 3 && remaining > 64.0 * f64::EPSILON {
            self.basis[self.rank] = residual.map(|v| v / remaining);
            self.rank += 1;
        }
        self.required_density = self.required_density.max(required);
        Ok(())
    }
    pub fn metric(&self) -> Result<MetricTensor3, MetricContractError> {
        if self.rank == 0 {
            return Ok(self.base);
        }
        let base = self.base;
        // Gershgorin gives a conservative lower bound on baseline density in
        // any direction. Zero remains safe for non-diagonally-dominant SPD.
        let lower = (base.xx - base.xy.abs() - base.xz.abs())
            .min(base.yy - base.xy.abs() - base.yz.abs())
            .min(base.zz - base.xz.abs() - base.yz.abs())
            .max(0.0);
        let delta = (self.required_density - lower).max(0.0);
        if delta == 0.0 {
            return Ok(base);
        }
        let projector: [[f64; 3]; 3] = if self.rank == 3 {
            [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        } else {
            std::array::from_fn(|i| {
                std::array::from_fn(|j| self.basis[..self.rank].iter().map(|v| v[i] * v[j]).sum())
            })
        };
        // Covers discarded orthogonalization residuals (angle^2) and the small
        // fixed dot-product inventory. This is sizing, not an interval proof;
        // independent constructive mesh acceptance still applies.
        let delta = delta * (1.0 + 128.0 * f64::EPSILON);
        let metric = MetricTensor3 {
            xx: base.xx + delta * projector[0][0],
            yy: base.yy + delta * projector[1][1],
            zz: base.zz + delta * projector[2][2],
            xy: base.xy + delta * projector[0][1],
            xz: base.xz + delta * projector[0][2],
            yz: base.yz + delta * projector[1][2],
        };
        metric.validate()?;
        Ok(metric)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: Sampled directional quadratic forms preserve the input SPD constraint and unrelated
    //   orthogonal directions while satisfying the measured directional length. Invalid direction and
    //   length rejection has separate tests. This is a finite numerical kernel family, not an interval
    //   proof or a complete geometric sizing qualification.
    #[test]
    fn refinement_preserves_unrelated_axes_and_existing_constraints() {
        let base = MetricTensor3::isotropic_length_m(0.01).unwrap();
        for direction in [[0., 1., 0.], [0., 3., 4.], [0., -4., 3.]] {
            let mut envelope = DirectionalMetricEnvelope::new(base).unwrap();
            envelope.include(direction, 0.0002).unwrap();
            let metric = envelope.metric().unwrap();
            assert_eq!(metric.xx, base.xx);
            let norm = direction[1].hypot(direction[2]);
            let unit = direction.map(|v| v / norm);
            let density = |v: [f64; 3]| {
                let [x, y, z] = v;
                metric.xx * x * x
                    + metric.yy * y * y
                    + metric.zz * z * z
                    + 2. * (metric.xy * x * y + metric.xz * x * z + metric.yz * y * z)
            };
            assert!(density(unit) >= 0.0002_f64.powi(-2) * (1. - 1e-12));
            let perpendicular = [0., -unit[2], unit[1]];
            assert!((density(perpendicular) / base.xx - 1.).abs() < 1e-12);
            let original = metric;
            envelope.include(direction, 0.01).unwrap();
            assert_eq!(envelope.metric().unwrap(), original);
        }
    }
    #[test]
    fn directional_input_rejects_invalid_geometry_and_lengths_without_mutation() {
        let base = MetricTensor3::isotropic_length_m(1.).unwrap();
        let mut envelope = DirectionalMetricEnvelope::new(base).unwrap();
        for direction in [[0.; 3], [f64::NAN, 0., 1.], [f64::INFINITY, 1., 0.]] {
            assert!(envelope.include(direction, 0.1).is_err());
            assert_eq!(envelope.metric().unwrap(), base);
        }
        for length in [0., -1., f64::NAN, f64::INFINITY, 1e-300] {
            assert!(envelope.include([1., 0., 0.], length).is_err());
            assert_eq!(envelope.metric().unwrap(), base);
        }
    }

    // witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
    //   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
    //   boundary: Nine,33 and4097 rotating tangent constraints retain one planar envelope with at most
    //   three basis directions, without repeated density growth or refinement of the orthogonal axis.
    //   This finite numerical component test is not an interval proof, a spatially local sizing policy
    //   or original CAD/PDE acceptance.
    #[test]
    fn repeated_rotating_tangents_have_one_bounded_planar_envelope() {
        let base = MetricTensor3::isotropic_length_m(0.01).unwrap();
        for count in [9, 33, 4097] {
            let mut envelope = DirectionalMetricEnvelope::new(base).unwrap();
            for index in 0..count {
                let a = std::f64::consts::TAU * index as f64 / (count - 1) as f64;
                envelope.include([a.cos(), a.sin(), 0.], 0.0002).unwrap();
            }
            let metric = envelope.metric().unwrap();
            let required = 0.0002_f64.powi(-2);
            assert_eq!(metric.zz, base.zz);
            assert!(metric.xx >= required * (1. - 1e-14) && metric.xx <= required * (1. + 1e-12));
            assert!(metric.yy >= required * (1. - 1e-14) && metric.yy <= required * (1. + 1e-12));
            assert!(metric.xy.abs() < required * 1e-14);
        }
    }
}
