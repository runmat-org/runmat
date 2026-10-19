//! One baseline and one bounded positive-subspace envelope for the whole source set.
//! Each B-A <= positive_part(B-A); the union projector contains every retained
//! positive eigendirection. Maximum density/error are independent of source count.
use super::super::{DirectionalMetricEnvelope, MetricContractError, MetricTensor3};

pub(super) struct TensorEnvelope {
    base: MetricTensor3,
    directions: DirectionalMetricEnvelope,
    lower: f64,
    error: f64,
}
impl TensorEnvelope {
    pub fn new(base: MetricTensor3) -> Result<Self, MetricContractError> {
        let lower = (base.xx - base.xy.abs() - base.xz.abs())
            .min(base.yy - base.xy.abs() - base.yz.abs())
            .min(base.zz - base.xz.abs() - base.yz.abs())
            .max(0.);
        Ok(Self {
            base,
            directions: DirectionalMetricEnvelope::new(base)?,
            lower,
            error: 0.,
        })
    }
    pub fn include(&mut self, metric: MetricTensor3) -> Result<(), MetricContractError> {
        metric.validate()?;
        let spectrum = super::spectral::difference(metric, self.base)?;
        // Unresolved tiny eigenvalues are covered by a small isotropic residual
        // allowance, rather than inventing an entire third strong direction.
        self.error = self.error.max(2. * spectrum.error);
        for (value, vector) in spectrum.values.into_iter().zip(spectrum.vectors) {
            if value > spectrum.error {
                self.directions
                    .include_density(vector, self.lower + value)?;
            }
        }
        Ok(())
    }
    pub fn metric(self) -> Result<MetricTensor3, MetricContractError> {
        let mut metric = self.directions.metric()?;
        metric.xx += self.error;
        metric.yy += self.error;
        metric.zz += self.error;
        metric.validate()?;
        Ok(metric)
    }
}
