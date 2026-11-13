//! Current trust-region widths. Integer widths are at least one, including
//! after scale reduction; this is distinct from the RBF's physical box metric.
use super::super::{domain::BoxView, invalid};
use crate::{
    builtins::math::optim::common::{feasible_cone::Diagonal, ScalarCooperation},
    BuiltinResult,
};
use num_rational::BigRational;
use num_traits::{One, ToPrimitive};

pub(in super::super) struct Metric<'a, 'domain> {
    domain: &'a BoxView<'domain>,
    scale: BigRational,
}
impl<'a, 'domain> Metric<'a, 'domain> {
    pub(in super::super) fn new(domain: &'a BoxView<'domain>, scale: f64) -> BuiltinResult<Self> {
        if !scale.is_finite() || scale <= 0.0 || scale > 0.8 {
            return Err(invalid("invalid sampler trust metric scale"));
        }
        Ok(Self {
            domain,
            scale: BigRational::from_float(scale).expect("finite positive scale"),
        })
    }
    pub(in super::super) async fn box_ray_into(
        &self,
        ray: &mut [f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        if ray.len() != self.dimension() {
            return Err(invalid("sampler ray changed physical extent"));
        }
        for (axis, value) in ray.iter_mut().enumerate() {
            let ratio = (self.factor(axis) / self.domain.exact_width(axis))
                .to_f64()
                .filter(|value| value.is_finite() && *value > 0.0)
                .ok_or_else(|| invalid("sampler trust metric is unrepresentable"))?;
            *value *= ratio;
            if !value.is_finite() {
                return Err(invalid("sampler ray is nonfinite"));
            }
            cooperation.completed(1).await?;
        }
        Ok(())
    }
}
impl Diagonal for Metric<'_, '_> {
    fn dimension(&self) -> usize {
        self.domain.dimensions()
    }
    fn factor(&self, axis: usize) -> BigRational {
        let width = self.domain.exact_width(axis) * &self.scale;
        if self
            .domain
            .integers()
            .binary_search(&self.domain.free_columns()[axis])
            .is_ok()
        {
            (width * BigRational::new(5.into(), 2.into())).max(BigRational::one())
        } else {
            width
        }
    }
}
