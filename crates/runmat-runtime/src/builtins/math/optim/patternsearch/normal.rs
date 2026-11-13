//! Pattern diagonal/error adapter; exact normal arithmetic is shared.
use super::scaling::Scaling;
use crate::builtins::math::optim::common::{feasible_cone, ScalarCooperation};
use crate::BuiltinResult;
impl feasible_cone::Diagonal for Scaling {
    fn dimension(&self) -> usize {
        Scaling::dimension(self)
    }
    fn factor(&self, column: usize) -> num_rational::BigRational {
        num_rational::BigRational::from_float(Scaling::factor(self, column))
            .expect("positive finite scale owner")
    }
}
pub(super) struct Normal(feasible_cone::Normal);
impl Normal {
    pub async fn new(
        row: &[(usize, f64)],
        scaling: &Scaling,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        feasible_cone::Normal::new(row, scaling, super::cone::policy(), cooperation)
            .await
            .map(Self)
    }
}
impl std::ops::Deref for Normal {
    type Target = feasible_cone::Normal;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
