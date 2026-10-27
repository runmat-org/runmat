//! Pattern-specific diagnostics for the shared feasible-cone implementation.
use crate::builtins::math::optim::common::{feasible_cone, ScalarCooperation};
use crate::BuiltinResult;
pub(super) fn policy() -> feasible_cone::Policy {
    feasible_cone::Policy {
        identity: runmat_builtins::BuiltinCatalogIdentity { name: super::NAME },
        invalid: &runmat_builtins::PATTERNSEARCH_ERROR_INVALID_ARGUMENT,
    }
}
pub(super) struct Cone(feasible_cone::Cone);
impl Cone {
    pub async fn full(
        dimension: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        feasible_cone::Cone::full(policy(), dimension, cooperation)
            .await
            .map(Self)
    }
    #[cfg(test)]
    pub async fn intersect(
        &mut self,
        row: &[(usize, f64)],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<()> {
        let normal = super::normal::Normal::new(
            row,
            &super::scaling::Scaling::unit(self.0.dimension()),
            cooperation,
        )
        .await?;
        self.0.intersect_normal(&normal, false, cooperation).await
    }
}
impl std::ops::Deref for Cone {
    type Target = feasible_cone::Cone;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for Cone {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
