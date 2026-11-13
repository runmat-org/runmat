//! Frozen selected lattice model; admission/controller ownership stays shared.
use crate::builtins::math::optim::common::ScalarCooperation;
use crate::BuiltinResult;
pub(in crate::builtins::math::optim::patternsearch) enum FrameModel {
    Triangular(super::generator::Generator),
    Orthogonal(super::orthogonal::Householder),
}
impl FrameModel {
    pub fn count(&self) -> BuiltinResult<usize> {
        match self {
            Self::Triangular(model) => model.count(),
            Self::Orthogonal(model) => model.count(),
        }
    }
    pub fn bound(&self) -> Option<f64> {
        match self {
            Self::Triangular(model) => model.bound(),
            Self::Orthogonal(model) => model.bound(),
        }
    }
    pub async fn direction(
        &self,
        index: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Option<Vec<f64>>> {
        match self {
            Self::Triangular(model) => model.direction(index, cooperation).await,
            Self::Orthogonal(model) => model.direction(index, cooperation).await,
        }
    }
}
