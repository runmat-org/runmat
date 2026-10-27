//! Surrogate diagnostic adapter for the shared cubic radial-basis owner.
#[cfg(test)]
pub(super) use crate::builtins::math::optim::common::cubic_rbf::Data;
pub(super) use crate::builtins::math::optim::common::cubic_rbf::{
    Fit, Generation, Model, Prediction, Training, TrainingOwner,
};
use crate::builtins::math::optim::common::{cubic_rbf, ScalarCooperation};
use crate::BuiltinResult;

pub(super) async fn fit(
    data: &impl Training,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Fit> {
    cubic_rbf::fit(
        cubic_rbf::Policy {
            identity: super::IDENTITY,
            invalid: &super::INVALID_ARGUMENT,
        },
        data,
        cooperation,
    )
    .await
}
