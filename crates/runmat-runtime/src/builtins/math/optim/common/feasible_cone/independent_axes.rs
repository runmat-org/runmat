//! Independent affine-tail columns of an exact equality-nullspace cone.
use super::{Basis, Cone};
use crate::{builtins::math::optim::common::ScalarCooperation, BuiltinResult};
pub(in crate::builtins::math::optim) enum IndependentAxes {
    Unique,
    Axes(Vec<usize>),
    Unrepresentable,
}
impl Cone {
    pub(in crate::builtins::math::optim) async fn independent_axes(
        &self,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<IndependentAxes> {
        if !self.rays.is_empty() {
            return Err(self
                .policy
                .invalid("independent axes require an equality nullspace"));
        }
        let count = self.len(Basis::Maximal)?;
        if count == 0 {
            return Ok(IndependentAxes::Unique);
        }
        if count % 2 != 0 {
            return Err(self
                .policy
                .invalid("independent axes require an equality nullspace"));
        }
        let rank = count / 2;
        let mut occurrences = self.policy.buffer(self.dimension)?;
        occurrences.resize(self.dimension, 0usize);
        // Exact elimination starts with coordinate vectors and only removes a
        // pivot basis vector at each equality cut. Each surviving vector retains
        // a coordinate absent from all other surviving vectors. Counting supports
        // therefore identifies an independent coordinate projection of this space.
        // The shared conversion rejects any nonzero component that rounds to zero.
        for index in 0..rank {
            let Some(vector) = self
                .oriented_direction(index, Basis::Maximal, false, cooperation)
                .await?
            else {
                return Ok(IndependentAxes::Unrepresentable);
            };
            for (axis, value) in vector.into_iter().enumerate() {
                if value != 0.0 {
                    occurrences[axis] = occurrences[axis]
                        .checked_add(1)
                        .ok_or_else(|| self.policy.invalid("nullspace support count overflow"))?;
                }
                cooperation.completed(1).await?;
            }
        }
        let mut axes = self.policy.buffer(rank)?;
        for index in 0..rank {
            let Some(vector) = self
                .oriented_direction(index, Basis::Maximal, false, cooperation)
                .await?
            else {
                return Ok(IndependentAxes::Unrepresentable);
            };
            let mut selected = None;
            for (axis, value) in vector.into_iter().enumerate() {
                if value != 0.0 && occurrences[axis] == 1 {
                    selected = Some(axis);
                }
                cooperation.completed(1).await?;
                if selected.is_some() {
                    break;
                }
            }
            let Some(axis) = selected else {
                return Ok(IndependentAxes::Unrepresentable);
            };
            occurrences[axis] = usize::MAX;
        }
        // Public tail metadata is canonical and checked once, not resorted per probe.
        for (axis, count) in occurrences.into_iter().enumerate() {
            if count == usize::MAX {
                axes.push(axis);
            }
            cooperation.completed(1).await?;
        }
        Ok(IndependentAxes::Axes(axes))
    }
}
