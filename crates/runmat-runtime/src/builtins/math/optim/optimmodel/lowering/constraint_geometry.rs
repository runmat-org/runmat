//! One validated broadcast mapping per original comparison array.
use super::*;
use crate::builtins::common::broadcast::{broadcast_index, compute_strides, BroadcastPlan};
pub(super) struct Geometry {
    pub plan: BroadcastPlan,
    left: SharedShape,
    right: SharedShape,
    left_strides: Vec<usize>,
    right_strides: Vec<usize>,
}
impl Geometry {
    pub(super) fn from_source(
        source: &evaluation::Plan,
        constraint: &formulation::Constraint,
        policy: Policy,
    ) -> BuiltinResult<Self> {
        Self::from_roots(
            source,
            constraint.left,
            constraint.right,
            &constraint.shape,
            policy,
        )
    }
    pub(super) fn from_roots(
        source: &evaluation::Plan,
        left: runmat_gc::GcHandle,
        right: runmat_gc::GcHandle,
        shape: &SharedShape,
        policy: Policy,
    ) -> BuiltinResult<Self> {
        let left = source
            .nodes
            .get(&left)
            .ok_or_else(|| policy.invalid("original left comparison root is unavailable"))?
            .node()
            .shape
            .clone();
        let right = source
            .nodes
            .get(&right)
            .ok_or_else(|| policy.invalid("original right comparison root is unavailable"))?
            .node()
            .shape
            .clone();
        let plan = BroadcastPlan::new(&left, &right).map_err(|e| policy.invalid(e))?;
        if !crate::builtins::common::shape::array_shapes_equal(plan.output_shape(), shape) {
            return Err(policy.invalid("original comparison broadcast geometry changed"));
        }
        let left_strides = compute_strides(&left);
        let right_strides = compute_strides(&right);
        Ok(Self {
            plan,
            left,
            right,
            left_strides,
            right_strides,
        })
    }
    pub(super) fn indices(&self, row: usize) -> Option<(usize, usize)> {
        if row >= self.plan.len() {
            return None;
        }
        Some((
            broadcast_index(
                row,
                self.plan.output_shape(),
                &self.left,
                &self.left_strides,
            ),
            broadcast_index(
                row,
                self.plan.output_shape(),
                &self.right,
                &self.right_strides,
            ),
        ))
    }
}
