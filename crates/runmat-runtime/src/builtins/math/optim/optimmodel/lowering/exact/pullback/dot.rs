//! Source-row exact dot slopes; original coordinates are never broadcast.
use super::*;
pub(super) async fn propagate(
    state: &mut State<'_>,
    constants: &Evaluated,
    left: GcHandle,
    right: GcHandle,
    arguments: &[Value],
    shape: &[usize],
    weights: Weights,
    c: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let policy = state.policy;
    let runtime = c.runtime().cloned();
    let left_shape = state.shape(left)?;
    let right_shape = state.shape(right)?;
    let plan = crate::context::RuntimeScopeFactory::new(runtime, || {
        crate::builtins::math::linalg::ops::dot::dot_geometry(
            left_shape,
            right_shape,
            crate::call::FunctionArgumentView::from_slice(arguments),
        )
    })
    .construct()
    .await?;
    if !crate::builtins::common::shape::array_shapes_equal(plan.output_shape(), shape) {
        return Err(policy.invalid("reverse dot geometry differs from original expression"));
    }
    for (row, weight) in weights {
        for axis in 0..plan.factor() {
            let source = plan
                .source(row, axis)
                .ok_or_else(|| policy.invalid("reverse dot exceeds geometry"))?;
            for (active, constant) in [(left, right), (right, left)] {
                if state.active_at(active, source)? {
                    let factor = constants
                        .value(constant, source, policy)?
                        .ok_or_else(|| policy.invalid("dot constant row is unassigned"))?;
                    state
                        .add(active, source, weight.mul(&factor, policy, c).await?, c)
                        .await?;
                }
                c.completed(1).await?;
            }
        }
    }
    Ok(())
}
