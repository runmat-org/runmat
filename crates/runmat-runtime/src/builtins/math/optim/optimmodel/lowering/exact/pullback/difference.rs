//! Sparse exact transpose of the same repeated adjacent-difference stages.
use super::*;
pub(super) async fn propagate(
    state: &mut State<'_>,
    source: GcHandle,
    arguments: &[Value],
    shape: &[usize],
    mut weights: Weights,
    c: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let policy = state.policy;
    let runtime = c.runtime().cloned();
    let plan = crate::context::with_original_scope(runtime.as_ref(), || {
        crate::builtins::math::reduction::diff::difference_geometry(
            state.shape(source)?,
            crate::call::FunctionArgumentView::from_slice(arguments),
        )
    })?;
    if !crate::builtins::common::shape::array_shapes_equal(&plan.output_shape(), shape) {
        return Err(policy.invalid("reverse difference geometry differs from original expression"));
    }
    for step in (0..plan.stages()).rev() {
        let mut next = Weights::new();
        for (row, value) in weights {
            let (a, b) = plan
                .pair(step, row)
                .ok_or_else(|| policy.invalid("reverse difference exceeds stage"))?;
            for (row, value) in [(a, -&value), (b, value)] {
                next.try_reserve(1).map_err(|e| policy.reserve(e))?;
                let previous = next.remove(&row).unwrap_or_else(Rational::zero);
                let sum = previous.add(&value, policy, c).await?;
                if !sum.is_zero() {
                    next.insert(row, sum);
                }
                c.completed(1).await?;
            }
        }
        weights = next;
    }
    for (row, value) in weights {
        state.add(source, row, value, c).await?;
        c.completed(1).await?;
    }
    Ok(())
}
