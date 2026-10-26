//! Exact adjacent subtraction through the numeric owner's stage plan.
use super::*;
pub(in crate::builtins::math::optim::optimmodel::lowering::exact) async fn evaluate(
    mut input: Data,
    arguments: &[Value],
    shape: SharedShape,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Data> {
    let runtime = c.runtime().cloned();
    let plan = crate::context::with_original_scope(runtime.as_ref(), || {
        crate::builtins::math::reduction::diff::difference_geometry(
            &input.shape,
            crate::call::FunctionArgumentView::from_slice(arguments),
        )
    })?;
    if !crate::builtins::common::shape::array_shapes_equal(&plan.output_shape(), &shape) {
        return Err(policy.invalid("exact difference geometry differs from original expression"));
    }
    for step in 0..plan.stages() {
        let output = SharedShape::new_unmanaged(plan.shape_after(step + 1));
        let count = evaluation::element_count(&output, policy)?;
        let mut values = data::buffer(count, policy)?;
        for row in 0..count {
            let (a, b) = plan
                .pair(step, row)
                .ok_or_else(|| policy.invalid("exact difference exceeds stage"))?;
            values.push(match (input.at(a, policy)?, input.at(b, policy)?) {
                (Some(a), Some(b)) => Some(b.sub(&a, policy, c).await?),
                _ => None,
            });
            c.completed(1).await?;
        }
        input = Data::explicit(output, values, policy)?;
        if count == 0 {
            break;
        }
    }
    input.reshaped(shape, policy)
}
