//! Exact original dot pairing uses the host owner's same geometry.
use super::*;
pub(in crate::builtins::math::optim::optimmodel::lowering::exact) async fn evaluate(
    left: Data,
    right: Data,
    arguments: &[Value],
    shape: SharedShape,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Data> {
    let runtime = c.runtime().cloned();
    let plan = crate::context::RuntimeScopeFactory::new(runtime, || {
        crate::builtins::math::linalg::ops::dot::dot_geometry(
            &left.shape,
            &right.shape,
            crate::call::FunctionArgumentView::from_slice(arguments),
        )
    })
    .construct()
    .await?;
    if !crate::builtins::common::shape::array_shapes_equal(plan.output_shape(), &shape) {
        return Err(policy.invalid("exact dot geometry differs from original expression"));
    }
    let mut values = data::buffer(plan.outputs(), policy)?;
    for row in 0..plan.outputs() {
        let mut sum = Some(Rational::zero());
        for axis in 0..plan.factor() {
            let source = plan
                .source(row, axis)
                .ok_or_else(|| policy.invalid("exact dot source exceeds geometry"))?;
            sum = match (sum, left.at(source, policy)?, right.at(source, policy)?) {
                (Some(sum), Some(a), Some(b)) => {
                    Some(sum.add(&a.mul(&b, policy, c).await?, policy, c).await?)
                }
                _ => None,
            };
            c.completed(1).await?;
        }
        values.push(sum);
        c.completed(1).await?;
    }
    Data::explicit(shape, values, policy)
}
