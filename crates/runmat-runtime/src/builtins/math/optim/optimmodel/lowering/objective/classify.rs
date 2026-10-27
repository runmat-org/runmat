//! Reverse original-topology propagation shares scalar weights at DAG joins.
//! Repeated references do not expand into exponentially many residual copies.
use super::*;
use std::collections::HashMap;
use step::Step;
async fn send(
    pending: &mut HashMap<GcHandle, Weight>,
    root: GcHandle,
    weight: Weight,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    pending.try_reserve(1).map_err(|e| policy.reserve(e))?;
    let value = match pending.remove(&root) {
        Some(previous) => previous.sum(&weight, policy, c).await?,
        None => weight,
    };
    pending.insert(root, value);
    c.completed(1).await
}
pub(in crate::builtins::math::optim::optimmodel::lowering) async fn classify(
    source: &evaluation::Plan,
    root: GcHandle,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Classification> {
    let mut pending = HashMap::new();
    send(&mut pending, root, Weight::one(), policy, c).await?;
    let mut terms = HashMap::<GcHandle, Weight>::new();
    let mut constant = Rational::zero();
    for &handle in source.order.iter().rev() {
        let Some(weight) = pending.remove(&handle) else {
            c.completed(1).await?;
            continue;
        };
        if !weight.proposal.is_finite() {
            return Ok(Classification::Unrepresentable);
        }
        match step::select(source, handle, policy, c).await? {
            Step::Constant => {
                let node = source.nodes[&handle].node();
                let Kind::Constant(value) = &node.kind else {
                    return Err(policy.invalid("constant objective step lost its literal"));
                };
                let offset = match array::Offset::from_value(value, policy)? {
                    Ok(value) => value,
                    Err(_) => return Ok(Classification::Unrepresentable),
                };
                for row in 0..evaluation::element_count(&node.shape, policy)? {
                    let value = Rational::from_float(offset.at(row, policy)?)
                        .ok_or_else(|| policy.invalid("objective constant must be finite"))?;
                    constant = constant
                        .add(&weight.exact.mul(&value, policy, c).await?, policy, c)
                        .await?;
                    c.completed(1).await?;
                }
            }
            Step::Add(left, right) => {
                send(&mut pending, left, weight.clone(), policy, c).await?;
                send(&mut pending, right, weight, policy, c).await?;
            }
            Step::Subtract(left, right) => {
                send(&mut pending, left, weight.clone(), policy, c).await?;
                let negative = weight.scale(-1., Scale::Multiply, policy, c).await?;
                send(&mut pending, right, negative, policy, c).await?;
            }
            Step::Forward(next) => send(&mut pending, next, weight, policy, c).await?,
            Step::Scale {
                next,
                factor,
                operation,
            } => {
                let scaled = weight.scale(factor, operation, policy, c).await?;
                send(&mut pending, next, scaled, policy, c).await?;
            }
            Step::Sum { next } => {
                let node = source
                    .nodes
                    .get(&handle)
                    .expect("selected original sum")
                    .node();
                let input = source
                    .nodes
                    .get(&next)
                    .ok_or_else(|| policy.invalid("sum lost original input"))?
                    .node();
                let Kind::Builtin { arguments, .. } = &node.kind else {
                    unreachable!("typed sum selection");
                };
                let geometry = crate::builtins::math::reduction::sum::reduction_geometry(
                    &input.shape,
                    crate::call::FunctionArgumentView::from_slice(arguments),
                    c.runtime(),
                )
                .await?;
                if !crate::builtins::common::shape::array_shapes_equal(
                    geometry.output_shape(),
                    &node.shape,
                ) {
                    return Err(policy.invalid("least-squares sum differs from original geometry"));
                }
                send(&mut pending, next, weight, policy, c).await?;
            }
            Step::Square { residual } => send(&mut terms, residual, weight, policy, c).await?,
            Step::Other => return Ok(Classification::Unrecognized),
        }
        c.completed(1).await?;
    }
    if !pending.is_empty() {
        return Err(policy.invalid("objective ancestry is outside its original topology"));
    }
    if terms.is_empty() {
        return Ok(Classification::Unrecognized);
    }
    // Preserve original topological order rather than HashMap iteration order.
    let mut selected =
        runmat_value::exact_capacity_vec(terms.len()).map_err(|e| policy.storage(e))?;
    for &handle in &source.order {
        if let Some(weight) = terms.remove(&handle) {
            if !weight.proposal.is_finite() {
                return Ok(Classification::Unrepresentable);
            }
            selected.push(Term {
                residual: handle,
                weight,
            });
        }
        c.completed(1).await?;
    }
    Ok(Classification::LeastSquares(LeastSquares {
        constant,
        terms: selected,
    }))
}
