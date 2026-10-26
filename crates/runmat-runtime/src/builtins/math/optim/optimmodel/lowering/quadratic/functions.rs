//! Existing admitted function geometry; sparse polynomial values are the only
//! changed scalar algebra. No source callback or numerical model probing.
use super::super::{
    linear::{exact_denominators, Linear},
    operations::Operation,
};
use super::*;
pub(super) async fn lower(
    mut inputs: Vec<Array>,
    operation: Operation,
    arguments: &[Value],
    shape: SharedShape,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> Lowered<Array> {
    if operation == Operation::Dot {
        return dot(inputs, arguments, shape, policy, c).await;
    }
    if inputs.len() != 1 {
        return Err(policy.invalid("quadratic function requires one original operand"));
    }
    let input = inputs.pop().expect("checked original function arity");
    match operation {
        Operation::Reshape => Ok(Ok(input.reshaped(shape, policy)?)),
        Operation::Sum
        | Operation::Mean
        | Operation::Trace
        | Operation::Cumsum
        | Operation::Diagonal => linear(input, operation, arguments, shape, policy, c).await,
        Operation::Difference => difference(input, arguments, shape, policy, c).await,
        Operation::Product | Operation::Cumprod => {
            product(input, operation, arguments, shape, policy, c).await
        }
        Operation::Dot => unreachable!("binary dot was selected before unary arity"),
    }
}
pub(super) async fn linear(
    input: Array,
    operation: Operation,
    arguments: &[Value],
    shape: SharedShape,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> Lowered<Array> {
    let Some(mut plan) = Linear::prepare(operation, &input.shape, arguments, policy, c).await?
    else {
        return Ok(Err(Stop::UnsupportedFunction));
    };
    plan.validate(&input.shape, &shape, policy)?;
    let inputs = input.count(policy)?;
    let outputs = evaluation::element_count(&shape, policy)?;
    let proposal = match plan.normalization(inputs, outputs, policy, c).await? {
        Ok(value) => value,
        Err(stop) => return Ok(Err(stop)),
    };
    let denominators = exact_denominators(&mut plan, inputs, outputs, policy, c).await?;
    for value in &denominators {
        if value.is_zero() {
            return Ok(Err(Stop::NonFinite));
        }
        c.completed(1).await?;
    }
    let mut rows = runmat_value::exact_capacity_vec(outputs).map_err(|e| policy.storage(e))?;
    for _ in 0..outputs {
        rows.push(Some(Polynomial::zero()));
        c.completed(1).await?;
    }
    if let Linear::Scan(scan) = &plan {
        let mut running = Some(Polynomial::zero());
        for step in scan.steps(false) {
            if step.first {
                running = Some(Polynomial::zero());
            }
            running = match (running, input.at(step.index, policy)?) {
                (Some(a), Some(b)) => Some(a.add(b, false, policy, c).await?),
                _ => None,
            };
            rows[step.index] = running.clone();
            c.completed(1).await?;
        }
    } else {
        for source in 0..inputs {
            for target in plan.outputs(source, policy)? {
                let raw = plan.weight(source);
                let mut weight = Weight::constant(raw, policy)?;
                if let Some(denominator) = denominators.get(target) {
                    weight.exact = weight.exact.div(denominator, policy, c).await?;
                }
                weight.proposal = proposal.weight(raw, target);
                rows[target] = match (rows[target].take(), input.at(source, policy)?) {
                    (Some(a), Some(b)) => Some(
                        a.add(b.scaled(&weight, policy, c).await?, false, policy, c)
                            .await?,
                    ),
                    _ => None,
                };
                c.completed(1).await?;
            }
            c.completed(1).await?;
        }
    }
    Ok(Ok(Array::rows(shape, rows, policy)?))
}
pub(super) async fn dot(
    inputs: Vec<Array>,
    arguments: &[Value],
    shape: SharedShape,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> Lowered<Array> {
    if inputs.len() != 2 {
        return Err(policy.invalid("quadratic dot requires two original operands"));
    }
    let mut inputs = inputs.into_iter();
    let left = inputs.next().expect("checked dot arity");
    let right = inputs.next().expect("checked dot arity");
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
    if !crate::builtins::common::shape::array_shapes_equal(plan.output_shape(), &shape)
        || left.count(policy)? != right.count(policy)?
    {
        return Err(policy.invalid("quadratic dot differs from original geometry"));
    }
    let mut rows =
        runmat_value::exact_capacity_vec(plan.outputs()).map_err(|e| policy.storage(e))?;
    for output in 0..plan.outputs() {
        let mut value = Some(Polynomial::zero());
        for axis in 0..plan.factor() {
            let source = plan
                .source(output, axis)
                .ok_or_else(|| policy.invalid("quadratic dot source exceeds geometry"))?;
            value = match (value, left.at(source, policy)?, right.at(source, policy)?) {
                (Some(sum), Some(a), Some(b)) => match a.product(&b, policy, c).await? {
                    Ok(product) => Some(sum.add(product, false, policy, c).await?),
                    Err(stop) => return Ok(Err(stop)),
                },
                _ => None,
            };
            c.completed(1).await?;
        }
        rows.push(value);
        c.completed(1).await?;
    }
    Ok(Ok(Array::rows(shape, rows, policy)?))
}
pub(super) async fn product(
    input: Array,
    operation: Operation,
    arguments: &[Value],
    shape: SharedShape,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> Lowered<Array> {
    let mut plan = super::super::factors::Factors::prepare(
        operation,
        &input.shape,
        arguments,
        &shape,
        policy,
        c,
    )
    .await?;
    let outputs = evaluation::element_count(&shape, policy)?;
    let mut rows = runmat_value::exact_capacity_vec(outputs).map_err(|e| policy.storage(e))?;
    for _ in 0..outputs {
        rows.push(Some(Polynomial::scalar(1., policy)?));
        c.completed(1).await?;
    }
    match &mut plan {
        super::super::factors::Factors::Reduction(reduction) => {
            for source in 0..input.count(policy)? {
                let target = reduction
                    .project(source)
                    .ok_or_else(|| policy.invalid("quadratic factor source exceeds geometry"))?;
                rows[target] = match (rows[target].take(), input.at(source, policy)?) {
                    (Some(a), Some(b)) => match a.product(&b, policy, c).await? {
                        Ok(value) => Some(value),
                        Err(stop) => return Ok(Err(stop)),
                    },
                    _ => None,
                };
                c.completed(1).await?;
            }
        }
        super::super::factors::Factors::Scan(scan) => {
            let mut running = Some(Polynomial::scalar(1., policy)?);
            for step in scan.steps(false) {
                if step.first {
                    running = Some(Polynomial::scalar(1., policy)?);
                }
                running = match (running, input.at(step.index, policy)?) {
                    (Some(a), Some(b)) => match a.product(&b, policy, c).await? {
                        Ok(value) => Some(value),
                        Err(stop) => return Ok(Err(stop)),
                    },
                    _ => None,
                };
                rows[step.index] = running.clone();
                c.completed(1).await?;
            }
        }
    }
    Ok(Ok(Array::rows(shape, rows, policy)?))
}
async fn difference(
    mut input: Array,
    arguments: &[Value],
    shape: SharedShape,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> Lowered<Array> {
    let runtime = c.runtime().cloned();
    let plan = crate::context::with_original_scope(runtime.as_ref(), || {
        crate::builtins::math::reduction::diff::difference_geometry(
            &input.shape,
            crate::call::FunctionArgumentView::from_slice(arguments),
        )
    })?;
    if !crate::builtins::common::shape::array_shapes_equal(&plan.output_shape(), &shape) {
        return Err(policy.invalid("quadratic difference differs from original geometry"));
    }
    for step in 0..plan.stages() {
        let output = SharedShape::new_unmanaged(plan.shape_after(step + 1));
        let count = evaluation::element_count(&output, policy)?;
        let mut rows = runmat_value::exact_capacity_vec(count).map_err(|e| policy.storage(e))?;
        for row in 0..count {
            let (a, b) = plan
                .pair(step, row)
                .ok_or_else(|| policy.invalid("quadratic difference source exceeds geometry"))?;
            rows.push(match (input.at(b, policy)?, input.at(a, policy)?) {
                (Some(a), Some(b)) => Some(a.add(b, true, policy, c).await?),
                _ => None,
            });
            c.completed(1).await?;
        }
        input = Array::rows(output, rows, policy)?;
        if count == 0 {
            break;
        }
    }
    Ok(Ok(input.reshaped(shape, policy)?))
}
