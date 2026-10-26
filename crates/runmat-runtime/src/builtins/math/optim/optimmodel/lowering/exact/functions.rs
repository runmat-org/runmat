//! Exact original linear rows use the same admitted geometry as proposals.
use super::super::{linear::Linear, operations::Operation};
use super::*;
mod difference;
mod dot;
mod products;
pub(super) use difference::evaluate as difference;
pub(super) use dot::evaluate as dot;
pub(super) use products::evaluate as products;
pub(super) async fn linear(
    input: Data,
    operation: Operation,
    arguments: &[Value],
    shape: SharedShape,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<Data> {
    let Some(mut plan) = Linear::prepare(operation, &input.shape, arguments, policy, c).await?
    else {
        return Ok(Data::uniform(shape, None));
    };
    plan.validate(&input.shape, &shape, policy)?;
    let count = input.count(policy)?;
    let outputs = evaluation::element_count(&shape, policy)?;
    let denominators = denominators(&mut plan, count, outputs, policy, c).await?;
    for value in &denominators {
        if value.is_zero() {
            return Ok(Data::uniform(shape, None));
        }
        c.completed(1).await?;
    }
    if let Some(value) = input.uniform_value() {
        let Some(value) = value else {
            return Ok(Data::uniform(shape, None));
        };
        let factor = match &plan {
            Linear::Sum(plan) => Some(plan.factor()),
            Linear::Mean(_) => Some(1),
            Linear::Trace(plan) => Some(plan.factor()),
            Linear::Diagonal(plan) if plan.copies_uniform() => Some(1),
            _ => None,
        };
        if let Some(factor) = factor {
            return Ok(Data::uniform(
                shape,
                Some(value.mul(&Rational::from_usize(factor), policy, c).await?),
            ));
        }
    }
    let mut values = HashMap::new();
    if let Linear::Trace(trace) = &plan {
        let mut sum = Rational::zero();
        for row in 0..trace.factor() {
            let source = trace
                .source(row)
                .ok_or_else(|| policy.invalid("exact trace source exceeds admitted diagonal"))?;
            let Some(value) = input.at(source, policy)? else {
                return Ok(Data::uniform(shape, None));
            };
            sum = sum.add(&value, policy, c).await?;
            c.completed(1).await?;
        }
        return Ok(Data::uniform(shape, Some(sum)));
    }
    if let Linear::Scan(scan) = &plan {
        let mut running = Rational::zero();
        for step in scan.steps(false) {
            if step.first {
                running = Rational::zero();
            }
            let Some(value) = input.at(step.index, policy)? else {
                return Ok(Data::uniform(shape, None));
            };
            running = running.add(&value, policy, c).await?;
            if !running.is_zero() {
                values.try_reserve(1).map_err(|e| policy.reserve(e))?;
                values.insert(step.index, running.clone());
            }
            c.completed(1).await?;
        }
    } else {
        for source in 0..count {
            let Some(value) = input.at(source, policy)? else {
                return Ok(Data::uniform(shape, None));
            };
            if !value.is_zero() {
                let raw = Rational::from_float(plan.weight(source))
                    .ok_or_else(|| policy.invalid("exact linear weight must be finite"))?;
                for row in plan.outputs(source, policy)? {
                    let mut value = value.mul(&raw, policy, c).await?;
                    if let Some(denominator) = denominators.get(row) {
                        if denominator.is_zero() {
                            return Ok(Data::uniform(shape, None));
                        }
                        value = value.div(denominator, policy, c).await?;
                    }
                    accumulate(&mut values, row, value, policy, c).await?;
                }
            }
            c.completed(1).await?;
        }
    }
    Ok(Data::routed(shape, values))
}
async fn accumulate(
    values: &mut HashMap<usize, Rational>,
    row: usize,
    value: Rational,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    if value.is_zero() {
        return Ok(());
    }
    values.try_reserve(1).map_err(|e| policy.reserve(e))?;
    match values.entry(row) {
        std::collections::hash_map::Entry::Vacant(entry) => {
            entry.insert(value);
        }
        std::collections::hash_map::Entry::Occupied(mut entry) => {
            let value = entry.get().add(&value, policy, c).await?;
            if value.is_zero() {
                entry.remove();
            } else {
                entry.insert(value);
            }
        }
    }
    Ok(())
}
pub(super) use super::super::linear::exact_denominators as denominators;
