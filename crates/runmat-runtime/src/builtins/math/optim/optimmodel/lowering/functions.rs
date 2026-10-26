//! Affine rows through one original geometry. No numerical basis probing.
use super::*;
mod products;
mod difference;
mod dot;
use super::{
    array::{Array, Assigned, Degree, Offset},
    linear::Linear,
    operations::Operation,
    terms::{Coefficients, Term},
};
pub(super) async fn linear(
    input: Array,
    operation: Operation,
    arguments: &[Value],
    shape: SharedShape,
    columns: usize,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> Lowered<Array> {
    let count = input.count(policy)?;
    let outputs = evaluation::element_count(&shape, policy)?;
    if !matches!(input.assigned, Assigned::All) {
        for row in 0..count {
            if !input.assigned.at(row, policy)? {
                return Ok(Err(Stop::Unassigned));
            }
            c.completed(1).await?;
        }
    }
    let Some(mut plan) = Linear::prepare(operation, &input.shape, arguments, policy, c).await?
    else {
        return Ok(Err(Stop::UnsupportedFunction));
    };
    plan.validate(&input.shape, &shape, policy)?;
    let normalization = match plan.normalization(count, outputs, policy, c).await? {
        Ok(value) => value,
        Err(stop) => return Ok(Err(stop)),
    };
    let offset = match offset(&input, &mut plan, &normalization, shape.clone(), policy, c).await? {
        Ok(value) => value,
        Err(stop) => return Ok(Err(stop)),
    };
    let coefficients = if input.coefficients.is_zero() {
        Coefficients::Zero
    } else {
        let mut entries = Vec::new();
        for term in input.coefficients.entries() {
            let raw = plan.weight(term.row);
            for row in plan.outputs(term.row, policy)? {
                let value = term.value * normalization.weight(raw, row);
                if value != 0. {
                    entries.try_reserve(1).map_err(|e| policy.reserve(e))?;
                    entries.push(Term {
                        column: term.column,
                        row,
                        value,
                    });
                }
                c.completed(1).await?;
            }
        }
        match Coefficients::from_entries(entries, outputs, columns, policy, c).await? {
            Ok(value) => value,
            Err(stop) => return Ok(Err(stop)),
        }
    };
    let mut dependent = std::collections::HashSet::new();
    for source in input.degree.rows(count) {
        if input.degree.at(source) {
            for row in plan.outputs(source, policy)? {
                dependent.try_reserve(1).map_err(|e| policy.reserve(e))?;
                dependent.insert(row);
                c.completed(1).await?;
            }
        }
        c.completed(1).await?;
    }
    let degree = Degree::collect(outputs, |row| Ok(dependent.contains(&row)), policy, c).await?;
    Ok(Ok(Array {
        shape,
        offset,
        coefficients,
        assigned: Assigned::All,
        degree,
    }))
}
async fn offset(
    input: &Array,
    plan: &mut Linear,
    normalization: &linear::Normalization,
    shape: SharedShape,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> Lowered<Offset> {
    if let Offset::Uniform(value) = &input.offset {
        if *value == 0. {
            return Ok(Ok(Offset::Uniform(0.)));
        }
        if let Some(value) = plan.uniform_result(*value) {
            return if value.is_finite() {
                Ok(Ok(Offset::Uniform(value)))
            } else {
                Ok(Err(Stop::NonFinite))
            };
        }
    }
    let mut entries = Vec::new();
    if let Linear::Trace(trace) = plan {
        let mut value = 0.;
        for row in 0..trace.factor() {
            value += input.offset.at(
                trace
                    .source(row)
                    .ok_or_else(|| policy.invalid("trace source exceeds admitted diagonal"))?,
                policy,
            )?;
            if !value.is_finite() {
                return Ok(Err(Stop::NonFinite));
            }
            c.completed(1).await?;
        }
        return Ok(Ok(Offset::Uniform(value)));
    }
    if let Linear::Scan(scan) = plan {
        let mut running = 0.;
        for step in scan.steps(false) {
            if step.first {
                running = 0.;
            }
            running += input.offset.at(step.index, policy)?;
            if !running.is_finite() {
                return Ok(Err(Stop::NonFinite));
            }
            if running != 0. {
                entries.try_reserve(1).map_err(|e| policy.reserve(e))?;
                entries.push(Term {
                    column: 0,
                    row: step.index,
                    value: running,
                });
            }
            c.completed(1).await?;
        }
    } else {
        for source in 0..input.count(policy)? {
            let value = input.offset.at(source, policy)?;
            if value != 0. {
                let raw = plan.weight(source);
                for row in plan.outputs(source, policy)? {
                    let value = value * normalization.weight(raw, row);
                    if value != 0. {
                        entries.try_reserve(1).map_err(|e| policy.reserve(e))?;
                        entries.push(Term {
                            column: 0,
                            row,
                            value,
                        });
                    }
                    c.completed(1).await?;
                }
            }
            c.completed(1).await?;
        }
    }
    let count = evaluation::element_count(&shape, policy)?;
    let reduced = match Coefficients::from_entries(entries, count, 1, policy, c).await? {
        Ok(value) => value,
        Err(stop) => return Ok(Err(stop)),
    };
    Ok(Ok(reduced.into_offset(shape, policy, c).await?))
}

pub(super) async fn lower(
    operation: Operation,
    inputs: Vec<Array>,
    arguments: &[Value],
    shape: SharedShape,
    columns: usize,
    policy: Policy,
    c: &mut ScalarCooperation,
) -> Lowered<Array> {
    if operation == Operation::Dot {
        return dot::lower(inputs, arguments, shape, columns, policy, c).await;
    }
    let expected = 1;
    if inputs.len() != expected {
        return Err(policy.invalid("canonical affine function has invalid source arity"));
    }
    let mut inputs = inputs.into_iter();
    let input = inputs.next().expect("validated function arity");
    match operation {
        Operation::Dot => unreachable!("dot arity handled before unary functions"),
        Operation::Difference => {
            difference::lower(input, arguments, shape, columns, policy, c).await
        }
        Operation::Product | Operation::Cumprod => {
            products::lower(input, operation, arguments, shape, columns, policy, c).await
        }
        Operation::Sum => input.sum(arguments, shape, columns, policy, c).await,
        Operation::Reshape => input.reshaped(arguments, shape, policy, c).await,
        Operation::Mean | Operation::Trace | Operation::Diagonal | Operation::Cumsum => {
            linear(input, operation, arguments, shape, columns, policy, c).await
        }
    }
}
