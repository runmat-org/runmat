//! Adjoint source rows, using the same typed geometry and exact mean weights.
use super::super::super::{linear::Linear, operations::Operation};
use super::*;
pub(super) async fn propagate(
    state: &mut State<'_>,
    handle: GcHandle,
    operation: Operation,
    source: GcHandle,
    arguments: &[Value],
    shape: &[usize],
    weights: Weights,
    c: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    if !state.active(source)? {
        return Ok(());
    }
    if state.plan.is_some() {
        return selected(
            state, handle, operation, source, arguments, shape, weights, c,
        )
        .await;
    }
    let policy = state.policy;
    let input = state
        .program
        .source
        .nodes
        .get(&source)
        .ok_or_else(|| policy.invalid("linear source node is unavailable"))?
        .node()
        .shape
        .clone();
    let Some(mut plan) = Linear::prepare(operation, &input, arguments, policy, c).await? else {
        return Err(policy.invalid("unadmitted reverse linear function"));
    };
    plan.validate(&input, shape, policy)?;
    let count = evaluation::element_count(&input, policy)?;
    let outputs = evaluation::element_count(shape, policy)?;
    let denominators =
        super::super::functions::denominators(&mut plan, count, outputs, policy, c).await?;
    if let Linear::Scan(scan) = &plan {
        let mut running = Rational::zero();
        for step in scan.steps(true) {
            if step.first {
                running = Rational::zero();
            }
            if let Some(value) = weights.get(&step.index) {
                running = running.add(value, policy, c).await?;
            }
            state.add(source, step.index, running.clone(), c).await?;
            c.completed(1).await?;
        }
        return Ok(());
    }
    if let Linear::Trace(trace) = &plan {
        if let Some(value) = weights.get(&0) {
            for row in 0..trace.factor() {
                let input = trace
                    .source(row)
                    .ok_or_else(|| policy.invalid("trace source exceeds original diagonal"))?;
                state.add(source, input, value.clone(), c).await?;
                c.completed(1).await?;
            }
        }
        return Ok(());
    }
    if let Linear::Diagonal(diagonal) = &plan {
        for (output, value) in weights {
            if let Some(input) = diagonal.source(output) {
                state.add(source, input, value, c).await?;
            }
            c.completed(1).await?;
        }
        return Ok(());
    }
    for input in 0..count {
        let raw = Rational::from_float(plan.weight(input))
            .ok_or_else(|| policy.invalid("reverse linear weight must be finite"))?;
        for output in plan.outputs(input, policy)? {
            if let Some(weight) = weights.get(&output) {
                let mut value = weight.mul(&raw, policy, c).await?;
                if let Some(denominator) = denominators.get(output) {
                    if denominator.is_zero() {
                        return Err(policy.invalid("reverse mean has zero exact weight"));
                    }
                    value = value.div(denominator, policy, c).await?;
                }
                state.add(source, input, value, c).await?;
            }
            c.completed(1).await?;
        }
    }
    Ok(())
}

async fn selected(
    state: &mut State<'_>,
    handle: GcHandle,
    operation: Operation,
    source: GcHandle,
    arguments: &[Value],
    shape: &[usize],
    weights: Weights,
    c: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    use std::rc::Rc;
    let policy = state.policy;
    let cached = state
        .plan
        .as_ref()
        .expect("selected reverse")
        .linear
        .get(&handle)
        .cloned();
    let cached = match cached {
        Some(plan) => plan,
        None => {
            let input = state.shape(source)?;
            let Some(mut plan) = Linear::prepare(operation, input, arguments, policy, c).await?
            else {
                return Err(policy.invalid("unadmitted reverse linear function"));
            };
            plan.validate(input, shape, policy)?;
            let denominators = super::super::functions::denominators(
                &mut plan,
                evaluation::element_count(input, policy)?,
                evaluation::element_count(shape, policy)?,
                policy,
                c,
            )
            .await?;
            let cached = Rc::new(super::sparse::LinearPlan { plan, denominators });
            let cache = &mut state.plan.as_mut().expect("selected reverse").linear;
            cache.try_reserve(1).map_err(|e| policy.reserve(e))?;
            cache.insert(handle, Rc::clone(&cached));
            cached
        }
    };
    for (output, value) in weights {
        let denominator = cached.denominators.get(output);
        match &cached.plan {
            Linear::Sum(reduction) => {
                for ordinal in 0..reduction.factor() {
                    let input = reduction
                        .source(output, ordinal)
                        .ok_or_else(|| policy.invalid("reverse sum exceeds original fiber"))?;
                    send(state, source, input, &value, 1., denominator, c).await?;
                }
            }
            Linear::Mean(mean) => {
                for ordinal in 0..mean.reduction.factor() {
                    let input = mean
                        .reduction
                        .source(output, ordinal)
                        .ok_or_else(|| policy.invalid("reverse mean exceeds original fiber"))?;
                    send(
                        state,
                        source,
                        input,
                        &value,
                        mean.weight(input),
                        denominator,
                        c,
                    )
                    .await?;
                }
            }
            Linear::Trace(trace) => {
                if output != 0 {
                    return Err(policy.invalid("reverse trace exceeds scalar output"));
                }
                for ordinal in 0..trace.factor() {
                    let input = trace
                        .source(ordinal)
                        .ok_or_else(|| policy.invalid("reverse trace exceeds original diagonal"))?;
                    send(state, source, input, &value, 1., None, c).await?;
                }
            }
            Linear::Diagonal(diagonal) => {
                if let Some(input) = diagonal.source(output) {
                    send(state, source, input, &value, 1., None, c).await?;
                }
            }
            Linear::Scan(scan) => {
                for input in scan
                    .inputs(output)
                    .ok_or_else(|| policy.invalid("reverse scan exceeds original lane"))?
                {
                    send(state, source, input, &value, 1., None, c).await?;
                }
            }
        }
        c.completed(1).await?;
    }
    Ok(())
}
async fn send(
    state: &mut State<'_>,
    source: GcHandle,
    input: usize,
    weight: &Rational,
    raw: f64,
    denominator: Option<&Rational>,
    c: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let policy = state.policy;
    let raw = Rational::from_float(raw)
        .ok_or_else(|| policy.invalid("reverse linear weight must be finite"))?;
    let mut value = weight.mul(&raw, policy, c).await?;
    if let Some(denominator) = denominator {
        if denominator.is_zero() {
            return Err(policy.invalid("reverse mean has zero exact weight"));
        }
        value = value.div(denominator, policy, c).await?;
    }
    state.add(source, input, value, c).await?;
    c.completed(1).await
}
