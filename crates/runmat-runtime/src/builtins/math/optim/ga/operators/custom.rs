//! Public custom operators are ordinary rooted semantic invocations. Genetic
//! parent indices are 1-based in the supplied population, never new identities.
use super::super::{invalid, options::Operator, population::Generation, ScalarCooperation, POLICY};
use crate::BuiltinResult;
use runmat_builtins::{OptimGeneticCrossover, OptimGeneticMutation, OptimGeneticPopulation};
use runmat_value::{Tensor, Value};
use std::ops::ControlFlow;

async fn indices(
    parents: &[usize],
    members: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Value> {
    let mut values = POLICY.capacity(parents.len(), cooperation.runtime())?;
    for &parent in parents {
        if parent >= members {
            return Err(invalid("custom operator parent is outside population"));
        }
        let one_based = parent
            .checked_add(1)
            .and_then(|index| u64::try_from(index).ok())
            .and_then(|index| runmat_value::IntValue::U64(index).to_exact_f64())
            .ok_or_else(|| invalid("custom operator parent index cannot be represented exactly"))?;
        values.push(one_based);
        cooperation.completed(1).await?;
    }
    Ok(Value::Tensor(
        Tensor::new(values, vec![1, parents.len()]).map_err(invalid)?,
    ))
}
fn genome_length(variables: usize) -> BuiltinResult<Value> {
    u64::try_from(variables)
        .ok()
        .and_then(|n| runmat_value::IntValue::U64(n).to_exact_f64())
        .map(Value::Num)
        .ok_or_else(|| invalid("custom operator genome length cannot be represented exactly"))
}

#[derive(Clone, Copy)]
pub(in super::super) struct Current<'a> {
    pub controls: &'a Value,
    pub problem: &'a Value,
    pub scores: &'a Value,
    pub population: &'a Generation,
    pub kind: OptimGeneticPopulation,
}

pub(in super::super) async fn crossover<I: super::super::callbacks::Invocation>(
    operator: &Operator<OptimGeneticCrossover>,
    parents: &[usize],
    current: &Current<'_>,
    invoker: &mut I,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<ControlFlow<I::Stop, Generation>> {
    use super::super::common::population::projection::Arguments;
    let Current {
        controls,
        problem,
        scores,
        population,
        kind,
    } = *current;
    if parents.is_empty() || !parents.len().is_multiple_of(2) {
        return Err(invalid(
            "custom crossover requires two parent indices per child",
        ));
    }
    let returned = operator
        .call_with(
            [
                indices(parents, population.members(), cooperation).await?,
                controls.clone(),
                genome_length(population.variables())?,
                problem.clone(),
                scores.clone(),
                population.value(),
            ],
            1,
            invoker,
            cooperation,
        )
        .await?;
    let returned = match returned {
        ControlFlow::Continue(returned) => returned,
        ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
    };
    let returned = {
        let _scope = cooperation
            .runtime()
            .map(crate::context::RuntimeContext::enter);
        returned.try_into_single()?
    };
    Generation::from_result_with(
        returned,
        parents.len() / 2,
        population.variables(),
        kind,
        invoker,
        cooperation,
    )
    .await
}

pub(in super::super) async fn mutation<I: super::super::callbacks::Invocation>(
    operator: &Operator<OptimGeneticMutation>,
    parents: &[usize],
    current: &Current<'_>,
    state: &Value,
    invoker: &mut I,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<ControlFlow<I::Stop, Generation>> {
    use super::super::common::population::projection::Arguments;
    let Current {
        controls,
        problem,
        scores,
        population,
        kind,
    } = *current;
    if parents.is_empty() {
        return Err(invalid("custom mutation requires parent indices"));
    }
    let returned = operator
        .call_with(
            [
                indices(parents, population.members(), cooperation).await?,
                controls.clone(),
                genome_length(population.variables())?,
                problem.clone(),
                state.clone(),
                scores.clone(),
                population.value(),
            ],
            1,
            invoker,
            cooperation,
        )
        .await?;
    let returned = match returned {
        ControlFlow::Continue(returned) => returned,
        ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
    };
    let returned = {
        let _scope = cooperation
            .runtime()
            .map(crate::context::RuntimeContext::enter);
        returned.try_into_single()?
    };
    Generation::from_result_with(
        returned,
        parents.len(),
        population.variables(),
        kind,
        invoker,
        cooperation,
    )
    .await
}
