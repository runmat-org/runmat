//! A subpopulation's rank, expectations and sampled parents refer to the
//! same immutable generation. Only row indices cross the island boundary.
use super::super::callbacks::Invocation;
use super::super::{
    invalid,
    operators::{ranking, scaling, selected, selection, shift_linear},
    options, random, ScalarCooperation, POLICY,
};
use crate::BuiltinResult;
use runmat_builtins::{OptimGeneticScaling, OptimGeneticSelection};
use runmat_value::{Tensor, Value};
use std::ops::ControlFlow;

pub(super) struct Parents {
    pub order: ranking::RankOrder,
    pub expectation: Tensor,
    pub selected: Vec<usize>,
}
pub(super) struct Request<'a> {
    pub comparison: &'a [ranking::Score],
    pub scores: &'a Value,
    pub offset: usize,
    pub count: usize,
    pub options: &'a options::Options,
    pub controls: Option<&'a Value>,
}
pub(super) async fn choose<I: Invocation>(
    request: Request<'_>,
    invoker: &mut I,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<ControlFlow<I::Stop, Parents>> {
    let Request {
        comparison,
        scores,
        offset,
        count,
        options,
        controls,
    } = request;
    let members = comparison.len();
    let end = offset
        .checked_add(members)
        .ok_or_else(|| invalid("parent score range overflow"))?;
    let total = match scores {
        Value::Tensor(value) => value.len(),
        Value::ComplexTensor(value) => value.len(),
        _ => return Err(invalid("parent scores must be an admitted numeric vector")),
    };
    if comparison.is_empty() || end > total {
        return Err(invalid("parent scores and comparison rows differ"));
    }
    let order = ranking::order(comparison, cooperation).await?;
    if count == 0 {
        let mut weights = POLICY.capacity(members, cooperation.runtime())?;
        for _ in comparison {
            weights.push(0.);
            cooperation.completed(1).await?;
        }
        return Ok(ControlFlow::Continue(Parents {
            order,
            expectation: Tensor::new(weights, vec![members, 1]).map_err(invalid)?,
            selected: POLICY.capacity(0, cooperation.runtime())?,
        }));
    }
    let expectation = match options.scaling.kind() {
        Some(OptimGeneticScaling::Rank) => {
            ranking::rank_expectation(&order, count, cooperation).await?
        }
        Some(OptimGeneticScaling::Proportional) => {
            let values = real_scores(comparison, cooperation).await?;
            scaling::proportional(&values, count, cooperation).await?
        }
        Some(OptimGeneticScaling::Top) => {
            let quantity = parameter(options.scaling.extra().first(), 0.4)?;
            scaling::top(&order, quantity, count, cooperation).await?
        }
        Some(OptimGeneticScaling::ShiftLinear) => {
            let rate = parameter(options.scaling.extra().first(), 2.)?;
            let values = real_scores(comparison, cooperation).await?;
            shift_linear::expectation(&values, &order, rate, count, cooperation).await?
        }
        None => {
            let value = score_argument(scores, offset..end, cooperation).await?;
            match scaling::custom(
                &options.scaling,
                value,
                members,
                count,
                invoker,
                cooperation,
            )
            .await?
            {
                ControlFlow::Continue(value) => value,
                ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
            }
        }
    };
    scaling::validate(&expectation, count, cooperation).await?;
    let expectation = Tensor::new(expectation, vec![members, 1]).map_err(invalid)?;
    let weights = expectation
        .as_f64_slice()
        .expect("double parent expectations");
    let selected = match options.selection.kind() {
        Some(OptimGeneticSelection::StochasticUniform) => {
            let draws = random::uniform(1, cooperation).await?;
            ranking::stochastic_uniform(weights, count, draws[0], cooperation).await?
        }
        Some(OptimGeneticSelection::Tournament) => {
            let draws = count
                .checked_mul(options.tournament_size)
                .ok_or_else(|| invalid("tournament draw count overflow"))?;
            let draws = random::uniform(draws, cooperation).await?;
            ranking::tournament(weights, &draws, options.tournament_size, cooperation).await?
        }
        Some(OptimGeneticSelection::Uniform) => {
            let draws = random::uniform(count, cooperation).await?;
            selection::uniform(members, &draws, cooperation).await?
        }
        Some(OptimGeneticSelection::Roulette) => {
            let draws = random::uniform(count, cooperation).await?;
            selection::roulette(weights, &draws, cooperation).await?
        }
        Some(OptimGeneticSelection::Remainder) => {
            let remainder = selection::Remainder::prepare(weights, count, cooperation).await?;
            let draws = random::uniform(remainder.remaining(), cooperation).await?;
            remainder.finish(&draws, cooperation).await?
        }
        None => {
            match selected::custom(
                &options.selection,
                Value::Tensor(expectation.clone()),
                count,
                members,
                controls
                    .ok_or_else(|| invalid("custom selection requires its effective controls"))?,
                invoker,
                cooperation,
            )
            .await?
            {
                ControlFlow::Continue(value) => value,
                ControlFlow::Break(stop) => return Ok(ControlFlow::Break(stop)),
            }
        }
    };
    if selected.len() != count || selected.iter().any(|&row| row >= members) {
        return Err(invalid(
            "selection changed the requested parent count or population",
        ));
    }
    Ok(ControlFlow::Continue(Parents {
        order,
        expectation,
        selected,
    }))
}

fn parameter(value: Option<&Value>, default: f64) -> BuiltinResult<f64> {
    value
        .map(|value| options::real(options::Owner::Solver, value))
        .transpose()
        .map(|value| value.unwrap_or(default))
}

async fn real_scores(
    comparison: &[ranking::Score],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut values = POLICY.capacity(comparison.len(), cooperation.runtime())?;
    for &score in comparison {
        values.push(match score {
            ranking::Score::Defined(value) => value,
            _ => f64::NAN,
        });
        cooperation.completed(1).await?;
    }
    Ok(values)
}
async fn score_argument(
    scores: &Value,
    range: std::ops::Range<usize>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Value> {
    let count = range.len();
    match scores {
        Value::Tensor(value) => {
            let source = value
                .as_f64_slice()
                .ok_or_else(|| invalid("internal scores must be double"))?;
            if range.start == 0 && range.end == source.len() {
                return Ok(scores.clone());
            }
            let mut data = POLICY.capacity(count, cooperation.runtime())?;
            for &value in &source[range] {
                data.push(value);
                cooperation.completed(1).await?;
            }
            Ok(Value::Tensor(
                Tensor::new(data, vec![count, 1]).map_err(invalid)?,
            ))
        }
        Value::ComplexTensor(value) => {
            let source = value
                .as_f64_slice()
                .ok_or_else(|| invalid("internal complex scores must be double"))?;
            if range.start == 0 && range.end == source.len() {
                return Ok(scores.clone());
            }
            let mut data = POLICY.capacity(count, cooperation.runtime())?;
            for value in &source[range] {
                data.push((value.0, value.1));
                cooperation.completed(1).await?;
            }
            Ok(Value::ComplexTensor(
                runmat_value::ComplexTensor::new(data, vec![count, 1]).map_err(invalid)?,
            ))
        }
        _ => Err(invalid("internal score class changed")),
    }
}
