//! Parent expectations are nonnegative counts, separate from raw fitness and
//! original constraint merit. They are never published as the user's fval.
use super::super::{common, invalid, numeric, options::Operator, ScalarCooperation, POLICY};
use super::ranking::RankOrder;
use crate::BuiltinResult;
use runmat_builtins::OptimGeneticScaling;
use runmat_value::Value;
use std::ops::ControlFlow;

/// Literal proportional scaling. Negative/undefined raw values do not define
/// nonnegative expectations; no inverse/rank rule is silently substituted.
pub(in super::super) async fn proportional(
    raw: &[f64],
    parents: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut scale = 0.0_f64;
    for &score in raw {
        if !score.is_finite() || score < 0. {
            return Err(invalid(
                "proportional fitness scaling requires finite nonnegative scores",
            ));
        }
        scale = scale.max(score);
        cooperation.completed(1).await?;
    }
    if raw.is_empty() && parents != 0 {
        return Err(invalid("positive nParents requires a nonempty population"));
    }
    let mut values = POLICY.capacity(raw.len(), cooperation.runtime())?;
    let mut sum = 0.;
    for &score in raw {
        let weight = if scale == 0. { 1. } else { score / scale };
        values.push(weight);
        sum += weight;
        cooperation.completed(1).await?;
    }
    if !raw.is_empty() {
        for value in &mut values {
            *value = (*value / sum) * parents as f64;
            cooperation.completed(1).await?;
        }
    }
    Ok(values)
}

pub(in super::super) async fn top(
    order: &RankOrder,
    quantity: f64,
    parents: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    if !quantity.is_finite() || quantity < 0. || quantity > 1. && quantity.fract() != 0. {
        return Err(invalid(
            "top scaling quantity must be a fraction in [0,1] or an integer count",
        ));
    }
    let count = if quantity <= 1. {
        (quantity * order.len() as f64).ceil() as usize
    } else {
        common::option_usize_value("ga", Some(&Value::Num(quantity)), "top scaling quantity", 0)?
    };
    if count > order.len() || count == 0 && parents != 0 {
        return Err(invalid("top scaling must select at least one eligible individual for positive nParents and cannot exceed the population"));
    }
    let mut values = POLICY.capacity(order.len(), cooperation.runtime())?;
    for _ in 0..order.len() {
        values.push(0.);
        cooperation.completed(1).await?;
    }
    if count != 0 {
        for &member in &order.as_slice()[..count] {
            values[member] = parents as f64 / count as f64;
            cooperation.completed(1).await?;
        }
    }
    Ok(values)
}

pub(in super::super) async fn custom<I: super::super::callbacks::Invocation>(
    operator: &Operator<OptimGeneticScaling>,
    scores: Value,
    members: usize,
    parents: usize,
    invoker: &mut I,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<ControlFlow<I::Stop, Vec<f64>>> {
    let count = u64::try_from(parents)
        .ok()
        .and_then(|count| runmat_value::IntValue::U64(count).to_exact_f64())
        .ok_or_else(|| invalid("nParents cannot be represented exactly for scaling callback"))?;
    let returned = operator
        .call_with([scores, Value::Num(count)], 1, invoker, cooperation)
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
    let values = returned
        .with_owned_async(|value| async {
            let value = common::constrained_numeric::read(
                value,
                "fitness scaling expectations",
                common::constrained_numeric::Geometry::Matrix(members, 1),
                &numeric::INPUT_POLICY,
                cooperation,
            )
            .await?;
            let common::constrained_numeric::Numeric::Real { values, .. } = value else {
                return Err(invalid(
                    "fitness scaling must return finite real expectations",
                ));
            };
            validate(&values, parents, cooperation).await?;
            Ok(ControlFlow::Continue(values))
        })
        .await?;
    Ok(values)
}

pub(in super::super) async fn validate(
    expectation: &[f64],
    parents: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<()> {
    let mut total = 0.;
    for &value in expectation {
        if !value.is_finite() || value < 0. {
            return Err(invalid(
                "fitness scaling expectations must be finite and nonnegative",
            ));
        }
        total += value;
        cooperation.completed(1).await?;
    }
    let tolerance = 32. * f64::EPSILON * (parents as f64).max(1.);
    if !total.is_finite() || (total - parents as f64).abs() > tolerance {
        return Err(invalid("fitness scaling expectations must sum to nParents"));
    }
    Ok(())
}
