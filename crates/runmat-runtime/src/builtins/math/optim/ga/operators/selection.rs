//! Parent sampling from expectations with one immutable CDF per wave.
//! RNG variates come from the selected stream; deterministic remainder copies
//! consume no random values, and every stochastic parent consumes one.
use super::super::{invalid, ScalarCooperation, POLICY};
use crate::BuiltinResult;

fn variate(value: f64) -> BuiltinResult<f64> {
    if value.is_finite() && (0.0..1.0).contains(&value) {
        Ok(value)
    } else {
        Err(invalid("selection requires uniform variates in [0,1)"))
    }
}
pub(in super::super) async fn uniform(
    members: usize,
    draws: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<usize>> {
    if members == 0 && !draws.is_empty() {
        return Err(invalid("cannot select parents from an empty population"));
    }
    let mut selected = POLICY.capacity(draws.len(), cooperation.runtime())?;
    for &draw in draws {
        selected.push(((variate(draw)? * members as f64) as usize).min(members - 1));
        cooperation.completed(1).await?;
    }
    Ok(selected)
}
pub(in super::super) async fn roulette(
    expectation: &[f64],
    draws: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<usize>> {
    let mut selected = POLICY.capacity(draws.len(), cooperation.runtime())?;
    if draws.is_empty() {
        return Ok(selected);
    }
    let mut cumulative = POLICY.capacity(expectation.len(), cooperation.runtime())?;
    let mut total = 0.;
    for &weight in expectation {
        if !weight.is_finite() || weight < 0. {
            return Err(invalid(
                "selection expectations must be finite and nonnegative",
            ));
        }
        total += weight;
        cumulative.push(total);
        cooperation.completed(1).await?;
    }
    if !total.is_finite() || total <= 0. {
        return Err(invalid(
            "selection expectations must have positive finite sum",
        ));
    }
    let last_inside = f64::from_bits(total.to_bits() - 1);
    for &draw in draws {
        let target = (variate(draw)? * total).min(last_inside);
        let mut low = 0;
        let mut high = cumulative.len();
        while low < high {
            let middle = low + (high - low) / 2;
            if cumulative[middle] <= target {
                low = middle + 1;
            } else {
                high = middle;
            }
            cooperation.completed(1).await?;
        }
        selected.push(low);
        cooperation.completed(1).await?;
    }
    Ok(selected)
}
pub(in super::super) struct Remainder {
    deterministic: Vec<usize>,
    residual: Vec<f64>,
    remaining: usize,
}
impl Remainder {
    pub async fn prepare(
        expectation: &[f64],
        parents: usize,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let mut deterministic = POLICY.capacity(parents, cooperation.runtime())?;
        let mut residual = POLICY.capacity(expectation.len(), cooperation.runtime())?;
        let mut total = 0.;
        for (member, &expected) in expectation.iter().enumerate() {
            if !expected.is_finite() || expected < 0. {
                return Err(invalid(
                    "remainder expectations must be finite and nonnegative",
                ));
            }
            total += expected;
            let whole = crate::builtins::math::optim::common::option_usize_value(
                "ga",
                Some(&runmat_value::Value::Num(expected.floor())),
                "integer offspring expectation",
                0,
            )?;
            if whole > parents - deterministic.len() {
                return Err(invalid("integer offspring expectations exceed nParents"));
            }
            for _ in 0..whole {
                deterministic.push(member);
                cooperation.completed(1).await?;
            }
            residual.push(expected - expected.floor());
            cooperation.completed(1).await?;
        }
        let tolerance = 32. * f64::EPSILON * (parents as f64).max(1.);
        if !total.is_finite() || (total - parents as f64).abs() > tolerance {
            return Err(invalid("offspring expectations must sum to nParents"));
        }
        let remaining = parents - deterministic.len();
        Ok(Self {
            deterministic,
            residual,
            remaining,
        })
    }
    pub fn remaining(&self) -> usize {
        self.remaining
    }
    pub async fn finish(
        mut self,
        draws: &[f64],
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Vec<usize>> {
        if draws.len() != self.remaining {
            return Err(invalid(
                "remainder selection variates must match remaining parents",
            ));
        }
        let tail = roulette(&self.residual, draws, cooperation).await?;
        // The deterministic buffer reserved the full parent count before any
        // random draw; appending moves only row indices and cannot reallocate.
        self.deterministic.extend(tail);
        Ok(self.deterministic)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stochastic_and_deterministic_parent_counts_are_distinct() {
        futures::executor::block_on(async {
            let mut cooperation =
                ScalarCooperation::for_builtin(super::super::super::IDENTITY, None);
            let draws = [0., 0.25, 0.5, f64::from_bits(1.0f64.to_bits() - 1)];
            assert_eq!(
                roulette(&[0., 2., 0., 2., 0.], &draws, &mut cooperation)
                    .await
                    .unwrap(),
                [1, 1, 3, 3]
            );
            assert_eq!(
                uniform(4, &draws, &mut cooperation).await.unwrap(),
                [0, 1, 2, 3]
            );
            let remainder = Remainder::prepare(&[2.25, 0.75, 1.], 4, &mut cooperation)
                .await
                .unwrap();
            assert_eq!(remainder.remaining(), 1);
            assert_eq!(
                remainder.finish(&[0.5], &mut cooperation).await.unwrap(),
                [0, 0, 2, 1]
            );
            let deterministic = Remainder::prepare(&[0., 2., 1.], 3, &mut cooperation)
                .await
                .unwrap();
            assert_eq!(deterministic.remaining(), 0);
            assert_eq!(
                deterministic.finish(&[], &mut cooperation).await.unwrap(),
                [1, 1, 2]
            );
        });
    }
}
