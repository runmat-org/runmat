//! Ordered population comparison and actual parent expectations.
use super::super::{invalid, ScalarCooperation, POLICY};
use crate::BuiltinResult;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
pub(in super::super) enum Score {
    Defined(f64),
    /// Strict shifted-barrier domain restoration, not penalty GA fitness.
    RestoreDomain(f64),
    Undefined,
}
impl Score {
    fn valid(self) -> bool {
        match self {
            Self::Defined(value) => value.is_finite(),
            Self::RestoreDomain(value) => value.is_finite() && value >= 0.0,
            Self::Undefined => true,
        }
    }
    pub(in super::super) fn compare(self, other: Self) -> std::cmp::Ordering {
        use std::cmp::Ordering::{Equal, Greater, Less};
        match (self, other) {
            (Self::Defined(a), Self::Defined(b))
            | (Self::RestoreDomain(a), Self::RestoreDomain(b)) => a.total_cmp(&b),
            (Self::Defined(_), _) => Less,
            (_, Self::Defined(_)) => Greater,
            (Self::RestoreDomain(_), Self::Undefined) => Less,
            (Self::Undefined, Self::RestoreDomain(_)) => Greater,
            (Self::Undefined, Self::Undefined) => Equal,
        }
    }
}

/// Only the actual comparison owner can mint a complete rank permutation.
pub(in super::super) struct RankOrder(Vec<usize>);
impl RankOrder {
    pub fn as_slice(&self) -> &[usize] {
        &self.0
    }
}
impl std::ops::Deref for RankOrder {
    type Target = [usize];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Invalid/undefined trials rank after real finite trials; no fabricated
/// objective value is published for them. Stable member order breaks ties.
pub(in super::super) async fn order(
    scores: &[Score],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RankOrder> {
    let mut order = POLICY.capacity(scores.len(), cooperation.runtime())?;
    for index in 0..scores.len() {
        if !scores[index].valid() {
            return Err(invalid("comparison scores must be finite or undefined"));
        }
        order.push(index);
        cooperation.completed(1).await?;
    }
    use crate::builtins::common::cooperative_sort::{SortProgress, StableSortWork};
    let mut sort = StableSortWork::new(&mut order, |&a: &usize, &b: &usize| {
        scores[a].compare(scores[b]).then(a.cmp(&b))
    });
    loop {
        let progress = sort.advance().map_err(|source| {
            crate::build_runtime_error("ga: population ranking allocation failed")
                .with_builtin("ga")
                .with_identifier("RunMat:ga:AllocationFailed")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .with_source(source.0)
                .build()
        })?;
        let (complete, work) = match progress {
            SortProgress::Pending(work) => (false, work),
            SortProgress::Complete(work) => (true, work),
        };
        cooperation.completed(work).await?;
        if complete {
            break;
        }
    }
    drop(sort);
    Ok(RankOrder(order))
}

/// Rank fitness scaling assigns expected offspring proportional to 1/sqrt(r).
/// Store by original member identity, not by sorted position.
pub(in super::super) async fn rank_expectation(
    order: &RankOrder,
    parents: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<f64>> {
    let mut expectation = POLICY.capacity(order.len(), cooperation.runtime())?;
    expectation.resize(order.len(), 0.0);
    let mut weight = 0.0;
    for rank in 1..=order.len() {
        weight += 1.0 / (rank as f64).sqrt();
        cooperation.completed(1).await?;
    }
    if !order.is_empty() && (!weight.is_finite() || weight <= 0.0) {
        return Err(invalid("rank expectation cannot be represented"));
    }
    for (rank, &member) in order.iter().enumerate() {
        let slot = expectation
            .get_mut(member)
            .ok_or_else(|| invalid("rank contains a nonexistent population member"))?;
        *slot = ((parents as f64) / weight) / ((rank + 1) as f64).sqrt();
        cooperation.completed(1).await?;
    }
    Ok(expectation)
}

async fn expectation_sum(
    expectation: &[f64],
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<f64>> {
    let mut total = 0.0;
    for &value in expectation {
        if !value.is_finite() || value < 0.0 {
            return Ok(None);
        }
        total += value;
        cooperation.completed(1).await?;
    }
    Ok((total.is_finite() && total > 0.0).then_some(total))
}

/// Equally spaced samples with one uniform origin. Variates have already been
/// reserved from the invocation's original RNG stream before this operation.
pub(in super::super) async fn stochastic_uniform(
    expectation: &[f64],
    parents: usize,
    origin: f64,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<usize>> {
    let mut selected = POLICY.capacity(parents, cooperation.runtime())?;
    if parents == 0 {
        return Ok(selected);
    }
    if !origin.is_finite() || !(0.0..1.0).contains(&origin) {
        return Err(invalid("selection requires a uniform variate in [0,1)"));
    }
    let total = expectation_sum(expectation, cooperation)
        .await?
        .ok_or_else(|| invalid("selection expectations must have positive finite sum"))?;
    let step = total / parents as f64;
    let mut member = 0;
    let mut cumulative = expectation[0];
    for parent in 0..parents {
        // The last addition can round up to exactly `parents`, even though
        // the original uniform variate is strictly below one. Keep the sample
        // inside the positive total rather than select a zero-weight tail.
        let last_inside = f64::from_bits(total.to_bits() - 1);
        let target = ((parent as f64 + origin) * step).min(last_inside);
        while cumulative <= target && member + 1 < expectation.len() {
            member += 1;
            cumulative += expectation[member];
            cooperation.completed(1).await?;
        }
        selected.push(member);
        cooperation.completed(1).await?;
    }
    Ok(selected)
}

/// Tournaments compare the scaled expectations supplied to SelectionFcn.
/// They never silently substitute a separate raw-fitness ordering.
pub(in super::super) async fn tournament(
    expectation: &[f64],
    draws: &[f64],
    size: usize,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Vec<usize>> {
    if expectation.is_empty() || size < 2 || !draws.len().is_multiple_of(size) {
        return Err(invalid("tournament geometry is invalid"));
    }
    if expectation_sum(expectation, cooperation).await?.is_none() {
        return Err(invalid(
            "tournament expectations require a positive finite sum",
        ));
    }
    let mut selected = POLICY.capacity(draws.len() / size, cooperation.runtime())?;
    for players in draws.chunks_exact(size) {
        let index = |draw: f64| {
            (draw.is_finite() && (0.0..1.0).contains(&draw))
                .then(|| ((draw * expectation.len() as f64) as usize).min(expectation.len() - 1))
        };
        let mut winner = None;
        for &player in players {
            let member = index(player).ok_or_else(|| invalid("invalid tournament variate"))?;
            if winner.is_none_or(|old: usize| expectation[member] > expectation[old]) {
                winner = Some(member);
            }
            cooperation.completed(1).await?;
        }
        selected.push(winner.expect("nonempty tournament"));
    }
    Ok(selected)
}
