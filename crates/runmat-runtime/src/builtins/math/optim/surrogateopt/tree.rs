//! Bounded host integer branch search with actual continuous SQP nodes.
//! RunMat visits assignments FIFO and cycles original integer coordinates by
//! depth. This internal policy is not a claim about unspecified trajectories
//! of another implementation. Exhaustion is never an infeasibility proof.
use super::{
    domain::BoxView,
    invalid,
    multistart::{self, Start},
    rbf::Training,
    refinement::Session,
    region::{Neighbor, Region, Scope},
    NAME,
};
use crate::builtins::math::optim::{common::ScalarCooperation, fmincon::internal::Refinement};
use crate::BuiltinResult;
mod nodes;
use nodes::Nodes;

#[derive(Clone, Copy)]
pub(super) enum Case {
    Local,
    Reset,
}
impl Case {
    fn steps(self) -> usize {
        match self {
            Self::Local => 30,
            Self::Reset => 60,
        }
    }
    fn scope(self, scale: f64) -> Scope {
        match self {
            Self::Local => Scope::Local(scale),
            Self::Reset => Scope::Original,
        }
    }
}
pub(super) async fn search<T: Training, E: Training>(
    case: Case,
    domain: BoxView<'_>,
    anchor: &[f64],
    scale: f64,
    session: &mut Session<'_, T, E>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Refinement>> {
    if !session.matches_domain(&domain) {
        return Err(invalid("integer tree has a different original model owner"));
    }
    let mut region = Region::new(domain, anchor, case.scope(scale), cooperation).await?;
    let mut integers =
        runmat_value::exact_capacity_vec(region.integers().len()).map_err(|source| {
            crate::build_runtime_error("surrogateopt: tree integer-index allocation failed")
                .with_builtin(NAME)
                .with_source(source)
                .build()
        })?;
    for &column in region.integers() {
        let (lower, upper) = region.bounds();
        if lower[column] != upper[column] {
            integers.push(column);
        }
        cooperation.completed(1).await?;
    }
    if integers.is_empty() {
        return Err(invalid("integer tree has no free integer coordinate"));
    }
    let mut nodes = Nodes::new(anchor, case.steps())?;
    let mut best = None;
    let mut solves = 0usize;
    while solves < case.steps() {
        cooperation.completed(0).await?;
        if session.expired() {
            break;
        }
        let Some(node) = nodes.next() else {
            break;
        };
        let point = nodes.point(node, &integers, cooperation).await?;
        region.restore(cooperation).await?;
        let mut branch = true;
        for index in 0..region.integers().len() {
            let column = region.integers()[index];
            branch &= region.pin_integer(column, point[column])?;
            cooperation.completed(1).await?;
        }
        if !branch {
            continue;
        }
        let (lower, upper) = region.bounds();
        // The bound counts actual SQP node attempts, including the root,
        // rather than queue visits or a multiplied inner solve allowance.
        solves += 1;
        let candidate = session.solve(point, lower, upper, cooperation).await?;
        let mut exact_assignment = candidate.point.len() == region.variables();
        for &column in region.integers() {
            exact_assignment &= candidate.point.get(column).copied() == Some(lower[column]);
            cooperation.completed(1).await?;
        }
        if exact_assignment && session.preferred(&candidate, best.as_ref()) {
            best = Some(candidate);
        }
        if solves == case.steps() || session.expired() {
            break;
        }
        region.restore(cooperation).await?;
        let column = integers[nodes.depth(node)? % integers.len()];
        let center = nodes.value(node, column, cooperation).await?;
        for direction in [Neighbor::Lower, Neighbor::Upper] {
            if let Some(value) = region.neighbor(column, center, direction)? {
                nodes
                    .append(node, column, value, &integers, cooperation)
                    .await?;
            }
        }
    }
    // Reuse the already-solved best as one MultiStart start. Its exact point
    // and modeled values stay paired; no working copy or expensive replay.
    region.restore(cooperation).await?;
    let seed = match best {
        Some(best) => Start::Refined(best),
        None => {
            let mut point = runmat_value::exact_capacity_vec(anchor.len()).map_err(|source| {
                crate::build_runtime_error("surrogateopt: tree seed allocation failed")
                    .with_builtin(NAME)
                    .with_source(source)
                    .build()
            })?;
            for &value in anchor {
                point.push(value);
                cooperation.completed(1).await?;
            }
            Start::Point(point)
        }
    };
    multistart::search(seed, &mut region, session, cooperation).await
}
