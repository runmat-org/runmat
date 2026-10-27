//! Deterministic branch-and-bound over exact certified original-model bounds.
use super::*;
use common::conic::{
    linear::{ExactLowerBound, LinearCertificates},
    solve::{self, Candidate},
};
use model::{Bound, Workspace};
use num_rational::BigRational;
use proofs::Incumbent;
use tree::Tree;
mod observation;
use observation::{accept_observed, observe, Event, Frontier};

#[derive(Clone, Copy)]
pub(super) enum Stop {
    Nodes,
    Time,
    FeasiblePoints,
    Numerical,
    Observer,
}
#[derive(Clone, Copy)]
pub(super) enum Terminal {
    Optimal,
    Infeasible,
    Unbounded,
    Stopped(Stop),
}
pub(super) struct Outcome {
    pub incumbent: Option<Incumbent>,
    pub lower: Option<BigRational>,
    pub terminal: Terminal,
}
fn prunable(
    bound: Option<&ExactLowerBound>,
    incumbent: Option<&Incumbent>,
    cutoff: Option<&BigRational>,
) -> bool {
    let Some(bound) = bound else {
        return false;
    };
    incumbent.is_some_and(|upper| bound.value() >= &upper.objective)
        || cutoff.is_some_and(|cutoff| bound.value() > cutoff)
}
pub(super) fn accept(
    candidate: Incumbent,
    best: &mut Option<Incumbent>,
    controls: &mut controls::Controls,
) -> BuiltinResult<bool> {
    // A cutoff excludes only objectives strictly greater than its value.
    if controls
        .cutoff
        .as_ref()
        .is_some_and(|c| &candidate.objective > c)
    {
        return Ok(false);
    }
    // Rechecking the retained incumbent is not a newly found point. Other
    // certified feasible candidates count even when they do not improve U.
    if best
        .as_ref()
        .is_some_and(|old| candidate.point == old.point)
    {
        return Ok(false);
    }
    controls.feasible_points = controls
        .feasible_points
        .checked_add(1)
        .ok_or_else(|| invalid("feasible-point counter overflow"))?;
    if best
        .as_ref()
        .is_none_or(|old| candidate.objective < old.objective)
    {
        controls.improving_points = controls
            .improving_points
            .checked_add(1)
            .ok_or_else(|| invalid("improving-point counter overflow"))?;
        *best = Some(candidate);
        return Ok(true);
    }
    Ok(false)
}

mod branch;
mod finish;
mod repair;
pub(super) use branch::exact_branch;
use branch::numeric_branch;
use finish::finish;
use repair::repair;

pub(super) async fn solve(
    workspace: Workspace,
    controls: &mut controls::Controls,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<(Outcome, Workspace)> {
    solve_validated(workspace, controls, cooperation, &mut compiled::Supplied).await
}
pub(super) async fn solve_validated<V: compiled::Validation>(
    mut workspace: Workspace,
    controls: &mut controls::Controls,
    cooperation: &mut ScalarCooperation,
    validation: &mut V,
) -> BuiltinResult<(Outcome, Workspace)> {
    let outcome = run(&mut workspace, controls, cooperation, validation).await?;
    workspace.reset(cooperation).await?;
    // No done replay on a failed/cancelled numerical or observer transaction.
    // A stop flag at done has no further solver work to interrupt.
    let _ = observe(
        Event {
            state: common::output_callback::OutputState::Done,
            new_point: false,
            best: outcome.incumbent.as_ref(),
            lower: outcome.lower.as_ref(),
        },
        &workspace,
        controls,
        cooperation,
        validation,
    )
    .await?;
    workspace.publish_bounds(cooperation).await?;
    Ok((outcome, workspace))
}
async fn run<V: compiled::Validation>(
    workspace: &mut Workspace,
    controls: &mut controls::Controls,
    cooperation: &mut ScalarCooperation,
    validation: &mut V,
) -> BuiltinResult<Outcome> {
    let mut tree = Tree::new()?;
    let mut best = None;
    if let Some(stop) = observe(
        Event {
            state: common::output_callback::OutputState::Init,
            new_point: false,
            best: None,
            lower: None,
        },
        workspace,
        controls,
        cooperation,
        validation,
    )
    .await?
    {
        return finish(&tree, best, Terminal::Stopped(stop), cooperation).await;
    }
    // x0 is a genuine repair proposal, never an unchecked incumbent. Repair
    // uses a separate fixed-integer relaxation and exact original-model check.
    if let Some(initial) = workspace.initial.take() {
        if let Some(point) = repair(workspace, &initial, controls, cooperation, validation).await? {
            if let Some(stop) = accept_observed(
                point,
                &mut best,
                Frontier {
                    tree: &tree,
                    current: None,
                },
                workspace,
                controls,
                cooperation,
                validation,
            )
            .await?
            {
                return finish(&tree, best, Terminal::Stopped(stop), cooperation).await;
            }
        }
    }
    loop {
        cooperation.completed(0).await?;
        if let Some(stop) = controls.stopped() {
            return finish(&tree, best, Terminal::Stopped(stop), cooperation).await;
        }
        let Some(id) = tree.open.pop() else {
            let terminal = if best.is_some() {
                Terminal::Optimal
            } else {
                Terminal::Infeasible
            };
            return finish(&tree, best, terminal, cooperation).await;
        };
        if prunable(
            tree.node(id).lower.as_ref(),
            best.as_ref(),
            controls.cutoff.as_ref(),
        ) {
            tree.record_cutoff(id, controls.cutoff.as_ref());
            continue;
        }
        tree.materialize(id, workspace, cooperation).await?;
        if !workspace.consistent() {
            continue;
        }
        let seed = workspace.seed(cooperation).await?;
        let model = workspace.view();
        let Some(certificates) = LinearCertificates::borrow(&model, cooperation).await? else {
            tree.restore_open(id)?;
            return finish(&tree, best, Terminal::Stopped(Stop::Numerical), cooperation).await;
        };
        if certificates.is_infeasible() && validation.matrix_is_original() {
            continue;
        }
        if workspace.n() == 0 {
            if let Some(point) = proofs::incumbent(
                &model,
                &workspace.integers,
                &workspace.semis,
                &[],
                cooperation,
            )
            .await?
            {
                if let Some(stop) = accept_observed(
                    point,
                    &mut best,
                    Frontier {
                        tree: &tree,
                        current: Some(id),
                    },
                    workspace,
                    controls,
                    cooperation,
                    validation,
                )
                .await?
                {
                    tree.restore_open(id)?;
                    return finish(&tree, best, Terminal::Stopped(stop), cooperation).await;
                }
                continue;
            }
            tree.restore_open(id)?;
            return finish(&tree, best, Terminal::Stopped(Stop::Numerical), cooperation).await;
        }
        controls.nodes = controls
            .nodes
            .checked_add(1)
            .ok_or_else(|| invalid("node count overflow"))?;
        let result = {
            let node = workspace.node();
            let mut original = compiled::Terminal {
                node: &node,
                validation: &mut *validation,
            };
            solve::candidate_validated(&model, &seed, controls, cooperation, &mut original).await?
        };
        controls.iteration_line(cooperation.runtime())?;
        if controls.time_limit_reached() {
            tree.restore_open(id)?;
            return finish(&tree, best, Terminal::Stopped(Stop::Time), cooperation).await;
        }
        let (x, z, point) = match result {
            Candidate::CertifiedInfeasible => continue,
            Candidate::CertifiedUnbounded {
                direction,
                feasible,
            } => {
                if proofs::integral_unbounded(
                    &feasible,
                    &direction,
                    &workspace.integers,
                    &workspace.semis,
                    cooperation,
                )
                .await?
                {
                    if let Some(point) = proofs::from_verified(
                        &model,
                        &workspace.integers,
                        &workspace.semis,
                        feasible,
                        cooperation,
                    )
                    .await?
                    {
                        if let Some(stop) = accept_observed(
                            point,
                            &mut best,
                            Frontier {
                                tree: &tree,
                                current: Some(id),
                            },
                            workspace,
                            controls,
                            cooperation,
                            validation,
                        )
                        .await?
                        {
                            tree.restore_open(id)?;
                            return finish(&tree, best, Terminal::Stopped(stop), cooperation).await;
                        }
                    }
                    return finish(&tree, best, Terminal::Unbounded, cooperation).await;
                }
                drop(direction);
                // The actual exact feasible fractional point defines branches;
                // an unbounded relaxation is never reinterpreted as MILP proof.
                (None, None, Some(feasible))
            }
            Candidate::Converged { x, z } => {
                let point =
                    common::conic::terminal::verified_point(&model, &x, cooperation).await?;
                (Some(x), Some(z), point)
            }
            Candidate::Stopped { kind, x } => {
                if let Some(x) = x {
                    if let Some(point) = proofs::incumbent(
                        &model,
                        &workspace.integers,
                        &workspace.semis,
                        &x,
                        cooperation,
                    )
                    .await?
                    {
                        if let Some(stop) = accept_observed(
                            point,
                            &mut best,
                            Frontier {
                                tree: &tree,
                                current: Some(id),
                            },
                            workspace,
                            controls,
                            cooperation,
                            validation,
                        )
                        .await?
                        {
                            tree.restore_open(id)?;
                            return finish(&tree, best, Terminal::Stopped(stop), cooperation).await;
                        }
                    }
                }
                tree.restore_open(id)?;
                let stop = if matches!(kind, solve::StopKind::TimeLimit) {
                    Stop::Time
                } else {
                    Stop::Numerical
                };
                return finish(&tree, best, Terminal::Stopped(stop), cooperation).await;
            }
            Candidate::Difficulty(_) => {
                tree.restore_open(id)?;
                return finish(&tree, best, Terminal::Stopped(Stop::Numerical), cooperation).await;
            }
        };
        if let Some(z) = z {
            tree.strengthen(
                id,
                validation
                    .lower_bound(&workspace.node(), &certificates, &z, cooperation)
                    .await?,
            );
        }
        drop(certificates);
        let split = if let Some(point) = point.as_ref() {
            match domains::exact_split(point.values(), &workspace.semis, cooperation).await? {
                Some(split) => Some(split),
                None => exact_branch(point.values(), &workspace.integers, cooperation).await?,
            }
        } else {
            x.as_ref().and_then(|x| {
                let progresses =
                    |column, floor, ceil| workspace.branch_progress(column, floor, ceil);
                domains::numeric_split(x, &workspace.semis, progresses)
                    .or_else(|| numeric_branch(x, &workspace.integers, progresses))
            })
        };
        if let Some(point) = point {
            if let Some(point) = proofs::from_verified(
                &model,
                &workspace.integers,
                &workspace.semis,
                point,
                cooperation,
            )
            .await?
            {
                if let Some(stop) = accept_observed(
                    point,
                    &mut best,
                    Frontier {
                        tree: &tree,
                        current: Some(id),
                    },
                    workspace,
                    controls,
                    cooperation,
                    validation,
                )
                .await?
                {
                    tree.restore_open(id)?;
                    return finish(&tree, best, Terminal::Stopped(stop), cooperation).await;
                }
            }
        }
        // Exact prunability is separate from the requested global gap stop.
        if prunable(
            tree.node(id).lower.as_ref(),
            best.as_ref(),
            controls.cutoff.as_ref(),
        ) {
            tree.record_cutoff(id, controls.cutoff.as_ref());
            continue;
        }
        if let Some(stop) = controls.stopped() {
            tree.restore_open(id)?;
            return finish(&tree, best, Terminal::Stopped(stop), cooperation).await;
        }
        if let Some(x) = x.as_deref().filter(|_| workspace.is_mixed()) {
            if let Some(point) = repair(workspace, x, controls, cooperation, validation).await? {
                if let Some(stop) = accept_observed(
                    point,
                    &mut best,
                    Frontier {
                        tree: &tree,
                        current: Some(id),
                    },
                    workspace,
                    controls,
                    cooperation,
                    validation,
                )
                .await?
                {
                    tree.restore_open(id)?;
                    return finish(&tree, best, Terminal::Stopped(stop), cooperation).await;
                }
            }
        }
        if prunable(
            tree.node(id).lower.as_ref(),
            best.as_ref(),
            controls.cutoff.as_ref(),
        ) {
            tree.record_cutoff(id, controls.cutoff.as_ref());
            continue;
        }
        // Keep the current leaf live while checking the global bound. A gap
        // stop must preserve the actual open bound rather than falsely report0.
        tree.restore_open(id)?;
        if let Some(stop) = controls.stopped() {
            return finish(&tree, best, Terminal::Stopped(stop), cooperation).await;
        }
        let lower = tree.lower_bound(cooperation).await?;
        if let Some(stop) = observe(
            Event {
                state: common::output_callback::OutputState::Iter,
                new_point: false,
                best: best.as_ref(),
                lower: lower.as_ref().map(|v| v.value()),
            },
            workspace,
            controls,
            cooperation,
            validation,
        )
        .await?
        {
            return finish(&tree, best, Terminal::Stopped(stop), cooperation).await;
        }
        if best.as_ref().zip(lower.as_ref()).is_some_and(|(u, l)| {
            l.value() >= &u.objective
                || gap::satisfies(
                    &u.objective,
                    l.value(),
                    controls.absolute_gap,
                    controls.relative_gap,
                )
        }) {
            return finish(&tree, best, Terminal::Optimal, cooperation).await;
        }
        if let Some(stop) = controls.stopped() {
            return finish(&tree, best, Terminal::Stopped(stop), cooperation).await;
        }
        tree.open.pop();
        let Some((column, floor, ceil)) = split else {
            tree.restore_open(id)?;
            return finish(&tree, best, Terminal::Stopped(Stop::Numerical), cooperation).await;
        };
        tree.branch(id, column, floor, ceil)?;
    }
}
