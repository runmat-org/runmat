//! A known-seed Nelder-Mead subsearch on the SAME original Pattern oracle.
use super::super::{
    buffer,
    evaluation::{self, Evaluator, Point},
    invalid,
    options::Options,
    poll::{Completion, Improvement, MeritChange, Outcome},
    timing::Deadline,
};
use crate::builtins::math::optim::{
    common::{nelder_mead as nm, ScalarCooperation},
    fminsearch,
};
use crate::BuiltinResult;
use runmat_builtins::OptimOptionField as Field;
struct Oracle<'a, 'b> {
    evaluator: &'a mut Evaluator<'b>,
    deadline: &'a Deadline,
    start: usize,
    limit: usize,
    check_values: bool,
}
impl nm::Oracle for Oracle<'_, '_> {
    fn cooperation(&mut self) -> &mut ScalarCooperation {
        self.evaluator.cooperation
    }
    fn calls(&self) -> usize {
        self.evaluator.state.objective_calls - self.start
    }
    fn remaining(&self) -> usize {
        self.limit
            .saturating_sub(nm::Oracle::calls(self))
            .min(self.evaluator.remaining())
    }
    async fn measure(&mut self, x: &[f64]) -> BuiltinResult<nm::Measurement> {
        if self.deadline.reached() {
            return Ok(nm::Measurement::Stopped(nm::OracleStop::Time));
        }
        if nm::Oracle::remaining(self) == 0 {
            return Ok(nm::Measurement::Stopped(nm::OracleStop::Evaluations));
        }
        Ok(match self.evaluator.measure_point(x).await? {
            Ok(value) => nm::Measurement::Defined(value.f),
            Err(evaluation::Stop::Budget) => nm::Measurement::Stopped(nm::OracleStop::Evaluations),
            Err(evaluation::Stop::Undefined) if self.check_values => {
                return Err(invalid(
                    "searchneldermead FunValCheck rejected an undefined objective trial",
                ))
            }
            Err(evaluation::Stop::Undefined | evaluation::Stop::Cached) => {
                nm::Measurement::Undefined
            }
        })
    }
}
pub(in crate::builtins::math::optim::patternsearch) async fn run(
    base: &Point,
    options: &Options,
    deadline: &Deadline,
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<Outcome> {
    if evaluator.context.constraints.is_some() {
        return Err(invalid(
            "searchneldermead supports unconstrained problems only",
        ));
    }
    let source = super::spec::nelder_options(
        options
            .callback(Field::SearchFcn)
            .ok_or_else(|| invalid("missing Nelder-Mead search generation"))?,
    )?;
    let local =
        fminsearch::options::Options::parse(source, base.x.len(), evaluator.cooperation.runtime())
            .await?;
    // This copy IS the first simplex vertex, not an evaluation/tracking mirror.
    let mut seed = buffer(base.x.len())?;
    for chunk in base.x.chunks(1024) {
        seed.extend_from_slice(chunk);
        evaluator.cooperation.completed(chunk.len() as u64).await?;
    }
    let mut observers = fminsearch::observers::Observers::new(
        &local,
        evaluator.context.shape,
        evaluator.context.scalar,
    );
    let mut events = fminsearch::simplex::Notifications {
        observers: &mut observers,
        display: local.display,
    };
    let mut oracle = Oracle {
        start: evaluator.state.objective_calls,
        evaluator,
        deadline,
        limit: local.max_calls,
        check_values: local.check_values,
    };
    let result = nm::minimize(
        nm::Policy {
            identity: super::super::IDENTITY,
            invalid: &runmat_builtins::PATTERNSEARCH_ERROR_INVALID_ARGUMENT,
            allocation_identifier: "RunMat:patternsearch:AllocationFailed",
            retry: crate::GpuGatherRetry::Never,
        },
        nm::Initial::Known(nm::Point { x: seed, f: base.f }),
        &mut oracle,
        &nm::Controls {
            tol_x: local.tol_x,
            tol_fun: local.tol_fun,
            max_iter: local.max_iter,
        },
        &mut events,
    )
    .await?;
    // Finish reporting uses the SAME local observer/display profile; it does
    // not restart an objective, publish public solver outputs, or set counts.
    fminsearch::simplex::finish_search(
        local.display,
        &result,
        nm::Oracle::cooperation(&mut oracle),
    )?;
    let completion = match result.reason {
        nm::Reason::Time => Completion::Time,
        nm::Reason::Evaluations if oracle.evaluator.remaining() == 0 => Completion::Budget,
        _ => Completion::Finished,
    };
    let improved = if result.point.f.is_finite() && result.point.f < base.f {
        Some(Improvement {
            merit: MeritChange {
                before: base.f,
                after: result.point.f,
            },
            point: Point {
                x: result.point.x,
                f: result.point.f,
                inequality: Vec::new().into(),
                equality: Vec::new().into(),
            },
            direction: None,
        })
    } else {
        None
    };
    Ok(Outcome {
        improved,
        completion,
        rotatable: false,
    })
}
