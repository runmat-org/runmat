//! Public fminsearch adapter to the shared Nelder-Mead engine.
use super::*;
use common::nelder_mead as nm;
fn policy() -> nm::Policy {
    nm::Policy {
        identity: FMINSEARCH_CATALOG_ENTRY.identity,
        invalid: &FMINSEARCH_ERROR_INVALID_ARGUMENT,
        allocation_identifier: "RunMat:fminsearch:AllocationFailed",
        retry: crate::GpuGatherRetry::Legacy,
    }
}
pub(super) fn geometry(n: usize) -> BuiltinResult<usize> {
    nm::geometry(policy(), n)
}
pub(super) struct Result {
    pub point: nm::Point,
    pub reason: nm::Reason,
    pub iterations: usize,
    pub func_count: usize,
    pub message: String,
}
pub(in crate::builtins::math::optim) struct Notifications<'a, 'b> {
    pub observers: &'a mut observers::Observers<'b>,
    pub display: runmat_builtins::OptimDisplay,
}
impl nm::Events for Notifications<'_, '_> {
    async fn notify(
        &mut self,
        phase: common::output_callback::OutputState,
        point: &nm::Point,
        iterations: usize,
        calls: usize,
        procedure: nm::Procedure,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<bool> {
        match phase {
            common::output_callback::OutputState::Init => {
                reporting::initial(self.display, point, calls, cooperation.runtime())?
            }
            common::output_callback::OutputState::Iter => reporting::iteration(
                self.display,
                point,
                iterations,
                calls,
                procedure,
                cooperation.runtime(),
            )?,
            _ => {}
        }
        self.observers
            .notify(phase, point, iterations, calls, procedure, cooperation)
            .await
    }
}
impl nm::Oracle for evaluation::Evaluator<'_> {
    fn cooperation(&mut self) -> &mut ScalarCooperation {
        self.cooperation
    }
    fn calls(&self) -> usize {
        self.calls
    }
    fn remaining(&self) -> usize {
        evaluation::Evaluator::remaining(self)
    }
    async fn measure(&mut self, x: &[f64]) -> BuiltinResult<nm::Measurement> {
        Ok(match evaluation::Evaluator::measure(self, x).await? {
            Some(f) => nm::Measurement::Defined(f),
            None => nm::Measurement::Stopped(nm::OracleStop::Evaluations),
        })
    }
}
pub(super) async fn minimize(
    x0: Vec<f64>,
    mut evaluator: evaluation::Evaluator<'_>,
    options: &options::Options<'_>,
    observers: &mut observers::Observers<'_>,
) -> BuiltinResult<Result> {
    let controls = nm::Controls {
        tol_x: options.tol_x,
        tol_fun: options.tol_fun,
        max_iter: options.max_iter,
    };
    let mut events = Notifications {
        observers,
        display: options.display,
    };
    let result = nm::minimize(
        policy(),
        nm::Initial::Evaluate(x0),
        &mut evaluator,
        &controls,
        &mut events,
    )
    .await?;
    let message = reporting::finish(
        options.display,
        result.reason,
        result.iterations,
        result.func_count,
        evaluator.cooperation.runtime(),
    )?;
    Ok(Result {
        point: result.point,
        reason: result.reason,
        iterations: result.iterations,
        func_count: result.func_count,
        message,
    })
}

pub(in crate::builtins::math::optim) fn finish_search(
    display: runmat_builtins::OptimDisplay,
    result: &nm::Outcome,
    cooperation: &ScalarCooperation,
) -> BuiltinResult<()> {
    reporting::finish(
        display,
        result.reason,
        result.iterations,
        result.func_count,
        cooperation.runtime(),
    )
    .map(|_| ())
}
