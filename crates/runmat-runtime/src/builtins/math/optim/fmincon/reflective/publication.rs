//! Reflective diagnostics and original-rooted last-used H/Hinfo publication.
pub(super) use super::super::publication::ReflectiveInfo as Info;
use super::*;
use crate::builtins::math::optim::common::{
    output_callback::OutputState, solver_timing::SolverTiming,
};
use crate::execution::RootedValueSequence;
pub(in crate::builtins::math::optim::fmincon) struct Solved {
    pub(super) outcome: Outcome,
    pub(super) model: Option<model::Model>,
    pub(super) info: Info,
}

pub(super) async fn observe(
    out: &Outcome,
    parameters: &Parameters<'_>,
    info: Info,
    state: OutputState,
    direction: &[f64],
    evaluator: &mut Evaluator<'_>,
) -> BuiltinResult<bool> {
    if parameters.display == runmat_builtins::OptimDisplay::Iter
        || (state == OutputState::Done
            && parameters.display == runmat_builtins::OptimDisplay::Final)
    {
        let _entry = evaluator
            .cooperation
            .runtime()
            .map(crate::context::RuntimeContext::enter);
        crate::console::record_console_line(crate::console::ConsoleStream::Stdout,format!("fmincon trust-region-reflective: iteration={} funcCount={} fval={:?} constraintViolation={:?} firstOrderOptimality={:?} trustRadius={} cgIterations={}{}",out.iterations,evaluator.budget.counts().points,out.point.as_ref().map(|point|point.f),out.violation,out.optimality,info.radius,info.cg_iterations,if state==OutputState::Done {out.termination.description()}else{""}))?;
    }
    let shape = evaluator.shape.clone();
    let scalar = evaluator.scalar;
    super::super::observation::notify_reflective(
        parameters.callback,
        state,
        super::super::observation::Snapshot {
            x: &out.x,
            point: out.point.as_ref(),
            shape: &shape,
            scalar,
            iterations: out.iterations,
            counts: evaluator.budget.counts(),
            violation: out.violation,
            optimality: out.optimality,
            step: out.step,
            alpha: None,
            direction,
        },
        info,
        evaluator.cooperation,
    )
    .await
}
pub(in crate::builtins::math::optim::fmincon) async fn finalize(
    solved: Solved,
    geometry: &Geometry,
    shape: &[usize],
    scalar: bool,
    timing: Option<SolverTiming>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<RootedValueSequence> {
    super::super::publication::finalize_reflective(
        solved.outcome,
        geometry,
        shape,
        scalar,
        timing,
        solved.model.map(model::Model::into_value),
        solved.info,
        cooperation,
    )
    .await
}
