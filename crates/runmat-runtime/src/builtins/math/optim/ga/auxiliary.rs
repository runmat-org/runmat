//! A borrowed subsearch enters the same genetic controller with its enclosing
//! oracle, original geometry and timing. No public solver call is involved.
use super::common::{global_time::TimeLimit, solver_timing::SolverTiming, ScalarCooperation};
use super::{
    callbacks::Sources,
    inputs::Problem,
    options::{Activity, Options, Prepared},
    POLICY,
};
use super::{evaluation::source::Source, linear::Geometry};
use crate::{execution::CallbackParallelMode, BuiltinResult};
use runmat_value::Value;
mod preparation;

pub(in crate::builtins::math::optim) fn metadata(
    value: &Value,
    runtime: Option<&crate::context::RuntimeContext>,
) -> BuiltinResult<()> {
    let population = Prepared::population(value)?;
    visit_fields(value, |field, value| {
        super::options::preflight_field(
            field,
            value,
            Activity::Linear,
            population,
            true,
            false,
            runtime,
        )?;
        Ok(())
    })
}
pub(in crate::builtins::math::optim) fn syntax(value: &Value) -> BuiltinResult<()> {
    visit_fields(value, |_, _| Ok(()))
}
fn visit_fields(
    value: &Value,
    mut visit: impl FnMut(runmat_builtins::OptimOptionField, &Value) -> BuiltinResult<()>,
) -> BuiltinResult<()> {
    if super::inputs::raw::unspecified(value) {
        return Ok(());
    }
    let Value::Struct(value) = value else {
        return Err(super::invalid("optionsGA must be a scalar structure or []"));
    };
    for (name, value) in &value.fields {
        if name.eq_ignore_ascii_case("Solver") {
            if runmat_builtins::OptimSolver::parse(&super::options::text(value)?)
                != Some(runmat_builtins::OptimSolver::Ga)
            {
                return Err(super::invalid("optionsGA belongs to another solver"));
            }
            continue;
        }
        let field = runmat_builtins::OptimOptionField::parse_for_solver(
            name,
            runmat_builtins::OptimSolver::Ga,
        )
        .ok_or_else(|| super::invalid(format!("unsupported ga option {name}")))?;
        visit(field, value)?;
    }
    Ok(())
}

pub(in crate::builtins::math::optim) struct PublicLinear<'a> {
    pub inequalities: Option<(&'a Value, &'a Value)>,
    pub equalities: Option<(&'a Value, &'a Value)>,
    pub lower: Option<&'a Value>,
    pub upper: Option<&'a Value>,
}
pub(in crate::builtins::math::optim) struct Controls {
    ready: super::initialization::Prepared,
    sources: Sources,
}
impl Controls {
    pub fn maximum(&self) -> TimeLimit {
        self.ready.options().maximum
    }
    pub fn tolerance(&self) -> f64 {
        self.ready.options().constraint_tolerance
    }
    pub fn prepare<'a>(
        fitness: Value,
        seed: &'a [f64],
        explicit: Option<&'a Value>,
        inherited: (bool, CallbackParallelMode),
        public: PublicLinear<'a>,
        bounds: (&'a [f64], &'a [f64]),
        has_linear: bool,
        cooperation: &'a mut ScalarCooperation,
    ) -> impl std::future::Future<Output = BuiltinResult<Self>> + 'a {
        preparation::prepare(preparation::Input {
            fitness,
            seed,
            explicit,
            inherited,
            public,
            bounds,
            has_linear,
            cooperation,
        })
    }
}
pub(in crate::builtins::math::optim) struct Completed<G, S> {
    finished: super::controller::Finished<G, std::ops::ControlFlow<S, super::outcome::Termination>>,
}
impl<G, S> Completed<G, S> {
    pub fn stopped(&self) -> Option<&S> {
        match &self.finished.reason {
            std::ops::ControlFlow::Break(stop) => Some(stop),
            std::ops::ControlFlow::Continue(_) => None,
        }
    }
}
pub(in crate::builtins::math::optim) fn run<'a, G: Geometry + 'a, S: Source + 'a>(
    controls: Controls,
    geometry: G,
    timing: SolverTiming,
    source: &'a mut S,
    cooperation: &'a mut ScalarCooperation,
) -> impl std::future::Future<Output = BuiltinResult<Completed<G, S::Stop>>> + 'a
where
    S::Stop: 'a,
{
    use futures::FutureExt;
    super::controller::run_borrowed(
        controls.ready,
        timing,
        controls.sources,
        super::creation::factory::Borrowed(geometry),
        source,
        cooperation,
    )
    .map(|result| result.map(|finished| Completed { finished }))
}
