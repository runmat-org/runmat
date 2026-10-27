//! Invocation-wide limits; backend iterations do not masquerade as branch nodes.
use super::*;
use crate::context::RuntimeContext;
use common::conic::solve::IterationControl;
use common::solver_timing::SolverTiming;
use runmat_builtins::OptimDisplay;
use runmat_builtins::{OptimOptionField as Field, OptimSolver};
pub(super) struct Controls {
    pub absolute_gap: f64,
    pub relative_gap: f64,
    pub constraint_tolerance: f64,
    pub max_nodes: usize,
    pub max_feasible_points: Option<usize>,
    pub max_time: f64,
    pub cutoff: Option<num_rational::BigRational>,
    pub display: OptimDisplay,
    pub nodes: usize,
    pub feasible_points: usize,
    pub improving_points: usize,
    pub observers: observers::Observers,
    deadline_observed: bool,
    timing: SolverTiming,
    feasibility_limit: Option<common::global_time::TimeLimit>,
}
impl Controls {
    pub async fn parse(
        value: Value,
        runtime: &RuntimeContext,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let mut result = Self::defaults(runtime)?;
        let options = match value {
            Value::Struct(v) => Some(v),
            Value::Tensor(v) if v.is_empty() => None,
            _ => return Err(invalid("options must be a scalar options struct or []")),
        };
        if let Some(options) = &options {
            for (name, value) in &options.fields {
                if name.eq_ignore_ascii_case("Solver") {
                    if OptimSolver::parse(&common::field_name(value)?)
                        != Some(OptimSolver::Intlinprog)
                    {
                        return Err(invalid("options belong to another solver"));
                    }
                } else if Field::parse_for_solver(name, OptimSolver::Intlinprog).is_none()
                    && !matches!(value,Value::Tensor(v) if v.is_empty())
                {
                    return Err(invalid(format!("unsupported intlinprog option {name}")));
                } else if Field::parse(name).is_none() {
                    return Err(invalid(format!("unknown intlinprog option {name}")));
                }
                if let Some(field) = Field::parse_for_solver(name, OptimSolver::Intlinprog) {
                    if matches!(value,Value::Tensor(v) if v.is_empty()) {
                        continue;
                    }
                    match field {
                        Field::OutputFcn | Field::PlotFcn => {
                            common::output_callback::validate_for_builtin(
                                NAME,
                                value,
                                common::output_callback::milp_form(field),
                            )?
                        }
                        Field::Display => {
                            let text =
                                common::option_string_value(Some(value), field.name(), "iter")?;
                            if !OptimDisplay::parse(&text)
                                .is_some_and(|v| OptimSolver::Intlinprog.accepts_display(v))
                            {
                                return Err(invalid("unsupported Display value"));
                            }
                        }
                        _ => {
                            numeric::preflight(value, numeric::Role::Control, Some(runtime))?;
                            common::preflight_option_scalar(NAME, value, field)?;
                        }
                    }
                }
            }
        }
        let prepared = common::prepare_solver_options(
            OptimSolver::Intlinprog,
            options,
            Some(&runmat_builtins::INTLINPROG_INTEGER_EXTENSION),
            None,
            Some(runtime),
            cooperation,
        )
        .await?;
        let fields =
            common::ScalarOptionFields::from_struct(prepared.as_ref().map(|v| v.structure()));
        result.absolute_gap =
            fields.real(NAME, Field::AbsoluteGapTolerance, result.absolute_gap)?;
        result.relative_gap =
            fields.real(NAME, Field::RelativeGapTolerance, result.relative_gap)?;
        result.constraint_tolerance =
            fields.real(NAME, Field::TolCon, result.constraint_tolerance)?;
        result.max_nodes = fields.count(NAME, Field::MaxNodes, result.max_nodes)?;
        result.max_feasible_points = match fields.exact(Field::MaxFeasiblePoints) {
            None => None,
            Some(v) if positive_infinity(v) => None,
            Some(v) => Some(common::option_usize_value(
                NAME,
                Some(v),
                Field::MaxFeasiblePoints.name(),
                0,
            )?),
        };
        result.max_time = if fields.exact(Field::MaxTime).is_some_and(positive_infinity) {
            f64::INFINITY
        } else {
            fields.real(NAME, Field::MaxTime, result.max_time)?
        };
        result.cutoff = match fields.exact(Field::ObjectiveCutOff) {
            None => None,
            Some(v) if positive_infinity(v) => None,
            Some(v) => Some(
                num_rational::BigRational::from_float(common::option_f64_value(
                    NAME,
                    Some(v),
                    Field::ObjectiveCutOff.name(),
                    0.0,
                )?)
                .expect("finite option"),
            ),
        };
        result.display = OptimDisplay::parse(&fields.text(Field::Display, "iter")?)
            .filter(|v| OptimSolver::Intlinprog.accepts_display(*v))
            .ok_or_else(|| invalid("unsupported Display value"))?;
        result.observers = observers::Observers::new(prepared)?;
        Ok(result)
    }
    pub fn defaults(runtime: &RuntimeContext) -> BuiltinResult<Self> {
        Self::with_timing(SolverTiming::start(Some(runtime))?)
    }
    fn with_timing(timing: SolverTiming) -> BuiltinResult<Self> {
        Ok(Self {
            absolute_gap: 1e-6,
            relative_gap: 1e-4,
            constraint_tolerance: 1e-6,
            max_nodes: 10_000_000,
            max_feasible_points: None,
            max_time: 7200.0,
            cutoff: None,
            display: OptimDisplay::Iter,
            nodes: 0,
            feasible_points: 0,
            improving_points: 0,
            observers: observers::Observers::new(None)?,
            deadline_observed: false,
            timing,
            feasibility_limit: None,
        })
    }
    /// Internal feasibility work shares the caller's already selected clock
    /// and cooperation budget. It is not a fresh public intlinprog invocation.
    pub(super) fn feasibility(
        timing: SolverTiming,
        max_time: common::global_time::TimeLimit,
        tolerance: f64,
    ) -> BuiltinResult<Self> {
        let mut result = Self::with_timing(timing)?;
        result.absolute_gap = 0.0;
        result.relative_gap = 0.0;
        result.constraint_tolerance = tolerance.min(1e-6);
        result.max_nodes = usize::MAX;
        result.max_feasible_points = Some(1);
        result.max_time = f64::INFINITY;
        result.feasibility_limit = Some(max_time);
        result.display = OptimDisplay::Off;
        Ok(result)
    }

    pub(super) fn retune_feasibility(
        &mut self,
        maximum: common::global_time::TimeLimit,
        tolerance: f64,
    ) {
        self.constraint_tolerance = tolerance.min(1e-6);
        self.feasibility_limit = Some(maximum);
        // A deadline already observed during work is not erased by a callback.
    }

    pub(super) fn begin_feasibility_repair(&mut self) {
        self.feasible_points = 0;
        self.improving_points = 0;
    }

    pub fn elapsed(&self) -> f64 {
        self.timing.elapsed().as_secs_f64()
    }
    pub fn time_left(&self) -> f64 {
        (self.max_time - self.elapsed()).max(0.0)
    }
    pub fn stopped(&self) -> Option<engine::Stop> {
        if self.time_limit_reached() {
            Some(engine::Stop::Time)
        } else if self.nodes >= self.max_nodes {
            Some(engine::Stop::Nodes)
        } else if self
            .max_feasible_points
            .is_some_and(|n| self.feasible_points >= n)
        {
            Some(engine::Stop::FeasiblePoints)
        } else {
            None
        }
    }
    pub fn time_limit_reached(&self) -> bool {
        self.deadline_observed
            || self.time_left() == 0.0
            || self
                .feasibility_limit
                .is_some_and(|limit| limit.reached(self.timing.elapsed()))
    }
    pub fn iteration_line(&self, runtime: Option<&RuntimeContext>) -> BuiltinResult<()> {
        if self.display == OptimDisplay::Iter {
            let _scope = runtime.map(RuntimeContext::enter);
            crate::console::record_console_line(
                crate::console::ConsoleStream::Stdout,
                format!(
                    "intlinprog nodes {} feasible points {} elapsed {:.3}s",
                    self.nodes,
                    self.feasible_points,
                    self.elapsed()
                ),
            )?;
        }
        Ok(())
    }
}
fn positive_infinity(value: &Value) -> bool {
    matches!(value,Value::Num(v) if *v==f64::INFINITY)
        || matches!(value,Value::Tensor(v) if v.len()==1 && v.integer_storage().is_none()
            && crate::builtins::common::tensor::tensor_value_f64(v,0)==f64::INFINITY)
}
impl IterationControl for Controls {
    fn accuracy(&self) -> f64 {
        self.constraint_tolerance
    }
    // MaxNodes counts actual branch relaxations, not interior-point iterations.
    // The backend's finite u32 range remains a truthful numerical frontier.
    fn remaining(&self) -> u32 {
        u32::MAX
    }
    fn progress_control(&mut self) -> clarabel::solver::ProgressControl {
        if self.time_limit_reached() {
            self.deadline_observed = true;
            clarabel::solver::ProgressControl::Stop(clarabel::solver::ProgressStop::TimeLimit)
        } else {
            clarabel::solver::ProgressControl::Continue
        }
    }
    fn record(
        &mut self,
        _previous: &mut u32,
        _info: &clarabel::solver::DefaultInfo<f64>,
        _scale: f64,
        _runtime: Option<&RuntimeContext>,
    ) -> BuiltinResult<()> {
        Ok(())
    }
}
