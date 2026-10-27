//! Only controls consumed by the implemented quadratic interior-point solver.
use super::*;
use runmat_builtins::{OptimAlgorithm, OptimDisplay, OptimOptionField as Field, OptimSolver};
mod active;
pub(super) use active::Active;
pub(super) struct Controls {
    pub algorithm: OptimAlgorithm,
    pub active: Option<Active>,
    dynamic_limit: bool,
    pub optimality: f64,
    pub feasibility: f64,
    limit: u32,
    iterations: u32,
    display: OptimDisplay,
}
impl conic::solve::IterationControl for Controls {
    fn progress_control(&mut self) -> clarabel::solver::ProgressControl {
        if self.time_reached() {
            clarabel::solver::ProgressControl::Stop(clarabel::solver::ProgressStop::TimeLimit)
        } else {
            clarabel::solver::ProgressControl::Continue
        }
    }
    fn accuracy(&self) -> f64 {
        self.feasibility.min(self.optimality)
    }
    fn remaining(&self) -> u32 {
        Controls::remaining(self)
    }
    fn record(
        &mut self,
        previous: &mut u32,
        info: &clarabel::solver::DefaultInfo<f64>,
        scale: f64,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<()> {
        Controls::record(self, previous, info, scale, runtime)
    }
}
impl Controls {
    pub async fn parse(value: Value, cooperation: &mut ScalarCooperation) -> BuiltinResult<Self> {
        let value = match value {
            Value::Struct(value) => Some(value),
            Value::Tensor(value) if value.is_empty() => None,
            _ => return Err(invalid("options must be a scalar struct or []")),
        };
        Self::preflight(value.as_ref(), cooperation)?;
        Self::prepare(value, cooperation).await
    }
    pub(super) async fn parse_borrowed(
        value: Option<&Value>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let value = match value {
            None => None,
            Some(Value::Struct(value)) => Some(value),
            Some(Value::Tensor(value)) if value.is_empty() => None,
            _ => return Err(invalid("options must be a scalar struct or []")),
        };
        Self::preflight(value, cooperation)?;
        Self::prepare(value.cloned(), cooperation).await
    }
    fn preflight(
        value: Option<&runmat_value::StructValue>,
        cooperation: &ScalarCooperation,
    ) -> BuiltinResult<()> {
        let fields = crate::builtins::math::optim::common::ScalarOptionFields::from_struct(value);
        let algorithm = OptimAlgorithm::parse(&fields.text(Field::Algorithm, ALGORITHM)?)
            .filter(|a| OptimSolver::Quadprog.accepts_algorithm(*a))
            .ok_or_else(|| invalid("unsupported quadprog Algorithm"))?;
        if let Some(value) = value {
            for (name, value) in &value.fields {
                let field = Field::parse_for_solver(name, OptimSolver::Quadprog);
                let text = name.eq_ignore_ascii_case("Solver")
                    || field
                        .is_some_and(|field| matches!(field, Field::Display | Field::Algorithm));
                if text
                    && !matches!(value,Value::Tensor(value)if value.is_empty())
                    && !matches!(value, Value::String(_))
                    && !matches!(value,Value::StringArray(value)if value.data().len()==1)
                    && !matches!(value,Value::CharArray(value)if value.rows==1)
                {
                    return Err(invalid(
                        "text controls must be character rows or scalar strings",
                    ));
                }
                if name.eq_ignore_ascii_case("Solver") {
                    if OptimSolver::parse(&crate::builtins::math::optim::common::field_name(value)?)
                        != Some(OptimSolver::Quadprog)
                    {
                        return Err(invalid("options belong to another solver"));
                    }
                } else if Field::parse_for_solver(name, OptimSolver::Quadprog).is_none() {
                    return Err(invalid(format!("unsupported quadprog option {name}")));
                } else if let Some(field) = Field::parse_for_solver(name, OptimSolver::Quadprog) {
                    if matches!(field, Field::TolX | Field::ObjectiveLimit | Field::MaxTime)
                        && algorithm != OptimAlgorithm::ActiveSet
                        && !matches!(value, Value::Tensor(t) if t.is_empty())
                    {
                        return Err(invalid(
                            "StepTolerance, ObjectiveLimit and MaxTime require active-set",
                        ));
                    }
                    if field == Field::MaxTime
                        && crate::builtins::duration::is_duration_object(value)
                    {
                        active::time(value)?;
                        continue;
                    }
                    if matches!(
                        field,
                        Field::TolFun
                            | Field::TolCon
                            | Field::MaxIter
                            | Field::TolX
                            | Field::ObjectiveLimit
                            | Field::MaxTime
                    ) && !matches!(value,Value::Tensor(t) if t.is_empty())
                    {
                        crate::builtins::math::optim::common::preflight_option_scalar(
                            NAME, value, field,
                        )?;
                        numeric::preflight(
                            value,
                            &runmat_builtins::QUADPROG_INTEGER_OPTION_EXTENSION,
                            cooperation.runtime(),
                            numeric::CompatibilityRole::Control,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
    async fn prepare(
        value: Option<runmat_value::StructValue>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let runtime = cooperation.runtime().cloned();
        let prepared = crate::builtins::math::optim::common::prepare_solver_options(
            OptimSolver::Quadprog,
            value,
            Some(&runmat_builtins::QUADPROG_INTEGER_OPTION_EXTENSION),
            Some(&runmat_builtins::QUADPROG_GPU_INPUT_EXTENSION),
            runtime.as_ref(),
            cooperation,
        )
        .await?;
        let fields = crate::builtins::math::optim::common::ScalarOptionFields::from_struct(
            prepared.as_ref().map(|v| v.structure()),
        );
        let algorithm = OptimAlgorithm::parse(&fields.text(Field::Algorithm, ALGORITHM)?)
            .filter(|a| OptimSolver::Quadprog.accepts_algorithm(*a))
            .ok_or_else(|| invalid("unsupported quadprog Algorithm"))?;
        let active = if algorithm == OptimAlgorithm::ActiveSet {
            Some(Active::parse(&fields, cooperation).await?)
        } else {
            None
        };
        let optimality = fields.real(NAME, Field::TolFun, 1e-8)?;
        let feasibility = fields.real(NAME, Field::TolCon, 1e-8)?;
        if optimality < 0.0
            || feasibility < 0.0
            || (active.is_none() && (optimality == 0.0 || feasibility == 0.0))
        {
            return Err(invalid(
                "optimality and constraint tolerances must be positive and finite",
            ));
        }
        let limit = u32::try_from(fields.count(NAME, Field::MaxIter, 200)?)
            .map_err(|_| invalid("MaxIterations must fit u32"))?;
        let display = OptimDisplay::parse(&fields.text(Field::Display, "final")?)
            .filter(|v| OptimSolver::Quadprog.accepts_display(*v))
            .ok_or_else(|| invalid("unsupported Display value"))?;
        Ok(Self {
            algorithm,
            active,
            dynamic_limit: fields.exact(Field::MaxIter).is_none(),
            optimality,
            feasibility,
            limit,
            iterations: 0,
            display,
        })
    }
    pub fn remaining(&self) -> u32 {
        self.limit - self.iterations
    }
    pub fn resolve_limit(&mut self, variables: usize, rows: usize) -> BuiltinResult<()> {
        if self.active.is_some() && self.dynamic_limit {
            self.limit = u32::try_from(
                variables
                    .checked_add(rows)
                    .and_then(|v| v.checked_mul(10))
                    .ok_or_else(|| invalid("default active iteration limit overflow"))?,
            )
            .map_err(|_| invalid("default active iteration limit must fit u32"))?;
        }
        Ok(())
    }
    pub fn bind_timing(
        &mut self,
        timing: Option<&SolverTiming>,
        c: &ScalarCooperation,
    ) -> BuiltinResult<()> {
        if let Some(active) = &mut self.active {
            active.bind(timing, c)?;
        }
        Ok(())
    }
    pub fn time_reached(&self) -> bool {
        self.active.as_ref().is_some_and(Active::reached)
    }
    pub fn require_active_clock(&self) -> BuiltinResult<()> {
        if let Some(active) = &self.active {
            active.require_clock()?;
        }
        Ok(())
    }
    pub fn advance(&mut self) -> BuiltinResult<()> {
        if self.remaining() == 0 {
            return Err(invalid("active-set iteration exceeded original limit"));
        }
        self.iterations += 1;
        Ok(())
    }
    pub fn record_active(
        &self,
        cost: f64,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<()> {
        if self.display == OptimDisplay::Iter {
            crate::context::with_original_scope(runtime, || {
                crate::console::record_console_line(
                    crate::console::ConsoleStream::Stdout,
                    format!("quadprog {:5} objective {:.10e}", self.iterations, cost),
                )
            })?;
        }
        Ok(())
    }
    pub fn iterations(&self) -> usize {
        self.iterations as usize
    }
    pub fn record(
        &mut self,
        previous: &mut u32,
        info: &clarabel::solver::DefaultInfo<f64>,
        scale: f64,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<()> {
        let delta = info
            .iterations
            .checked_sub(*previous)
            .ok_or_else(|| invalid("backend iteration counter regressed"))?;
        if delta > self.remaining() {
            return Err(invalid("backend exceeded invocation iteration limit"));
        }
        self.iterations += delta;
        *previous = info.iterations;
        if self.display == OptimDisplay::Iter {
            crate::context::with_original_scope(runtime, || {
                crate::console::record_console_line(
                    crate::console::ConsoleStream::Stdout,
                    format!(
                        "quadprog {:5} objective {:.10e} primal {:.3e} dual {:.3e} gap {:.3e}",
                        self.iterations,
                        info.cost_primal * scale,
                        info.res_primal,
                        info.res_dual,
                        info.gap_abs * scale
                    ),
                )
            })?;
        }
        Ok(())
    }
    pub fn summary(
        &self,
        outcome: &Outcome,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<()> {
        if matches!(self.display, OptimDisplay::Final | OptimDisplay::Iter) {
            crate::context::with_original_scope(runtime, || {
                crate::console::record_console_line(
                    crate::console::ConsoleStream::Stdout,
                    format!(
                        "quadprog: {} Iterations: {}.",
                        outcome.message, outcome.iterations
                    ),
                )
            })?;
        }
        Ok(())
    }
}
