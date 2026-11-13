//! Canonical controls and effective algorithm-specific operand applicability.
use super::*;
use runmat_builtins::{OptimAlgorithm, OptimDisplay, OptimOptionField as Field, OptimSolver};
mod active;
mod input;
mod reflective;
pub(super) use reflective::{Method, Reflective};
pub(super) struct Controls {
    pub optimality: f64,
    pub algorithm: OptimAlgorithm,
    pub reflective: Option<Reflective>,
    pub active: Option<active::Active>,
    pub feasibility: f64,
    limit: u32,
    iterations: u32,
    display: OptimDisplay,
}
impl Controls {
    pub(super) fn warm_carriers(value: &Value) -> BuiltinResult<()> {
        let source = input::Input::borrowed(Some(value))?;
        if let Some(value) = source.as_ref() {
            for (name, value) in &value.fields {
                if Field::parse_for_solver(name, OptimSolver::Lsqlin)
                    .is_some_and(|field| reflective::active(field, OptimAlgorithm::ActiveSet))
                    && matches!(value, Value::GpuTensor(_))
                {
                    return Err(invalid(
                        "warm-start options require host active control carriers",
                    ));
                }
            }
        }
        Ok(())
    }
    pub(super) fn selected(
        value: &Value,
        linear_constraints: bool,
    ) -> BuiltinResult<OptimAlgorithm> {
        let source = input::Input::borrowed(Some(value))?;
        reflective::selected_input(source.as_ref(), linear_constraints)
    }
    pub(super) fn preflight_container(value: &Value) -> BuiltinResult<()> {
        input::Input::borrowed(Some(value)).map(|_| ())
    }
    pub async fn parse(
        value: Value,
        linear_constraints: bool,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        Self::parse_input(input::Input::owned(value)?, linear_constraints, cooperation).await
    }
    pub(super) async fn parse_borrowed(
        value: Option<&Value>,
        linear_constraints: bool,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        Self::parse_input(
            input::Input::borrowed(value)?,
            linear_constraints,
            cooperation,
        )
        .await
    }
    async fn parse_input(
        source: input::Input<'_>,
        linear_constraints: bool,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let value = source.as_ref();
        let algorithm = reflective::selected_input(value, linear_constraints)?;
        if let Some(value) = &value {
            for (name, value) in &value.fields {
                let field = Field::parse_for_solver(name, OptimSolver::Lsqlin);
                let solver = name.eq_ignore_ascii_case("Solver");
                if field.is_some_and(|field| !reflective::active(field, algorithm)) {
                    continue;
                }
                let text = solver
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
                if solver {
                    if OptimSolver::parse(&crate::builtins::math::optim::common::field_name(value)?)
                        != Some(OptimSolver::Lsqlin)
                    {
                        return Err(invalid("options belong to another solver"));
                    }
                } else if field.is_none() {
                    return Err(invalid(format!("unsupported lsqlin option {name}")));
                } else if let Some(field) = field {
                    if matches!(field, Field::TolFun | Field::TolCon | Field::MaxIter) {
                        if matches!(value, Value::Tensor(value) if value.is_empty()) {
                            continue;
                        }
                        crate::builtins::math::optim::common::preflight_option_scalar(
                            NAME, value, field,
                        )?;
                        numeric::compatibility(
                            value,
                            &runmat_builtins::LSQLIN_INTEGER_OPTION_EXTENSION,
                            cooperation.runtime(),
                            numeric::CompatibilityRole::Control,
                        )?;
                    }
                }
            }
        }
        if algorithm == OptimAlgorithm::InteriorPoint
            && crate::builtins::math::optim::common::ScalarOptionFields::from_struct(value)
                .exact(Field::TolX)
                .is_some()
        {
            return Err(invalid(
                "StepTolerance is not yet implemented for interior-point lsqlin",
            ));
        }
        let method = reflective::select(value, algorithm, cooperation).await?;
        let runtime = cooperation.runtime().cloned();
        let prepared = crate::builtins::math::optim::common::prepare_solver_options(
            OptimSolver::Lsqlin,
            source.into_active_owned(algorithm, method),
            Some(&runmat_builtins::LSQLIN_INTEGER_OPTION_EXTENSION),
            None,
            runtime.as_ref(),
            cooperation,
        )
        .await?;
        let fields = crate::builtins::math::optim::common::ScalarOptionFields::from_struct(
            prepared.as_ref().map(|v| v.structure()),
        );
        let trust = algorithm == OptimAlgorithm::TrustRegionReflective;
        let default = if trust { 100.0 * f64::EPSILON } else { 1e-8 };
        let legacy = fields.real(NAME, Field::TolFun, default)?;
        let optimality = fields.real(NAME, Field::OptimalityTolerance, legacy)?;
        let feasibility = fields.real(NAME, Field::TolCon, 1e-8)?;
        if !optimality.is_finite()
            || optimality < 0.
            || (!trust && optimality == 0.)
            || feasibility <= 0.
        {
            return Err(invalid("tolerances must be finite; interior-point requires positive optimality and constraint tolerances"));
        }
        let limit = u32::try_from(fields.count(
            NAME,
            Field::MaxIter,
            if algorithm == OptimAlgorithm::ActiveSet {
                2000
            } else {
                200
            },
        )?)
        .map_err(|_| invalid("MaxIterations must fit u32"))?;
        let display = OptimDisplay::parse(&fields.text(Field::Display, "final")?)
            .filter(|v| OptimSolver::Lsqlin.accepts_display(*v))
            .filter(|v| {
                !trust
                    || matches!(
                        v,
                        OptimDisplay::Off | OptimDisplay::None | OptimDisplay::Final
                    )
            })
            .ok_or_else(|| invalid("unsupported Display value for selected lsqlin algorithm"))?;
        let reflective = if trust {
            Some(Reflective::parse(
                &fields,
                legacy,
                method.expect("selected reflective method"),
            )?)
        } else {
            None
        };
        let active = if algorithm == OptimAlgorithm::ActiveSet {
            Some(active::Active::parse(&fields)?)
        } else {
            None
        };
        Ok(Self {
            active,
            optimality,
            algorithm,
            reflective,
            feasibility,
            limit,
            iterations: 0,
            display,
        })
    }
    pub(super) fn advance(&mut self) -> BuiltinResult<()> {
        if self.remaining() == 0 {
            return Err(invalid(
                "selected lsqlin controller exceeded iteration limit",
            ));
        }
        self.iterations += 1;
        Ok(())
    }
    pub fn remaining(&self) -> u32 {
        self.limit - self.iterations
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
        self.record_count(previous, info)?;
        if self.display == OptimDisplay::Iter {
            crate::context::with_original_scope(runtime, || {
                crate::console::record_console_line(
                    crate::console::ConsoleStream::Stdout,
                    format!(
                        "lsqlin {:5} objective {:.10e} primal {:.3e} dual {:.3e} gap {:.3e}",
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
    fn record_count(
        &mut self,
        previous: &mut u32,
        info: &clarabel::solver::DefaultInfo<f64>,
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
        Ok(())
    }
    pub(super) fn record_phase(
        &mut self,
        previous: &mut u32,
        info: &clarabel::solver::DefaultInfo<f64>,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<()> {
        self.record_count(previous, info)?;
        if self.display == OptimDisplay::Iter {
            crate::context::with_original_scope(runtime, || {
                crate::console::record_console_line(
                    crate::console::ConsoleStream::Stdout,
                    format!(
                        "lsqlin phase-I {:5} quadratic {:.10e} primal {:.3e}",
                        self.iterations, info.cost_primal, info.res_primal
                    ),
                )
            })?;
        }
        Ok(())
    }
    pub(super) fn record_active(
        &self,
        resnorm: f64,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<()> {
        if self.display == OptimDisplay::Iter {
            crate::context::with_original_scope(runtime, || {
                crate::console::record_console_line(
                    crate::console::ConsoleStream::Stdout,
                    format!(
                        "lsqlin active-set {:5} squared residual {:.10e}",
                        self.iterations, resnorm
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
                        "lsqlin: {} Iterations: {}.",
                        outcome.message, outcome.iterations
                    ),
                )
            })?;
        }
        Ok(())
    }
}

impl conic::solve::IterationControl for Controls {
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
