//! Typed consumed controls for the actual sparse interior-point algorithm.
use super::*;
use clarabel::solver::DefaultInfo;
use runmat_builtins::{OptimAlgorithm, OptimDisplay, OptimOptionField, OptimSolver};

pub(super) struct Controls {
    pub tol: f64,
    max_iter: usize,
    iterations: usize,
    display: OptimDisplay,
}
impl Default for Controls {
    fn default() -> Self {
        Self {
            tol: DEFAULT_TOL,
            max_iter: DEFAULT_MAX_ITER,
            iterations: 0,
            display: OptimDisplay::Off,
        }
    }
}
impl Controls {
    pub fn validate(options: &StructValue) -> BuiltinResult<()> {
        for (name, value) in &options.fields {
            if name.eq_ignore_ascii_case("Solver") {
                if OptimSolver::parse(&crate::builtins::math::optim::common::field_name(value)?)
                    != Some(OptimSolver::Coneprog)
                {
                    return Err(invalid("options belong to another solver"));
                }
            } else if !OptimOptionField::parse(name)
                .is_some_and(|field| OptimSolver::Coneprog.accepts(field))
            {
                return Err(invalid(&format!("unsupported option {name}")));
            }
        }
        Ok(())
    }
    pub async fn parse(
        value: Option<StructValue>,
        runtime: Option<&crate::context::RuntimeContext>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        if let Some(value) = &value {
            Self::validate(value)?;
        }
        let value = crate::builtins::math::optim::common::prepare_solver_options(
            OptimSolver::Coneprog,
            value,
            None,
            None,
            runtime,
            cooperation,
        )
        .await?;
        let fields = crate::builtins::math::optim::common::ScalarOptionFields::from_struct(
            value.as_ref().map(|value| value.structure()),
        );
        let tol = fields
            .real(CONEPROG, OptimOptionField::TolFun, DEFAULT_TOL)?
            .min(fields.real(CONEPROG, OptimOptionField::TolX, DEFAULT_TOL)?);
        if tol <= 0.0 {
            return Err(invalid(
                "TolFun and TolX must be positive finite tolerances",
            ));
        }
        let max_iter = fields.count(CONEPROG, OptimOptionField::MaxIter, DEFAULT_MAX_ITER)?;
        u32::try_from(max_iter).map_err(|_| {
            invalid("MaxIterations exceeds the backend's exact u32 iteration range")
        })?;
        let display = OptimDisplay::parse(&fields.text(OptimOptionField::Display, "off")?)
            .filter(|display| OptimSolver::Coneprog.accepts_display(*display))
            .ok_or_else(|| invalid("unsupported Display value"))?;
        // Both accepted public spellings select the implemented interior-point
        // family, not a fake switch to a different numerical algorithm.
        OptimAlgorithm::parse(&fields.text(OptimOptionField::Algorithm, "interior-point")?)
            .filter(|algorithm| OptimSolver::Coneprog.accepts_algorithm(*algorithm))
            .ok_or_else(|| invalid("unsupported Algorithm value"))?;
        Ok(Self {
            tol,
            max_iter,
            iterations: 0,
            display,
        })
    }
    pub fn iterations(&self) -> usize {
        self.iterations
    }
    pub fn remaining(&self) -> usize {
        self.max_iter.saturating_sub(self.iterations)
    }
    pub fn limit(&self) -> u32 {
        u32::try_from(self.remaining()).expect("admitted u32 iteration limit")
    }
    pub fn begin_phase(
        &self,
        feasibility: bool,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<()> {
        if self.display == OptimDisplay::Iter {
            let _entry = runtime.map(crate::context::RuntimeContext::enter);
            line(
                if feasibility {
                    "coneprog: auxiliary feasibility phase"
                } else {
                    "coneprog: optimization phase"
                }
                .into(),
            )?;
            line(
                "Iteration  Primal objective  Primal residual  Dual residual  Duality gap".into(),
            )?;
        }
        Ok(())
    }
    pub fn record(
        &mut self,
        previous: &mut u32,
        info: &DefaultInfo<f64>,
        scale: f64,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<()> {
        let delta = info
            .iterations
            .checked_sub(*previous)
            .ok_or_else(|| invalid("backend iteration counter moved backwards"))?
            as usize;
        if delta > self.remaining() {
            return Err(invalid(
                "backend exceeded the original invocation iteration limit",
            ));
        }
        self.iterations += delta;
        *previous = info.iterations;
        if self.display == OptimDisplay::Iter {
            let _entry = runtime.map(crate::context::RuntimeContext::enter);
            line(format!(
                "{:9}  {:16.8e}  {:15.6e}  {:13.6e}  {:11.6e}",
                self.iterations,
                info.cost_primal * scale,
                info.res_primal,
                info.res_dual,
                info.gap_abs * scale
            ))?;
        }
        Ok(())
    }
    pub fn summary(
        &self,
        outcome: &ConeOutcome,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<()> {
        if matches!(self.display, OptimDisplay::Iter | OptimDisplay::Final) {
            let _entry = runtime.map(crate::context::RuntimeContext::enter);
            line(format!(
                "coneprog: {} Iterations: {}.",
                outcome.message, self.iterations
            ))?;
        }
        Ok(())
    }
}
fn line(message: String) -> BuiltinResult<()> {
    crate::console::record_console_line(crate::console::ConsoleStream::Stdout, message)
}
fn invalid(detail: &str) -> RuntimeError {
    coneprog_error(&ERROR_INVALID_ARGUMENT, detail)
}
