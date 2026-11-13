//! Actual controls for sparse interior-point iterations, not aliases for
//! unimplemented MATLAB LP algorithms.
use super::*;
use runmat_builtins::{OptimDisplay, OptimOptionField, OptimSolver};

pub(super) struct Controls {
    max_iterations: usize,
    iterations: usize,
    display: OptimDisplay,
}
impl Default for Controls {
    fn default() -> Self {
        Self {
            max_iterations: 100_000,
            iterations: 0,
            display: OptimDisplay::Off,
        }
    }
}
impl Controls {
    pub(super) fn iterations(&self) -> usize {
        self.iterations
    }
    pub(super) async fn parse(
        value: Option<&Value>,
        runtime: Option<&crate::context::RuntimeContext>,
        cooperation: &mut ScalarCooperation,
    ) -> BuiltinResult<Self> {
        let source = match value {
            None => return Ok(Self::default()),
            Some(Value::Tensor(value)) if value.is_empty() => return Ok(Self::default()),
            Some(Value::Struct(value)) => value,
            _ => {
                return Err(linprog_error_with_detail(
                    &LINPROG_ERROR_INVALID_ARGUMENT,
                    "options must be a scalar structure or []",
                ))
            }
        };
        for (key, value) in &source.fields {
            if key.eq_ignore_ascii_case("Solver") {
                let solver = crate::builtins::math::optim::common::field_name(value)?;
                if OptimSolver::parse(&solver) != Some(OptimSolver::Linprog) {
                    return Err(linprog_error_with_detail(
                        &LINPROG_ERROR_INVALID_ARGUMENT,
                        "options belong to another solver",
                    ));
                }
            } else if !OptimOptionField::parse(key)
                .is_some_and(|field| OptimSolver::Linprog.accepts(field))
            {
                return Err(linprog_error_with_detail(
                    &LINPROG_ERROR_INVALID_ARGUMENT,
                    format!("unsupported linprog option {key}"),
                ));
            }
        }
        let source = crate::builtins::math::optim::common::prepare_solver_options(
            OptimSolver::Linprog,
            Some(source.clone()),
            Some(&runmat_builtins::LINPROG_INTEGER_OPTION_EXTENSION),
            Some(&LINPROG_GPU_INPUT_EXTENSION),
            runtime,
            cooperation,
        )
        .await?;
        let fields = crate::builtins::math::optim::common::ScalarOptionFields::from_struct(
            source.as_ref().map(|source| source.structure()),
        );
        let max_iterations = fields.count(NAME, OptimOptionField::MaxIter, 100_000)?;
        u32::try_from(max_iterations).map_err(|_| {
            linprog_error_with_detail(
                &LINPROG_ERROR_INVALID_ARGUMENT,
                "MaxIterations exceeds the backend's exact 32-bit iteration range",
            )
        })?;
        let display = fields.text(OptimOptionField::Display, "off")?;
        let display = OptimDisplay::parse(&display)
            .filter(|value| OptimSolver::Linprog.accepts_display(*value))
            .ok_or_else(|| {
                linprog_error_with_detail(
                    &LINPROG_ERROR_INVALID_ARGUMENT,
                    "unsupported Display value",
                )
            })?;
        Ok(Self {
            max_iterations,
            display,
            iterations: 0,
        })
    }
    pub(super) fn remaining(&self) -> usize {
        self.max_iterations.saturating_sub(self.iterations)
    }
    pub(super) fn backend_limit(&self) -> BuiltinResult<u32> {
        u32::try_from(self.remaining()).map_err(|_| {
            linprog_error_with_detail(
                &LINPROG_ERROR_INVALID_ARGUMENT,
                "MaxIterations exceeds the backend's exact 32-bit iteration range",
            )
        })
    }
    pub(super) fn record_iteration(
        &mut self,
        previous: &mut u32,
        current: u32,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<()> {
        let delta = current.checked_sub(*previous).ok_or_else(|| {
            linprog_error_with_detail(
                &LINPROG_ERROR_INVALID_INPUT,
                "backend iteration counter moved backwards",
            )
        })? as usize;
        if delta > self.remaining() {
            return Err(linprog_error_with_detail(
                &LINPROG_ERROR_INVALID_INPUT,
                "backend exceeded the original invocation iteration limit",
            ));
        }
        self.iterations += delta;
        *previous = current;
        if self.display == OptimDisplay::Iter {
            let _entry = runtime.map(crate::context::RuntimeContext::enter);
            crate::console::record_console_line(
                crate::console::ConsoleStream::Stdout,
                format!("linprog: interior-point iteration {}", self.iterations),
            )?;
        }
        Ok(())
    }
    pub(super) fn summary(
        &self,
        result: &LinprogOutcome,
        runtime: Option<&crate::context::RuntimeContext>,
    ) -> BuiltinResult<()> {
        if matches!(self.display, OptimDisplay::Final | OptimDisplay::Iter) {
            let _entry = runtime.map(crate::context::RuntimeContext::enter);
            crate::console::record_console_line(
                crate::console::ConsoleStream::Stdout,
                format!(
                    "linprog: {} Iterations: {}.",
                    result.message, self.iterations
                ),
            )?;
        }
        Ok(())
    }
}
