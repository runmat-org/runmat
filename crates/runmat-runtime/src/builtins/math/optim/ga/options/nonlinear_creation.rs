//! The documented nonlinear-creation parameters are one closed grammar;
//! arbitrary custom callback extras remain outside this parser.
use super::{admission, common, Field, Operator, Owner, ScalarCooperation};
use crate::{context::RuntimeContext, execution::CallbackParallelMode, BuiltinResult};
use runmat_builtins::OptimGeneticCreation;
use runmat_value::Value;

#[derive(Clone, Copy)]
enum Parameter {
    Solver,
    Parallel,
    Starts,
}
impl Parameter {
    fn parse(value: &Value) -> Option<Self> {
        let name = crate::builtins::common::validation::value_text_scalar_string(value)?;
        if name.eq_ignore_ascii_case("SolverOpts") {
            Some(Self::Solver)
        } else if name.eq_ignore_ascii_case("UseParallel") {
            Some(Self::Parallel)
        } else if name.eq_ignore_ascii_case("NumStartPts") {
            Some(Self::Starts)
        } else {
            None
        }
    }
}
#[derive(Default)]
struct Raw<'a> {
    solver: Option<&'a Value>,
    parallel: Option<&'a Value>,
    starts: Option<&'a Value>,
}
impl<'a> Raw<'a> {
    fn parse(owner: Owner, extra: &'a [Value]) -> BuiltinResult<Self> {
        if extra.len() > 6 || !extra.len().is_multiple_of(2) {
            return Err(owner.invalid("nonlinear creation accepts only SolverOpts, UseParallel and NumStartPts name/value pairs"));
        }
        let mut result = Self::default();
        for pair in extra.chunks_exact(2) {
            let slot = match Parameter::parse(&pair[0])
                .ok_or_else(|| owner.invalid("unknown nonlinear creation parameter"))?
            {
                Parameter::Solver => &mut result.solver,
                Parameter::Parallel => &mut result.parallel,
                Parameter::Starts => &mut result.starts,
            };
            if slot.replace(&pair[1]).is_some() {
                return Err(owner.invalid("duplicate nonlinear creation parameter"));
            }
        }
        Ok(result)
    }
    fn validate(&self, owner: Owner) -> BuiltinResult<()> {
        if let Some(value) = self.solver {
            if !matches!(value, Value::Struct(_)) && !super::super::inputs::raw::unspecified(value)
            {
                return Err(
                    owner.invalid("SolverOpts must be a scalar fmincon options structure or []")
                );
            }
        }
        if let Some(value) = self.parallel {
            if super::super::inputs::raw::unspecified(value) {
                return Err(owner.invalid("UseParallel requires off/auto/on or logical scalar"));
            }
            if !matches!(value, Value::GpuTensor(_)) {
                common::quasi_newton_controls::parallel_mode(owner.name(), value)?;
            }
        }
        if let Some(value) = self.starts {
            if super::super::inputs::raw::unspecified(value) {
                return Err(owner.invalid("NumStartPts must be positive"));
            }
            if !matches!(value, Value::GpuTensor(_))
                && common::option_usize_value(owner.name(), Some(value), "NumStartPts", 0)? == 0
            {
                return Err(owner.invalid("NumStartPts must be positive"));
            }
        }
        Ok(())
    }
    fn preflight(&self, owner: Owner, runtime: Option<&RuntimeContext>) -> BuiltinResult<()> {
        self.validate(owner)?;
        if let Some(value) = self.parallel {
            admission::metadata(owner, Field::UseParallel, value, runtime)?;
        }
        if let Some(value) = self.starts {
            admission::metadata(owner, Field::MaxIter, value, runtime)?;
        }
        Ok(())
    }
}
pub(super) fn validate(owner: Owner, extra: &[Value]) -> BuiltinResult<()> {
    Raw::parse(owner, extra)?.validate(owner)
}
pub(super) fn preflight(
    owner: Owner,
    value: &Value,
    runtime: Option<&RuntimeContext>,
) -> BuiltinResult<()> {
    let operator = Operator::parse(value, OptimGeneticCreation::parse)?;
    if operator.kind() == Some(OptimGeneticCreation::NonlinearFeasible) {
        Raw::parse(owner, operator.extra())?.preflight(owner, runtime)?;
    }
    Ok(())
}

pub(in super::super) struct Prepared {
    pub solver: super::super::super::fmincon::feasibility::Prepared,
    pub parallel: CallbackParallelMode,
    pub starts: Option<usize>,
}
pub(in super::super) async fn default(
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Prepared> {
    Ok(Prepared {
        solver: super::super::super::fmincon::feasibility::prepare_options(
            super::super::inputs::raw::empty(),
            cooperation,
        )
        .await?,
        parallel: CallbackParallelMode::Off,
        starts: None,
    })
}
pub(super) async fn prepare(
    value: Option<&Value>,
    cooperation: &mut ScalarCooperation,
) -> BuiltinResult<Option<Prepared>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let operator = Operator::parse(value, OptimGeneticCreation::parse)?;
    if operator.kind() != Some(OptimGeneticCreation::NonlinearFeasible) {
        return Ok(None);
    }
    let raw = Raw::parse(Owner::Solver, operator.extra())?;
    raw.preflight(Owner::Solver, cooperation.runtime())?;
    // Prepare nested controls FIRST. Their canonical fmincon preflight owns
    // all nested class/intent gates before either nested or GA control reads.
    let solver = super::super::super::fmincon::feasibility::prepare_options(
        raw.solver
            .cloned()
            .unwrap_or_else(super::super::inputs::raw::empty),
        cooperation,
    )
    .await?;
    let parallel = if let Some(value) = raw.parallel {
        let value =
            crate::dispatcher::gather_if_needed_in_context_async(value, cooperation.runtime())
                .await?;
        common::quasi_newton_controls::parallel_mode("ga", &value)?
    } else {
        CallbackParallelMode::Off
    };
    let starts = if let Some(value) = raw.starts {
        let value =
            crate::dispatcher::gather_if_needed_in_context_async(value, cooperation.runtime())
                .await?;
        let count = common::option_usize_value("ga", Some(&value), "NumStartPts", 0)?;
        if count == 0 {
            return Err(Owner::Solver.invalid("NumStartPts must be positive"));
        }
        Some(count)
    } else {
        None
    };
    Ok(Some(Prepared {
        solver,
        parallel,
        starts,
    }))
}
