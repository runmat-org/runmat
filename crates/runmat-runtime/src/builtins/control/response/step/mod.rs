//! Pure, host-side step-response simulation shared by step and stepinfo.
//! Admission and provider ownership stay with the calling builtin.

use runmat_builtins::{
    BuiltinErrorDescriptor, STEPINFO_ERROR_INTERNAL, STEPINFO_ERROR_INVALID_DATA,
    STEPINFO_ERROR_INVALID_SYSTEM, STEPINFO_ERROR_UNSUPPORTED_MODEL, STEP_ERROR_DISCRETE_LIMIT,
    STEP_ERROR_INTERNAL, STEP_ERROR_INVALID_MODEL, STEP_ERROR_INVALID_TIME,
    STEP_ERROR_UNSUPPORTED_MODEL,
};
use runmat_value::Value;

use crate::builtins::control::tf_model::RealTfModel;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const EPS: f64 = 1.0e-12;
pub(crate) const DEFAULT_SAMPLES: usize = 101;
pub(crate) const MAX_DISCRETE_SAMPLES: usize = 1_000_000;

mod solver;
mod time;

pub(crate) use solver::StepResponse;
pub(crate) use time::TimeSpec;

#[derive(Clone, Copy, Debug)]
pub(crate) enum StepSimulationCaller {
    Step,
    StepInfo,
}

impl StepSimulationCaller {
    fn name(self) -> &'static str {
        match self {
            Self::Step => "step",
            Self::StepInfo => "stepinfo",
        }
    }

    fn descriptor(self, kind: SimulationErrorKind) -> &'static BuiltinErrorDescriptor {
        match self {
            Self::Step => match kind {
                SimulationErrorKind::InvalidModel => &STEP_ERROR_INVALID_MODEL,
                SimulationErrorKind::InvalidTime => &STEP_ERROR_INVALID_TIME,
                SimulationErrorKind::UnsupportedModel => &STEP_ERROR_UNSUPPORTED_MODEL,
                SimulationErrorKind::DiscreteLimit => &STEP_ERROR_DISCRETE_LIMIT,
                SimulationErrorKind::Internal => &STEP_ERROR_INTERNAL,
            },
            Self::StepInfo => match kind {
                SimulationErrorKind::InvalidModel => &STEPINFO_ERROR_INVALID_SYSTEM,
                SimulationErrorKind::InvalidTime => &STEPINFO_ERROR_INVALID_DATA,
                SimulationErrorKind::UnsupportedModel => &STEPINFO_ERROR_UNSUPPORTED_MODEL,
                SimulationErrorKind::DiscreteLimit | SimulationErrorKind::Internal => {
                    &STEPINFO_ERROR_INTERNAL
                }
            },
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum SimulationErrorKind {
    InvalidModel,
    InvalidTime,
    UnsupportedModel,
    DiscreteLimit,
    Internal,
}

enum SimulationError {
    Domain(SimulationErrorKind, String),
}

type SimulationResult<T> = Result<T, SimulationError>;

fn simulation_error(kind: SimulationErrorKind, detail: impl ToString) -> SimulationError {
    SimulationError::Domain(kind, detail.to_string())
}

impl SimulationError {
    fn into_runtime(self, caller: StepSimulationCaller) -> RuntimeError {
        match self {
            Self::Domain(kind, detail) => {
                let descriptor = caller.descriptor(kind);
                let mut builder = build_runtime_error(format!("{}: {detail}", descriptor.message))
                    .with_builtin(caller.name());
                if let Some(identifier) = descriptor.identifier {
                    builder = builder.with_identifier(identifier);
                }
                builder.build()
            }
        }
    }
}

/// Parse host-owned time after the caller has completed compatibility and
/// residency admission. Grid alignment depends on the admitted model.
pub(crate) fn admit_time(
    value: Option<&Value>,
    sample_time: f64,
    caller: StepSimulationCaller,
) -> BuiltinResult<TimeSpec> {
    TimeSpec::parse(value, sample_time).map_err(|error| error.into_runtime(caller))
}

/// This typed simulation boundary performs no downloads, compatibility
/// checks, raw-value parsing, or graphics side effects.
pub(crate) fn simulate(
    model: &RealTfModel,
    time: TimeSpec,
    caller: StepSimulationCaller,
) -> BuiltinResult<StepResponse> {
    model.ensure_zero_delays(caller.name())?;
    if model.sample_time() < 0.0 {
        return Err(simulation_error(
            SimulationErrorKind::InvalidModel,
            "step response requires a known nonnegative sample time",
        )
        .into_runtime(caller));
    }
    if model.numerator.len().saturating_sub(1) > model.denominator.len().saturating_sub(1) {
        return Err(simulation_error(
            SimulationErrorKind::UnsupportedModel,
            "improper transfer functions are not supported yet",
        )
        .into_runtime(caller));
    }
    solver::evaluate_step(model, time, caller)
}

pub(crate) fn preflight_host_time(
    value: Option<&Value>,
    caller: StepSimulationCaller,
) -> BuiltinResult<()> {
    TimeSpec::preflight_host(value).map_err(|error| error.into_runtime(caller))
}
