//! Pure, host-side step-response simulation shared by step and stepinfo.
//! Admission and provider ownership stay with the calling builtin.

use runmat_builtins::{
    BuiltinErrorDescriptor, STEP_ERROR_DISCRETE_LIMIT, STEP_ERROR_INTERNAL,
    STEP_ERROR_INVALID_MODEL, STEP_ERROR_INVALID_TIME, STEP_ERROR_UNSUPPORTED_MODEL,
};
use runmat_value::Value;

use crate::builtins::control::tf_model::RealTfModel;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const EPS: f64 = 1.0e-12;
pub(crate) const DEFAULT_SAMPLES: usize = 101;
const MAX_DISCRETE_SAMPLES: usize = 1_000_000;

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

    fn descriptor(self, kind: SimulationErrorKind) -> Option<&'static BuiltinErrorDescriptor> {
        match self {
            Self::Step => Some(match kind {
                SimulationErrorKind::InvalidModel => &STEP_ERROR_INVALID_MODEL,
                SimulationErrorKind::InvalidTime => &STEP_ERROR_INVALID_TIME,
                SimulationErrorKind::UnsupportedModel => &STEP_ERROR_UNSUPPORTED_MODEL,
                SimulationErrorKind::DiscreteLimit => &STEP_ERROR_DISCRETE_LIMIT,
                SimulationErrorKind::Internal => &STEP_ERROR_INTERNAL,
            }),
            Self::StepInfo => None,
        }
    }

    fn identifier(self, kind: SimulationErrorKind) -> &'static str {
        match kind {
            SimulationErrorKind::InvalidModel => "RunMat:stepinfo:InvalidSystem",
            SimulationErrorKind::InvalidTime => "RunMat:stepinfo:InvalidData",
            SimulationErrorKind::UnsupportedModel => "RunMat:stepinfo:UnsupportedModel",
            SimulationErrorKind::DiscreteLimit | SimulationErrorKind::Internal => {
                "RunMat:stepinfo:Internal"
            }
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
    Model(RuntimeError),
}

type SimulationResult<T> = Result<T, SimulationError>;

fn simulation_error(kind: SimulationErrorKind, detail: impl ToString) -> SimulationError {
    SimulationError::Domain(kind, detail.to_string())
}

impl SimulationError {
    fn into_runtime(self, caller: StepSimulationCaller) -> RuntimeError {
        match self {
            Self::Model(error) => error,
            Self::Domain(kind, detail) => {
                if let Some(descriptor) = caller.descriptor(kind) {
                    let mut builder =
                        build_runtime_error(format!("{}: {detail}", descriptor.message))
                            .with_builtin(caller.name());
                    if let Some(identifier) = descriptor.identifier {
                        builder = builder.with_identifier(identifier);
                    }
                    builder.build()
                } else {
                    build_runtime_error(format!("{}: {detail}", caller.name()))
                        .with_builtin(caller.name())
                        .with_identifier(caller.identifier(kind))
                        .build()
                }
            }
        }
    }
}

/// `time` is host-owned and already admitted by the caller. This function
/// performs no downloads, compatibility checks, or graphics side effects.
pub(crate) fn simulate(
    model: &RealTfModel,
    time: Option<&Value>,
    caller: StepSimulationCaller,
) -> BuiltinResult<StepResponse> {
    let result = (|| {
        if model.sample_time < 0.0 {
            return Err(simulation_error(
                SimulationErrorKind::InvalidModel,
                "step response requires a known nonnegative sample time",
            ));
        }
        if model.numerator.len().saturating_sub(1) > model.denominator.len().saturating_sub(1) {
            return Err(simulation_error(
                SimulationErrorKind::UnsupportedModel,
                "improper transfer functions are not supported yet",
            ));
        }
        let time = TimeSpec::parse(time, model.sample_time)?;
        solver::evaluate_step(model, time, caller)
    })();
    result.map_err(|error| error.into_runtime(caller))
}

pub(crate) fn preflight_host_time(
    value: Option<&Value>,
    caller: StepSimulationCaller,
) -> BuiltinResult<()> {
    TimeSpec::preflight_host(value).map_err(|error| error.into_runtime(caller))
}
