//! Solver-specific coefficients admitted once per native-class iteration.

use crate::builtins::deep_learning::model::execution::host::numeric::Real;
use crate::BuiltinResult;

use super::super::super::{Solver, TrainingConfig};
use super::fail;

#[derive(Clone, Copy)]
pub(super) enum NativeControls<T> {
    Sgdm {
        momentum_complement: T,
    },
    Rmsprop {
        squared_complement: T,
        epsilon: T,
    },
    Adam {
        mean_complement: T,
        squared_complement: T,
        mean_correction: T,
        squared_correction: T,
        epsilon: T,
    },
}

fn unit(value: f64, function: &'static str) -> BuiltinResult<()> {
    if value.is_finite() && (0.0..1.0).contains(&value) {
        Ok(())
    } else {
        Err(fail(
            function,
            "optimizer decay or momentum must be in [0, 1)",
        ))
    }
}

fn complement<T: Real>(value: f64, function: &'static str) -> BuiltinResult<T> {
    unit(value, function)?;
    // Subtract before native conversion. A valid binary64 decay can round to
    // exactly one in binary32; `1 - native_decay` would then erase the update.
    let converted = T::from_f64(1.0 - value);
    if converted.is_finite() && converted > T::zero() {
        Ok(converted)
    } else {
        Err(fail(
            function,
            "optimizer decay complement is not representable in parameter precision",
        ))
    }
}

fn correction<T: Real>(decay: f64, iteration: usize, function: &'static str) -> BuiltinResult<T> {
    let remaining = 1.0 - decay;
    // 1 - decay^iteration loses the small first-step correction near one.
    let corrected = -((iteration as f64) * (-remaining).ln_1p()).exp_m1();
    let native = T::from_f64(corrected);
    if native.is_finite() && native > T::zero() {
        Ok(native)
    } else {
        Err(fail(function, "Adam bias correction is not representable"))
    }
}

fn epsilon<T: Real>(value: f64, function: &'static str) -> BuiltinResult<T> {
    let native = T::from_f64(value);
    if value.is_finite() && value > 0.0 && native.is_finite() && native > T::zero() {
        Ok(native)
    } else {
        Err(fail(
            function,
            "optimizer epsilon is not representable in parameter precision",
        ))
    }
}

impl<T: Real> NativeControls<T> {
    pub(super) fn admit(
        config: &TrainingConfig,
        iteration: usize,
        function: &'static str,
    ) -> BuiltinResult<Self> {
        if iteration == 0 {
            return Err(fail(function, "optimizer iteration must be positive"));
        }
        match config.solver {
            Solver::Sgdm => Ok(Self::Sgdm {
                momentum_complement: complement(config.momentum, function)?,
            }),
            Solver::Rmsprop => Ok(Self::Rmsprop {
                squared_complement: complement(config.squared_gradient_decay_factor, function)?,
                epsilon: epsilon(config.epsilon, function)?,
            }),
            Solver::Adam => Ok(Self::Adam {
                mean_complement: complement(config.gradient_decay_factor, function)?,
                squared_complement: complement(config.squared_gradient_decay_factor, function)?,
                mean_correction: correction(config.gradient_decay_factor, iteration, function)?,
                squared_correction: correction(
                    config.squared_gradient_decay_factor,
                    iteration,
                    function,
                )?,
                epsilon: epsilon(config.epsilon, function)?,
            }),
        }
    }
}
