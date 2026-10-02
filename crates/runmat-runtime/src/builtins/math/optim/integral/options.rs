//! Typed option values and one ordered admission authority for `integral`.

use super::{
    integral_error_with_detail, IntegralCallbackMode, IntegralOptionName, DEFAULT_MAX_FUN_EVALS,
};
use crate::builtins::common::tensor;
use crate::BuiltinResult;
use runmat_builtins::INTEGRAL_ERROR_INVALID_ARGUMENT;
use runmat_value::{LogicalArray, Value};

#[derive(Clone, Copy)]
pub(super) struct IntegralOptions {
    pub(super) abs_tol: f64,
    pub(super) rel_tol: f64,
    pub(super) max_fun_evals: usize,
    pub(super) callback_mode: IntegralCallbackMode,
    pub(super) array_valued: bool,
}

impl IntegralOptions {
    fn defaults() -> Self {
        Self {
            abs_tol: IntegralOptionName::DEFAULT_ABS_TOL,
            rel_tol: IntegralOptionName::DEFAULT_REL_TOL,
            max_fun_evals: DEFAULT_MAX_FUN_EVALS,
            callback_mode: IntegralCallbackMode::Vectorized,
            array_valued: false,
        }
    }

    pub(super) fn preflight_host(rest: &[(IntegralOptionName, &Value)]) -> BuiltinResult<()> {
        let mut check = Self::defaults();
        let mut known_abs_tol = Some(check.abs_tol);
        let mut known_rel_tol = Some(check.rel_tol);
        for &(name, value) in rest {
            if matches!(value, Value::GpuTensor(_)) {
                match name {
                    IntegralOptionName::AbsTol => known_abs_tol = None,
                    IntegralOptionName::RelTol => known_rel_tol = None,
                    IntegralOptionName::MaxFunEvals
                    | IntegralOptionName::MaxIntervalCount
                    | IntegralOptionName::ArrayValued
                    | IntegralOptionName::Vectorized => {}
                }
                continue;
            }
            check.apply_option(name, value)?;
            match name {
                IntegralOptionName::AbsTol => known_abs_tol = Some(check.abs_tol),
                IntegralOptionName::RelTol => known_rel_tol = Some(check.rel_tol),
                IntegralOptionName::MaxFunEvals
                | IntegralOptionName::MaxIntervalCount
                | IntegralOptionName::ArrayValued
                | IntegralOptionName::Vectorized => {}
            }
        }
        if known_abs_tol == Some(0.0) && known_rel_tol == Some(0.0) {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_ARGUMENT,
                "AbsTol and RelTol cannot both be zero",
            ));
        }
        if known_abs_tol.is_some_and(|value| value < 0.0) {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_ARGUMENT,
                "AbsTol must be nonnegative",
            ));
        }
        if known_rel_tol.is_some_and(|value| value < 0.0) {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_ARGUMENT,
                "RelTol must be nonnegative",
            ));
        }
        Ok(())
    }

    pub(super) async fn parse(rest: &[(IntegralOptionName, &Value)]) -> BuiltinResult<Self> {
        let mut options = Self::defaults();
        for &(name, value) in rest {
            if matches!(value, Value::GpuTensor(_)) {
                let gathered = crate::dispatcher::gather_if_needed_async(value).await?;
                options.apply_option(name, &gathered)?;
            } else {
                options.apply_option(name, value)?;
            }
        }
        options.validate()?;
        Ok(options)
    }

    fn apply_option(&mut self, name: IntegralOptionName, value: &Value) -> BuiltinResult<()> {
        match name {
            IntegralOptionName::AbsTol => {
                self.abs_tol = numeric_option(name.display_name(), value)?
            }
            IntegralOptionName::RelTol => {
                self.rel_tol = numeric_option(name.display_name(), value)?
            }
            IntegralOptionName::MaxFunEvals | IntegralOptionName::MaxIntervalCount => {
                let parsed = integer_option(name.display_name(), value)?;
                if parsed < 5 {
                    return Err(integral_error_with_detail(
                        &INTEGRAL_ERROR_INVALID_ARGUMENT,
                        "MaxFunEvals must be an integer scalar >= 5",
                    ));
                }
                self.max_fun_evals = parsed;
            }
            IntegralOptionName::ArrayValued => {
                self.array_valued = bool_option("ArrayValued", value)?;
            }
            IntegralOptionName::Vectorized => {
                self.callback_mode =
                    IntegralCallbackMode::from_vectorized(bool_option("Vectorized", value)?);
            }
        }
        Ok(())
    }

    fn validate(&self) -> BuiltinResult<()> {
        if self.abs_tol < 0.0 {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_ARGUMENT,
                "AbsTol must be nonnegative",
            ));
        }
        if self.rel_tol < 0.0 {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_ARGUMENT,
                "RelTol must be nonnegative",
            ));
        }
        if self.abs_tol == 0.0 && self.rel_tol == 0.0 {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_ARGUMENT,
                "AbsTol and RelTol cannot both be zero",
            ));
        }
        Ok(())
    }
}

fn numeric_option(name: &str, value: &Value) -> BuiltinResult<f64> {
    if !crate::builtins::common::validation::native_integer_value_is_exact_f64(value) {
        return Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_ARGUMENT,
            format!("option {name} must be exactly representable as double"),
        ));
    }
    let parsed = match value {
        Value::Num(n) => *n,
        Value::Int(i) => i.to_f64(),
        Value::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            tensor::tensor_value_f64(tensor, 0)
        }
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => {
            if data[0] != 0 {
                1.0
            } else {
                0.0
            }
        }
        other => {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_ARGUMENT,
                format!("option {name} must be numeric, got {other:?}"),
            ))
        }
    };
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_ARGUMENT,
            format!("option {name} must be finite"),
        ))
    }
}

fn integer_option(name: &str, value: &Value) -> BuiltinResult<usize> {
    if let Some(integer) = tensor::scalar_integer_value(value) {
        return integer.try_to_usize().ok_or_else(|| {
            integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_ARGUMENT,
                format!("option {name} must be nonnegative"),
            )
        });
    }
    let parsed = numeric_option(name, value)?;
    if parsed < 0.0 {
        return Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_ARGUMENT,
            format!("option {name} must be nonnegative"),
        ));
    }
    if parsed.fract() != 0.0 {
        return Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_ARGUMENT,
            format!("option {name} must be an integer scalar"),
        ));
    }
    if parsed > usize::MAX as f64 || (usize::BITS == 64 && parsed == usize::MAX as f64) {
        return Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_ARGUMENT,
            format!("option {name} exceeds maximum supported size"),
        ));
    }
    Ok(parsed as usize)
}

fn bool_option(name: &str, value: &Value) -> BuiltinResult<bool> {
    match value {
        Value::Bool(flag) => Ok(*flag),
        Value::Num(n) if *n == 0.0 || *n == 1.0 => Ok(*n != 0.0),
        Value::Int(i) => {
            let raw = i.to_i64();
            if raw == 0 || raw == 1 {
                Ok(raw != 0)
            } else {
                Err(integral_error_with_detail(
                    &INTEGRAL_ERROR_INVALID_ARGUMENT,
                    format!("option {name} must be logical scalar"),
                ))
            }
        }
        Value::LogicalArray(LogicalArray { data, .. }) if data.len() == 1 => Ok(data[0] != 0),
        Value::Tensor(tensor) if tensor::is_scalar_tensor(tensor) => {
            if let Some(integer) = tensor::scalar_integer_value(value) {
                match integer.try_to_usize() {
                    Some(0) => Ok(false),
                    Some(1) => Ok(true),
                    _ => Err(integral_error_with_detail(
                        &INTEGRAL_ERROR_INVALID_ARGUMENT,
                        format!("option {name} must be logical scalar"),
                    )),
                }
            } else {
                match tensor::tensor_value_f64(tensor, 0) {
                    0.0 => Ok(false),
                    1.0 => Ok(true),
                    _ => Err(integral_error_with_detail(
                        &INTEGRAL_ERROR_INVALID_ARGUMENT,
                        format!("option {name} must be logical scalar"),
                    )),
                }
            }
        }
        other => Err(integral_error_with_detail(
            &INTEGRAL_ERROR_INVALID_ARGUMENT,
            format!("option {name} must be logical scalar, got {other:?}"),
        )),
    }
}
