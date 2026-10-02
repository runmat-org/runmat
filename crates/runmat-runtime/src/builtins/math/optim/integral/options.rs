//! Typed option values and one ordered admission authority for `integral`.

use super::{integral_error_with_detail, IntegralOptionName, DEFAULT_MAX_FUN_EVALS};
use crate::builtins::common::tensor;
use crate::BuiltinResult;
use num_complex::Complex;
use runmat_builtins::{IntegralCallbackMode, INTEGRAL_ERROR_INVALID_ARGUMENT};
use runmat_value::{LogicalArray, Value};
use std::num::NonZeroUsize;

use super::waypoints;
use crate::builtins::math::optim::quadrature::QuadratureCooperation;

pub(super) struct IntegralOptions {
    pub(super) abs_tol: f64,
    pub(super) rel_tol: f64,
    pub(super) max_fun_evals: usize,
    pub(super) max_interval_count: Option<NonZeroUsize>,
    pub(super) callback_mode: IntegralCallbackMode,
    pub(super) array_valued: bool,
    pub(super) waypoints: Vec<Complex<f64>>,
}

impl IntegralOptions {
    fn defaults() -> Self {
        Self {
            abs_tol: IntegralOptionName::DEFAULT_ABS_TOL,
            rel_tol: IntegralOptionName::DEFAULT_REL_TOL,
            max_fun_evals: DEFAULT_MAX_FUN_EVALS,
            max_interval_count: None,
            callback_mode: IntegralCallbackMode::Vectorized,
            array_valued: false,
            waypoints: Vec::new(),
        }
    }

    pub(super) async fn preflight_host(
        rest: &[(IntegralOptionName, &Value)],
        cooperation: &mut QuadratureCooperation,
    ) -> BuiltinResult<Vec<Option<Vec<Complex<f64>>>>> {
        let mut check = Self::defaults();
        let mut known_abs_tol = Some(check.abs_tol);
        let mut known_rel_tol = Some(check.rel_tol);
        let mut waypoint_receipts = Vec::new();
        waypoint_receipts
            .try_reserve_exact(rest.len())
            .map_err(|source| {
                crate::build_runtime_error("integral: could not reserve option admission")
                    .with_identifier("RunMat:Internal")
                    .with_builtin("integral")
                    .with_source(source)
                    .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                    .build()
            })?;
        for &(name, value) in rest {
            if name == IntegralOptionName::Waypoints {
                waypoints::admit_shape(value)?;
                waypoint_receipts.push(match value {
                    Value::GpuTensor(_) => None,
                    host => Some(waypoints::decode(host, cooperation).await?),
                });
                continue;
            }
            waypoint_receipts.push(None);
            if matches!(value, Value::GpuTensor(_)) {
                match name {
                    IntegralOptionName::AbsTol | IntegralOptionName::AbsoluteTolerance => {
                        known_abs_tol = None
                    }
                    IntegralOptionName::RelTol | IntegralOptionName::RelativeTolerance => {
                        known_rel_tol = None
                    }
                    IntegralOptionName::MaxFunEvals
                    | IntegralOptionName::MaxIntervalCount
                    | IntegralOptionName::ArrayValued
                    | IntegralOptionName::Vectorized
                    | IntegralOptionName::Waypoints => {}
                }
                continue;
            }
            check.apply_option(name, value, cooperation).await?;
            match name {
                IntegralOptionName::AbsTol | IntegralOptionName::AbsoluteTolerance => {
                    known_abs_tol = Some(check.abs_tol)
                }
                IntegralOptionName::RelTol | IntegralOptionName::RelativeTolerance => {
                    known_rel_tol = Some(check.rel_tol)
                }
                IntegralOptionName::MaxFunEvals
                | IntegralOptionName::MaxIntervalCount
                | IntegralOptionName::ArrayValued
                | IntegralOptionName::Vectorized
                | IntegralOptionName::Waypoints => {}
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
        Ok(waypoint_receipts)
    }

    pub(super) async fn parse(
        rest: &[(IntegralOptionName, &Value)],
        mut waypoint_receipts: Vec<Option<Vec<Complex<f64>>>>,
        cooperation: &mut QuadratureCooperation,
    ) -> BuiltinResult<Self> {
        let mut options = Self::defaults();
        for (index, &(name, value)) in rest.iter().enumerate() {
            if name == IntegralOptionName::Waypoints {
                options.waypoints = match waypoint_receipts[index].take() {
                    Some(host) => host,
                    None => {
                        let gathered = materialize_waypoint(value, cooperation).await?;
                        waypoints::decode_exact(&gathered, cooperation).await?
                    }
                };
                continue;
            }
            if matches!(value, Value::GpuTensor(_)) {
                let gathered = crate::dispatcher::gather_if_needed_async(value).await?;
                options.apply_option(name, &gathered, cooperation).await?;
            } else {
                options.apply_option(name, value, cooperation).await?;
            }
        }
        options.validate()?;
        Ok(options)
    }

    async fn apply_option(
        &mut self,
        name: IntegralOptionName,
        value: &Value,
        cooperation: &mut QuadratureCooperation,
    ) -> BuiltinResult<()> {
        match name {
            IntegralOptionName::AbsTol | IntegralOptionName::AbsoluteTolerance => {
                self.abs_tol = numeric_option(name.display_name(), value)?
            }
            IntegralOptionName::RelTol | IntegralOptionName::RelativeTolerance => {
                self.rel_tol = numeric_option(name.display_name(), value)?
            }
            IntegralOptionName::MaxFunEvals => {
                let parsed = integer_option(name.display_name(), value)?;
                if parsed < 5 {
                    return Err(integral_error_with_detail(
                        &INTEGRAL_ERROR_INVALID_ARGUMENT,
                        "MaxFunEvals must be an integer scalar >= 5",
                    ));
                }
                self.max_fun_evals = parsed;
            }
            IntegralOptionName::MaxIntervalCount => {
                let parsed = integer_option(name.display_name(), value)?;
                self.max_interval_count = Some(NonZeroUsize::new(parsed).ok_or_else(|| {
                    integral_error_with_detail(
                        &INTEGRAL_ERROR_INVALID_ARGUMENT,
                        "MaxIntervalCount must be a positive integer scalar",
                    )
                })?);
            }
            IntegralOptionName::ArrayValued => {
                self.array_valued = bool_option("ArrayValued", value)?;
            }
            IntegralOptionName::Vectorized => {
                self.callback_mode =
                    IntegralCallbackMode::from_vectorized(bool_option("Vectorized", value)?);
            }
            IntegralOptionName::Waypoints => {
                self.waypoints = waypoints::decode(value, cooperation).await?;
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

async fn materialize_waypoint(
    value: &Value,
    cooperation: &mut QuadratureCooperation,
) -> BuiltinResult<Value> {
    use crate::builtins::common::validation::{
        value_has_native_integer_class, PreparedIntegerPayload,
    };
    if value_has_native_integer_class(value) {
        let prepared = cooperation.prepare_integer_boundary(value).await?;
        if !prepared.is_exact() {
            return Err(integral_error_with_detail(
                &INTEGRAL_ERROR_INVALID_ARGUMENT,
                "Waypoints integer values must convert to double exactly",
            ));
        }
        if let Some(payload) = prepared.into_resident_payload() {
            let converted = match payload {
                PreparedIntegerPayload::Real(gathered) => runmat_value::Tensor::new_integer(
                    crate::builtins::common::gpu_helpers::integer_storage_from_owned(gathered.data),
                    gathered.shape,
                )
                .map(Value::Tensor),
                PreparedIntegerPayload::Complex(gathered) => {
                    crate::builtins::common::gpu_helpers::value_from_numeric_download(
                        gathered, false,
                    )
                }
            };
            return converted.map_err(|detail| {
                crate::build_runtime_error(format!(
                    "integral: invalid native waypoint download: {detail}"
                ))
                .with_identifier("RunMat:Internal")
                .with_builtin("integral")
                .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
                .build()
            });
        }
    }
    crate::dispatcher::gather_if_needed_async(value).await
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
            ));
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
