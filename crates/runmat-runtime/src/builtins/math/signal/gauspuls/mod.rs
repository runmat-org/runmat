//! Gaussian-modulated sinusoid samples and cutoff times.
mod admission;
mod error;
mod execution;
mod math;
mod parameters;
#[cfg(test)]
mod provider_tests;
#[cfg(test)]
mod tests;
use crate::builtins::common::tensor::{scalar_f64_from_value_async, tensor_into_value};
use crate::builtins::common::{gpu_helpers, tensor};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};
use error::*;
pub(crate) use math::{default_params, gauspuls_scalar, validate_params, GauspulsParams};
use math::{gauspuls_components_tensor, gauspuls_cutoff, gauspuls_tensor, validate_tpe};
use runmat_builtins::{
    BuiltinErrorDescriptor, GAUSPULS_ERROR_ARG_COUNT, GAUSPULS_ERROR_INTERNAL,
    GAUSPULS_ERROR_INVALID_INPUT, GAUSPULS_ERROR_INVALID_PARAMETER,
    GAUSPULS_INTEGER_CONTROL_EXTENSION, GAUSPULS_INTEGER_TIME_EXTENSION,
    GAUSPULS_LOGICAL_CONTROL_EXTENSION, GAUSPULS_LOGICAL_TIME_EXTENSION,
    GAUSPULS_RESIDENT_INPUT_EXTENSION, GAUSPULS_SINGLE_CONTROL_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, NumericStorage, Tensor, Value};
use std::f64::consts::PI;
const BUILTIN_NAME: &str = "gauspuls";
const DEFAULT_FC: f64 = 1000.;
const DEFAULT_BW: f64 = 0.5;
const DEFAULT_BWR: f64 = -6.;
const DEFAULT_TPE: f64 = -60.;
#[runtime_builtin(
    name = "gauspuls",
    binding_variant = "default",
    builtin_path = "crate::builtins::math::signal::gauspuls"
)]
async fn gauspuls_builtin(t: Value, rest: crate::call::BuiltinArguments) -> BuiltinResult<Value> {
    if let Some(mode) = parameters::text_scalar(&t)? {
        return execution::gauspuls_mode(mode, rest).await;
    }
    if rest.len() > 3 {
        return Err(gauspuls_error_with_detail(
            &GAUSPULS_ERROR_ARG_COUNT,
            format!("got {}", rest.len() + 1),
        ));
    }
    admission::ensure_gauspuls_time_extensions(&t)?;
    admission::ensure_gauspuls_control_extensions(rest.view())?;
    let params = parameters::parse_params(rest.view()).await?;
    if let Some(out_count) = crate::output_count::current_output_count() {
        return execution::gauspuls_with_output_count(t, params, out_count).await;
    }
    execution::gauspuls_value(t, params).await
}
