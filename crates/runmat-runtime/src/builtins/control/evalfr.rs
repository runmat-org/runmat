//! Evaluate one admitted SISO transfer function at an s- or z-plane point.

use num_complex::Complex64;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::control::frequency_response;
use crate::builtins::control::frequency_response::FrequencyApi;
use crate::builtins::control::tf_model::control_error;
use crate::context::RuntimeContext;
use crate::BuiltinResult;

const NAME: &str = "evalfr";

#[runtime_builtin(
    name = "evalfr",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::control::evalfr"
)]
async fn evalfr_builtin(
    runtime: Option<RuntimeContext>,
    system: Value,
    point: Value,
) -> BuiltinResult<Value> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 1) {
        return Err(control_error(
            NAME,
            "RunMat:evalfr:InvalidArgument",
            "evalfr returns one response",
        ));
    }
    let point = frequency_response::point_argument(FrequencyApi::Evalfr, &point)?;
    let model = frequency_response::model(FrequencyApi::Evalfr, system, runtime.as_ref()).await?;
    let value =
        frequency_response::at_point(FrequencyApi::Evalfr, &model, point, runtime.as_ref())?;
    Ok(scalar_value(value))
}

fn scalar_value(value: Complex64) -> Value {
    if value.im == 0.0 {
        Value::Num(value.re)
    } else {
        Value::Complex(value.re, value.im)
    }
}
