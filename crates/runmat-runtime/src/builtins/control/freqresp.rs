//! Frequency response of one SISO transfer function, returned as 1×1×N.

use runmat_macros::runtime_builtin;
use runmat_value::{ComplexStorage, ComplexTensor, Value};

use crate::builtins::control::frequency_response;
use crate::builtins::control::frequency_response::FrequencyApi;
use crate::builtins::control::tf_model::control_error;
use crate::context::RuntimeContext;
use crate::BuiltinResult;

const NAME: &str = "freqresp";

#[runtime_builtin(
    name = "freqresp",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::control::freqresp"
)]
async fn freqresp_builtin(
    runtime: Option<RuntimeContext>,
    system: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if rest.len() > 1 || crate::output_count::current_output_count().is_some_and(|count| count > 2)
    {
        return Err(control_error(
            NAME,
            "RunMat:freqresp:InvalidArgument",
            "freqresp expects a model, optional frequency vector, and at most two outputs",
        ));
    }
    // Refuse an unowned managed materialization before scanning or copying an
    // arbitrarily large caller-provided frequency vector.
    frequency_response::require_host_owner(FrequencyApi::Freqresp, runtime.as_ref())?;
    // Validate and materialize supplied frequencies before a potentially
    // resident model read. The owner policy has already been checked.
    let supplied = rest
        .first()
        .map(|value| {
            frequency_response::explicit_grid(FrequencyApi::Freqresp, value, runtime.as_ref())
        })
        .transpose()?;
    let model = frequency_response::model(FrequencyApi::Freqresp, system, runtime.as_ref()).await?;
    let frequencies = match supplied {
        Some(values) => values,
        None => frequency_response::default_grid(FrequencyApi::Freqresp, &model)?,
    };
    let response = frequency_response::evaluate_grid(
        FrequencyApi::Freqresp,
        &model,
        &frequencies,
        runtime.as_ref(),
    )?;
    let count = response.len();
    let values: Vec<(f64, f64)> = response
        .into_iter()
        .map(|value| (value.re, value.im))
        .collect();
    let result =
        ComplexTensor::from_complex_storage(ComplexStorage::F64(values.into()), vec![1, 1, count])
            .map(Value::ComplexTensor)
            .map_err(|error| {
                control_error(
                    NAME,
                    "RunMat:freqresp:Internal",
                    format!("freqresp: response materialization failed: {error}"),
                )
            })?;
    match crate::output_count::current_output_count().unwrap_or(1) {
        0 => Ok(Value::output_list(Vec::new())),
        1 => Ok(result),
        2 => Ok(Value::output_list(vec![
            result,
            frequency_response::frequency_column(FrequencyApi::Freqresp, frequencies)?,
        ])),
        _ => unreachable!("output count checked before model evaluation"),
    }
}
