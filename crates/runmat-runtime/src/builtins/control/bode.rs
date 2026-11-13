//! SISO magnitude/phase response through the shared frequency evaluator.

use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::builtins::control::frequency_response;
use crate::builtins::control::frequency_response::FrequencyApi;
use crate::builtins::control::tf_model::control_error;
use crate::context::RuntimeContext;
use crate::BuiltinResult;

const NAME: &str = "bode";

#[runtime_builtin(
    name = "bode",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::control::bode"
)]
async fn bode_builtin(
    runtime: Option<RuntimeContext>,
    system: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if rest.len() > 1 || crate::output_count::current_output_count().is_some_and(|count| count > 3)
    {
        return Err(control_error(
            NAME,
            "RunMat:bode:InvalidArgument",
            "bode expects a model, optional frequency vector, and at most three outputs",
        ));
    }
    // The host grid scan and copy are not admitted under managed allocation.
    frequency_response::require_host_owner(FrequencyApi::Bode, runtime.as_ref())?;
    let supplied = rest
        .first()
        .map(|value| frequency_response::explicit_grid(FrequencyApi::Bode, value, runtime.as_ref()))
        .transpose()?;
    let model = frequency_response::model(FrequencyApi::Bode, system, runtime.as_ref()).await?;
    let frequencies = match supplied {
        Some(values) => values,
        None => frequency_response::default_grid(FrequencyApi::Bode, &model)?,
    };
    let values = frequency_response::evaluate_grid(
        FrequencyApi::Bode,
        &model,
        &frequencies,
        runtime.as_ref(),
    )?;
    let mut magnitude = Vec::with_capacity(values.len());
    let mut phase = Vec::with_capacity(values.len());
    let mut previous = None;
    for value in values {
        magnitude.push(value.norm());
        let mut degrees = value.arg().to_degrees();
        if let Some(last) = previous {
            while degrees - last > 180.0 {
                degrees -= 360.0;
            }
            while degrees - last < -180.0 {
                degrees += 360.0;
            }
        }
        previous = Some(degrees);
        phase.push(degrees);
    }
    let response = BodeResponse {
        magnitude,
        phase,
        frequencies,
    };
    let requested = crate::output_count::current_output_count();
    if requested == Some(0)
        || (requested.is_none() && crate::output_context::requested_output_count() == Some(0))
    {
        render_plot(&response).await?;
        return Ok(Value::output_list(Vec::new()));
    }
    response.outputs(requested.unwrap_or(1))
}

struct BodeResponse {
    magnitude: Vec<f64>,
    phase: Vec<f64>,
    frequencies: Vec<f64>,
}

impl BodeResponse {
    fn response_array(data: Vec<f64>) -> BuiltinResult<Value> {
        let count = data.len();
        Tensor::new(data, vec![1, 1, count])
            .map(Value::Tensor)
            .map_err(|error| {
                control_error(
                    NAME,
                    "RunMat:bode:Internal",
                    format!("bode: response materialization failed: {error}"),
                )
            })
    }

    fn outputs(self, count: usize) -> BuiltinResult<Value> {
        let magnitude = Self::response_array(self.magnitude)?;
        match count {
            1 => Ok(magnitude),
            2 => Ok(Value::output_list(vec![
                magnitude,
                Self::response_array(self.phase)?,
            ])),
            3 => Ok(Value::output_list(vec![
                magnitude,
                Self::response_array(self.phase)?,
                frequency_response::frequency_column(FrequencyApi::Bode, self.frequencies)?,
            ])),
            _ => unreachable!("output count checked before response computation"),
        }
    }
}

#[cfg(feature = "plot-core")]
async fn render_plot(response: &BodeResponse) -> BuiltinResult<()> {
    use crate::builtins::plotting::state::presentation_failure_kind;

    let count = response.frequencies.len();
    let frequencies =
        || frequency_response::frequency_column(FrequencyApi::Bode, response.frequencies.clone());
    let magnitude_db = response
        .magnitude
        .iter()
        .map(|value| 20.0 * value.log10())
        .collect();
    let magnitude_db = Tensor::new(magnitude_db, vec![count, 1])
        .map(Value::Tensor)
        .map_err(|error| control_error(NAME, "RunMat:bode:Internal", error))?;
    let phase = Tensor::new(response.phase.clone(), vec![count, 1])
        .map(Value::Tensor)
        .map_err(|error| control_error(NAME, "RunMat:bode:Internal", error))?;

    let plot = async {
        crate::call_builtin_async(
            "subplot",
            &[Value::Num(2.0), Value::Num(1.0), Value::Num(1.0)],
        )
        .await?;
        crate::call_builtin_async("semilogx", &[frequencies()?, magnitude_db]).await?;
        let _ = crate::call_builtin_async("ylabel", &[Value::from("Magnitude (dB)")]).await;
        crate::call_builtin_async(
            "subplot",
            &[Value::Num(2.0), Value::Num(1.0), Value::Num(2.0)],
        )
        .await?;
        crate::call_builtin_async("semilogx", &[frequencies()?, phase]).await?;
        let _ = crate::call_builtin_async("xlabel", &[Value::from("Frequency (rad/s)")]).await;
        let _ = crate::call_builtin_async("ylabel", &[Value::from("Phase (deg)")]).await;
        Ok::<(), crate::RuntimeError>(())
    }
    .await;
    match plot {
        Ok(()) => Ok(()),
        Err(error) if presentation_failure_kind(&error).is_some() => Ok(()),
        Err(error) => Err(crate::builtins::plotting::state::with_render_cause(
            control_error(
                NAME,
                "RunMat:bode:PlotFailed",
                format!("bode: plotting failed: {}", error.message()),
            ),
            error,
        )),
    }
}

#[cfg(not(feature = "plot-core"))]
async fn render_plot(_response: &BodeResponse) -> BuiltinResult<()> {
    Err(crate::plotting_hooks::unavailable_presentation())
}
