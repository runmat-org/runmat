//! MATLAB-compatible `step` response builtin for RunMat.

#[cfg(feature = "plot-core")]
use runmat_builtins::STEP_ERROR_PLOT_FAILED;
use runmat_builtins::{
    BuiltinErrorDescriptor, STEP_ERROR_INTERNAL, STEP_ERROR_INVALID_ARGUMENT,
    STEP_ERROR_INVALID_TIME, STEP_INTEGER_TIME_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::common::validation::{
    native_integer_value_is_exact_f64, value_has_native_integer_class,
};
use crate::builtins::control::response::step::{
    admit_time, preflight_host_time, simulate, StepResponse, StepSimulationCaller,
};
use crate::builtins::control::response::{
    parse_real_model, preflight_response_model_operand, ResponseKind,
};
#[cfg(feature = "plot-core")]
use crate::builtins::plotting::style::{parse_line_style_args, LineStyleParseOptions};
#[cfg(feature = "plot-core")]
use crate::plotting_hooks::{presentation_failed, with_render_cause};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "step";

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::step")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "step",
    op_kind: GpuOpKind::Custom("control-step-response"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Step-response simulation runs on the host from transfer-function metadata.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::step")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "step",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "step simulates a dynamic system and terminates numeric fusion chains.",
};

fn step_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    step_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn step_error_with_message(
    message: impl Into<String>,
    error: &'static BuiltinErrorDescriptor,
) -> RuntimeError {
    let mut builder = build_runtime_error(message).with_builtin(BUILTIN_NAME);
    if let Some(identifier) = error.identifier {
        builder = builder.with_identifier(identifier);
    }
    builder.build()
}

#[runtime_builtin(
    name = "step",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::control::step"
)]
async fn step_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    sys: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if !is_statement_form_call() && rest.len() > 1 {
        return Err(step_error_with_detail(
            &STEP_ERROR_INVALID_ARGUMENT,
            "expected step(sys), step(sys, tFinal), or step(sys, t)",
        ));
    }
    if crate::output_count::current_output_count().is_some_and(|count| count > 2) {
        return Err(step_error_with_detail(
            &STEP_ERROR_INVALID_ARGUMENT,
            "step supports at most two outputs",
        ));
    }
    if !is_statement_form_call() {
        preflight_step_time(rest.first())?;
        preflight_response_model_operand(&sys, ResponseKind::Step)?;
    }
    if is_statement_form_call() {
        plot_multiple_step_responses(runtime.as_ref(), sys, rest).await?;
        return Ok(Value::output_list(Vec::new()));
    }

    let model = parse_real_model(runtime.as_ref(), sys, ResponseKind::Step).await?;
    let time_host = gather_step_time(rest.first()).await?;
    let time = admit_time(
        time_host.as_ref(),
        model.sample_time(),
        StepSimulationCaller::Step,
    )?;
    let eval = simulate(&model, time, StepSimulationCaller::Step)?;

    if crate::output_context::requested_output_count() == Some(0)
        && crate::output_count::current_output_count().is_none()
    {
        plot_response(&eval).await?;
        return Ok(Value::output_list(Vec::new()));
    }

    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            plot_response(&eval).await?;
            return Ok(Value::output_list(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::output_list(vec![eval.y_value()?]));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            eval.outputs()?,
        ));
    }

    eval.y_value()
}

fn is_statement_form_call() -> bool {
    matches!(crate::output_count::current_output_count(), Some(0))
        || (crate::output_context::requested_output_count() == Some(0)
            && crate::output_count::current_output_count().is_none())
}

#[cfg(feature = "plot-core")]
async fn plot_multiple_step_responses(
    runtime: Option<&crate::context::RuntimeContext>,
    first_sys: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<()> {
    let mut systems = vec![(first_sys, None)];
    let mut time_arg = None;
    for arg in rest {
        if is_plot_style_arg(&arg) {
            parse_line_style_args(std::slice::from_ref(&arg), &LineStyleParseOptions::plot())
                .map_err(|error| {
                    step_error_with_detail(&STEP_ERROR_INVALID_ARGUMENT, error.message())
                })?;
            if let Some((_, style)) = systems.last_mut() {
                if style.is_some() {
                    return Err(step_error_with_detail(
                        &STEP_ERROR_INVALID_ARGUMENT,
                        "only one style argument is supported per system",
                    ));
                }
                *style = Some(arg);
                continue;
            }
            continue;
        }
        if matches!(arg, Value::Object(_)) {
            if time_arg.is_some() {
                return Err(step_error_with_detail(
                    &STEP_ERROR_INVALID_ARGUMENT,
                    "time argument must follow all systems in statement-form step plots",
                ));
            }
            systems.push((arg, None));
            continue;
        }
        if time_arg.is_none() {
            time_arg = Some(arg);
            continue;
        }
        return Err(step_error_with_detail(
            &STEP_ERROR_INVALID_ARGUMENT,
            "unsupported statement-form step plot argument",
        ));
    }

    // Every locally decidable error is checked before any owner is asked to
    // download model or time data. Plotting starts only after every response
    // has been calculated, so a later invalid system cannot leave a partial
    // figure behind.
    preflight_step_time(time_arg.as_ref())?;
    for (system, _) in &systems {
        preflight_response_model_operand(system, ResponseKind::Step)?;
    }
    let mut models = Vec::with_capacity(systems.len());
    for (system, style) in systems {
        models.push((
            parse_real_model(runtime, system, ResponseKind::Step).await?,
            style,
        ));
    }
    let gathered_time = gather_step_time(time_arg.as_ref()).await?;
    let mut responses = Vec::with_capacity(models.len());
    for (model, style) in models {
        let time = admit_time(
            gathered_time.as_ref(),
            model.sample_time(),
            StepSimulationCaller::Step,
        )?;
        responses.push((simulate(&model, time, StepSimulationCaller::Step)?, style));
    }

    let mut first = true;
    let mut hold_enabled = false;
    let result: BuiltinResult<()> = async {
        for (eval, style) in responses {
            plot_response_with_style(&eval, style.as_ref()).await?;
            if first {
                first = false;
                let _ = crate::call_builtin_async("hold", &[Value::from("on")]).await;
                hold_enabled = true;
            }
        }
        Ok(())
    }
    .await;
    if hold_enabled {
        let _ = crate::call_builtin_async("hold", &[Value::from("off")]).await;
    }
    result
}

#[cfg(not(feature = "plot-core"))]
async fn plot_multiple_step_responses(
    _runtime: Option<&crate::context::RuntimeContext>,
    _first_sys: Value,
    _rest: crate::call::BuiltinArguments,
) -> BuiltinResult<()> {
    Err(crate::plotting_hooks::unavailable_presentation())
}

#[cfg(feature = "plot-core")]
fn is_plot_style_arg(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    )
}

fn preflight_step_time(value: Option<&Value>) -> BuiltinResult<()> {
    let Some(value) = value else {
        return Ok(());
    };
    if value_has_native_integer_class(value) {
        crate::compatibility::ensure_builtin_extension_enabled(
            &STEP_INTEGER_TIME_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if !matches!(value, Value::GpuTensor(_)) && !native_integer_value_is_exact_f64(value) {
        return Err(step_error_with_detail(
            &STEP_ERROR_INVALID_TIME,
            "integer time values must be exactly representable as double",
        ));
    }
    preflight_host_time(Some(value), StepSimulationCaller::Step)
}

async fn gather_step_time(value: Option<&Value>) -> BuiltinResult<Option<Value>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let gathered = crate::gather_if_needed_async(value).await?;
    if !native_integer_value_is_exact_f64(&gathered) {
        return Err(step_error_with_detail(
            &STEP_ERROR_INVALID_TIME,
            "integer time values must be exactly representable as double",
        ));
    }
    Ok(Some(gathered))
}

impl StepResponse {
    fn y_value(&self) -> BuiltinResult<Value> {
        column_tensor(self.y.clone())
    }

    fn t_value(&self) -> BuiltinResult<Value> {
        column_tensor(self.t.clone())
    }

    fn outputs(&self) -> BuiltinResult<Vec<Value>> {
        Ok(vec![self.y_value()?, self.t_value()?])
    }
}

fn column_tensor(data: Vec<f64>) -> BuiltinResult<Value> {
    let rows = data.len();
    let tensor = Tensor::new(data, vec![rows, 1]).map_err(|err| {
        step_error_with_detail(
            &STEP_ERROR_INTERNAL,
            format!("failed to build response tensor: {err}"),
        )
    })?;
    Ok(Value::Tensor(tensor))
}

async fn plot_response(eval: &StepResponse) -> BuiltinResult<()> {
    plot_response_with_style(eval, None).await
}

#[cfg(feature = "plot-core")]
async fn plot_response_with_style(eval: &StepResponse, style: Option<&Value>) -> BuiltinResult<()> {
    let t = eval.t_value()?;
    let y = eval.y_value()?;
    let args = if let Some(style) = style {
        vec![t, y, style.clone()]
    } else {
        vec![t, y]
    };
    if let Err(err) = crate::call_builtin_async("plot", &args).await {
        if presentation_failed(&err) {
            return Ok(());
        }
        return Err(with_render_cause(
            step_error_with_detail(&STEP_ERROR_PLOT_FAILED, err.message()),
            err,
        ));
    }
    let _ = crate::call_builtin_async("title", &[Value::from("Step Response")]).await;
    let _ = crate::call_builtin_async("xlabel", &[Value::from("Time")]).await;
    let _ = crate::call_builtin_async("ylabel", &[Value::from("Amplitude")]).await;
    Ok(())
}

#[cfg(not(feature = "plot-core"))]
async fn plot_response_with_style(
    _eval: &StepResponse,
    _style: Option<&Value>,
) -> BuiltinResult<()> {
    Err(crate::plotting_hooks::unavailable_presentation())
}

#[cfg(test)]
mod tests;
