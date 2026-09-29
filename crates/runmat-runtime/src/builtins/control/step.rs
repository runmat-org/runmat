//! MATLAB-compatible `step` response builtin for RunMat.

use runmat_builtins::{
    BuiltinErrorDescriptor, STEP_ERROR_DISCRETE_LIMIT, STEP_ERROR_INTERNAL,
    STEP_ERROR_INVALID_ARGUMENT, STEP_ERROR_INVALID_MODEL, STEP_ERROR_INVALID_TIME,
    STEP_ERROR_PLOT_FAILED, STEP_ERROR_UNSUPPORTED_MODEL, STEP_INTEGER_TIME_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::control::plot_setup_error::is_nonfatal_plot_setup_error;
use crate::builtins::control::response::{
    parse_real_model, preflight_response_model_operand, ResponseKind,
};
use crate::builtins::plotting::style::{parse_line_style_args, LineStyleParseOptions};
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "step";
const EPS: f64 = 1.0e-12;
const DEFAULT_SAMPLES: usize = 101;
const MAX_DISCRETE_SAMPLES: usize = 1_000_000;

mod solver;
mod time;

use solver::{evaluate_step, StepEval};
use time::TimeSpec;

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
    category = "control",
    summary = "Compute or plot step responses of SISO transfer-function models.",
    keywords = "step,response,control system,transfer function,tf",
    sink = true,
    suppress_auto_output = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::control::step"
)]
async fn step_builtin(sys: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
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
        TimeSpec::preflight_host(rest.first())?;
        preflight_response_model_operand(&sys, ResponseKind::Step)?;
    }
    if is_statement_form_call() {
        plot_multiple_step_responses(sys, rest).await?;
        return Ok(Value::OutputList(Vec::new()));
    }

    if let Some(value) = rest.first() {
        crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
            value,
            &STEP_INTEGER_TIME_EXTENSION,
            BUILTIN_NAME,
            "time",
        )
        .await?;
    }

    let model = TransferFunction::from_value_async(sys).await?;
    let time_host = match rest.first() {
        Some(value) => Some(crate::gather_if_needed_async(value).await?),
        None => None,
    };
    let time = TimeSpec::parse(time_host.as_ref(), model.sample_time)?;
    let eval = evaluate_step(&model, time)?;

    if crate::output_context::requested_output_count() == Some(0)
        && crate::output_count::current_output_count().is_none()
    {
        plot_response(&eval).await?;
        return Ok(Value::OutputList(Vec::new()));
    }

    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            plot_response(&eval).await?;
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![eval.y_value()?]));
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

async fn plot_multiple_step_responses(first_sys: Value, rest: Vec<Value>) -> BuiltinResult<()> {
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
    TimeSpec::preflight_host(time_arg.as_ref())?;
    for (system, _) in &systems {
        preflight_response_model_operand(system, ResponseKind::Step)?;
    }
    if let Some(time) = &time_arg {
        crate::builtins::common::validation::ensure_runmat_integer_f64_boundary(
            time,
            &STEP_INTEGER_TIME_EXTENSION,
            BUILTIN_NAME,
            "time",
        )
        .await?;
    }
    let mut models = Vec::with_capacity(systems.len());
    for (system, style) in systems {
        models.push((TransferFunction::from_value_async(system).await?, style));
    }
    let gathered_time = match time_arg {
        Some(time) => Some(crate::gather_if_needed_async(&time).await?),
        None => None,
    };
    let mut responses = Vec::with_capacity(models.len());
    for (model, style) in models {
        let time = TimeSpec::parse(gathered_time.as_ref(), model.sample_time)?;
        responses.push((evaluate_step(&model, time)?, style));
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

fn is_plot_style_arg(value: &Value) -> bool {
    matches!(
        value,
        Value::String(_) | Value::StringArray(_) | Value::CharArray(_)
    )
}

#[derive(Clone, Debug)]
struct TransferFunction {
    numerator: Vec<f64>,
    denominator: Vec<f64>,
    sample_time: f64,
}

impl TransferFunction {
    async fn from_value_async(value: Value) -> BuiltinResult<Self> {
        let real = parse_real_model(value, ResponseKind::Step).await?;
        if real.sample_time < 0.0 {
            return Err(step_error_with_detail(
                &STEP_ERROR_INVALID_MODEL,
                "step requires a known nonnegative sample time",
            ));
        }
        let numerator = real.numerator;
        let denominator = real.denominator;
        let sample_time = real.sample_time;
        if numerator.len().saturating_sub(1) > denominator.len().saturating_sub(1) {
            return Err(step_error_with_detail(
                &STEP_ERROR_UNSUPPORTED_MODEL,
                "improper transfer functions are not supported yet",
            ));
        }

        Ok(Self {
            numerator,
            denominator,
            sample_time,
        })
    }

    fn normalized(&self) -> (Vec<f64>, Vec<f64>) {
        let leading = self.denominator[0];
        let den = self
            .denominator
            .iter()
            .map(|value| value / leading)
            .collect::<Vec<_>>();
        let num = self
            .numerator
            .iter()
            .map(|value| value / leading)
            .collect::<Vec<_>>();
        (num, den)
    }
}

async fn plot_response(eval: &StepEval) -> BuiltinResult<()> {
    plot_response_with_style(eval, None).await
}

async fn plot_response_with_style(eval: &StepEval, style: Option<&Value>) -> BuiltinResult<()> {
    let t = eval.t_value()?;
    let y = eval.y_value()?;
    let args = if let Some(style) = style {
        vec![t, y, style.clone()]
    } else {
        vec![t, y]
    };
    if let Err(err) = crate::call_builtin_async("plot", &args).await {
        if is_nonfatal_plot_setup_error(&err) {
            return Ok(());
        }
        return Err(step_error_with_detail(
            &STEP_ERROR_PLOT_FAILED,
            err.message(),
        ));
    }
    let _ = crate::call_builtin_async("title", &[Value::from("Step Response")]).await;
    let _ = crate::call_builtin_async("xlabel", &[Value::from("Time")]).await;
    let _ = crate::call_builtin_async("ylabel", &[Value::from("Amplitude")]).await;
    Ok(())
}

#[cfg(test)]
mod tests;
