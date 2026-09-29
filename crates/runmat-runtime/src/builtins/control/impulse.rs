//! MATLAB-compatible `impulse` response builtin for supported control models.

use runmat_builtins::{
    BuiltinErrorDescriptor, IMPULSE_ERROR_DISCRETE_LIMIT, IMPULSE_ERROR_INTERNAL,
    IMPULSE_ERROR_INVALID_ARGUMENT, IMPULSE_ERROR_INVALID_MODEL, IMPULSE_ERROR_INVALID_TIME,
    IMPULSE_ERROR_UNSUPPORTED_MODEL, IMPULSE_INTEGER_EXTENSION, IMPULSE_LOGICAL_EXTENSION,
    IMPULSE_RESIDENT_EXTENSION, IMPULSE_SINGLE_EXTENSION,
};
use runmat_macros::runtime_builtin;
use runmat_value::{NumericDType, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::control::plot_setup_error::is_nonfatal_plot_setup_error;
use crate::builtins::control::response::{
    parse_real_model, preflight_response_model_operand, ResponseKind,
};
use crate::builtins::control::tf_model::RealTfModel;
use crate::{build_runtime_error, BuiltinResult, RuntimeError};

const BUILTIN_NAME: &str = "impulse";
const DEFAULT_POINTS: usize = 100;
const MAX_DISCRETE_SAMPLES: usize = 1_000_000;

mod solver;
mod time;

use solver::{evaluate_impulse, ImpulseResponse};
#[cfg(test)]
use time::time_vector_from_value;
use time::{checked_discrete_sample_index, TimeSpec};

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::impulse")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "impulse",
    op_kind: GpuOpKind::Custom("control-impulse-response"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Control-system response evaluation runs on the host. GPU-resident metadata is gathered before simulation.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::impulse")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "impulse",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Impulse-response simulation materialises host-side time and output vectors and terminates fusion chains.",
};

fn impulse_error_with_detail(
    error: &'static BuiltinErrorDescriptor,
    detail: impl AsRef<str>,
) -> RuntimeError {
    impulse_error_with_message(format!("{}: {}", error.message, detail.as_ref()), error)
}

fn impulse_error_with_message(
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
    name = "impulse",
    category = "control",
    summary = "Compute or plot impulse responses.",
    keywords = "impulse,control system,transfer function,response,tf",
    sink = true,
    suppress_auto_output = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::control::impulse"
)]
async fn impulse_builtin(system: Value, rest: Vec<Value>) -> BuiltinResult<Value> {
    if rest.len() > 1 || crate::output_count::current_output_count().is_some_and(|count| count > 2)
    {
        return Err(impulse_error_with_detail(
            &IMPULSE_ERROR_INVALID_ARGUMENT,
            "impulse accepts at most one time argument and two outputs",
        ));
    }
    TimeSpec::preflight_host(&rest)?;
    preflight_response_model_operand(&system, ResponseKind::Impulse)?;
    ensure_impulse_extensions(&system, &rest)?;
    let system = TfSystem::parse(system).await?;
    let time = TimeSpec::parse(&system, &rest).await?;
    let response = evaluate_impulse(&system, time)?;

    if let Some(out_count) = crate::output_count::current_output_count() {
        if out_count == 0 {
            emit_impulse_plot(&response).await?;
            return Ok(Value::OutputList(Vec::new()));
        }
        if out_count == 1 {
            return Ok(Value::OutputList(vec![response.y_value()?]));
        }
        return Ok(crate::output_count::output_list_with_padding(
            out_count,
            vec![response.y_value()?, response.t_value()?],
        ));
    }

    if crate::output_context::requested_output_count() == Some(0) {
        emit_impulse_plot(&response).await?;
        return Ok(Value::OutputList(Vec::new()));
    }

    response.y_value()
}

fn ensure_impulse_extensions(system: &Value, rest: &[Value]) -> BuiltinResult<()> {
    let mut values = consumed_numeric_inputs(system, rest).into_iter();
    let integer = values.clone().any(value_contains_typed_integer);
    let logical = values.clone().any(value_contains_logical);
    let single = values.clone().any(value_contains_single);
    let resident = values.any(|value| matches!(value, Value::GpuTensor(_)));
    if integer {
        crate::compatibility::ensure_builtin_extension_enabled(
            &IMPULSE_INTEGER_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if logical {
        crate::compatibility::ensure_builtin_extension_enabled(
            &IMPULSE_LOGICAL_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if single {
        crate::compatibility::ensure_builtin_extension_enabled(
            &IMPULSE_SINGLE_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    if resident {
        crate::compatibility::ensure_builtin_extension_enabled(
            &IMPULSE_RESIDENT_EXTENSION,
            BUILTIN_NAME,
        )?;
    }
    Ok(())
}

fn consumed_numeric_inputs<'a>(system: &'a Value, rest: &'a [Value]) -> Vec<&'a Value> {
    let mut values = Vec::with_capacity(7);
    if let Value::Object(object) = system {
        for name in [
            "Numerator",
            "Denominator",
            "Ts",
            "InputDelay",
            "OutputDelay",
            "Variable",
        ] {
            if let Some(value) = object.properties.get(name) {
                values.push(value);
            }
        }
    } else {
        values.push(system);
    }
    values.extend(rest.iter());
    values
}

fn value_contains_typed_integer(value: &Value) -> bool {
    match value {
        Value::Int(_) => true,
        Value::Tensor(tensor) => tensor.integer_storage().is_some(),
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_integer_type(handle).is_some(),
        _ => false,
    }
}

fn value_contains_logical(value: &Value) -> bool {
    match value {
        Value::Bool(_) | Value::LogicalArray(_) => true,
        Value::GpuTensor(handle) => runmat_accelerate_api::handle_is_logical(handle),
        _ => false,
    }
}

fn value_contains_single(value: &Value) -> bool {
    match value {
        Value::Tensor(tensor) => tensor.numeric_dtype() == NumericDType::F32,
        Value::GpuTensor(handle) => {
            runmat_accelerate_api::handle_integer_type(handle).is_none()
                && !runmat_accelerate_api::handle_is_logical(handle)
                && runmat_accelerate_api::handle_precision(handle)
                    == Some(runmat_accelerate_api::ProviderPrecision::F32)
        }
        _ => false,
    }
}

async fn emit_impulse_plot(response: &ImpulseResponse) -> BuiltinResult<()> {
    if let Err(err) = render_impulse_plot(response).await {
        if is_nonfatal_plot_setup_error(&err) {
            return Ok(());
        }
        return Err(err);
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct TfSystem {
    numerator: Vec<f64>,
    denominator: Vec<f64>,
    sample_time: f64,
}

impl TfSystem {
    async fn parse(value: Value) -> BuiltinResult<Self> {
        let model = parse_real_model(value, ResponseKind::Impulse).await?;
        if model.sample_time < 0.0 {
            return Err(impulse_error_with_detail(
                &IMPULSE_ERROR_INVALID_MODEL,
                "impulse requires a known nonnegative sample time",
            ));
        }
        let numerator = model.numerator;
        let denominator = model.denominator;
        let sample_time = model.sample_time;
        if numerator.iter().all(|value| *value == 0.0) {
            return Ok(Self {
                numerator: Vec::new(),
                denominator,
                sample_time,
            });
        }
        if denominator.len() <= 1 {
            return Err(impulse_error_with_detail(
                &IMPULSE_ERROR_UNSUPPORTED_MODEL,
                "static-gain transfer functions do not have a finite impulse-response vector",
            ));
        }
        if numerator.len() >= denominator.len() {
            return Err(impulse_error_with_detail(
                &IMPULSE_ERROR_UNSUPPORTED_MODEL,
                "only strictly proper SISO tf models are currently supported",
            ));
        }

        Ok(Self {
            numerator,
            denominator,
            sample_time,
        })
    }

    fn is_discrete(&self) -> bool {
        self.sample_time > 0.0
    }

    fn normalized(&self) -> BuiltinResult<(Vec<f64>, Vec<f64>)> {
        RealTfModel {
            numerator: self.numerator.clone(),
            denominator: self.denominator.clone(),
            sample_time: self.sample_time,
            input_delay: 0.0,
            output_delay: 0.0,
        }
        .normalized(BUILTIN_NAME)
    }
}

#[cfg(feature = "plot-core")]
async fn render_impulse_plot(response: &ImpulseResponse) -> BuiltinResult<Value> {
    let t = response.t_value()?;
    let y = response.y_value()?;
    let plot_name = if response.discrete { "stem" } else { "plot" };
    crate::dispatcher::call_builtin_async(plot_name, &[t, y]).await
}

#[cfg(not(feature = "plot-core"))]
async fn render_impulse_plot(_response: &ImpulseResponse) -> BuiltinResult<Value> {
    Ok(Value::Num(f64::NAN))
}

#[cfg(test)]
mod tests;
