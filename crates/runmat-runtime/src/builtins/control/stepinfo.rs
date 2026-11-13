//! Step-response metrics for sampled data or a supported real SISO model.

mod admission;
mod metrics;
#[cfg(test)]
mod tests;

use runmat_builtins::STEPINFO_EXTENSIONS;
use runmat_macros::runtime_builtin;
use runmat_value::Value;

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::control::response::step::{
    admit_time, preflight_host_time, simulate, StepSimulationCaller,
};
use crate::builtins::control::response::{
    parse_admitted_model, preflight_response_model_operand, ResponseKind,
};
use crate::builtins::control::tf_model::{control_error, TfModel};
use crate::BuiltinResult;

const BUILTIN_NAME: &str = "stepinfo";
const DEFAULT_SETTLING_THRESHOLD: f64 = 0.02;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::stepinfo")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "stepinfo",
    op_kind: GpuOpKind::Custom("control-stepinfo"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Admitted sampled data and consumed model properties gather to the host before scalar metrics.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::stepinfo")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "stepinfo",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Step metrics are a host struct result, not a numeric fusion operation.",
};

#[runtime_builtin(
    name = "stepinfo",
    context = true,
    binding_variant = "default",
    builtin_path = "crate::builtins::control::stepinfo"
)]
async fn stepinfo_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    input: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if crate::output_count::current_output_count().is_some_and(|count| count > 1) {
        return Err(control_error(
            BUILTIN_NAME,
            "RunMat:stepinfo:InvalidArgument",
            "stepinfo returns one metrics structure",
        ));
    }
    let request = admission::preflight(input, rest)?;
    let threshold = admission::gather_threshold(&request).await?;
    let (y, t, y_final) = match request.form {
        admission::Form::Model(model) => model_response(runtime.as_ref(), model).await?,
        admission::Form::Samples(samples) => {
            let admission::Samples { y, time, y_final } = *samples;
            admission::gather_samples(y, time, y_final).await?
        }
    };
    metrics::to_value(metrics::compute(&y, &t, y_final, threshold)?)
}

async fn model_response(
    runtime: Option<&crate::context::RuntimeContext>,
    system: Value,
) -> BuiltinResult<(Vec<f64>, Vec<f64>, f64)> {
    // The response boundary owns exact coefficient admission and one model
    // parse. The simulation service is pure and does not redispatch `step`.
    preflight_response_model_operand(&system, ResponseKind::StepInfo)?;
    let parsed = parse_admitted_model(runtime, system, ResponseKind::StepInfo).await?;
    preflight_host_time(None, StepSimulationCaller::StepInfo)?;
    let time = admit_time(
        None,
        parsed.real.sample_time(),
        StepSimulationCaller::StepInfo,
    )?;
    let response = simulate(&parsed.real, time, StepSimulationCaller::StepInfo)?;
    let final_value = model_final_value(&parsed.model, &response.y)?;
    Ok((response.y, response.t, final_value))
}

fn model_final_value(model: &TfModel, sampled: &[f64]) -> BuiltinResult<f64> {
    let gain = model.dc_gain()?;
    if gain.im == 0.0 && gain.re.is_finite() {
        Ok(gain.re)
    } else {
        sampled.last().copied().ok_or_else(|| {
            control_error(
                BUILTIN_NAME,
                "RunMat:stepinfo:Internal",
                "stepinfo: empty model response",
            )
        })
    }
}
