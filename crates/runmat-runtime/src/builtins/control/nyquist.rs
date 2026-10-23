//! SISO Nyquist response admission, evaluation and optional presentation.

mod admission;
mod frequency;
#[cfg(feature = "plot-core")]
mod plot;
mod solver;
#[cfg(test)]
mod tests;

use runmat_builtins::NYQUIST_EXTENSIONS;
use runmat_macros::runtime_builtin;
use runmat_value::{Tensor, Value};

use crate::builtins::common::spec::{
    BroadcastSemantics, BuiltinFusionSpec, BuiltinGpuSpec, ConstantStrategy, GpuOpKind,
    ReductionNaN, ResidencyPolicy, ShapeRequirements,
};
use crate::builtins::control::tf_model::{control_error, TfModel};
use crate::BuiltinResult;

const BUILTIN_NAME: &str = "nyquist";
#[cfg(all(test, feature = "plot-core"))]
const DEFAULT_FREQUENCY_POINTS: usize = 200;

#[runmat_macros::register_gpu_spec(builtin_path = "crate::builtins::control::nyquist")]
pub const GPU_SPEC: BuiltinGpuSpec = BuiltinGpuSpec {
    name: "nyquist",
    op_kind: GpuOpKind::Custom("control-nyquist-frequency-response"),
    supported_precisions: &[],
    broadcast: BroadcastSemantics::None,
    provider_hooks: &[],
    constant_strategy: ConstantStrategy::InlineLiteral,
    residency: ResidencyPolicy::GatherImmediately,
    nan_mode: ReductionNaN::Include,
    two_pass_threshold: None,
    workgroup_size: None,
    accepts_nan_mode: false,
    notes: "Frequency evaluation is a host control-model operation; admitted consumed resident values gather through their owner.",
};

#[runmat_macros::register_fusion_spec(builtin_path = "crate::builtins::control::nyquist")]
pub const FUSION_SPEC: BuiltinFusionSpec = BuiltinFusionSpec {
    name: "nyquist",
    shape: ShapeRequirements::Any,
    constant_strategy: ConstantStrategy::InlineLiteral,
    elementwise: None,
    reduction: None,
    emits_nan: false,
    notes: "Frequency-response vectors materialize on the host and end numeric fusion.",
};

#[runtime_builtin(
    name = "nyquist",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::control::nyquist"
)]
async fn nyquist_builtin(
    runtime: Option<crate::context::RuntimeContext>,
    system: Value,
    rest: crate::call::BuiltinArguments,
) -> BuiltinResult<Value> {
    if rest.len() > 1 || crate::output_count::current_output_count().is_some_and(|n| n > 3) {
        return Err(control_error(
            BUILTIN_NAME,
            "RunMat:nyquist:InvalidArgument",
            "nyquist accepts one model, optional frequency vector and at most three outputs",
        ));
    }
    let frequency = rest.first();
    admission::preflight(&system, frequency)?;
    let model = admission::parse_model(runtime.as_ref(), system).await?;
    let w = frequency::parse(&model, frequency).await?;
    let response = solver::evaluate(&model, w)?;

    if let Some(n) = crate::output_count::current_output_count() {
        if n == 0 {
            render_plot(&response).await?;
            return Ok(Value::output_list(Vec::new()));
        }
        return Ok(Value::output_list(response.outputs(n)?));
    }
    if crate::output_context::requested_output_count() == Some(0) {
        render_plot(&response).await?;
        return Ok(Value::output_list(Vec::new()));
    }
    response.re_value()
}

#[cfg(feature = "plot-core")]
async fn render_plot(response: &solver::NyquistResponse) -> BuiltinResult<()> {
    plot::render(response).await
}

#[cfg(not(feature = "plot-core"))]
async fn render_plot(_response: &solver::NyquistResponse) -> BuiltinResult<()> {
    Err(crate::plotting_hooks::unavailable_presentation())
}

fn column_tensor(data: Vec<f64>) -> BuiltinResult<Value> {
    let rows = data.len();
    shaped_tensor(data, vec![rows, 1])
}

#[cfg(feature = "plot-core")]
fn row_tensor(data: Vec<f64>) -> BuiltinResult<Value> {
    let columns = data.len();
    shaped_tensor(data, vec![1, columns])
}

fn shaped_tensor(data: Vec<f64>, shape: Vec<usize>) -> BuiltinResult<Value> {
    Tensor::new(data, shape).map(Value::Tensor).map_err(|err| {
        control_error(
            BUILTIN_NAME,
            "RunMat:nyquist:Internal",
            format!("nyquist: failed to materialize response: {err}"),
        )
    })
}
