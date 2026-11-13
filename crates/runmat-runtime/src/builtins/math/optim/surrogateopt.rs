//! Surrogate optimization over one measured history and original domain.
mod affine;
mod campaign;
mod candidates;
mod checkpoint;
mod construction;
mod construction_phase;
mod controller;
mod crossover;
mod dispatch;
mod domain;
mod driver;
mod evaluation;
mod frame;
mod gss;
mod history;
mod incumbent;
mod initial;
mod inputs;
mod linear_view;
mod merit;
mod model_phase;
mod multistart;
mod notice;
mod numeric;
mod observation;
mod observers;
mod options;
mod oracle;
mod parameters;
mod phase;
mod preparation;
mod proposal;
mod public_values;
mod publication;
mod queue;
mod rbf;
mod refinement;
mod region;
mod sampler_pool;
mod sampling;
mod schedule;
mod spacing;
#[cfg(test)]
mod tests;
mod tree;

const CONTROL_STORAGE: super::common::retained_control::Policy =
    super::common::retained_control::Policy {
        identifier: "RunMat:surrogateopt:ControlAllocation",
        operation: "surrogate optimization control state allocation",
    };

const NAME: &str = "surrogateopt";
const IDENTITY: runmat_builtins::BuiltinCatalogIdentity =
    runmat_builtins::BuiltinCatalogIdentity { name: NAME };
use runmat_builtins::SURROGATEOPT_ERROR_INVALID_ARGUMENT as INVALID_ARGUMENT;
fn invalid(message: impl std::fmt::Display) -> crate::RuntimeError {
    crate::build_runtime_error(format!("surrogateopt: {message}"))
        .with_builtin(NAME)
        .with_identifier(
            INVALID_ARGUMENT
                .identifier
                .expect("fixed invalid identifier"),
        )
        .with_gpu_gather_retry(crate::GpuGatherRetry::Never)
        .build()
}

#[runmat_macros::runtime_builtin(
    name = "surrogateopt",
    binding_variant = "default",
    context = true,
    builtin_path = "crate::builtins::math::optim::surrogateopt"
)]
async fn registered(
    runtime: Option<crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValueSequence> {
    execute(runtime.as_ref(), args).await.map_err(|mut error| {
        error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        error
    })
}
async fn execute(
    runtime: Option<&crate::context::RuntimeContext>,
    args: crate::call::BuiltinArguments,
) -> crate::BuiltinResult<crate::execution::RootedValueSequence> {
    crate::context::host_allocation::require_unbounded_workspace(runtime, NAME,
        "finite host-memory execution has no complete surrogate numerical, callback, checkpoint and output workspace bound")?;
    crate::output_count::validate_bounded_outputs_in(
        &runmat_builtins::SURROGATEOPT_DESCRIPTOR,
        &INVALID_ARGUMENT,
        NAME,
        runtime,
    )?;
    let _original = args.fork();
    let mut cooperation =
        crate::builtins::math::optim::common::ScalarCooperation::for_builtin(IDENTITY, runtime);
    cooperation.completed(0).await?;
    let form = inputs::grammar(&args, &cooperation)?;
    let plan = preparation::prepare(form, &mut cooperation).await?;
    let result = phase::launch(plan, &mut cooperation)?.await;
    result
}
