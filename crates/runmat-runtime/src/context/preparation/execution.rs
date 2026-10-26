//! Retire begin/operand extraction before polling the selected body. The same
//! observation and root lease own body completion, adoption, failure and abort.
use super::*;
use futures::FutureExt;

struct Started {
    values: Vec<Value>,
    compound: Option<AdmittedBuiltinCompound>,
    numerical_output: Option<PreparedHostNumericalOutput>,
    // Retain roots/observation after extracted children on abandonment.
    prepared: PreparedBuiltinOperands,
}
pub(super) fn execute_prepared_resources<F, Fut>(
    mut prepared: PreparedBuiltinOperands,
    execute: F,
) -> impl Future<Output = Result<RootedValue, RuntimeError>>
where
    F: FnOnce(
        Vec<Value>,
        Option<AdmittedBuiltinCompound>,
        Option<PreparedHostNumericalOutput>,
    ) -> Fut,
    Fut: Future<Output = Result<RootedValue, RuntimeError>>,
{
    let recorder = prepared.provider_recorder();
    let no_replay = prepared.forbids_gpu_gather_retry();
    runmat_accelerate_api::observation::observe_provider_future(
        recorder,
        crate::execution::after_result(
            futures::future::lazy(move |_| {
                prepared.begin_execution();
                let compound = prepared.take_compound();
                let numerical_output = prepared.take_host_numerical_output();
                let values = prepared.take_operands();
                Ok::<_, RuntimeError>(Started {
                    values,
                    compound,
                    numerical_output,
                    prepared,
                })
            }),
            move |started| {
                let Started {
                    values,
                    compound,
                    numerical_output,
                    prepared,
                } = started;
                // The body error is DATA here: both success and failure must
                // run finish, rather than short-circuit the completion hook.
                crate::execution::after_result(
                    execute(values, compound, numerical_output).map(Ok::<_, RuntimeError>),
                    move |result| async move { finish(prepared, result, no_replay) },
                )
            },
        ),
    )
}
fn finish(
    mut prepared: PreparedBuiltinOperands,
    result: Result<RootedValue, RuntimeError>,
    no_replay: bool,
) -> Result<RootedValue, RuntimeError> {
    let cleanup = prepared.finish(result.as_ref().map(|value| value.value()));
    let mut result = crate::runtime_error::complete_with_cleanup(result, cleanup);
    if no_replay {
        if let Err(error) = &mut result {
            error.gpu_gather_retry = crate::GpuGatherRetry::Never;
        }
    }
    result
}
