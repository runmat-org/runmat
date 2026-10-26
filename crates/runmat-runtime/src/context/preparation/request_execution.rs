//! Service/host-output preparation retires before the selected body is polled.
use super::*;
use futures::future::Either;

struct Prepared {
    host_output: Option<PreparedHostNumericalOutput>,
    prepared: PreparedBuiltinOperands,
}
pub(super) fn execute_request_resources<'a, F, Fut>(
    context: Option<&'a RuntimeContext>,
    request: AdmittedBuiltinPreparation,
    consumes_host_numerical_output: bool,
    execute: F,
) -> impl Future<Output = Result<RootedValue, RuntimeError>> + 'a
where
    F: FnOnce(
            Vec<Value>,
            Option<AdmittedBuiltinCompound>,
            Option<PreparedHostNumericalOutput>,
        ) -> Fut
        + 'a,
    Fut: Future<Output = Result<RootedValue, RuntimeError>> + 'a,
{
    crate::execution::after_result(
        prepare(context, request, consumes_host_numerical_output),
        move |Prepared {
                  host_output,
                  prepared,
              }| {
            execute_prepared_resources(prepared, move |values, compound, numerical_output| {
                // The no-service fallback has no observation/output. Refuse
                // contradictory delivery before constructing the body.
                if host_output.is_some() && numerical_output.is_some() {
                    Either::Left(std::future::ready(Err(missing_host_numerical_consumer())))
                } else {
                    Either::Right(execute(values, compound, numerical_output.or(host_output)))
                }
            })
        },
    )
}
async fn prepare(
    context: Option<&RuntimeContext>,
    request: AdmittedBuiltinPreparation,
    consumes_host_numerical_output: bool,
) -> Result<Prepared, RuntimeError> {
    match context {
        Some(context) => {
            // Preserve the original no-service finite-output authority. A
            // service remains solely responsible for its own output delivery.
            let host_output = if consumes_host_numerical_output
                && context.service_ports().acceleration().is_none()
            {
                use super::super::host_allocation::{
                    host_allocation_runtime_error, HostAllocationPolicy, HostAllocationReceipt,
                };
                let error = |cause| {
                    host_allocation_runtime_error(
                        Some(context),
                        "RunMat:Placement:HostNumericalOutput",
                        "admitted host numerical output preparation failed",
                        cause,
                    )
                };
                match HostAllocationReceipt::policy(Some(context)).map_err(error)? {
                    HostAllocationPolicy::Managed => request
                        .prepare_host_numerical_output(context)
                        .map_err(error)?,
                    HostAllocationPolicy::Unbounded => None,
                }
            } else {
                None
            };
            let prepared = prepare_admitted(context, request).await?;
            Ok(Prepared {
                host_output,
                prepared,
            })
        }
        None => {
            let prepared = request
                .into_operands()
                .with_unbounded_values(PreparedBuiltinOperands::passthrough)??;
            Ok(Prepared {
                host_output: None,
                prepared,
            })
        }
    }
}
