//! Application stream ownership around the private control protocol.
use super::*;

impl IsolatedMexClient {
    pub async fn invoke(
        &mut self,
        arguments: &crate::call::RootedFunctionArguments,
        requested_outputs: usize,
        api: Option<MexApi>,
        tier: MexBinaryTier,
        runtime: RuntimeContext,
        timeout: Option<Duration>,
    ) -> Result<RootedMexInvocation, RuntimeError> {
        self.settle_pending().await?;
        let output = self.output.clone();
        isolation_output::drive(
            &output,
            Some(&runtime),
            Completion::AvailableOutput,
            async {
                self.invoke_protocol(
                    arguments,
                    requested_outputs,
                    api,
                    tier,
                    runtime.clone(),
                    timeout,
                )
                .await
                .map_err(wire_runtime_error)
            },
        )
        .await
    }

    pub async fn clear(
        &mut self,
        runtime: RuntimeContext,
        timeout: Option<Duration>,
    ) -> Result<MexLifecycleOutcome, RuntimeError> {
        self.settle_pending().await?;
        let output = self.output.clone();
        isolation_output::drive_selected(&output, Completion::AvailableOutput, async {
            self.clear_protocol(runtime, timeout)
                .await
                .map_err(wire_runtime_error)
        })
        .await
    }

    pub(in crate::foreign::mex) async fn shutdown(
        &mut self,
        runtime: RuntimeContext,
        timeout: Option<Duration>,
    ) -> IsolatedMexRetirement {
        if self.output.failed() {
            let error = wire_error(
                "RunMat:MEX:HostTransport",
                "MEX application output previously failed",
                None,
            );
            let _ = self.poison_transport(error).await;
        }
        if self.terminal_error.is_none() {
            if let Err(error) = self.settle_pending().await {
                return IsolatedMexRetirement::Retained(error);
            }
        }
        let output = self.output.clone();
        let result =
            isolation_output::drive_selected(&output, Completion::AvailableOutput, async {
                match self.shutdown_protocol(runtime, timeout).await {
                    IsolatedMexRetirement::Retired(result) => result.map_err(wire_runtime_error),
                    IsolatedMexRetirement::Retained(error) => Err(wire_runtime_error(error)),
                }
            })
            .await;
        if self.tree_retired {
            let drained = tokio::time::timeout(
                timeout.unwrap_or(DEFAULT_LIFECYCLE_TIMEOUT),
                isolation_output::drive_selected(&output, Completion::EndOfStreams, async {
                    Ok(())
                }),
            )
            .await
            .unwrap_or_else(|_| {
                Err(crate::build_runtime_error(
                    "MEX application output did not close after retirement",
                )
                .with_identifier("RunMat:MEX:Timeout")
                .build())
            });
            IsolatedMexRetirement::Retired(crate::runtime_error::complete_with_cleanup(
                result, drained,
            ))
        } else {
            IsolatedMexRetirement::Retained(result.err().unwrap_or_else(|| {
                crate::build_runtime_error("MEX host retirement was not proved")
                    .with_identifier("RunMat:MEX:Lifecycle")
                    .build()
            }))
        }
    }

    pub async fn service_ready_background(&mut self) -> Result<bool, RuntimeError> {
        if self.output.failed() && self.terminal_error.is_none() {
            let error = wire_error(
                "RunMat:MEX:HostTransport",
                "MEX application output previously failed",
                None,
            );
            let _ = self.poison_transport(error.clone()).await;
            return Err(wire_runtime_error(error));
        }
        let output = self.output.clone();
        isolation_output::drive_selected(&output, Completion::AvailableOutput, async {
            self.service_ready_background_protocol()
                .await
                .map_err(wire_runtime_error)
        })
        .await
    }

    pub(in crate::foreign::mex) fn poll_application_output(
        &self,
        context: &mut Context<'_>,
    ) -> Result<bool, RuntimeError> {
        if self.terminal_error.is_some() || self.output.failed() {
            return Ok(false);
        }
        self.output.poll_available(context)
    }
}
