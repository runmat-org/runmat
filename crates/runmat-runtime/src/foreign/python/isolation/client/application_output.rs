//! Application stream ownership around the private control protocol.
use super::*;

impl IsolatedPythonClient {
    pub async fn invoke(
        &mut self,
        runtime: RuntimeContext,
        call: ForeignCall,
        handles: &ForeignHandleRegistry,
        host_identity: &str,
        released_local_handles: Vec<u64>,
        nested: &NestedPythonQueue,
    ) -> Result<RootedValues, RuntimeError> {
        if self.output.failed() {
            return Err(console_transport_error(
                build_runtime_error("foreign output transport previously failed").build(),
            ));
        }
        let output = self.output.clone();
        let result = isolation_output::drive(
            &output,
            Some(&runtime),
            Completion::AvailableOutput,
            self.invoke_protocol(
                runtime.clone(),
                call,
                handles,
                host_identity,
                released_local_handles,
                nested,
            ),
        )
        .await;
        result
    }

    pub fn transport_output_unavailable(&self) -> bool {
        self.output.failed()
    }

    pub async fn shutdown(
        &mut self,
        runtime: Option<&RuntimeContext>,
        released: Vec<u64>,
    ) -> Result<(), PythonWireError> {
        match self.shutdown_retirement(runtime, released).await {
            PythonClientRetirement::Retired { result, .. } => result,
            PythonClientRetirement::Retained { retirement, .. } => Err(retirement),
        }
    }

    pub async fn shutdown_retirement(
        &mut self,
        runtime: Option<&RuntimeContext>,
        released: Vec<u64>,
    ) -> PythonClientRetirement {
        let output = self.output.clone();
        if output.failed() && matches!(self.shutdown_progress, ShutdownProgress::Open) {
            self.shutdown_progress = ShutdownProgress::Protocol(Err(wire_error(
                "RunMat:Python:HostTransport",
                "Python application output previously failed",
            )));
        }
        if matches!(self.shutdown_progress, ShutdownProgress::Open) {
            // A cancelled shutdown cannot resend an unknown-effect record.
            self.shutdown_progress = ShutdownProgress::Attempted;
            let result =
                isolation_output::drive(&output, runtime, Completion::AvailableOutput, async {
                    self.shutdown_protocol(released)
                        .await
                        .map_err(runtime_error)
                })
                .await
                .map_err(runtime_wire_error);
            self.shutdown_progress = ShutdownProgress::Protocol(result);
        }
        let retirement = self.child.terminate_contained_child().await;
        let drained = if retirement.is_ok() {
            tokio::time::timeout(
                DEFAULT_SHUTDOWN_TIMEOUT,
                isolation_output::drive(&output, runtime, Completion::EndOfStreams, async {
                    Ok(())
                }),
            )
            .await
            .unwrap_or_else(|_| {
                Err(runtime_error(wire_error(
                    "RunMat:Python:Timeout",
                    "Python application output did not close after retirement",
                )))
            })
        } else {
            Ok(())
        };
        match retirement {
            Ok(scope) => {
                let result = match std::mem::replace(
                    &mut self.shutdown_progress,
                    ShutdownProgress::Attempted,
                ) {
                    ShutdownProgress::Protocol(result) => result,
                    ShutdownProgress::Attempted => Err(wire_error(
                        "RunMat:Python:ShutdownInterrupted",
                        "isolated Python shutdown was interrupted before retirement",
                    )),
                    ShutdownProgress::Open => unreachable!("shutdown attempted before proof"),
                };
                let result = crate::runtime_error::complete_with_cleanup(
                    result.map_err(runtime_error),
                    drained,
                )
                .map_err(runtime_wire_error);
                PythonClientRetirement::Retired { scope, result }
            }
            Err(error) => PythonClientRetirement::Retained {
                retirement: wire_error("RunMat:Python:RetirementUnverified", error.to_string()),
                prior_protocol: match &self.shutdown_progress {
                    ShutdownProgress::Protocol(Err(prior)) => Some(prior.clone()),
                    _ => None,
                },
            },
        }
    }
}
