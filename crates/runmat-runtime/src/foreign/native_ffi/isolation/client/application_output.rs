//! Application stream ownership around the private control protocol.
use super::*;

impl NativeFfiHostClient {
    pub async fn load_project(
        &mut self,
        runtime: RuntimeContext,
        request: NativeProjectLoadRequest,
        handles: &ForeignHandleRegistry,
        host_identity: &str,
        timeout: Option<Duration>,
    ) -> Result<(), RuntimeError> {
        let output = self.output.clone();
        isolation_output::drive(
            &output,
            Some(&runtime),
            Completion::AvailableOutput,
            self.load_project_protocol(runtime.clone(), request, handles, host_identity, timeout),
        )
        .await
    }

    pub async fn invoke(
        &mut self,
        runtime: RuntimeContext,
        call: ForeignCall,
        handles: &ForeignHandleRegistry,
        host_identity: &str,
        released_local_handles: Vec<u64>,
        timeout: Option<Duration>,
    ) -> Result<RootedValues, RuntimeError> {
        let output = self.output.clone();
        isolation_output::drive(
            &output,
            Some(&runtime),
            Completion::AvailableOutput,
            self.invoke_protocol(
                runtime.clone(),
                call,
                handles,
                host_identity,
                released_local_handles,
                timeout,
            ),
        )
        .await
    }

    pub async fn shutdown_retirement(
        &mut self,
        runtime: Option<&RuntimeContext>,
        released_local_handles: Vec<u64>,
    ) -> NativeClientRetirement {
        let output = self.output.clone();
        if output.failed() && matches!(self.shutdown_progress, ShutdownProgress::Open) {
            self.shutdown_progress = ShutdownProgress::Protocol(Err(wire_error(
                "RunMat:NativeFFI:HostTransport",
                "native application output previously failed",
            )));
        }
        if matches!(self.shutdown_progress, ShutdownProgress::Open) {
            // If this future is cancelled after a write, retry must not send
            // another ambiguous Shutdown record to the same child.
            self.shutdown_progress = ShutdownProgress::Attempted;
            let result =
                isolation_output::drive(&output, runtime, Completion::AvailableOutput, async {
                    self.shutdown_protocol(released_local_handles)
                        .await
                        .map_err(runtime_error)
                })
                .await
                .map_err(NativeWireError::from_runtime);
            self.shutdown_progress = ShutdownProgress::Protocol(result);
        }
        let retirement = self.abort_host().await;
        let drained = if retirement.is_ok() {
            shutdown_step(
                isolation_output::drive(&output, runtime, Completion::EndOfStreams, async {
                    Ok(())
                }),
                "native application output retirement",
            )
            .await
            .and_then(|result| result.map_err(NativeWireError::from_runtime))
        } else {
            Ok(())
        };
        match retirement {
            Ok(scope) => {
                match std::mem::replace(&mut self.shutdown_progress, ShutdownProgress::Attempted) {
                    ShutdownProgress::Protocol(result) => {
                        let result = crate::runtime_error::complete_with_cleanup(
                            result.map_err(runtime_error),
                            drained.map_err(runtime_error),
                        )
                        .map_err(NativeWireError::from_runtime);
                        NativeClientRetirement::Retired { scope, result }
                    }
                    ShutdownProgress::Attempted => NativeClientRetirement::Retired {
                        scope,
                        result: if matches!(
                            self.user_state,
                            NativeUserState::ProjectLoadUnverifiedWithoutUser
                        ) {
                            drained
                        } else {
                            Err(wire_error(
                                "RunMat:NativeFFI:ShutdownInterrupted",
                                "isolated native-library operation was interrupted before retirement",
                            ))
                        },
                    },
                    ShutdownProgress::Open => unreachable!("shutdown was attempted before proof"),
                }
            }
            Err(error) => NativeClientRetirement::Retained {
                retirement: wire_error("RunMat:NativeFFI:RetirementUnverified", error.to_string()),
                // Keep the original phase for a same-child retry. The clone
                // supplies diagnostic ownership without resending Shutdown.
                prior_protocol: match &self.shutdown_progress {
                    ShutdownProgress::Protocol(Err(prior)) => Some(prior.clone()),
                    _ => None,
                },
            },
        }
    }
}
