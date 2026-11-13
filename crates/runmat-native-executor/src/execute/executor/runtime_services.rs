//! One admission check for the full selected effect-service closure.
use super::*;

impl NativeExecutor {
    pub(super) fn require_runtime_services(
        &self,
        runtime: &RuntimeContext,
    ) -> NativeExecutorResult<()> {
        if self.selected_deterministic
            && !self.selected_services.iter().any(|issuer| {
                issuer.binding().kind
                    == runmat_execution::host::NativeDeploymentServiceKind::Effect(
                        runmat_types::EffectKind::Clock,
                    )
            })
            && runtime.service_ports().timing().is_ok()
        {
            return Err(NativeExecutorError::Executable(
                "deterministic selected invocation retained an unadmitted timing port".into(),
            ));
        }
        for issuer in &self.selected_services {
            match issuer.binding().kind {
                runmat_execution::host::NativeDeploymentServiceKind::SourceEvaluation => {
                    runtime
                        .service_ports()
                        .require_selected_source_evaluation(issuer)?;
                }
                runmat_execution::host::NativeDeploymentServiceKind::Effect(
                    runmat_types::EffectKind::Clock,
                ) => {
                    let invocation = self.selected_clock_invocation.as_ref().ok_or_else(|| {
                        NativeExecutorError::Executable(
                            "selected Clock lacks exact invocation inputs".into(),
                        )
                    })?;
                    runtime
                        .service_ports()
                        .require_selected_timing(issuer, invocation)?;
                }
                runmat_execution::host::NativeDeploymentServiceKind::Effect(
                    runmat_types::EffectKind::Network,
                ) => {
                    let invocation = self.selected_local_invocation.ok_or_else(|| {
                        NativeExecutorError::Executable(
                            "selected Network lacks exact invocation identity".into(),
                        )
                    })?;
                    runtime
                        .service_ports()
                        .require_selected_network(issuer, invocation)?;
                }
                runmat_execution::host::NativeDeploymentServiceKind::Effect(
                    runmat_types::EffectKind::UserInterface,
                ) => {
                    let invocation = self.selected_local_invocation.ok_or_else(|| {
                        NativeExecutorError::Executable(
                            "selected linked host lacks exact invocation identity".into(),
                        )
                    })?;
                    runtime
                        .service_ports()
                        .require_selected_linked_host(issuer, invocation)?;
                }
                runmat_execution::host::NativeDeploymentServiceKind::Effect(
                    runmat_types::EffectKind::HostCallback,
                ) => {
                    let invocation = self.selected_local_invocation.ok_or_else(|| {
                        NativeExecutorError::Executable(
                            "selected callable lacks exact invocation identity".into(),
                        )
                    })?;
                    runtime
                        .service_ports()
                        .require_selected_linked_callback(issuer, invocation)?;
                }
                runmat_execution::host::NativeDeploymentServiceKind::HostedDynamicDispatch => {
                    let invocation = self.selected_local_invocation.ok_or_else(|| {
                        NativeExecutorError::Executable(
                            "selected dynamic bindings lack exact invocation identity".into(),
                        )
                    })?;
                    runtime
                        .service_ports()
                        .require_selected_linked_dynamic(issuer, invocation)?;
                }
                runmat_execution::host::NativeDeploymentServiceKind::Effect(
                    runmat_types::EffectKind::WorkspaceRead
                    | runmat_types::EffectKind::WorkspaceWrite
                    | runmat_types::EffectKind::EnvironmentRead
                    | runmat_types::EffectKind::EnvironmentWrite
                    | runmat_types::EffectKind::FilesystemRead
                    | runmat_types::EffectKind::FilesystemWrite
                    | runmat_types::EffectKind::Randomness,
                ) => {
                    let invocation = self.selected_local_invocation.ok_or_else(|| {
                        NativeExecutorError::Executable(
                            "selected local service lacks exact invocation identity".into(),
                        )
                    })?;
                    runtime
                        .service_ports()
                        .require_selected_local_service(issuer, invocation)?;
                }
                runmat_execution::host::NativeDeploymentServiceKind::Effect(
                    runmat_types::EffectKind::MaySuspend
                    | runmat_types::EffectKind::MayThrow
                    | runmat_types::EffectKind::Unknown,
                ) => {
                    return Err(NativeExecutorError::Executable(
                        "selected native runtime service is unsupported".into(),
                    ))
                }
            }
        }
        if self
            .required_runtime_services
            .contains(&runmat_execution::host::NativeDeploymentServiceKind::SourceEvaluation)
            && !self.selected_services.iter().any(|issuer| {
                issuer.binding().kind
                    == runmat_execution::host::NativeDeploymentServiceKind::SourceEvaluation
            })
        {
            runtime
                .service_ports()
                .require_source_evaluation("execute native source evaluation")
                .map_err(|error| NativeExecutorError::from(error.into_runtime_error()))?;
        }
        Ok(())
    }
}
