//! Portable execution is a session invocation of an original compiler product.
use super::RunMatSession;
use crate::{ExecutableUnit, InvocationControl, ProcedureInvocation, ProcedureTarget};
use runmat_execution::ExecutableEntrypointKind;

impl RunMatSession {
    /// Invoke a complete compiler product through this session's existing host
    /// ports, language policy and frozen project. Artifact transport belongs to
    /// the caller; core owns the executable and invocation semantics.
    pub async fn invoke_portable_unit(
        &mut self,
        envelope: &runmat_execution::ExecutableUnitEnvelope,
        arguments: Vec<runmat_value::Value>,
        requested_outputs: usize,
        control: &InvocationControl,
    ) -> Result<runmat_runtime::execution::RootedValueSequence, String> {
        let unit = ExecutableUnit::from_portable_envelope(envelope)?;
        if unit.revision().program_revision.environment()
            != crate::program_environment(self.compat_mode())
        {
            return Err("portable program environment differs from the configured session".into());
        }
        if let Some(project) = self.project_revision() {
            let revision = &unit.revision().program_revision;
            if revision.graph_digest().bytes() != project.graph_digest.bytes()
                || revision.source_digest().bytes() != project.source_revision.bytes()
            {
                return Err(
                    "portable program revision differs from the installed frozen project".into(),
                );
            }
        }
        let target = match envelope.manifest.identity.entrypoint_kind {
            ExecutableEntrypointKind::Script => ProcedureTarget::Entrypoint,
            ExecutableEntrypointKind::DeclarationOnly => {
                return Err("declaration-only product has no invocable entrypoint".into())
            }
            ExecutableEntrypointKind::Function => ProcedureTarget::SemanticFunction(
                envelope
                    .manifest
                    .identity
                    .entrypoint_function
                    .ok_or("portable function product has no entrypoint function")?,
            ),
        };
        let result = self
            .invoke_executable(
                &unit,
                ProcedureInvocation {
                    target,
                    arguments: crate::ProcedureArguments::Source(arguments),
                    requested_outputs,
                },
                control,
            )
            .await
            .map_err(|error| error.to_string());
        result
    }
}

#[cfg(test)]
mod tests;
