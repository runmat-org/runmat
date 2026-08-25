use std::collections::BTreeSet;

use runmat_execution_artifact::{
    ExecutableForm, ProgramExecutionRequest, ProgramExecutionResponse,
};
use runmat_types::{CapabilityRequirement, CapabilitySet};

pub(crate) async fn execute_local_program(
    request: ProgramExecutionRequest,
) -> ProgramExecutionResponse {
    if request.validate_for_portable_host().is_ok()
        && request.artifact.form == ExecutableForm::ExecutableUnitV3
    {
        if let Ok(Some(envelope)) = request.artifact.executable_unit() {
            if let Err(error) = envelope
                .manifest
                .validate_capabilities_for(&browser_capabilities())
            {
                return ProgramExecutionResponse::Failure {
                    message: rejection_message(&envelope.manifest, &error.to_string()),
                };
            }
        }
    }
    runmat_vm::execute_program_request(request).await
}

fn browser_capabilities() -> CapabilitySet {
    CapabilitySet(BTreeSet::from([
        CapabilityRequirement::HostRuntime,
        CapabilityRequirement::Filesystem,
        CapabilityRequirement::Network,
        CapabilityRequirement::UserInterface,
        CapabilityRequirement::Accelerator,
        CapabilityRequirement::ParallelRuntime,
    ]))
}

fn rejection_message(manifest: &runmat_execution::ExecutableUnitManifest, error: &str) -> String {
    let adapters = manifest
        .interop
        .adapters
        .iter()
        .map(|adapter| {
            if adapter.artifact_identities.is_empty() {
                adapter.adapter.clone()
            } else {
                format!(
                    "{} [{}]",
                    adapter.adapter,
                    adapter.artifact_identities.join(", ")
                )
            }
        })
        .collect::<Vec<_>>();
    let interop = if adapters.is_empty() {
        String::new()
    } else {
        format!("; foreign adapters: {}", adapters.join("; "))
    };
    format!("browser host rejected unavailable executable capabilities: {error}{interop}")
}
