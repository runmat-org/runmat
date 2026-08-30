use super::model::BrowserExecutionCapabilities;
use super::resources::{
    browser_accelerators, browser_execution_host_inventory, browser_inventory, browser_request,
};
use runmat_execution::security::ExecutionTrustTier;
use runmat_execution_artifact::{ProgramExecutionRequest, ProgramExecutionResponse};

pub(crate) async fn execute_local_program(
    request: ProgramExecutionRequest,
) -> ProgramExecutionResponse {
    if let Err(message) = validate_browser_admission(&request) {
        return ProgramExecutionResponse::Failure { message };
    }
    runmat_vm::execute_program_request(request).await
}

fn validate_browser_admission(request: &ProgramExecutionRequest) -> Result<(), String> {
    request
        .validate_for_portable_host()
        .map_err(|error| format!("browser host rejected invalid program artifact: {error}"))?;
    let host_requirement = request
        .artifact
        .execution_host_requirement(
            &request.recipe,
            [ExecutionTrustTier::CustomerTrusted].into_iter().collect(),
        )
        .map_err(|error| format!("browser host rejected invalid program contract: {error}"))?;
    // Take one provider snapshot and use it for both semantic host admission
    // and physical resource placement. This keeps the two contracts aligned
    // even if the process-local provider registry changes between requests.
    let accelerators = browser_accelerators()
        .map_err(|error| format!("browser accelerator inventory is unavailable: {error}"))?;
    let host = browser_execution_host_inventory(&accelerators)
        .map_err(|error| format!("browser host inventory is unavailable: {error}"))?;
    host_requirement.is_satisfied_by(&host).map_err(|error| {
        let detail = request
            .artifact
            .executable_unit()
            .ok()
            .flatten()
            .map(|envelope| rejection_message(&envelope.manifest, &error.to_string()))
            .unwrap_or_else(|| {
                format!("browser host rejected unavailable executable capabilities: {error}")
            });
        detail
    })?;
    let resources = browser_request(
        BrowserExecutionCapabilities::default(),
        request.recipe.accelerators.clone(),
    );
    request
        .recipe
        .validate_resource_request(&resources)
        .map_err(|error| {
            format!("browser host rejected unavailable execution resources: {error}")
        })?;
    let inventory = browser_inventory(BrowserExecutionCapabilities::default(), accelerators);
    if !runmat_execution_runner::scheduler::fits(
        &inventory,
        &runmat_execution_runner::scheduler::ResourceAllocation::default(),
        &resources,
    ) {
        return Err("browser host rejected unavailable execution resources".into());
    }
    Ok(())
}

fn rejection_message(manifest: &runmat_execution::ExecutableUnitManifest, error: &str) -> String {
    let adapters = manifest
        .interop
        .adapters
        .iter()
        .map(|adapter| {
            if adapter.artifact_identities.is_empty() {
                adapter.adapter.as_str().to_owned()
            } else {
                format!(
                    "{} [{}]",
                    adapter.adapter,
                    adapter
                        .artifact_identities
                        .iter()
                        .map(|identity| identity.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
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
