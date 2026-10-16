use super::model::{BrowserExecutionCapabilities, BrowserWorkerTopology};
use super::resources::{
    browser_accelerators, browser_execution_host_inventory, browser_inventory, browser_request,
};
use runmat_execution::security::ExecutionTrustTier;
use runmat_execution_artifact::{ProgramExecutionRequest, ProgramExecutionResponse};

use super::request::BrowserWorkerRequest;

pub(super) async fn execute_local_worker_program(
    request: BrowserWorkerRequest,
) -> ProgramExecutionResponse {
    match request {
        BrowserWorkerRequest::Direct(request) => execute_local_program(request).await,
        BrowserWorkerRequest::Graph(request) => {
            if let Err(message) = validate_browser_graph_admission(&request) {
                return ProgramExecutionResponse::Failure { message };
            }
            runmat_vm::execute_graph_request(request).await
        }
        BrowserWorkerRequest::OwnedBuiltin(request) => execute_local_owned_builtin(request).await,
    }
}

async fn execute_local_owned_builtin(
    request: runmat_execution::OwnedBuiltinWorkRequest,
) -> ProgramExecutionResponse {
    if let Err(message) = validate_browser_owned_builtin_admission(&request) {
        return ProgramExecutionResponse::Failure { message };
    }
    match runmat_runtime::execution::execute_owned_builtin_work(request).await {
        runmat_execution::OwnedBuiltinWorkResponse::Success { outputs } => {
            ProgramExecutionResponse::Success { outputs }
        }
        runmat_execution::OwnedBuiltinWorkResponse::RuntimeFailure { failure } => {
            ProgramExecutionResponse::RuntimeFailure { failure }
        }
        runmat_execution::OwnedBuiltinWorkResponse::InfrastructureFailure { message } => {
            ProgramExecutionResponse::Failure { message }
        }
    }
}

fn validate_browser_owned_builtin_admission(
    request: &runmat_execution::OwnedBuiltinWorkRequest,
) -> Result<(), String> {
    request.validate_wire().map_err(|error| error.to_string())?;
    let assignment = request
        .assignment
        .as_ref()
        .ok_or_else(|| "owned builtin work has no physical worker assignment".to_string())?;
    if assignment.backend != runmat_execution::PoolBackend::BrowserWorkers {
        return Err("owned builtin work requires an isolated browser worker".into());
    }
    runmat_runtime::execution::admit_owned_builtin_work(request)?;
    let isolated_worker = BrowserExecutionCapabilities {
        topology: BrowserWorkerTopology::Coordinator,
        max_workers: 1,
        owned_builtin_work: true,
    };
    let inventory = browser_inventory(isolated_worker, Vec::new());
    if !runmat_execution_runner::scheduler::fits(
        &inventory,
        &runmat_execution_runner::scheduler::ResourceAllocation::default(),
        &request.resources,
    ) {
        return Err("browser worker lacks owned builtin resources".into());
    }
    Ok(())
}

fn validate_browser_graph_admission(
    request: &runmat_execution_artifact::GraphExecutionRequest,
) -> Result<(), String> {
    request
        .validate()
        .map_err(|error| format!("browser host rejected invalid graph artifact: {error}"))?;
    let host_requirement = request
        .execution_host_requirement([ExecutionTrustTier::CustomerTrusted].into_iter().collect())
        .map_err(|error| format!("browser host rejected invalid graph contract: {error}"))?;
    let accelerators = browser_accelerators()
        .map_err(|error| format!("browser accelerator inventory is unavailable: {error}"))?;
    let host = browser_execution_host_inventory(&accelerators)
        .map_err(|error| format!("browser host inventory is unavailable: {error}"))?;
    host_requirement.is_satisfied_by(&host).map_err(|error| {
        format!("browser host rejected unavailable graph capabilities: {error}")
    })?;
    let mut resources = browser_request(
        BrowserExecutionCapabilities::default(),
        request.recipe.accelerators.clone(),
    );
    let wire_bytes = serde_json::to_vec(request)
        .map_err(|error| format!("browser graph request cannot be encoded: {error}"))?;
    resources.memory_bytes = request
        .input_allocation_floor_bytes(wire_bytes.len())
        .map_err(|error| format!("browser graph demand is invalid: {error}"))?;
    resources.scratch_bytes = u64::try_from(wire_bytes.len())
        .map_err(|_| "browser graph wire length exceeds u64".to_string())?
        .max(1024 * 1024);
    request
        .recipe
        .validate_resource_request(&resources)
        .map_err(|error| format!("browser graph resources are unavailable: {error}"))?;
    let inventory = browser_inventory(BrowserExecutionCapabilities::default(), accelerators);
    if !runmat_execution_runner::scheduler::fits(
        &inventory,
        &runmat_execution_runner::scheduler::ResourceAllocation::default(),
        &resources,
    ) {
        return Err("browser graph resources are unavailable".into());
    }
    Ok(())
}

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
