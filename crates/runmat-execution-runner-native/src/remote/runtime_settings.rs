//! Worker allocation limits always dominate project performance preferences.
use runmat_execution::resource::ResourceRequest;
use runmat_execution_artifact::ExecutionBundle;

pub(super) fn admit(bundle: &ExecutionBundle, resources: &ResourceRequest) -> Result<(), String> {
    let config = runmat_core::cluster_settings::runtime_config(bundle)?
        .ok_or("cluster execution requires digest-bound project-only runtime settings")?;
    admit_config(&config, resources)
}

pub(super) fn admit_graph(
    graph: &runmat_execution_artifact::GraphExecutionRequest,
    resources: &ResourceRequest,
) -> Result<(), String> {
    let config = runmat_core::cluster_settings::validate_graph_policy(graph)?
        .runtime_config()
        .map_err(|error| error.to_string())?;
    admit_config(&config, resources)
}

pub(crate) fn admit_config(
    config: &runmat_config::runtime::RunMatRuntimeConfig,
    resources: &ResourceRequest,
) -> Result<(), String> {
    admit_limits(
        config,
        u64::from(resources.cpu_millicores),
        resources.memory_bytes,
        !resources.accelerators.is_empty(),
    )
}

pub(super) fn admit_limits(
    config: &runmat_config::runtime::RunMatRuntimeConfig,
    cpu_millicores: u64,
    memory_bytes: u64,
    has_accelerator: bool,
) -> Result<(), String> {
    let memory =
        usize::try_from(memory_bytes).map_err(|_| "worker memory allocation exceeds this host")?;
    let threads = usize::try_from(cpu_millicores.div_ceil(1_000))
        .map_err(|_| "worker CPU allocation exceeds this host")?;
    if config.accelerate.enabled && !config.accelerate.allow_inprocess_fallback && !has_accelerator
    {
        return Err("project requires GPU acceleration without CPU fallback, but this attempt has no admitted accelerator".into());
    }
    // Device assignment is owned by the existing worker host inventory and
    // request assignment. Project policy is never authority to select another
    // GPU or enlarge this request's accelerator/CPU/memory reservation.
    runmat_core::RunMatSession::configure_gc_with_resource_limits(config, memory, threads)
        .map_err(|error| error.to_string())
}

#[cfg(test)]
pub(crate) fn fixture_policy(
    revision: &runmat_execution::ProgramRevision,
) -> runmat_execution_artifact::LogicalObject {
    // Fixture programs explicitly choose their compiler mode; construct the
    // corresponding project document rather than consulting the operator's home.
    let source = format!(
        "[runtime.language]\ncompat = '{}'\n",
        revision.environment().compatibility_mode.as_ref()
    );
    let policy = runmat_config::runtime::ClusterRuntimePolicy::from_project_document(
        &source,
        runmat_config::document::RunmatConfigFormat::Toml,
    )
    .unwrap();
    runmat_core::cluster_settings::capture_project_policy(&policy).unwrap()
}
