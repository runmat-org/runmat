//! Worker allocation limits always dominate project performance preferences.
use runmat_execution::resource::ResourceRequest;
use runmat_execution_artifact::ExecutionBundle;

pub(super) fn admit(
    bundle: &ExecutionBundle,
    resources: &ResourceRequest,
    assignment: Option<&runmat_execution::ProgramExecutionAssignment>,
) -> Result<(), String> {
    let config = runmat_core::cluster_settings::runtime_config(bundle)?
        .ok_or("cluster execution requires digest-bound project-only runtime settings")?;
    admit_config(&config, resources, assignment)
}

pub(super) fn admit_graph(
    graph: &runmat_execution_artifact::GraphExecutionRequest,
    resources: &ResourceRequest,
) -> Result<(), String> {
    let config = runmat_core::cluster_settings::validate_graph_policy(graph)?
        .runtime_config()
        .map_err(|error| error.to_string())?;
    admit_config(&config, resources, graph.assignment.as_ref())
}

pub(crate) fn admit_config(
    config: &runmat_config::runtime::RunMatRuntimeConfig,
    resources: &ResourceRequest,
    assignment: Option<&runmat_execution::ProgramExecutionAssignment>,
) -> Result<(), String> {
    admit_assigned_accelerators(config, resources, assignment)?;
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
    // Lack of a GPU never forbids ordinary CPU computation. Actual GPU
    // operations are fenced by the invocation context and exact assignment.
    if has_accelerator {
        admit_fixed_adapter_preferences(config)?;
    }
    // Device assignment is owned by the existing worker host inventory and
    // request assignment. Project policy is never authority to select another
    // GPU or enlarge this request's accelerator/CPU/memory reservation.
    runmat_core::RunMatSession::configure_gc_with_resource_limits(config, memory, threads)
        .map_err(|error| error.to_string())
}

fn admit_fixed_adapter_preferences(
    config: &runmat_config::runtime::RunMatRuntimeConfig,
) -> Result<(), String> {
    use runmat_config::runtime::{AccelPowerPreference, AccelerateProviderPreference};
    let acceleration = &config.accelerate;
    if !acceleration.enabled || acceleration.provider == AccelerateProviderPreference::InProcess {
        return Err("project disables GPU execution but this attempt reserves an accelerator; submit a CPU request".into());
    }
    if acceleration.wgpu_power_preference != AccelPowerPreference::Auto
        || acceleration.wgpu_force_fallback_adapter
    {
        return Err("fixed worker accelerator assignments do not attest WGPU power class or fallback-adapter identity; use power=auto and force_fallback_adapter=false, or request an inventory with that attestation".into());
    }
    Ok(())
}

fn admit_assigned_accelerators(
    config: &runmat_config::runtime::RunMatRuntimeConfig,
    resources: &ResourceRequest,
    assignment: Option<&runmat_execution::ProgramExecutionAssignment>,
) -> Result<(), String> {
    if resources.accelerators.is_empty() {
        return Ok(());
    }
    admit_fixed_adapter_preferences(config)?;
    let assignment = assignment
        .ok_or("GPU execution requires an exact authenticated worker device assignment")?;
    assignment
        .resources
        .validate_for_request(resources)
        .map_err(|error| error.to_string())?;
    let providers = runmat_accelerate_api::registered_providers();
    for lease in &assignment.resources.accelerator_leases {
        let compatible = providers.iter().any(|provider| {
            runmat_accelerate_api::provider_matches_execution_lease(*provider, lease)
                && (config.accelerate.provider
                    != runmat_config::runtime::AccelerateProviderPreference::Wgpu
                    || provider.wgpu_device_access().is_some())
        });
        if !compatible {
            return Err("project provider preference is incompatible with the exact assigned worker accelerator; no process-global provider substitution is permitted".into());
        }
    }
    Ok(())
}

/// Synchronous domain kernels use the same invocation provider fence as Core.
/// The scope is created on the executing thread, never carried across awaits.
pub(crate) fn with_domain_policy<R>(
    config: &runmat_config::runtime::RunMatRuntimeConfig,
    assignment: Option<runmat_execution::ProgramExecutionAssignment>,
    body: impl FnOnce() -> R,
) -> R {
    let runtime = runmat_runtime::context::RuntimeContext::new(std::rc::Rc::new(
        runmat_runtime::execution::RuntimeExecutionService::new(),
    ))
    .with_acceleration_policy(runmat_core::cluster_settings::acceleration_policy(config));
    let _assignment = runtime.enter_execution_assignment(assignment);
    let _scope = runtime.enter();
    use runmat_analysis_fea::accelerator_admission::{
        with_accelerator_admission_policy, AcceleratorAdmissionPolicy,
    };
    let policy = if config.accelerate.allow_inprocess_fallback {
        AcceleratorAdmissionPolicy::FallbackToCpu
    } else {
        AcceleratorAdmissionPolicy::Required
    };
    with_accelerator_admission_policy(policy, body)
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
