//! Digest-bound project policy admission. This never reads a worker home/CWD,
//! environment, local user configuration, or collector credentials.
use runmat_config::runtime::{ClusterRuntimePolicy, LanguageCompatMode, RunMatRuntimeConfig};
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::{ExecutionBundle, LogicalObject};

/// Narrow immutable execution projection; it never initializes a process provider.
pub fn acceleration_policy(
    config: &RunMatRuntimeConfig,
) -> runmat_runtime::context::RuntimeAccelerationPolicy {
    use runmat_config::runtime::AccelerateProviderPreference as Source;
    use runmat_runtime::context::{RuntimeAccelerationPolicy, RuntimeProviderPreference as Target};
    RuntimeAccelerationPolicy {
        enabled: config.accelerate.enabled,
        provider: match config.accelerate.provider {
            Source::Auto => Target::Auto,
            Source::Wgpu => Target::Wgpu,
            Source::InProcess => Target::InProcess,
        },
        automatic_offload: config.accelerate.auto_offload.enabled,
        allow_host_fallback: config.accelerate.allow_inprocess_fallback,
        calibrate: config.accelerate.auto_offload.calibrate,
    }
}

pub fn decode_project_policy_object(
    object: &LogicalObject,
    revision: &ProgramRevision,
) -> Result<ClusterRuntimePolicy, String> {
    let policy: ClusterRuntimePolicy =
        runmat_execution_artifact::decode_typed_project_runtime_policy(object)
            .map_err(|error| error.to_string())?;
    policy.validate().map_err(|error| error.to_string())?;
    validate_program_environment(&policy, revision)?;
    Ok(policy)
}

/// Bind project-only policy to unassigned immutable selected code. This never
/// changes that code's language or foreign authority; mismatches must recompile.
pub fn prepare_graph_for_cluster(
    mut graph: runmat_execution_artifact::GraphExecutionRequest,
    policy: &ClusterRuntimePolicy,
) -> Result<runmat_execution_artifact::GraphExecutionRequest, String> {
    graph.validate().map_err(|error| error.to_string())?;
    if graph.assignment.is_some() || graph.job_id.is_some() {
        return Err("cannot change cluster settings after graph assignment".into());
    }
    if graph.schema_version < runmat_execution::value::GRAPH_REQUEST_SCHEMA_V12
        && matches!(
            graph.roots,
            runmat_execution_artifact::GraphInvocationRoots::Callable { .. }
        )
    {
        return Err("legacy flattened callable Graphs must be rematerialized with supplied-only callable arguments before cluster preparation".into());
    }
    graph.project_runtime_policy = Some(capture_project_policy(policy)?);
    graph.schema_version = graph
        .schema_version
        .max(runmat_execution::value::GRAPH_REQUEST_SCHEMA_V13);
    validate_graph_policy(&graph)?;
    graph.validate().map_err(|error| error.to_string())?;
    Ok(graph)
}

pub fn validate_graph_policy(
    graph: &runmat_execution_artifact::GraphExecutionRequest,
) -> Result<ClusterRuntimePolicy, String> {
    let object = graph.project_runtime_policy.as_ref().ok_or("cluster Graph lacks digest-bound project-only runtime policy; prepare the Graph for its cluster target before submitting")?;
    let policy = decode_project_policy_object(object, &graph.recipe.program_revision)?;
    let expected_mode = graph
        .recipe
        .program_revision
        .environment()
        .compatibility_mode;
    if graph
        .selected_unit_host_requirements()
        .map_err(|error| error.to_string())?
        .iter()
        .any(|unit| unit.environment.compatibility_mode != expected_mode)
    {
        return Err("selected Graph contains code compiled under a different language policy; recompile that code for the cluster target".into());
    }
    let foreign = runmat_execution_artifact::capture_typed_foreign_policy(&policy.foreign)
        .map_err(|error| error.to_string())?;
    let foreign_ref = runmat_execution_artifact::foreign_policy_ref(&foreign.descriptor)
        .map_err(|error| error.to_string())?;
    if graph
        .selected_foreign_policies
        .iter()
        .flatten()
        .any(|reference| reference != &foreign_ref)
    {
        return Err("selected Graph foreign authority differs from project-only policy; recompile for the cluster target".into());
    }
    Ok(policy)
}

pub fn capture_project_policy(policy: &ClusterRuntimePolicy) -> Result<LogicalObject, String> {
    policy.validate().map_err(|error| error.to_string())?;
    runmat_execution_artifact::capture_typed_project_runtime_policy(policy)
        .map_err(|error| error.to_string())
}

pub fn decode_project_policy(
    bundle: &ExecutionBundle,
) -> Result<Option<ClusterRuntimePolicy>, String> {
    let Some(object) = bundle
        .project_runtime_policy_object()
        .map_err(|error| error.to_string())?
    else {
        return Ok(None);
    };
    let policy = decode_project_policy_object(object, &bundle.manifest.program_revision)?;
    let foreign = runmat_execution_artifact::capture_typed_foreign_policy(&policy.foreign)
        .map_err(|error| error.to_string())?;
    if bundle
        .manifest
        .recipes
        .iter()
        .filter_map(|recipe| recipe.foreign_policy.as_ref())
        .any(|descriptor| descriptor != &foreign.descriptor)
    {
        return Err("compiled foreign policy differs from project-only cluster policy; recompile for the cluster target before submitting".into());
    }
    Ok(Some(policy))
}

pub fn validate_program_environment(
    policy: &ClusterRuntimePolicy,
    revision: &ProgramRevision,
) -> Result<(), String> {
    let expected = match policy.language.compat {
        LanguageCompatMode::RunMat => runmat_execution::LanguageCompatibilityMode::RunMat,
        LanguageCompatMode::Matlab => runmat_execution::LanguageCompatibilityMode::Matlab,
        LanguageCompatMode::Strict => runmat_execution::LanguageCompatibilityMode::Strict,
    };
    if revision.environment().compatibility_mode != expected {
        return Err("compiled program language differs from project-only cluster policy; recompile for the cluster target before submitting".into());
    }
    Ok(())
}

/// Decode only the bounded execution settings; no logging/telemetry process
/// configuration is accepted from a run's project or submitter.
pub fn runtime_config(bundle: &ExecutionBundle) -> Result<Option<RunMatRuntimeConfig>, String> {
    decode_project_policy(bundle)?
        .map(|policy| policy.runtime_config().map_err(|error| error.to_string()))
        .transpose()
}
