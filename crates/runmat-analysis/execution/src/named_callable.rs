//! Shared portable program descriptor for admitted named analysis callables.
use crate::{AnalysisBridgeError, AnalysisBridgeResult};
use runmat_execution::{
    Digest, ExecutableEntrypointKind, ExecutableUnitEnvelope, OutputContract, ProgramRevision,
};
use runmat_execution_artifact::{
    BundleCallable, ExecutableForm, ProgramArtifact, ProgramBuildRecipe,
    ProgramExecutionDescriptor, ProgramTarget, PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};
use runmat_types::{CapabilityRequirement, ProgramFunctionId};
use std::collections::BTreeSet;

pub const ANALYSIS_CALLABLE_TARGET_PROFILE: &str = "portable-executable-unit-v4";
#[derive(Clone, Debug)]
pub struct NamedCallableExecutionContext {
    pub scope_id: runmat_execution::ExecutionScopeId,
    pub pool_id: runmat_execution::PoolId,
    /// Scheduler admission facts derived from the compiled callable. Callers may
    /// narrow its trust tiers but never invent runtime or capability facts.
    pub host: runmat_execution::host::ExecutionHostRequirement,
    pub resources: runmat_execution::resource::ResourceRequest,
    pub retry: runmat_execution::task::RetryPolicy,
    pub deadline_unix_millis: Option<u64>,
    pub priority: i16,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnalysisCallablePurpose {
    Generator,
    Assessment,
}
impl AnalysisCallablePurpose {
    pub const fn execution_mode(self) -> &'static str {
        match self {
            Self::Generator => "analysis-generator",
            Self::Assessment => "analysis-assessment",
        }
    }
}
pub struct NamedCallableAdmissionRef<'a> {
    pub program_revision: &'a ProgramRevision,
    pub callable: &'a BundleCallable,
    pub executable_source_digest: Digest,
    pub function: ProgramFunctionId,
    pub required_capabilities: &'a BTreeSet<CapabilityRequirement>,
}
pub fn prepare_named_callable_descriptor(
    admission: NamedCallableAdmissionRef<'_>,
    executable: ExecutableUnitEnvelope,
    purpose: AnalysisCallablePurpose,
) -> AnalysisBridgeResult<ProgramExecutionDescriptor> {
    executable.validate()?;
    let identity = &executable.manifest.identity;
    if &identity.program != admission.program_revision
        || identity.root_package != admission.callable.owner_identity
        || identity.entrypoint != admission.callable.qualified_name
        || identity.entrypoint_kind != ExecutableEntrypointKind::Function
        || identity.entrypoint_function != Some(admission.function)
        || identity.source_digest != admission.executable_source_digest
    {
        return Err(AnalysisBridgeError::Identity(
            "portable executable differs from the admitted frozen callable",
        ));
    }
    if !admission
        .required_capabilities
        .is_subset(&executable.manifest.capabilities.0)
    {
        return Err(AnalysisBridgeError::Identity(
            "portable executable omits admitted callable capabilities",
        ));
    }
    let entrypoint = identity.entrypoint.clone();
    let interop = executable.manifest.interop.clone();
    let accelerators = runmat_execution::resource::accelerator_requirements_for_capabilities(
        &executable.manifest.capabilities,
    )?;
    let executable_bytes = executable.canonical_bytes()?;
    let recipe = ProgramBuildRecipe {
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: admission.program_revision.clone(),
        entrypoint,
        outputs: OutputContract {
            requested_outputs: 1,
            owned_work: None,
        },
        execution_mode: purpose.execution_mode().into(),
        target: ProgramTarget::portable(ANALYSIS_CALLABLE_TARGET_PROFILE),
        interop,
        accelerators,
        features: BTreeSet::new(),
        compile_options: BTreeSet::new(),
        source_objects: Vec::new(),
        foreign_policy: None,
        expected_artifact_id: None,
    };
    let artifact =
        ProgramArtifact::materialize(&recipe, ExecutableForm::ExecutableUnitV4, executable_bytes)?;
    let descriptor = ProgramExecutionDescriptor {
        console: None,
        environment: runmat_execution::EnvironmentSnapshot::default(),
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
        recipe,
        artifact,
        callable: named_program_callable(&admission),
        requested_outputs: 1,
    };
    descriptor.validate_for_portable_host()?;
    Ok(descriptor)
}

/// The scheduler and worker identify an admitted named callable by its frozen
/// function and its checked qualified name.
pub(crate) fn named_program_callable(
    admission: &NamedCallableAdmissionRef<'_>,
) -> runmat_execution::ProgramCallable {
    runmat_execution::ProgramCallable::semantic(
        admission.function,
        Some(admission.callable.qualified_name.clone()),
    )
}

/// Rejects a task host requirement that is not the one derived from the compiled
/// program under the caller's trust policy.
pub(crate) fn validate_program_host(
    request: &runmat_execution_artifact::ProgramExecutionRequest,
    host: &runmat_execution::host::ExecutionHostRequirement,
) -> AnalysisBridgeResult<()> {
    let derived = request
        .artifact
        .execution_host_requirement(&request.recipe, host.permitted_trust_tiers.clone())?;
    if &derived != host {
        return Err(AnalysisBridgeError::Identity(
            "task host requirement differs from its compiled program",
        ));
    }
    Ok(())
}

/// Scheduler metadata for a direct invocation of one exact program request.
pub(crate) fn program_task_callable(
    owner_identity: &str,
    request: &runmat_execution_artifact::ProgramExecutionRequest,
) -> runmat_execution::task::Callable {
    runmat_execution::task::Callable::for_program(owner_identity, &request.callable)
}
