//! Shared portable program descriptor for admitted named analysis callables.
use crate::{AnalysisBridgeError, AnalysisBridgeResult};
use runmat_execution::{
    Digest, ExecutableEntrypointKind, ExecutableUnitEnvelope, OutputContract, ProgramRevision,
};
use runmat_execution_artifact::{
    BundleCallable, ExecutableForm, ProgramArtifact, ProgramBuildRecipe,
    ProgramExecutionDescriptor, ProgramTarget, PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
    PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
use runmat_types::{CapabilityRequirement, ProgramFunctionId};
use std::collections::BTreeSet;

pub const ANALYSIS_CALLABLE_TARGET_PROFILE: &str = "portable-executable-unit-v4";
#[derive(Clone, Debug)]
pub struct NamedCallableExecutionContext {
    pub scope_id: runmat_execution::ExecutionScopeId,
    pub pool_id: runmat_execution::PoolId,
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
        || identity.entrypoint_kind != ExecutableEntrypointKind::Function
        || identity.entrypoint_function != admission.function
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
    let executable_bytes = executable.canonical_bytes()?;
    let entrypoint = usize::try_from(admission.function.0).map_err(|_| {
        AnalysisBridgeError::Invalid("callable function identity exceeds this host".into())
    })?;
    let recipe = ProgramBuildRecipe {
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision: admission.program_revision.clone(),
        entrypoint: entrypoint.to_string(),
        outputs: OutputContract {
            requested_outputs: 1,
        },
        execution_mode: purpose.execution_mode().into(),
        target: ProgramTarget::portable(ANALYSIS_CALLABLE_TARGET_PROFILE),
        features: BTreeSet::new(),
        compile_options: BTreeSet::new(),
        source_objects: Vec::new(),
        expected_artifact_id: None,
    };
    let artifact =
        ProgramArtifact::materialize(&recipe, ExecutableForm::ExecutableUnitV4, executable_bytes)?;
    let descriptor = ProgramExecutionDescriptor {
        schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
        recipe,
        artifact,
        function: entrypoint,
        requested_outputs: 1,
    };
    descriptor.validate_for_portable_host()?;
    Ok(descriptor)
}
