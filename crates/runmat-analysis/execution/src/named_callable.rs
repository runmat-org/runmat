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

/// Borrowed original source/compiled selection used by assessment preparation.
#[derive(Clone, Copy)]
pub struct NamedCallableSourceRef<'a> {
    pub program_revision: &'a ProgramRevision,
    pub callable: &'a BundleCallable,
    pub compiled_binding: &'a CompiledCallableBinding,
}

/// Registry execution name selected by the original trusted compiler/admission
/// owner. This is distinct from BundleCallable's package-qualified source label.
/// Capture it before freezing admission, never from a later descriptor candidate.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompiledCallableBinding {
    registry_name: String,
    qualified_label: String,
    function: ProgramFunctionId,
    source_digest: Digest,
}

impl CompiledCallableBinding {
    pub fn from_selected_identity(
        identity: &runmat_execution::ExecutableIdentity,
        qualified_label: &str,
    ) -> AnalysisBridgeResult<Self> {
        identity.program.validate()?;
        identity.validate()?;
        if identity.entrypoint_kind != ExecutableEntrypointKind::Function {
            return Err(AnalysisBridgeError::Identity(
                "compiled callable entrypoint kind is not function",
            ));
        }
        if qualified_label.trim().is_empty() || qualified_label.chars().any(char::is_control) {
            return Err(AnalysisBridgeError::Identity(
                "compiled callable qualified label is empty or contains control characters",
            ));
        }
        Ok(Self {
            registry_name: identity.entrypoint.clone(),
            qualified_label: qualified_label.to_owned(),
            function: identity
                .entrypoint_function
                .expect("validated function entrypoint"),
            source_digest: identity.source_digest,
        })
    }

    pub(crate) fn function(&self) -> ProgramFunctionId {
        self.function
    }
    pub(crate) fn source_digest(&self) -> Digest {
        self.source_digest
    }

    pub(crate) fn validate_admission(
        &self,
        callable: &BundleCallable,
        function: ProgramFunctionId,
        source: Digest,
    ) -> AnalysisBridgeResult<()> {
        if self.qualified_label != callable.qualified_name {
            return Err(AnalysisBridgeError::Identity(
                "admitted callable qualified label differs from its original compiled binding",
            ));
        }
        if self.function != function {
            return Err(AnalysisBridgeError::Identity(
                "compiled callable binding differs from admitted function facts",
            ));
        }
        if self.source_digest != source {
            return Err(AnalysisBridgeError::Identity(
                "compiled callable binding differs from admitted executable source digest",
            ));
        }
        Ok(())
    }

    pub(crate) fn validate_candidate(
        &self,
        identity: &runmat_execution::ExecutableIdentity,
        program_revision: &ProgramRevision,
        callable: &BundleCallable,
    ) -> AnalysisBridgeResult<()> {
        if self.qualified_label != callable.qualified_name {
            return Err(AnalysisBridgeError::Identity(
                "admitted callable qualified label differs from its original compiled binding",
            ));
        }
        for (matches, diagnostic) in [
            (
                &identity.program == program_revision,
                "portable executable program differs from the admitted frozen callable",
            ),
            (
                identity.root_package == callable.owner_identity,
                "portable executable root package differs from the admitted frozen callable",
            ),
            (
                identity.entrypoint == self.registry_name,
                "portable executable registry entrypoint differs from the admitted frozen callable",
            ),
            (
                identity.entrypoint_kind == ExecutableEntrypointKind::Function,
                "portable executable entrypoint kind differs from the admitted frozen callable",
            ),
            (
                identity.entrypoint_function == Some(self.function),
                "portable executable function differs from the admitted frozen callable",
            ),
            (
                identity.source_digest == self.source_digest,
                "portable executable source digest differs from the admitted frozen callable",
            ),
        ] {
            if !matches {
                return Err(AnalysisBridgeError::Identity(diagnostic));
            }
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn fixture(
        name: &str,
        qualified_label: &str,
        function: ProgramFunctionId,
        source_digest: Digest,
    ) -> Self {
        Self {
            registry_name: name.to_owned(),
            qualified_label: qualified_label.to_owned(),
            function,
            source_digest,
        }
    }
}
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
    pub compiled_binding: &'a CompiledCallableBinding,
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
    admission.compiled_binding.validate_admission(
        admission.callable,
        admission.function,
        admission.executable_source_digest,
    )?;
    let identity = &executable.manifest.identity;
    admission.compiled_binding.validate_candidate(
        identity,
        admission.program_revision,
        admission.callable,
    )?;
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
