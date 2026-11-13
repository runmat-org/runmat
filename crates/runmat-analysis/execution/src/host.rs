use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::value::{ValuePayload, ValueRef, ValueRefKind};
use runmat_execution::{OutputContract, ProgramRevision};
use runmat_execution_artifact::cache::{CacheExport, CacheImport};
use runmat_execution_artifact::{
    ArtifactAccess, ExecutableForm, LogicalObject, ObjectNamespace, ProgramArtifact,
    ProgramBuildRecipe, ProgramExecutionRequest, ProgramTarget,
    PROGRAM_BUILD_RECIPE_SCHEMA_VERSION, PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
};
use serde::{Deserialize, Serialize};

use crate::{AnalysisBridgeError, AnalysisBridgeResult, AnalysisWorkloadRequest};

pub const ANALYSIS_HOST_ABI: &str = "host-v2";
pub const ANALYSIS_HOST_EXECUTION_MODE: &str = "analysis";
pub const ANALYSIS_HOST_TARGET_PROFILE: &str = "portable-analysis-host-v2";
const HOST_PROGRAM_BYTES: &[u8] = b"runmat-analysis-host\0";
pub(crate) const ANALYSIS_HOST_ENTRYPOINT: &str = "analysis_workload";
const HOST_PREFIX: &[u8] = b"runmat-analysis-host-invocation/v2\0";
const HOST_DOMAIN: &str = "host-invocation";
const HOST_LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);
pub const ANALYSIS_HOST_INVOCATION_MEDIA_TYPE: &str =
    "application/vnd.runmat.analysis-host-invocation+cbor";
pub const ANALYSIS_HOST_INVOCATION_VALUE_SCHEMA: &str = "runmat.analysis.host-invocation";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnalysisHostWorkload {
    pub schema_version: u16,
    pub workload: AnalysisWorkloadRequest,
    pub artifact_access: ArtifactAccess,
    pub output_policy: crate::AnalysisOutputPolicy,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AnalysisHostInvocation {
    schema_version: u16,
    workload: AnalysisWorkloadRequest,
    output_policy: crate::AnalysisOutputPolicy,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnalysisHostProgramArtifact {
    pub recipe: ProgramBuildRecipe,
    pub artifact: ProgramArtifact,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedAnalysisHostInvocation {
    request: ProgramExecutionRequest,
    descriptor: LogicalObject,
}

/// The single program callable of the analysis host workload.
pub(crate) fn analysis_host_callable() -> runmat_execution::ProgramCallable {
    runmat_execution::ProgramCallable::semantic(
        runmat_execution::ProgramFunctionId(0),
        Some(ANALYSIS_HOST_ENTRYPOINT.into()),
    )
}

pub fn prepare_analysis_host_program(
    program_revision: ProgramRevision,
) -> AnalysisBridgeResult<AnalysisHostProgramArtifact> {
    let recipe = ProgramBuildRecipe {
        schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
        program_revision,
        entrypoint: ANALYSIS_HOST_ENTRYPOINT.into(),
        outputs: OutputContract {
            requested_outputs: 1,
            owned_work: None,
        },
        execution_mode: ANALYSIS_HOST_EXECUTION_MODE.into(),
        target: ProgramTarget::portable(ANALYSIS_HOST_TARGET_PROFILE),
        interop: runmat_types::InteropManifest::empty(),
        accelerators: Vec::new(),
        features: Default::default(),
        compile_options: Default::default(),
        source_objects: Vec::new(),
        foreign_policy: None,
        expected_artifact_id: None,
    };
    let artifact = ProgramArtifact::materialize(
        &recipe,
        ExecutableForm::AnalysisWorkload,
        HOST_PROGRAM_BYTES.to_vec(),
    )?;
    Ok(AnalysisHostProgramArtifact { recipe, artifact })
}

impl PreparedAnalysisHostInvocation {
    pub const fn request(&self) -> &ProgramExecutionRequest {
        &self.request
    }

    pub const fn descriptor(&self) -> &LogicalObject {
        &self.descriptor
    }

    pub fn write_to(&self, store: &mut impl CacheExport) -> AnalysisBridgeResult<()> {
        store.write_verified(&self.descriptor)?;
        Ok(())
    }

    pub fn into_parts(self) -> (ProgramExecutionRequest, LogicalObject) {
        (self.request, self.descriptor)
    }
}

impl AnalysisHostWorkload {
    pub fn new(
        workload: AnalysisWorkloadRequest,
        artifact_access: ArtifactAccess,
        output_policy: crate::AnalysisOutputPolicy,
    ) -> AnalysisBridgeResult<Self> {
        let host = Self {
            schema_version: 2,
            workload,
            artifact_access,
            output_policy,
        };
        host.validate()?;
        Ok(host)
    }

    pub fn validate(&self) -> AnalysisBridgeResult<()> {
        if self.schema_version != 2 {
            return Err(AnalysisBridgeError::Invalid(
                "unsupported analysis host workload schema".into(),
            ));
        }
        self.workload.validate()?;
        self.artifact_access.validate()?;
        self.output_policy.validate()?;
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisBridgeResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            HOST_PREFIX,
            HOST_DOMAIN,
            &AnalysisHostInvocation {
                schema_version: self.schema_version,
                workload: self.workload.clone(),
                output_policy: self.output_policy,
            },
            HOST_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(
        bytes: &[u8],
        artifact_access: ArtifactAccess,
    ) -> AnalysisBridgeResult<Self> {
        let program: AnalysisHostInvocation =
            decode_contract(HOST_PREFIX, HOST_DOMAIN, bytes, HOST_LIMITS)?;
        let host = Self {
            schema_version: program.schema_version,
            workload: program.workload,
            artifact_access,
            output_policy: program.output_policy,
        };
        host.validate()?;
        Ok(host)
    }

    pub(crate) fn invocation_descriptor(&self) -> AnalysisBridgeResult<(LogicalObject, ValueRef)> {
        let descriptor = LogicalObject::new(
            ObjectNamespace::InputValue,
            format!(
                "analysis/host-invocations/{}",
                self.workload.identity.digest()
            ),
            ANALYSIS_HOST_INVOCATION_MEDIA_TYPE,
            self.canonical_bytes()?,
        )?;
        let descriptor_root = self.artifact_access.reference(
            &descriptor,
            ValueRefKind::DriverObject,
            ANALYSIS_HOST_INVOCATION_VALUE_SCHEMA,
        )?;
        Ok((descriptor, descriptor_root))
    }

    pub fn prepare_program_request(
        &self,
        program_revision: ProgramRevision,
        input_roots: &[ValueRef],
    ) -> AnalysisBridgeResult<PreparedAnalysisHostInvocation> {
        self.validate_inputs(input_roots)?;
        let (descriptor, descriptor_root) = self.invocation_descriptor()?;
        let AnalysisHostProgramArtifact { recipe, artifact } =
            prepare_analysis_host_program(program_revision)?;
        let mut arguments = Vec::with_capacity(input_roots.len() + 1);
        arguments.push(ValuePayload::Object(Box::new(descriptor_root)));
        arguments.extend(
            input_roots
                .iter()
                .cloned()
                .map(|root| ValuePayload::Object(Box::new(root))),
        );
        let request = ProgramExecutionRequest {
            console: None,
            environment: runmat_execution::EnvironmentSnapshot::default(),
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_VERSION,
            recipe,
            artifact,
            callable: analysis_host_callable(),
            context: runmat_execution::ProgramInvocationContext::Direct,
            assignment: None,
            job_id: None,
            arguments,
            requested_outputs: 1,
        };
        request.validate()?;
        Ok(PreparedAnalysisHostInvocation {
            request,
            descriptor,
        })
    }

    pub fn from_program_request(
        request: &ProgramExecutionRequest,
        store: &impl CacheImport,
    ) -> AnalysisBridgeResult<Self> {
        Self::from_program_request_admitted(request, store, None)
    }

    pub(crate) fn from_program_request_admitted(
        request: &ProgramExecutionRequest,
        store: &impl CacheImport,
        admission: Option<&crate::object::AnalysisInputAdmission>,
    ) -> AnalysisBridgeResult<Self> {
        request.validate_for_portable_host()?;
        if request.artifact.form != ExecutableForm::AnalysisWorkload
            || request.recipe.execution_mode != ANALYSIS_HOST_EXECUTION_MODE
            || request.recipe.target != ProgramTarget::portable(ANALYSIS_HOST_TARGET_PROFILE)
            || request.artifact.executable_bytes != HOST_PROGRAM_BYTES
            || !request.recipe.compile_options.is_empty()
        {
            return Err(AnalysisBridgeError::Invalid(
                "program request is not an analysis host workload".into(),
            ));
        }
        let (descriptor, inputs) = request.arguments.split_first().ok_or_else(|| {
            AnalysisBridgeError::Invalid(
                "analysis host request is missing its invocation descriptor".into(),
            )
        })?;
        let ValuePayload::Object(descriptor_root) = descriptor else {
            return Err(AnalysisBridgeError::Invalid(
                "analysis host invocation must remain externalized".into(),
            ));
        };
        if descriptor_root.kind != ValueRefKind::DriverObject
            || descriptor_root.media_type != ANALYSIS_HOST_INVOCATION_MEDIA_TYPE
            || descriptor_root.value_schema != ANALYSIS_HOST_INVOCATION_VALUE_SCHEMA
            || descriptor_root.encoded_length == 0
            || descriptor_root.encoded_length > HOST_LIMITS.maximum_encoded_bytes as u64
        {
            return Err(AnalysisBridgeError::Invalid(
                "analysis host invocation reference is malformed".into(),
            ));
        }
        let bytes = crate::object::read_input(
            store,
            descriptor_root.logical_digest,
            descriptor_root.encoded_length,
            admission,
        )?
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid("analysis host invocation is unavailable".into())
        })?;
        if bytes.len() as u64 != descriptor_root.encoded_length
            || runmat_execution::Digest::sha256(&bytes) != descriptor_root.logical_digest
        {
            return Err(AnalysisBridgeError::Identity(
                "analysis host invocation bytes differ from their reference",
            ));
        }
        let artifact_access = ArtifactAccess {
            authorization_scope: descriptor_root.authorization_scope.clone(),
            encryption_context: descriptor_root.encryption_context,
        };
        if let Some(admission) = admission {
            admission.admit_canonical_decode(&bytes)?;
        }
        let host = Self::from_canonical_bytes(&bytes, artifact_access)?;
        host.artifact_access.validate_reference(
            descriptor_root,
            ValueRefKind::DriverObject,
            ANALYSIS_HOST_INVOCATION_MEDIA_TYPE,
            ANALYSIS_HOST_INVOCATION_VALUE_SCHEMA,
        )?;
        let roots = inputs
            .iter()
            .map(|value| match value {
                ValuePayload::Object(reference) => Ok((**reference).clone()),
                _ => Err(AnalysisBridgeError::Invalid(
                    "analysis host inputs must be externalized manifest roots".into(),
                )),
            })
            .collect::<AnalysisBridgeResult<Vec<_>>>()?;
        host.validate_inputs(&roots)?;
        Ok(host)
    }

    pub fn semantic_input_roots(
        request: &ProgramExecutionRequest,
    ) -> AnalysisBridgeResult<Vec<ValueRef>> {
        request
            .arguments
            .get(1..)
            .ok_or_else(|| {
                AnalysisBridgeError::Invalid(
                    "analysis host request is missing its invocation descriptor".into(),
                )
            })?
            .iter()
            .map(|value| match value {
                ValuePayload::Object(root) => Ok((**root).clone()),
                _ => Err(AnalysisBridgeError::Invalid(
                    "analysis host inputs must be externalized root manifests".into(),
                )),
            })
            .collect()
    }

    pub(crate) fn validate_inputs(&self, roots: &[ValueRef]) -> AnalysisBridgeResult<()> {
        self.validate()?;
        if roots.len() != self.workload.inputs.len() {
            return Err(AnalysisBridgeError::Invalid(
                "analysis input root count differs from workload inputs".into(),
            ));
        }
        for (root, input) in roots.iter().zip(&self.workload.inputs) {
            self.artifact_access.validate_reference(
                root,
                input.reference_kind,
                runmat_analysis_fea::execution::ANALYSIS_MANIFEST_MEDIA_TYPE,
                runmat_analysis_fea::execution::ANALYSIS_MANIFEST_VALUE_SCHEMA,
            )?;
            if root.logical_digest != input.manifest_digest
                || root.encoded_length != input.manifest_encoded_length
            {
                return Err(AnalysisBridgeError::Identity(
                    "analysis input root differs from its workload descriptor",
                ));
            }
        }
        Ok(())
    }
}
