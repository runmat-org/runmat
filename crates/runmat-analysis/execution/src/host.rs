use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::value::{ValuePayload, ValueRef};
use runmat_execution::{OutputContract, ProgramRevision};
use runmat_execution_artifact::{
    ArtifactAccess, ExecutableForm, ProgramArtifact, ProgramBuildRecipe, ProgramExecutionRequest,
    ProgramTarget, PROGRAM_BUILD_RECIPE_SCHEMA_VERSION, PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
};
use serde::{Deserialize, Serialize};

use crate::{AnalysisBridgeError, AnalysisBridgeResult, AnalysisWorkloadRequest};

pub const ANALYSIS_HOST_ABI: &str = "host-v1";
pub const ANALYSIS_HOST_EXECUTION_MODE: &str = "analysis";
pub const ANALYSIS_HOST_TARGET_PROFILE: &str = "portable-analysis-host-v1";
const HOST_PREFIX: &[u8] = b"runmat-analysis-host-workload/v1\0";
const HOST_DOMAIN: &str = "host-workload";
const HOST_LIMITS: CanonicalLimits = CanonicalLimits::new(64 * 1024 * 1024, 1_000_000, 65_536, 64);

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisHostWorkload {
    pub schema_version: u16,
    pub workload: AnalysisWorkloadRequest,
    pub artifact_access: ArtifactAccess,
}

impl AnalysisHostWorkload {
    pub fn new(
        workload: AnalysisWorkloadRequest,
        artifact_access: ArtifactAccess,
    ) -> AnalysisBridgeResult<Self> {
        let host = Self {
            schema_version: 1,
            workload,
            artifact_access,
        };
        host.validate()?;
        Ok(host)
    }

    pub fn validate(&self) -> AnalysisBridgeResult<()> {
        if self.schema_version != 1 {
            return Err(AnalysisBridgeError::Invalid(
                "unsupported analysis host workload schema".into(),
            ));
        }
        self.workload.validate()?;
        self.artifact_access.validate()?;
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisBridgeResult<Vec<u8>> {
        self.validate()?;
        Ok(encode_contract(
            HOST_PREFIX,
            HOST_DOMAIN,
            self,
            HOST_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisBridgeResult<Self> {
        let host: Self = decode_contract(HOST_PREFIX, HOST_DOMAIN, bytes, HOST_LIMITS)?;
        host.validate()?;
        Ok(host)
    }

    pub fn program_request(
        &self,
        program_revision: ProgramRevision,
        input_roots: &[ValueRef],
    ) -> AnalysisBridgeResult<ProgramExecutionRequest> {
        self.validate_inputs(input_roots)?;
        let requested_outputs = u16::try_from(self.workload.outputs.len()).map_err(|_| {
            AnalysisBridgeError::Invalid("analysis output count is not representable".into())
        })?;
        let executable_bytes = self.canonical_bytes()?;
        let recipe = ProgramBuildRecipe {
            schema_version: PROGRAM_BUILD_RECIPE_SCHEMA_VERSION,
            program_revision,
            entrypoint: "analysis_workload".into(),
            outputs: OutputContract { requested_outputs },
            execution_mode: ANALYSIS_HOST_EXECUTION_MODE.into(),
            target: ProgramTarget::portable(ANALYSIS_HOST_TARGET_PROFILE),
            features: Default::default(),
            compile_options: std::collections::BTreeSet::from([format!(
                "analysis-workload:{}",
                self.workload.identity.digest()
            )]),
            source_objects: Vec::new(),
            expected_artifact_id: None,
        };
        let artifact = ProgramArtifact::materialize(
            &recipe,
            ExecutableForm::AnalysisWorkload,
            executable_bytes,
        )?;
        let request = ProgramExecutionRequest {
            schema_version: PROGRAM_EXECUTION_REQUEST_SCHEMA_V1,
            recipe,
            artifact,
            function: 0,
            arguments: input_roots
                .iter()
                .cloned()
                .map(|root| ValuePayload::Object(Box::new(root)))
                .collect(),
            requested_outputs,
        };
        request.validate()?;
        Ok(request)
    }

    pub fn from_program_request(request: &ProgramExecutionRequest) -> AnalysisBridgeResult<Self> {
        request.validate_for_portable_host()?;
        if request.artifact.form != ExecutableForm::AnalysisWorkload
            || request.recipe.execution_mode != ANALYSIS_HOST_EXECUTION_MODE
            || request.recipe.target != ProgramTarget::portable(ANALYSIS_HOST_TARGET_PROFILE)
        {
            return Err(AnalysisBridgeError::Invalid(
                "program request is not an analysis host workload".into(),
            ));
        }
        let host = Self::from_canonical_bytes(&request.artifact.executable_bytes)?;
        let roots = request
            .arguments
            .iter()
            .map(|value| match value {
                ValuePayload::Object(reference) => Ok((**reference).clone()),
                ValuePayload::Inline(_) => Err(AnalysisBridgeError::Invalid(
                    "analysis host inputs must be externalized manifest roots".into(),
                )),
            })
            .collect::<AnalysisBridgeResult<Vec<_>>>()?;
        host.validate_inputs(&roots)?;
        Ok(host)
    }

    fn validate_inputs(&self, roots: &[ValueRef]) -> AnalysisBridgeResult<()> {
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
