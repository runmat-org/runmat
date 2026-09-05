use std::collections::BTreeSet;

use runmat_analysis_fea::execution::AnalysisResultIdentity;
use runmat_execution::value::{ValuePayload, ValueRef, ValueRefKind};
use runmat_execution_artifact::ProgramExecutionResponse;
use serde::{Deserialize, Serialize};

use crate::{
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisHostWorkload, CompletedAnalysisWorkload,
};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisHostResponse {
    pub schema_version: u16,
    pub result_identity: AnalysisResultIdentity,
    pub output_payload_digests: Vec<runmat_execution::Digest>,
    pub roots: Vec<ValueRef>,
    pub result_objects: Vec<ValueRef>,
}

impl AnalysisHostResponse {
    pub fn completed(
        host: &AnalysisHostWorkload,
        completed: &CompletedAnalysisWorkload,
    ) -> AnalysisBridgeResult<Self> {
        let response = Self {
            schema_version: 1,
            result_identity: completed.identity,
            output_payload_digests: completed
                .outputs
                .iter()
                .map(|output| output.manifest.payload_digest)
                .collect(),
            roots: completed.roots.clone(),
            result_objects: completed.result_objects.clone(),
        };
        response.validate_against(host)?;
        Ok(response)
    }

    pub fn validate_against(&self, host: &AnalysisHostWorkload) -> AnalysisBridgeResult<()> {
        host.validate()?;
        if self.schema_version != 1
            || self.roots.len() != host.workload.outputs.len()
            || self.output_payload_digests.len() != self.roots.len()
        {
            return Err(AnalysisBridgeError::Invalid(
                "analysis host response shape differs from its workload".into(),
            ));
        }
        let expected = AnalysisResultIdentity::new(
            host.workload.identity,
            self.output_payload_digests.clone(),
        )?;
        if self.result_identity != expected {
            return Err(AnalysisBridgeError::Identity(
                "analysis result identity differs from its workload and payloads",
            ));
        }
        let mut ids = BTreeSet::new();
        let mut digests = BTreeSet::new();
        for reference in &self.result_objects {
            host.artifact_access.validate_reference(
                reference,
                ValueRefKind::ResultObject,
                &reference.media_type,
                &reference.value_schema,
            )?;
            if !ids.insert(reference.id) || !digests.insert(reference.logical_digest) {
                return Err(AnalysisBridgeError::Invalid(
                    "analysis result inventory contains duplicate objects".into(),
                ));
            }
        }
        for root in &self.roots {
            host.artifact_access.validate_reference(
                root,
                ValueRefKind::ResultObject,
                runmat_analysis_fea::execution::ANALYSIS_MANIFEST_MEDIA_TYPE,
                runmat_analysis_fea::execution::ANALYSIS_MANIFEST_VALUE_SCHEMA,
            )?;
            if !self.result_objects.contains(root) {
                return Err(AnalysisBridgeError::Invalid(
                    "analysis result inventory omits an output root".into(),
                ));
            }
        }
        Ok(())
    }

    pub fn program_response(&self) -> ProgramExecutionResponse {
        ProgramExecutionResponse::ExternalizedSuccess {
            outputs: self
                .roots
                .iter()
                .cloned()
                .map(|root| ValuePayload::Object(Box::new(root)))
                .collect(),
            result_objects: self.result_objects.clone(),
        }
    }
}
