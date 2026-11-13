use std::collections::BTreeSet;

use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisResultIdentity, AnalysisWorkloadIdentity,
    ANALYSIS_MANIFEST_MEDIA_TYPE, ANALYSIS_MANIFEST_VALUE_SCHEMA,
};
use runmat_analysis_fea::planning::resource::PreparedOutputSetResourceFacts;
use runmat_canonical_codec::{decode_contract, encode_contract, CanonicalLimits};
use runmat_execution::value::{ValuePayload, ValueRef, ValueRefKind};
use runmat_execution::Digest;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::{LogicalObject, ObjectNamespace, ProgramExecutionResponse};
use serde::{Deserialize, Serialize};

use crate::{
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisHostWorkload, CompletedAnalysisWorkload,
    PreparedAnalysisObjects,
};

pub const ANALYSIS_OUTPUT_SET_MEDIA_TYPE: &str = "application/vnd.runmat.analysis-output-set+cbor";
pub const ANALYSIS_OUTPUT_SET_VALUE_SCHEMA: &str = "runmat.analysis.output-set.v1";
const OUTPUT_SET_PREFIX: &[u8] = b"runmat-analysis-output-set/v1\0";
pub(crate) use runmat_analysis_fea::execution::ANALYSIS_OUTPUT_SET_MAX_ENCODED_BYTES;
const OUTPUT_SET_LIMITS: CanonicalLimits = CanonicalLimits::new(
    ANALYSIS_OUTPUT_SET_MAX_ENCODED_BYTES as usize,
    1_000_000,
    65_536,
    32,
);

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisOutputRoot {
    pub kind: AnalysisNumericalObjectKind,
    pub logical_identity: Digest,
    pub payload_digest: Digest,
    pub root: ValueRef,
}

/// Transport manifest returned as the analysis host's single stable ABI output.
///
/// Numerical output count and kinds remain in the workload and this manifest. They do not alter
/// the bundle-authorized host executable, so dynamically planned stages reuse one artifact.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisOutputSet {
    pub schema_version: u16,
    pub workload: AnalysisWorkloadIdentity,
    pub result: AnalysisResultIdentity,
    pub outputs: Vec<AnalysisOutputRoot>,
}

impl AnalysisOutputSet {
    pub fn new(
        host: &AnalysisHostWorkload,
        result: AnalysisResultIdentity,
        outputs: &[PreparedAnalysisObjects],
        roots: &[ValueRef],
    ) -> AnalysisBridgeResult<Self> {
        if outputs.len() != roots.len() {
            return Err(AnalysisBridgeError::Invalid(
                "analysis numerical outputs differ from their roots".into(),
            ));
        }
        let output_set = Self {
            schema_version: 1,
            workload: host.workload.identity,
            result,
            outputs: outputs
                .iter()
                .zip(roots)
                .map(|(output, root)| AnalysisOutputRoot {
                    kind: output.manifest.kind,
                    logical_identity: output.manifest.logical_identity,
                    payload_digest: output.manifest.payload_digest,
                    root: root.clone(),
                })
                .collect(),
        };
        output_set.validate_against(host)?;
        Ok(output_set)
    }

    pub fn validate_against(&self, host: &AnalysisHostWorkload) -> AnalysisBridgeResult<()> {
        host.validate()?;
        if self.schema_version != 1
            || self.workload != host.workload.identity
            || self.outputs.len() != host.workload.outputs.len()
            || self
                .outputs
                .iter()
                .zip(&host.workload.outputs)
                .any(|(output, expected)| output.kind != *expected)
        {
            return Err(AnalysisBridgeError::Identity(
                "analysis output set differs from its workload contract",
            ));
        }
        let expected_result = AnalysisResultIdentity::new(
            self.workload,
            self.outputs
                .iter()
                .map(|output| output.payload_digest)
                .collect(),
        )?;
        if self.result != expected_result {
            return Err(AnalysisBridgeError::Identity(
                "analysis output set result identity does not converge",
            ));
        }
        let mut root_ids = BTreeSet::new();
        let mut root_digests = BTreeSet::new();
        for output in &self.outputs {
            host.artifact_access.validate_reference(
                &output.root,
                ValueRefKind::ResultObject,
                ANALYSIS_MANIFEST_MEDIA_TYPE,
                ANALYSIS_MANIFEST_VALUE_SCHEMA,
            )?;
            if !root_ids.insert(output.root.id) || !root_digests.insert(output.root.logical_digest)
            {
                return Err(AnalysisBridgeError::Invalid(
                    "analysis output set contains duplicate numerical roots".into(),
                ));
            }
        }
        Ok(())
    }

    pub fn canonical_bytes(&self) -> AnalysisBridgeResult<Vec<u8>> {
        Ok(encode_contract(
            OUTPUT_SET_PREFIX,
            "analysis-output-set",
            self,
            OUTPUT_SET_LIMITS,
        )?)
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> AnalysisBridgeResult<Self> {
        Ok(decode_contract(
            OUTPUT_SET_PREFIX,
            "analysis-output-set",
            bytes,
            OUTPUT_SET_LIMITS,
        )?)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImportedAnalysisOutputSet {
    pub manifest: AnalysisOutputSet,
    pub root: ValueRef,
    pub result_objects: Vec<ValueRef>,
}

pub fn import_analysis_output_set(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    host: &AnalysisHostWorkload,
) -> AnalysisBridgeResult<ImportedAnalysisOutputSet> {
    let (outputs, result_objects) = match response {
        ProgramExecutionResponse::ExternalizedSuccess {
            outputs,
            result_objects,
        } => (outputs, result_objects),
        ProgramExecutionResponse::Failure { message } => {
            return Err(AnalysisBridgeError::HostExecution(message))
        }
        ProgramExecutionResponse::RuntimeFailure { failure } => {
            return Err(AnalysisBridgeError::HostExecution(failure.message))
        }
        ProgramExecutionResponse::Success { .. }
        | ProgramExecutionResponse::GraphSuccess { .. }
        | ProgramExecutionResponse::SpmdSuccess { .. } => {
            return Err(AnalysisBridgeError::Invalid(
                "analysis host did not return an externalized success".into(),
            ))
        }
    };
    let [ValuePayload::Object(root)] = outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "analysis host must return exactly one output-set root".into(),
        ));
    };
    host.artifact_access.validate_reference(
        root,
        ValueRefKind::ResultObject,
        ANALYSIS_OUTPUT_SET_MEDIA_TYPE,
        ANALYSIS_OUTPUT_SET_VALUE_SCHEMA,
    )?;
    if !result_objects.contains(root) {
        return Err(AnalysisBridgeError::Invalid(
            "analysis result inventory omits its output-set root".into(),
        ));
    }
    if root.encoded_length == 0 || root.encoded_length > ANALYSIS_OUTPUT_SET_MAX_ENCODED_BYTES {
        return Err(AnalysisBridgeError::Invalid(
            "analysis output-set root exceeds its schema bound".into(),
        ));
    }
    let bytes = source
        .read_verified(root.logical_digest)?
        .ok_or_else(|| AnalysisBridgeError::Invalid("analysis output set is absent".into()))?;
    if bytes.len() as u64 != root.encoded_length || Digest::sha256(&bytes) != root.logical_digest {
        return Err(AnalysisBridgeError::Identity(
            "analysis output-set bytes differ from their reference",
        ));
    }
    let manifest = AnalysisOutputSet::from_canonical_bytes(&bytes)?;
    manifest.validate_against(host)?;
    Ok(ImportedAnalysisOutputSet {
        manifest,
        root: (**root).clone(),
        result_objects,
    })
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisHostResponse {
    pub schema_version: u16,
    pub result_identity: AnalysisResultIdentity,
    pub output_set_root: ValueRef,
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
            output_set_root: completed.output_set_root.clone(),
            result_objects: completed.result_objects.clone(),
        };
        response.validate_against(host, completed)?;
        Ok(response)
    }

    pub fn validate_against(
        &self,
        host: &AnalysisHostWorkload,
        completed: &CompletedAnalysisWorkload,
    ) -> AnalysisBridgeResult<()> {
        completed.output_set.validate_against(host)?;
        if self.schema_version != 1
            || self.result_identity != completed.identity
            || self.output_set_root != completed.output_set_root
            || self.result_objects != completed.result_objects
            || !self.result_objects.contains(&self.output_set_root)
        {
            return Err(AnalysisBridgeError::Identity(
                "analysis host response differs from its completed output set",
            ));
        }
        Ok(())
    }

    pub fn program_response(&self) -> ProgramExecutionResponse {
        ProgramExecutionResponse::ExternalizedSuccess {
            outputs: vec![ValuePayload::Object(Box::new(self.output_set_root.clone()))],
            result_objects: self.result_objects.clone(),
        }
    }
}

pub(crate) struct PreparedAnalysisOutputSet {
    pub output_set: AnalysisOutputSet,
    pub object: LogicalObject,
    pub root: ValueRef,
}

impl PreparedAnalysisOutputSet {
    pub fn validate_resource_facts(
        &self,
        admitted: &PreparedOutputSetResourceFacts,
    ) -> AnalysisBridgeResult<()> {
        if self.object.bytes.len() as u64 > admitted.encoded_bytes_upper_bound.0 {
            return Err(AnalysisBridgeError::Invalid(
                "analysis output set exceeds its resource-plan envelope".into(),
            ));
        }
        Ok(())
    }
}

pub(crate) fn prepare_analysis_output_set(
    host: &AnalysisHostWorkload,
    result: AnalysisResultIdentity,
    outputs: &[PreparedAnalysisObjects],
    roots: &[ValueRef],
) -> AnalysisBridgeResult<PreparedAnalysisOutputSet> {
    let output_set = AnalysisOutputSet::new(host, result, outputs, roots)?;
    let object = LogicalObject::new(
        ObjectNamespace::ResultValue,
        format!("analysis/output-sets/{}", result.digest()),
        ANALYSIS_OUTPUT_SET_MEDIA_TYPE,
        output_set.canonical_bytes()?,
    )?;
    let root = host.artifact_access.reference(
        &object,
        ValueRefKind::ResultObject,
        ANALYSIS_OUTPUT_SET_VALUE_SCHEMA,
    )?;
    Ok(PreparedAnalysisOutputSet {
        output_set,
        object,
        root,
    })
}
