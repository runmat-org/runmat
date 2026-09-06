use std::collections::BTreeSet;

use runmat_analysis_core::StateOutput;
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisObjectManifest, AnalysisWorkKind, CanonicalSemanticState,
    SemanticStatePublicationRequest, SolverResultIdentity,
};
use runmat_execution::value::{ValuePayload, ValueRefKind};
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ObjectNamespace, ProgramExecutionResponse};

use crate::{
    import_analysis_objects, prepare_analysis_objects, prepare_analysis_program,
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisCapabilityRequirement, AnalysisHostWorkload,
    AnalysisWorkloadInput, AnalysisWorkloadRequest, PreparedAnalysisObjects,
    PreparedAnalysisProgram, ANALYSIS_HOST_ABI,
};

pub struct SemanticStatePublicationPreparation {
    pub source_host: AnalysisHostWorkload,
    pub source_result: PreparedAnalysisObjects,
    pub declaration: StateOutput,
    pub topology_digest: Digest,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub struct PreparedSemanticStatePublication {
    pub request: SemanticStatePublicationRequest,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
}

pub fn prepare_semantic_state_publication(
    preparation: SemanticStatePublicationPreparation,
) -> AnalysisBridgeResult<PreparedSemanticStatePublication> {
    preparation.source_host.validate()?;
    if preparation.source_host.workload.kind != AnalysisWorkKind::Solve
        || preparation.source_result.manifest.kind != AnalysisNumericalObjectKind::SolverResult
        || preparation.chunk_bytes == 0
    {
        return Err(AnalysisBridgeError::Invalid(
            "semantic-state publication requires one completed solver result and bounded chunks"
                .into(),
        ));
    }
    let source_result = SolverResultIdentity::new(
        preparation.source_host.workload.identity,
        preparation.source_result.manifest.payload_digest,
    )?;
    if preparation.source_result.manifest.logical_identity != source_result.digest() {
        return Err(AnalysisBridgeError::Identity(
            "solver result identity does not match its source workload and content",
        ));
    }
    let request = SemanticStatePublicationRequest::structural_linear(
        preparation.source_host.workload.step,
        preparation.declaration,
        preparation.topology_digest,
        source_result,
    )?;
    let request_bytes = request.canonical_bytes()?;
    let request_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::StatePublicationRequest,
        Digest::sha256(&request_bytes),
        &request_bytes,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let inputs = vec![request_objects, preparation.source_result];
    let workload_inputs = inputs
        .iter()
        .map(|input| AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: ValueRefKind::DriverObject,
        })
        .collect();
    let required_capabilities = BTreeSet::from([AnalysisCapabilityRequirement::Host {
        abi: ANALYSIS_HOST_ABI.into(),
    }]);
    let workload = AnalysisWorkloadRequest::new(
        preparation.source_host.workload.study,
        preparation.source_host.workload.step,
        preparation.source_host.workload.execution_plan,
        AnalysisWorkKind::RecoverField,
        workload_inputs,
        vec![AnalysisNumericalObjectKind::SemanticState],
        required_capabilities,
    )?;
    let host = AnalysisHostWorkload::new(workload, preparation.source_host.artifact_access)?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &inputs)?;
    Ok(PreparedSemanticStatePublication {
        request,
        host,
        program,
        inputs,
    })
}

pub fn import_semantic_state(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedSemanticStatePublication,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<CanonicalSemanticState> {
    response.validate_against(&prepared.program.request)?;
    let ProgramExecutionResponse::ExternalizedSuccess {
        outputs,
        result_objects,
    } = response
    else {
        return Err(AnalysisBridgeError::Invalid(
            "semantic-state publication did not return an externalized success".into(),
        ));
    };
    let [ValuePayload::Object(root)] = outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "semantic-state publication must return exactly one state root".into(),
        ));
    };
    let manifest_bytes = source
        .read_verified(root.logical_digest)?
        .ok_or_else(|| AnalysisBridgeError::Invalid("semantic-state manifest is absent".into()))?;
    let manifest = AnalysisObjectManifest::from_canonical_bytes(&manifest_bytes)?;
    let imported = import_analysis_objects(
        source,
        root,
        AnalysisNumericalObjectKind::SemanticState,
        manifest.logical_identity,
        manifest.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let expected =
        imported.references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?;
    if result_objects != expected {
        return Err(AnalysisBridgeError::Identity(
            "semantic-state result inventory differs from its canonical closure",
        ));
    }
    let state = CanonicalSemanticState::from_canonical_bytes(&imported.payload())?;
    if manifest.logical_identity != state.identity.digest()
        || state.producer != prepared.request.producer
        || state.state_id != prepared.request.declaration.state_id
        || state.kind != prepared.request.declaration.kind
        || state.completeness != prepared.request.declaration.completeness
        || state.persistence != prepared.request.declaration.persistence
        || state.topology_digest != prepared.request.topology_digest
        || state.source_result != prepared.request.source_result
    {
        return Err(AnalysisBridgeError::Identity(
            "published semantic state differs from its admitted request",
        ));
    }
    Ok(state)
}
