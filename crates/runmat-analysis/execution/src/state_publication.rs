use std::collections::BTreeSet;

use runmat_analysis_core::StateOutput;
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, CanonicalSemanticState,
    SemanticStatePublicationRequest, SemanticStatePublicationSource, SolverResultIdentity,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution::{Digest, ProgramRevision};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ObjectNamespace, ProgramExecutionResponse};

use crate::{
    import_analysis_objects, import_analysis_output_set, prepare_analysis_objects,
    prepare_analysis_program, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisCapabilityRequirement, AnalysisHostWorkload, AnalysisWorkloadInput,
    AnalysisWorkloadRequest, PreparedAnalysisObjects, PreparedAnalysisProgram, ANALYSIS_HOST_ABI,
};

pub struct SemanticStatePublicationPreparation {
    pub source_host: AnalysisHostWorkload,
    pub source_result: PreparedAnalysisObjects,
    pub source: SemanticStatePublicationSource,
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

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedSemanticState {
    pub state: CanonicalSemanticState,
    pub objects: PreparedAnalysisObjects,
    pub response: ProgramExecutionResponse,
}

pub fn prepare_semantic_state_publication(
    preparation: SemanticStatePublicationPreparation,
) -> AnalysisBridgeResult<PreparedSemanticStatePublication> {
    preparation.source_host.validate()?;
    let source_kind_matches = matches!(
        (preparation.source, preparation.source_host.workload.kind),
        (
            SemanticStatePublicationSource::StructuralPrimaryFields,
            AnalysisWorkKind::Solve
        ) | (
            SemanticStatePublicationSource::StructuralLinearization,
            AnalysisWorkKind::Solve
        ) | (
            SemanticStatePublicationSource::ModalBasis,
            AnalysisWorkKind::ExtractStructuralModes
        ) | (
            SemanticStatePublicationSource::StructuralResponseEnvelope,
            AnalysisWorkKind::AssembleStructuralModalResponseSpectrumResult
        ) | (
            SemanticStatePublicationSource::StructuralRandomStatistics,
            AnalysisWorkKind::AssembleStructuralRandomResponseResult
        )
    );
    if !source_kind_matches
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
    let request = SemanticStatePublicationRequest::new(
        preparation.source,
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
    let inputs = vec![
        (request_objects, ValueRefKind::DriverObject),
        (preparation.source_result, ValueRefKind::ResultObject),
    ];
    let workload_inputs = inputs
        .iter()
        .map(|(input, reference_kind)| AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: *reference_kind,
        })
        .collect();
    let inputs = inputs
        .into_iter()
        .map(|(input, _)| input)
        .collect::<Vec<_>>();
    let required_capabilities = BTreeSet::from([AnalysisCapabilityRequirement::Host {
        abi: ANALYSIS_HOST_ABI.into(),
    }]);
    let workload = AnalysisWorkloadRequest::new(
        preparation.source_host.workload.study,
        preparation.source_host.workload.step,
        preparation.source_host.workload.execution_plan,
        AnalysisWorkKind::PublishState,
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
    Ok(import_semantic_state_completion(source, response, prepared, limits)?.state)
}

pub fn import_semantic_state_completion(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedSemanticStatePublication,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedSemanticState> {
    response.validate_against(&prepared.program.request)?;
    let retained_response = response.clone();
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [output] = output_set.manifest.outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "semantic-state publication must return exactly one state root".into(),
        ));
    };
    let imported = import_analysis_objects(
        source,
        &output.root,
        AnalysisNumericalObjectKind::SemanticState,
        output.logical_identity,
        output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let mut expected =
        imported.unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?;
    expected.push(output_set.root);
    if output_set.result_objects != expected {
        return Err(AnalysisBridgeError::Identity(
            "semantic-state result inventory differs from its canonical closure",
        ));
    }
    let state = CanonicalSemanticState::from_canonical_bytes(&imported.payload())?;
    if output.logical_identity != state.identity.digest()
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
    Ok(ImportedSemanticState {
        state,
        objects: imported,
        response: retained_response,
    })
}
