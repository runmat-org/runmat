use std::collections::BTreeSet;

use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, StructuralTransientBlockRequest,
    StructuralTransientCheckpoint, StructuralTransientSegment, StructuralTransientWorkload,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::ProgramExecutionResponse;
use runmat_meshing_core::CanonicalMeshingContract;

use crate::{
    decode_prepared_structural_transient_checkpoint, decode_prepared_structural_transient_segment,
    decode_prepared_structural_transient_workload, import_analysis_objects,
    import_analysis_output_set, prepare_analysis_program, AnalysisBridgeError,
    AnalysisBridgeResult, AnalysisCapabilityRequirement, AnalysisHostWorkload,
    AnalysisWorkloadInput, AnalysisWorkloadRequest, PreparedAnalysisObjects,
    PreparedAnalysisProgram, ANALYSIS_HOST_ABI,
};

pub struct StructuralTransientBlockPreparation {
    pub workload: PreparedAnalysisObjects,
    pub mesh: PreparedAnalysisObjects,
    pub request: PreparedAnalysisObjects,
    pub predecessor: Option<PreparedAnalysisObjects>,
    pub artifact_access: runmat_execution_artifact::ArtifactAccess,
    pub program_revision: ProgramRevision,
}

pub struct PreparedStructuralTransientBlockTask {
    pub workload: StructuralTransientWorkload,
    pub request: StructuralTransientBlockRequest,
    pub predecessor: Option<StructuralTransientCheckpoint>,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedStructuralTransientBlock {
    pub segment: StructuralTransientSegment,
    pub objects: PreparedAnalysisObjects,
}

pub fn prepare_structural_transient_block_task(
    preparation: StructuralTransientBlockPreparation,
) -> AnalysisBridgeResult<PreparedStructuralTransientBlockTask> {
    let workload = decode_prepared_structural_transient_workload(&preparation.workload)?;
    if preparation.mesh.manifest.kind != AnalysisNumericalObjectKind::Mesh {
        return Err(AnalysisBridgeError::Invalid(
            "structural transient block requires one canonical mesh object".into(),
        ));
    }
    let request = crate::decode_prepared_structural_transient_block_request(&preparation.request)?;
    let predecessor = preparation
        .predecessor
        .as_ref()
        .map(decode_prepared_structural_transient_checkpoint)
        .transpose()?;
    request.validate_against_predecessor(predecessor.as_ref())?;
    let solver_mesh =
        runmat_meshing_core::SolverMeshArtifact::canonical_decode(&preparation.mesh.payload())
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if preparation.workload.manifest.logical_identity != request.scope.source_workload.digest()
        || preparation.mesh.manifest.logical_identity
            != runmat_execution::Digest::from_bytes(*solver_mesh.canonical_digest.bytes())
    {
        return Err(AnalysisBridgeError::Identity(
            "structural transient workload or mesh differs from the block scope",
        ));
    }
    let mut typed_inputs = vec![
        (preparation.workload, ValueRefKind::DriverObject),
        (preparation.mesh, ValueRefKind::DriverObject),
        (preparation.request, ValueRefKind::DriverObject),
    ];
    if let Some(predecessor) = preparation.predecessor {
        typed_inputs.push((predecessor, ValueRefKind::ResultObject));
    }
    let workload_inputs = typed_inputs
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
    let inputs = typed_inputs
        .into_iter()
        .map(|(input, _)| input)
        .collect::<Vec<_>>();
    let analysis_workload = AnalysisWorkloadRequest::new(
        request.scope.study,
        request.scope.step,
        request.scope.execution_plan,
        AnalysisWorkKind::SolveStructuralTransientBlock,
        workload_inputs,
        vec![AnalysisNumericalObjectKind::StructuralTransientSegment],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )?;
    let host = AnalysisHostWorkload::new(analysis_workload, preparation.artifact_access)?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &inputs)?;
    Ok(PreparedStructuralTransientBlockTask {
        workload,
        request,
        predecessor,
        host,
        program,
        inputs,
    })
}

pub fn import_structural_transient_block(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedStructuralTransientBlockTask,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedStructuralTransientBlock> {
    response.validate_against(&prepared.program.request)?;
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [output] = output_set.manifest.outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "structural transient block must return exactly one segment".into(),
        ));
    };
    let objects = import_analysis_objects(
        source,
        &output.root,
        AnalysisNumericalObjectKind::StructuralTransientSegment,
        output.logical_identity,
        output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let segment = decode_prepared_structural_transient_segment(&objects)?;
    if segment.manifest().request != prepared.request
        || segment.manifest().terminal_checkpoint.predecessor
            != prepared.predecessor.as_ref().map(|value| value.identity)
    {
        return Err(AnalysisBridgeError::Identity(
            "structural transient segment differs from its prepared block lineage",
        ));
    }
    let mut expected_objects =
        objects.unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?;
    expected_objects.push(output_set.root);
    if output_set.result_objects != expected_objects {
        return Err(AnalysisBridgeError::Identity(
            "structural transient block result inventory differs from its canonical closure",
        ));
    }
    Ok(ImportedStructuralTransientBlock { segment, objects })
}
