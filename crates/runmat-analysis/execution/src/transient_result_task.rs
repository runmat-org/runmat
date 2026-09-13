use std::collections::{BTreeMap, BTreeSet};

use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, StructuralTransientNumericalReceipt,
    StructuralTransientResultAssemblyRequest, StructuralTransientSolverResult,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ArtifactAccess, ObjectNamespace, ProgramExecutionResponse};

use crate::{
    decode_prepared_structural_transient_numerical_receipt,
    decode_prepared_structural_transient_solver_result, import_analysis_objects,
    import_analysis_output_set, prepare_analysis_program,
    prepare_structural_transient_result_assembly_request_objects, AnalysisBridgeError,
    AnalysisBridgeResult, AnalysisCapabilityRequirement, AnalysisHostWorkload,
    AnalysisWorkloadInput, AnalysisWorkloadRequest, ImportedStructuralTransientBlock,
    PreparedAnalysisObjects, PreparedAnalysisProgram, ANALYSIS_HOST_ABI,
};

pub struct StructuralTransientResultPreparation {
    pub blocks: Vec<ImportedStructuralTransientBlock>,
    pub artifact_access: ArtifactAccess,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub struct PreparedStructuralTransientResultTask {
    pub request: StructuralTransientResultAssemblyRequest,
    pub blocks: Vec<ImportedStructuralTransientBlock>,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedStructuralTransientResult {
    pub result: StructuralTransientSolverResult,
    pub numerical_receipt: StructuralTransientNumericalReceipt,
    pub result_objects: PreparedAnalysisObjects,
    pub receipt_objects: PreparedAnalysisObjects,
}

pub fn prepare_structural_transient_result_task(
    preparation: StructuralTransientResultPreparation,
) -> AnalysisBridgeResult<PreparedStructuralTransientResultTask> {
    if preparation.chunk_bytes == 0 || preparation.blocks.is_empty() {
        return Err(AnalysisBridgeError::Invalid(
            "structural transient result assembly requires blocks and bounded chunks".into(),
        ));
    }
    let mut by_ordinal = BTreeMap::new();
    for block in preparation.blocks {
        let ordinal = block.segment.manifest().request.block_ordinal;
        if by_ordinal.insert(ordinal, block).is_some() {
            return Err(AnalysisBridgeError::Identity(
                "structural transient result assembly contains a duplicate block ordinal",
            ));
        }
    }
    let blocks = by_ordinal.into_values().collect::<Vec<_>>();
    let segments = blocks
        .iter()
        .map(|block| block.segment.clone())
        .collect::<Vec<_>>();
    let request = StructuralTransientResultAssemblyRequest::new(&segments)?;
    let request_objects = prepare_structural_transient_result_assembly_request_objects(
        &request,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let mut typed_inputs = vec![(request_objects, ValueRefKind::DriverObject)];
    typed_inputs.extend(
        blocks
            .iter()
            .map(|block| (block.objects.clone(), ValueRefKind::ResultObject)),
    );
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
    let workload = AnalysisWorkloadRequest::new(
        request.scope.study,
        request.scope.step,
        request.scope.execution_plan,
        AnalysisWorkKind::AssembleStructuralTransientResult,
        workload_inputs,
        vec![
            AnalysisNumericalObjectKind::SolverResult,
            AnalysisNumericalObjectKind::NumericalReceipt,
        ],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )?;
    let host = AnalysisHostWorkload::new(workload, preparation.artifact_access)?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &inputs)?;
    Ok(PreparedStructuralTransientResultTask {
        request,
        blocks,
        host,
        program,
        inputs,
    })
}

pub fn import_structural_transient_result(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedStructuralTransientResultTask,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedStructuralTransientResult> {
    response.validate_against(&prepared.program.request)?;
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [result_output, receipt_output] = output_set.manifest.outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "structural transient assembly must return result and numerical-receipt roots".into(),
        ));
    };
    let result_objects = import_analysis_objects(
        source,
        &result_output.root,
        AnalysisNumericalObjectKind::SolverResult,
        result_output.logical_identity,
        result_output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let receipt_objects = import_analysis_objects(
        source,
        &receipt_output.root,
        AnalysisNumericalObjectKind::NumericalReceipt,
        receipt_output.logical_identity,
        receipt_output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let result = decode_prepared_structural_transient_solver_result(&result_objects)?;
    let numerical_receipt =
        decode_prepared_structural_transient_numerical_receipt(&receipt_objects)?;
    let segments = prepared
        .blocks
        .iter()
        .map(|block| block.segment.clone())
        .collect::<Vec<_>>();
    numerical_receipt.validate_for(&result, &prepared.request, &segments)?;
    if result.manifest().identity.digest() != result_output.logical_identity
        || numerical_receipt.identity.digest() != receipt_output.logical_identity
    {
        return Err(AnalysisBridgeError::Identity(
            "structural transient outputs differ from their advertised identities",
        ));
    }
    let mut expected = result_objects
        .unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?;
    expected.extend(
        receipt_objects
            .unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?,
    );
    crate::object::retain_unique_references(&mut expected);
    expected.push(output_set.root);
    if output_set.result_objects != expected {
        return Err(AnalysisBridgeError::Identity(
            "structural transient result inventory differs from its canonical closure",
        ));
    }
    Ok(ImportedStructuralTransientResult {
        result,
        numerical_receipt,
        result_objects,
        receipt_objects,
    })
}
