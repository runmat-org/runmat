//! Validate completed thermal task outputs without reconstructing numerical field arrays.
use crate::{
    decode_prepared_thermal_manifest, import_analysis_objects, import_analysis_output_set,
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisHostWorkload, PreparedAnalysisObjects,
    PreparedThermalTask,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, ThermalNumericalReceiptArtifact, ThermalSolverResultManifest,
    ThermalWorkload,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ProgramExecutionResponse,
};

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedThermalResult {
    pub host: AnalysisHostWorkload,
    pub manifest: ThermalSolverResultManifest,
    pub objects: PreparedAnalysisObjects,
    pub numerical_receipt: ThermalNumericalReceiptArtifact,
    pub numerical_receipt_objects: PreparedAnalysisObjects,
    pub response: ProgramExecutionResponse,
}

pub fn import_thermal_result(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedThermalTask,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedThermalResult> {
    response.validate_against(&prepared.program.request)?;
    let output_set = import_analysis_output_set(source, response.clone(), &prepared.host)?;
    let [result, receipt] = output_set.manifest.outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "thermal task requires one result and one receipt".into(),
        ));
    };
    let objects = import_analysis_objects(
        source,
        &result.root,
        AnalysisNumericalObjectKind::SolverResult,
        result.logical_identity,
        result.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let receipt_objects = import_analysis_objects(
        source,
        &receipt.root,
        AnalysisNumericalObjectKind::NumericalReceipt,
        receipt.logical_identity,
        receipt.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let mut expected =
        objects.unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?;
    expected.extend(
        receipt_objects
            .unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?,
    );
    crate::object::retain_unique_references(&mut expected);
    expected.push(output_set.root);
    if expected != output_set.result_objects {
        return Err(AnalysisBridgeError::Identity(
            "thermal result inventory differs from canonical closure",
        ));
    }
    let workload_objects = prepared
        .inputs
        .iter()
        .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::ThermalWorkload)
        .ok_or_else(|| AnalysisBridgeError::Invalid("thermal workload absent".into()))?;
    let workload = ThermalWorkload::from_canonical_bytes(&workload_objects.payload())?;
    if workload.identity()?.digest() != workload_objects.manifest.logical_identity {
        return Err(AnalysisBridgeError::Identity(
            "thermal workload input identity differs",
        ));
    }
    let manifest = decode_prepared_thermal_manifest(&objects)?;
    let numerical_receipt =
        ThermalNumericalReceiptArtifact::from_canonical_bytes(&receipt_objects.payload())?;
    numerical_receipt.validate_against(&workload, &manifest)?;
    if numerical_receipt.identity().digest() != receipt.logical_identity
        || manifest.producer != prepared.host.workload.step
        || !prepared.inputs.iter().any(|input| {
            input.manifest.kind == AnalysisNumericalObjectKind::Mesh
                && input.manifest.logical_identity == manifest.topology_digest
        })
    {
        return Err(AnalysisBridgeError::Identity(
            "thermal result producer, topology or receipt differs",
        ));
    }
    Ok(ImportedThermalResult {
        host: prepared.host.clone(),
        manifest,
        objects,
        numerical_receipt,
        numerical_receipt_objects: receipt_objects,
        response,
    })
}
