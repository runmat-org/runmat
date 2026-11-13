//! Import canonical acoustic fields through the shared output-set/inventory authority.
use crate::{
    import_analysis_objects, import_analysis_output_set, AnalysisBridgeError, AnalysisBridgeResult,
    PreparedAcousticHarmonicTask, PreparedAnalysisObjects,
};
use runmat_analysis_fea::execution::{
    AcousticHarmonicSolverResult, AcousticHarmonicWorkload, AcousticNumericalReceiptArtifact,
    AnalysisNumericalObjectKind,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ProgramExecutionResponse,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

pub struct ImportedAcousticHarmonicResult {
    pub result: AcousticHarmonicSolverResult,
    pub objects: PreparedAnalysisObjects,
    pub source_profiles: PreparedAnalysisObjects,
    pub numerical_receipt: AcousticNumericalReceiptArtifact,
    pub numerical_receipt_objects: PreparedAnalysisObjects,
}

pub fn import_acoustic_harmonic_result(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedAcousticHarmonicTask,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedAcousticHarmonicResult> {
    response.validate_against(&prepared.program.request)?;
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [output, receipt, profiles] = output_set.manifest.outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "acoustic task requires exact result, receipt and publication-profile outputs".into(),
        ));
    };
    if output.kind != AnalysisNumericalObjectKind::SolverResult
        || receipt.kind != AnalysisNumericalObjectKind::NumericalReceipt
        || profiles.kind != AnalysisNumericalObjectKind::StatePublicationProfiles
    {
        return Err(AnalysisBridgeError::Identity(
            "acoustic output kinds differ",
        ));
    }
    let objects = import_analysis_objects(
        source,
        &output.root,
        AnalysisNumericalObjectKind::SolverResult,
        output.logical_identity,
        output.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let source_profiles = import_analysis_objects(
        source,
        &profiles.root,
        AnalysisNumericalObjectKind::StatePublicationProfiles,
        profiles.logical_identity,
        profiles.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let numerical_receipt_objects = import_analysis_objects(
        source,
        &receipt.root,
        AnalysisNumericalObjectKind::NumericalReceipt,
        receipt.logical_identity,
        receipt.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    let mut references =
        objects.unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?;
    references.extend(
        numerical_receipt_objects
            .unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?,
    );
    references.extend(
        source_profiles
            .unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?,
    );
    crate::object::retain_unique_references(&mut references);
    references.push(output_set.root);
    if references != output_set.result_objects {
        return Err(AnalysisBridgeError::Identity(
            "acoustic output inventory differs from canonical closure",
        ));
    }
    let input = |kind| -> AnalysisBridgeResult<&PreparedAnalysisObjects> {
        let mut matches = prepared
            .inputs
            .iter()
            .filter(|input| input.manifest.kind == kind);
        let first = matches
            .next()
            .ok_or_else(|| AnalysisBridgeError::Invalid("acoustic task input absent".into()))?;
        if matches.next().is_some() {
            return Err(AnalysisBridgeError::Invalid(
                "duplicate acoustic input".into(),
            ));
        }
        Ok(first)
    };
    let workload_input = input(AnalysisNumericalObjectKind::AcousticHarmonicWorkload)?;
    let mesh_input = input(AnalysisNumericalObjectKind::Mesh)?;
    let workload = AcousticHarmonicWorkload::from_canonical_bytes(&workload_input.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    mesh.validate_canonical()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if workload.identity()?.digest() != workload_input.manifest.logical_identity
        || runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes())
            != mesh_input.manifest.logical_identity
    {
        return Err(AnalysisBridgeError::Identity(
            "acoustic source input identity differs",
        ));
    }
    let result = AcousticHarmonicSolverResult::from_canonical_bytes(&objects.payload())?;
    result.validate_for(&workload, &mesh)?;
    if result.source_result_identity()?.digest() != objects.manifest.logical_identity {
        return Err(AnalysisBridgeError::Identity(
            "acoustic result logical identity differs",
        ));
    }
    let numerical_receipt = AcousticNumericalReceiptArtifact::from_canonical_bytes(
        &numerical_receipt_objects.payload(),
    )?;
    numerical_receipt.validate_against(&workload, &result, &mesh)?;
    if numerical_receipt.identity().digest() != numerical_receipt_objects.manifest.logical_identity
    {
        return Err(AnalysisBridgeError::Identity(
            "acoustic receipt logical identity differs",
        ));
    }
    let catalog = runmat_analysis_fea::execution::StatePublicationProfiles::from_canonical_bytes(
        &source_profiles.payload(),
    )?;
    if catalog.identity() != source_profiles.manifest.logical_identity
        || catalog != result.state_publication_profiles(objects.manifest.payload_digest)?
    {
        return Err(AnalysisBridgeError::Identity(
            "acoustic publication profiles differ from canonical source",
        ));
    }
    Ok(ImportedAcousticHarmonicResult {
        result,
        objects,
        source_profiles,
        numerical_receipt,
        numerical_receipt_objects,
    })
}
