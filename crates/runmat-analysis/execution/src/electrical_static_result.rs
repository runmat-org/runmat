//! Import canonical electrical fields through the shared output-set/inventory authority.
use crate::{
    import_analysis_objects, import_analysis_output_set, AnalysisBridgeError, AnalysisBridgeResult,
    PreparedAnalysisObjects, PreparedElectricalStaticTask,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, ElectricalNumericalReceiptArtifact, ElectricalStaticSolverResult,
    ElectricalStaticWorkload,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ProgramExecutionResponse,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

pub struct ImportedElectricalStaticResult {
    pub result: ElectricalStaticSolverResult,
    pub objects: PreparedAnalysisObjects,
    pub source_profiles: PreparedAnalysisObjects,
    pub numerical_receipt: ElectricalNumericalReceiptArtifact,
    pub numerical_receipt_objects: PreparedAnalysisObjects,
}

pub fn import_electrical_static_result(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedElectricalStaticTask,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedElectricalStaticResult> {
    prepared.host.workload.validate()?;
    if prepared.host.workload.kind
        != runmat_analysis_fea::execution::AnalysisWorkKind::SolveElectricalStatic
        || prepared.host.workload.outputs
            != [
                AnalysisNumericalObjectKind::SolverResult,
                AnalysisNumericalObjectKind::NumericalReceipt,
                AnalysisNumericalObjectKind::StatePublicationProfiles,
            ]
    {
        return Err(AnalysisBridgeError::Identity(
            "electrical task kind or declared output set differs",
        ));
    }
    if prepared.inputs.len() != 3 || prepared.host.workload.inputs.len() != 3 {
        return Err(AnalysisBridgeError::Identity(
            "electrical source input inventory differs",
        ));
    }
    let expected_inputs = prepared
        .inputs
        .iter()
        .map(|input| crate::AnalysisWorkloadInput {
            kind: input.manifest.kind,
            logical_identity: input.manifest.logical_identity,
            payload_digest: input.manifest.payload_digest,
            manifest_digest: input.root.digest,
            manifest_encoded_length: input.root.encoded_length,
            reference_kind: ValueRefKind::DriverObject,
        })
        .collect::<Vec<_>>();
    if expected_inputs != prepared.host.workload.inputs {
        return Err(AnalysisBridgeError::Identity(
            "electrical inputs differ from their immutable task",
        ));
    }
    response.validate_against(&prepared.program.request)?;
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [output, receipt, profiles] = output_set.manifest.outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "electrical task requires exact result, receipt and publication-profile outputs".into(),
        ));
    };
    if output.kind != AnalysisNumericalObjectKind::SolverResult
        || receipt.kind != AnalysisNumericalObjectKind::NumericalReceipt
        || profiles.kind != AnalysisNumericalObjectKind::StatePublicationProfiles
    {
        return Err(AnalysisBridgeError::Identity(
            "electrical output kinds differ",
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
            "electrical output inventory differs from canonical closure",
        ));
    }
    let input = |kind| -> AnalysisBridgeResult<&PreparedAnalysisObjects> {
        let mut matches = prepared
            .inputs
            .iter()
            .filter(|input| input.manifest.kind == kind);
        let first = matches
            .next()
            .ok_or_else(|| AnalysisBridgeError::Invalid("electrical task input absent".into()))?;
        if matches.next().is_some() {
            return Err(AnalysisBridgeError::Invalid(
                "duplicate electrical input".into(),
            ));
        }
        Ok(first)
    };
    let workload_input = input(AnalysisNumericalObjectKind::ElectricalStaticWorkload)?;
    let mesh_input = input(AnalysisNumericalObjectKind::Mesh)?;
    let plan_input = input(AnalysisNumericalObjectKind::ResourcePlan)?;
    let workload = ElectricalStaticWorkload::from_canonical_bytes(&workload_input.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    mesh.validate_canonical()
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let rederived = crate::rederive_electrical_static_resource_plan(
        &workload,
        &mesh,
        &prepared.inputs,
        prepared.output_policy,
    )?;
    if rederived != prepared.resource_plan {
        return Err(AnalysisBridgeError::Identity(
            "electrical task resource policy differs from canonical source authority",
        ));
    }
    let plan = runmat_analysis_fea::planning::resource::FeaResourcePlan::from_canonical_bytes(
        &plan_input.payload(),
    )?;
    let numerical =
        runmat_analysis_fea::execution::NumericalPlanIdentity::from_electrical_static_workload(
            workload.identity()?,
            runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes()),
        )?;
    let execution = runmat_analysis_fea::execution::ExecutionPlanIdentity::new(
        numerical,
        plan.identity.digest(),
    )?;
    if plan != prepared.resource_plan
        || plan_input.manifest.logical_identity != plan.identity.digest()
        || prepared.host.workload.execution_plan != execution
        || prepared.host.workload.step != workload.producer_identity()?
    {
        return Err(AnalysisBridgeError::Identity(
            "electrical plan or producer differs from its task",
        ));
    }
    if workload.identity()?.digest() != workload_input.manifest.logical_identity
        || runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes())
            != mesh_input.manifest.logical_identity
    {
        return Err(AnalysisBridgeError::Identity(
            "electrical source input identity differs",
        ));
    }
    let result = ElectricalStaticSolverResult::from_canonical_bytes(&objects.payload())?;
    result.validate_for(&workload, &mesh)?;
    if result.source_result_identity()?.digest() != objects.manifest.logical_identity {
        return Err(AnalysisBridgeError::Identity(
            "electrical result logical identity differs",
        ));
    }
    let numerical_receipt = ElectricalNumericalReceiptArtifact::from_canonical_bytes(
        &numerical_receipt_objects.payload(),
    )?;
    numerical_receipt.validate_against(&workload, &result, &mesh)?;
    if numerical_receipt.identity().digest() != numerical_receipt_objects.manifest.logical_identity
    {
        return Err(AnalysisBridgeError::Identity(
            "electrical receipt logical identity differs",
        ));
    }
    let catalog = runmat_analysis_fea::execution::StatePublicationProfiles::from_canonical_bytes(
        &source_profiles.payload(),
    )?;
    if catalog.identity() != source_profiles.manifest.logical_identity
        || catalog != result.state_publication_profiles(objects.manifest.payload_digest)?
    {
        return Err(AnalysisBridgeError::Identity(
            "electrical publication profiles differ from canonical source",
        ));
    }
    Ok(ImportedElectricalStaticResult {
        result,
        objects,
        source_profiles,
        numerical_receipt,
        numerical_receipt_objects,
    })
}
