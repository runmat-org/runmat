//! Import canonical Maxwell fields through the shared output-set/inventory authority.
use crate::{
    import_analysis_objects, import_analysis_output_set, AnalysisBridgeError, AnalysisBridgeResult,
    PreparedAnalysisObjects, PreparedMaxwellHarmonicTask,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, MaxwellHarmonicSolverResult, MaxwellHarmonicWorkload,
    MaxwellNumericalReceiptArtifact,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ProgramExecutionResponse,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};
mod admission;

pub struct ImportedMaxwellHarmonicResult {
    pub result: MaxwellHarmonicSolverResult,
    pub objects: PreparedAnalysisObjects,
    pub source_profiles: PreparedAnalysisObjects,
    pub numerical_receipt: MaxwellNumericalReceiptArtifact,
    pub numerical_receipt_objects: PreparedAnalysisObjects,
}

pub fn import_maxwell_harmonic_result(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedMaxwellHarmonicTask,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedMaxwellHarmonicResult> {
    prepared.host.workload.validate()?;
    if prepared.host.workload.kind
        != runmat_analysis_fea::execution::AnalysisWorkKind::SolveMaxwellHarmonic
        || prepared.host.workload.outputs
            != [
                AnalysisNumericalObjectKind::SolverResult,
                AnalysisNumericalObjectKind::NumericalReceipt,
                AnalysisNumericalObjectKind::StatePublicationProfiles,
            ]
    {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell task kind or declared output set differs",
        ));
    }
    if prepared.inputs.len() != 3 || prepared.host.workload.inputs.len() != 3 {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell source input inventory differs",
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
            "Maxwell inputs differ from their immutable task",
        ));
    }
    admission::require_source_decode_budget(prepared, limits)?;
    let input = |kind| -> AnalysisBridgeResult<&PreparedAnalysisObjects> {
        let mut matches = prepared
            .inputs
            .iter()
            .filter(|input| input.manifest.kind == kind);
        let first = matches
            .next()
            .ok_or_else(|| AnalysisBridgeError::Invalid("Maxwell task input absent".into()))?;
        if matches.next().is_some() {
            return Err(AnalysisBridgeError::Invalid(
                "duplicate Maxwell input".into(),
            ));
        }
        Ok(first)
    };
    let workload_input = input(AnalysisNumericalObjectKind::MaxwellHarmonicWorkload)?;
    let mesh_input = input(AnalysisNumericalObjectKind::Mesh)?;
    let plan_input = input(AnalysisNumericalObjectKind::ResourcePlan)?;
    let workload = MaxwellHarmonicWorkload::from_canonical_bytes(&workload_input.payload())?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_input.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let rederived = crate::rederive_maxwell_harmonic_resource_plan(
        &workload,
        &mesh,
        &prepared.inputs,
        prepared.output_policy,
    )?;
    if rederived != prepared.resource_plan {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell task resource policy differs from canonical source authority",
        ));
    }
    let plan = runmat_analysis_fea::planning::resource::FeaResourcePlan::from_canonical_bytes(
        &plan_input.payload(),
    )?;
    let numerical =
        runmat_analysis_fea::execution::NumericalPlanIdentity::from_maxwell_harmonic_workload(
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
            "Maxwell plan or producer differs from its task",
        ));
    }
    if workload.identity()?.digest() != workload_input.manifest.logical_identity
        || runmat_execution::Digest::from_bytes(*mesh.canonical_digest.bytes())
            != mesh_input.manifest.logical_identity
    {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell source input identity differs",
        ));
    }

    response.validate_against(&prepared.program.request)?;
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
    let [output, receipt, profiles] = output_set.manifest.outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "Maxwell task requires exact result, receipt and publication-profile outputs".into(),
        ));
    };
    if output.kind != AnalysisNumericalObjectKind::SolverResult
        || receipt.kind != AnalysisNumericalObjectKind::NumericalReceipt
        || profiles.kind != AnalysisNumericalObjectKind::StatePublicationProfiles
    {
        return Err(AnalysisBridgeError::Identity("Maxwell output kinds differ"));
    }
    let runmat_analysis_fea::planning::resource::FeaResourceDemand::MaxwellHarmonic(demand) =
        &rederived.input.demand
    else {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell output resource family differs",
        ));
    };
    let bounded_limits =
        |facts: &runmat_analysis_fea::planning::resource::PreparedOutputResourceFacts| {
            let mut bounded = limits;
            bounded.max_total_bytes = bounded
                .max_total_bytes
                .min(facts.transport_closure_bytes_upper_bound.0);
            bounded
        };
    let objects = import_analysis_objects(
        source,
        &output.root,
        AnalysisNumericalObjectKind::SolverResult,
        output.logical_identity,
        output.payload_digest,
        &prepared.host.artifact_access,
        bounded_limits(&demand.result_output),
    )?;
    let source_profiles = import_analysis_objects(
        source,
        &profiles.root,
        AnalysisNumericalObjectKind::StatePublicationProfiles,
        profiles.logical_identity,
        profiles.payload_digest,
        &prepared.host.artifact_access,
        bounded_limits(&demand.state_profiles_output),
    )?;
    let numerical_receipt_objects = import_analysis_objects(
        source,
        &receipt.root,
        AnalysisNumericalObjectKind::NumericalReceipt,
        receipt.logical_identity,
        receipt.payload_digest,
        &prepared.host.artifact_access,
        bounded_limits(&demand.receipt_output),
    )?;
    if objects.manifest.encoded_length > demand.result_output.encoded_payload_bytes_upper_bound.0
        || source_profiles.manifest.encoded_length
            > demand
                .state_profiles_output
                .encoded_payload_bytes_upper_bound
                .0
        || numerical_receipt_objects.manifest.encoded_length
            > demand.receipt_output.encoded_payload_bytes_upper_bound.0
    {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell outputs exceed admitted payload bounds",
        ));
    }
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
            "Maxwell output inventory differs from canonical closure",
        ));
    }
    let result = MaxwellHarmonicSolverResult::from_canonical_bytes_for(
        &objects.payload(),
        objects.manifest.payload_digest,
        &workload,
        &mesh,
    )?;
    if result.identity()?.digest() != objects.manifest.logical_identity {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell result logical identity differs",
        ));
    }
    let numerical_receipt = MaxwellNumericalReceiptArtifact::from_canonical_bytes(
        &numerical_receipt_objects.payload(),
    )?;
    numerical_receipt.validate_against(&workload, &result, &mesh)?;
    if numerical_receipt.identity().digest() != numerical_receipt_objects.manifest.logical_identity
    {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell receipt logical identity differs",
        ));
    }
    let catalog = runmat_analysis_fea::execution::StatePublicationProfiles::from_canonical_bytes(
        &source_profiles.payload(),
    )?;
    if catalog.identity() != source_profiles.manifest.logical_identity
        || catalog != result.state_publication_profiles(objects.manifest.payload_digest)?
    {
        return Err(AnalysisBridgeError::Identity(
            "Maxwell publication profiles differ from canonical source",
        ));
    }
    Ok(ImportedMaxwellHarmonicResult {
        result,
        objects,
        source_profiles,
        numerical_receipt,
        numerical_receipt_objects,
    })
}
