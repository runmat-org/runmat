//! Validate completed thermal task outputs without reconstructing numerical field arrays.
use crate::{
    decode_prepared_thermal_manifest, import_analysis_objects, import_analysis_output_set,
    thermal_state_publication_profiles, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisHostWorkload, PreparedAnalysisObjects, PreparedThermalTask,
};
use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, StatePublicationProfiles, ThermalNumericalReceiptArtifact,
    ThermalSolverResultManifest, ThermalWorkload,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution_artifact::{
    cache::CacheImport, object::ObjectInventoryLimits, ProgramExecutionResponse,
};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedThermalResult {
    pub host: AnalysisHostWorkload,
    pub manifest: ThermalSolverResultManifest,
    pub objects: PreparedAnalysisObjects,
    pub numerical_receipt: ThermalNumericalReceiptArtifact,
    pub numerical_receipt_objects: PreparedAnalysisObjects,
    pub state_publication_profiles: StatePublicationProfiles,
    pub state_publication_profile_objects: PreparedAnalysisObjects,
    pub response: ProgramExecutionResponse,
}

pub fn import_thermal_result(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedThermalTask,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedThermalResult> {
    response.validate_against(&prepared.program.request)?;
    // Preserve the original task closure: freshly rehashed source/plan objects cannot be
    // substituted at import while retaining a response for another prepared program.
    if prepared.host.workload.inputs.len() != prepared.inputs.len()
        || prepared.inputs.iter().any(|objects| {
            prepared
                .host
                .workload
                .inputs
                .iter()
                .filter(|input| {
                    input.kind == objects.manifest.kind
                        && input.logical_identity == objects.manifest.logical_identity
                        && input.payload_digest == objects.manifest.payload_digest
                        && input.manifest_digest == objects.root.digest
                        && input.manifest_encoded_length == objects.root.encoded_length
                })
                .count()
                != 1
        })
    {
        return Err(AnalysisBridgeError::Identity(
            "thermal original input closure differs",
        ));
    }
    let workload_objects = prepared
        .inputs
        .iter()
        .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::ThermalWorkload)
        .ok_or_else(|| AnalysisBridgeError::Invalid("thermal workload absent".into()))?;
    let workload = ThermalWorkload::from_canonical_bytes(&workload_objects.payload())?;
    let mesh_objects = prepared
        .inputs
        .iter()
        .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::Mesh)
        .ok_or_else(|| AnalysisBridgeError::Invalid("thermal mesh absent".into()))?;
    let mesh = SolverMeshArtifact::canonical_decode(&mesh_objects.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    let plan_objects = prepared
        .inputs
        .iter()
        .find(|input| input.manifest.kind == AnalysisNumericalObjectKind::ResourcePlan)
        .ok_or_else(|| AnalysisBridgeError::Invalid("thermal resource plan absent".into()))?;
    let original_plan =
        runmat_analysis_fea::planning::resource::FeaResourcePlan::from_canonical_bytes(
            &plan_objects.payload(),
        )?;
    let fresh_plan = crate::rederive_thermal_resource_plan(
        &workload,
        &mesh,
        &prepared.inputs,
        prepared.output_policy,
    )?;
    if original_plan != prepared.resource_plan
        || fresh_plan != original_plan
        || original_plan.identity.digest() != plan_objects.manifest.logical_identity
    {
        return Err(AnalysisBridgeError::Identity(
            "thermal admitted plan differs from original retained-source resource derivation",
        ));
    }
    let field_source =
        crate::thermal_plan::validate_thermal_task_source(&workload, &mesh, &prepared.inputs)?;
    let output_set = import_analysis_output_set(source, response.clone(), &prepared.host)?;
    let [result, receipt, profiles] = output_set.manifest.outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "thermal task requires one result, one receipt and one publication profile catalog"
                .into(),
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
    let profile_objects = import_analysis_objects(
        source,
        &profiles.root,
        AnalysisNumericalObjectKind::StatePublicationProfiles,
        profiles.logical_identity,
        profiles.payload_digest,
        &prepared.host.artifact_access,
        limits,
    )?;
    expected.extend(
        receipt_objects
            .unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?,
    );
    expected.extend(
        profile_objects
            .unique_references(&prepared.host.artifact_access, ValueRefKind::ResultObject)?,
    );
    crate::object::retain_unique_references(&mut expected);
    expected.push(output_set.root);
    if expected != output_set.result_objects {
        return Err(AnalysisBridgeError::Identity(
            "thermal result inventory differs from canonical closure",
        ));
    }
    if workload.identity()?.digest() != workload_objects.manifest.logical_identity {
        return Err(AnalysisBridgeError::Identity(
            "thermal workload input identity differs",
        ));
    }
    let manifest = decode_prepared_thermal_manifest(&objects)?;
    let numerical_receipt =
        ThermalNumericalReceiptArtifact::from_canonical_bytes(&receipt_objects.payload())?;
    numerical_receipt.validate_against(&workload, &manifest)?;
    if let Some(source) = field_source {
        let [balance] = manifest.energy_balances.as_slice() else {
            return Err(AnalysisBridgeError::Invalid(
                "retained Joule source requires one steady thermal energy balance".into(),
            ));
        };
        let expected = source.transfer().power_w();
        let actual = balance.applied_power_w.canonical_value();
        let scale = expected.abs().max(actual.abs());
        let relative = if scale == 0.0 {
            0.0
        } else {
            (actual - expected).abs() / scale
        };
        if !relative.is_finite() || relative > workload.residual_target.max(1e-10) * 10.0 {
            return Err(AnalysisBridgeError::Identity(
                "thermal applied source power differs from original retained Joule result",
            ));
        }
    }
    let state_publication_profiles =
        StatePublicationProfiles::from_canonical_bytes(&profile_objects.payload())?;
    state_publication_profiles.validate_source(
        manifest.producer,
        manifest.identity(),
        manifest.topology_digest,
        objects.manifest.payload_digest,
    )?;
    if state_publication_profiles.identity() != profiles.logical_identity
        || state_publication_profiles != thermal_state_publication_profiles(&workload, &objects)?
    {
        return Err(AnalysisBridgeError::Identity(
            "thermal publication profiles differ from canonical metadata projection",
        ));
    }
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
        state_publication_profiles,
        state_publication_profile_objects: profile_objects,
        response,
    })
}
