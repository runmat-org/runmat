mod preparation;
mod resource;
pub use preparation::prepare_structural_field_recovery;
pub use resource::{
    build_prepared_structural_recovery_resource_plan,
    build_structural_recovery_metadata_resource_plan, require_structural_recovery_decode_budget,
};
mod objects;
pub use objects::{prepare_recovered_field_objects, recovered_field_transport_reservation_bytes};

use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, FieldRecoveryRequest, FieldRecoverySource,
};
use runmat_analysis_fea::results::field::{
    reconstruct_recovered_field, RecoveredFieldArtifact, RecoveredFieldArtifactManifest,
    RecoveredFieldPartition,
};
use runmat_execution::{value::ValueRefKind, ProgramRevision};
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ObjectNamespace, ProgramExecutionResponse};

use crate::{
    analysis_object_references, import_analysis_output_set, import_analysis_partition,
    AnalysisBridgeError, AnalysisBridgeResult, AnalysisHostWorkload, AnalysisOutputRoot,
    AnalysisPartitionImport, ImportedAnalysisPartition, PreparedAnalysisObjects,
    PreparedAnalysisProgram,
};

pub struct StructuralFieldRecoveryPreparation {
    pub source_host: AnalysisHostWorkload,
    pub source_invocation: PreparedAnalysisObjects,
    pub source_result: PreparedAnalysisObjects,
    pub mesh: Option<PreparedAnalysisObjects>,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
    /// The study's admitted local execution plan; sizes the recovery worker.
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
}

pub struct PreparedStructuralFieldRecovery {
    pub request: FieldRecoveryRequest,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
    pub resource_plan: runmat_analysis_fea::planning::resource::FeaResourcePlan,
    /// The study's admitted local execution plan; sizes the recovery worker.
    pub local_execution: runmat_analysis_fea::planning::resource::LocalExecutionPlan,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedRecoveredFields {
    pub manifest: RecoveredFieldArtifactManifest,
    pub catalog_objects: ImportedAnalysisPartition,
    pub output: AnalysisOutputRoot,
}

pub fn import_structural_recovered_fields(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedStructuralFieldRecovery,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedRecoveredFields> {
    require_structural_recovery_decode_budget(&prepared.inputs, &prepared.resource_plan, limits)?;
    if prepared.resource_plan
        != build_prepared_structural_recovery_resource_plan(
            &prepared.request,
            &prepared.inputs,
            prepared.host.output_policy,
            prepared.local_execution,
        )?
    {
        return Err(AnalysisBridgeError::Identity(
            "cold static recovery plan differs from original inputs",
        ));
    }
    let demand = prepared
        .resource_plan
        .input
        .demand
        .output_facts(AnalysisNumericalObjectKind::Field)
        .ok_or_else(|| AnalysisBridgeError::Invalid("static field output demand absent".into()))?;
    let mut limits = limits;
    limits.max_total_bytes = limits
        .max_total_bytes
        .min(demand.transport_closure_bytes_upper_bound.0);
    import_recovered_fields(
        source,
        response,
        &prepared.request,
        &prepared.host,
        &prepared.program,
        limits,
    )
}

pub(crate) fn import_recovered_fields(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    request: &FieldRecoveryRequest,
    host: &AnalysisHostWorkload,
    program: &PreparedAnalysisProgram,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedRecoveredFields> {
    response.validate_against(&program.request)?;
    let output_set = import_analysis_output_set(source, response, host)?;
    let [output] = output_set.manifest.outputs.as_slice() else {
        return Err(AnalysisBridgeError::Invalid(
            "field recovery must return exactly one field artifact root".into(),
        ));
    };
    let imported = import_analysis_partition(
        source,
        AnalysisPartitionImport {
            root: &output.root,
            expected_kind: AnalysisNumericalObjectKind::Field,
            expected_logical_identity: output.logical_identity,
            expected_payload_digest: output.payload_digest,
            partition_identity: output.logical_identity,
            access: &host.artifact_access,
            limits,
        },
    )?;
    let mut expected = analysis_object_references(
        &imported.manifest,
        &output.root,
        &host.artifact_access,
        ValueRefKind::ResultObject,
    )?;
    expected.push(output_set.root);
    if output_set.result_objects != expected {
        return Err(AnalysisBridgeError::Identity(
            "field result inventory differs from its canonical closure",
        ));
    }
    let manifest = RecoveredFieldArtifactManifest::from_canonical_bytes(&imported.payload())
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if manifest.identity.digest() != output.logical_identity
        || manifest.source != request.source
        || manifest.topology_digest != request.topology.discretization_digest()
        || manifest.recovery_request != request.identity()
    {
        return Err(AnalysisBridgeError::Identity(
            "recovered fields differ from their admitted source request",
        ));
    }
    Ok(ImportedRecoveredFields {
        manifest,
        catalog_objects: imported,
        output: output.clone(),
    })
}

pub fn decode_recovered_field_artifact(
    objects: &PreparedAnalysisObjects,
) -> AnalysisBridgeResult<RecoveredFieldArtifact> {
    if objects.manifest.kind != AnalysisNumericalObjectKind::Field {
        return Err(AnalysisBridgeError::Invalid(
            "recovered-field decoder requires a field numerical object".into(),
        ));
    }
    let manifest = RecoveredFieldArtifactManifest::from_canonical_bytes(
        &objects.partition_payload(objects.manifest.logical_identity)?,
    )
    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if manifest.identity.digest() != objects.manifest.logical_identity {
        return Err(AnalysisBridgeError::Identity(
            "recovered-field manifest differs from its numerical object identity",
        ));
    }
    let fields = manifest
        .fields
        .iter()
        .map(|entry| {
            let partitions = entry
                .partitions
                .iter()
                .map(|descriptor| {
                    RecoveredFieldPartition::from_canonical_bytes(
                        &objects.partition_payload(descriptor.partition.digest())?,
                    )
                    .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
                })
                .collect::<AnalysisBridgeResult<Vec<_>>>()?;
            reconstruct_recovered_field(
                entry.field,
                entry.definition.clone(),
                entry.sample,
                partitions,
                entry.definition.stored_entities(manifest.entity_counts),
            )
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    RecoveredFieldArtifact::from_manifest_and_fields(manifest, fields)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
}
