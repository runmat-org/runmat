use std::collections::BTreeSet;

use runmat_analysis_fea::execution::{
    AnalysisNumericalObjectKind, AnalysisWorkKind, FieldRecoveryRequest, SolverResultIdentity,
    StructuralLinearWorkload, StructuralLinearWorkloadResult,
};
use runmat_analysis_fea::results::field::{
    reconstruct_recovered_field, RecoveredFieldArtifact, RecoveredFieldArtifactManifest,
    RecoveredFieldPartition, StructuralRecoveredFieldTopology,
};
use runmat_execution::value::ValueRefKind;
use runmat_execution::ProgramRevision;
use runmat_execution_artifact::cache::CacheImport;
use runmat_execution_artifact::object::ObjectInventoryLimits;
use runmat_execution_artifact::{ObjectNamespace, ProgramExecutionResponse};
use runmat_meshing_core::{CanonicalMeshingContract, SolverMeshArtifact};

use crate::{
    analysis_object_references, import_analysis_output_set, import_analysis_partition,
    prepare_analysis_objects, prepare_analysis_program, AnalysisBridgeError, AnalysisBridgeResult,
    AnalysisCapabilityRequirement, AnalysisHostWorkload, AnalysisOutputRoot,
    AnalysisPartitionImport, AnalysisWorkloadInput, AnalysisWorkloadRequest,
    ImportedAnalysisPartition, PreparedAnalysisObjects, PreparedAnalysisProgram, ANALYSIS_HOST_ABI,
};

pub struct StructuralFieldRecoveryPreparation {
    pub source_host: AnalysisHostWorkload,
    pub source_invocation: PreparedAnalysisObjects,
    pub source_result: PreparedAnalysisObjects,
    pub mesh: Option<PreparedAnalysisObjects>,
    pub program_revision: ProgramRevision,
    pub chunk_bytes: usize,
    pub inventory_limits: ObjectInventoryLimits,
}

pub struct PreparedStructuralFieldRecovery {
    pub request: FieldRecoveryRequest,
    pub host: AnalysisHostWorkload,
    pub program: PreparedAnalysisProgram,
    pub inputs: Vec<PreparedAnalysisObjects>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedRecoveredFields {
    pub manifest: RecoveredFieldArtifactManifest,
    pub catalog_objects: ImportedAnalysisPartition,
    pub output: AnalysisOutputRoot,
}

pub fn prepare_structural_field_recovery(
    preparation: StructuralFieldRecoveryPreparation,
) -> AnalysisBridgeResult<PreparedStructuralFieldRecovery> {
    preparation.source_host.validate()?;
    if preparation.source_host.workload.kind != AnalysisWorkKind::Solve
        || preparation.source_result.manifest.kind != AnalysisNumericalObjectKind::SolverResult
        || preparation.source_invocation.manifest.kind
            != AnalysisNumericalObjectKind::SolverInvocation
        || preparation
            .mesh
            .as_ref()
            .is_some_and(|mesh| mesh.manifest.kind != AnalysisNumericalObjectKind::Mesh)
        || preparation.chunk_bytes == 0
    {
        return Err(AnalysisBridgeError::Invalid(
            "field recovery requires one completed solve, its invocation, optional canonical mesh, and bounded chunks"
                .into(),
        ));
    }
    let source_invocation = preparation
        .source_host
        .workload
        .inputs
        .iter()
        .find(|input| input.kind == AnalysisNumericalObjectKind::SolverInvocation)
        .ok_or_else(|| {
            AnalysisBridgeError::Invalid("source solve has no invocation input".into())
        })?;
    if source_invocation.logical_identity != preparation.source_invocation.manifest.logical_identity
        || source_invocation.payload_digest != preparation.source_invocation.manifest.payload_digest
        || source_invocation.manifest_digest != preparation.source_invocation.root.digest
    {
        return Err(AnalysisBridgeError::Identity(
            "field recovery invocation differs from the source solve invocation",
        ));
    }
    let source_mesh = preparation
        .source_host
        .workload
        .inputs
        .iter()
        .find(|input| input.kind == AnalysisNumericalObjectKind::Mesh);
    match (source_mesh, preparation.mesh.as_ref()) {
        (Some(source), Some(mesh))
            if source.logical_identity == mesh.manifest.logical_identity
                && source.payload_digest == mesh.manifest.payload_digest
                && source.manifest_digest == mesh.root.digest => {}
        (None, None) => {}
        _ => {
            return Err(AnalysisBridgeError::Identity(
                "field recovery mesh presence or identity differs from the source solve",
            ))
        }
    }
    let source_result = SolverResultIdentity::new(
        preparation.source_host.workload.identity,
        preparation.source_result.manifest.payload_digest,
    )?;
    if preparation.source_result.manifest.logical_identity != source_result.digest() {
        return Err(AnalysisBridgeError::Identity(
            "field recovery solver result differs from its source workload",
        ));
    }
    let workload =
        StructuralLinearWorkload::from_canonical_bytes(&preparation.source_invocation.payload())?;
    let result =
        StructuralLinearWorkloadResult::from_canonical_bytes(&preparation.source_result.payload())?;
    let mesh_artifact = preparation
        .mesh
        .as_ref()
        .map(|mesh| {
            SolverMeshArtifact::canonical_decode(&mesh.payload())
                .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
        })
        .transpose()?;
    let topology =
        StructuralRecoveredFieldTopology::derive(&workload.model, mesh_artifact.as_ref())
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))?;
    if result.constraint_result.topology != topology.identity {
        return Err(AnalysisBridgeError::Identity(
            "field recovery topology differs from the completed solve",
        ));
    }
    let request = FieldRecoveryRequest::new(
        preparation.source_host.workload.step,
        source_result,
        topology.identity,
    )?;
    let request_bytes = request.canonical_bytes()?;
    let request_objects = prepare_analysis_objects(
        AnalysisNumericalObjectKind::FieldRecoveryRequest,
        request.identity().digest(),
        &request_bytes,
        preparation.chunk_bytes,
        ObjectNamespace::InputValue,
        preparation.inventory_limits,
    )?;
    let inputs = vec![
        (request_objects, ValueRefKind::DriverObject),
        (preparation.source_invocation, ValueRefKind::DriverObject),
        (preparation.source_result, ValueRefKind::ResultObject),
    ];
    let mut inputs = inputs;
    if let Some(mesh) = preparation.mesh {
        inputs.push((mesh, ValueRefKind::DriverObject));
    }
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
    let workload = AnalysisWorkloadRequest::new(
        preparation.source_host.workload.study,
        preparation.source_host.workload.step,
        preparation.source_host.workload.execution_plan,
        AnalysisWorkKind::RecoverField,
        workload_inputs,
        vec![AnalysisNumericalObjectKind::Field],
        BTreeSet::from([AnalysisCapabilityRequirement::Host {
            abi: ANALYSIS_HOST_ABI.into(),
        }]),
    )?;
    let host = AnalysisHostWorkload::new(workload, preparation.source_host.artifact_access)?;
    let program = prepare_analysis_program(&host, preparation.program_revision, &inputs)?;
    Ok(PreparedStructuralFieldRecovery {
        request,
        host,
        program,
        inputs,
    })
}

pub fn import_structural_recovered_fields(
    source: &impl CacheImport,
    response: ProgramExecutionResponse,
    prepared: &PreparedStructuralFieldRecovery,
    limits: ObjectInventoryLimits,
) -> AnalysisBridgeResult<ImportedRecoveredFields> {
    response.validate_against(&prepared.program.request)?;
    let output_set = import_analysis_output_set(source, response, &prepared.host)?;
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
            access: &prepared.host.artifact_access,
            limits,
        },
    )?;
    let mut expected = analysis_object_references(
        &imported.manifest,
        &output.root,
        &prepared.host.artifact_access,
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
        || manifest.source_result != prepared.request.source_result
        || manifest.topology_digest != prepared.request.topology.discretization_digest()
        || manifest.recovery_request != prepared.request.identity()
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
                manifest
                    .entity_counts
                    .entity_count(entry.definition.entity_domain()),
            )
            .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
        })
        .collect::<AnalysisBridgeResult<Vec<_>>>()?;
    RecoveredFieldArtifact::from_manifest_and_fields(manifest, fields)
        .map_err(|error| AnalysisBridgeError::Invalid(error.to_string()))
}
