//! Prepare and store the complete immutable stage publication.
use super::*;
use runmat_geometry_core::{GeometryAllocationScope, GeometryEvaluationControl};
use runmat_meshing_core::{
    close_stage_payload_with_admission, InvariantEvidence, MeshingManifestMetadata,
};
mod storage;

pub(super) fn publish_stage<S: CacheExport>(
    host: &MeshingHostWorkload,
    store: &mut S,
    control: &mut MeshingStageControl<'_>,
    invariant_summary_digest: StableDigest,
    payload: runmat_meshing_core::MeshingChunkedPayload,
    limits: ObjectInventoryLimits,
) -> Result<CompletedMeshingStage, MeshingSerialExecutionError> {
    let geometry = control.geometry_evaluation_control();
    let scope = GeometryAllocationScope::new(&geometry);
    let codec = crate::codec_control::CodecControl::new(&scope);
    let mut admit = |bytes| {
        codec
            .consume_allocation_bytes(bytes)
            .map_err(|error| MeshingExecutionError::Invalid(error.to_string()))
    };
    admit(host.workload.partition.owned_buffer_bytes()?)
        .map_err(|error| publication_failure(host, error, &codec))?;
    let partition_identity = MeshingPartitionIdentity {
        schema_version: MESHING_IDENTITY_SCHEMA_VERSION,
        stage_identity_digest: host.workload.stage_identity_digest,
        partition: host.workload.partition.clone(),
    };
    let result_kind = match host.workload.partition.kind {
        MeshingPartitionKind::WholeStage => MeshingStageResultKind::WholeStage,
        MeshingPartitionKind::DeterministicJoin => MeshingStageResultKind::DeterministicJoin,
        MeshingPartitionKind::CanonicalEntityBatch
        | MeshingPartitionKind::DisconnectedComponent => MeshingStageResultKind::Partition,
    };
    admit(storage::slots::<StableDigest>(host.workload.inputs.len())?)
        .map_err(|error| publication_failure(host, error, &codec))?;
    let mut prerequisite_manifest_digests = host
        .workload
        .inputs
        .iter()
        .map(|input| input.digest)
        .collect::<Vec<_>>();
    prerequisite_manifest_digests.sort_unstable();
    admit(
        partition_identity
            .canonical_encoding_profile()?
            .resident_workspace_bytes_estimate,
    )
    .map_err(|error| publication_failure(host, error, &codec))?;
    let producer_identity_digest = partition_identity.canonical_digest()?;
    let closed = close_stage_payload_with_admission(
        MeshingManifestMetadata {
            stage: host.workload.stage,
            result_kind,
            producer_identity_digest,
            invariant_summary_digest,
            disposition: MeshingManifestDisposition::ValidatedDependency,
        },
        prerequisite_manifest_digests,
        payload,
        &codec,
        |bytes| {
            codec.consume_allocation_bytes(bytes).map_err(|error| {
                runmat_meshing_core::MeshingContractError::invalid(
                    "publication admission",
                    error.to_string(),
                )
            })
        },
    )
    .map_err(|error| publication_failure(host, error.into(), &codec))?;
    let objects = crate::objects::prepare_closed_stage_objects(closed, limits, &mut admit)
        .map_err(|error| publication_failure(host, error, &codec))?;
    let publication = crate::publication::prepare_validated_result_publication(
        objects,
        &host.artifact_access,
        &mut admit,
    )
    .map_err(|error| publication_failure(host, error, &codec))?;
    let manifest_digest =
        StableDigest::from_bytes(*publication.stage_objects().root.digest.bytes());
    let workload_result = MeshingWorkloadResult::Validated {
        stage_manifest_digest: manifest_digest,
    };
    workload_result.validate_against(&host.workload)?;
    admit(
        host.workload.partition.owned_buffer_bytes()?
            + storage::slots::<InvariantEvidence>(1)?
            + "canonical_stage_validation".len() as u64,
    )
    .map_err(|error| publication_failure(host, error, &codec))?;
    let stage_evidence = control.stage_evidence(host.workload.partition.clone(), manifest_digest);
    stage_evidence.validate()?;
    storage::admit_evidence(&stage_evidence, &host.artifact_access, &mut admit)
        .map_err(|error| publication_failure(host, error, &codec))?;
    let prepared_evidence = crate::stage_evidence_object::prepare_stage_evidence_observation(
        host,
        &stage_evidence,
        limits,
    )?;
    crate::stage_evidence_object::validate_complete_result_inventory(
        &publication.stage_objects().objects,
        &prepared_evidence.object,
        limits,
    )?;
    for object in &publication.stage_objects().objects {
        store
            .write_verified(object)
            .map_err(MeshingExecutionError::from)?;
    }
    store
        .write_verified(&prepared_evidence.object)
        .map_err(MeshingExecutionError::from)?;
    Ok(CompletedMeshingStage {
        workload_result,
        publication,
        stage_evidence,
        stage_evidence_reference: prepared_evidence.reference,
    })
}

fn publication_failure(
    host: &MeshingHostWorkload,
    error: MeshingExecutionError,
    codec: &crate::codec_control::CodecControl<'_>,
) -> MeshingSerialExecutionError {
    if let Some(original) = codec.take_failure() {
        MeshingSerialExecutionError::Stage(failure(
            host.workload.stage,
            crate::geometry_failure::category(original.kind),
            &original.reason,
            None,
        ))
    } else {
        map_artifact_limit(host, error)
    }
}
