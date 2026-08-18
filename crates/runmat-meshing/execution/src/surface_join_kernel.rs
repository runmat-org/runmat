use runmat_execution::Digest;
use runmat_meshing_core::{
    MeshingChunkMediaType, MeshingChunkStream, MeshingFailure, MeshingFailureCategory,
    MeshingPartitionKind, MeshingStageKind, MeshingStageResultKind, StableDigest,
};
use runmat_meshing_curve::decode_shared_curve_mesh;
use runmat_meshing_surface::{
    decide_exact_surface_pass, decode_exact_face_partition_result,
    encode_decided_exact_surface_pass_result, ExactSurfacePassOutcome,
    EXACT_FACE_PARTITION_RESULT_SCHEMA_VERSION, EXACT_SURFACE_PASS_RESULT_SCHEMA_VERSION,
    MAX_EXACT_FACE_PARTITIONS,
};

use crate::surface_kernel::{
    checked_count, curve_record,
    error::{invalid_input, map_convergence_error, surface_failure},
    exact_surface_join_options,
};
use crate::{
    MeshingStageCheckpoint, MeshingStageInvocation, MeshingStageKernel,
    PreparedExactGeometryObjects, PreparedMeshingInput, PreparedMeshingResultPublication,
    ValidatedMeshingStageOutput,
};

#[derive(Clone, Copy, Debug, Default)]
pub struct ExactSurfaceJoinKernel;

impl MeshingStageKernel for ExactSurfaceJoinKernel {
    fn execute(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        if invocation.host.workload.stage != MeshingStageKind::SurfaceMesh
            || invocation.host.workload.partition.kind != MeshingPartitionKind::DeterministicJoin
        {
            return Err(surface_failure(
                MeshingFailureCategory::InternalInvariantViolation,
                None,
                "submit the surface barrier with its deterministic-join descriptor",
                "surface join stage shape",
            ));
        }
        let (geometry, curve_publication, partition_publications) = inputs(invocation.inputs)?;
        invocation
            .control
            .checkpoint(MeshingStageCheckpoint::default())?;
        let curve_bytes = curve_record(geometry, curve_publication)?;
        let curves = decode_shared_curve_mesh(&curve_bytes, &geometry.topology)
            .map_err(crate::curve_kernel::map_curve_error)?;
        let geometry_digest = StableDigest::from_bytes(*geometry.root.digest.bytes());
        let curve_digest =
            StableDigest::from_bytes(*curve_publication.stage_objects().root.digest.bytes());
        let mut encoded_input_bytes = checked_count(
            std::iter::once(curve_bytes.len()),
            "shared curve byte count overflowed",
        )?;
        let mut partitions = Vec::with_capacity(partition_publications.len());
        for publication in partition_publications {
            let bytes = partition_record(publication, geometry_digest, curve_digest)?;
            let byte_count = checked_count(
                std::iter::once(bytes.len()),
                "surface partition byte count overflowed",
            )?;
            encoded_input_bytes = encoded_input_bytes
                .checked_add(byte_count)
                .ok_or_else(|| invalid_input("surface partition byte count overflowed"))?;
            partitions.push(
                decode_exact_face_partition_result(&bytes, &geometry.topology, &curves).map_err(
                    |error| {
                        surface_failure(
                            MeshingFailureCategory::InvalidGeometry,
                            error.source_face_id.as_deref().cloned(),
                            "regenerate every face partition from the same current shared curve",
                            &error.to_string(),
                        )
                    },
                )?,
            );
            invocation
                .control
                .checkpoint(MeshingStageCheckpoint::default())?;
        }
        let options = exact_surface_join_options(invocation.host);
        let result = decide_exact_surface_pass(&curves, &partitions, &geometry.topology, options)
            .map_err(map_convergence_error)?;
        invocation
            .control
            .checkpoint(MeshingStageCheckpoint::default())?;
        let encoded = encode_decided_exact_surface_pass_result(&result)
            .map_err(|error| invalid_input(&error.to_string()))?;

        let face_count = checked_count(
            std::iter::once(geometry.topology.faces.len()),
            "surface face count overflowed",
        )?;
        let (completed_work, node_count, element_count, split_count) = match &result.outcome {
            ExactSurfacePassOutcome::Converged { surface } => (
                checked_count(
                    std::iter::once(surface.face_ids.len()),
                    "joined surface face count overflowed",
                )?,
                checked_count(
                    std::iter::once(surface.nodes.len()),
                    "joined surface node count overflowed",
                )?,
                checked_count(
                    std::iter::once(surface.triangles.len()),
                    "joined surface triangle count overflowed",
                )?,
                0,
            ),
            ExactSurfacePassOutcome::RequiresCurveSplits { splits } => (
                face_count,
                0,
                0,
                checked_count(
                    std::iter::once(splits.len()),
                    "curve split demand count overflowed",
                )?,
            ),
        };
        let mut entity_counts = std::collections::BTreeMap::new();
        entity_counts.insert("surface_faces".into(), completed_work);
        entity_counts.insert("surface_nodes".into(), node_count);
        entity_counts.insert("surface_triangles".into(), element_count);
        entity_counts.insert("curve_split_demands".into(), split_count);
        let checkpoint = MeshingStageCheckpoint {
            completed_work,
            estimated_work: face_count,
            node_count,
            element_count,
            peak_memory_bytes: encoded_input_bytes.saturating_add(checked_count(
                std::iter::once(encoded.len()),
                "surface pass byte count overflowed",
            )?),
            entity_counts,
            ..MeshingStageCheckpoint::default()
        };
        invocation.control.checkpoint(checkpoint.clone())?;

        Ok(ValidatedMeshingStageOutput {
            invariant_summary_digest: validation_digest(&encoded),
            streams: vec![MeshingChunkStream {
                media_type: MeshingChunkMediaType::SurfacePartitions,
                schema_version: EXACT_SURFACE_PASS_RESULT_SCHEMA_VERSION,
                records: vec![encoded],
            }],
            final_checkpoint: checkpoint,
        })
    }
}

fn inputs(
    inputs: &[PreparedMeshingInput],
) -> Result<
    (
        &PreparedExactGeometryObjects,
        &PreparedMeshingResultPublication,
        Vec<&PreparedMeshingResultPublication>,
    ),
    Box<MeshingFailure>,
> {
    let mut geometry = None;
    let mut curves = None;
    let mut partitions = Vec::new();
    for input in inputs {
        match input {
            PreparedMeshingInput::ExactGeometry(input) if geometry.is_none() => {
                geometry = Some(input.geometry_objects());
            }
            PreparedMeshingInput::StageArtifact(input)
                if input.stage_objects().result_identity.stage == MeshingStageKind::CurveMesh
                    && curves.is_none() =>
            {
                curves = Some(input.as_ref());
            }
            PreparedMeshingInput::StageArtifact(input) => partitions.push(input.as_ref()),
            PreparedMeshingInput::ExactGeometry(_) | PreparedMeshingInput::FacetedGeometry(_) => {
                return Err(invalid_input("surface join input kinds are invalid"));
            }
        }
    }
    if partitions.is_empty() || partitions.len() > MAX_EXACT_FACE_PARTITIONS {
        return Err(invalid_input(
            "surface join requires one through 62 face partition artifacts",
        ));
    }
    Ok((
        geometry.ok_or_else(|| invalid_input("surface join lacks exact geometry"))?,
        curves.ok_or_else(|| invalid_input("surface join lacks a current shared curve"))?,
        partitions,
    ))
}

pub(crate) fn partition_record(
    publication: &PreparedMeshingResultPublication,
    geometry_digest: StableDigest,
    curve_digest: StableDigest,
) -> Result<Vec<u8>, Box<MeshingFailure>> {
    let streams = publication
        .stage_objects()
        .decoded_streams()
        .map_err(|error| invalid_input(&error.to_string()))?;
    let [stream] = streams.as_slice() else {
        return Err(invalid_input(
            "surface partition closure must contain one logical stream",
        ));
    };
    let prerequisites = &publication
        .stage_objects()
        .manifest
        .prerequisite_manifest_digests;
    if publication.stage_objects().result_identity.stage != MeshingStageKind::SurfaceMesh
        || publication.stage_objects().result_identity.result_kind
            != MeshingStageResultKind::Partition
        || prerequisites.binary_search(&geometry_digest).is_err()
        || prerequisites.binary_search(&curve_digest).is_err()
        || stream.media_type != MeshingChunkMediaType::SurfacePartitions
        || stream.schema_version != EXACT_FACE_PARTITION_RESULT_SCHEMA_VERSION
        || stream.records.len() != 1
    {
        return Err(invalid_input(
            "surface partition identity, prerequisites, media, schema, or record count is invalid",
        ));
    }
    Ok(stream.records[0].clone())
}

fn validation_digest(encoded: &[u8]) -> StableDigest {
    let mut bytes = b"runmat-exact-surface-pass-validation/v1\0".to_vec();
    bytes.extend_from_slice(encoded);
    StableDigest::from_bytes(*Digest::sha256(&bytes).bytes())
}
