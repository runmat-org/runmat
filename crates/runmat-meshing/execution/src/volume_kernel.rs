use runmat_execution::Digest;
use runmat_meshing_core::{
    MeshingChunkMediaType, MeshingChunkStream, MeshingFailure, MeshingFailureCategory,
    MeshingPartitionKind, MeshingStageKind, MeshingStageResultKind, StableDigest,
};
use runmat_meshing_surface::{
    decode_published_exact_surface_mesh, ExactSurfaceJoinOptions,
    EXACT_SURFACE_MESH_SCHEMA_VERSION, EXACT_SURFACE_PASS_RESULT_SCHEMA_VERSION,
};
use runmat_meshing_tetrahedron::cdt::{
    construct_delaunay_volume_mesh, encode_delaunay_volume_mesh,
    DELAUNAY_VOLUME_MESH_SCHEMA_VERSION,
};

mod error;
mod options;
#[cfg(test)]
mod tests;

use error::{invalid_input, map_codec_error, map_volume_error, volume_failure};
pub(crate) use options::volume_options;

use crate::{
    MeshingStageCheckpoint, MeshingStageInvocation, MeshingStageKernel,
    PreparedExactGeometryObjects, PreparedMeshingInput, PreparedMeshingResultPublication,
    ValidatedMeshingStageOutput,
};

#[derive(Clone, Copy, Debug, Default)]
pub struct ExactVolumeKernel;

impl MeshingStageKernel for ExactVolumeKernel {
    fn execute(
        &self,
        invocation: MeshingStageInvocation<'_, '_>,
    ) -> Result<ValidatedMeshingStageOutput, Box<MeshingFailure>> {
        if invocation.host.workload.stage != MeshingStageKind::Tetrahedralization
            || invocation.host.workload.partition.kind != MeshingPartitionKind::WholeStage
        {
            return Err(volume_failure(
                MeshingFailureCategory::InternalInvariantViolation,
                "submit general CDT construction as one whole connected volume stage",
                "tetrahedralization stage shape",
            ));
        }
        let (geometry, surface_publication) = inputs(invocation.inputs)?;
        invocation
            .control
            .checkpoint(MeshingStageCheckpoint::default())?;
        let surface_bytes = surface_record(geometry, surface_publication)?;
        let surface_options = ExactSurfaceJoinOptions {
            coordinate_tolerance_m: invocation.host.resolved_request.tolerance.absolute_floor_m,
            maximum_nodes: invocation.host.resolved_request.resources.maximum_nodes,
            maximum_triangles: invocation.host.resolved_request.resources.maximum_elements,
            maximum_boundary_segments: invocation.host.resolved_request.resources.maximum_elements,
        };
        let surface = decode_published_exact_surface_mesh(
            &surface_bytes,
            &geometry.topology,
            surface_options,
        )
        .map_err(|error| invalid_input(&error.to_string()))?;
        let options = volume_options(&invocation.host.resolved_request);
        let cancellation = invocation.control.geometry_evaluation_control();
        let mesh = construct_delaunay_volume_mesh(
            &geometry.topology,
            &surface,
            &invocation.host.resolved_request.metric,
            options,
            &cancellation,
        )
        .map_err(map_volume_error)?;
        let encoded = encode_delaunay_volume_mesh(
            &mesh,
            &geometry.topology,
            &surface,
            &invocation.host.resolved_request.metric,
            options,
            &cancellation,
        )
        .map_err(map_codec_error)?;

        let node_count = mesh.topology.nodes.len() as u64;
        let element_count = mesh.topology.tetrahedra.len() as u64;
        let mut entity_counts = std::collections::BTreeMap::new();
        entity_counts.insert("volume_nodes".into(), node_count);
        entity_counts.insert("volume_tetrahedra".into(), element_count);
        entity_counts.insert(
            "volume_boundary_facets".into(),
            mesh.provenance.facets.len() as u64,
        );
        entity_counts.insert("volume_mutations".into(), mesh.mutations.len() as u64);
        let checkpoint = MeshingStageCheckpoint {
            completed_work: element_count,
            estimated_work: element_count,
            node_count,
            element_count,
            peak_memory_bytes: (surface_bytes.len() as u64).saturating_add(encoded.len() as u64),
            iterations: mesh.mutations.len() as u64,
            entity_counts,
            ..MeshingStageCheckpoint::default()
        };
        invocation.control.checkpoint(checkpoint.clone())?;

        Ok(ValidatedMeshingStageOutput {
            invariant_summary_digest: validation_digest(&encoded),
            streams: vec![MeshingChunkStream {
                media_type: MeshingChunkMediaType::VolumeTopology,
                schema_version: DELAUNAY_VOLUME_MESH_SCHEMA_VERSION,
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
    ),
    Box<MeshingFailure>,
> {
    match inputs {
        [PreparedMeshingInput::ExactGeometry(geometry), PreparedMeshingInput::StageArtifact(surface)]
        | [PreparedMeshingInput::StageArtifact(surface), PreparedMeshingInput::ExactGeometry(geometry)] => {
            Ok((geometry.geometry_objects(), surface))
        }
        _ => Err(invalid_input(
            "tetrahedralization requires exactly one exact geometry and one final surface artifact",
        )),
    }
}

pub(crate) fn surface_record(
    geometry: &PreparedExactGeometryObjects,
    publication: &PreparedMeshingResultPublication,
) -> Result<Vec<u8>, Box<MeshingFailure>> {
    let stage = publication.stage_objects();
    let streams = stage
        .decoded_streams()
        .map_err(|error| invalid_input(&error.to_string()))?;
    let geometry_digest = StableDigest::from_bytes(*geometry.root.digest.bytes());
    if stage.result_identity.stage != MeshingStageKind::SurfaceMesh
        || stage.result_identity.result_kind != MeshingStageResultKind::DeterministicJoin
        || stage
            .manifest
            .prerequisite_manifest_digests
            .binary_search(&geometry_digest)
            .is_err()
    {
        return Err(invalid_input(
            "final surface identity or exact-geometry prerequisite is invalid",
        ));
    }
    let [pass, stream] = streams.as_slice() else {
        return Err(invalid_input(
            "final surface publication must contain its pass and exact surface streams",
        ));
    };
    if pass.media_type != MeshingChunkMediaType::SurfacePartitions
        || pass.schema_version != EXACT_SURFACE_PASS_RESULT_SCHEMA_VERSION
        || pass.records.len() != 1
        || stream.media_type != MeshingChunkMediaType::SurfaceMesh
        || stream.schema_version != EXACT_SURFACE_MESH_SCHEMA_VERSION
        || stream.records.len() != 1
    {
        return Err(invalid_input(
            "final surface publication stream media, schema, or record count is invalid",
        ));
    }
    Ok(stream.records[0].clone())
}

fn validation_digest(encoded: &[u8]) -> StableDigest {
    let mut bytes = b"runmat-general-cdt-volume-publication-validation/v1\0".to_vec();
    bytes.extend_from_slice(encoded);
    StableDigest::from_bytes(*Digest::sha256(&bytes).bytes())
}
