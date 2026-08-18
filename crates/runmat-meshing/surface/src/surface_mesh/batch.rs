use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_core::{CanonicalEntityRange, MeshingPartitionDescriptor, MeshingPartitionKind};

use crate::{face_mesh::validate_exact_face_mesh_contract, ExactFaceMesh};

use super::{
    ExactFaceMeshBatch, ExactSurfaceMeshError, ExactSurfaceMeshErrorKind,
    EXACT_FACE_MESH_BATCH_SCHEMA_VERSION, MAX_EXACT_FACE_PARTITIONS,
};

// One exact-geometry prerequisite plus this many partition roots fits the shared 64-input join.
pub fn face_partition_descriptors(
    topology: &ExactBRepTopology,
    preferred_faces_per_partition: u32,
) -> Result<Vec<MeshingPartitionDescriptor>, ExactSurfaceMeshError> {
    if preferred_faces_per_partition == 0 || topology.faces.is_empty() {
        return Err(invalid_options(
            "maximum faces and exact face inventory must be nonzero",
        ));
    }
    let minimum_batch_size = topology.faces.len().div_ceil(MAX_EXACT_FACE_PARTITIONS);
    let batch_size = (preferred_faces_per_partition as usize).max(minimum_batch_size);
    let partition_count = topology.faces.len().div_ceil(batch_size);
    Ok(topology
        .faces
        .chunks(batch_size)
        .enumerate()
        .map(|(index, faces)| MeshingPartitionDescriptor {
            kind: MeshingPartitionKind::CanonicalEntityBatch,
            partition_index: index as u32,
            partition_count: partition_count as u32,
            entity_range: Some(CanonicalEntityRange {
                first: faces[0].id.clone(),
                last: faces.last().expect("chunk is nonempty").id.clone(),
                entity_count: faces.len() as u64,
            }),
        })
        .collect())
}

pub fn build_exact_face_mesh_batch(
    topology: &ExactBRepTopology,
    partition: MeshingPartitionDescriptor,
    faces: Vec<ExactFaceMesh>,
) -> Result<ExactFaceMeshBatch, ExactSurfaceMeshError> {
    let batch = ExactFaceMeshBatch {
        schema_version: EXACT_FACE_MESH_BATCH_SCHEMA_VERSION,
        partition,
        faces,
    };
    validate_exact_face_mesh_batch(&batch, topology)?;
    Ok(batch)
}

pub fn validate_exact_face_mesh_batch(
    batch: &ExactFaceMeshBatch,
    topology: &ExactBRepTopology,
) -> Result<(), ExactSurfaceMeshError> {
    validate_partition(&batch.partition)?;
    if batch.schema_version != EXACT_FACE_MESH_BATCH_SCHEMA_VERSION || batch.faces.is_empty() {
        return Err(invalid_input("face batch schema or inventory is invalid"));
    }
    let range = batch
        .partition
        .entity_range
        .as_ref()
        .expect("partition kind validated");
    let expected = topology
        .faces
        .iter()
        .filter(|face| face.id >= range.first && face.id <= range.last)
        .collect::<Vec<_>>();
    if expected.len() as u64 != range.entity_count
        || expected.first().is_none_or(|face| face.id != range.first)
        || expected.last().is_none_or(|face| face.id != range.last)
        || batch.faces.len() != expected.len()
        || batch
            .faces
            .iter()
            .zip(&expected)
            .any(|(mesh, face)| mesh.source_face_id != face.id)
    {
        return Err(invalid_input(
            "face batch does not exactly cover its canonical topology range",
        ));
    }
    for mesh in &batch.faces {
        validate_exact_face_mesh_contract(mesh, topology).map_err(|error| {
            ExactSurfaceMeshError::new(ExactSurfaceMeshErrorKind::InvalidInput, error.reason)
                .with_face(&mesh.source_face_id)
        })?;
    }
    Ok(())
}

fn validate_partition(partition: &MeshingPartitionDescriptor) -> Result<(), ExactSurfaceMeshError> {
    partition
        .validate()
        .map_err(|error| invalid_input(error.to_string()))?;
    if partition.kind != MeshingPartitionKind::CanonicalEntityBatch
        || partition.partition_count as usize > MAX_EXACT_FACE_PARTITIONS
    {
        return Err(invalid_input(
            "face work requires a bounded canonical entity batch",
        ));
    }
    Ok(())
}

fn invalid_options(reason: &str) -> ExactSurfaceMeshError {
    ExactSurfaceMeshError::new(ExactSurfaceMeshErrorKind::InvalidOptions, reason)
}

fn invalid_input(reason: impl Into<String>) -> ExactSurfaceMeshError {
    ExactSurfaceMeshError::new(ExactSurfaceMeshErrorKind::InvalidInput, reason)
}

#[cfg(test)]
mod tests {
    use super::face_partition_descriptors;

    #[test]
    fn face_partitions_coarsen_to_the_shared_join_fan_in() {
        let (_, mut topology, _) = runmat_geometry_fixtures::exact_circle();
        let template = topology.faces[0].clone();
        topology.faces = (0..4_097)
            .map(|index| {
                let mut face = template.clone();
                face.id.source_topology_id = format!("face-{index:05}");
                face
            })
            .collect();

        let partitions = face_partition_descriptors(&topology, 1).unwrap();

        assert_eq!(partitions.len(), 63);
        assert_eq!(
            partitions
                .iter()
                .map(|partition| partition.entity_range.as_ref().unwrap().entity_count)
                .sum::<u64>(),
            4_097
        );
        assert!(partitions.iter().enumerate().all(|(index, partition)| {
            partition.partition_index == index as u32
                && partition.partition_count == partitions.len() as u32
        }));
    }
}
