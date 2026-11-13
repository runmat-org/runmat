use std::collections::{BTreeMap, BTreeSet};
pub(super) mod nodes;

use super::control::checkpoint;
use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};
pub(super) mod conformity;
pub(super) mod validation;
use conformity::{shell_evidence_with_control, validate_edge_conformity_with_control};

use super::{
    ExactFaceMeshBatch, ExactSurfaceJoinOptions, ExactSurfaceMesh, ExactSurfaceMeshError,
    ExactSurfaceMeshErrorKind, EXACT_SURFACE_MESH_SCHEMA_VERSION, MAX_EXACT_FACE_PARTITIONS,
};
use crate::ExactFaceMeshNode;

pub fn join_exact_face_mesh_batches(
    topology: &ExactBRepTopology,
    batches: Vec<ExactFaceMeshBatch>,
    options: ExactSurfaceJoinOptions,
) -> Result<ExactSurfaceMesh, ExactSurfaceMeshError> {
    join_with_control(topology, batches, options, None)
}

pub(super) fn join_with_control(
    topology: &ExactBRepTopology,
    batches: Vec<ExactFaceMeshBatch>,
    options: ExactSurfaceJoinOptions,
    control: Option<&dyn MeshingCancellationSignal>,
) -> Result<ExactSurfaceMesh, ExactSurfaceMeshError> {
    join_with_admission(topology, batches, options, control, None)
}

pub(super) fn join_with_admission(
    topology: &ExactBRepTopology,
    mut batches: Vec<ExactFaceMeshBatch>,
    options: ExactSurfaceJoinOptions,
    control: Option<&dyn MeshingCancellationSignal>,
    allocations: Option<&dyn runmat_geometry_core::GeometryEvaluationControl>,
) -> Result<ExactSurfaceMesh, ExactSurfaceMeshError> {
    checkpoint(control)?;
    validate_options(options)?;
    if batches.is_empty() || batches.len() > MAX_EXACT_FACE_PARTITIONS {
        return Err(invalid(
            "surface join requires a bounded nonempty partition set",
        ));
    }
    batches.sort_by_key(|batch| batch.partition.partition_index);
    for (index, batch) in batches.iter().enumerate() {
        checkpoint(control)?;
        super::batch::validate_exact_face_mesh_batch_parts_with_cancellation(
            batch.schema_version,
            &batch.partition,
            &batch.faces,
            topology,
            control,
        )?;
        if batch.partition.partition_index != index as u32
            || batch.partition.partition_count != batches.len() as u32
        {
            return Err(invalid(
                "surface join partitions do not form one complete canonical set",
            ));
        }
    }
    let face_count = batches
        .iter()
        .try_fold(0usize, |sum, batch| sum.checked_add(batch.faces.len()))
        .ok_or_else(|| limit("surface face inventory overflow"))?;
    let _face_storage = nodes::reserve::<crate::ExactFaceMesh>(face_count, allocations)?;
    let mut faces = Vec::with_capacity(face_count);
    for batch in batches {
        faces.extend(batch.faces);
    }
    if faces.len() != topology.faces.len()
        || faces
            .iter()
            .zip(&topology.faces)
            .any(|(mesh, face)| mesh.source_face_id != face.id)
    {
        return Err(invalid(
            "surface join does not exactly cover the canonical face inventory",
        ));
    }
    topology
        .validate_solid_shell_boundaries()
        .map_err(|error| invalid(error.to_string()))?;

    let joined_nodes = nodes::merge(topology, &mut faces, options, control, allocations)?;
    let triangle_count = faces
        .iter()
        .try_fold(0usize, |sum, face| sum.checked_add(face.triangles.len()))
        .ok_or_else(|| limit("surface triangle inventory overflow"))?;
    let boundary_count = faces
        .iter()
        .try_fold(0usize, |sum, face| {
            sum.checked_add(face.boundary_segments.len())
        })
        .ok_or_else(|| limit("surface boundary inventory overflow"))?;
    if triangle_count as u64 > options.maximum_triangles
        || boundary_count as u64 > options.maximum_boundary_segments
    {
        return Err(limit(
            "surface join exceeds triangle or boundary hard limits",
        ));
    }
    let mut triangle_storage =
        nodes::reserve::<crate::ExactFaceMeshTriangle>(triangle_count, allocations)?;
    let _id_storage = nodes::reserve::<StableDigest>(triangle_count, allocations)?;
    let mut boundary_storage =
        nodes::reserve::<crate::ExactFaceMeshBoundarySegment>(boundary_count, allocations)?;
    let mut face_storage =
        nodes::reserve::<runmat_geometry_core::PersistentEntityId>(faces.len(), allocations)?;
    let mut triangle_ids = Vec::with_capacity(triangle_count);
    let mut triangles = Vec::with_capacity(triangle_count);
    let mut boundary_segments = Vec::with_capacity(boundary_count);
    let mut maximum_chordal_deviation_m = 0.0_f64;
    let mut maximum_normal_deviation_rad = 0.0_f64;
    let mut face_ids = Vec::with_capacity(faces.len());
    for face in faces {
        checkpoint(control)?;
        maximum_chordal_deviation_m =
            maximum_chordal_deviation_m.max(face.maximum_chordal_deviation_m.canonical_value());
        maximum_normal_deviation_rad =
            maximum_normal_deviation_rad.max(face.maximum_normal_deviation_rad);
        for triangle in face.triangles {
            checkpoint(control)?;
            if triangles.len() as u64 >= options.maximum_triangles {
                return Err(limit("surface join exceeded its triangle hard limit"));
            }
            triangle_ids.push(triangle.triangle_id);
            triangles.push(triangle);
        }
        if boundary_segments.len() as u64 + face.boundary_segments.len() as u64
            > options.maximum_boundary_segments
        {
            return Err(limit(
                "surface join exceeded its boundary-segment hard limit",
            ));
        }
        for segment in face.boundary_segments {
            checkpoint(control)?;
            boundary_segments.push(segment);
        }
        face_ids.push(face.source_face_id);
    }
    runmat_cpu_execution::sort_in_place_with_checkpoint(&mut triangle_ids, Ord::cmp, || {
        checkpoint(control)
    })?;
    for pair in triangle_ids.windows(2) {
        checkpoint(control)?;
        if pair[0] == pair[1] {
            let face = &triangles
                .iter()
                .filter(|triangle| triangle.triangle_id == pair[0])
                .nth(1)
                .expect("duplicate original ID located")
                .source_face_id;
            return Err(
                invalid("surface join contains a duplicate triangle identity").with_face(face),
            );
        }
    }
    validate_edge_conformity_with_control(topology, &boundary_segments, control)?;
    let shells = shell_evidence_with_control(topology, control)?;
    if let Some(storage) = &mut triangle_storage {
        storage.retain_bytes(
            (triangles.capacity() * std::mem::size_of::<crate::ExactFaceMeshTriangle>()) as u64,
        );
    }
    if let Some(storage) = &mut boundary_storage {
        storage.retain_bytes(
            (boundary_segments.capacity()
                * std::mem::size_of::<crate::ExactFaceMeshBoundarySegment>()) as u64,
        );
    }
    if let Some(storage) = &mut face_storage {
        storage.retain_bytes(
            (face_ids.capacity() * std::mem::size_of::<runmat_geometry_core::PersistentEntityId>())
                as u64,
        );
    }
    let result = ExactSurfaceMesh {
        schema_version: EXACT_SURFACE_MESH_SCHEMA_VERSION,
        face_ids,
        nodes: joined_nodes,
        triangles,
        boundary_segments,
        shells,
        maximum_chordal_deviation_m: runmat_types::physical::Length::new(
            maximum_chordal_deviation_m,
        ),
        maximum_normal_deviation_rad,
    };
    Ok(result)
}

pub fn validate_exact_surface_mesh(
    result: &ExactSurfaceMesh,
    topology: &ExactBRepTopology,
    batches: &[ExactFaceMeshBatch],
    options: ExactSurfaceJoinOptions,
) -> Result<(), ExactSurfaceMeshError> {
    validate_with_control(result, topology, batches, options, None)
}

pub(super) fn validate_with_control(
    result: &ExactSurfaceMesh,
    topology: &ExactBRepTopology,
    batches: &[ExactFaceMeshBatch],
    options: ExactSurfaceJoinOptions,
    control: Option<&dyn MeshingCancellationSignal>,
) -> Result<(), ExactSurfaceMeshError> {
    validation::validate(
        result,
        topology,
        batches
            .iter()
            .map(|batch| validation::Batch {
                schema_version: batch.schema_version,
                partition: &batch.partition,
                faces: &batch.faces,
            })
            .collect(),
        options,
        control,
    )
}

pub(super) fn validate_options(
    options: ExactSurfaceJoinOptions,
) -> Result<(), ExactSurfaceMeshError> {
    if !options.coordinate_tolerance_m.canonical_value().is_finite()
        || options.coordinate_tolerance_m.canonical_value() <= 0.0
        || options.maximum_nodes == 0
        || options.maximum_triangles == 0
        || options.maximum_boundary_segments == 0
    {
        return Err(ExactSurfaceMeshError::new(
            ExactSurfaceMeshErrorKind::InvalidOptions,
            "surface join tolerance and hard limits must be finite and positive",
        ));
    }
    Ok(())
}

fn distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| (left - right) * (left - right))
        .sum::<f64>()
        .sqrt()
}

fn invalid(reason: impl Into<String>) -> ExactSurfaceMeshError {
    ExactSurfaceMeshError::new(ExactSurfaceMeshErrorKind::InvalidInput, reason)
}

fn limit(reason: &str) -> ExactSurfaceMeshError {
    ExactSurfaceMeshError::new(ExactSurfaceMeshErrorKind::ResourceLimit, reason)
}
