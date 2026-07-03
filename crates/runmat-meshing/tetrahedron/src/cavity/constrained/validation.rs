use std::collections::{BTreeMap, BTreeSet};

use super::{
    topology::{
        boundary_face_map, boundary_face_source_edges, face_edges, sorted_edge, sorted_face,
        sorted_region_ids, sorted_u32_ids,
    },
    ConstrainedCavity, ConstrainedCavityBoundaryFace, ConstrainedCavityRefillError,
    ConstrainedCavityRefillOptions, ConstrainedCavityValidationError,
    ConstrainedCavityValidationReport,
};

pub fn validate_constrained_cavity(
    cavity: &ConstrainedCavity,
) -> Result<ConstrainedCavityValidationReport, ConstrainedCavityValidationError> {
    if cavity.removed_tetrahedron_ids.is_empty() {
        return Err(ConstrainedCavityValidationError::EmptyRemovedTetrahedronSet);
    }
    if !cavity.target_volume_m3.is_finite() || cavity.target_volume_m3 <= 0.0 {
        return Err(ConstrainedCavityValidationError::InvalidTargetVolume {
            target_volume_m3: cavity.target_volume_m3,
        });
    }
    if cavity.boundary_faces.len() < 4 {
        return Err(ConstrainedCavityValidationError::TooFewBoundaryFaces {
            boundary_face_count: cavity.boundary_faces.len(),
        });
    }

    let mut boundary_faces = BTreeSet::<[u32; 3]>::new();
    let mut boundary_edges = BTreeMap::<[u32; 2], usize>::new();
    let mut boundary_nodes = BTreeSet::<u32>::new();
    for (face_index, face) in cavity.boundary_faces.iter().enumerate() {
        if face.node_ids[0] == face.node_ids[1]
            || face.node_ids[0] == face.node_ids[2]
            || face.node_ids[1] == face.node_ids[2]
        {
            return Err(ConstrainedCavityValidationError::DegenerateBoundaryFace {
                face_index,
                node_ids: face.node_ids,
            });
        }
        let sorted_face = sorted_face(face.node_ids);
        if !boundary_faces.insert(sorted_face) {
            return Err(ConstrainedCavityValidationError::DuplicateBoundaryFace {
                node_ids: sorted_face,
            });
        }
        for node_id in face.node_ids {
            boundary_nodes.insert(node_id);
        }
        for edge in face_edges(face.node_ids) {
            *boundary_edges.entry(sorted_edge(edge)).or_default() += 1;
        }
    }

    for (edge, face_count) in &boundary_edges {
        if *face_count != 2 {
            return Err(ConstrainedCavityValidationError::NonManifoldBoundaryEdge {
                node_ids: *edge,
                face_count: *face_count,
            });
        }
    }

    for node_id in &cavity.protected_node_ids {
        if !boundary_nodes.contains(node_id) {
            return Err(
                ConstrainedCavityValidationError::ProtectedNodeOutsideBoundary {
                    node_id: *node_id,
                },
            );
        }
    }

    Ok(ConstrainedCavityValidationReport {
        boundary_face_count: cavity.boundary_faces.len(),
        boundary_edge_count: boundary_edges.len(),
        boundary_node_count: boundary_nodes.len(),
        protected_node_count: cavity.protected_node_ids.len(),
        target_volume_m3: cavity.target_volume_m3,
    })
}

pub fn validate_constrained_cavity_refill_volume(
    target_volume_m3: f64,
    candidate_volume_m3: f64,
    relative_tolerance: f64,
) -> Result<(), ConstrainedCavityValidationError> {
    if !target_volume_m3.is_finite() || target_volume_m3 <= 0.0 {
        return Err(ConstrainedCavityValidationError::InvalidTargetVolume { target_volume_m3 });
    }
    let tolerance_m3 = target_volume_m3.max(1.0e-18) * relative_tolerance.max(0.0);
    if !candidate_volume_m3.is_finite()
        || candidate_volume_m3 <= 0.0
        || (candidate_volume_m3 - target_volume_m3).abs() > tolerance_m3
    {
        return Err(ConstrainedCavityValidationError::InvalidRefillVolume {
            target_volume_m3,
            candidate_volume_m3,
            tolerance_m3,
        });
    }
    Ok(())
}

pub fn validate_constrained_cavity_boundary_preserved(
    cavity: &ConstrainedCavity,
    candidate_boundary_faces: &[ConstrainedCavityBoundaryFace],
) -> Result<(), ConstrainedCavityValidationError> {
    if cavity.boundary_faces.len() != candidate_boundary_faces.len() {
        return Err(
            ConstrainedCavityValidationError::BoundaryFaceCountMismatch {
                expected_count: cavity.boundary_faces.len(),
                candidate_count: candidate_boundary_faces.len(),
            },
        );
    }

    let expected_faces = boundary_face_map(&cavity.boundary_faces)?;
    let candidate_faces = boundary_face_map(candidate_boundary_faces)?;

    for expected_face in expected_faces.keys() {
        if !candidate_faces.contains_key(expected_face) {
            return Err(ConstrainedCavityValidationError::MissingBoundaryFace {
                node_ids: *expected_face,
            });
        }
    }
    for candidate_face in candidate_faces.keys() {
        if !expected_faces.contains_key(candidate_face) {
            return Err(ConstrainedCavityValidationError::UnexpectedBoundaryFace {
                node_ids: *candidate_face,
            });
        }
    }

    for (face_key, expected) in &expected_faces {
        let candidate = candidate_faces
            .get(face_key)
            .expect("candidate face should exist after key comparison");
        let expected_outside_tetrahedron_ids = sorted_u32_ids(&expected.outside_tetrahedron_ids);
        let candidate_outside_tetrahedron_ids = sorted_u32_ids(&candidate.outside_tetrahedron_ids);
        if expected_outside_tetrahedron_ids != candidate_outside_tetrahedron_ids {
            return Err(
                ConstrainedCavityValidationError::BoundaryOutsideTetrahedronMismatch {
                    node_ids: *face_key,
                    expected_outside_tetrahedron_ids,
                    candidate_outside_tetrahedron_ids,
                },
            );
        }
        if expected.source_face_id != candidate.source_face_id {
            return Err(
                ConstrainedCavityValidationError::BoundarySourceFaceMismatch {
                    node_ids: *face_key,
                    expected_source_face_id: expected.source_face_id,
                    candidate_source_face_id: candidate.source_face_id,
                },
            );
        }
        let expected_edges = boundary_face_source_edges(expected)?;
        let candidate_edges = boundary_face_source_edges(candidate)?;
        for (edge_key, expected_source_edge_id) in expected_edges {
            let candidate_source_edge_id = candidate_edges.get(&edge_key).copied().flatten();
            if expected_source_edge_id != candidate_source_edge_id {
                return Err(
                    ConstrainedCavityValidationError::BoundarySourceEdgeMismatch {
                        node_ids: edge_key,
                        expected_source_edge_id,
                        candidate_source_edge_id,
                    },
                );
            }
        }
        let expected_regions = sorted_region_ids(&expected.region_ids);
        let candidate_regions = sorted_region_ids(&candidate.region_ids);
        if expected_regions != candidate_regions {
            return Err(ConstrainedCavityValidationError::BoundaryRegionMismatch {
                node_ids: *face_key,
                expected_region_ids: expected_regions,
                candidate_region_ids: candidate_regions,
            });
        }
    }

    Ok(())
}

pub(super) fn validate_refill_options(
    options: ConstrainedCavityRefillOptions,
) -> Result<(), ConstrainedCavityRefillError> {
    if !options.min_volume_m3.is_finite()
        || options.min_volume_m3 <= 0.0
        || !options.max_aspect_ratio.is_finite()
        || options.max_aspect_ratio <= 0.0
        || !options.min_scaled_jacobian.is_finite()
        || options.min_scaled_jacobian < 0.0
        || !options.volume_relative_tolerance.is_finite()
        || options.volume_relative_tolerance < 0.0
        || !options.min_protected_node_distance_m.is_finite()
        || options.min_protected_node_distance_m < 0.0
    {
        return Err(ConstrainedCavityRefillError::InvalidOptions);
    }
    Ok(())
}
