use super::*;
use crate::topology::FiniteVolumeProjectionPolicy;

fn source() -> (AnalysisProjection, FiniteVolumeTopology) {
    let projection =
        AnalysisProjection::derive(&runmat_meshing_core::fixtures::slab_mesh(1)).unwrap();
    let geometry = FiniteVolumeTopology::derive(
        &projection,
        FiniteVolumeProjectionPolicy {
            max_cells: Some(100),
            max_face_incidences: Some(400),
            max_boundary_faces: Some(400),
            max_scratch_bytes: Some(10_000_000),
            max_traversal_operations: Some(1_000_000),
        },
    )
    .unwrap();
    (projection, geometry)
}

#[test]
fn internal_faces_remain_distinct_from_exterior_faces_and_volume_cells() {
    let (projection, geometry) = source();
    let layout = FiniteVolumeFieldTopology::derive(&projection, &geometry).unwrap();
    assert!(layout
        .oriented_faces()
        .iter()
        .any(|face| face.neighbor.is_some()));
    assert_eq!(layout.entity_counts().oriented_faces, geometry.faces.len());
    assert_eq!(layout.entity_counts().volume_elements, geometry.cells.len());
    assert!(layout.entity_counts().oriented_faces > projection.boundary_faces.len());
    assert!(layout.identity().require_structural().is_err());
}

#[test]
fn internally_valid_reordered_cold_layout_has_no_original_source_authority() {
    let (projection, geometry) = source();
    let mut layout = FiniteVolumeFieldTopology::derive(&projection, &geometry).unwrap();
    layout.cell_identities.reverse();
    let cold = FiniteVolumeFieldTopology::from_canonical_bytes(&layout.canonical_bytes().unwrap())
        .unwrap();
    cold.validate().unwrap();
    assert!(cold.validate_for(&projection, &geometry).is_err());
}

#[test]
fn self_consistent_owner_reversal_cannot_substitute_original_flux_orientation() {
    let (projection, geometry) = source();
    let mut layout = FiniteVolumeFieldTopology::derive(&projection, &geometry).unwrap();
    let face = layout
        .oriented_faces
        .iter_mut()
        .find(|face| face.neighbor.is_some())
        .unwrap();
    let old_owner = face.owner;
    face.owner = face.neighbor.unwrap();
    face.neighbor = Some(old_owner);
    face.identity = face_identity(face.corner_identities, face.owner, face.neighbor).unwrap();
    layout.validate().unwrap();
    assert!(layout.validate_for(&projection, &geometry).is_err());
}

#[test]
fn counted_layout_validation_preserves_original_bytes_and_rejects_bad_references() {
    let (projection, geometry) = source();
    let original = FiniteVolumeFieldTopology::derive(&projection, &geometry).unwrap();
    let bytes = original.canonical_bytes().unwrap();
    let cold = FiniteVolumeFieldTopology::from_canonical_bytes(&bytes).unwrap();
    cold.validate_for(&projection, &geometry).unwrap();
    assert_eq!(cold.canonical_bytes().unwrap(), bytes);
    for case in 0..7 {
        let mut value = original.clone();
        match case {
            0 => value.node_identities[1] = value.node_identities[0],
            1 => value.cell_identities[1] = value.cell_identities[0],
            2 => value.oriented_faces[1].identity = value.oriented_faces[0].identity,
            3 => value.oriented_faces[0].owner = StableDigest::from_bytes([255; 32]),
            4 => value.oriented_faces[0].corner_identities[0] = StableDigest::ZERO,
            5 => value.oriented_faces[0].neighbor = Some(value.oriented_faces[0].owner),
            6 => value.node_identities[0] = StableDigest::ZERO,
            _ => unreachable!(),
        }
        assert!(value.validate().is_err(), "case {case}");
    }
    assert!(FiniteVolumeFieldTopology::validation_workspace_bytes(usize::MAX, 1, 1).is_err());
}
