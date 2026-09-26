use super::*;
use runmat_meshing_core::{
    fixtures::canonical_tetrahedron_solver_mesh, ElementOrder, StableDigest,
};

#[test]
fn estimate_covers_retained_projection_and_validation_for_both_orders() {
    let mut estimates = Vec::new();
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let original = mesh.clone();
        let estimate = AnalysisProjection::derivation_resident_bytes_estimate(&mesh).unwrap();
        let validation = mesh.validation_resident_bytes_estimate().unwrap();
        let projection = AnalysisProjection::derive(&mesh).unwrap();
        let record_slots = projection.nodes.capacity() * size_of::<ProjectedNode>()
            + projection.elements.capacity() * size_of::<ProjectedElement>()
            + projection.boundary_faces.capacity() * size_of::<ProjectedBoundaryFace>()
            + projection.node_neighbors.capacity() * size_of::<Vec<usize>>();
        let references = projection
            .elements
            .iter()
            .map(|element| element.node_indices.capacity())
            .sum::<usize>()
            + projection
                .boundary_faces
                .iter()
                .map(|face| face.node_indices.capacity() + face.adjacent_element_indices.capacity())
                .sum::<usize>()
            + projection
                .node_neighbors
                .iter()
                .map(Vec::capacity)
                .sum::<usize>();
        // Lower-bound payload assertion, not an allocator-exact measurement of the tree indexes.
        assert!(estimate >= validation + (record_slots + references * size_of::<usize>()) as u64);
        assert_eq!(mesh, original);
        assert_eq!(
            estimate,
            AnalysisProjection::derivation_resident_bytes_estimate(&mesh).unwrap()
        );
        estimates.push(estimate);
    }
    assert!(estimates[1] > estimates[0]);
}

#[test]
fn estimate_counts_cloned_metadata_and_malformed_raw_connectivity() {
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let original = AnalysisProjection::derivation_resident_bytes_estimate(&mesh).unwrap();
    mesh.canonical_digest = StableDigest::ZERO;
    assert_eq!(
        original,
        AnalysisProjection::derivation_resident_bytes_estimate(&mesh).unwrap()
    );
    assert!(AnalysisProjection::derive(&mesh).is_err());
    mesh.topology.volume_elements[0]
        .region_id
        .assembly_path
        .push("region_instance".repeat(512));
    let region = AnalysisProjection::derivation_resident_bytes_estimate(&mesh).unwrap();
    assert!(region > original);
    mesh.topology.boundary_faces[0].provenance[0]
        .assembly_path
        .push("boundary_instance".repeat(512));
    let boundary = AnalysisProjection::derivation_resident_bytes_estimate(&mesh).unwrap();
    assert!(boundary > region);
    // No 4/10-node cap is assumed before the validator rejects malformed input.
    mesh.topology.volume_elements[0].node_ids.extend(1000..2000);
    let element = AnalysisProjection::derivation_resident_bytes_estimate(&mesh).unwrap();
    assert!(element > boundary);
    mesh.topology.boundary_faces[0].node_ids.extend(3000..4000);
    mesh.topology.boundary_faces[0]
        .adjacent_volume_element_ids
        .extend(4000..5000);
    assert!(AnalysisProjection::derivation_resident_bytes_estimate(&mesh).unwrap() > element);
}

#[test]
fn resource_arithmetic_is_checked_and_malformed_serialization_fails_closed() {
    let mut total = u64::MAX;
    assert!(add(&mut total, 1).is_err());
    assert!(scale(u64::MAX, 2).is_err());
    assert!(index_count::<usize>(u64::MAX).is_err());
    if usize::BITS == 64 {
        assert!(vector::<ProjectedElement>(usize::MAX).is_err());
        assert!(slots::<ProjectedElement>(usize::MAX, 2).is_err());
    }
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    mesh.topology.nodes[0].coordinates_m[0] = f64::NAN;
    assert!(matches!(
        AnalysisProjection::derivation_resident_bytes_estimate(&mesh),
        Err(AnalysisProjectionError::ResourceEstimate(_))
    ));
}
