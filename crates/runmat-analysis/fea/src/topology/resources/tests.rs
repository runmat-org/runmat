use super::*;
use runmat_meshing_core::{
    fixtures::canonical_tetrahedron_solver_mesh, ElementOrder, StableDigest,
};
use std::mem::size_of;

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
        // Retained counted buffers; this remains a payload lower bound, not measured RSS.
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
    assert!(slots::<usize>(usize::MAX, 2).is_err());
    if usize::BITS == 64 {
        assert!(vector::<ProjectedElement>(usize::MAX).is_err());
        assert!(slots::<ProjectedElement>(usize::MAX, 2).is_err());
    }
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    {
        let quantity_change = f64::NAN;
        let quantity_index = 0;
        let quantity_slot = &mut mesh.topology.nodes[0].coordinates_m;
        let mut quantity_coordinates = quantity_slot.canonical_value();
        quantity_coordinates[quantity_index] = quantity_change;
        *quantity_slot = runmat_types::physical::Quantity::<
            runmat_types::physical::kind::Position,
            _,
        >::new(quantity_coordinates);
    };
    assert!(matches!(
        AnalysisProjection::derivation_resident_bytes_estimate(&mesh),
        Err(AnalysisProjectionError::ResourceEstimate(_))
    ));
}

pub(super) mod allocation;

#[test]
fn counted_retained_bound_covers_original_geometry_without_constructing_or_validating_it() {
    for mesh in [
        canonical_tetrahedron_solver_mesh(ElementOrder::Tet4),
        canonical_tetrahedron_solver_mesh(ElementOrder::Tet10),
        runmat_meshing_core::fixtures::slab_mesh(2),
        runmat_meshing_core::fixtures::quadratic_slab_mesh(2),
    ] {
        let (bound, live, peak) =
            measured(|| AnalysisProjection::retained_resident_bytes_upper_bound(&mesh).unwrap());
        assert_eq!((live, peak), (0, 0));
        let (projection, live, _) = measured(|| AnalysisProjection::derive(&mesh).unwrap());
        assert!(live as u64 <= bound);
        assert!(projection.clone_resident_bytes_estimate().unwrap() <= bound);
        let mut foreign = mesh.clone();
        foreign.canonical_digest = StableDigest::ZERO;
        assert_eq!(
            AnalysisProjection::retained_resident_bytes_upper_bound(&foreign).unwrap(),
            bound
        );
        assert!(AnalysisProjection::derive(&foreign).is_err());
    }
}
