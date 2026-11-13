use super::*;
use crate::{fixtures::canonical_tetrahedron_solver_mesh, CanonicalMeshingContract, ElementOrder};

#[test]
fn validation_estimate_covers_the_direct_encoding_for_both_orders() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let original = mesh.clone();
        let estimate = mesh.validation_resident_bytes_estimate().unwrap();
        let encoded = mesh.canonical_encode().unwrap();
        // Identity validation runs the direct encoder; no serde value tree is materialized, so
        // the estimate covers the encoded bytes, not a value-tree materialization.
        assert!(
            estimate >= encoded.len() as u64,
            "{estimate} vs {}",
            encoded.len()
        );
        assert_eq!(mesh, original);
        assert_eq!(estimate, mesh.validation_resident_bytes_estimate().unwrap());
    }
}

#[test]
fn profiling_does_not_validate_or_trust_claimed_topology_sizes() {
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let base = mesh.validation_resident_bytes_estimate().unwrap();
    let scratch =
        validation_scratch_bytes(&mesh.topology, mesh.resolved_request.element_order).unwrap();
    mesh.canonical_digest = StableDigest::ZERO;
    mesh.root_stage_manifest_digest = StableDigest::ZERO;
    // Digest byte values confer no validation authority: a zeroed seal neither fails the
    // resource estimate nor changes it, so admission is decided before validation rejects it.
    let invalid_profile = mesh.validation_resident_bytes_estimate().unwrap();
    assert_eq!(invalid_profile, base);
    assert!(invalid_profile >= scratch);
    assert_eq!(
        scratch,
        validation_scratch_bytes(&mesh.topology, mesh.resolved_request.element_order).unwrap()
    );
    assert!(mesh.validate_canonical().is_err());
    // Codec materialization counts actual supplied entries even though validation borrows them.
    mesh.topology.field_topologies[0]
        .ordered_entity_ids
        .extend(10_000..20_000);
    let fields = mesh.validation_resident_bytes_estimate().unwrap();
    assert!(fields > base);
    // No assumption that an invalid element has already been bounded to 4/10 nodes.
    mesh.topology.volume_elements[0]
        .node_ids
        .extend(30_000..40_000);
    assert!(mesh.validation_resident_bytes_estimate().unwrap() > fields);
}

#[test]
fn validation_estimate_counts_nested_metadata_and_checks_overflow() {
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let base = mesh.validation_resident_bytes_estimate().unwrap();
    mesh.topology.nodes[0].provenance[0]
        .assembly_path
        .push("part".repeat(4096));
    assert!(mesh.validation_resident_bytes_estimate().unwrap() > base);
    let mut total = u64::MAX;
    assert!(add(&mut total, 1).is_err());
    if usize::BITS == 64 {
        assert!(slots::<StableDigest>(usize::MAX).is_err());
        assert!(volume_edge_occurrences(usize::MAX).is_err());
    }
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
    assert!(mesh.validation_resident_bytes_estimate().is_err());
}

#[test]
fn malformed_lists_do_not_allocate_validator_copies() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let mut invalid = mesh.topology.clone();
        // These lists are rejected in place; encoding still profiles their supplied bytes.
        invalid.volume_elements[0].node_ids.extend(10_000..20_000);
        invalid.field_topologies[0]
            .ordered_entity_ids
            .extend(10_000..20_000);
        assert_eq!(
            validation_scratch_bytes(&invalid, order).unwrap(),
            validation_scratch_bytes(&mesh.topology, order).unwrap()
        );
        assert!(crate::validate_solver_mesh_topology(&invalid, &mesh.resolved_request).is_err());
    }
}

#[test]
fn owned_encoding_estimate_covers_wire_and_nested_metadata_without_changing_original() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mut mesh = canonical_tetrahedron_solver_mesh(order);
        let original = mesh.clone();
        let estimate = mesh.canonical_encoding_resident_bytes_estimate().unwrap();
        assert!(estimate >= mesh.canonical_encode().unwrap().len() as u64);
        assert!(estimate >= validation_scratch_bytes(&mesh.topology, order).unwrap());
        assert_eq!(mesh, original);
        mesh.topology.nodes[0].provenance[0]
            .assembly_path
            .push("part".repeat(4096));
        assert!(mesh.canonical_encoding_resident_bytes_estimate().unwrap() > estimate);
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
        assert!(mesh.canonical_encoding_resident_bytes_estimate().is_err());
    }
}
