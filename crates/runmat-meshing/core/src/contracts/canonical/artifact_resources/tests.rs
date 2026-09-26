use super::*;
use crate::{fixtures::canonical_tetrahedron_solver_mesh, CanonicalMeshingContract, ElementOrder};
use runmat_canonical_codec::canonical_payload_materialization_resident_bytes_estimate;

#[test]
fn validation_estimate_covers_canonical_materialization_for_both_orders() {
    for order in [ElementOrder::Tet4, ElementOrder::Tet10] {
        let mesh = canonical_tetrahedron_solver_mesh(order);
        let original = mesh.clone();
        let estimate = mesh.validation_resident_bytes_estimate().unwrap();
        let encoded = mesh.canonical_encode().unwrap();
        assert!(
            estimate
                >= canonical_payload_materialization_resident_bytes_estimate(&encoded).unwrap()
        );
        assert_eq!(mesh, original);
        assert_eq!(estimate, mesh.validation_resident_bytes_estimate().unwrap());
    }
}

#[test]
fn profiling_does_not_validate_or_trust_claimed_topology_sizes() {
    let mut mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet10);
    let base = mesh.validation_resident_bytes_estimate().unwrap();
    mesh.canonical_digest = StableDigest::ZERO;
    assert_eq!(base, mesh.validation_resident_bytes_estimate().unwrap());
    assert!(mesh.validate_canonical().is_err());
    // Field inventories allocate before their completeness check; count actual supplied entries.
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
        assert!(index::<StableDigest>(usize::MAX).is_err());
        assert!(slots::<StableDigest>(usize::MAX, 2).is_err());
    }
    mesh.topology.nodes[0].coordinates_m[0] = f64::NAN;
    assert!(mesh.validation_resident_bytes_estimate().is_err());
}
