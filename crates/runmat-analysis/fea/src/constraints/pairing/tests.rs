use super::*;
use runmat_analysis_core::RegionTarget;
use runmat_execution::Digest;
use runmat_meshing_core::{fixtures::canonical_tetrahedron_solver_mesh, ElementOrder};

fn target(ids: Vec<StableDigest>) -> ResolvedConstraintTarget {
    ResolvedConstraintTarget {
        authored: RegionTarget::boundary("pair"),
        matched_entities: Vec::new(),
        matched_node_identities: ids,
        selection_digest: Digest::sha256("pair"),
    }
}

#[test]
fn streaming_matching_keeps_identity_ties_and_does_not_skip_later_errors() {
    let mesh = canonical_tetrahedron_solver_mesh(ElementOrder::Tet4);
    let [a, b, c, d] = [1, 2, 3, 4].map(|id| StableDigest::from_bytes([id; 32]));
    let epsilon = matching_tolerance(&mesh) / 4.0;
    let mut nodes = BTreeMap::from([
        (a, [-epsilon, 0.0, 0.0]),
        (b, [epsilon, 0.0, 0.0]),
        (c, [0.0; 3]),
        (d, [epsilon, 0.0, 0.0]),
    ]);
    let secondary = target(vec![c, d]);
    for primary in [target(vec![a, b]), target(vec![b, a])] {
        let pairs = pair_matching(
            &mesh,
            &nodes,
            &primary,
            &secondary,
            &PeriodicTransform::Identity,
            "tie",
        )
        .unwrap();
        assert_eq!(pairs[0].primary_weights[0].primary_node_identity, a);
        assert_eq!(pairs[1].primary_weights[0].primary_node_identity, b);
    }
    nodes.remove(&b);
    // A valid nearest candidate must not short-circuit validation of later candidates.
    assert!(pair_matching(
        &mesh,
        &nodes,
        &target(vec![a, b]),
        &secondary,
        &PeriodicTransform::Identity,
        "missing"
    )
    .is_err());
}
