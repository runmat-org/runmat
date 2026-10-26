use super::*;
use crate::cdt::insertion::validate_canonical_conforming_volume_topology;
use crate::cdt::DelaunayInsertionOptions;
use runmat_meshing_core::NeverCancelled;

fn constructed() -> CanonicalVolumeTopology {
    let nodes = [
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0., 0., -1.],
    ]
    .into_iter()
    .enumerate()
    .map(|(i, point)| {
        DelaunayVolumeNode::explicit(StableDigest::from_bytes([i as u8 + 1; 32]), point)
    })
    .collect();
    build_canonical_volume_topology(
        nodes,
        vec![([0, 1, 2, 3], None), ([0, 1, 2, 4], None)],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Construction-only immutable topology authority matches independent complete
//   admission. Consuming the carrier discards authority; a raw handoff with broken neighbors
//   rejects when authority is restored. This bounded component is not proof of local-only
//   construction, arbitrary connected-mesh scaling, native peak memory or complete study
//   acceptance.
#[test]
fn construction_authority_preserves_independent_admission_and_rejects_edited_handoff() {
    let built = constructed();
    validate_canonical_volume_topology(&built, DelaunayTopologyOptions::default(), &NeverCancelled)
        .unwrap();
    let mut raw = built.into_inner();
    raw.tetrahedra[0].neighbors = [None; 4];
    assert!(CanonicalVolumeTopology::validate(
        raw,
        DelaunayTopologyOptions::default(),
        &NeverCancelled
    )
    .is_err());
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: An immutable construction carrier still checks the current protected-face inventory,
//   lower topology caps and immediate cancellation. Present faces pass; absent faces, tighter caps
//   and an already cancelled small candidate reject. This component does not establish arbitrary
//   feature recovery, native deadline latency or full physics qualification.
#[test]
fn construction_authority_does_not_waive_protected_faces_or_current_limits() {
    let built = constructed();
    let options = DelaunayInsertionOptions::default();
    let present = [[
        built.nodes[0].identity,
        built.nodes[1].identity,
        built.nodes[3].identity,
    ]];
    validate_canonical_conforming_volume_topology(&built, &present, options, &NeverCancelled)
        .unwrap();
    let absent = [[
        built.nodes[2].identity,
        built.nodes[3].identity,
        built.nodes[4].identity,
    ]];
    assert!(validate_canonical_conforming_volume_topology(
        &built,
        &absent,
        options,
        &NeverCancelled
    )
    .is_err());
    let limited = DelaunayInsertionOptions {
        topology: DelaunayTopologyOptions {
            maximum_tetrahedra: 1,
            ..options.topology
        },
        ..options
    };
    assert!(validate_canonical_conforming_volume_topology(
        &built,
        &present,
        limited,
        &NeverCancelled
    )
    .is_err());
    struct Cancelled;
    impl MeshingCancellationSignal for Cancelled {
        fn is_cancelled(&self) -> bool {
            true
        }
    }
    assert!(
        validate_canonical_conforming_volume_topology(&built, &present, options, &Cancelled)
            .is_err()
    );
}

// witness: FEA-SHARED-RESOURCES-001/bounded_local_search @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: Contiguous sorted face incidence preserves complete canonical node, cell, neighbor
//   and boundary/region/star inventory across sixteen input permutations. Separated duplicate and
//   same-side overlap inputs reject. This component establishes deterministic topology equivalence,
//   not measured production-scale performance or accepted realistic native studies.
#[test]
fn compact_face_inventory_preserves_canonical_topology_under_input_permutations() {
    let baseline = constructed().into_inner();
    let cells = [[0, 1, 2, 3], [0, 1, 2, 4]];
    for a in 0..4 {
        for b in 0..4 {
            let mut first = cells[0];
            let mut second = cells[1];
            first.rotate_left(a);
            second.rotate_left(b);
            second.swap(0, 1);
            let actual = build_delaunay_volume_topology(
                baseline.nodes.clone(),
                vec![second, first],
                DelaunayTopologyOptions::default(),
                &NeverCancelled,
            )
            .unwrap();
            assert_eq!(actual, baseline);
        }
    }
    // A permuted duplicate must be rejected even if separated in input order.
    assert!(build_delaunay_volume_topology(
        baseline.nodes.clone(),
        vec![cells[0], cells[1], [3, 2, 1, 0]],
        DelaunayTopologyOptions::default(),
        &NeverCancelled
    )
    .is_err());
    let mut overlapping = baseline.nodes;
    overlapping[4].coordinates_m = runmat_types::physical::Quantity::<
        runmat_types::physical::kind::Position,
        _,
    >::new([0., 0., 2.]);
    assert_eq!(
        build_delaunay_volume_topology(
            overlapping,
            cells.to_vec(),
            DelaunayTopologyOptions::default(),
            &NeverCancelled
        )
        .unwrap_err()
        .kind,
        DelaunayTopologyErrorKind::InvalidTetrahedron
    );
}
