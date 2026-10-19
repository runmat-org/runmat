use super::*;
use crate::cdt::{topology::build_delaunay_volume_topology_with_regions, DelaunayTopologyOptions};
use runmat_meshing_core::NeverCancelled;

fn bipyramid(pole: f64) -> DelaunayVolumeTopology {
    let points = [
        [0., 0., -pole],
        [0., 0., pole],
        [5., 0., 0.],
        [-3., 4., 0.],
        [-3., -4., 0.],
    ];
    build_delaunay_volume_topology_with_regions(
        points
            .into_iter()
            .enumerate()
            .map(|(index, point)| {
                DelaunayVolumeNode::explicit(
                    StableDigest::from_bytes([(index + 1) as u8; 32]),
                    point,
                )
            })
            .collect(),
        vec![
            ([0, 1, 2, 3], None),
            ([0, 1, 3, 4], None),
            ([0, 1, 4, 2], None),
        ],
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

#[test]
fn physical_cospherical_seed_admission_preserves_valid_connectivity() {
    // All five integer points lie exactly on the radius-five sphere. Either
    // diagonal is physically admissible; no symbolic rearrangement is needed.
    let candidate = bipyramid(5.);
    let options = DelaunayInsertionOptions::default();
    validate_constrained_delaunay_volume_topology(&candidate, &[], options, &NeverCancelled)
        .unwrap();
    let admitted = admit_or_repair(candidate.clone(), &[], options, &NeverCancelled).unwrap();
    assert_eq!(admitted, candidate);
    assert_eq!(admitted.tetrahedra.len(), 3);
}

#[test]
fn physically_illegal_seed_retains_complete_repair_and_independent_admission() {
    let candidate = bipyramid(10.);
    let options = DelaunayInsertionOptions::default();
    assert!(validate_constrained_delaunay_volume_topology(
        &candidate,
        &[],
        options,
        &NeverCancelled
    )
    .is_err());
    let admitted = admit_or_repair(candidate.clone(), &[], options, &NeverCancelled).unwrap();
    assert_eq!(admitted.nodes, candidate.nodes);
    assert_eq!(admitted.tetrahedra.len(), 2);
    validate_constrained_delaunay_volume_topology(&admitted, &[], options, &NeverCancelled)
        .unwrap();
    let boundary = |topology: &DelaunayVolumeTopology| {
        topology
            .incidence
            .boundary_facets
            .iter()
            .map(|face| face.vertex_indices)
            .collect::<BTreeSet<_>>()
    };
    assert_eq!(boundary(&admitted), boundary(&candidate));
}

#[test]
fn seed_admission_rejects_missing_protected_face() {
    let candidate = bipyramid(5.);
    let absent = [
        StableDigest::from_bytes([21; 32]),
        StableDigest::from_bytes([22; 32]),
        StableDigest::from_bytes([23; 32]),
    ];
    assert!(admit_or_repair(
        candidate,
        &[absent],
        DelaunayInsertionOptions::default(),
        &NeverCancelled
    )
    .is_err());
}
