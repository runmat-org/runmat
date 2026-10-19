use super::*;
use crate::cdt::{
    build_delaunay_constraints, DelaunayConstraintNode, DelaunayConstraintOptions,
    DelaunayInsertionOptions,
};
use runmat_meshing_core::{NeverCancelled, StableDigest};

#[test]
fn visible_nondelaunay_stars_decline_without_weakening_physical_admission() {
    let (topology, surface) = crate::cdt::constraints::tests::tetrahedron();
    let original = build_delaunay_constraints(
        &topology,
        &surface,
        DelaunayConstraintOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let DelaunayConstraintFacetSide::Region(region) = &original.facets[0].negative_side else {
        panic!("fixture region")
    };
    assert!(try_build_constraint_star(
        region,
        &original,
        DelaunayInsertionOptions::default(),
        &NeverCancelled
    )
    .unwrap()
    .is_some());
    let mut rejected = 0;
    for facet_index in 0..original.facets.len() {
        for weights in [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.2, 0.3, 0.5],
        ] {
            let mut constraints = original.clone();
            let facet = constraints.facets.remove(facet_index);
            let coordinates_m = std::array::from_fn(|axis| {
                (0..3)
                    .map(|i| {
                        weights[i]
                            * constraints.nodes[facet.vertex_indices[i] as usize].coordinates_m
                                [axis]
                    })
                    .sum()
            });
            let added = constraints.nodes.len() as u32;
            constraints.nodes.push(DelaunayConstraintNode {
                identity: StableDigest::from_bytes([0x71; 32]),
                source_vertex_id: None,
                coordinates_m,
            });
            for edge in 0..3 {
                let mut part = facet.clone();
                part.facet_id = StableDigest::from_bytes([0x80 + edge as u8; 32]);
                part.vertex_indices = [
                    facet.vertex_indices[edge],
                    facet.vertex_indices[(edge + 1) % 3],
                    added,
                ];
                constraints.facets.push(part);
            }
            let raw = build_visible_constraint_star(
                region,
                &constraints,
                Default::default(),
                &NeverCancelled,
            )
            .unwrap()
            .expect("interior face subdivision retains visibility");
            let verdict = crate::cdt::insertion::validate_locally_delaunay_volume_topology(
                &raw,
                Default::default(),
                &NeverCancelled,
            );
            let accepted = try_build_constraint_star(
                region,
                &constraints,
                Default::default(),
                &NeverCancelled,
            )
            .unwrap();
            match verdict {
                Ok(()) => assert!(accepted.is_some()),
                Err(failure) => {
                    assert_eq!(
                        failure.kind,
                        crate::cdt::DelaunayInsertionErrorKind::InvalidTopology
                    );
                    assert!(accepted.is_none());
                    rejected += 1;
                }
            }
        }
    }
    assert!(
        rejected > 0,
        "family must contain visible but physically illegal stars"
    );
}
