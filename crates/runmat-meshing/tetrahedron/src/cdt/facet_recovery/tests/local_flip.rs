use super::*;
use crate::cdt::mutation::TopologyMutation;

// witness: FEA-SHARED-RESOURCES-001/machine_cpu_execution @
//   shared_infrastructure/canonical_mixed_topology/numerical_kernel;
//   boundary: A local facet flip matches the independently rebuilt canonical operator with zero, 64
//   and 2,048 unrelated cells, under a fixed 128-step local search budget. This checks owned
//   mutation equivalence and query locality; it does not establish parallel throughput, allocator
//   demand, arbitrary cavity/refinement providers or whole-study scaling.
#[test]
fn local_flip_matches_full_operator_and_is_independent_of_unrelated_cells() {
    for unrelated in [0, 64, 2048] {
        let constraints = constraints(false);
        let mut source = segment_recovery(true, &constraints);
        let mut nodes = source.topology.nodes.clone();
        let mut cells = source
            .topology
            .tetrahedra
            .iter()
            .map(|cell| cell.vertex_indices)
            .collect::<Vec<_>>();
        for island in 0..unrelated {
            let offset = nodes.len() as u32;
            for (corner, point) in [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
                .into_iter()
                .enumerate()
            {
                let mut identity = [0x80; 32];
                identity[24..].copy_from_slice(&((island * 4 + corner) as u64).to_be_bytes());
                nodes.push(DelaunayVolumeNode::explicit(
                    StableDigest::from_bytes(identity),
                    [point[0] + 100. + island as f64 * 10., point[1], point[2]],
                ));
            }
            cells.push([offset, offset + 1, offset + 2, offset + 3]);
        }
        source.topology = build_delaunay_volume_topology(
            nodes,
            cells,
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap();
        let facet = constraints.facets[0]
            .vertex_indices
            .map(|v| constraints.nodes[v as usize].identity);
        let options = DelaunayFacetRecoveryOptions::default();
        let reference = super::super::flip::try_recover_facet_with_edge_flip(
            &source,
            facet,
            &[],
            0,
            &mut FacetRecoveryWork::new(options, &NeverCancelled),
        )
        .unwrap()
        .unwrap();
        let protected_edges = super::super::support_refinement::recovered_segment_edges(&source);
        let mut mutable = TopologyMutation::from_canonical(source.topology);
        let mut bounded = options;
        bounded.maximum_search_steps = 128;
        assert!(super::super::flip::try_recover_facet_with_local_edge_flip(
            &mut mutable,
            facet,
            &protected_edges,
            &std::collections::BTreeSet::new(),
            0,
            &mut FacetRecoveryWork::new(bounded, &NeverCancelled)
        )
        .unwrap());
        let actual = mutable
            .freeze(options.segment_recovery.insertion.topology, &NeverCancelled)
            .unwrap();
        assert_eq!(actual, reference, "unrelated cells: {unrelated}");
    }
}

#[test]
fn protected_edge_and_internal_face_preserve_original_mutation() {
    let constraints = constraints(false);
    let source = segment_recovery(true, &constraints);
    let facet = constraints.facets[0]
        .vertex_indices
        .map(|v| constraints.nodes[v as usize].identity);
    let edge = [constraints.nodes[3].identity, constraints.nodes[4].identity];
    let mut face = [edge[0], edge[1], facet[0]];
    face.sort_unstable();
    for (edges, faces) in [(vec![edge], vec![]), (vec![], vec![face])] {
        let mut mutable = TopologyMutation::from_canonical(source.topology.clone());
        let options = DelaunayFacetRecoveryOptions::default();
        assert!(!super::super::flip::try_recover_facet_with_local_edge_flip(
            &mut mutable,
            facet,
            &edges,
            &faces.into_iter().collect(),
            0,
            &mut FacetRecoveryWork::new(options, &NeverCancelled)
        )
        .unwrap());
        assert_eq!(
            mutable
                .freeze(options.segment_recovery.insertion.topology, &NeverCancelled)
                .unwrap(),
            source.topology
        );
    }
}

#[test]
fn admission_budget_failure_rolls_back_the_complete_local_trial() {
    let constraints = constraints(false);
    let source = segment_recovery(true, &constraints);
    let facet = constraints.facets[0]
        .vertex_indices
        .map(|v| constraints.nodes[v as usize].identity);
    let mut mutable = TopologyMutation::from_canonical(source.topology.clone());
    let options = DelaunayFacetRecoveryOptions {
        maximum_search_steps: 7,
        ..DelaunayFacetRecoveryOptions::default()
    };
    let failure = super::super::flip::try_recover_facet_with_local_edge_flip(
        &mut mutable,
        facet,
        &[],
        &std::collections::BTreeSet::new(),
        0,
        &mut FacetRecoveryWork::new(options, &NeverCancelled),
    )
    .unwrap_err();
    assert_eq!(failure.kind, DelaunayFacetRecoveryErrorKind::ResourceLimit);
    assert_eq!(
        mutable
            .freeze(options.segment_recovery.insertion.topology, &NeverCancelled)
            .unwrap(),
        source.topology
    );
}
