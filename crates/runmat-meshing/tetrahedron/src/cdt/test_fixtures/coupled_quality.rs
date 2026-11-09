//! A compliant octahedral boundary encloses a long interior edge and thin
//! collars. Optional unrelated closed tetrahedra support bounded-local-work tests.
use crate::cdt::*;
use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::{NeverCancelled, StableDigest};
use runmat_meshing_size::metric::{
    MetricCombinationRule, MetricContribution, MetricContributionScope, MetricFieldRequest,
    MetricSourceKind, MetricTensor3,
};

fn region() -> PersistentEntityId {
    PersistentEntityId::new(
        PersistentEntityKind::Region,
        "octahedron".into(),
        Vec::new(),
    )
}
pub(in crate::cdt) fn fixture(
    permuted: bool,
    background: usize,
) -> (DelaunayVolumeTopology, DelaunayVolumeProvenance) {
    fixture_scaled(permuted, background, [1.0; 3])
}

pub(in crate::cdt) fn fixture_scaled(
    permuted: bool,
    background: usize,
    scale: [f64; 3],
) -> (DelaunayVolumeTopology, DelaunayVolumeProvenance) {
    let points = [
        [1., 0., 0.],
        [-1., 0., 0.],
        [0., 1., 0.],
        [0., -1., 0.],
        [0., 0., 1.],
        [0., 0., -1.],
        [0.95, 0., 0.],
        [-0.95, 0., 0.],
    ]
    .map(|point| std::array::from_fn(|axis| point[axis] * scale[axis]));
    let nodes = points
        .into_iter()
        .enumerate()
        .map(|(i, p)| {
            let identity = if permuted { 100 - i as u8 } else { i as u8 + 1 };
            DelaunayVolumeNode::explicit(StableDigest::from_bytes([identity; 32]), p)
        })
        .collect();
    let base =
        build_delaunay_volume_point_set(nodes, DelaunayPointSetOptions::default(), &NeverCancelled)
            .unwrap();
    let mut cells = base
        .tetrahedra
        .iter()
        .map(|cell| cell.vertex_indices)
        .collect::<Vec<_>>();
    let mut regions = vec![region(); cells.len()];
    let mut nodes = base.nodes;
    for index in 0..background {
        let offset = nodes.len() as u32;
        let x = 10. + index as f64 * 4.;
        for (corner, point) in [[x, 0., 0.], [x + 0.1, 0., 0.], [x, 0.1, 0.], [x, 0., 0.1]]
            .into_iter()
            .enumerate()
        {
            let mut identity = [208; 32];
            // Canonical topology requires lexicographically ordered identities.
            identity[1..9].copy_from_slice(&((index * 4 + corner + 1) as u64).to_be_bytes());
            nodes.push(DelaunayVolumeNode::explicit(
                StableDigest::from_bytes(identity),
                point,
            ));
        }
        cells.push([offset, offset + 1, offset + 2, offset + 3]);
        regions.push(PersistentEntityId::new(
            PersistentEntityKind::Region,
            format!("background-{index}"),
            Vec::new(),
        ));
    }
    let topology = build_delaunay_volume_topology(
        nodes,
        cells,
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let topology = assign_delaunay_volume_regions(
        topology,
        regions,
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let mut facets = topology
        .incidence
        .boundary_facets
        .iter()
        .map(|face| {
            let mut identities = face
                .vertex_indices
                .map(|i| topology.nodes[i as usize].identity);
            identities.sort_unstable();
            let mut points = face.vertex_indices.map(|i| {
                topology.nodes[i as usize]
                    .coordinates_m
                    .canonical_value()
                    .map(f64::to_bits)
            });
            points.sort_unstable();
            let entity = PersistentEntityId::new(
                PersistentEntityKind::Face,
                format!("outer-{points:?}"),
                Vec::new(),
            );
            let region = topology.tetrahedra[face.tetrahedron_index as usize]
                .region_id
                .clone()
                .unwrap();
            DelaunayFacetProvenance {
                node_identities: identities,
                chart_id: StableDigest::from_bytes([200; 32]),
                entity_ids: vec![entity],
                region_ids: vec![region],
            }
        })
        .collect::<Vec<_>>();
    facets.sort_by_key(|face| face.node_identities);
    assert_eq!(facets.len(), 8 + 4 * background);
    (
        topology,
        DelaunayVolumeProvenance {
            nodes: Vec::new(),
            segments: Vec::new(),
            facets,
        },
    )
}
pub(in crate::cdt) fn metric(
    provenance: &DelaunayVolumeProvenance,
    scoped: bool,
) -> MetricFieldRequest {
    let mut request = MetricFieldRequest {
        entity_supports: Vec::new(),
        located_sources: Vec::new(),
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3::isotropic_length_m(1.5).unwrap(),
        maximum_grading_ratio: 1.5,
        contributions: Vec::new(),
        cells_across_gap: None,
    };
    if scoped {
        // The original six vertices use byte-repeated IDs below 208. Restrict
        // this metric to the same octahedral face in every background size.
        let entity = provenance
            .facets
            .iter()
            .filter(|face| face.node_identities.iter().all(|id| id.bytes()[0] < 208))
            .flat_map(|face| face.entity_ids.iter())
            .min()
            .unwrap()
            .clone();
        request.contributions.push(MetricContribution {
            source: MetricSourceKind::Face,
            scope: MetricContributionScope::Entity { entity_id: entity },
            metric: MetricTensor3::isotropic_length_m(1.45).unwrap(),
        });
    }
    request
}
