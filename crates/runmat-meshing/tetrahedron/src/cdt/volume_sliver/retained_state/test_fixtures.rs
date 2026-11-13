//! Shared physical bipyramid and unrelated-region fixtures for compact search state.
use crate::cdt::{
    evaluate_delaunay_volume_quality, topology::build_delaunay_volume_topology,
    DelaunayTopologyOptions, DelaunayVolumeNode, DelaunayVolumeProvenance, DelaunayVolumeQuality,
    DelaunayVolumeQualityOptions, DelaunayVolumeTopology,
};
use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::{NeverCancelled, StableDigest};

pub(in crate::cdt::volume_sliver) fn pair(
    unrelated: usize,
) -> (DelaunayVolumeTopology, DelaunayVolumeTopology) {
    let mut points = vec![
        [0., 0., -5.],
        [0., 0., 5.],
        [5., 0., 0.],
        [-3., 4., 0.],
        [-3., -4., 0.],
    ];
    let mut retained = Vec::new();
    for ordinal in 0..unrelated {
        let start = points.len() as u32;
        let x = 20. + ordinal as f64 * 3.;
        points.extend([[x, 0., 0.], [x + 1., 0., 0.], [x, 1., 0.], [x, 0., 1.]]);
        retained.push([start, start + 1, start + 2, start + 3]);
    }
    let nodes = points
        .into_iter()
        .enumerate()
        .map(|(index, point)| {
            let mut identity = [0; 32];
            identity[24..].copy_from_slice(&((index + 1) as u64).to_be_bytes());
            DelaunayVolumeNode::explicit(StableDigest::from_bytes(identity), point)
        })
        .collect::<Vec<_>>();
    let build = |mut cells: Vec<[u32; 4]>| {
        cells.extend(retained.iter().copied());
        build_delaunay_volume_topology(
            nodes.clone(),
            cells,
            DelaunayTopologyOptions::default(),
            &NeverCancelled,
        )
        .unwrap()
    };
    (
        build(vec![[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 2]]),
        build(vec![[0, 2, 3, 4], [1, 2, 3, 4]]),
    )
}

pub(in crate::cdt::volume_sliver) fn assigned(
    topology: DelaunayVolumeTopology,
) -> DelaunayVolumeTopology {
    let regions = topology
        .tetrahedra
        .iter()
        .map(|cell| {
            PersistentEntityId::new(
                PersistentEntityKind::Region,
                if topology.nodes[cell.vertex_indices[0] as usize]
                    .coordinates_m
                    .canonical_value()[0]
                    >= 20.
                {
                    "unrelated-part"
                } else {
                    "local-part"
                }
                .into(),
                Vec::new(),
            )
        })
        .collect();
    crate::cdt::assign_delaunay_volume_regions(
        topology,
        regions,
        crate::cdt::DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

pub(in crate::cdt::volume_sliver) fn quality(
    topology: &DelaunayVolumeTopology,
) -> DelaunayVolumeQuality {
    let metric = MetricFieldRequest {
        entity_supports: Vec::new(),
        located_sources: Vec::new(),
        combination: MetricCombinationRule::MostRestrictiveIntersection,
        global_metric: MetricTensor3::isotropic_length_m(10.).unwrap(),
        maximum_grading_ratio: 1.5,
        contributions: Vec::new(),
        cells_across_gap: None,
    };
    evaluate_delaunay_volume_quality(
        topology,
        &metric,
        &DelaunayVolumeProvenance {
            nodes: vec![],
            segments: vec![],
            facets: vec![],
        },
        DelaunayVolumeQualityOptions::default(),
        &NeverCancelled,
    )
    .unwrap()
}

use runmat_meshing_size::metric::{MetricCombinationRule, MetricFieldRequest, MetricTensor3};
