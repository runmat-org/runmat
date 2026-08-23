use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::{
    quality::predicate::{orient3d, PredicateSign},
    StableDigest,
};

use super::{
    DelaunayConstraintFacet, DelaunayConstraintFacetSide, DelaunayConstraintNode,
    DelaunayConstraintSegment, DelaunayConstraints, DelaunayVolumeTopology,
};

pub(super) fn same_facet_multi_steiner_constraints() -> DelaunayConstraints {
    constraints(vec![internal_facet()], vec![[0, 1], [0, 2], [1, 2]])
}

pub(super) fn closed_same_facet_multi_steiner_constraints(
    initial: &DelaunayVolumeTopology,
) -> DelaunayConstraints {
    let coordinates = same_facet_multi_steiner_coordinates();
    let mut facets = initial
        .incidence
        .boundary_facets
        .iter()
        .enumerate()
        .map(|(index, boundary)| {
            let tetrahedron = &initial.tetrahedra[boundary.tetrahedron_index as usize];
            let interior_node = tetrahedron.vertex_indices[boundary.opposite_vertex_slot as usize];
            let face_z = boundary
                .oriented_vertex_indices
                .map(|node| coordinates[node as usize][2]);
            assert!(
                !(face_z.iter().any(|z| *z > 0.0) && face_z.iter().any(|z| *z < 0.0)),
                "the closed fixture boundary must be conformal to its internal interface"
            );
            let region = if face_z.iter().any(|z| *z > 0.0) {
                "upper"
            } else {
                "lower"
            };
            let (positive_side, negative_side) = region_and_exterior_sides(
                coordinates,
                boundary.oriented_vertex_indices,
                interior_node,
                region,
            );
            DelaunayConstraintFacet {
                facet_id: StableDigest::from_bytes([(index + 100) as u8; 32]),
                chart_id: StableDigest::from_bytes([(index + 140) as u8; 32]),
                vertex_indices: boundary.oriented_vertex_indices,
                source_face_id: entity(PersistentEntityKind::Face, &format!("outer:{index}")),
                positive_side,
                negative_side,
                contact_ids: Vec::new(),
            }
        })
        .collect::<Vec<_>>();
    facets.push(internal_facet());
    facets.sort_by_key(|facet| {
        let mut vertices = facet.vertex_indices;
        vertices.sort_unstable();
        (vertices, facet.facet_id)
    });
    let mut segments = facets
        .iter()
        .flat_map(|facet| {
            let vertices = facet.vertex_indices;
            [
                [vertices[0], vertices[1]],
                [vertices[1], vertices[2]],
                [vertices[2], vertices[0]],
            ]
            .map(|mut segment| {
                segment.sort_unstable();
                segment
            })
        })
        .collect::<Vec<_>>();
    segments.sort_unstable();
    segments.dedup();
    constraints(facets, segments)
}

pub(super) fn same_facet_multi_steiner_nodes() -> Vec<DelaunayConstraintNode> {
    same_facet_multi_steiner_coordinates()
        .into_iter()
        .enumerate()
        .map(|(index, coordinates_m)| DelaunayConstraintNode {
            identity: StableDigest::from_bytes([(index + 1) as u8; 32]),
            source_vertex_id: None,
            coordinates_m,
        })
        .collect()
}

fn constraints(
    facets: Vec<DelaunayConstraintFacet>,
    segments: Vec<[u32; 2]>,
) -> DelaunayConstraints {
    DelaunayConstraints {
        nodes: same_facet_multi_steiner_nodes(),
        segments: segments
            .into_iter()
            .map(|vertex_indices| DelaunayConstraintSegment {
                vertex_indices,
                source_edge_id: None,
                source_edge_parameters: None,
            })
            .collect(),
        facets,
    }
}

fn internal_facet() -> DelaunayConstraintFacet {
    let coordinates = same_facet_multi_steiner_coordinates();
    let upper_is_positive = orient3d([
        coordinates[0],
        coordinates[1],
        coordinates[2],
        coordinates[3],
    ])
    .expect("the fixture uses finite coordinates")
        == PredicateSign::Negative;
    let upper = DelaunayConstraintFacetSide::Region(entity(PersistentEntityKind::Region, "upper"));
    let lower = DelaunayConstraintFacetSide::Region(entity(PersistentEntityKind::Region, "lower"));
    DelaunayConstraintFacet {
        facet_id: StableDigest::from_bytes([90; 32]),
        chart_id: StableDigest::from_bytes([91; 32]),
        vertex_indices: [0, 1, 2],
        source_face_id: entity(PersistentEntityKind::Face, "interface"),
        positive_side: if upper_is_positive {
            upper.clone()
        } else {
            lower.clone()
        },
        negative_side: if upper_is_positive { lower } else { upper },
        contact_ids: Vec::new(),
    }
}

fn region_and_exterior_sides(
    coordinates: [[f64; 3]; 12],
    facet: [u32; 3],
    interior_node: u32,
    region: &str,
) -> (DelaunayConstraintFacetSide, DelaunayConstraintFacetSide) {
    let region = DelaunayConstraintFacetSide::Region(entity(PersistentEntityKind::Region, region));
    let points = [
        coordinates[facet[0] as usize],
        coordinates[facet[1] as usize],
        coordinates[facet[2] as usize],
        coordinates[interior_node as usize],
    ];
    if orient3d(points).expect("the fixture uses finite coordinates") == PredicateSign::Negative {
        (region, DelaunayConstraintFacetSide::Exterior)
    } else {
        (DelaunayConstraintFacetSide::Exterior, region)
    }
}

fn entity(kind: PersistentEntityKind, source_topology_id: &str) -> PersistentEntityId {
    PersistentEntityId {
        kind,
        source_topology_id: source_topology_id.to_owned(),
        assembly_path: Vec::new(),
    }
}

fn same_facet_multi_steiner_coordinates() -> [[f64; 3]; 12] {
    [
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [
            2.146_286_764_315_602,
            1.740_015_072_675_053_6,
            3.949_560_185_614_555,
        ],
        [
            0.672_028_477_081_060_7,
            3.812_110_815_504_620_4,
            -2.865_885_590_115_422_3,
        ],
        [
            2.268_524_209_207_779,
            2.535_807_099_982_384,
            0.921_485_771_101_690_4,
        ],
        [
            3.123_550_544_898_799,
            1.509_084_834_193_832,
            -0.287_310_370_063_132,
        ],
        [
            1.318_734_874_744_291_5,
            1.712_278_473_534_934_6,
            0.326_696_181_000_617_26,
        ],
        [
            1.984_278_580_220_735_7,
            3.295_687_031_023_837_8,
            -0.442_500_790_535_345_3,
        ],
        [
            0.821_271_736_879_066_4,
            3.475_122_822_136_877_5,
            3.375_402_648_270_280_4,
        ],
        [
            2.932_055_176_698_859_6,
            2.076_521_826_754_142,
            -0.032_682_551_834_443_88,
        ],
        [
            2.421_687_614_392_703_3,
            2.931_676_642_503_659_6,
            0.726_344_109_359_386_3,
        ],
    ]
}
