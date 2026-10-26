//! Analytic tetrahedron red-refinement. This fixture builds a known conforming
//! complex, not a meshing solve, so projection scale can be tested independently.
use super::*;
use crate::cdt::*;
use runmat_geometry_core::ExactBRepTopology;
use runmat_meshing_core::StableDigest;
use runmat_meshing_surface::{ExactFaceMeshNode, ExactFaceMeshNodeUse, ExactSurfaceMesh};
use sha2::{Digest as _, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn refined(levels: usize) -> (ExactBRepTopology, ExactSurfaceMesh, DelaunayVolumeMesh) {
    let (cad, mut surface) = crate::cdt::constraints::tests::tetrahedron();
    let request = request(ElementOrder::Tet4);
    // Use the four exact corners directly. Running the mesher here would add
    // its own optimization points and make this reference depend on that solver.
    let constraints = build_delaunay_constraints(
        &cad,
        &surface,
        DelaunayConstraintOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let mut nodes = constraints.volume_nodes();
    assert_eq!(nodes.len(), 4);
    let mut cells = vec![std::array::from_fn(|index| nodes[index].identity)];
    for _ in 0..levels {
        let points = nodes
            .iter()
            .map(|node| (node.identity, node.coordinates_m.canonical_value()))
            .collect::<BTreeMap<_, _>>();
        let mut midpoint_ids = BTreeMap::new();
        let mut next_cells = Vec::with_capacity(cells.len() * 8);
        for [a, b, c, d] in cells {
            let mut midpoint = |mut edge: [StableDigest; 2]| {
                edge.sort_unstable();
                *midpoint_ids.entry(edge).or_insert_with(|| {
                    let mut hash = Sha256::new();
                    hash.update(b"analytic-tetrahedron-midpoint/v1\0");
                    for id in edge {
                        hash.update(id.bytes());
                    }
                    let identity = StableDigest::from_bytes(hash.finalize().into());
                    let coordinates = std::array::from_fn(|axis| {
                        (points[&edge[0]][axis] + points[&edge[1]][axis]) * 0.5
                    });
                    nodes.push(DelaunayVolumeNode::explicit(identity, coordinates));
                    identity
                })
            };
            let [ab, ac, ad, bc, bd, cd] =
                [[a, b], [a, c], [a, d], [b, c], [b, d], [c, d]].map(&mut midpoint);
            next_cells.extend([
                [a, ab, ac, ad],
                [b, ab, bc, bd],
                [c, ac, bc, cd],
                [d, ad, bd, cd],
            ]);
            // Shortest octahedron diagonal preserves the reference family's
            // shape under repeated refinement; boundary subdivision is identical.
            let diagonals = [
                ([a, b], [c, d], [ab, cd], [ac, ad, bd, bc]),
                ([a, c], [b, d], [ac, bd], [ab, ad, cd, bc]),
                ([a, d], [b, c], [ad, bc], [ab, ac, cd, bd]),
            ];
            let (_, _, diagonal, cycle) = diagonals
                .into_iter()
                .min_by(|left, right| {
                    let length = |(first, second, _, _): &(
                        [StableDigest; 2],
                        [StableDigest; 2],
                        [StableDigest; 2],
                        [StableDigest; 4],
                    )| {
                        (0..3)
                            .map(|axis| {
                                let delta = (points[&first[0]][axis] + points[&first[1]][axis]
                                    - points[&second[0]][axis]
                                    - points[&second[1]][axis])
                                    * 0.5;
                                delta * delta
                            })
                            .sum::<f64>()
                    };
                    length(left)
                        .total_cmp(&length(right))
                        .then_with(|| left.2.cmp(&right.2))
                })
                .unwrap();
            for index in 0..4 {
                next_cells.push([
                    diagonal[0],
                    diagonal[1],
                    cycle[index],
                    cycle[(index + 1) % 4],
                ]);
            }
        }
        cells = next_cells;
        let midpoint = |mut edge: [StableDigest; 2]| {
            edge.sort_unstable();
            midpoint_ids[&edge]
        };
        let mut triangles = Vec::with_capacity(surface.triangles.len() * 4);
        for triangle in surface.triangles {
            let [a, b, c] = triangle.node_ids;
            let [ab, bc, ca] = [[a, b], [b, c], [c, a]].map(midpoint);
            for mut ids in [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]] {
                let first = (0..3).min_by_key(|&index| ids[index]).unwrap();
                ids.rotate_left(first);
                let mut child = triangle.clone();
                child.node_ids = ids;
                child.physical_area_m2 *= 0.25;
                let mut hash = Sha256::new();
                hash.update(b"analytic-tetrahedron-facet/v1\0");
                hash.update(child.chart_id.bytes());
                for id in ids {
                    hash.update(id.bytes());
                }
                child.triangle_id = StableDigest::from_bytes(hash.finalize().into());
                triangles.push(child);
            }
        }
        surface.triangles = triangles;
        surface.boundary_segments = surface
            .boundary_segments
            .into_iter()
            .flat_map(|segment| {
                let mut left = segment.clone();
                let mut right = segment;
                let center = midpoint(left.node_ids);
                let parameter = (left.edge_parameters[0] + left.edge_parameters[1]) * 0.5;
                left.node_ids[1] = center;
                left.edge_parameters[1] = parameter;
                right.node_ids[0] = center;
                right.edge_parameters[0] = parameter;
                [left, right]
            })
            .collect();
    }
    let points = nodes
        .iter()
        .map(|node| (node.identity, node.coordinates_m.canonical_value()))
        .collect::<BTreeMap<_, _>>();
    let mut uses = BTreeMap::<_, BTreeMap<_, ExactFaceMeshNodeUse>>::new();
    for triangle in &surface.triangles {
        let face: usize = triangle
            .source_face_id
            .source_topology_id
            .strip_prefix("face:")
            .unwrap()
            .parse()
            .unwrap();
        for id in triangle.node_ids {
            let point = points[&id];
            let uv = match face {
                0 => [point[0], point[1]],
                1 => [point[0], point[2]],
                _ => [point[1], point[2]],
            };
            uses.entry(id)
                .or_default()
                .entry((triangle.source_face_id.clone(), triangle.chart_id))
                .or_insert_with(|| ExactFaceMeshNodeUse {
                    source_face_id: triangle.source_face_id.clone(),
                    chart_id: triangle.chart_id,
                    uv,
                    evaluator_uv: uv,
                    exact_edge_parameters: vec![],
                });
        }
    }
    surface.nodes = uses
        .into_iter()
        .map(|(id, uses)| ExactFaceMeshNode {
            node_id: id,
            point_m: points[&id],
            uses: uses.into_values().collect(),
        })
        .collect();
    surface
        .triangles
        .sort_by_key(|triangle| triangle.triangle_id);
    nodes.sort_by_key(|node| node.identity);
    let slots = nodes
        .iter()
        .enumerate()
        .map(|(slot, node)| (node.identity, slot as u32))
        .collect::<BTreeMap<_, _>>();
    let region = cad.regions[0].id.clone();
    let topology = crate::cdt::topology::build_delaunay_volume_topology_with_regions(
        nodes,
        cells
            .into_iter()
            .map(|ids| (ids.map(|id| slots[&id]), Some(region.clone())))
            .collect(),
        DelaunayTopologyOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    let mut segments = BTreeMap::new();
    for segment in &surface.boundary_segments {
        let mut ids = segment.node_ids;
        let mut parameters = segment.edge_parameters;
        if ids[0] > ids[1] {
            ids.swap(0, 1);
            parameters.swap(0, 1);
        }
        segments
            .entry(ids)
            .or_insert_with(|| DelaunaySegmentProvenance {
                node_identities: ids,
                entity_ids: vec![segment.source_edge_id.clone()],
                edge_parameters: parameters,
            });
    }
    let mut facets = surface
        .triangles
        .iter()
        .map(|triangle| {
            let mut ids = triangle.node_ids;
            ids.sort_unstable();
            DelaunayFacetProvenance {
                node_identities: ids,
                chart_id: triangle.chart_id,
                entity_ids: vec![triangle.source_face_id.clone()],
                region_ids: vec![region.clone()],
            }
        })
        .collect::<Vec<_>>();
    facets.sort_by_key(|facet| facet.node_identities);
    let provenance = DelaunayVolumeProvenance {
        nodes: constraints
            .nodes
            .iter()
            .filter_map(|node| {
                node.source_vertex_id
                    .as_ref()
                    .map(|id| DelaunayNodeProvenance {
                        node_identity: node.identity,
                        entity_ids: vec![id.clone()],
                    })
            })
            .collect(),
        segments: segments.into_values().collect(),
        facets,
    };
    validate_delaunay_volume_provenance(
        &topology,
        &provenance,
        DelaunayVolumeProvenanceOptions::default(),
        &NeverCancelled,
    )
    .unwrap();
    assert_eq!(topology.tetrahedra.len(), 8usize.pow(levels as u32));
    let boundary = topology
        .incidence
        .boundary_facets
        .iter()
        .map(|face| {
            let mut ids = face
                .vertex_indices
                .map(|slot| topology.nodes[slot as usize].identity);
            ids.sort_unstable();
            ids
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        boundary,
        provenance
            .facets
            .iter()
            .map(|facet| facet.node_identities)
            .collect()
    );
    let quality = evaluate_delaunay_volume_quality(
        &topology,
        &request.metric,
        &provenance,
        volume_options().quality,
        &NeverCancelled,
    )
    .unwrap();
    (
        cad,
        surface,
        DelaunayVolumeMesh {
            topology,
            provenance,
            quality,
            facet_recovery_insertions: vec![],
            quality_mutations: vec![],
        },
    )
}
