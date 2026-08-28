use std::collections::{BTreeSet, VecDeque};

use runmat_geometry_core::PersistentEntityId;

mod sides;

use sides::classify_facet_sides;

use super::{
    error, CarvingWork, DelaunayCarvedFacet, DelaunayCarving, DelaunayCarvingError,
    DelaunayCarvingErrorKind, DelaunayConstraintFacetSide, DelaunayConstraints,
    DelaunayFacetRecovery, DelaunayVolumeTopology,
};
use crate::cdt::topology::build_delaunay_volume_topology_with_regions;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Classification {
    Exterior,
    Void,
    Region(PersistentEntityId),
}

pub(super) fn classify_and_build(
    recovery: &DelaunayFacetRecovery,
    constraints: &DelaunayConstraints,
    work: &mut CarvingWork<'_>,
) -> Result<DelaunayCarving, DelaunayCarvingError> {
    let topology = &recovery.topology;
    let blocked = blocked_faces(recovery)?;
    let mut classifications = vec![None; topology.tetrahedra.len()];
    let exterior = topology
        .incidence
        .boundary_facets
        .iter()
        .filter(|facet| !blocked.contains(&facet.vertex_indices))
        .map(|facet| facet.tetrahedron_index)
        .collect::<BTreeSet<_>>();
    flood(
        topology,
        &blocked,
        exterior,
        Classification::Exterior,
        &mut classifications,
        work,
    )?;

    classify_facet_sides(recovery, constraints, &blocked, &mut classifications, work)?;
    if classifications.iter().any(Option::is_none) {
        return Err(error(
            DelaunayCarvingErrorKind::AmbiguousClassification,
            None,
            "one or more facet-bounded components have no authoritative facet-side classification",
        ));
    }
    let facets = classify_facets(recovery, constraints, &classifications)?;

    let mut retained = Vec::new();
    let mut removed_tetrahedra = Vec::new();
    for (tetrahedron, classification) in topology.tetrahedra.iter().zip(classifications) {
        let classification = classification.ok_or_else(|| {
            error(
                DelaunayCarvingErrorKind::AmbiguousClassification,
                None,
                "tetrahedron classification disappeared before carving",
            )
        })?;
        match classification {
            Classification::Region(region_id) => {
                retained.push((tetrahedron.vertex_indices, Some(region_id)));
            }
            Classification::Exterior | Classification::Void => {
                let mut key = tetrahedron
                    .vertex_indices
                    .map(|index| topology.nodes[index as usize].identity);
                key.sort_unstable();
                removed_tetrahedra.push(key);
            }
        }
    }
    let used_nodes = retained
        .iter()
        .flat_map(|(vertices, _)| vertices)
        .copied()
        .collect::<BTreeSet<_>>();
    let mut remap = vec![None; topology.nodes.len()];
    let mut nodes = Vec::with_capacity(used_nodes.len());
    for (index, node) in topology.nodes.iter().enumerate() {
        let index_u32 = u32::try_from(index).map_err(|_| {
            error(
                DelaunayCarvingErrorKind::ResourceLimit,
                None,
                "source node inventory exceeds index capacity",
            )
        })?;
        if used_nodes.contains(&index_u32) {
            remap[index] = Some(u32::try_from(nodes.len()).map_err(|_| {
                error(
                    DelaunayCarvingErrorKind::ResourceLimit,
                    None,
                    "carved node inventory exceeds index capacity",
                )
            })?);
            nodes.push(*node);
        }
    }
    for (vertices, _) in &mut retained {
        for vertex in vertices {
            *vertex = remap[*vertex as usize].ok_or_else(|| {
                error(
                    DelaunayCarvingErrorKind::InvalidTopology,
                    None,
                    "retained tetrahedron node was omitted during carving compaction",
                )
            })?;
        }
    }
    let topology = build_delaunay_volume_topology_with_regions(
        nodes,
        retained,
        work.options
            .facet_recovery
            .segment_recovery
            .insertion
            .topology,
        work.cancellation,
    )
    .map_err(|topology_error| {
        error(
            match topology_error.kind {
                crate::cdt::DelaunayTopologyErrorKind::ResourceLimit => {
                    DelaunayCarvingErrorKind::ResourceLimit
                }
                crate::cdt::DelaunayTopologyErrorKind::Cancelled => {
                    DelaunayCarvingErrorKind::Cancelled
                }
                _ => DelaunayCarvingErrorKind::InvalidTopology,
            },
            None,
            topology_error.to_string(),
        )
    })?;
    Ok(DelaunayCarving {
        topology,
        removed_tetrahedra,
        facets,
    })
}

fn classify_facets(
    recovery: &DelaunayFacetRecovery,
    constraints: &DelaunayConstraints,
    classifications: &[Option<Classification>],
) -> Result<Vec<DelaunayCarvedFacet>, DelaunayCarvingError> {
    let topology = &recovery.topology;
    let mut result = Vec::with_capacity(recovery.facets.len());
    for facet in &recovery.facets {
        let mut expected = None;
        for triangle in &facet.triangles {
            let indices = triangle.node_identities.map(|identity| {
                topology
                    .nodes
                    .binary_search_by_key(&identity, |node| node.identity)
                    .map(|index| index as u32)
                    .map_err(|_| {
                        error(
                            DelaunayCarvingErrorKind::InvalidTopology,
                            None,
                            "facet-classification node is missing from topology",
                        )
                    })
            });
            let [first, second, third] = indices;
            let indices = [first?, second?, third?];
            let uses = topology.incidence.vertex_stars[indices[0] as usize]
                .iter()
                .copied()
                .filter(|tetrahedron| {
                    let vertices = topology.tetrahedra[*tetrahedron as usize].vertex_indices;
                    vertices.contains(&indices[1]) && vertices.contains(&indices[2])
                })
                .collect::<Vec<_>>();
            if uses.is_empty() || uses.len() > 2 {
                return Err(error(
                    DelaunayCarvingErrorKind::InvalidTopology,
                    None,
                    "recovered facet support has invalid tetrahedron incidence",
                ));
            }
            let mut signature = FacetSignature {
                region_ids: BTreeSet::new(),
                borders_exterior: uses.len() == 1,
                borders_void: false,
            };
            for tetrahedron in uses {
                match classifications[tetrahedron as usize]
                    .as_ref()
                    .ok_or_else(|| {
                        error(
                            DelaunayCarvingErrorKind::AmbiguousClassification,
                            None,
                            "facet support references an unclassified tetrahedron",
                        )
                    })? {
                    Classification::Exterior => signature.borders_exterior = true,
                    Classification::Void => signature.borders_void = true,
                    Classification::Region(region_id) => {
                        signature.region_ids.insert(region_id.clone());
                    }
                }
            }
            if expected.as_ref().is_some_and(|value| value != &signature) {
                return Err(error(
                    DelaunayCarvingErrorKind::AmbiguousClassification,
                    None,
                    "one PLC facet has inconsistent classification across its support",
                ));
            }
            expected = Some(signature);
        }
        let signature = expected.ok_or_else(|| {
            error(
                DelaunayCarvingErrorKind::InvalidTopology,
                None,
                "recovered facet has no support triangles",
            )
        })?;
        let constraint = &constraints.facets[facet.constraint_index as usize];
        let authored = authored_signature(&constraint.positive_side, &constraint.negative_side);
        if signature != authored {
            return Err(error(
                DelaunayCarvingErrorKind::AmbiguousClassification,
                Some(facet.constraint_index),
                "recovered facet adjacency disagrees with its authoritative side classification",
            ));
        }
        result.push(DelaunayCarvedFacet {
            constraint_index: facet.constraint_index,
            region_ids: signature.region_ids.into_iter().collect(),
            borders_exterior: signature.borders_exterior,
            borders_void: signature.borders_void,
        });
    }
    Ok(result)
}

fn authored_signature(
    positive: &DelaunayConstraintFacetSide,
    negative: &DelaunayConstraintFacetSide,
) -> FacetSignature {
    let mut signature = FacetSignature {
        region_ids: BTreeSet::new(),
        borders_exterior: false,
        borders_void: false,
    };
    for side in [positive, negative] {
        match side {
            DelaunayConstraintFacetSide::Region(region) => {
                signature.region_ids.insert(region.clone());
            }
            DelaunayConstraintFacetSide::Exterior => signature.borders_exterior = true,
            DelaunayConstraintFacetSide::Void => signature.borders_void = true,
        }
    }
    signature
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct FacetSignature {
    region_ids: BTreeSet<PersistentEntityId>,
    borders_exterior: bool,
    borders_void: bool,
}

fn blocked_faces(
    recovery: &DelaunayFacetRecovery,
) -> Result<BTreeSet<[u32; 3]>, DelaunayCarvingError> {
    recovery
        .facets
        .iter()
        .flat_map(|facet| &facet.triangles)
        .map(|triangle| {
            let indices = triangle.node_identities.map(|identity| {
                recovery
                    .segment_recovery
                    .topology
                    .nodes
                    .binary_search_by_key(&identity, |node| node.identity)
                    .map(|index| index as u32)
                    .map_err(|_| {
                        error(
                            DelaunayCarvingErrorKind::InvalidTopology,
                            None,
                            "recovered facet node is missing from carving topology",
                        )
                    })
            });
            let [first, second, third] = indices;
            let mut face = [first?, second?, third?];
            face.sort_unstable();
            Ok(face)
        })
        .collect()
}

fn flood(
    topology: &DelaunayVolumeTopology,
    blocked: &BTreeSet<[u32; 3]>,
    starts: BTreeSet<u32>,
    classification: Classification,
    classifications: &mut [Option<Classification>],
    work: &mut CarvingWork<'_>,
) -> Result<(), DelaunayCarvingError> {
    let mut queue = VecDeque::from_iter(starts);
    while let Some(tetrahedron_index) = queue.pop_front() {
        work.flood()?;
        let slot = &mut classifications[tetrahedron_index as usize];
        match slot {
            Some(existing) if *existing == classification => continue,
            Some(_) => {
                return Err(error(
                    DelaunayCarvingErrorKind::AmbiguousClassification,
                    None,
                    "carving floods overlap without a recovered facet between them",
                ));
            }
            None => *slot = Some(classification.clone()),
        }
        let tetrahedron = &topology.tetrahedra[tetrahedron_index as usize];
        for (opposite, neighbor) in tetrahedron.neighbors.iter().enumerate() {
            let Some(neighbor) = neighbor else {
                continue;
            };
            let mut face = tetrahedron
                .vertex_indices
                .into_iter()
                .enumerate()
                .filter_map(|(slot, vertex)| (slot != opposite).then_some(vertex))
                .collect::<Vec<_>>();
            face.sort_unstable();
            if !blocked.contains(&[face[0], face[1], face[2]]) {
                queue.push_back(*neighbor);
            }
        }
    }
    Ok(())
}
