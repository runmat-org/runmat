use std::collections::{BTreeSet, VecDeque};

use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::quality::predicate::{orient3d, PredicateSign};

use super::{
    error, CarvingWork, DelaunayCarving, DelaunayCarvingError, DelaunayCarvingErrorKind,
    DelaunayCarvingSeeds, DelaunayFacetRecovery, DelaunayVolumeTopology,
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
    seeds: &DelaunayCarvingSeeds,
    work: &mut CarvingWork<'_>,
) -> Result<DelaunayCarving, DelaunayCarvingError> {
    let topology = &recovery.segment_recovery.topology;
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

    for (index, seed) in seeds.voids.iter().enumerate() {
        let tetrahedron = locate_seed(topology, seed.coordinates_m, index as u32, work)?;
        flood(
            topology,
            &blocked,
            BTreeSet::from([tetrahedron]),
            Classification::Void,
            &mut classifications,
            work,
        )?;
    }
    for (index, seed) in seeds.regions.iter().enumerate() {
        let tetrahedron = locate_seed(topology, seed.coordinates_m, index as u32, work)?;
        flood(
            topology,
            &blocked,
            BTreeSet::from([tetrahedron]),
            Classification::Region(seed.region_id.clone()),
            &mut classifications,
            work,
        )?;
    }
    if classifications.iter().any(Option::is_none) {
        return Err(error(
            DelaunayCarvingErrorKind::AmbiguousClassification,
            None,
            "one or more facet-bounded components have no exterior, void, or region seed",
        ));
    }

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
    })
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

fn locate_seed(
    topology: &DelaunayVolumeTopology,
    point: [f64; 3],
    seed_index: u32,
    work: &mut CarvingWork<'_>,
) -> Result<u32, DelaunayCarvingError> {
    for (tetrahedron_index, tetrahedron) in topology.tetrahedra.iter().enumerate() {
        work.location(seed_index)?;
        let vertices = tetrahedron
            .vertex_indices
            .map(|index| topology.nodes[index as usize].coordinates_m);
        let signs = (0..4)
            .map(|slot| {
                let mut candidate = vertices;
                candidate[slot] = point;
                orient3d(candidate)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|predicate| {
                error(
                    DelaunayCarvingErrorKind::InvalidSeeds,
                    Some(seed_index),
                    format!("seed location predicate failed: {predicate:?}"),
                )
            })?;
        if signs.iter().all(|sign| *sign != PredicateSign::Negative) {
            if signs.contains(&PredicateSign::Zero) {
                return Err(error(
                    DelaunayCarvingErrorKind::AmbiguousClassification,
                    Some(seed_index),
                    "seed lies on tetrahedron boundary",
                ));
            }
            return Ok(tetrahedron_index as u32);
        }
    }
    Err(error(
        DelaunayCarvingErrorKind::InvalidSeeds,
        Some(seed_index),
        "seed lies outside the tetrahedralized domain",
    ))
}
