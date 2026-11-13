mod assembly;
mod provenance;
#[cfg(test)]
pub(super) mod reference;
pub(in crate::cdt) use assembly::assemble;

use std::collections::{BTreeMap, BTreeSet};

use runmat_geometry_core::PersistentEntityId;

use crate::cdt::{
    DelaunayConstraintFacet, DelaunayConstraintFacetSide, DelaunayConstraintNode,
    DelaunayConstraintSegment, DelaunayConstraints, DelaunayFacetSteinerInsertion,
    DelaunayVolumeMeshError, DelaunayVolumeMeshErrorKind, DelaunayVolumeMeshStage,
};

pub(super) struct ConstraintComponent {
    pub region: PersistentEntityId,
    /// More than one region, joined by conformal interfaces.
    pub joined_regions: bool,
    pub constraints: DelaunayConstraints,
    pub global_facet_indices: Vec<u32>,
}

pub(super) fn partition_constraints(
    constraints: &DelaunayConstraints,
) -> Result<Vec<ConstraintComponent>, DelaunayVolumeMeshError> {
    // Regions joined by a conformal interface are one component: meshing them apart lets one
    // side's segment or facet recovery split the shared boundary where the other side does
    // not, and the assembled mesh is then not conformal on the interface. Regions that touch
    // only through contact or not at all stay separate.
    let mut region_components = regions(&constraints.facets)
        .into_iter()
        .map(|region| BTreeSet::from([region]))
        .collect::<Vec<_>>();
    for facet in &constraints.facets {
        let joined = facet_regions(facet);
        if joined.len() < 2 {
            continue;
        }
        let mut merged = BTreeSet::new();
        region_components.retain(|component| {
            if joined.iter().any(|region| component.contains(region)) {
                merged.extend(component.iter().cloned());
                false
            } else {
                true
            }
        });
        region_components.push(merged);
    }
    if region_components.is_empty() {
        return Err(super::error(
            DelaunayVolumeMeshStage::Constraints,
            DelaunayVolumeMeshErrorKind::InvalidGeometry,
            "volume constraints contain no persistent region",
        ));
    }
    region_components
        .iter()
        .map(|regions| build_component(constraints, regions))
        .collect()
}

pub(super) fn remap_facet_insertions(
    insertions: &mut [DelaunayFacetSteinerInsertion],
    global_facet_indices: &[u32],
) -> Result<(), DelaunayVolumeMeshError> {
    for insertion in insertions {
        insertion.constraint_index = *global_facet_indices
            .get(insertion.constraint_index as usize)
            .ok_or_else(|| {
                super::error(
                    DelaunayVolumeMeshStage::FacetRecovery,
                    DelaunayVolumeMeshErrorKind::InvalidTopology,
                    "component facet insertion references an absent local constraint",
                )
            })?;
    }
    Ok(())
}

fn regions(facets: &[DelaunayConstraintFacet]) -> BTreeSet<PersistentEntityId> {
    let mut regions = BTreeSet::new();
    for facet in facets {
        regions.extend(facet_regions(facet));
    }
    regions
}

fn build_component(
    constraints: &DelaunayConstraints,
    regions: &BTreeSet<PersistentEntityId>,
) -> Result<ConstraintComponent, DelaunayVolumeMeshError> {
    let selected_facets = constraints
        .facets
        .iter()
        .enumerate()
        .filter(|(_, facet)| {
            facet_regions(facet)
                .iter()
                .any(|region| regions.contains(region))
        })
        .collect::<Vec<_>>();
    let referenced_indices = selected_facets
        .iter()
        .flat_map(|(_, facet)| facet.vertex_indices)
        .collect::<BTreeSet<_>>();
    let nodes = referenced_indices
        .iter()
        .map(|index| constraints.nodes[*index as usize].clone())
        .collect::<Vec<DelaunayConstraintNode>>();
    let new_index_by_old = referenced_indices
        .iter()
        .enumerate()
        .map(|(new, old)| (*old, new as u32))
        .collect::<BTreeMap<_, _>>();
    let facets = selected_facets
        .iter()
        .map(|(_, facet)| DelaunayConstraintFacet {
            vertex_indices: facet.vertex_indices.map(|index| new_index_by_old[&index]),
            positive_side: localize_side(&facet.positive_side, regions),
            negative_side: localize_side(&facet.negative_side, regions),
            ..(*facet).clone()
        })
        .collect::<Vec<_>>();
    let facet_segment_keys = facets
        .iter()
        .flat_map(|facet| {
            (0..3).map(|edge| {
                sorted_segment([
                    facet.vertex_indices[edge],
                    facet.vertex_indices[(edge + 1) % 3],
                ])
            })
        })
        .collect::<BTreeSet<_>>();
    let segments_by_identity = constraints
        .segments
        .iter()
        .map(|segment| {
            let mut identities = segment
                .vertex_indices
                .map(|index| constraints.nodes[index as usize].identity);
            identities.sort_unstable();
            (identities, segment)
        })
        .collect::<BTreeMap<_, _>>();
    let segments = facet_segment_keys
        .into_iter()
        .map(|indices| {
            let identities = indices.map(|index| nodes[index as usize].identity);
            let source = segments_by_identity.get(&identities).ok_or_else(|| {
                super::error(
                    DelaunayVolumeMeshStage::Constraints,
                    DelaunayVolumeMeshErrorKind::InvalidTopology,
                    "component facet edge is absent from the full constraint inventory",
                )
            })?;
            Ok(DelaunayConstraintSegment {
                vertex_indices: indices,
                source_edge_id: source.source_edge_id.clone(),
                source_edge_parameters: source.source_edge_parameters,
            })
        })
        .collect::<Result<Vec<_>, DelaunayVolumeMeshError>>()?;
    Ok(ConstraintComponent {
        region: regions
            .first()
            .expect("region component is nonempty")
            .clone(),
        joined_regions: regions.len() > 1,
        constraints: DelaunayConstraints {
            nodes,
            segments,
            facets,
        },
        global_facet_indices: selected_facets
            .iter()
            .map(|(index, _)| *index as u32)
            .collect(),
    })
}

fn localize_side(
    side: &DelaunayConstraintFacetSide,
    regions: &BTreeSet<PersistentEntityId>,
) -> DelaunayConstraintFacetSide {
    match side {
        DelaunayConstraintFacetSide::Region(region) if regions.contains(region) => {
            DelaunayConstraintFacetSide::Region(region.clone())
        }
        DelaunayConstraintFacetSide::Region(_)
        | DelaunayConstraintFacetSide::Exterior
        | DelaunayConstraintFacetSide::Void => DelaunayConstraintFacetSide::Exterior,
    }
}

fn facet_regions(facet: &DelaunayConstraintFacet) -> Vec<PersistentEntityId> {
    let mut regions = [&facet.positive_side, &facet.negative_side]
        .into_iter()
        .filter_map(|side| match side {
            DelaunayConstraintFacetSide::Region(region) => Some(region.clone()),
            DelaunayConstraintFacetSide::Exterior | DelaunayConstraintFacetSide::Void => None,
        })
        .collect::<Vec<_>>();
    regions.sort();
    regions.dedup();
    regions
}

fn sorted_segment(mut vertices: [u32; 2]) -> [u32; 2] {
    vertices.sort_unstable();
    vertices
}

#[cfg(test)]
mod tests;
