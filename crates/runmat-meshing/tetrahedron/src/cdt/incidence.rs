use std::collections::BTreeMap;

use runmat_geometry_core::{PersistentEntityId, PersistentEntityKind};
use runmat_meshing_core::{
    quality::predicate::{orient3d, PredicateSign},
    MeshingCancellationSignal,
};

use super::{
    topology::{build_delaunay_volume_topology_with_regions, error},
    DelaunayTopologyError, DelaunayTopologyErrorKind, DelaunayTopologyOptions, DelaunayVolumeNode,
    DelaunayVolumeTetrahedron, DelaunayVolumeTopology,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayBoundaryFacet {
    pub vertex_indices: [u32; 3],
    /// Oriented so the incident tetrahedron's opposite vertex has positive
    /// robust orientation relative to this face.
    pub oriented_vertex_indices: [u32; 3],
    pub tetrahedron_index: u32,
    pub opposite_vertex_slot: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayRegionIncidence {
    pub region_id: PersistentEntityId,
    pub tetrahedron_indices: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayVolumeIncidence {
    pub vertex_stars: Vec<Vec<u32>>,
    pub boundary_facets: Vec<DelaunayBoundaryFacet>,
    pub regions: Vec<DelaunayRegionIncidence>,
    pub unassigned_tetrahedron_indices: Vec<u32>,
}

pub fn assign_delaunay_volume_regions(
    mut topology: DelaunayVolumeTopology,
    region_ids: Vec<PersistentEntityId>,
    options: DelaunayTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<DelaunayVolumeTopology, DelaunayTopologyError> {
    let rebuilt = build_delaunay_volume_topology_with_regions(
        topology.nodes.clone(),
        topology
            .tetrahedra
            .iter()
            .map(|tetrahedron| (tetrahedron.vertex_indices, tetrahedron.region_id.clone()))
            .collect(),
        options,
        cancellation,
    )?;
    if rebuilt != topology {
        return Err(error(
            DelaunayTopologyErrorKind::InvalidTetrahedron,
            "region assignment requires canonical checked topology and incidence",
        ));
    }
    if region_ids.len() != topology.tetrahedra.len() {
        return Err(error(
            DelaunayTopologyErrorKind::InvalidRegion,
            "region assignment count must equal the tetrahedron count",
        ));
    }
    for (tetrahedron, region_id) in topology.tetrahedra.iter_mut().zip(region_ids) {
        validate_region(&region_id)?;
        tetrahedron.region_id = Some(region_id);
    }
    topology.incidence =
        build_volume_incidence(&topology.nodes, &topology.tetrahedra, cancellation, options)?;
    Ok(topology)
}

pub(super) fn build_volume_incidence(
    nodes: &[DelaunayVolumeNode],
    tetrahedra: &[DelaunayVolumeTetrahedron],
    cancellation: &dyn MeshingCancellationSignal,
    options: DelaunayTopologyOptions,
) -> Result<DelaunayVolumeIncidence, DelaunayTopologyError> {
    let mut vertex_stars = vec![Vec::new(); nodes.len()];
    let mut boundary_facets = Vec::new();
    let mut regions = BTreeMap::<PersistentEntityId, Vec<u32>>::new();
    let mut unassigned = Vec::new();
    for (tetrahedron_index, tetrahedron) in tetrahedra.iter().enumerate() {
        if (tetrahedron_index as u64).is_multiple_of(options.cancellation_check_interval)
            && cancellation.is_cancelled()
        {
            return Err(error(DelaunayTopologyErrorKind::Cancelled, "cancelled"));
        }
        for vertex in tetrahedron.vertex_indices {
            vertex_stars[vertex as usize].push(tetrahedron_index as u32);
        }
        match &tetrahedron.region_id {
            Some(region_id) => {
                validate_region(region_id)?;
                regions
                    .entry(region_id.clone())
                    .or_default()
                    .push(tetrahedron_index as u32);
            }
            None => unassigned.push(tetrahedron_index as u32),
        }
        for opposite in 0..4 {
            if tetrahedron.neighbors[opposite].is_some() {
                continue;
            }
            boundary_facets.push(boundary_facet(
                nodes,
                tetrahedron,
                tetrahedron_index,
                opposite,
            )?);
        }
    }
    boundary_facets.sort_by_key(|facet| facet.vertex_indices);
    Ok(DelaunayVolumeIncidence {
        vertex_stars,
        boundary_facets,
        regions: regions
            .into_iter()
            .map(|(region_id, tetrahedron_indices)| DelaunayRegionIncidence {
                region_id,
                tetrahedron_indices,
            })
            .collect(),
        unassigned_tetrahedron_indices: unassigned,
    })
}

fn boundary_facet(
    nodes: &[DelaunayVolumeNode],
    tetrahedron: &DelaunayVolumeTetrahedron,
    tetrahedron_index: usize,
    opposite: usize,
) -> Result<DelaunayBoundaryFacet, DelaunayTopologyError> {
    let mut oriented = [0; 3];
    let mut cursor = 0;
    for (slot, vertex) in tetrahedron.vertex_indices.iter().enumerate() {
        if slot != opposite {
            oriented[cursor] = *vertex;
            cursor += 1;
        }
    }
    let opposite_vertex = tetrahedron.vertex_indices[opposite];
    let points = [
        nodes[oriented[0] as usize].coordinates_m,
        nodes[oriented[1] as usize].coordinates_m,
        nodes[oriented[2] as usize].coordinates_m,
        nodes[opposite_vertex as usize].coordinates_m,
    ];
    match orient3d(points).map_err(|predicate| {
        error(
            DelaunayTopologyErrorKind::InvalidNode,
            format!("boundary orientation predicate failed: {predicate:?}"),
        )
    })? {
        PredicateSign::Positive => {}
        PredicateSign::Negative => oriented.swap(0, 1),
        PredicateSign::Zero => {
            return Err(error(
                DelaunayTopologyErrorKind::DegenerateTetrahedron,
                "boundary facet belongs to a degenerate tetrahedron",
            ));
        }
    }
    let mut canonical = oriented;
    canonical.sort_unstable();
    Ok(DelaunayBoundaryFacet {
        vertex_indices: canonical,
        oriented_vertex_indices: oriented,
        tetrahedron_index: tetrahedron_index as u32,
        opposite_vertex_slot: opposite as u8,
    })
}

fn validate_region(region_id: &PersistentEntityId) -> Result<(), DelaunayTopologyError> {
    region_id.validate().map_err(|validation| {
        error(
            DelaunayTopologyErrorKind::InvalidRegion,
            format!("region identity is invalid: {validation}"),
        )
    })?;
    if region_id.kind != PersistentEntityKind::Region {
        return Err(error(
            DelaunayTopologyErrorKind::InvalidRegion,
            "tetrahedron region identity must have region kind",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "incidence/tests.rs"]
mod tests;
