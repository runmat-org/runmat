//! Edge-collapse planning follows only the changed star. The explicit complete
//! candidate is materialized once after local deduplication and inventory admission.
use super::*;

struct CollapsePlan {
    removed: BTreeSet<usize>,
    changed: BTreeMap<[u32; 4], ([u32; 4], Option<runmat_geometry_core::PersistentEntityId>)>,
}

fn prepare(
    topology: &DelaunayVolumeTopology,
    source_index: usize,
    target_index: usize,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<CollapsePlan>, DelaunayVolumeSliverError> {
    if source_index == target_index {
        return Ok(None);
    }
    let removed = topology.incidence.vertex_stars[source_index]
        .iter()
        .map(|index| *index as usize)
        .collect::<BTreeSet<_>>();
    let mut changed = BTreeMap::new();
    for (position, index) in removed.iter().copied().enumerate() {
        super::super::checkpoint(position as u64, options, cancellation)?;
        let cell = &topology.tetrahedra[index];
        let vertices = cell.vertex_indices.map(|vertex| {
            if vertex as usize == source_index {
                target_index as u32
            } else {
                vertex
            }
        });
        let mut key = vertices;
        key.sort_unstable();
        if key.windows(2).any(|pair| pair[0] == pair[1]) {
            continue;
        }
        if let Some(existing) =
            crate::cdt::topology::queries::canonical_tetrahedron_index(topology, key, &mut || {})
        {
            if !removed.contains(&existing) {
                if topology.tetrahedra[existing].region_id != cell.region_id {
                    return Ok(None);
                }
                continue;
            }
        }
        match changed.entry(key) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert((vertices, cell.region_id.clone()));
            }
            std::collections::btree_map::Entry::Occupied(entry) => {
                if entry.get().1 != cell.region_id {
                    return Ok(None);
                }
            }
        }
    }
    Ok(Some(CollapsePlan { removed, changed }))
}

pub(super) fn collapse_interior_edge(
    topology: &DelaunayVolumeTopology,
    source_index: usize,
    target_identity: StableDigest,
    options: DelaunayVolumeSliverOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<DelaunayVolumeTopology>, DelaunayVolumeSliverError> {
    let target = topology
        .nodes
        .binary_search_by_key(&target_identity, |node| node.identity)
        .map_err(|_| {
            error(
                DelaunayVolumeSliverErrorKind::InvalidTopology,
                "interior edge-collapse target is absent from canonical topology",
            )
        })?;
    let Some(plan) = prepare(topology, source_index, target, options, cancellation)? else {
        return Ok(None);
    };
    let count = topology
        .tetrahedra
        .len()
        .checked_sub(plan.removed.len())
        .and_then(|count| count.checked_add(plan.changed.len()))
        .ok_or_else(|| {
            error(
                DelaunayVolumeSliverErrorKind::ResourceLimit,
                "collapse inventory overflow",
            )
        })?;
    if topology.nodes.len().saturating_sub(1) as u64 > options.insertion.topology.maximum_nodes
        || count as u64 > options.insertion.topology.maximum_tetrahedra
    {
        return Err(error(
            DelaunayVolumeSliverErrorKind::ResourceLimit,
            "collapse inventory exceeds original topology limits",
        ));
    }
    let mut nodes = Vec::with_capacity(topology.nodes.len().saturating_sub(1));
    for (index, node) in topology.nodes.iter().copied().enumerate() {
        super::super::checkpoint(index as u64, options, cancellation)?;
        if index != source_index {
            nodes.push(node);
        }
    }
    let remap = |vertices: [u32; 4]| {
        vertices.map(|vertex| vertex - u32::from(vertex as usize > source_index))
    };
    let mut cells = Vec::with_capacity(count);
    for (index, cell) in topology.tetrahedra.iter().enumerate() {
        super::super::checkpoint(index as u64, options, cancellation)?;
        if !plan.removed.contains(&index) {
            cells.push((remap(cell.vertex_indices), cell.region_id.clone()));
        }
    }
    cells.extend(
        plan.changed
            .into_values()
            .map(|(vertices, region)| (remap(vertices), region)),
    );
    match build_delaunay_volume_topology_with_regions(
        nodes,
        cells,
        options.insertion.topology,
        cancellation,
    ) {
        Ok(topology) => Ok(Some(topology)),
        Err(failure) => match failure.kind {
            DelaunayTopologyErrorKind::ResourceLimit => Err(error(
                DelaunayVolumeSliverErrorKind::ResourceLimit,
                failure.to_string(),
            )),
            DelaunayTopologyErrorKind::Cancelled => Err(error(
                DelaunayVolumeSliverErrorKind::Cancelled,
                failure.to_string(),
            )),
            _ => Ok(None),
        },
    }
}

#[cfg(test)]
mod tests;
