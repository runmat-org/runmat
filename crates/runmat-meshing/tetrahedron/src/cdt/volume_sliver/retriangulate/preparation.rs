//! Immutable vertex-star preparation, borrowed from exactly one canonical baseline.
//! Face neighbors retain the original outside-owner evidence without a whole-mesh scan.
use super::*;

pub(super) fn prepare<'a>(
    topology: &'a DelaunayVolumeTopology,
    source_index: usize,
    insertion_options: crate::cdt::DelaunayInsertionOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<PreparedVertexRemoval<'a>>, DelaunayVolumeSliverError> {
    if cancellation.is_cancelled() {
        return Err(error(
            DelaunayVolumeSliverErrorKind::Cancelled,
            "vertex-star preparation cancelled",
        ));
    }
    if source_index >= topology.nodes.len() {
        return Ok(None);
    }
    let selected = topology.incidence.vertex_stars[source_index]
        .iter()
        .map(|index| *index as usize)
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Ok(None);
    }
    let regions = selected
        .iter()
        .filter_map(|index| topology.tetrahedra[*index].region_id.clone())
        .collect::<BTreeSet<_>>();
    if regions.len() != 1 {
        return Ok(None);
    }
    let region = regions
        .into_iter()
        .next()
        .expect("one region was established");

    let (cavity_tetrahedra, local_selected) =
        local_cavity_inventory(topology, &selected).map_err(|cause| {
            error(
                DelaunayVolumeSliverErrorKind::Accounting(cause),
                "vertex-star numerical accounting failed",
            )
        })?;
    let cavity = match constrained_cavity_from_selected_tetrahedra(
        &cavity_tetrahedra,
        &local_selected,
        Vec::new(),
    ) {
        Ok(cavity) => cavity,
        Err(ConstrainedCavityExtractionError::Validation(_)) => return Ok(None),
        Err(failure) => {
            return Err(error(
                DelaunayVolumeSliverErrorKind::InvalidTopology,
                format!("interior vertex-star cavity extraction failed: {failure:?}"),
            ))
        }
    };
    let boundary_node_ids = constrained_cavity_boundary_node_ids(&cavity);
    let mut star_refill = None;
    for apex in &boundary_node_ids {
        if let Some(tetrahedra) = constrained_cavity_star_refill_for_apex(&cavity, *apex, |index| {
            topology
                .nodes
                .get(index as usize)
                .map(|node| node.coordinates_m.canonical_value())
        }) {
            star_refill = Some(tetrahedra);
            break;
        }
    }
    // The first admitted star refill owns this choice. The expensive fallible
    // fallback must run only when no star refill exists, not as an eager argument.
    let refill = match star_refill {
        Some(tetrahedra) => Some(tetrahedra),
        None => convex_delaunay_refill(
            topology,
            &cavity,
            &boundary_node_ids,
            insertion_options,
            cancellation,
        )?,
    };
    let refill_tetrahedra = match refill {
        Some(tetrahedra) => tetrahedra,
        None => return Ok(None),
    };

    Ok(Some(PreparedVertexRemoval {
        topology,
        source_index,
        insertion_options,
        selected: selected.into_iter().collect(),
        region,
        refill_tetrahedra,
    }))
}

fn local_cavity_inventory(
    topology: &DelaunayVolumeTopology,
    selected: &[usize],
) -> Result<(Vec<CavityTetrahedron>, Vec<usize>), runmat_simulation_usage::LedgerError> {
    let inventory = selected
        .iter()
        .copied()
        .chain(selected.iter().flat_map(|index| {
            topology.tetrahedra[*index]
                .neighbors
                .iter()
                .flatten()
                .map(|index| *index as usize)
        }))
        .collect::<BTreeSet<_>>();
    let mut local_selected = Vec::with_capacity(selected.len());
    let selected_set = selected.iter().copied().collect::<BTreeSet<_>>();
    let tetrahedra = inventory
        .into_iter()
        .enumerate()
        .map(|(local_index, index)| {
            if selected_set.contains(&index) {
                local_selected.push(local_index);
            }
            let tetrahedron = &topology.tetrahedra[index];
            let points = tetrahedron.vertex_indices.map(|vertex| {
                topology.nodes[vertex as usize]
                    .coordinates_m
                    .canonical_value()
            });
            Ok(CavityTetrahedron {
                tetrahedron_id: index as u32,
                component_id: 0,
                node_ids: tetrahedron.vertex_indices,
                source_surface_element_id: 0,
                region_ids: tetrahedron
                    .region_id
                    .iter()
                    .map(|region| region.source_topology_id.clone())
                    .collect(),
                volume_m3: tetrahedron_volume(points),
                aspect_ratio: tetrahedron_edge_aspect_ratio(points),
                exact_scaled_jacobian: tetrahedron_scaled_jacobian(points)?,
            })
        })
        .collect::<Result<Vec<_>, runmat_simulation_usage::LedgerError>>()?;
    Ok((tetrahedra, local_selected))
}

#[cfg(test)]
mod tests;
