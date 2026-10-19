//! Verify complete vertex and region incidence against borrowed canonical cells.
use super::*;

pub(super) fn validate_incidence(
    topology: &DelaunayVolumeTopology,
    options: DelaunayTopologyOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayTopologyError> {
    let incidence = &topology.incidence;
    let invalid = || {
        error(
            DelaunayTopologyErrorKind::InvalidTetrahedron,
            "vertex or region incidence differs from canonical topology",
        )
    };
    if incidence.vertex_stars.len() != topology.nodes.len() {
        return Err(invalid());
    }
    for (vertex, star) in incidence.vertex_stars.iter().enumerate() {
        checkpoint(vertex as u64, options, cancellation)?;
        if star.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(invalid());
        }
        for (offset, index) in star.iter().enumerate() {
            checkpoint(offset as u64, options, cancellation)?;
            if !topology
                .tetrahedra
                .get(*index as usize)
                .is_some_and(|cell| cell.vertex_indices.contains(&(vertex as u32)))
            {
                return Err(invalid());
            }
        }
    }
    for (index, region) in incidence.regions.iter().enumerate() {
        checkpoint(index as u64, options, cancellation)?;
        super::super::super::incidence::validate_region(&region.region_id)?;
        if index > 0 && incidence.regions[index - 1].region_id >= region.region_id
            || region.tetrahedron_indices.is_empty()
        {
            return Err(invalid());
        }
    }
    let mut cursors = vec![0; incidence.regions.len()];
    let mut unassigned = 0;
    for (index, cell) in topology.tetrahedra.iter().enumerate() {
        checkpoint(index as u64, options, cancellation)?;
        for vertex in cell.vertex_indices {
            if incidence.vertex_stars[vertex as usize]
                .binary_search(&(index as u32))
                .is_err()
            {
                return Err(invalid());
            }
        }
        match &cell.region_id {
            Some(identity) => {
                let region = incidence
                    .regions
                    .binary_search_by_key(&identity, |region| &region.region_id)
                    .map_err(|_| invalid())?;
                if incidence.regions[region]
                    .tetrahedron_indices
                    .get(cursors[region])
                    != Some(&(index as u32))
                {
                    return Err(invalid());
                }
                cursors[region] += 1;
            }
            None => {
                if incidence.unassigned_tetrahedron_indices.get(unassigned) != Some(&(index as u32))
                {
                    return Err(invalid());
                }
                unassigned += 1;
            }
        }
    }
    if unassigned != incidence.unassigned_tetrahedron_indices.len()
        || cursors
            .iter()
            .zip(&incidence.regions)
            .any(|(count, region)| *count != region.tetrahedron_indices.len())
    {
        return Err(invalid());
    }
    Ok(())
}
