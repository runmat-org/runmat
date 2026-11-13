//! Metric-aware, target-sized protected-face proposals. Acute corners need
//! offcenters that centroid-normal probes cannot represent. These are proposals,
//! never permission to relax protected geometry or full current quality.
use super::*;
#[cfg(test)]
mod tests;

pub(super) fn choices(
    topology: &(impl VolumeTopologyRead + ?Sized),
    source: &DelaunayTetrahedronQuality,
    protected: &BTreeSet<[StableDigest; 3]>,
    target: f64,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Vec<EdgeStarChoice>, DelaunayVolumeSliverError> {
    let vertices = source
        .node_identities
        .map(|id| topology.node_index(id).expect("source corner"));
    let cell = topology.cell(topology.tetrahedron_index(vertices).expect("source cell"));
    let Some(neighbor_index) = cell
        .neighbors
        .into_iter()
        .flatten()
        .find(|index| topology.cell(*index).region_id == cell.region_id)
    else {
        return Ok(Vec::new());
    };
    let mut result = Vec::new();
    for opposite in 0..4 {
        let mut face = vertices
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| (i != opposite).then_some(v))
            .collect::<Vec<_>>();
        face.sort_unstable_by_key(|v| topology.node(*v).identity);
        let face: [u32; 3] = face.try_into().expect("three corners");
        let ids = face.map(|v| topology.node(v).identity);
        if !protected.contains(&ids) {
            continue;
        }
        let points = face.map(|v| topology.node(v).coordinates_m.canonical_value());
        for (index, (point, score)) in super::super::offcenter::points_for_face(
            points,
            topology
                .node(vertices[opposite])
                .coordinates_m
                .canonical_value(),
            source.resolved_metric,
            target,
            cancellation,
        )?
        .into_iter()
        .enumerate()
        {
            result.push(EdgeStarChoice {
                edge_identities: [ids[0], ids[1]],
                index: 30_000 + (opposite * 32 + index) as u16,
                coordinates_m: runmat_types::physical::Quantity::<
                    runmat_types::physical::kind::Position,
                    _,
                >::new(point),
                construction: Point3Construction::Explicit,
                proposal_score: score,
                strategy: InsertionStrategy::BoundaryOffcenter { neighbor_index },
            });
        }
    }
    result.truncate(MAXIMUM_CAVITY_SEARCH_DIRECTIONS);
    Ok(result)
}
