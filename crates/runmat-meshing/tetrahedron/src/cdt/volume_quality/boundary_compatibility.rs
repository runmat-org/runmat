//! Sufficient fixed-PLC incompatibility evidence, never a feasibility claim.
//! Only one geometrically strongest candidate receives a full provenance audit;
//! this avoids rescanning the complete provenance inventory per boundary edge.
use super::working::PreparedLocalQuality;
use super::*;
use crate::cdt::{
    insertion::topology_access::TetrahedralTopology, topology::orient_materialized_volume_nodes,
};
use runmat_meshing_core::quality::predicate::PredicateSign;
mod interval;
mod math;
mod scope;
#[derive(Debug)]
pub(in crate::cdt) struct BoundaryMetricConflict {
    pub(in crate::cdt) edge: [StableDigest; 2],
    pub(in crate::cdt) jacobian_upper_bound: f64,
}
struct Candidate {
    edge: [u32; 2],
    region: PersistentEntityId,
    upper: f64,
}
pub(in crate::cdt) fn certify(
    topology: &DelaunayVolumeTopology,
    prepared: &PreparedLocalQuality<'_>,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<Option<BoundaryMetricConflict>, DelaunayVolumeQualityError> {
    checkpoint(cancellation)?;
    let target = prepared.options().minimum_metric_scaled_jacobian;
    let mut best: Option<Candidate> = None;
    for segment in &prepared.provenance().segments {
        checkpoint(cancellation)?;
        let Some(candidate) = candidate(topology, prepared, segment.node_identities)? else {
            continue;
        };
        if candidate.upper < target && best.as_ref().is_none_or(|old| candidate.upper < old.upper) {
            best = Some(candidate);
        }
    }
    let Some(candidate) = best else {
        return Ok(None);
    };
    if !scope::common_metric(topology, prepared, &candidate, cancellation)? {
        return Ok(None);
    }
    Ok(Some(BoundaryMetricConflict {
        edge: candidate
            .edge
            .map(|index| topology.nodes[index as usize].identity),
        jacobian_upper_bound: candidate.upper,
    }))
}
fn checkpoint(
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayVolumeQualityError> {
    if cancellation.is_cancelled() {
        Err(error(
            DelaunayVolumeQualityErrorKind::Cancelled,
            None,
            "boundary metric compatibility cancelled",
        ))
    } else {
        Ok(())
    }
}
fn candidate(
    topology: &DelaunayVolumeTopology,
    prepared: &PreparedLocalQuality<'_>,
    ids: [StableDigest; 2],
) -> Result<Option<Candidate>, DelaunayVolumeQualityError> {
    let [Some(a), Some(b)] = ids.map(|id| topology.node_index(id)) else {
        return Ok(None);
    };
    let edge = [a, b];
    let star = edge_star(topology, edge);
    let Some(first) = star.clone().next() else {
        return Ok(None);
    };
    let Some(region) = topology.tetrahedra[first as usize].region_id.clone() else {
        return Ok(None);
    };
    let mut opposite = [0; 2];
    let mut opposite_count = 0;
    for index in star.clone() {
        let cell = &topology.tetrahedra[index as usize];
        if cell.region_id.as_ref() != Some(&region) {
            return Ok(None);
        }
        for slot in 0..4 {
            if cell.neighbors[slot].is_some() || edge.contains(&cell.vertex_indices[slot]) {
                continue;
            }
            let face = std::array::from_fn::<_, 3, _>(|offset| {
                cell.vertex_indices[(slot + offset + 1) % 4]
            });
            if !edge.iter().all(|vertex| face.contains(vertex)) {
                continue;
            }
            let mut key = face.map(|index| topology.nodes[index as usize].identity);
            key.sort_unstable();
            if !prepared.protected_facets().contains(&key) {
                return Ok(None);
            }
            if opposite_count >= opposite.len() {
                return Ok(None);
            }
            opposite[opposite_count] = face
                .into_iter()
                .find(|vertex| !edge.contains(vertex))
                .expect("face contains one opposite corner");
            opposite_count += 1;
        }
    }
    if opposite_count != 2 {
        return Ok(None);
    }
    let metric = prepared.metric_for_corners(
        [ids[0], ids[1], StableDigest::ZERO, StableDigest::ZERO],
        &region,
    )?;
    let points = [a, b, opposite[0], opposite[1]]
        .map(|i| topology.nodes[i as usize].coordinates_m.canonical_value());
    let Some(upper) = math::edge_star_upper_bound(points, metric) else {
        return Ok(None);
    };
    let [na, nb, p, q] = [a, b, opposite[0], opposite[1]].map(|i| topology.nodes[i as usize]);
    let sign = orient_materialized_volume_nodes([na, nb, p, q]).map_err(predicate_error)?;
    if sign == PredicateSign::Zero {
        return Ok(None);
    }
    // The present valid edge star fixes which wedge is the solid's interior.
    // All its radial vertices must be in the same acute sector as the endpoints.
    for index in star {
        for vertex in topology.tetrahedra[index as usize].vertex_indices {
            if [a, b, opposite[0], opposite[1]].contains(&vertex) {
                continue;
            }
            let r = topology.nodes[vertex as usize];
            for nodes in [[na, nb, p, r], [na, nb, r, q]] {
                let observed = orient_materialized_volume_nodes(nodes).map_err(predicate_error)?;
                if observed != PredicateSign::Zero && observed != sign {
                    return Ok(None);
                }
            }
        }
    }
    Ok(Some(Candidate {
        edge,
        region,
        upper,
    }))
}
/// Borrow the maintained incidence instead of cloning a potentially large star.
fn edge_star(
    topology: &DelaunayVolumeTopology,
    edge: [u32; 2],
) -> impl Iterator<Item = u32> + Clone + '_ {
    let stars = &topology.incidence.vertex_stars;
    let [first, second] = if stars[edge[0] as usize].len() <= stars[edge[1] as usize].len() {
        edge
    } else {
        [edge[1], edge[0]]
    };
    stars[first as usize].iter().copied().filter(move |index| {
        topology.tetrahedra[*index as usize]
            .vertex_indices
            .contains(&second)
    })
}

fn predicate_error(failure: impl std::fmt::Debug) -> DelaunayVolumeQualityError {
    error(
        DelaunayVolumeQualityErrorKind::InvalidTopology,
        None,
        format!("boundary metric wedge predicate failed: {failure:?}"),
    )
}
