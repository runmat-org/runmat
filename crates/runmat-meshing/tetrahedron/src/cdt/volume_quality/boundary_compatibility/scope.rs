//! Classify every way a future four-corner cell can gain a different metric.
//! New interior points have no PLC provenance. Node/single-endpoint activators
//! remain indeterminate; two-extra-corner activators are finite exact cell cases.
use super::*;
use runmat_meshing_size::metric::MetricContributionScope;
pub(super) fn common_metric(
    topology: &DelaunayVolumeTopology,
    prepared: &PreparedLocalQuality<'_>,
    candidate: &Candidate,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<bool, DelaunayVolumeQualityError> {
    let edge = candidate
        .edge
        .map(|index| topology.nodes[index as usize].identity);
    let context = crate::cdt::volume_provenance::derive_metric_context_for_corners(
        [edge[0], edge[1], StableDigest::ZERO, StableDigest::ZERO],
        candidate.region.clone(),
        prepared.provenance(),
    );
    let mut common_input = None;
    for contribution in &prepared.request().contributions {
        let active = match &contribution.scope {
            MetricContributionScope::Region { region_id } => *region_id == candidate.region,
            MetricContributionScope::Entity { entity_id } => {
                context.incident_entity_ids.contains(entity_id)
            }
        };
        if active {
            if common_input.is_some_and(|metric| metric != contribution.metric) {
                return Ok(false);
            }
            common_input = Some(contribution.metric);
        }
    }
    // Exact duplicate source matrices cannot change a most-restrictive
    // intersection. Source occurrence counts are not density multipliers.
    let mut noncommon = prepared
        .request()
        .contributions
        .iter()
        .filter_map(|contribution| {
            if Some(contribution.metric) == common_input {
                return None;
            }
            match &contribution.scope {
                MetricContributionScope::Entity { entity_id } => Some(entity_id),
                _ => None,
            }
        })
        .collect::<Vec<_>>();
    noncommon.sort_unstable();
    noncommon.dedup();
    let activates = |entities: &[PersistentEntityId]| {
        entities
            .iter()
            .any(|entity| noncommon.binary_search(&entity).is_ok())
    };
    for binding in &prepared.provenance().nodes {
        checkpoint(cancellation)?;
        if activates(&binding.entity_ids) {
            return Ok(false);
        }
    }
    for binding in &prepared.provenance().segments {
        checkpoint(cancellation)?;
        if !activates(&binding.entity_ids) {
            continue;
        }
        if binding.node_identities.iter().any(|id| edge.contains(id)) {
            return Ok(false);
        }
        if !excluded(
            topology,
            prepared,
            candidate,
            [
                edge[0],
                edge[1],
                binding.node_identities[0],
                binding.node_identities[1],
            ],
        )? {
            return Ok(false);
        }
    }
    for binding in &prepared.provenance().facets {
        checkpoint(cancellation)?;
        if !activates(&binding.entity_ids) {
            continue;
        }
        let extra = binding
            .node_identities
            .into_iter()
            .filter(|id| !edge.contains(id))
            .collect::<Vec<_>>();
        match extra.as_slice() {
            [a, b] => {
                if !excluded(topology, prepared, candidate, [edge[0], edge[1], *a, *b])? {
                    return Ok(false);
                }
            }
            [_] | [] => return Ok(false),
            _ => {}
        }
    }
    Ok(true)
}
fn excluded(
    topology: &DelaunayVolumeTopology,
    prepared: &PreparedLocalQuality<'_>,
    candidate: &Candidate,
    corners: [StableDigest; 4],
) -> Result<bool, DelaunayVolumeQualityError> {
    let indices = corners.map(|id| topology.node_index(id));
    let [Some(a), Some(b), Some(c), Some(d)] = indices else {
        return Ok(false);
    };
    let metric = prepared.metric_for_corners(corners, &candidate.region)?;
    let points = [a, b, c, d].map(|i| topology.nodes[i as usize].coordinates_m.canonical_value());
    Ok(math::tetrahedron_jacobian_upper_bound(points, metric)
        .is_some_and(|upper| upper < prepared.options().minimum_metric_scaled_jacobian))
}
