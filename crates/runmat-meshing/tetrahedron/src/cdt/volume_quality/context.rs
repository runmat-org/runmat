use runmat_geometry_core::PersistentEntityId;
use runmat_meshing_core::StableDigest;

use super::{
    error, DelaunayVolumeQualityError, DelaunayVolumeQualityErrorKind, DelaunayVolumeTopology,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelaunayVolumeMetricContext {
    pub tetrahedron_node_identities: [StableDigest; 4],
    pub incident_entity_ids: Vec<PersistentEntityId>,
}

pub(super) fn validate_metric_contexts(
    topology: &DelaunayVolumeTopology,
    contexts: &[DelaunayVolumeMetricContext],
) -> Result<(), DelaunayVolumeQualityError> {
    if contexts.len() != topology.tetrahedra.len() {
        return Err(invalid(
            None,
            "metric context count must equal the tetrahedron count",
        ));
    }
    for (index, (tetrahedron, context)) in topology.tetrahedra.iter().zip(contexts).enumerate() {
        let identities = tetrahedron
            .vertex_indices
            .map(|vertex| topology.nodes[vertex as usize].identity);
        if context.tetrahedron_node_identities != identities {
            return Err(invalid(
                Some(index),
                "metric context must bind the tetrahedron's stable node identities",
            ));
        }
        if context.incident_entity_ids.is_empty()
            || context
                .incident_entity_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid(
                Some(index),
                "incident metric entities must be nonempty, unique, and canonically ordered",
            ));
        }
        for entity_id in &context.incident_entity_ids {
            entity_id.validate().map_err(|failure| {
                invalid(
                    Some(index),
                    format!(
                        "incident metric entity is invalid at {}: {}",
                        failure.field, failure.reason
                    ),
                )
            })?;
        }
        let region_id = tetrahedron.region_id.as_ref().ok_or_else(|| {
            invalid(
                Some(index),
                "metric context requires an assigned tetrahedron region",
            )
        })?;
        if context
            .incident_entity_ids
            .binary_search(region_id)
            .is_err()
        {
            return Err(invalid(
                Some(index),
                "metric context must include the assigned persistent region",
            ));
        }
    }
    Ok(())
}

fn invalid(index: Option<usize>, reason: impl Into<String>) -> DelaunayVolumeQualityError {
    error(
        DelaunayVolumeQualityErrorKind::InvalidMetricContext,
        index,
        reason,
    )
}
