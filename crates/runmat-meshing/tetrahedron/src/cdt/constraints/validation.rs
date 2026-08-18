use std::collections::BTreeSet;

use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use super::{
    checkpoint, error, resource, sorted_segment, validate_entity_id, validate_options,
    validate_token, DelaunayConstraintError, DelaunayConstraintErrorKind,
    DelaunayConstraintOptions, DelaunayConstraints,
};

pub fn validate_delaunay_constraints(
    constraints: &DelaunayConstraints,
    options: DelaunayConstraintOptions,
    cancellation: &dyn MeshingCancellationSignal,
) -> Result<(), DelaunayConstraintError> {
    validate_options(options)?;
    if constraints.nodes.is_empty()
        || constraints.nodes.len() as u64 > options.maximum_nodes
        || constraints.segments.len() as u64 > options.maximum_segments
        || constraints.facets.len() as u64 > options.maximum_facets
    {
        return Err(resource(
            "constraint inventory is empty or exceeds its hard limits",
        ));
    }
    let mut source_nodes = BTreeSet::new();
    let mut positions = BTreeSet::new();
    for (index, node) in constraints.nodes.iter().enumerate() {
        checkpoint(index, options, cancellation)?;
        validate_entity_id(&node.source_node_id)?;
        if node.identity == StableDigest::ZERO
            || node.coordinates_m.iter().any(|value| !value.is_finite())
            || index > 0 && constraints.nodes[index - 1].identity >= node.identity
            || !source_nodes.insert(node.source_node_id.clone())
            || !positions.insert(node.coordinates_m.map(coordinate_bits))
        {
            return Err(error(
                DelaunayConstraintErrorKind::InvalidIdentity,
                "constraint nodes require ordered unique identities, source nodes, and finite positions",
            ));
        }
    }

    let mut previous_segment = None;
    for (index, segment) in constraints.segments.iter().enumerate() {
        checkpoint(index, options, cancellation)?;
        if segment.vertex_indices != sorted_segment(segment.vertex_indices)
            || segment.vertex_indices[0] == segment.vertex_indices[1]
            || segment
                .vertex_indices
                .iter()
                .any(|vertex| *vertex as usize >= constraints.nodes.len())
            || previous_segment.is_some_and(|previous| previous >= segment.vertex_indices)
            || segment.protected_edge_id.is_some() != segment.source_edge_id.is_some()
        {
            return Err(error(
                DelaunayConstraintErrorKind::InvalidPlc,
                "segments must be unique, ordered, in range, and carry complete provenance",
            ));
        }
        if let Some(edge_id) = &segment.protected_edge_id {
            validate_entity_id(edge_id)?;
        }
        if let Some(source_id) = &segment.source_edge_id {
            validate_entity_id(source_id)?;
        }
        previous_segment = Some(segment.vertex_indices);
    }

    let mut facet_keys = BTreeSet::new();
    for (index, facet) in constraints.facets.iter().enumerate() {
        checkpoint(index, options, cancellation)?;
        validate_entity_id(&facet.facet_id)?;
        validate_entity_id(&facet.source_face_id)?;
        let mut key = facet.vertex_indices;
        key.sort_unstable();
        if key.windows(2).any(|pair| pair[0] == pair[1])
            || key
                .iter()
                .any(|vertex| *vertex as usize >= constraints.nodes.len())
            || !facet_keys.insert(key)
            || facet
                .material_interface_ids
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(error(
                DelaunayConstraintErrorKind::InvalidPlc,
                "facets must be unique, in range, oriented, and carry ordered interface identity",
            ));
        }
        for interface_id in &facet.material_interface_ids {
            validate_token("material interface", interface_id)?;
        }
    }
    Ok(())
}

fn coordinate_bits(value: f64) -> u64 {
    if value == 0.0 {
        0
    } else {
        value.to_bits()
    }
}
