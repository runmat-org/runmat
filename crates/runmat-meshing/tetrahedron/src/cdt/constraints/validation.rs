use std::collections::BTreeSet;

use runmat_geometry_core::PersistentEntityKind;
use runmat_meshing_core::{MeshingCancellationSignal, StableDigest};

use super::{
    checkpoint, error, resource, sorted_segment, validate_options, DelaunayConstraintError,
    DelaunayConstraintErrorKind, DelaunayConstraintFacetSide, DelaunayConstraintOptions,
    DelaunayConstraints,
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
    let mut source_vertices = BTreeSet::new();
    for (index, node) in constraints.nodes.iter().enumerate() {
        checkpoint(index, options, cancellation)?;
        if node.identity == StableDigest::ZERO
            || node.coordinates_m.iter().any(|value| !value.is_finite())
            || index > 0 && constraints.nodes[index - 1].identity >= node.identity
        {
            return Err(invalid_identity(
                "constraint nodes require ordered unique nonzero identities and finite positions",
            ));
        }
        if let Some(vertex) = &node.source_vertex_id {
            validate_kind(vertex, PersistentEntityKind::Vertex)?;
            if !source_vertices.insert(vertex.clone()) {
                return Err(invalid_identity(
                    "one persistent vertex must bind exactly one constraint node",
                ));
            }
        }
    }

    let mut segment_keys = BTreeSet::new();
    for (index, segment) in constraints.segments.iter().enumerate() {
        checkpoint(index, options, cancellation)?;
        if segment.vertex_indices != sorted_segment(segment.vertex_indices)
            || segment.vertex_indices[0] == segment.vertex_indices[1]
            || segment
                .vertex_indices
                .iter()
                .any(|vertex| *vertex as usize >= constraints.nodes.len())
            || !segment_keys.insert(segment.vertex_indices)
        {
            return Err(invalid_boundary(
                "segments must be canonical, unique, distinct, and in range",
            ));
        }
        if let Some(edge) = &segment.source_edge_id {
            validate_kind(edge, PersistentEntityKind::Edge)?;
        }
    }
    if constraints
        .segments
        .windows(2)
        .any(|pair| pair[0].vertex_indices >= pair[1].vertex_indices)
    {
        return Err(invalid_boundary(
            "constraint segments must be strictly ordered",
        ));
    }

    let mut facet_keys = BTreeSet::new();
    let mut facet_ids = BTreeSet::new();
    let mut required_segments = BTreeSet::new();
    let mut previous_key = None;
    for (index, facet) in constraints.facets.iter().enumerate() {
        checkpoint(index, options, cancellation)?;
        validate_kind(&facet.source_face_id, PersistentEntityKind::Face)?;
        if facet.facet_id == StableDigest::ZERO || !facet_ids.insert(facet.facet_id) {
            return Err(invalid_identity(
                "constraint facet identities must be nonzero and unique",
            ));
        }
        let mut key = facet.vertex_indices;
        key.sort_unstable();
        if key.windows(2).any(|pair| pair[0] == pair[1])
            || key
                .iter()
                .any(|vertex| *vertex as usize >= constraints.nodes.len())
            || !facet_keys.insert(key)
            || previous_key.is_some_and(|previous| previous >= (key, facet.facet_id))
        {
            return Err(invalid_boundary(
                "facets must be canonical, unique, oriented, and in range",
            ));
        }
        previous_key = Some((key, facet.facet_id));
        validate_sides(&facet.positive_side, &facet.negative_side)?;
        if facet.contact_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(invalid_identity(
                "facet contact identities must be canonical and unique",
            ));
        }
        for contact in &facet.contact_ids {
            validate_kind(contact, PersistentEntityKind::Contact)?;
        }
        for edge in 0..3 {
            required_segments.insert(sorted_segment([
                facet.vertex_indices[edge],
                facet.vertex_indices[(edge + 1) % 3],
            ]));
        }
    }
    if !required_segments.is_subset(&segment_keys) {
        return Err(invalid_boundary(
            "constraint segments must cover every facet edge",
        ));
    }
    Ok(())
}

fn validate_sides(
    positive: &DelaunayConstraintFacetSide,
    negative: &DelaunayConstraintFacetSide,
) -> Result<(), DelaunayConstraintError> {
    for side in [positive, negative] {
        if let DelaunayConstraintFacetSide::Region(region) = side {
            validate_kind(region, PersistentEntityKind::Region)?;
        }
    }
    match (positive, negative) {
        (DelaunayConstraintFacetSide::Region(left), DelaunayConstraintFacetSide::Region(right))
            if left != right =>
        {
            Ok(())
        }
        (DelaunayConstraintFacetSide::Region(_), DelaunayConstraintFacetSide::Exterior)
        | (DelaunayConstraintFacetSide::Exterior, DelaunayConstraintFacetSide::Region(_))
        | (DelaunayConstraintFacetSide::Region(_), DelaunayConstraintFacetSide::Void)
        | (DelaunayConstraintFacetSide::Void, DelaunayConstraintFacetSide::Region(_)) => Ok(()),
        _ => Err(invalid_boundary(
            "facet sides must classify one region boundary or two distinct interface regions",
        )),
    }
}

fn validate_kind(
    identity: &runmat_geometry_core::PersistentEntityId,
    expected: PersistentEntityKind,
) -> Result<(), DelaunayConstraintError> {
    identity
        .validate()
        .map_err(|failure| invalid_identity(failure.to_string()))?;
    if identity.kind != expected {
        return Err(invalid_identity(
            "persistent constraint provenance has the wrong entity kind",
        ));
    }
    Ok(())
}

fn invalid_identity(reason: impl Into<String>) -> DelaunayConstraintError {
    error(DelaunayConstraintErrorKind::InvalidIdentity, reason)
}

fn invalid_boundary(reason: impl Into<String>) -> DelaunayConstraintError {
    error(DelaunayConstraintErrorKind::InvalidBoundary, reason)
}
