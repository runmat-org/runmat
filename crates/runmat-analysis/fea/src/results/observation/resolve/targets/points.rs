use super::*;

pub(super) fn resolve_interpolated_points(
    spec: &ObservationSpec,
    field_domain: RecoveredFieldEntityDomain,
    topology: &ObservationTopology,
    limits: ObservationLimits,
) -> Result<ResolvedObservationTarget, ObservationError> {
    if spec.source_location != FieldLocation::Node
        || spec.output_location != FieldLocation::Point
        || field_domain != RecoveredFieldEntityDomain::Nodes
    {
        return Err(target_error(
            spec,
            "finite-element point interpolation requires a nodal structural field and point output",
        ));
    }
    let positions = match &spec.target {
        ObservationTarget::Point { position } => std::slice::from_ref(position),
        ObservationTarget::Path { positions } => positions.as_slice(),
        _ => {
            return Err(target_error(
                spec,
                "point interpolation requires a point or path target",
            ))
        }
    };
    require_selected(positions.len(), limits.maximum_selected_entities)?;
    let projection = topology
        .solid_projection
        .as_ref()
        .ok_or(ObservationError::Unsupported(
            "point and path interpolation require admitted solid topology",
        ))?;
    let bytes = resources::interpolated_points_resident_bytes(projection, positions.len())?;
    if bytes > limits.maximum_peak_resident_bytes {
        return Err(ObservationError::Limit(
            "point target construction residency",
        ));
    }
    let mut samples = reserved(positions.len())?;
    for (ordinal, position) in positions.iter().enumerate() {
        let location = projection
            .locate_volume_point(position.canonical_value())?
            .ok_or_else(|| target_error(spec, "point lies outside the admitted solid topology"))?;
        let ordinal =
            u64::try_from(ordinal).map_err(|_| ObservationError::Limit("path sample ordinal"))?;
        let mut sample = ResolvedObservationPoint {
            identity: StableDigest::ZERO,
            ordinal,
            position_m: location.position_m,
            source_element_index: location.element_index,
            source_element_identity: location.element_identity,
            source_node_indices: location.node_indices,
            source_node_identities: location.node_identities,
            shape_values: location.shape_values,
        };
        sample.identity = derive_resolved_point_identity(topology.digest(), &sample)?;
        samples.push(sample);
    }
    Ok(ResolvedObservationTarget::Points { samples })
}

pub(super) fn resolve_probe(
    spec: &ObservationSpec,
    field_domain: RecoveredFieldEntityDomain,
    topology: &ObservationTopology,
    maximum: usize,
) -> Result<ResolvedObservationTarget, ObservationError> {
    let ObservationTarget::Point { position } = spec.target else {
        return Err(target_error(
            spec,
            "probe requests require an exact point target",
        ));
    };
    match materialized_field_location(spec) {
        FieldLocation::Node => {
            let mut matches = topology.nodes.iter().enumerate().filter(|(_, node)| {
                node.coordinates_m.canonical_value() == position.canonical_value()
            });
            match (matches.next(), matches.next()) {
                (Some((index, _)), None) => nodes_target(
                    selected_indices(std::iter::once(index), maximum)?,
                    topology,
                    maximum,
                ),
                (None, _) => Err(target_error(
                    spec,
                    "point does not exactly coincide with an admitted mesh node",
                )),
                _ => Err(target_error(
                    spec,
                    "point matches multiple admitted mesh nodes",
                )),
            }
        }
        FieldLocation::Element | FieldLocation::Cell => {
            let elements = topology.elements(field_domain)?;
            // A point at an element's centroid names that element even outside a solid volume
            // (sampled and face element domains).
            let mut exact = elements
                .iter()
                .enumerate()
                .filter(|(_, element)| element.centroid_m == position.canonical_value());
            if let (Some((index, _)), None) = (exact.next(), exact.next()) {
                return elements_target(
                    selected_indices(std::iter::once(index), maximum)?,
                    field_domain,
                    topology,
                    maximum,
                );
            }
            // Otherwise a point samples the native value of the volume element containing it.
            let projection =
                topology
                    .solid_projection
                    .as_ref()
                    .ok_or(ObservationError::Unsupported(
                        "element point samples require admitted solid topology",
                    ))?;
            let location = projection
                .locate_volume_point(position.canonical_value())?
                .ok_or_else(|| {
                    target_error(spec, "point lies outside the admitted solid topology")
                })?;
            let index = elements
                .iter()
                .position(|element| element.identity == location.element_identity)
                .ok_or_else(|| {
                    target_error(spec, "containing element is absent from the field domain")
                })?;
            elements_target(
                selected_indices(std::iter::once(index), maximum)?,
                field_domain,
                topology,
                maximum,
            )
        }
        _ => Err(ObservationError::Unsupported(
            "this evaluator accepts only nodal, element and cell probes",
        )),
    }
}

mod resources;
