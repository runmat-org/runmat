use super::*;

pub(super) fn resolve_interpolated_points(
    spec: &ObservationSpec,
    field_domain: RecoveredFieldEntityDomain,
    topology: &StructuralObservationTopology,
    maximum: usize,
) -> Result<ResolvedObservationTarget, ObservationError> {
    if spec.source_location != FieldLocation::Node
        || spec.output_location != FieldLocation::Point
        || field_domain != RecoveredFieldEntityDomain::StructuralNodes
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
    require_selected(positions.len(), maximum)?;
    let projection = topology
        .solid_projection
        .as_ref()
        .ok_or(ObservationError::Unsupported(
            "point and path interpolation require admitted solid topology",
        ))?;
    let mut samples = reserved(positions.len())?;
    for (ordinal, position) in positions.iter().enumerate() {
        let location = projection
            .locate_volume_point(position.meters)
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
    topology: &StructuralObservationTopology,
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
            let mut matches = topology
                .nodes
                .iter()
                .enumerate()
                .filter(|(_, node)| node.coordinates_m == position.meters);
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
        FieldLocation::Element => {
            let elements = topology.elements(field_domain)?;
            let mut matches = elements
                .iter()
                .enumerate()
                .filter(|(_, element)| element.centroid_m == position.meters);
            match (matches.next(), matches.next()) {
                (Some((index, _)), None) => elements_target(
                    selected_indices(std::iter::once(index), maximum)?,
                    field_domain,
                    topology,
                    maximum,
                ),
                (None, _) => Err(target_error(
                    spec,
                    "point does not exactly coincide with an admitted element centroid",
                )),
                _ => Err(target_error(
                    spec,
                    "point matches multiple element centroids",
                )),
            }
        }
        _ => Err(ObservationError::Unsupported(
            "this evaluator accepts only nodal and element probes",
        )),
    }
}
