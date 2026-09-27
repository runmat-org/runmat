use super::*;

pub(super) fn resolve_field_target(
    spec: &ObservationSpec,
    field_domain: RecoveredFieldEntityDomain,
    topology: &StructuralObservationTopology,
    maximum: usize,
) -> Result<ResolvedObservationTarget, ObservationError> {
    if matches!(spec.target, ObservationTarget::Path { .. })
        || spec.output_location == FieldLocation::Point
    {
        return resolve_interpolated_points(spec, field_domain, topology, maximum);
    }
    match &spec.target {
        ObservationTarget::WholeModel
        | ObservationTarget::Spatial(ExcitationTarget::WholeModel) => {
            match materialized_field_location(spec) {
                FieldLocation::Node => nodes_target(
                    selected_indices(0..topology.nodes.len(), maximum)?,
                    topology,
                    maximum,
                ),
                FieldLocation::Element | FieldLocation::IntegrationPoint => {
                    let elements = topology.elements(field_domain)?;
                    elements_target(
                        selected_indices(0..elements.len(), maximum)?,
                        field_domain,
                        topology,
                        maximum,
                    )
                }
                _ => Err(ObservationError::Unsupported("unsupported field location")),
            }
        }
        ObservationTarget::Spatial(ExcitationTarget::Region(region)) => {
            match (
                region.entity_dimension,
                region.expected_topology,
                materialized_field_location(spec),
            ) {
                (EntityDimension::Point, ExpectedTopology::MeshNode, FieldLocation::Node) => {
                    let id = parse_node_selector(&region.region_id, spec)?;
                    let index = topology.node_index(id)?.ok_or_else(|| {
                        target_error(spec, "structural-node selector does not resolve")
                    })?;
                    enforce_cardinality(region.cardinality, 1, spec)?;
                    nodes_target(
                        selected_indices(std::iter::once(index), maximum)?,
                        topology,
                        maximum,
                    )
                }
                (EntityDimension::Face, ExpectedTopology::BoundaryFace, FieldLocation::Node) => {
                    let projection = solid_projection(topology, spec)?;
                    let faces = matching_faces(region, projection);
                    enforce_cardinality(region.cardinality, faces.clone().count(), spec)?;
                    let indices = unique_nodes(
                        faces.flat_map(|index| {
                            projection.boundary_faces[index]
                                .node_indices
                                .iter()
                                .copied()
                        }),
                        maximum,
                    )?;
                    nodes_target(indices, topology, maximum)
                }
                (
                    EntityDimension::Volume,
                    ExpectedTopology::MeshElement,
                    FieldLocation::Element,
                ) => {
                    let indices = matching_elements(region, field_domain, topology)?;
                    enforce_cardinality(region.cardinality, indices.clone().count(), spec)?;
                    elements_target(
                        selected_indices(indices, maximum)?,
                        field_domain,
                        topology,
                        maximum,
                    )
                }
                (
                    EntityDimension::Volume,
                    ExpectedTopology::MeshElement,
                    FieldLocation::IntegrationPoint,
                ) => {
                    let parents = matching_elements(
                        region,
                        RecoveredFieldEntityDomain::SolidElements,
                        topology,
                    )?;
                    enforce_cardinality(region.cardinality, parents.clone().count(), spec)?;
                    elements_target(
                        integration_indices(parents, maximum)?,
                        field_domain,
                        topology,
                        maximum,
                    )
                }
                (EntityDimension::Volume, ExpectedTopology::MeshElement, FieldLocation::Node) => {
                    let projection = solid_projection(topology, spec)?;
                    let elements = matching_elements(
                        region,
                        RecoveredFieldEntityDomain::SolidElements,
                        topology,
                    )?;
                    enforce_cardinality(region.cardinality, elements.clone().count(), spec)?;
                    let indices = unique_nodes(
                        elements.flat_map(|index| {
                            projection.elements[index].node_indices.iter().copied()
                        }),
                        maximum,
                    )?;
                    nodes_target(indices, topology, maximum)
                }
                _ => Err(target_error(
                    spec,
                    "target topology is incompatible with the recovered field location",
                )),
            }
        }
        _ => Err(target_error(
            spec,
            "field requests support whole-model and admitted region targets",
        )),
    }
}
