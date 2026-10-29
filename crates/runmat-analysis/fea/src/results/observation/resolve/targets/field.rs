use super::*;

pub(super) fn resolve_field_target(
    spec: &ObservationSpec,
    field_domain: RecoveredFieldEntityDomain,
    topology: &ObservationTopology,
    limits: ObservationLimits,
) -> Result<ResolvedObservationTarget, ObservationError> {
    let maximum = limits.maximum_selected_entities;
    if matches!(spec.target, ObservationTarget::Path { .. })
        || spec.output_location == FieldLocation::Point
    {
        return resolve_interpolated_points(spec, field_domain, topology, limits);
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
                FieldLocation::Element
                | FieldLocation::Cell
                | FieldLocation::Face
                | FieldLocation::IntegrationPoint
                | FieldLocation::Edge => {
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
                        topology.nodes.len(),
                        limits,
                    )?;
                    nodes_target(indices, topology, maximum)
                }
                // A volume-element field on a boundary region is the set of elements that own the
                // region's faces (the field's surface trace), used by selections and extrema.
                (EntityDimension::Face, ExpectedTopology::BoundaryFace, FieldLocation::Element)
                    if field_domain == RecoveredFieldEntityDomain::VolumeElements =>
                {
                    let projection = solid_projection(topology, spec)?;
                    let faces = matching_faces(region, projection);
                    enforce_cardinality(region.cardinality, faces.clone().count(), spec)?;
                    let mut owners = std::collections::BTreeSet::new();
                    for index in faces {
                        let [owner] = projection.boundary_faces[index]
                            .adjacent_element_indices
                            .as_slice()
                        else {
                            return Err(target_error(
                                spec,
                                "element fields on a boundary need faces with one owning element",
                            ));
                        };
                        owners.insert(*owner);
                        if owners.len() > maximum {
                            return Err(ObservationError::Limit("selected topology entities"));
                        }
                    }
                    // Volume-element field indices follow the projection's element order.
                    elements_target(
                        owners.into_iter().collect(),
                        field_domain,
                        topology,
                        maximum,
                    )
                }
                (EntityDimension::Face, ExpectedTopology::BoundaryFace, FieldLocation::Face) => {
                    if field_domain != RecoveredFieldEntityDomain::OrientedFaces
                        || topology.finite_volume.is_none()
                    {
                        return Err(target_error(
                            spec,
                            "face fields require exact all-face topology",
                        ));
                    }
                    let projection = solid_projection(topology, spec)?;
                    require_workspace(
                        face_selection_resident_bytes(region, projection, topology, limits)?,
                        limits,
                    )?;
                    validate_exact_coverage(
                        region,
                        projection
                            .boundary_faces
                            .iter()
                            .flat_map(|f| f.provenance.iter()),
                    )?;
                    let matches = matching_faces(region, projection);
                    enforce_cardinality(region.cardinality, matches.clone().count(), spec)?;
                    let faces = crate::topology::storage::SortedKeys::from_entries(
                        selected_indices(matches, usize::MAX)?,
                    )
                    .map_err(storage_error)?;
                    let indices = topology
                        .face_boundary_sources
                        .iter()
                        .enumerate()
                        .filter_map(|(i, source)| source.filter(|s| faces.contains(s)).map(|_| i));
                    elements_target(
                        selected_indices(indices, maximum)?,
                        field_domain,
                        topology,
                        maximum,
                    )
                }
                (
                    EntityDimension::Volume,
                    ExpectedTopology::MeshElement | ExpectedTopology::Cell,
                    FieldLocation::Element | FieldLocation::Cell,
                ) => {
                    if region.expected_topology == ExpectedTopology::Cell
                        && topology.finite_volume.is_none()
                    {
                        return Err(target_error(
                            spec,
                            "Cell selection requires exact finite-volume topology",
                        ));
                    }
                    let indices = matching_elements(
                        region,
                        field_domain,
                        topology,
                        limits,
                        selection_output_bytes(topology.elements(field_domain)?.len(), maximum)?,
                    )?;
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
                        RecoveredFieldEntityDomain::VolumeElements,
                        topology,
                        limits,
                        selection_output_bytes(topology.volume_integration_point_count(), maximum)?,
                    )?;
                    enforce_cardinality(region.cardinality, parents.clone().count(), spec)?;
                    elements_target(
                        topology.integration_indices(parents, maximum)?,
                        field_domain,
                        topology,
                        maximum,
                    )
                }
                (EntityDimension::Volume, ExpectedTopology::MeshElement, FieldLocation::Node) => {
                    let projection = solid_projection(topology, spec)?;
                    let elements = matching_elements(
                        region,
                        RecoveredFieldEntityDomain::VolumeElements,
                        topology,
                        limits,
                        node_union_resident_bytes(topology.nodes.len(), maximum)?,
                    )?;
                    enforce_cardinality(region.cardinality, elements.clone().count(), spec)?;
                    let indices = unique_nodes(
                        elements.flat_map(|index| {
                            projection.elements[index].node_indices.iter().copied()
                        }),
                        topology.nodes.len(),
                        limits,
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
