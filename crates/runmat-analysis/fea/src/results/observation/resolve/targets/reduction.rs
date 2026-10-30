use super::*;

pub(super) fn resolve_reduction_target(
    spec: &ObservationSpec,
    field_domain: RecoveredFieldEntityDomain,
    topology: &ObservationTopology,
    constraint_plan: Option<&CanonicalStructuralConstraintPlan>,
    limits: ObservationLimits,
) -> Result<ResolvedObservationTarget, ObservationError> {
    let maximum = limits.maximum_selected_entities;
    if let ObservationTarget::RegionPair { first, second } = &spec.target {
        return resolve_pair_target(spec, first, second, field_domain, topology, maximum);
    }
    if field_domain == RecoveredFieldEntityDomain::ConstraintGroups {
        if spec.source_location != FieldLocation::Constraint
            || !matches!(
                spec.quantity,
                ObservationQuantity::ReactionForce | ObservationQuantity::ReactionMoment
            )
            || !matches!(
                spec.reduction.as_ref().map(|reduction| reduction.spatial),
                Some(SpatialReduction::Resultant)
            )
        {
            return Err(ObservationError::Invalid(
                "constraint-group fields support reaction resultants only".into(),
            ));
        }
        let ObservationTarget::Constraint { constraint_id } = &spec.target else {
            return Err(target_error(
                spec,
                "constraint-group reaction fields require one authored constraint target",
            ));
        };
        let plan = constraint_plan.ok_or(ObservationError::Unsupported(
            "constraint-target reactions require the canonical executed constraint plan",
        ))?;
        let (index, group) = plan
            .groups
            .iter()
            .enumerate()
            .find(|(_, group)| group.constraint_id == *constraint_id)
            .ok_or_else(|| {
                target_error(spec, "constraint target is absent from the canonical plan")
            })?;
        let indices = selected_indices(std::iter::once(index), maximum)?;
        let mut identities = reserved(1)?;
        identities.push(StableDigest::from_bytes(*group.source_digest.bytes()));
        return Ok(ResolvedObservationTarget::ConstraintGroups {
            indices,
            identities,
        });
    }
    if !matches!(
        field_domain,
        RecoveredFieldEntityDomain::Nodes | RecoveredFieldEntityDomain::VolumeElements
    ) {
        return Err(ObservationError::Unsupported(
            "beam and shell reductions require formulation-owned integration rules",
        ));
    }
    if matches!(
        spec.quantity,
        ObservationQuantity::ReactionForce | ObservationQuantity::ReactionMoment
    ) {
        if spec.source_location != FieldLocation::Node
            || field_domain != RecoveredFieldEntityDomain::Nodes
        {
            return Err(ObservationError::Invalid(
                "reaction resultants require a nodal structural reaction field".into(),
            ));
        }
        return match &spec.target {
            ObservationTarget::WholeModel | ObservationTarget::Spatial(ExcitationTarget::WholeModel) => {
                nodes_target(selected_indices(0..topology.nodes.len(), maximum)?, topology, maximum)
            }
            ObservationTarget::Spatial(ExcitationTarget::Region(region))
                if region.entity_dimension == EntityDimension::Point && region.expected_topology == ExpectedTopology::MeshNode => {
                let id = parse_node_selector(&region.region_id, spec)?;
                let index = topology.node_index(id)?.ok_or_else(|| target_error(spec, "structural-node selector does not resolve"))?;
                enforce_cardinality(region.cardinality, 1, spec)?;
                nodes_target(selected_indices(std::iter::once(index), maximum)?, topology, maximum)
            }
            // Reactions over a face region are the discrete sum at the nodes of its faces.
            ObservationTarget::Spatial(ExcitationTarget::Region(region))
                if region.entity_dimension == EntityDimension::Face
                    && region.expected_topology == ExpectedTopology::BoundaryFace =>
            {
                let projection = solid_projection(topology, spec)?;
                let faces = matching_faces(region, projection);
                enforce_cardinality(region.cardinality, faces.clone().count(), spec)?;
                let indices = unique_nodes(
                    faces.flat_map(|index| {
                        projection.boundary_faces[index].node_indices.iter().copied()
                    }),
                    topology.nodes.len(),
                    limits,
                )?;
                nodes_target(indices, topology, maximum)
            }
            ObservationTarget::Constraint { .. } => Err(ObservationError::Unsupported("constraint-target reaction resultants require the canonical executed constraint plan")),
            _ => Err(target_error(spec, "reaction resultants support the whole model, an exact structural node, or a boundary face region")),
        };
    }
    let projection = solid_projection(topology, spec)?;
    match &spec.target {
        ObservationTarget::WholeModel
        | ObservationTarget::Spatial(ExcitationTarget::WholeModel) => volume_target(
            selected_indices(0..projection.elements.len(), maximum)?,
            projection,
        ),
        ObservationTarget::Spatial(ExcitationTarget::Region(region)) => {
            match (region.entity_dimension, region.expected_topology) {
                (EntityDimension::Volume, ExpectedTopology::MeshElement) => {
                    let indices = matching_elements(
                        region,
                        RecoveredFieldEntityDomain::VolumeElements,
                        topology,
                        limits,
                        selection_output_bytes(projection.elements.len(), maximum)?,
                    )?;
                    enforce_cardinality(region.cardinality, indices.clone().count(), spec)?;
                    volume_target(selected_indices(indices, maximum)?, projection)
                }
                // An element field on a boundary reads each face's owning element: its trace on
                // the surface is piecewise constant, weighted by face area.
                (EntityDimension::Face, ExpectedTopology::BoundaryFace)
                    if materialized_field_location(spec) == FieldLocation::Node
                        || (materialized_field_location(spec) == FieldLocation::Element
                            && field_domain == RecoveredFieldEntityDomain::VolumeElements) =>
                {
                    let indices = matching_faces(region, projection);
                    enforce_cardinality(region.cardinality, indices.clone().count(), spec)?;
                    let face_indices = selected_indices(indices, maximum)?;
                    let mut face_identities = reserved(face_indices.len())?;
                    face_identities.extend(
                        face_indices
                            .iter()
                            .map(|&index| projection.boundary_faces[index].stable_identity),
                    );
                    Ok(ResolvedObservationTarget::Boundary {
                        face_indices,
                        face_identities,
                    })
                }
                _ => Err(target_error(
                    spec,
                    "reductions support admitted volumes, and nodal or volume-element fields on admitted boundaries",
                )),
            }
        }
        _ => Err(target_error(
            spec,
            "reductions support whole-model and admitted region targets",
        )),
    }
}

fn volume_target(
    element_indices: Vec<usize>,
    projection: &AnalysisProjection,
) -> Result<ResolvedObservationTarget, ObservationError> {
    let mut element_identities = reserved(element_indices.len())?;
    element_identities.extend(
        element_indices
            .iter()
            .map(|&index| projection.elements[index].stable_identity),
    );
    Ok(ResolvedObservationTarget::Volume {
        element_indices,
        element_identities,
    })
}

/// Two disjoint face regions read by one paired reduction. Nodal fields are read at face
/// quadrature points, element and cell fields in each face's owner, and finite-volume face
/// fields at the oriented mesh face of each boundary face.
fn resolve_pair_target(
    spec: &ObservationSpec,
    first: &runmat_analysis_core::RegionTarget,
    second: &runmat_analysis_core::RegionTarget,
    field_domain: RecoveredFieldEntityDomain,
    topology: &ObservationTopology,
    maximum: usize,
) -> Result<ResolvedObservationTarget, ObservationError> {
    let readable = match (materialized_field_location(spec), field_domain) {
        (FieldLocation::Node, RecoveredFieldEntityDomain::Nodes) => true,
        (FieldLocation::Element | FieldLocation::Cell, RecoveredFieldEntityDomain::VolumeElements) => true,
        (FieldLocation::Face, RecoveredFieldEntityDomain::OrientedFaces) => {
            topology.finite_volume.is_some()
        }
        _ => false,
    };
    if !readable {
        return Err(target_error(
            spec,
            "a paired boundary reduction reads a nodal, element, cell, or finite-volume face field",
        ));
    }
    let projection = solid_projection(topology, spec)?;
    let mut selected = Vec::with_capacity(2);
    for region in [first, second] {
        if region.entity_dimension != EntityDimension::Face
            || region.expected_topology != ExpectedTopology::BoundaryFace
        {
            return Err(target_error(spec, "a paired reduction reads two boundary face regions"));
        }
        let faces = matching_faces(region, projection);
        enforce_cardinality(region.cardinality, faces.clone().count(), spec)?;
        let faces = selected_indices(faces, maximum)?;
        if faces.is_empty() {
            return Err(target_error(
                spec,
                format!("face region `{}` selects no boundary faces", region.region_id),
            ));
        }
        selected.push(faces);
    }
    let second_faces = selected.pop().expect("two regions");
    let first_faces = selected.pop().expect("two regions");
    if first_faces
        .iter()
        .any(|index| second_faces.binary_search(index).is_ok())
    {
        return Err(target_error(
            spec,
            "the two regions of a paired reduction share boundary faces",
        ));
    }
    let first_count = first_faces.len();
    let face_indices = selected_indices(
        first_faces.iter().chain(&second_faces).copied(),
        maximum,
    )?;
    let mut face_identities = reserved(face_indices.len())?;
    face_identities.extend(
        face_indices
            .iter()
            .map(|&index| projection.boundary_faces[index].stable_identity),
    );
    Ok(ResolvedObservationTarget::BoundaryPair {
        face_indices,
        face_identities,
        first_count,
    })
}
