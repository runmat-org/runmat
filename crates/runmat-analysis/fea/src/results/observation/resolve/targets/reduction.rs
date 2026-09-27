use super::*;

pub(super) fn resolve_reduction_target(
    spec: &ObservationSpec,
    field_domain: RecoveredFieldEntityDomain,
    topology: &StructuralObservationTopology,
    constraint_plan: Option<&CanonicalStructuralConstraintPlan>,
    maximum: usize,
) -> Result<ResolvedObservationTarget, ObservationError> {
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
        RecoveredFieldEntityDomain::StructuralNodes | RecoveredFieldEntityDomain::SolidElements
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
            || field_domain != RecoveredFieldEntityDomain::StructuralNodes
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
            ObservationTarget::Constraint { .. } => Err(ObservationError::Unsupported("constraint-target reaction resultants require the canonical executed constraint plan")),
            _ => Err(target_error(spec, "reaction resultants support the whole model or an exact structural node")),
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
                        RecoveredFieldEntityDomain::SolidElements,
                        topology,
                    )?;
                    enforce_cardinality(region.cardinality, indices.clone().count(), spec)?;
                    volume_target(selected_indices(indices, maximum)?, projection)
                }
                (EntityDimension::Face, ExpectedTopology::BoundaryFace)
                    if materialized_field_location(spec) == FieldLocation::Node =>
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
                    "reductions support admitted volumes and nodal fields on admitted boundaries",
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
