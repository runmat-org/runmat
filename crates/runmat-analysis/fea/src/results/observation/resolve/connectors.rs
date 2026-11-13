use super::*;

pub(super) fn compile_connector_observation_plan(
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
    constraint_plan: &CanonicalStructuralConstraintPlan,
    topology_node_count: usize,
) -> Result<CanonicalStructuralConnectorPlan, ObservationError> {
    if model.connectors.is_empty() {
        CanonicalStructuralConnectorPlan::empty(constraint_plan.topology, topology_node_count)
            .map_err(|error| ObservationError::Invalid(error.to_string()))
    } else {
        crate::assembly::explicit_connectors::compile_model_connector_plan(
            model,
            mesh,
            constraint_plan.reference_points.len(),
        )
        .map_err(|error| match error {
            crate::assembly::StructuralAssemblyError::Accounting(cause) => {
                ObservationError::Accounting(cause)
            }
            crate::assembly::StructuralAssemblyError::Cancelled => {
                ObservationError::Topology(crate::topology::AnalysisProjectionError::Cancelled)
            }
            other => ObservationError::Invalid(other.to_string()),
        })
    }
}

pub(super) fn resolve_connector_target(
    spec: &ObservationSpec,
    connector_plan: Option<&CanonicalStructuralConnectorPlan>,
    maximum_selected_entities: usize,
) -> Result<ResolvedObservationTarget, ObservationError> {
    if spec.source_location != FieldLocation::Connector
        || spec.output_location != FieldLocation::Connector
        || !matches!(
            spec.quantity,
            ObservationQuantity::ReactionForce | ObservationQuantity::StrainEnergy
        )
        || spec.reduction.as_ref().is_some_and(|reduction| {
            !matches!(
                (spec.quantity, reduction.spatial),
                (_, SpatialReduction::None)
                    | (
                        ObservationQuantity::ReactionForce,
                        SpatialReduction::Resultant
                    )
            )
        })
    {
        return Err(ObservationError::Invalid(
            "connector fields support connector force and stored energy without spatial aggregation"
                .into(),
        ));
    }
    let ObservationTarget::Connector { connector_id } = &spec.target else {
        return Err(target_error(
            spec,
            "connector fields require one authored connector target",
        ));
    };
    let plan = connector_plan.ok_or(ObservationError::Unsupported(
        "connector observations require the canonical executed connector plan",
    ))?;
    let (index, entry) = plan
        .entries
        .iter()
        .enumerate()
        .find(|(_, entry)| entry.connector_id() == connector_id)
        .ok_or_else(|| target_error(spec, "connector target is absent from the canonical plan"))?;
    if maximum_selected_entities == 0 {
        return Err(ObservationError::Limit("selected topology entities"));
    }
    Ok(ResolvedObservationTarget::Connectors {
        indices: vec![index],
        identities: vec![StableDigest::from_bytes(*entry.source_digest().bytes())],
    })
}
