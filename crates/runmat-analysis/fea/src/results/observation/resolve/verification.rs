//! Source reprojection reuses the resolver, retaining at most one reconstructed observation.
//! The supplied plan is a cardinality ceiling, never scientific authority: every entry must
//! equal the independently resolved authored request before verification succeeds.
use super::*;

pub(crate) fn verify_authored_observations(
    step: &ProcedureStep,
    plan: &ResolvedObservationPlan,
) -> Result<(), ObservationError> {
    if plan.step_id != step.step_id
        || plan.independent_domain != step.independent_domain
        || plan.observations.len() != step.observations.len()
    {
        return Err(ObservationError::Identity(
            "observation plan differs from authored step coverage",
        ));
    }
    for (request, expected) in step.observations.iter().zip(&plan.observations) {
        let spec = match request {
            ObservationRequest::Field(spec)
            | ObservationRequest::History(spec)
            | ObservationRequest::Probe(spec)
            | ObservationRequest::Reduction(spec) => spec,
            _ => {
                return Err(ObservationError::Unsupported(
                    "diagnostics and monitors are evaluated by their owning pipeline",
                ))
            }
        };
        if spec != &expected.spec {
            return Err(ObservationError::Identity(
                "observation declaration differs from authored request",
            ));
        }
        if let ObservationTarget::Path { positions } = &spec.target {
            if positions.len() > expected.target.selected_entity_count() {
                return Err(ObservationError::Limit("selected topology entities"));
            }
        }
    }
    plan.validate_identity()
}

pub(crate) fn verify_observation_plan(
    step: &ProcedureStep,
    projection: &AnalysisProjection,
    fields: &[RecoveredFieldDefinition],
    plan: &ResolvedObservationPlan,
) -> Result<(), ObservationError> {
    verify_authored_observations(step, plan)?;
    let topology = StructuralObservationTopology::from_projection(projection);
    verify_for_topology(step, &topology, None, None, fields, plan)
}

pub(crate) fn verify_structural_observation_plan(
    step: &ProcedureStep,
    model: &AnalysisModel,
    mesh: Option<&SolverMeshArtifact>,
    constraint_plan: &CanonicalStructuralConstraintPlan,
    fields: &[RecoveredFieldDefinition],
    plan: &ResolvedObservationPlan,
) -> Result<(), ObservationError> {
    verify_authored_observations(step, plan)?;
    let topology = StructuralObservationTopology::derive(model, mesh)?;
    let connectors =
        compile_connector_observation_plan(model, mesh, constraint_plan, topology.nodes.len())?;
    validate_structural_context(model, &topology, constraint_plan, &connectors)?;
    verify_for_topology(
        step,
        &topology,
        Some(constraint_plan),
        Some(&connectors),
        fields,
        plan,
    )
}

fn verify_for_topology(
    step: &ProcedureStep,
    topology: &StructuralObservationTopology,
    constraints: Option<&CanonicalStructuralConstraintPlan>,
    connectors: Option<&CanonicalStructuralConnectorPlan>,
    fields: &[RecoveredFieldDefinition],
    plan: &ResolvedObservationPlan,
) -> Result<(), ObservationError> {
    if plan.topology_digest != topology.digest() {
        return Err(ObservationError::Identity(
            "observation plan differs from authoritative topology",
        ));
    }
    validate_field_catalog(fields)?;
    for (request, expected) in step.observations.iter().zip(&plan.observations) {
        let limits = verification_limits(expected)?;
        let actual = resolve_request(
            step,
            request,
            topology,
            constraints,
            connectors,
            fields,
            limits,
        )?;
        if &actual != expected {
            return Err(ObservationError::Identity(
                "resolved observation differs from authoritative mesh and fields",
            ));
        }
        // `actual` and its target buffers are released here, not retained in a second plan.
    }
    Ok(())
}

fn verification_limits(
    expected: &ResolvedObservation,
) -> Result<ObservationLimits, ObservationError> {
    let selected = expected.target.selected_entity_count();
    let components = observation_result_storage_component_count(
        shape_component_count(&expected.result_value_shape)?,
        expected.spec.complex_representation,
    )?;
    let maximum_values = selected
        .max(1)
        .checked_mul(components)
        .ok_or(ObservationError::Limit("observation value count"))?;
    let resolved = &expected.resolved_cost;
    // A valid authored upper bound can exceed the realized cost. It remains checked by the
    // normal resolver and cannot authorize a scientifically different reconstructed entry.
    let authored = expected.spec.cost.upper_bound();
    Ok(ObservationLimits {
        maximum_observations: 1,
        maximum_selected_entities: selected,
        maximum_values,
        maximum_operation_count: resolved
            .operation_count
            .max(authored.map_or(0, |cost| cost.operation_count)),
        maximum_peak_resident_bytes: resolved
            .peak_resident_bytes
            .max(authored.map_or(0, |cost| cost.peak_resident_bytes)),
        maximum_stored_bytes: resolved
            .stored_bytes
            .max(authored.map_or(0, |cost| cost.stored_bytes)),
        maximum_transfer_bytes: resolved
            .transfer_bytes
            .max(authored.map_or(0, |cost| cost.transfer_bytes)),
    })
}

#[cfg(test)]
mod tests;
