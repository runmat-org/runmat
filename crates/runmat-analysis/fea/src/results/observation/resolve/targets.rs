//! Selection-bounded target materialization; limits count final entities, not parent matches.
use super::*;

mod field;
mod points;
mod reduction;
mod selection;
#[cfg(test)]
mod tests;

use field::resolve_field_target;
use points::{resolve_interpolated_points, resolve_probe};
use reduction::resolve_reduction_target;
use selection::*;

pub(super) fn resolve_target(
    request: &ObservationRequest,
    spec: &ObservationSpec,
    field_domain: RecoveredFieldEntityDomain,
    topology: &StructuralObservationTopology,
    constraint_plan: Option<&CanonicalStructuralConstraintPlan>,
    connector_plan: Option<&CanonicalStructuralConnectorPlan>,
    maximum_selected_entities: usize,
) -> Result<ResolvedObservationTarget, ObservationError> {
    let maximum = maximum_selected_entities;
    if field_domain == RecoveredFieldEntityDomain::Connectors {
        return resolve_connector_target(spec, connector_plan, maximum);
    }
    if spec.reduction.as_ref().is_some_and(|reduction| {
        matches!(
            reduction.spatial,
            SpatialReduction::Minimum | SpatialReduction::Maximum
        )
    }) {
        return resolve_field_target(spec, field_domain, topology, maximum);
    }
    match request {
        ObservationRequest::Probe(_) if spec.output_location == FieldLocation::Point => {
            resolve_interpolated_points(spec, field_domain, topology, maximum)
        }
        ObservationRequest::Probe(_) => resolve_probe(spec, field_domain, topology, maximum),
        ObservationRequest::Reduction(_) => {
            resolve_reduction_target(spec, field_domain, topology, constraint_plan, maximum)
        }
        ObservationRequest::Field(_) => resolve_field_target(spec, field_domain, topology, maximum),
        ObservationRequest::History(_)
            if spec
                .reduction
                .as_ref()
                .is_some_and(|reduction| reduction.spatial != SpatialReduction::None) =>
        {
            resolve_reduction_target(spec, field_domain, topology, constraint_plan, maximum)
        }
        ObservationRequest::History(_)
            if matches!(spec.target, ObservationTarget::Path { .. })
                || spec.output_location == FieldLocation::Point =>
        {
            resolve_interpolated_points(spec, field_domain, topology, maximum)
        }
        ObservationRequest::History(_)
            if matches!(spec.target, ObservationTarget::Point { .. }) =>
        {
            resolve_probe(spec, field_domain, topology, maximum)
        }
        ObservationRequest::History(_) => {
            resolve_field_target(spec, field_domain, topology, maximum)
        }
        _ => Err(ObservationError::Unsupported(
            "unsupported observation request",
        )),
    }
}
