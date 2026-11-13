//! Verify the retained target against original fields using the existing observation owner.
use super::*;
use crate::results::field::{RecoveredFieldCatalog, StructuralRecoveredFieldTopology};
use crate::results::observation::{evaluate_structural_observations, ResolvedObservationPlan};

pub(super) fn verify(
    input: StructuralAdaptationSolvedSourceInput<'_>,
    plan: &ResolvedObservationPlan,
    original: &ObservationResult,
) -> Result<(), Error> {
    // The shared canonical materialization envelope has already been admitted before
    // copying retained field payloads. This is original-field evaluation, not another solve.
    let mut fields = input.run.fields.clone();
    let mut recovered =
        crate::post::fields::take_structural_observable_fields(&mut fields).map_err(invalid)?;
    let constraints = input
        .run
        .structural_constraint_result
        .as_ref()
        .ok_or_else(|| invalid("missing constraints"))?;
    let connectors = input
        .run
        .structural_connector_result
        .as_ref()
        .ok_or_else(|| invalid("missing connectors"))?;
    recovered.extend(
        crate::post::recover_constraint_group_observable_fields(constraints).map_err(invalid)?,
    );
    recovered
        .extend(crate::post::recover_connector_observable_fields(connectors).map_err(invalid)?);
    let mut topology =
        StructuralRecoveredFieldTopology::derive(input.model, Some(input.mesh)).map_err(invalid)?;
    topology.entity_counts.constraint_groups = constraints.groups.len();
    topology.entity_counts.connectors = connectors.responses.len();
    let catalog = RecoveredFieldCatalog::new_for_topology(
        topology.identity.discretization_digest(),
        topology.entity_counts,
        recovered,
    )
    .map_err(invalid)?;
    let evaluated = evaluate_structural_observations(plan, input.model, Some(input.mesh), &catalog)
        .map_err(invalid)?;
    if evaluated
        .iter()
        .find(|o| &o.observation_id == input.observation_id)
        != Some(original)
    {
        return Err(invalid(
            "retained target observation differs from original physical fields",
        ));
    }
    Ok(())
}
