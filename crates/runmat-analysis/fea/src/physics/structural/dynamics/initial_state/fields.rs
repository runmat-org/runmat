//! Borrow admitted canonical nodal content without interpolation or numerical recovery.
use super::{assignment::RealizationContext, StructuralInitialStateError};
use crate::execution::AdmittedInitialField;
use runmat_analysis_core::{InitialConditionValue, InitialValue};

pub(super) fn field_values<'a, Q>(
    context: &RealizationContext<'a>,
    condition: &InitialConditionValue<Q>,
) -> Result<Option<AdmittedInitialField<'a>>, StructuralInitialStateError> {
    let InitialValue::FieldArtifact(reference) = &condition.value else {
        return Ok(None);
    };
    let binding = context
        .fields
        .iter()
        .find(|binding| binding.reference() == reference)
        .ok_or_else(
            || StructuralInitialStateError::UnmaterializedFieldArtifact {
                initial_condition_id: condition.initial_condition_id.clone(),
                artifact_id: reference.artifact_id.clone(),
            },
        )?;
    let selected = context
        .model
        .initial_conditions
        .iter()
        .find(|selected| selected.initial_condition_id() == condition.initial_condition_id)
        .ok_or_else(|| StructuralInitialStateError::InvalidFieldArtifact {
            initial_condition_id: condition.initial_condition_id.clone(),
            detail: "initial condition is not model-owned".into(),
        })?;
    binding
        .admit(
            selected,
            context.topology_digest,
            context.topology.nodes.len(),
        )
        .map(Some)
        .map_err(|error| StructuralInitialStateError::InvalidFieldArtifact {
            initial_condition_id: condition.initial_condition_id.clone(),
            detail: error.to_string(),
        })
}
