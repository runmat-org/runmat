use runmat_analysis_core::{
    AnalysisField, FieldLocation, ObservationQuantity, ObservationValueShape,
};

use crate::{
    constraints::CanonicalStructuralConstraintResult,
    contracts::{
        FEA_FIELD_STRUCTURAL_CONSTRAINT_REACTION_FORCE,
        FEA_FIELD_STRUCTURAL_CONSTRAINT_REACTION_MOMENT_ABOUT_GLOBAL_ORIGIN,
    },
    results::field::{
        RecoveredField, RecoveredFieldDefinition, RecoveredFieldEntityDomain, RecoveredFieldError,
        RecoveredFieldNumericRepresentation, RecoveredFieldSample,
    },
};

/// Recovers exact authored-constraint reaction totals as a compact field domain.
///
/// These values come from canonical multiplier ownership. They must not be reconstructed by
/// selecting the aggregate nodal reaction field because independently authored constraints may
/// share nodes and degrees of freedom.
pub fn recover_constraint_group_observable_fields(
    result: &CanonicalStructuralConstraintResult,
) -> Result<Vec<RecoveredField>, RecoveredFieldError> {
    result
        .validate()
        .map_err(|error| RecoveredFieldError::InvalidArtifact(error.to_string()))?;
    if result.groups.is_empty() {
        return Ok(Vec::new());
    }
    let force = result
        .groups
        .iter()
        .flat_map(|group| group.resultant_force_newtons)
        .collect::<Vec<_>>();
    let moment = result
        .groups
        .iter()
        .flat_map(|group| group.resultant_moment_about_global_origin_newton_meters)
        .collect::<Vec<_>>();
    [
        (
            FEA_FIELD_STRUCTURAL_CONSTRAINT_REACTION_FORCE,
            ObservationQuantity::ReactionForce,
            force,
        ),
        (
            FEA_FIELD_STRUCTURAL_CONSTRAINT_REACTION_MOMENT_ABOUT_GLOBAL_ORIGIN,
            ObservationQuantity::ReactionMoment,
            moment,
        ),
    ]
    .into_iter()
    .map(|(field_id, quantity, values)| {
        RecoveredField::new(
            RecoveredFieldDefinition::new(
                field_id,
                quantity,
                FieldLocation::Constraint,
                RecoveredFieldEntityDomain::ConstraintGroups,
                ObservationValueShape::Vector3,
                RecoveredFieldNumericRepresentation::Real,
            )?,
            RecoveredFieldSample::Final,
            AnalysisField::host_f64(field_id, vec![result.groups.len(), 3], values),
        )
    })
    .collect()
}
