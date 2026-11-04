use runmat_analysis_core::AnalysisFieldLayout;
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
            AnalysisField::host_f64(
                field_id,
                AnalysisFieldLayout::vector3(quantity, FieldLocation::Constraint),
                vec![result.groups.len(), 3],
                values,
            ),
        )
    })
    .collect()
}

/// Recovers exact authored-constraint reaction phasors without collapsing row ownership.
pub fn recover_complex_constraint_group_observable_fields(
    real: &CanonicalStructuralConstraintResult,
    imaginary: &CanonicalStructuralConstraintResult,
    frequency_hz: f64,
) -> Result<Vec<RecoveredField>, RecoveredFieldError> {
    real.validate()
        .map_err(|error| RecoveredFieldError::InvalidArtifact(error.to_string()))?;
    imaginary
        .validate()
        .map_err(|error| RecoveredFieldError::InvalidArtifact(error.to_string()))?;
    if real.constraint_plan_digest != imaginary.constraint_plan_digest
        || real.topology != imaginary.topology
        || real.groups.len() != imaginary.groups.len()
        || real
            .groups
            .iter()
            .zip(&imaginary.groups)
            .any(|(left, right)| {
                left.constraint_id != right.constraint_id
                    || left.active_row_count != right.active_row_count
                    || left.redundant_row_count != right.redundant_row_count
                    || left
                        .row_reactions
                        .iter()
                        .map(|row| row.row_identity)
                        .ne(right.row_reactions.iter().map(|row| row.row_identity))
            })
    {
        return Err(RecoveredFieldError::InvalidArtifact(
            "complex constraint reaction components have different canonical ownership".into(),
        ));
    }
    let real_fields = recover_constraint_group_observable_fields(real)?;
    let imaginary_fields = recover_constraint_group_observable_fields(imaginary)?;
    real_fields
        .iter()
        .zip(&imaginary_fields)
        .map(|(real, imaginary)| {
            RecoveredField::new(
                RecoveredFieldDefinition::new(
                    real.definition().field_id(),
                    real.definition().quantity(),
                    real.definition().location(),
                    real.definition().entity_domain(),
                    real.definition().value_shape().clone(),
                    RecoveredFieldNumericRepresentation::ComplexInterleaved,
                )?,
                RecoveredFieldSample::FrequencyHertz(frequency_hz),
                super::complex_fields::interleave(real.storage(), imaginary.storage())
                    .map_err(|error| RecoveredFieldError::InvalidArtifact(error.to_string()))?,
            )
        })
        .collect()
}
