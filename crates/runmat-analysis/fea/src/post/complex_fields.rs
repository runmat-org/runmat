use std::collections::BTreeMap;

use runmat_analysis_core::{AnalysisField, ObservationQuantity};
use thiserror::Error;

use crate::{
    assembly::AssemblySummary,
    results::field::{
        RecoveredField, RecoveredFieldDefinition, RecoveredFieldError,
        RecoveredFieldNumericRepresentation, RecoveredFieldSample,
    },
};

use super::{
    fields::{recover_deformation_result_fields, structural_observable_field_definitions},
    StructuralFieldRecoveryError,
};

#[derive(Clone, Debug, Error, PartialEq)]
pub enum ComplexStructuralFieldRecoveryError {
    #[error(transparent)]
    PhysicalRecovery(#[from] StructuralFieldRecoveryError),
    #[error(transparent)]
    RecoveredField(#[from] RecoveredFieldError),
    #[error("complex structural recovery inventories have different lengths")]
    InventoryLengthMismatch,
    #[error("complex structural field `{field_id}` has no typed definition")]
    MissingDefinition { field_id: String },
    #[error("complex structural field parts have different definitions")]
    DefinitionMismatch,
    #[error("complex structural field part is not host real64")]
    NonHostStorage,
    #[error("complex structural field parts have different storage lengths")]
    StorageLengthMismatch,
}

/// Recovers one physical complex structural response from its real and imaginary DOF vectors.
///
/// Formulation-owned real recovery remains the single authority for the physical quantities. This
/// boundary only interleaves quantities whose recovery is linear in displacement. Nonlinear
/// invariants such as von Mises stress and quadratic energy require their own explicitly defined
/// harmonic representation and are therefore omitted rather than mislabeled as phasors.
pub fn recover_complex_deformation_result_fields(
    summary: &AssemblySummary,
    frequency_hz: f64,
    displacement_real: &[f64],
    displacement_imaginary: &[f64],
) -> Result<Vec<RecoveredField>, ComplexStructuralFieldRecoveryError> {
    let real = recover_deformation_result_fields(summary, displacement_real)?;
    let imaginary = recover_deformation_result_fields(summary, displacement_imaginary)?;
    let definitions = structural_observable_field_definitions()?
        .into_iter()
        .map(|definition| (definition.field_id().to_string(), definition))
        .collect::<BTreeMap<_, _>>();
    if real.len() != imaginary.len() {
        return Err(ComplexStructuralFieldRecoveryError::InventoryLengthMismatch);
    }
    real.iter()
        .zip(&imaginary)
        .filter(|(real, imaginary)| {
            real.shape.first().copied().unwrap_or_default() > 0
                || imaginary.shape.first().copied().unwrap_or_default() > 0
        })
        .filter(|(real, _)| {
            definitions
                .get(&real.field_id)
                .is_some_and(|definition| is_linear_phasor_quantity(definition.quantity()))
        })
        .map(|(real, imaginary)| {
            let definition = definitions.get(&real.field_id).ok_or_else(|| {
                ComplexStructuralFieldRecoveryError::MissingDefinition {
                    field_id: real.field_id.clone(),
                }
            })?;
            let complex_definition = RecoveredFieldDefinition::new(
                definition.field_id(),
                definition.quantity(),
                definition.location(),
                definition.entity_domain(),
                definition.value_shape().clone(),
                RecoveredFieldNumericRepresentation::ComplexInterleaved,
            )?;
            Ok(RecoveredField::new(
                complex_definition,
                RecoveredFieldSample::FrequencyHertz(frequency_hz),
                interleave(real, imaginary)?,
            )?)
        })
        .collect()
}

fn is_linear_phasor_quantity(quantity: ObservationQuantity) -> bool {
    !matches!(
        quantity,
        ObservationQuantity::VonMisesStress
            | ObservationQuantity::StrainEnergy
            | ObservationQuantity::ShellVonMises
    )
}

fn interleave(
    real: &AnalysisField,
    imaginary: &AnalysisField,
) -> Result<AnalysisField, ComplexStructuralFieldRecoveryError> {
    if real.field_id != imaginary.field_id || real.shape != imaginary.shape {
        return Err(ComplexStructuralFieldRecoveryError::DefinitionMismatch);
    }
    let real_values = real
        .as_host_f64()
        .ok_or(ComplexStructuralFieldRecoveryError::NonHostStorage)?;
    let imaginary_values = imaginary
        .as_host_f64()
        .ok_or(ComplexStructuralFieldRecoveryError::NonHostStorage)?;
    if real_values.len() != imaginary_values.len() {
        return Err(ComplexStructuralFieldRecoveryError::StorageLengthMismatch);
    }
    let mut shape = real.shape.clone();
    shape.push(2);
    Ok(AnalysisField::host_f64(
        &real.field_id,
        shape,
        real_values
            .iter()
            .zip(imaginary_values)
            .flat_map(|(real, imaginary)| [*real, *imaginary])
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::FixtureId;

    #[test]
    fn complex_recovery_uses_canonical_interleaved_fields() {
        let model = crate::fixtures::fixture_model(FixtureId::CantileverLinearStatic);
        let excitations = crate::excitation::realize_step_excitations(
            &model,
            &model.sequence.steps[0],
            crate::excitation::ExcitationSamplePoint::Static,
        )
        .unwrap();
        let summary = crate::assembly::assemble_structural_system(
            &model,
            &excitations,
            None,
            crate::assembly::StructuralFormulationRequirement::ExperimentalSyntheticOperator,
        )
        .unwrap();
        let real = vec![0.1; summary.dof_count];
        let imaginary = vec![-0.2; summary.dof_count];
        let fields =
            recover_complex_deformation_result_fields(&summary, 60.0, &real, &imaginary).unwrap();
        let displacement = fields
            .iter()
            .find(|field| {
                field.definition().quantity()
                    == runmat_analysis_core::ObservationQuantity::Displacement
            })
            .unwrap();
        assert_eq!(
            displacement.definition().numeric_representation(),
            RecoveredFieldNumericRepresentation::ComplexInterleaved
        );
        assert_eq!(
            displacement.sample(),
            RecoveredFieldSample::FrequencyHertz(60.0)
        );
        assert!(displacement
            .storage()
            .as_host_f64()
            .unwrap()
            .chunks_exact(2)
            .all(|pair| pair == [0.1, -0.2]));
        assert!(fields.iter().all(|field| !matches!(
            field.definition().quantity(),
            ObservationQuantity::VonMisesStress
                | ObservationQuantity::StrainEnergy
                | ObservationQuantity::ShellVonMises
        )));
    }
}
