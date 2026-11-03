use runmat_analysis_core::{AnalysisFieldLayout, FieldLocation, ComplexRepresentation};
use std::collections::BTreeMap;

use runmat_analysis_core::{AnalysisField, ObservationQuantity};
use thiserror::Error;

use crate::{
    assembly::{dofs::StructuralDofKind, AssemblySummary},
    contracts::{FEA_FIELD_STRUCTURAL_REACTION_FORCE, FEA_FIELD_STRUCTURAL_REACTION_MOMENT},
    results::field::{
        RecoveredField, RecoveredFieldDefinition, RecoveredFieldEntityCounts, RecoveredFieldError,
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
    #[error(transparent)]
    Connector(#[from] crate::connectors::ConnectorRecoveryError),
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
    #[error("complex structural reaction length differs from its canonical DOF layout")]
    ReactionLengthMismatch,
}

pub struct ComplexStructuralFieldState<'a> {
    pub displacement_real: &'a [f64],
    pub displacement_imaginary: &'a [f64],
    pub reaction_real: &'a [f64],
    pub reaction_imaginary: &'a [f64],
}

/// Exact formulation-owned inventory for one complex structural recovery task.
///
/// Resource projection and physical recovery consume this same layout so field selection,
/// entity cardinality, and complex storage width cannot drift independently.
#[derive(Clone, Debug, PartialEq)]
pub struct ComplexStructuralFieldLayout {
    pub entity_counts: RecoveredFieldEntityCounts,
    pub definitions: Vec<RecoveredFieldDefinition>,
    pub storage_value_count: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ComplexStructuralFieldCardinality {
    pub nodes: usize,
    pub volume_elements: usize,
    pub beam_elements: usize,
    pub shell_elements: usize,
    pub connectors: usize,
    pub constraint_groups: usize,
    pub has_rotational_dofs: bool,
}

pub fn complex_structural_field_layout(
    cardinality: ComplexStructuralFieldCardinality,
) -> Result<ComplexStructuralFieldLayout, ComplexStructuralFieldRecoveryError> {
    let entity_counts = RecoveredFieldEntityCounts {
        oriented_faces: 0,
        nodes: cardinality.nodes,
        volume_elements: cardinality.volume_elements,
        volume_integration_points: cardinality.volume_elements.saturating_mul(4),
        oriented_edges: 0,
        beam_elements: cardinality.beam_elements,
        shell_elements: cardinality.shell_elements,
        connectors: cardinality.connectors,
        constraint_groups: cardinality.constraint_groups,
    };
    let mut definitions = structural_observable_field_definitions()?
        .into_iter()
        .filter(|definition| is_deformation_phasor_quantity(definition.quantity()))
        .filter(|definition| {
            entity_counts.entity_count(definition.entity_domain()) > 0
                && (definition.quantity() != ObservationQuantity::Rotation
                    || cardinality.has_rotational_dofs)
        })
        .map(|definition| complex_definition(&definition))
        .collect::<Result<Vec<_>, _>>()?;
    let declarations = structural_observable_field_definitions()?
        .into_iter()
        .map(|definition| (definition.field_id().to_owned(), definition))
        .collect::<BTreeMap<_, _>>();
    for field_id in [
        FEA_FIELD_STRUCTURAL_REACTION_FORCE,
        FEA_FIELD_STRUCTURAL_REACTION_MOMENT,
    ] {
        if field_id == FEA_FIELD_STRUCTURAL_REACTION_MOMENT && !cardinality.has_rotational_dofs {
            continue;
        }
        definitions.push(complex_definition(declarations.get(field_id).ok_or_else(
            || ComplexStructuralFieldRecoveryError::MissingDefinition {
                field_id: field_id.to_owned(),
            },
        )?)?);
    }
    if cardinality.constraint_groups > 0 {
        for field_id in [
            crate::contracts::FEA_FIELD_STRUCTURAL_CONSTRAINT_REACTION_FORCE,
            crate::contracts::FEA_FIELD_STRUCTURAL_CONSTRAINT_REACTION_MOMENT_ABOUT_GLOBAL_ORIGIN,
        ] {
            definitions.push(complex_definition(declarations.get(field_id).ok_or_else(
                || ComplexStructuralFieldRecoveryError::MissingDefinition {
                    field_id: field_id.to_owned(),
                },
            )?)?);
        }
    }
    if cardinality.connectors > 0 {
        definitions.push(RecoveredFieldDefinition::new(
            crate::contracts::FEA_FIELD_STRUCTURAL_CONNECTOR_FORCE,
            ObservationQuantity::ReactionForce,
            runmat_analysis_core::FieldLocation::Connector,
            crate::results::field::RecoveredFieldEntityDomain::Connectors,
            runmat_analysis_core::ObservationValueShape::Vector3,
            RecoveredFieldNumericRepresentation::ComplexInterleaved,
        )?);
    }
    let storage_value_count = definitions.iter().try_fold(0_usize, |total, definition| {
        let values = entity_counts
            .entity_count(definition.entity_domain())
            .checked_mul(definition.storage_component_count()?)
            .ok_or(RecoveredFieldError::StorageShapeMismatch)?;
        total
            .checked_add(values)
            .ok_or(RecoveredFieldError::StorageShapeMismatch)
    })?;
    Ok(ComplexStructuralFieldLayout {
        entity_counts,
        definitions,
        storage_value_count,
    })
}

fn complex_definition(
    definition: &RecoveredFieldDefinition,
) -> Result<RecoveredFieldDefinition, RecoveredFieldError> {
    RecoveredFieldDefinition::new(
        definition.field_id(),
        definition.quantity(),
        definition.location(),
        definition.entity_domain(),
        definition.value_shape().clone(),
        RecoveredFieldNumericRepresentation::ComplexInterleaved,
    )
}

/// Recovers one physical complex structural response from its real and imaginary DOF vectors.
///
/// Formulation-owned real recovery remains the single authority for the physical quantities. This
/// boundary only interleaves quantities whose recovery is linear in displacement. Nonlinear
/// invariants such as von Mises stress and quadratic energy require their own explicitly defined
/// harmonic representation and are therefore omitted rather than mislabeled as phasors.
pub fn recover_complex_structural_result_fields(
    summary: &AssemblySummary,
    frequency_hz: f64,
    state: ComplexStructuralFieldState<'_>,
) -> Result<Vec<RecoveredField>, ComplexStructuralFieldRecoveryError> {
    let layout = complex_structural_field_layout(ComplexStructuralFieldCardinality {
        nodes: summary.structural_dof_layout.node_count(),
        volume_elements: summary.structural_solid_recovery.len(),
        beam_elements: summary.structural_beam_recovery.len(),
        shell_elements: summary.structural_shell_recovery.len(),
        connectors: 0,
        constraint_groups: 0,
        has_rotational_dofs: summary.structural_rotational_dof_count > 0,
    })?;
    let real = recover_deformation_result_fields(summary, state.displacement_real)?;
    let imaginary = recover_deformation_result_fields(summary, state.displacement_imaginary)?;
    let definitions = layout
        .definitions
        .into_iter()
        .map(|definition| (definition.field_id().to_string(), definition))
        .collect::<BTreeMap<_, _>>();
    if real.len() != imaginary.len() {
        return Err(ComplexStructuralFieldRecoveryError::InventoryLengthMismatch);
    }
    let mut fields = real
        .iter()
        .zip(&imaginary)
        .filter(|(real, imaginary)| {
            real.shape.first().copied().unwrap_or_default() > 0
                || imaginary.shape.first().copied().unwrap_or_default() > 0
        })
        .filter(|(real, _)| {
            definitions
                .get(&real.field_id)
                .is_some_and(|definition| is_deformation_phasor_quantity(definition.quantity()))
        })
        .map(
            |(real, imaginary)| -> Result<RecoveredField, ComplexStructuralFieldRecoveryError> {
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
            },
        )
        .collect::<Result<Vec<_>, _>>()?;
    append_exact_reaction_fields(
        summary,
        frequency_hz,
        state.reaction_real,
        state.reaction_imaginary,
        &definitions,
        &mut fields,
    )?;
    Ok(fields)
}

fn is_deformation_phasor_quantity(quantity: ObservationQuantity) -> bool {
    !matches!(
        quantity,
        ObservationQuantity::VonMisesStress
            | ObservationQuantity::StrainEnergy
            | ObservationQuantity::ShellVonMises
            | ObservationQuantity::ReactionForce
            | ObservationQuantity::ReactionMoment
    )
}

fn append_exact_reaction_fields(
    summary: &AssemblySummary,
    frequency_hz: f64,
    reaction_real: &[f64],
    reaction_imaginary: &[f64],
    definitions: &BTreeMap<String, RecoveredFieldDefinition>,
    fields: &mut Vec<RecoveredField>,
) -> Result<(), ComplexStructuralFieldRecoveryError> {
    let real = project_nodal_reaction(summary, reaction_real)?;
    let imaginary = project_nodal_reaction(summary, reaction_imaginary)?;
    let mut pairs = vec![(
        FEA_FIELD_STRUCTURAL_REACTION_FORCE,
        real.force,
        imaginary.force,
    )];
    if let (Some(real), Some(imaginary)) = (real.moment, imaginary.moment) {
        pairs.push((FEA_FIELD_STRUCTURAL_REACTION_MOMENT, real, imaginary));
    }
    for (field_id, real, imaginary) in pairs {
        let definition = definitions.get(field_id).ok_or_else(|| {
            ComplexStructuralFieldRecoveryError::MissingDefinition {
                field_id: field_id.to_owned(),
            }
        })?;
        fields.push(RecoveredField::new(
            RecoveredFieldDefinition::new(
                definition.field_id(),
                definition.quantity(),
                definition.location(),
                definition.entity_domain(),
                definition.value_shape().clone(),
                RecoveredFieldNumericRepresentation::ComplexInterleaved,
            )?,
            RecoveredFieldSample::FrequencyHertz(frequency_hz),
            interleave(&real, &imaginary)?,
        )?);
    }
    Ok(())
}

struct NodalReactionFields {
    force: AnalysisField,
    moment: Option<AnalysisField>,
}

fn project_nodal_reaction(
    summary: &AssemblySummary,
    reaction: &[f64],
) -> Result<NodalReactionFields, ComplexStructuralFieldRecoveryError> {
    let layout = &summary.structural_dof_layout;
    if reaction.len() != layout.total_dof_count() {
        return Err(ComplexStructuralFieldRecoveryError::ReactionLengthMismatch);
    }
    let mut force = vec![0.0; layout.node_count() * 3];
    let mut moment = vec![0.0; layout.node_count() * 3];
    let mut has_rotation = false;
    for (row, value) in reaction.iter().take(layout.node_dof_count()).enumerate() {
        let address = layout
            .address(row)
            .ok_or(ComplexStructuralFieldRecoveryError::ReactionLengthMismatch)?;
        let component = match address.kind {
            StructuralDofKind::Ux | StructuralDofKind::Rx => 0,
            StructuralDofKind::Uy | StructuralDofKind::Ry => 1,
            StructuralDofKind::Uz | StructuralDofKind::Rz => 2,
        };
        let target = address.node_index * 3 + component;
        if address.kind.is_translational() {
            force[target] = *value;
        } else {
            has_rotation = true;
            moment[target] = *value;
        }
    }
    Ok(NodalReactionFields {
        force: AnalysisField::host_f64(
            FEA_FIELD_STRUCTURAL_REACTION_FORCE, AnalysisFieldLayout::vector3(ObservationQuantity::ReactionForce, FieldLocation::Node).representation(ComplexRepresentation::Complex),
            vec![layout.node_count(), 3],
            force,
        ),
        moment: has_rotation.then(|| {
            AnalysisField::host_f64(
                FEA_FIELD_STRUCTURAL_REACTION_MOMENT, AnalysisFieldLayout::vector3(ObservationQuantity::ReactionMoment, FieldLocation::Node).representation(ComplexRepresentation::Complex),
                vec![layout.node_count(), 3],
                moment,
            )
        }),
    })
}

pub(super) fn interleave(
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
        &real.field_id, real.layout.clone().representation(ComplexRepresentation::Complex),
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
            crate::assembly::StructuralFormulationRequirement::SyntheticOperator,
        )
        .unwrap();
        let real = vec![0.1; summary.dof_count];
        let imaginary = vec![-0.2; summary.dof_count];
        let reaction_real = vec![0.3; summary.dof_count];
        let reaction_imaginary = vec![-0.4; summary.dof_count];
        let fields = recover_complex_structural_result_fields(
            &summary,
            60.0,
            ComplexStructuralFieldState {
                displacement_real: &real,
                displacement_imaginary: &imaginary,
                reaction_real: &reaction_real,
                reaction_imaginary: &reaction_imaginary,
            },
        )
        .unwrap();
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
        let reaction = fields
            .iter()
            .find(|field| field.definition().quantity() == ObservationQuantity::ReactionForce)
            .unwrap();
        assert!(reaction
            .storage()
            .as_host_f64()
            .unwrap()
            .chunks_exact(2)
            .all(|pair| pair == [0.3, -0.4]));
    }
}
