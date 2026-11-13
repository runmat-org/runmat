use runmat_analysis_core::{
    AnalysisField, FieldLocation, ObservationQuantity, ObservationValueShape,
};

use crate::{
    contracts::{
        FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE, FEA_FIELD_STRUCTURAL_BEAM_BENDING_MOMENT,
        FEA_FIELD_STRUCTURAL_BEAM_BENDING_STRESS, FEA_FIELD_STRUCTURAL_BEAM_SHEAR_FORCE,
        FEA_FIELD_STRUCTURAL_BEAM_TORSION_MOMENT, FEA_FIELD_STRUCTURAL_BEAM_TORSION_STRESS,
        FEA_FIELD_STRUCTURAL_CONSTRAINT_REACTION_FORCE,
        FEA_FIELD_STRUCTURAL_CONSTRAINT_REACTION_MOMENT_ABOUT_GLOBAL_ORIGIN,
        FEA_FIELD_STRUCTURAL_DISPLACEMENT, FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRAIN,
        FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS, FEA_FIELD_STRUCTURAL_NODAL_STRESS,
        FEA_FIELD_STRUCTURAL_NODAL_VON_MISES, FEA_FIELD_STRUCTURAL_REACTION_FORCE,
        FEA_FIELD_STRUCTURAL_ROTATION, FEA_FIELD_STRUCTURAL_SHELL_BENDING_MOMENT,
        FEA_FIELD_STRUCTURAL_SHELL_MEMBRANE_FORCE, FEA_FIELD_STRUCTURAL_SHELL_TRANSVERSE_SHEAR,
        FEA_FIELD_STRUCTURAL_SHELL_VON_MISES, FEA_FIELD_STRUCTURAL_STRAIN,
        FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY, FEA_FIELD_STRUCTURAL_STRESS,
        FEA_FIELD_STRUCTURAL_VON_MISES,
    },
    results::field::{
        RecoveredField, RecoveredFieldDefinition, RecoveredFieldDerivation,
        RecoveredFieldEntityDomain, RecoveredFieldError, RecoveredFieldNumericRepresentation,
        RecoveredFieldSample,
    },
};

/// Structural quantities whose recovery is linear in the displacement state.
///
/// Statistical modal procedures may combine these component transfers before applying any
/// nonlinear derived transform. Equivalent/principal stress, energy, and beam peak-end selectors
/// are intentionally absent.
#[derive(
    Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum StructuralLinearFieldQuantity {
    Displacement,
    Rotation,
    Strain,
    Stress,
    ShellMembraneForce,
    ShellBendingMoment,
    ShellTransverseShear,
}

impl StructuralLinearFieldQuantity {
    pub const ALL: [Self; 7] = [
        Self::Displacement,
        Self::Rotation,
        Self::Strain,
        Self::Stress,
        Self::ShellMembraneForce,
        Self::ShellBendingMoment,
        Self::ShellTransverseShear,
    ];

    pub const fn source_field_id(self) -> &'static str {
        match self {
            Self::Displacement => FEA_FIELD_STRUCTURAL_DISPLACEMENT,
            Self::Rotation => FEA_FIELD_STRUCTURAL_ROTATION,
            Self::Strain => FEA_FIELD_STRUCTURAL_STRAIN,
            Self::Stress => FEA_FIELD_STRUCTURAL_STRESS,
            Self::ShellMembraneForce => FEA_FIELD_STRUCTURAL_SHELL_MEMBRANE_FORCE,
            Self::ShellBendingMoment => FEA_FIELD_STRUCTURAL_SHELL_BENDING_MOMENT,
            Self::ShellTransverseShear => FEA_FIELD_STRUCTURAL_SHELL_TRANSVERSE_SHEAR,
        }
    }

    /// Physical quantity represented by this linear recovery channel.
    ///
    /// Statistical solvers retain this closed quantity instead of attaching unit strings. The
    /// model's canonical unit system and this quantity determine the units of RMS; variance uses
    /// the square of those units.
    pub fn layout(self) -> runmat_analysis_core::AnalysisFieldLayout {
        runmat_analysis_core::AnalysisFieldLayout::canonical(
            self.observation_quantity(),
            self.location(),
        )
    }

    pub const fn observation_quantity(self) -> ObservationQuantity {
        match self {
            Self::Displacement => ObservationQuantity::Displacement,
            Self::Rotation => ObservationQuantity::Rotation,
            Self::Strain => ObservationQuantity::Strain,
            Self::Stress => ObservationQuantity::Stress,
            Self::ShellMembraneForce => ObservationQuantity::ShellMembraneForce,
            Self::ShellBendingMoment => ObservationQuantity::ShellBendingMoment,
            Self::ShellTransverseShear => ObservationQuantity::ShellTransverseShear,
        }
    }

    /// Canonical topology location of the recovered quantity.
    pub const fn location(self) -> FieldLocation {
        match self {
            Self::Displacement | Self::Rotation => FieldLocation::Node,
            Self::Strain
            | Self::Stress
            | Self::ShellMembraneForce
            | Self::ShellBendingMoment
            | Self::ShellTransverseShear => FieldLocation::Element,
        }
    }
}

pub fn take_structural_observable_fields(
    fields: &mut Vec<AnalysisField>,
) -> Result<Vec<RecoveredField>, RecoveredFieldError> {
    let inventory = structural_retained_field_inventory(fields, 0)?;
    let mut recovered = Vec::with_capacity(inventory.len());
    for (definition, _) in inventory {
        let index = fields
            .iter()
            .position(|field| {
                field.field_id == definition.field_id()
                    && field.shape.first().is_some_and(|count| *count > 0)
            })
            .ok_or(RecoveredFieldError::StorageShapeMismatch)?;
        let storage = fields.remove(index);
        recovered.push(RecoveredField::new(
            definition,
            RecoveredFieldSample::Final,
            storage,
        )?);
    }
    Ok(recovered)
}

/// Shared original field inventory used both by move-only recovery and its resource planner.
/// Constraint-group channels are generated from canonical reactions, not nodal totals.
pub fn structural_retained_field_inventory(
    fields: &[AnalysisField],
    constraint_group_count: usize,
) -> Result<Vec<(RecoveredFieldDefinition, usize)>, RecoveredFieldError> {
    let declarations = structural_observable_field_definitions()?;
    let mut inventory = Vec::with_capacity(declarations.len());
    for definition in declarations {
        let count = if definition.entity_domain() == RecoveredFieldEntityDomain::ConstraintGroups {
            constraint_group_count
        } else {
            let Some(storage) = fields.iter().find(|field| {
                field.field_id == definition.field_id()
                    && field.shape.first().is_some_and(|count| *count > 0)
            }) else {
                continue;
            };
            let count = storage.shape[0];
            if storage.shape != definition.storage_shape(count)? {
                return Err(RecoveredFieldError::StorageShapeMismatch);
            }
            count
        };
        if count > 0 {
            inventory.push((definition, count));
        }
    }
    Ok(inventory)
}

/// Semantic field inventory produced by the structural solid recovery path.
///
/// Resource and observation planning use the same declarations as postprocessing so output
/// selection cannot drift from the fields that are actually recoverable.
pub fn structural_observable_field_definitions(
) -> Result<Vec<RecoveredFieldDefinition>, RecoveredFieldError> {
    let declarations = [
        (
            FEA_FIELD_STRUCTURAL_DISPLACEMENT,
            ObservationQuantity::Displacement,
            FieldLocation::Node,
            RecoveredFieldEntityDomain::Nodes,
            ObservationValueShape::Vector3,
        ),
        (
            FEA_FIELD_STRUCTURAL_ROTATION,
            ObservationQuantity::Rotation,
            FieldLocation::Node,
            RecoveredFieldEntityDomain::Nodes,
            ObservationValueShape::Vector3,
        ),
        (
            FEA_FIELD_STRUCTURAL_STRAIN,
            ObservationQuantity::Strain,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::VolumeElements,
            ObservationValueShape::SymmetricTensor3,
        ),
        (
            FEA_FIELD_STRUCTURAL_STRESS,
            ObservationQuantity::Stress,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::VolumeElements,
            ObservationValueShape::SymmetricTensor3,
        ),
        (
            FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRAIN,
            ObservationQuantity::Strain,
            FieldLocation::IntegrationPoint,
            RecoveredFieldEntityDomain::VolumeIntegrationPoints,
            ObservationValueShape::SymmetricTensor3,
        ),
        (
            FEA_FIELD_STRUCTURAL_INTEGRATION_POINT_STRESS,
            ObservationQuantity::Stress,
            FieldLocation::IntegrationPoint,
            RecoveredFieldEntityDomain::VolumeIntegrationPoints,
            ObservationValueShape::SymmetricTensor3,
        ),
        (
            FEA_FIELD_STRUCTURAL_VON_MISES,
            ObservationQuantity::VonMisesStress,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::VolumeElements,
            ObservationValueShape::Scalar,
        ),
        (
            FEA_FIELD_STRUCTURAL_NODAL_VON_MISES,
            ObservationQuantity::VonMisesStress,
            FieldLocation::Node,
            RecoveredFieldEntityDomain::Nodes,
            ObservationValueShape::Scalar,
        ),
        (
            FEA_FIELD_STRUCTURAL_NODAL_STRESS,
            ObservationQuantity::Stress,
            FieldLocation::Node,
            RecoveredFieldEntityDomain::Nodes,
            ObservationValueShape::SymmetricTensor3,
        ),
        (
            FEA_FIELD_STRUCTURAL_STRAIN_ENERGY_DENSITY,
            ObservationQuantity::StrainEnergy,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::VolumeElements,
            ObservationValueShape::Scalar,
        ),
        (
            FEA_FIELD_STRUCTURAL_REACTION_FORCE,
            ObservationQuantity::ReactionForce,
            FieldLocation::Node,
            RecoveredFieldEntityDomain::Nodes,
            ObservationValueShape::Vector3,
        ),
        (
            FEA_FIELD_STRUCTURAL_CONSTRAINT_REACTION_FORCE,
            ObservationQuantity::ReactionForce,
            FieldLocation::Constraint,
            RecoveredFieldEntityDomain::ConstraintGroups,
            ObservationValueShape::Vector3,
        ),
        (
            FEA_FIELD_STRUCTURAL_CONSTRAINT_REACTION_MOMENT_ABOUT_GLOBAL_ORIGIN,
            ObservationQuantity::ReactionMoment,
            FieldLocation::Constraint,
            RecoveredFieldEntityDomain::ConstraintGroups,
            ObservationValueShape::Vector3,
        ),
        (
            FEA_FIELD_STRUCTURAL_BEAM_AXIAL_FORCE,
            ObservationQuantity::BeamAxialForce,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::BeamElements,
            ObservationValueShape::Scalar,
        ),
        (
            FEA_FIELD_STRUCTURAL_BEAM_SHEAR_FORCE,
            ObservationQuantity::BeamShearForce,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::BeamElements,
            ObservationValueShape::Tensor { extents: vec![2] },
        ),
        (
            FEA_FIELD_STRUCTURAL_BEAM_TORSION_MOMENT,
            ObservationQuantity::BeamTorsionMoment,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::BeamElements,
            ObservationValueShape::Scalar,
        ),
        (
            FEA_FIELD_STRUCTURAL_BEAM_BENDING_MOMENT,
            ObservationQuantity::BeamBendingMoment,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::BeamElements,
            ObservationValueShape::Tensor { extents: vec![2] },
        ),
        (
            FEA_FIELD_STRUCTURAL_BEAM_BENDING_STRESS,
            ObservationQuantity::BeamBendingStress,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::BeamElements,
            ObservationValueShape::Tensor { extents: vec![2] },
        ),
        (
            FEA_FIELD_STRUCTURAL_BEAM_TORSION_STRESS,
            ObservationQuantity::BeamTorsionStress,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::BeamElements,
            ObservationValueShape::Scalar,
        ),
        (
            FEA_FIELD_STRUCTURAL_SHELL_MEMBRANE_FORCE,
            ObservationQuantity::ShellMembraneForce,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::ShellElements,
            ObservationValueShape::Tensor { extents: vec![3] },
        ),
        (
            FEA_FIELD_STRUCTURAL_SHELL_BENDING_MOMENT,
            ObservationQuantity::ShellBendingMoment,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::ShellElements,
            ObservationValueShape::Tensor { extents: vec![3] },
        ),
        (
            FEA_FIELD_STRUCTURAL_SHELL_TRANSVERSE_SHEAR,
            ObservationQuantity::ShellTransverseShear,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::ShellElements,
            ObservationValueShape::Tensor { extents: vec![2] },
        ),
        (
            FEA_FIELD_STRUCTURAL_SHELL_VON_MISES,
            ObservationQuantity::ShellVonMises,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::ShellElements,
            ObservationValueShape::Scalar,
        ),
    ];
    declarations
        .into_iter()
        .map(
            |(field_id, quantity, location, entity_domain, value_shape)| {
                if field_id == FEA_FIELD_STRUCTURAL_NODAL_VON_MISES
                    || field_id == FEA_FIELD_STRUCTURAL_NODAL_STRESS
                {
                    RecoveredFieldDefinition::new_derived(
                        field_id,
                        quantity,
                        location,
                        entity_domain,
                        value_shape,
                        RecoveredFieldNumericRepresentation::Real,
                        RecoveredFieldDerivation::LocationTransfer {
                            source_location: FieldLocation::Element,
                            recovery: runmat_analysis_core::RecoveryPolicy::VolumeWeightedAverage,
                        },
                    )
                } else {
                    RecoveredFieldDefinition::new(
                        field_id,
                        quantity,
                        location,
                        entity_domain,
                        value_shape,
                        RecoveredFieldNumericRepresentation::Real,
                    )
                }
            },
        )
        .collect()
}

/// The static numerical result also publishes connector force and stored energy when connectors
/// are present. Use their recovery-owned definitions during preassembly observation admission.
pub fn structural_static_observable_field_definitions(
) -> Result<Vec<RecoveredFieldDefinition>, RecoveredFieldError> {
    let mut fields = structural_observable_field_definitions()?;
    fields.extend(crate::post::connector_observable_field_definitions()?);
    Ok(fields)
}
