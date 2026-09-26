//! One definition inventory for thermal history planning and evaluation.
use crate::results::field::{
    RecoveredFieldDefinition, RecoveredFieldEntityDomain, RecoveredFieldError,
    RecoveredFieldNumericRepresentation,
};
use runmat_analysis_core::{FieldLocation, ObservationQuantity, ObservationValueShape};

/// Same ordinal-derived field definition used by publication and preassembly profiling.
pub(crate) fn snapshot_definition(
    index: usize,
    ordinal: usize,
) -> Result<RecoveredFieldDefinition, RecoveredFieldError> {
    let definitions = definitions()?;
    let template = definitions
        .get(index)
        .ok_or(RecoveredFieldError::StorageShapeMismatch)?;
    let ids = [
        crate::contracts::fea_thermal_temperature_field_id,
        crate::contracts::fea_thermal_temperature_gradient_field_id,
        crate::contracts::fea_thermal_heat_flux_field_id,
        crate::contracts::fea_thermal_energy_storage_rate_field_id,
    ];
    RecoveredFieldDefinition::new_derived(
        ids[index](ordinal),
        template.quantity(),
        template.location(),
        template.entity_domain(),
        template.value_shape().clone(),
        template.numeric_representation(),
        template.derivation(),
    )
}

pub(crate) fn definitions() -> Result<Vec<RecoveredFieldDefinition>, RecoveredFieldError> {
    [
        (
            "thermal.temperature",
            ObservationQuantity::Temperature,
            FieldLocation::Node,
            RecoveredFieldEntityDomain::StructuralNodes,
            ObservationValueShape::Scalar,
        ),
        (
            "thermal.temperature_gradient",
            ObservationQuantity::TemperatureGradient,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::SolidElements,
            ObservationValueShape::Vector3,
        ),
        (
            "thermal.heat_flux",
            ObservationQuantity::HeatFlux,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::SolidElements,
            ObservationValueShape::Vector3,
        ),
        (
            "thermal.energy_storage_rate",
            ObservationQuantity::ThermalEnergyStorageRateDensity,
            FieldLocation::Element,
            RecoveredFieldEntityDomain::SolidElements,
            ObservationValueShape::Scalar,
        ),
    ]
    .into_iter()
    .map(|(id, quantity, location, domain, shape)| {
        RecoveredFieldDefinition::new(
            id,
            quantity,
            location,
            domain,
            shape,
            RecoveredFieldNumericRepresentation::Real,
        )
    })
    .collect()
}
